"""Travel-cost networks of the segment-DP experiments and their composite over distance segments.

TravelCost   method/network.py unchanged: two positions (q, g), symmetric pair features.
OffsetCost   position + offset: T(q, g) = |d| (1 + softplus(f(q, d))) with d = g - q. Same modules and widths as
             TravelCost (shared encoder E, 64-wide pair head); the pair is [E(q), E(d), E(q)*E(d), cos(2 pi h d), |d|]
             instead of [E(q)+E(g), (E(q)-E(g))^2, E(q)*E(g), cos(2 pi h (q-g))], so the offset, and with it the
             distance |d|, is an input of its own. 338,241 parameters (TravelCost + 64 for |d|).
Ensemble     M independent models of one architecture evaluated in one batched pass (inputs [M, B, 6]); the same
             computation as the single modules, used to train one model per distance segment in one process.
SegmentField the composite: model m owns segment [e_(m-1), e_m] of |q - g|; across each inner edge the weight ramps
             linearly from model m to model m+1 over [e_m - blend, e_m + blend], so the field and its gradient are
             continuous. Every model is trained on its segment widened by the blend. A one-segment field is its model.
"""
import sys
sys.path.insert(0, '/clean/method')
import math, re
import torch
from torch import nn
from torch.nn import functional as F
from network import TravelCost, CONFIG, ResidualBlock, JointFeatures

ARCHS = ('pair', 'offset', 'add', 'exp')
ARCH_RE = re.compile(r'^pair-h(\d+)(?:-i(\d+(?:\.\d+)?))?$')      # 'pair-h16' / 'pair-h16-i4': TravelCost with H harmonics and an input scale (resolution knobs, 2026-11-01)


def parse_arch(arch):
    """('pair', H, inscale) for the pair family, else (arch, 4, 1.0)."""
    m = ARCH_RE.match(arch)
    if m: return 'pair', int(m.group(1)), float(m.group(2) or 1.)
    assert arch in ARCHS, arch; return arch, 4, 1.


def valid_arch(arch):
    return arch in ARCHS or ARCH_RE.match(arch) is not None


class PairCost(TravelCost):
    """TravelCost with H harmonics of the joint features and the input coordinates scaled by inscale before the features
    (frequencies 2 pi h inscale, h = 1..H, instead of 2 pi h, h = 1..4); same trunk widths. With (4, 1) it is TravelCost."""
    def __init__(self, harmonics=4, inscale=1.):
        super().__init__(); self.H, self.inscale = int(harmonics), float(inscale)
        self.state_input = nn.Linear(6+12*self.H, 256); self.pair_input = nn.Linear(768+6*self.H, 64)

    def feats(self, q):
        h = torch.arange(1, self.H+1, device=q.device, dtype=q.dtype); phase = (2*math.pi*(self.inscale*q)[:, :, None]*h).flatten(1)
        return torch.cat((q, torch.sin(phase), torch.cos(phase)), 1)

    def encode(self, q):
        h = F.silu(self.state_input(self.feats(q)))
        for block in self.state_blocks: h = block(h)
        return self.state_norm(h)

    def forward(self, q, goal):
        encoded = self.encode(torch.cat((q, goal), 0)); a, b = encoded[:len(q)], encoded[len(q):]
        h = torch.arange(1, self.H+1, device=q.device, dtype=q.dtype); phase = (2*math.pi*(self.inscale*(q-goal))[:, :, None]*h).flatten(1)
        pair = torch.cat((a+b, (a-b).square(), a*b, torch.cos(phase)), 1); x = F.silu(self.pair_input(pair))
        for block in self.pair_blocks: x = block(x)
        residual = self.output(self.pair_norm(x)); return torch.linalg.vector_norm(q-goal, dim=1, keepdim=True)*(1+F.softplus(residual))


class OffsetCost(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = JointFeatures()
        self.state_input = nn.Linear(54, 256)
        self.state_blocks = nn.ModuleList([ResidualBlock(256) for _ in range(2)])
        self.state_norm = nn.LayerNorm(256)
        self.pair_input = nn.Linear(793, 64)
        self.pair_blocks = nn.ModuleList([ResidualBlock(64)])
        self.pair_norm = nn.LayerNorm(64)
        self.output = nn.Linear(64, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, math.log(math.expm1(0.05)))

    def encode(self, q):
        h = F.silu(self.state_input(self.features(q)))
        for block in self.state_blocks:
            h = block(h)
        return self.state_norm(h)

    def forward(self, q, goal):
        offset = goal-q
        encoded = self.encode(torch.cat((q, offset), dim=0))
        a, b = encoded[:len(q)], encoded[len(q):]
        harmonic = torch.arange(1, 5, device=q.device, dtype=q.dtype)
        phase = (2.0*math.pi*offset[:, :, None]*harmonic).flatten(1)
        distance = torch.linalg.vector_norm(offset, dim=1, keepdim=True)
        pair = torch.cat((a, b, a*b, torch.cos(phase), distance), dim=1)
        h = F.silu(self.pair_input(pair))
        for block in self.pair_blocks:
            h = block(h)
        residual = self.output(self.pair_norm(h))
        return distance*(1.0+F.softplus(residual))


class DecodedCost(TravelCost):
    """TravelCost's trunk with another decoder of the residual r (dp3.py, task 2). The travel time is never below
    the lower bound |q - g| (speed <= 1) for 'add'; 'exp' is the unbounded control.
        add   T = |q - g| + softplus(r)        (an addition over the minimum possible travel time)
        exp   T = |q - g| exp(r)               (multiplicative, no bound)
    TravelCost itself is 'mult': T = |q - g| (1 + softplus(r)). Same 338,177 parameters."""

    def __init__(self, decoder):
        super().__init__(); assert decoder in ('add', 'exp'); self.decoder = decoder
        if decoder == 'exp':
            nn.init.constant_(self.output.bias, math.log(1.05))

    def forward(self, q, goal):
        encoded = self.encode(torch.cat((q, goal), dim=0))
        a, b = encoded[:len(q)], encoded[len(q):]
        harmonic = torch.arange(1, 5, device=q.device, dtype=q.dtype)
        phase = (2.0*math.pi*(q-goal)[:, :, None]*harmonic).flatten(1)
        pair = torch.cat((a+b, (a-b).square(), a*b, torch.cos(phase)), dim=1)
        h = F.silu(self.pair_input(pair))
        for block in self.pair_blocks:
            h = block(h)
        residual = self.output(self.pair_norm(h))
        distance = torch.linalg.vector_norm(q-goal, dim=1, keepdim=True)
        return distance+F.softplus(residual) if self.decoder == 'add' else distance*torch.exp(residual)


def build(arch):
    kind, H, inscale = parse_arch(arch)
    if kind == 'pair' and arch != 'pair': return PairCost(H, inscale)
    return TravelCost() if arch == 'pair' else OffsetCost() if arch == 'offset' else DecodedCost(arch)


class Ensemble(nn.Module):
    """M models of one architecture stacked along a leading dimension; forward(q, goal) takes [M, B, 6] each."""

    def __init__(self, models):
        super().__init__()
        self.arch = 'offset' if isinstance(models[0], OffsetCost) else 'pair'
        self.H, self.inscale = getattr(models[0], 'H', 4), getattr(models[0], 'inscale', 1.)
        self.names = [n for n, _ in models[0].named_parameters()]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.params = nn.ParameterList([nn.Parameter(torch.stack([dict(m.named_parameters())[n].detach() for m in models])) for n in self.names])
        self.count = len(models)

    def p(self, name):
        return self.params[self.index[name]]

    def linear(self, x, name):
        return torch.baddbmm(self.p(name+'.bias')[:, None, :], x, self.p(name+'.weight').transpose(1, 2))

    def norm(self, x, name):
        return F.layer_norm(x, x.shape[-1:])*self.p(name+'.weight')[:, None, :]+self.p(name+'.bias')[:, None, :]

    def block(self, x, name):
        return x+self.linear(F.silu(self.linear(self.norm(x, name+'.norm'), name+'.first')), name+'.second')/math.sqrt(2.0)

    def features(self, q):
        harmonic = torch.arange(1, self.H+1, device=q.device, dtype=q.dtype)
        phase = (2.0*math.pi*(self.inscale*q)[..., None]*harmonic).flatten(2)
        return torch.cat((q, torch.sin(phase), torch.cos(phase)), dim=2)

    def encode(self, x):
        h = F.silu(self.linear(self.features(x), 'state_input'))
        for i in range(2):
            h = self.block(h, f'state_blocks.{i}')
        return self.norm(h, 'state_norm')

    def forward(self, q, goal):
        count = q.shape[1]
        harmonic = torch.arange(1, (self.H if self.arch == 'pair' else 4)+1, device=q.device, dtype=q.dtype)
        if self.arch == 'pair':
            encoded = self.encode(torch.cat((q, goal), dim=1)); a, b = encoded[:, :count], encoded[:, count:]
            phase = (2.0*math.pi*(self.inscale*(q-goal))[..., None]*harmonic).flatten(2)
            distance = torch.linalg.vector_norm(q-goal, dim=2, keepdim=True)
            pair = torch.cat((a+b, (a-b).square(), a*b, torch.cos(phase)), dim=2)
        else:
            offset = goal-q
            encoded = self.encode(torch.cat((q, offset), dim=1)); a, b = encoded[:, :count], encoded[:, count:]
            phase = (2.0*math.pi*offset[..., None]*harmonic).flatten(2)
            distance = torch.linalg.vector_norm(offset, dim=2, keepdim=True)
            pair = torch.cat((a, b, a*b, torch.cos(phase), distance), dim=2)
        h = F.silu(self.linear(pair, 'pair_input'))
        h = self.block(h, 'pair_blocks.0')
        residual = self.linear(self.norm(h, 'pair_norm'), 'output')
        return distance*(1.0+F.softplus(residual))

    def state_dict_of(self, m):
        return {n: self.p(n)[m].detach().clone() for n in self.names}


def clip_per_model(parameters, max_norm):
    """Global L2 gradient clipping (as method/optimizer.py) separately for every model of an Ensemble."""
    parameters = [p for p in parameters if p.grad is not None]
    square = sum(p.grad.detach().double().square().flatten(1).sum(1) for p in parameters)
    norm = square.sqrt(); coefficient = (max_norm/(norm+1e-12)).clamp(max=1.0)
    for p in parameters:
        p.grad.mul_(coefficient.to(p.grad.dtype).view(-1, *[1]*(p.grad.dim()-1)))
    return norm


class SegmentField(nn.Module):
    def __init__(self, models, edges, blend):
        super().__init__()
        assert len(edges) == len(models)+1 and all(b-a >= 2*blend for a, b in zip(edges[:-1], edges[1:])), (edges, blend)
        self.models = nn.ModuleList(models); self.edges = [float(e) for e in edges]; self.blend = float(blend)

    def weights(self, distance):
        if len(self.models) == 1:
            return distance.new_ones((len(distance), 1))
        inner = torch.tensor(self.edges[1:-1], device=distance.device, dtype=distance.dtype)
        ramp = ((distance[:, None]-(inner-self.blend))/(2*self.blend)).clamp(0, 1)      # share of the upper model
        ones = distance.new_ones((len(distance), 1)); zeros = distance.new_zeros((len(distance), 1))
        return torch.cat((ones, ramp), dim=1)-torch.cat((ramp, zeros), dim=1)

    def forward(self, q, goal):
        if len(self.models) == 1:
            return self.models[0](q, goal)
        distance = torch.linalg.vector_norm(q-goal, dim=1)
        weight = self.weights(distance); out = q.new_zeros((len(q), 1))
        # (model, row) of every nonzero weight, grouped by model: one host sync for the whole routing, so the models'
        # forwards are launched back to back (same rows, order and accumulation as one pass per model)
        pairs = (weight > 0).T.nonzero(); counts = torch.bincount(pairs[:, 0], minlength=len(self.models)).tolist(); start = 0
        for m, count in enumerate(counts):
            if count == len(q):
                out = out+weight[:, m:m+1]*self.models[m](q, goal)
            elif count:
                rows = pairs[start:start+count, 1]
                out = out.index_add(0, rows, weight[rows, m:m+1]*self.models[m](q[rows], goal[rows]))
            start += count
        return out


def save_field(path, arch, edges, blend, state_dicts, archs=None, **extra):
    """arch: the architecture of every model, or archs: one per model (dp3 chains mix the base's with their own)."""
    payload = dict(kind='segment_field', arch=arch, edges=[float(e) for e in edges], blend=float(blend), model_config=CONFIG, model_state_dicts=state_dicts, **extra)
    if archs is not None:
        assert len(archs) == len(state_dicts) and all(valid_arch(x) for x in archs); payload['archs'] = list(archs)
    torch.save(payload, path)


def load_field(path, device='cuda:0'):
    """A segment-field checkpoint (save_field) or a single TravelCost checkpoint (method/network.py format)."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get('kind') != 'segment_field':
        assert payload['model_config'] == CONFIG
        net = TravelCost().to(device); net.load_state_dict(payload['model_state_dict'], strict=True)
        return net.eval().requires_grad_(False)
    assert payload['model_config'] == CONFIG and valid_arch(payload['arch'])
    archs = payload.get('archs') or [payload['arch']]*len(payload['model_state_dicts']); models = []
    for arch, state in zip(archs, payload['model_state_dicts']):
        net = build(arch).to(device); net.load_state_dict(state, strict=True); models.append(net)
    return SegmentField(models, payload['edges'], payload['blend']).to(device).eval().requires_grad_(False)
