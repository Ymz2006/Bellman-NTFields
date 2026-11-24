"""Fast single-query MPPI on an LBS field (lift + ntrl-demo metric network): the planner of jeffrey/maze_scale/plan.py with the work per step
cut down, plus a bidirectional version.

What bench.py of jeffrey/lbs_final does per MPPI step (the reference, `reference()` here): 100 (state, goal) pairs go through NN.out, which
encodes the 100 states AND 100 copies of the goal, recomputes the Lipschitz-normalized weights, computes the lift height of all 200 rows
from the exact clearance, and builds an autograd graph that is thrown away.

    Field        the frozen pieces of one field: normalized weights, the Fourier matrix with the lift column, the lift profile G(d) and the
                 clearance function d(q) (the scene's exact one, or a replacement: dist= a learned network / a table, Task 2)
    kernel       one MPPI step for A agents as one pure tensor function (torch.compile-able): sample, clip, the two rollout ends, lift,
                 encoder on A x S x 2 rows only, metric against A cached target features, softmax update. The arithmetic of the sampling
                 and the update is plan.mppi's, so with the same random numbers the path is the reference's up to float rounding.
    meet kernel  the same step for a pair of agents that chase each other (Task 3): agent 0 starts at the start, agent 1 at the goal, each
                 one's target is the other's current position (their features come out of the same encoder call: two more rows). Within
                 `freeze` of each other agent 1 stops and agent 0 finishes on it with the usual tolerance.
    plan_*       host loops: one step per call (CPU) or blocks of K steps per call with one read-back per block (GPU, CUDA graphs).

Units: everything in the network's normalized coordinates, as plan.mppi. Nothing here checks collisions: the path is only generated.
"""
import math
import time
import torch
import torch._dynamo
import fastnet as F                      # jeffrey/lbs snapshot (verbatim midpoint_dp/fastnet.py)
from maze import MARGIN_CELLS, OFFSET_CELLS

# One process compiles many planners from the same two functions (kernel, block) with different constants and shapes. Floats must stay
# constants of the graph (a symbolic float becomes a CPU scalar input and CUDA graphs are skipped), and every planner needs its own entry.
torch._dynamo.config.specialize_float = True
torch._dynamo.config.recompile_limit = 256
if hasattr(torch._dynamo.config, 'accumulated_recompile_limit'):
    torch._dynamo.config.accumulated_recompile_limit = 4096


class Field:
    """Frozen LBS field. dist: None = the scene's exact clearance (scene.signed_cells), else a callable q (N, 3) -> clearance in cells."""

    def __init__(self, net, scene, dist=None):
        self.scene = scene
        self.lift = getattr(net, 'lift_kind', 'none') != 'none'
        B2pi = (2.0 * math.pi * net.B).detach().clone()
        if self.lift:
            assert net.lift_kind == 'time' and not getattr(net, 'K', 0), 'only the scalar travel-time lift without octaves'
            self.W = torch.cat([B2pi, (2.0 * math.pi * net.b_h.detach().clone())[None, :]], 0).contiguous()
            pr = net.profile
            self.table, self.nodes = pr.table.clone(), int(pr.nodes)
            self.G_offset, self.floor_slope, self.depth = float(pr.G_offset), float(pr.floor_slope), float(pr.depth)
            self.hscale = float(net.lift_gain) / float(scene.E)
        else:
            self.W = B2pi.contiguous()
        fz = F.Frozen(net)
        self.layers = [(w.contiguous(), b.contiguous()) for w, b in fz.layers]
        self.gates, self.nl1 = fz.gates, fz.nl1
        self.dist = dist if dist is not None else scene.signed_cells
        self.lo2, self.hi2 = float(scene.lo[2]), float(scene.hi[2])
        self.dim_feat = self.layers[-1][0].shape[0]

    def G(self, d):
        """lift.Profile.__call__(d)[0] for kind 'time' (the slope is not needed without gradients)."""
        dc = d.clamp(OFFSET_CELLS, MARGIN_CELLS)
        u = (dc - OFFSET_CELLS) / (MARGIN_CELLS - OFFSET_CELLS) * (self.nodes - 1)
        i = u.floor().long().clamp(0, self.nodes - 2)
        w = u - i
        G = (1 - w) * self.table[i] + w * self.table[i + 1]
        return torch.where(d < OFFSET_CELLS, self.G_offset + (OFFSET_CELLS - d.clamp(min=-self.depth)) * self.floor_slope, G)

    def rows(self, q):
        if not self.lift:
            return q
        return torch.cat([q, (self.hscale * self.G(self.dist(q)))[:, None]], 1)

    def encode(self, q):
        return F._forward(self.W, self.layers, self.gates, self.nl1, self.rows(q))

    @torch.no_grad()
    def travel(self, pairs):
        return F.metric(self.encode(pairs[:, :3]), self.encode(pairs[:, 3:]))


def make_kernel(field, step, temperature, meet=False, freeze=0.0, first_weight=10.0):
    """One MPPI step for A agents. Returns f(cur (A, 3), prior (A, 3), tfeat (A, C) or None, z0 (A, S, 1, 3), zh (A, S, H, 3)) ->
    (new (A, 3), update (A, 3)). meet: A = 2 P, agents p and p + P are partners; tfeat is ignored."""
    lo2, hi2 = field.lo2, field.hi2
    step, temperature, freeze = float(step), float(temperature), float(freeze)

    def kernel(cur, prior, tfeat, z0, zh):
        A, S, H = zh.shape[0], zh.shape[1], zh.shape[2]
        dp = step * z0 + step * zh
        dp = dp + 2 * prior[:, None, None, :]
        dp = dp / (torch.linalg.vector_norm(dp, dim=3, keepdim=True).clamp(min=step) / step)
        cs = torch.cumsum(dp, dim=2)
        ends = cur[:, None, None, :] + torch.stack([cs[:, :, 0, :], cs[:, :, H - 1, :]], 2)       # (A, S, 2, 3)
        ends = torch.cat([ends[..., :2], ends[..., 2:].clamp(lo2, hi2)], -1)
        flat = ends.reshape(-1, 3)
        if meet:
            f = field.encode(torch.cat([flat, cur], 0))
            fc = f[A * S * 2:]
            tf = torch.roll(fc, A // 2, 0)
            f = f[:A * S * 2]
        else:
            f = field.encode(flat)
            tf = tfeat
        C = f.shape[1]
        T = F.metric(f, tf[:, None, :].expand(A, S * 2, C).reshape(-1, C)).reshape(A, S, 2)
        c = first_weight * T[..., 0] + T[..., 1]
        weight = torch.softmax(-temperature * c, dim=1)
        update = torch.einsum('bs,bsd->bd', weight, dp[:, :, 0, :])
        if meet:
            P = A // 2
            near = torch.linalg.vector_norm(cur[:P] - cur[P:], dim=1, keepdim=True) < freeze
            update = torch.cat([update[:P], torch.where(near, torch.zeros_like(update[P:]), update[P:])], 0)
        new = cur + update
        new = torch.cat([new[:, :2], new[:, 2:].clamp(lo2, hi2)], 1)
        return new, update

    return kernel


def make_block(kernel, K):
    """K steps in one graph: f(cur, prior, tfeat, Z0 (K, A, S, 1, 3), ZH (K, A, S, H, 3)) -> (cur, prior, positions (K, A, 3))."""
    def block(cur, prior, tfeat, Z0, ZH):
        out = []
        for k in range(K):
            cur, prior = kernel(cur, prior, tfeat, Z0[k], ZH[k])
            out.append(cur)
        return cur, prior, torch.stack(out)
    return block


class Planner:
    """Single-query planner on one device. variant:
         'goal'  one agent from the start to the goal (plan.mppi)
         'both'  two independent agents, start -> goal and goal -> start, fixed targets (bidirectional, first arrival or first meeting)
         'meet'  two agents chasing each other (bidirectional, moving targets)
       compile: None (eager) | 'default' | 'reduce-overhead' (CUDA graphs).  block: steps per call (1 = read back every step).
       noise: 'ref' draws the reference's random numbers call for call (torch.manual_seed(seed); randn(A, S, 1, 3), randn(A, S, H, 3) per
       step); 'bulk' one randn per block from a private generator."""

    def __init__(self, field, st, variant='goal', samples=None, horizon=None, step_mult=1.0, compile=None, block=1, noise='ref', freeze_steps=3.0, meet_tol=None):
        self.field, self.variant, self.noise = field, variant, noise
        self.S = int(samples or st['sample_num'])
        self.H = int(horizon or st['horizon'])
        self.step = float(st['step']) * float(step_mult)
        self.temperature = float(st['temperature']) / float(step_mult)
        self.tol = float(st['tol'])
        self.meet_tol = float(meet_tol) if meet_tol is not None else self.tol
        self.A = 1 if variant == 'goal' else 2
        self.K = int(block)
        self.freeze = freeze_steps * self.step + self.tol
        self.dev = field.W.device
        kern = make_kernel(field, self.step, self.temperature, meet=(variant == 'meet'), freeze=self.freeze)
        fn = kern if self.K == 1 else make_block(kern, self.K)
        self.raw = fn
        self.fn = torch.compile(fn, dynamic=False, mode=compile) if compile else fn
        self.graphs = compile == 'reduce-overhead' and self.dev.type == 'cuda'     # CUDA graphs: outputs are overwritten by the next call
        self.enc = field.encode
        self.gen = torch.Generator(self.dev)
        self.setup_seconds = 0.0

    @torch.no_grad()
    def warm(self, start, goal, calls=3):
        """Compile / build the CUDA graph outside the timed region."""
        t0 = time.perf_counter()
        for _ in range(calls):
            self.plan(start, goal, budget=2 * self.K, seed=0)
        if self.dev.type == 'cuda':
            torch.cuda.synchronize()
        self.setup_seconds = time.perf_counter() - t0

    def _noise(self, seed_first):
        A, S, H, K, dev = self.A, self.S, self.H, self.K, self.dev
        if self.noise == 'ref':
            if K == 1:
                return torch.randn(A, S, 1, 3, device=dev), torch.randn(A, S, H, 3, device=dev)
            z = [(torch.randn(A, S, 1, 3, device=dev), torch.randn(A, S, H, 3, device=dev)) for _ in range(K)]
            return torch.stack([a for a, _ in z]), torch.stack([b for _, b in z])
        if K == 1:
            z = torch.randn(A, S, H + 1, 3, device=dev, generator=self.gen)
            return z[:, :, :1], z[:, :, 1:]
        z = torch.randn(K, A, S, H + 1, 3, device=dev, generator=self.gen)
        return z[:, :, :, :1], z[:, :, :, 1:]

    @torch.no_grad()
    def plan(self, start, goal, budget, seed=0):
        """start, goal: (3,) tensors on the planner's device. budget: step limit (iterations). Returns dict(path (L, 3) cpu tensor from the
        start to the goal, steps, converged, and for the two-agent variants how the path was closed)."""
        if self.noise == 'ref':
            torch.manual_seed(seed)
        else:
            self.gen.manual_seed(seed)
        A, K, tol = self.A, self.K, self.tol
        variant = self.variant
        if variant == 'goal':
            cur = start[None, :].clone()
            tfeat = self.enc(goal[None, :])
        else:
            cur = torch.stack([start, goal])
            tfeat = self.enc(torch.stack([goal, start])) if variant == 'both' else None
        prior = torch.zeros_like(cur)
        target = goal[None, :] if variant == 'goal' else torch.stack([goal, start])
        blocks = []
        steps, converged, how, cut = int(budget), False, 'budget', None
        it = 0
        trace_np = None
        while it < budget:
            z0, zh = self._noise(seed)
            if K == 1:
                cur, prior = self.fn(cur, prior, tfeat, z0, zh)
                if self.graphs:
                    cur, prior = cur.clone(), prior.clone()
                pos = cur[None]
            else:
                cur, prior, pos = self.fn(cur, prior, tfeat, z0, zh)
                if self.graphs:
                    cur, prior, pos = cur.clone(), prior.clone(), pos.clone()
            blocks.append(pos)
            if variant == 'goal':
                d = torch.linalg.vector_norm(target[None] - pos, dim=2)[:, 0]            # (K,)
                hit = torch.nonzero(d < tol)
                if len(hit):
                    steps, converged, how = it + int(hit[0]) + 1, True, 'goal'
                    break
            elif variant == 'meet':
                d = torch.linalg.vector_norm(pos[:, 0] - pos[:, 1], dim=1)              # (K,)
                hit = torch.nonzero(d < tol)
                if len(hit):
                    steps, converged, how = it + int(hit[0]) + 1, True, 'meet'
                    break
            else:
                # first arrival of either agent at its own target, or first meeting of one agent's position with the other's trace
                new = pos.cpu()
                trace_np = new if trace_np is None else torch.cat([trace_np, new])       # (L, 2, 3) positions after steps 1..L
                L0 = len(trace_np) - len(new)
                tg = target.cpu()
                arr = torch.linalg.vector_norm(tg[None] - new, dim=2) < tol              # (K, 2)
                full0 = torch.cat([start.cpu()[None], trace_np[:, 0]])                   # agent 0 positions 0..L
                full1 = torch.cat([goal.cpu()[None], trace_np[:, 1]])
                d01 = torch.cdist(new[:, 0], full1)                                      # (K, L + 1): agent 0's new positions vs agent 1's trace
                d10 = torch.cdist(new[:, 1], full0)
                idx = torch.arange(L0 + 1, L0 + len(new) + 1)[:, None]
                ok = torch.arange(len(full1))[None, :] <= idx                            # only the other's positions up to the same step
                d01 = torch.where(ok, d01, torch.full_like(d01, float('inf')))
                d10 = torch.where(ok, d10, torch.full_like(d10, float('inf')))
                m01, j01 = d01.min(1)
                m10, j10 = d10.min(1)
                stop = None
                for k in range(len(new)):
                    i = L0 + k + 1
                    if bool(arr[k, 0]):
                        stop = ('goal', i, None)
                    elif bool(arr[k, 1]):
                        stop = ('start', i, None)
                    elif float(m01[k]) < self.meet_tol:
                        stop = ('meet01', i, int(j01[k]))
                    elif float(m10[k]) < self.meet_tol:
                        stop = ('meet10', i, int(j10[k]))
                    if stop:
                        break
                if stop:
                    how, steps, cut = stop
                    converged = True
                    break
            it += K
        steps = min(steps, int(budget)) if not converged else steps
        P = torch.cat(blocks).cpu() if variant != 'both' or trace_np is None else trace_np
        s, g = start.cpu()[None], goal.cpu()[None]
        if variant == 'goal':
            path = torch.cat([s, P[:steps, 0], g])
        elif variant == 'meet':
            path = torch.cat([s, P[:steps, 0], P[:steps, 1].flip(0), g])
        else:
            f0 = torch.cat([s, P[:steps, 0]])
            f1 = torch.cat([g, P[:steps, 1]])
            if how == 'goal' or how == 'budget':
                path = torch.cat([f0, g])
            elif how == 'start':
                path = torch.cat([s, f1.flip(0)])
            elif how == 'meet01':                    # agent 0 at step `steps` is next to agent 1's position `cut`
                path = torch.cat([f0, f1[:cut + 1].flip(0)])
            else:                                    # agent 1 at step `steps` is next to agent 0's position `cut`
                path = torch.cat([f0[:cut + 1], f1.flip(0)])
        return dict(path=path, steps=int(steps), converged=bool(converged), how=how)


@torch.no_grad()
def reference(net, scene, st, start, goal, budget, seed=0):
    """bench.py of jeffrey/lbs_final: plan.mppi with plan.field_cost(net) on one query. Returns the same dict as Planner.plan."""
    import plan as P
    cost = P.field_cost(net)
    pp, stop, conv = P.mppi(cost, start[None], goal[None], seed, torch.tensor([int(budget)]), scene.lo, scene.hi, **st)
    return dict(path=pp[0].cpu(), steps=int(stop[0]), converged=bool(conv[0]), how='goal' if bool(conv[0]) else 'budget')
