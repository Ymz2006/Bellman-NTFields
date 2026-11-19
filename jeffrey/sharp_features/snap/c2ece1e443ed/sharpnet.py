"""ntrl-demo's metric network with a sharp-feature input mapping (features.py plan), unchanged otherwise.

Replace specs (width 256): only input_mapping differs; plain rows are computed by the original expressions
(w = 2 pi B, x @ w, sin, cos), so the spec 'octave' is ntrl-demo's network bit for bit, and a replace spec has the same
parameters, the same initialization (the same global-generator draws) and the same data order as the octave run of the
same seed.

Add specs (width 256 + m): the three layers that read the input (pe_gate[0], pe_gate[1], encoder[1]) are
Linear(256 + m, 256) and are initialized by the package's own init_weights (std sqrt(2 / (fan_in + fan_out))); every
other layer is unchanged. ntrl-demo's first block adds its input to the 256-wide hidden state (x = (1 - w) x_in +
w sin(y)); with appended features that residual carries the 256 base features, so the appended ones enter through
the gating vectors u, v and the first block's linear map. Every add spec of the same width m has the same shapes, the
same initialization and the same data order, so a sharp feature and its plain control differ only in the features.

The plan's arrays are buffers (in the checkpoint's model_state_dict), so load_field rebuilds the plan from (spec, B, E,
n, seed) and a strict load checks it.
"""
import math
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import Linear

HERE = Path(__file__).resolve().parent
ROOT = next((q for q in HERE.parents if (q / 'baselines/ntrl-demo').is_dir()), Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield')))
sys.path.insert(0, str(HERE))
import features as FT  # noqa: E402

H_SIZE = 256


def network_module():
    """ntrl-demo's models/metric network module (unchanged on disk)."""
    ntrl = ROOT / 'baselines/ntrl-demo'
    if str(ntrl) not in sys.path:
        sys.path.insert(0, str(ntrl))
    from models.metric import model_network_metric as mn
    assert Path(mn.__file__).resolve() == ntrl / 'models/metric/model_network_metric.py', 'ntrl-demo models package shadowed'
    return mn


_CLASSES = {}


def sharp_class(NN):
    """Subclass of ntrl-demo's NN class (pass the class itself: the trainer patch replaces mn.NN while it runs)."""
    if id(NN) in _CLASSES:
        return _CLASSES[id(NN)]

    class SharpNN(NN):
        def __init__(self, device, dim, B, plan):
            super().__init__(device, dim, B)
            self.spec = plan['spec']
            self.extra = int(plan['extra_width'])
            self.rep_parts, self.add_parts = [], []
            for k, blk in enumerate(plan['replace']):
                self.register_buffer(f'rep{k}_mask', torch.tensor(blk['mask'], dtype=torch.bool))
                self.register_buffer(f'rep{k}_source', torch.tensor(blk['source'], dtype=torch.long))
                partner = bool((blk['source'] != np.arange(len(blk['source']))).any())
                if blk['p'] is not None:
                    self.register_buffer(f'rep{k}_p', torch.tensor(blk['p'], dtype=torch.float32))
                self.rep_parts.append(dict(part=blk['part'], partner=partner, has_p=blk['p'] is not None))
            for k, blk in enumerate(plan['add']):
                info = dict(part=blk['part'], new=blk['rows'] is None, gate='h' in blk, has_p='p' in blk, width=int(blk['width']))
                if info['new']:
                    self.register_buffer(f'add{k}_B', torch.tensor(blk['B_new'].T.copy(), dtype=torch.float32))   # (3, M) like self.B
                else:
                    self.register_buffer(f'add{k}_rows', torch.tensor(blk['rows'], dtype=torch.long))
                if info['gate']:
                    self.register_buffer(f'add{k}_grows', torch.tensor(blk['gate_rows'], dtype=torch.long))
                    self.register_buffer(f'add{k}_h', torch.tensor(blk['h'], dtype=torch.float32))
                if info['has_p']:
                    self.register_buffer(f'add{k}_p', torch.tensor(blk['p'], dtype=torch.float32))
                self.add_parts.append(info)
            if self.extra:
                self.pe_gate[0] = Linear(H_SIZE + self.extra, H_SIZE)
                self.pe_gate[1] = Linear(H_SIZE + self.extra, H_SIZE)
                self.encoder[1] = Linear(H_SIZE + self.extra, H_SIZE)
            # no .to(device) here: the trainer applies init_weights on the CPU first (CPU generator), then moves the net

        def input_mapping(self, x):
            if not self.rep_parts and not self.add_parts:
                return super().input_mapping(x)
            w = 2. * np.pi * self.B
            x_proj = x @ w
            S, C = torch.sin(x_proj), torch.cos(x_proj)
            for k, info in enumerate(self.rep_parts):
                mask = getattr(self, f'rep{k}_mask')
                z = x_proj[:, getattr(self, f'rep{k}_source')] if info['partner'] else x_proj
                f1, f2 = FT.pair(info['part'], z, getattr(self, f'rep{k}_p') if info['has_p'] else None)
                S = torch.where(mask, f1, S)
                C = torch.where(mask, f2, C)
            feats = [S, C]
            for k, info in enumerate(self.add_parts):
                part = info['part']
                p_rows = getattr(self, f'add{k}_p') if info['has_p'] else None
                if info['gate']:
                    z = x_proj[:, getattr(self, f'add{k}_grows')]
                    feats.append(FT.add_features(part, z, None, getattr(self, f'add{k}_h')))
                    continue
                if info['new']:
                    z = x @ (2. * np.pi * getattr(self, f'add{k}_B'))
                else:
                    z = x_proj[:, getattr(self, f'add{k}_rows')]
                feats.append(FT.add_features(part, z, p_rows))
            out = torch.cat(feats, dim=-1)
            assert out.shape[-1] == H_SIZE + self.extra
            return out

        def out(self, coords):
            if not self.extra:
                return super().out(coords)
            # ntrl-demo's NN.out with one change: the first block's residual takes the 256 base features
            coords = coords.clone().detach().requires_grad_(True)
            size = coords.shape[0]
            x0 = coords[:, :self.dim]
            x1 = coords[:, self.dim:]
            x = torch.vstack((x0, x1))
            x = self.input_mapping(x)
            w = self.pe_gate[0].weight
            b = self.pe_gate[0].bias
            w = self.lip_norm(w)
            u = torch.sin(x @ w.T + b)
            w = self.pe_gate[1].weight
            b = self.pe_gate[1].bias
            w = self.lip_norm(w)
            v = torch.sin(x @ w.T + b)
            for ii in range(0, self.nl1):
                x_tmp = x[:, :H_SIZE] if ii == 0 else x
                w = self.encoder[3 * ii + 1].weight
                b = self.encoder[3 * ii + 1].bias
                w = self.lip_norm(w)
                y = x @ w.T + b
                x = u * torch.sin(y) + v * (1 - torch.sin(y))
                w = self.encoder[3 * ii + 2].weight
                b = self.encoder[3 * ii + 2].bias
                w = self.lip_norm(w)
                y = x @ w.T + b
                x = u * torch.sin(y) + v * (1 - torch.sin(y))
                w = self.encoder[3 * ii + 3].weight
                b = self.encoder[3 * ii + 3].bias
                w = self.lip_norm(w)
                y = x @ w.T + b
                weight = torch.sigmoid(0.1 * self.gate[ii].weight)
                x = (1 - weight) * x_tmp + (weight) * torch.sin(y)
            w = self.encoder[-1].weight
            b = self.encoder[-1].bias
            w = self.lip_norm(w)
            y = x @ w.T + b
            y = self.encoder_norm(y)
            x0 = y[:size, ...]
            x1 = y[size:, ...]
            x = torch.sqrt((x0 - x1) ** 2 + 1e-6)
            x = x.view(x.shape[0], -1, 16)
            x = (torch.logsumexp(10 * x, 2) - np.log(16)) / 10
            x = 0.2 * (torch.sum(x, dim=1, keepdim=True))
            return x, w, coords

    _CLASSES[id(NN)] = SharpNN
    return SharpNN


def build(NN, device, B, plan):
    """The network as the trainer constructs it (the trainer then applies init_weights and moves it)."""
    return sharp_class(NN)(device, 3, B, plan)


def load_field(checkpoint, spec, E, n, seed, device='cuda:0'):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    mn = network_module()
    B = payload['B_state_dict']
    plan = FT.plan(spec, B, E, n, seed)
    net = build(mn.NN, device, B, plan)
    net.load_state_dict(payload['model_state_dict'], strict=True)
    ref = build(mn.NN, device, B, plan)                     # buffers rebuilt from the spec must equal the checkpoint's
    for name, buf in ref.named_buffers():
        assert torch.equal(buf.to(device), dict(net.named_buffers())[name].to(device)), f'plan buffer {name} differs from the checkpoint'
    del ref
    net.to(torch.device(device)).float().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def parameter_count(net):
    return int(sum(p.numel() for p in net.parameters()))
