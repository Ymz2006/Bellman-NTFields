"""Travel-time fields for segment DP on mazes: a chain of ntrl-demo networks over distance segments.

ChainField: model 0 (the base, e.g. ntrl-demo's trained field) owns gaps |q - g| < e_1, model m owns [e_m, e_(m+1)],
the last model everything above; across every inner edge e_k the weight ramps linearly from the lower to the upper
model over [e_k - b_k, e_k + b_k] (b_k = blend * e_k), so the field and its gradient are continuous (as
jeffrey/segment_dp/fields.py SegmentField, with per-edge blend widths for geometric edges). The gap is the C-space
distance in normalized coordinates (x, y, phi). Every model is an ntrl-demo models/metric NN (same B-matrix
convention, loaded with its own B).
"""
import torch
import plan as P


class ChainField:
    def __init__(self, models, edges, blend):
        """models: list of NN (len L + 1); edges: inner edges e_1..e_L (normalized gap units, increasing)."""
        assert len(models) == len(edges) + 1, (len(models), len(edges))
        self.models = models
        self.edges = torch.tensor(edges, dtype=torch.float32, device=next(models[0].parameters()).device) if len(edges) else None
        self.blend = float(blend)

    def ramps(self, gap):
        """(P, L) share of the upper model at each inner edge, and its derivative with respect to the gap."""
        lo = self.edges * (1 - self.blend)
        width = 2 * self.blend * self.edges
        r = (gap[:, None] - lo) / width
        inside = (r > 0) & (r < 1)
        return r.clamp(0, 1), inside.float() / width

    def weights(self, gap):
        if self.edges is None:
            return gap.new_ones(len(gap), 1), gap.new_zeros(len(gap), 1)
        r, dr = self.ramps(gap)
        one, zero = gap.new_ones(len(gap), 1), gap.new_zeros(len(gap), 1)
        w = torch.cat([one, r], 1) - torch.cat([r, zero], 1)
        dw = torch.cat([zero, dr], 1) - torch.cat([dr, zero], 1)
        return w, dw

    def cost(self, pairs, ids=None, chunk=200_000):
        out = []
        for s in range(0, len(pairs), chunk):
            x = pairs[s:s + chunk]
            gap = torch.linalg.vector_norm(x[:, :3] - x[:, 3:], dim=1)
            w, _ = self.weights(gap)
            t = x.new_zeros(len(x))
            for m, net in enumerate(self.models):
                rows = torch.nonzero(w[:, m] > 0).squeeze(1)
                if len(rows):
                    with torch.no_grad():
                        t = t.index_add(0, rows, w[rows, m] * net.out(x[rows])[0][:, 0].detach())
            out.append(t)
        return torch.cat(out) if out else pairs.new_zeros(0)

    def cost_grad(self, pairs, chunk=100_000):
        """T and dT/dx0 of the blended field (product rule through the gap-dependent weights)."""
        ts, gs = [], []
        for s in range(0, len(pairs), chunk):
            x = pairs[s:s + chunk]
            d = x[:, :3] - x[:, 3:]
            gap = torch.linalg.vector_norm(d, dim=1)
            w, dw = self.weights(gap)
            unit = d / gap.clamp(min=1e-12)[:, None]
            t = x.new_zeros(len(x))
            g = x.new_zeros(len(x), 3)
            for m, net in enumerate(self.models):
                rows = torch.nonzero(w[:, m] > 0).squeeze(1)
                if not len(rows):
                    continue
                tm, gm = P.field_grad(net, x[rows])
                t = t.index_add(0, rows, w[rows, m] * tm)
                g = g.index_add(0, rows, w[rows, m, None] * gm + (dw[rows, m] * tm)[:, None] * unit[rows])
            ts.append(t)
            gs.append(g)
        return torch.cat(ts), torch.cat(gs)


def save_chain(path, base_checkpoint, edges, blend, state_dicts, Bs, **extra):
    torch.save(dict(kind='maze_chain', base_checkpoint=str(base_checkpoint), edges=[float(e) for e in edges], blend=float(blend),
                    level_state_dicts=state_dicts, level_B=Bs, **extra), path)


def load_chain(path, device='cuda:0', upto=None):
    """ChainField from save_chain(); upto = number of DP levels to keep (None = all)."""
    payload = torch.load(path, map_location=device, weights_only=False)
    assert payload['kind'] == 'maze_chain'
    models = [P.load_field(payload['base_checkpoint'], 'metric', device)]
    L = len(payload['level_state_dicts']) if upto is None else upto
    NN = P.network_class('metric')
    for sd, B in zip(payload['level_state_dicts'][:L], payload['level_B'][:L]):
        net = NN(device, 3, B)
        net.load_state_dict(sd, strict=True)
        net.to(torch.device(device)).float().eval()
        for p in net.parameters():
            p.requires_grad_(False)
        models.append(net)
    return ChainField(models, payload['edges'][:L], payload['blend']), payload
