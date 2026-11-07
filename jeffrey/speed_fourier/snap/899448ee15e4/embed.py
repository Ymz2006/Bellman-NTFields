"""The embedding of the speed-Fourier fields: what replaces ntrl-demo's NN.input_mapping.

ntrl-demo: q -> [sin 2 pi B q, cos 2 pi B q], 256 numbers, then the unchanged network. Here 256 numbers again, so the network,
its parameter count and its seeded initialization stay exactly ntrl-demo's; only the first map changes:

  mix    [sin, cos] of 64 octave rows (every second row of jeffrey/grid_sync's `octave`) + 128 speed rows S_k(x, y)
  top    [sin, cos] of 120 octave rows (every 16th dropped) + 16 speed rows: nearly the whole octave embedding, plus the speed
  pure   8 heading-only rows [sin, cos](2 pi b phi) + 240 speed rows: position enters through the S_k alone
  plain  [sin, cos] of 128 given rows (the environment's own frequencies, or any other B): no speed rows
  half   mix with the speed rows zeroed [floor of mix: what the 64 octave rows do alone]

Speed rows come from linefit.fit as a table on a node grid; a cubic B-spline in (x, y) reads it. Pure tensor ops, so the
gradient with respect to q and its derivatives with respect to the weights exist (the loss differentiates |dT/dq|;
torch.grid_sample has no double backward with respect to the grid). Rows are scaled to mean 0 and RMS 1 / sqrt(2), the RMS of a
sin or cos feature, over the node grid.
"""
import hashlib
import math
import types
import numpy as np
import torch

RMS = 1 / math.sqrt(2)
SPLIT = dict(mix=(64, 128), pure=(8, 240), plain=(128, 0), half=(64, 128), top=(120, 16))


def standardize(table):
    t = table.astype(np.float64)
    t = t - t.mean((0, 1), keepdims=True)
    s = t.std((0, 1), keepdims=True)
    return (t / np.maximum(s, 1e-9) * RMS).astype(np.float32)              # an unused channel (all zero) stays zero


class Embed:
    def __init__(self, rows, table, device):
        """rows: (r, 3) frequency rows or None; table: (R, R, K) standardized or None."""
        self.w = None if rows is None else (2 * math.pi * torch.as_tensor(rows, dtype=torch.float32, device=device)).T.contiguous()
        self.table = None
        if table is not None:
            self.R = table.shape[0]
            self.table = torch.as_tensor(table, dtype=torch.float32, device=device).reshape(self.R * self.R, -1).contiguous()

    def speed_rows(self, xy):
        R = self.R
        u = ((xy + 0.5) * (R - 1)).clamp(0, R - 1 - 1e-4)
        base = torch.floor(u.detach())
        f = u - base
        i = base.long()
        wts = torch.stack([(1 - f) ** 3 / 6, (3 * f ** 3 - 6 * f ** 2 + 4) / 6, (-3 * f ** 3 + 3 * f ** 2 + 3 * f + 1) / 6, f ** 3 / 6], 0)    # (4, N, 2)
        out = 0
        for a in range(4):
            ia = (i[:, 0] + a - 1).clamp(0, R - 1) * R
            for b in range(4):
                ib = (i[:, 1] + b - 1).clamp(0, R - 1)
                out = out + (wts[a, :, 0] * wts[b, :, 1])[:, None] * self.table[ia + ib]
        return out

    def __call__(self, q):
        parts = []
        if self.w is not None:
            p = q @ self.w
            parts += [torch.sin(p), torch.cos(p)]
        if self.table is not None:
            parts.append(self.speed_rows(q[:, :2]))
        return torch.cat(parts, -1)


def install(net, embed):
    """Replace net.input_mapping (the instance attribute leaves the state dict and the checkpoint format as they are)."""
    net.input_mapping = types.MethodType(lambda self, x: embed(x), net)
    return net


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]
