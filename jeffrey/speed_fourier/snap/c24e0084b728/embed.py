"""The embedding of the speed-Fourier fields: what replaces ntrl-demo's NN.input_mapping.

ntrl-demo: q -> [sin 2 pi B q, cos 2 pi B q], 256 numbers, then the unchanged network. Here 256 numbers again, so the network,
its parameter count and its seeded initialization stay exactly ntrl-demo's; only the first map changes:

  mix    [sin, cos] of 64 octave rows (every second row of jeffrey/grid_sync's `octave`) + 128 speed rows S_k(x, y)
  top    [sin, cos] of 120 octave rows (every 16th dropped) + 16 speed rows: nearly the whole octave embedding, plus the speed
  topzero  top with the speed rows zeroed [what the 120 octave rows do alone]
  oracle   top with the 16 speed rows all equal to the exact heading-resolved signal o(x, y, theta), read from a cube of the exact
         clearance: the embedding holds the speed itself [upper bound on what capturing the speed can give]
  pure   8 heading-only rows [sin, cos](2 pi b phi) + 240 speed rows: position enters through the S_k alone
  plain  [sin, cos] of 128 given rows (the environment's own frequencies, the grid-anchored rows of gridrows.py with their phase, or any other B): no speed rows
  half   mix with the speed rows zeroed [floor of mix: what the 64 octave rows do alone]
  geo    top with the 16 speed rows replaced by the corridor distance to 16 landmark cells (linefit.landmark_table) [mechanism control: rows that
         differ between the two sides of a wall, where a speed row reads the same]

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
SPLIT = dict(mix=(64, 128), pure=(8, 240), plain=(128, 0), half=(64, 128), top=(120, 16), topzero=(120, 16), oracle=(120, 16), geo=(120, 16))


def standardize(table):
    t = table.astype(np.float64)
    t = t - t.mean((0, 1), keepdims=True)
    s = t.std((0, 1), keepdims=True)
    return (t / np.maximum(s, 1e-9) * RMS).astype(np.float32)              # an unused channel (all zero) stays zero


class Embed:
    def __init__(self, rows, table, device, cube=None, phi_max=None, copies=16, phase=None):
        """rows: (r, 3) frequency rows or None; table: (R, R, K) standardized or None; cube: (R, R, J) over heading knots;
        phase: (r,) radians added to each row's 2 pi b . q (gridrows.py), or None."""
        self.phase = None if phase is None else torch.as_tensor(phase, dtype=torch.float32, device=device)
        self.w = None if rows is None else (2 * math.pi * torch.as_tensor(rows, dtype=torch.float32, device=device)).T.contiguous()
        self.table = None
        if table is not None:
            self.R = table.shape[0]
            self.table = torch.as_tensor(table, dtype=torch.float32, device=device).reshape(self.R * self.R, -1).contiguous()
        self.cube = None
        if cube is not None:
            self.R, self.J, self.phi_max, self.copies = cube.shape[0], cube.shape[2], phi_max, copies
            self.cube = torch.as_tensor(cube, dtype=torch.float32, device=device).reshape(self.R * self.R, -1).contiguous()

    @staticmethod
    def knots(f):
        return torch.stack([(1 - f) ** 3 / 6, (3 * f ** 3 - 6 * f ** 2 + 4) / 6, (-3 * f ** 3 + 3 * f ** 2 + 3 * f + 1) / 6, f ** 3 / 6], 0)

    def speed_rows(self, xy, table=None):
        table = self.table if table is None else table
        R = self.R
        u = ((xy + 0.5) * (R - 1)).clamp(0, R - 1 - 1e-4)
        base = torch.floor(u.detach())
        f = u - base
        i = base.long()
        wts = self.knots(f)                                                  # (4, N, 2)
        out = 0
        for a in range(4):
            ia = (i[:, 0] + a - 1).clamp(0, R - 1) * R
            for b in range(4):
                ib = (i[:, 1] + b - 1).clamp(0, R - 1)
                out = out + (wts[a, :, 0] * wts[b, :, 1])[:, None] * table[ia + ib]
        return out

    def cube_row(self, q):
        v = self.speed_rows(q[:, :2], self.cube)                             # (N, J)
        J = self.J
        u = ((q[:, 2] / self.phi_max + 1) / 2 * (J - 1)).clamp(0, J - 1 - 1e-4)
        base = torch.floor(u.detach())
        w = self.knots(u - base)                                             # (4, N)
        i = base.long()
        out = 0
        for a in range(4):
            out = out + w[a] * v.gather(1, (i + a - 1).clamp(0, J - 1)[:, None])[:, 0]
        return out

    def __call__(self, q):
        parts = []
        if self.w is not None:
            p = q @ self.w
            if self.phase is not None:
                p = p + self.phase
            parts += [torch.sin(p), torch.cos(p)]
        if self.table is not None:
            parts.append(self.speed_rows(q[:, :2]))
        if self.cube is not None:
            parts.append(self.cube_row(q)[:, None].expand(-1, self.copies))
        return torch.cat(parts, -1)


def install(net, embed):
    """Replace net.input_mapping (the instance attribute leaves the state dict and the checkpoint format as they are)."""
    net.input_mapping = types.MethodType(lambda self, x: embed(x), net)
    return net


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]
