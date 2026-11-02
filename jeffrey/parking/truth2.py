"""truth.py for thin C-space walls: every edge cost is integrated from the speed sampled along the edge.

truth.Grid prices an edge from its two end nodes only, so an edge whose ends are both outside a wall thinner than
the grid steps (0.005 in x, y; 98-neighbourhood edges span up to 0.01) passes through it for free. Here each of the
98 edge directions gets the exact speed at N_SUB - 1 interior points (spacing <= 0.0014 for the longest edge) and
the cost is the trapezoid rule of 1/S along the edge, so walls down to ~0.002 thick are priced. Same grid, solver,
interpolation and interface as truth.Grid.
"""
import math
import torch
import torch.nn.functional as F

import truth as T1
from truth import HX, HT, THETA_MAX, RADIUS

N_SUB = 8


class Grid(T1.Grid):
    def __init__(self, scene):
        dev = scene.device
        self.scene = scene
        self.xs = torch.linspace(-0.5, 0.5, int(round(1 / HX)) + 1, device=dev)
        self.ts = torch.linspace(-THETA_MAX, THETA_MAX, int(round(2 * THETA_MAX / HT)) + 1, device=dev)
        self.shape = (len(self.xs), len(self.xs), len(self.ts))
        q = torch.stack(torch.meshgrid(self.xs, self.xs, self.ts, indexing='ij'), -1).reshape(-1, 3)
        d, _ = scene.clearance(q)
        self.clear = d.reshape(self.shape)
        self.speed = scene.train_speed(d).reshape(self.shape)
        inv = 1 / self.speed
        r = RADIUS
        pad = F.pad(inv[None, None], (r, r, r, r, r, r), value=float('inf'))[0, 0]
        nx, ny, nt = self.shape
        self.offsets, self.costs = [], []
        step = torch.tensor([HX, HX, HT], device=dev)
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                for c in range(-r, r + 1):
                    if (a, b, c) == (0, 0, 0) or math.gcd(math.gcd(abs(a), abs(b)), abs(c)) != 1:
                        continue
                    length = math.sqrt((a * HX) ** 2 + (b * HX) ** 2 + (c * HT) ** 2)
                    nb = pad[r + a:r + a + nx, r + b:r + b + ny, r + c:r + c + nt]
                    total = 0.5 * (inv + nb)
                    o = torch.tensor([a, b, c], device=dev, dtype=q.dtype) * step
                    for k in range(1, N_SUB):
                        dk, _ = scene.clearance(q + o * (k / N_SUB))
                        total = total + 1 / scene.train_speed(dk).reshape(self.shape)
                    self.offsets.append((a, b, c))
                    self.costs.append(length * total / N_SUB)
