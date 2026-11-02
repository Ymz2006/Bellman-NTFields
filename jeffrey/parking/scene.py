"""Parking scene: a rectangular car with configuration (x, y, theta) in a small walled lot with two bays.

Configuration space = ntrl-demo's box [-0.5, 0.5]^3 with q = (x, y, theta / 2 pi) (the arm package's q / 2 pi
convention for angles). The lot is the workspace square [-0.5, 0.5]^2, about 3.3 car lengths a side, closed by
outer walls. Two bays open towards -x on the right; the divider between them (thickness w) is the wall the sweep
thins. Getting from bay A (upper) to bay B (lower): back out (-x), move down (-y), drive in (+x).

    y=+0.5 +--------------------------------+
           |                     +----------|
           |       aisle         |  block   |
           |                     +----------|
           |                     |  bay A   |   <- car parked nose-in (theta = 0)
           |                     +==========|   <- divider, thickness w
           |                     |  bay B   |
           |                     +----------|
           |                     |  block   |
           |                     +----------|
    y=-0.5 +--------------------------------+
         x=-0.5                 X_D       x=+0.5

Clearance is the exact signed distance between the car rectangle and the union of wall boxes (2-D convex
polygons: the closest points of disjoint rectangles always include a vertex of one of them, so the minimum of
the 8 vertex-to-rectangle distances is exact; overlapping rectangles get minus the separating-axis penetration
depth, which is exact for convex polygons). Speed label = ntrl-demo's point recipe clip(d / margin, offset /
margin, 1) with margin = limit / 15 (its default) and offset = margin / 10.
"""
import math
import numpy as np
import torch

CAR_L, CAR_W = 0.30, 0.13           # car rectangle (length along its heading, width; Scene(car_w=) for the C-space gap control)
BOUND, T_WALL = 0.5, 0.02           # lot half-size, outer wall thickness
X_D = 0.10                          # bays (and divider) occupy x in [X_D, 0.48]; the aisle is x < X_D
CLEAR = 0.05                        # bay side clearance of a centred car
BAY_W = CAR_W + 2 * CLEAR           # bay width along y (default car)
PARK_X = 0.29                       # nominal parked x (car front 0.04 from the back wall, rear 0.04 inside the bay)
THETA = 2 * math.pi                 # q[2] = theta / THETA
LIMIT = 0.5
MARGIN = LIMIT / 15.0               # ntrl-demo default margin (speed_sampling_gpu_kdtree_normal.py, non-'test' tasks)
OFFSET = MARGIN / 10.0
ALPHA = 1.025                       # ntrl-demo models/metric training stretch: alpha * s^2 (2 - s)^2 + 1 - alpha
WALLS = [0.30, 0.20, 0.14, 0.10, 0.07, 0.05, 0.035, 0.02, 0.01, 0.005]
NAMES = ['left wall', 'back wall', 'bottom wall', 'top wall', 'divider', 'block above A', 'block below B']


def boxes(w, car_w=CAR_W, bay_w=None):
    """Obstacle boxes [xmin, ymin, xmax, ymax] for divider thickness w (index 4 is the divider); bays are bay_w wide
    (default car_w + 2 CLEAR, i.e. 0.05 side clearance for a centred car)."""
    i = BOUND - T_WALL
    bay = car_w + 2 * CLEAR if bay_w is None else bay_w
    assert 0 < w and w / 2 + bay < i - 0.04 and bay > car_w, 'divider too thick for the lot, or car wider than its bay'
    return np.array([
        [-BOUND, -BOUND, -i, BOUND],
        [i, -BOUND, BOUND, BOUND],
        [-BOUND, -BOUND, BOUND, -i],
        [-BOUND, i, BOUND, BOUND],
        [X_D, -w / 2, i, w / 2],
        [X_D, w / 2 + bay, i, i],
        [X_D, -i, i, -w / 2 - bay],
    ], dtype=np.float64)


def bay_y(w, bay, car_w=CAR_W, bay_w=None):
    """Centre line of bay 'A' (upper) or 'B' (lower); bays are bay_w wide (default car_w + 2 CLEAR)."""
    return (1 if bay == 'A' else -1) * (w / 2 + (car_w + 2 * CLEAR if bay_w is None else bay_w) / 2)


class Scene:
    def __init__(self, w, device='cuda:0', dtype=torch.float32, car_w=CAR_W, bay_w=None):
        self.w = float(w)
        self.car_w = float(car_w)
        self.bay_w = self.car_w + 2 * CLEAR if bay_w is None else float(bay_w)
        self.side_gap = (self.bay_w - self.car_w) / 2       # side clearance of a centred parked car
        self.device = device
        self.dtype = dtype
        b = torch.tensor(boxes(w, car_w, self.bay_w), dtype=dtype, device=device)
        self.centre = (b[:, :2] + b[:, 2:]) / 2          # (M, 2)
        self.half = (b[:, 2:] - b[:, :2]) / 2            # (M, 2)
        self.corners = torch.stack([self.centre + self.half * torch.tensor(s, dtype=dtype, device=device)
                                    for s in ([1, 1], [1, -1], [-1, -1], [-1, 1])], 1)   # (M, 4, 2)

    def signed(self, q):
        """Signed clearance of each wall box, (N, M); negative = penetration depth."""
        q = q.to(self.dtype)
        th = q[:, 2] * THETA
        c, s = torch.cos(th), torch.sin(th)
        u = torch.stack([c, s], -1)                       # heading
        v = torch.stack([-s, c], -1)
        p = q[:, :2]
        hl, hw = CAR_L / 2, self.car_w / 2
        car = torch.stack([p + a * hl * u + b * hw * v for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))], 1)   # (N, 4, 2)
        # car corners -> boxes (point-to-AABB distance)
        e = (car[:, None] - self.centre[None, :, None]).abs() - self.half[None, :, None]          # (N, M, 4, 2)
        d1 = torch.sqrt((e.clamp(min=0) ** 2).sum(-1) + 1e-12).amin(-1)                       # (N, M)
        # box corners -> car (in the car frame)
        r = self.corners[None] - p[:, None, None]                                                  # (N, M, 4, 2)
        lu = (r * u[:, None, None]).sum(-1).abs() - hl
        lv = (r * v[:, None, None]).sum(-1).abs() - hw
        d2 = torch.sqrt(lu.clamp(min=0) ** 2 + lv.clamp(min=0) ** 2 + 1e-12).amin(-1)
        separated = torch.minimum(d1, d2)
        # separating axes x, y, u, v: gap = |centre distance| - projected half extents
        k = p[:, None] - self.centre[None]                                                          # (N, M, 2)
        axes = torch.stack([torch.ones_like(c), torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)], -1).view(-1, 2, 2)
        axes = torch.cat([axes, u[:, None], v[:, None]], 1)                                        # (N, 4, 2)
        dist = (k[:, :, None] * axes[:, None]).sum(-1).abs()                                       # (N, M, 4)
        rc = hl * (axes * u[:, None]).sum(-1).abs() + hw * (axes * v[:, None]).sum(-1).abs()      # (N, 4)
        rb = (self.half[None, :, None] * axes[:, None].abs()).sum(-1)                              # (N, M, 4)
        gap = (dist - rc[:, None] - rb).amax(-1)                                                   # (N, M)
        return torch.where(gap > 0, separated, gap)

    def clearance(self, q, chunk=262144):
        """Signed clearance min over boxes and the index of the closest box."""
        ds, idx = [], []
        for i in range(0, len(q), chunk):
            d = self.signed(q[i:i + chunk])
            m, j = d.min(1)
            ds.append(m)
            idx.append(j)
        return torch.cat(ds), torch.cat(idx)

    def clearance_normal(self, q, chunk=262144):
        """Clearance and its unit gradient in the C-space coordinates (ntrl-demo's normal label)."""
        ds, ns = [], []
        for i in range(0, len(q), chunk):
            x = q[i:i + chunk].detach().clone().requires_grad_(True)
            with torch.enable_grad():
                d = self.signed(x).amin(1)
                g, = torch.autograd.grad(d.sum(), x)
            ds.append(d.detach())
            ns.append(torch.nn.functional.normalize(g, dim=1))
        return torch.cat(ds), torch.cat(ns)

    def unit_speed(self, d):
        """ntrl-demo label s = clip(d / margin, offset / margin, 1)."""
        return (d / MARGIN).clamp(OFFSET / MARGIN, 1.0)

    def train_speed(self, d):
        """The speed the metric trainer fits: alpha s^2 (2 - s)^2 + 1 - alpha."""
        s = self.unit_speed(d)
        return ALPHA * s * s * (2 - s) * (2 - s) + 1 - ALPHA


def parked(w, bay, n, gen, device, car_w=CAR_W, bay_w=None):
    """n parked poses in a bay: x in [0.27, 0.31], y within 0.015 of the bay centre, |theta| <= 3 deg. In bays with
    less than CLEAR side clearance the y range shrinks to 0.3 x the side gap and the theta range to 0.3 x the tilt that
    closes it (2 gap / CAR_L), so a parked pose keeps >= 0.4 x the gap."""
    gap = CLEAR if bay_w is None else (bay_w - car_w) / 2
    tight = gap < CLEAR - 1e-9
    dy = 0.3 * gap if tight else 0.015
    dth = min(math.radians(3), 0.6 * gap / CAR_L) if tight else math.radians(3)
    r = torch.rand(n, 3, generator=gen, device=device) * 2 - 1
    q = torch.stack([PARK_X + 0.02 * r[:, 0], bay_y(w, bay, car_w, bay_w) + dy * r[:, 1], dth / THETA * r[:, 2]], 1)
    return q


def aisle(w, n, gen, device):
    """n aisle poses: x in [-0.30, -0.06], y in [-0.32, 0.32], |theta| <= 20 deg."""
    r = torch.rand(n, 3, generator=gen, device=device) * 2 - 1
    return torch.stack([-0.18 + 0.12 * r[:, 0], 0.32 * r[:, 1], math.radians(20) / THETA * r[:, 2]], 1)


X_BACK = X_D - CAR_L / 2 - 0.06      # witness: car centre x with the front 0.06 clear of the bay mouth


def witness(start, goal):
    """Certifying path (back out, move along y in the aisle, drive in); theta is reset to 0 while backing out."""
    a = torch.stack([torch.full_like(start[:, 0], X_BACK), start[:, 1], torch.zeros_like(start[:, 2])], 1)
    b = torch.stack([torch.full_like(start[:, 0], X_BACK), goal[:, 1], torch.zeros_like(start[:, 2])], 1)
    return torch.stack([start, a, b, goal], 1)                      # (N, 4, 3)
