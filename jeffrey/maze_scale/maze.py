"""Maze scenes for the maze-scale test: n x n cell mazes, a rectangular robot in SE(2), ntrl-demo's normalization.

Maze units: cell pitch 1, cells [0, n]^2, walls of thickness T_WALL centred on the grid lines; each wall box spans its
cell edge plus T_WALL / 2 at both ends, so adjacent walls join (no free-standing posts). Layout: a perfect maze
(randomized Kruskal over the cell graph), then every remaining interior wall is removed with probability LOOP_P, which
opens loops as in ntrl-demo's demo maze (datasets/test/maze: 7 x 7 cells, open). The same generator and parameters
make every size; SIZES = 4, 7, 13, 22, 40 cells a side (about x1.78 per step, 10x from the first to the last).

Normalization = ntrl-demo's to_off (dataprocessing/convert_to_scaled_off.py): centre on the bounding box of the mesh
and divide by its largest extent, E = n + T_WALL (the wall height H_OBJ < E), so the maze spans [-0.5, 0.5]^2 and the
normalized pitch is 1 / E. write_obj() writes the mesh in maze units; to_off on it gives exactly these coordinates.

Robot: a ROBOT_L x ROBOT_W rectangle (cell units), heading theta in [-pi/2, pi/2] (the rectangle is pi-symmetric, so
this interval holds every orientation; the C-space is the box, no wrap-around). Configuration q = (x, y, phi), x and y
normalized, phi = RHO * theta / E: rotating by d theta costs like moving RHO d theta cells, at every maze size (the
C-space metric is the same in cell units, so a bigger maze is the same world with more cells). On the corridor centre
line the robot can turn in place (half-diagonal 0.27 < half-corridor 0.45); driving along a corridor broadside leaves
0.2 clearance at its ends (slow), aligned 0.35 (fast).

Clearance: exact signed distance between the robot rectangle and the union of wall boxes (as jeffrey/parking/scene.py:
disjoint convex polygons -> min of the 8 vertex-to-polygon distances; overlapping -> minus the separating-axis
penetration depth), evaluated against the walls on the edges of the 3 x 3 block of cells around the robot's cell
(24 candidate edges): exact whenever the clearance is below 0.68 cells (walls outside the block are farther), and
>= 0.68 otherwise. Speed label: ntrl-demo's point recipe clip(d / margin, offset / margin, 1) with
margin = MARGIN_CELLS cells (0.175 = ntrl-demo's 'test' maze ratio: margin limit/20 = 0.025 at pitch 1/7) and
offset = margin / 10; the metric trainer stretches it to ALPHA s^2 (2 - s)^2 + 1 - ALPHA (floor 0.012).
"""
import math
import numpy as np
import torch

SIZES = [4, 7, 13, 22, 40]
T_WALL = 0.1                 # wall thickness (cells)
LOOP_P = 0.15                # extra interior walls removed after the perfect maze
H_OBJ = 0.5                  # wall height in the .obj (cells); < E so to_off scales by the xy extent
ROBOT_L, ROBOT_W = 0.5, 0.2  # robot rectangle (cells): length along its heading, width
RHO = 0.25                   # cells per radian in the C-space metric (about the robot's half length)
MARGIN_CELLS = 0.175         # ntrl-demo margin in cells
OFFSET_CELLS = MARGIN_CELLS / 10
ALPHA = 1.025                # models/metric training stretch
EXACT_RANGE = 0.68           # clearance is exact below this (cells)
REF_N = 4                    # planner scales are ntrl-demo's defaults at this maze size


def maze_seed(n):
    return 1000 + n


def generate(n, seed, loop_p=LOOP_P):
    """Wall occupancy: V (n+1, n) vertical walls at x = i over y in [j, j+1]; H (n, n+1) horizontal walls at y = j over
    x in [i, i+1]. Border walls always present."""
    rng = np.random.default_rng(seed)
    V = np.ones((n + 1, n), bool)
    H = np.ones((n, n + 1), bool)
    parent = np.arange(n * n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edges = [('V', i, j) for i in range(1, n) for j in range(n)] + [('H', i, j) for i in range(n) for j in range(1, n)]
    for k in rng.permutation(len(edges)):
        kind, i, j = edges[k]
        a, b = ((i - 1) * n + j, i * n + j) if kind == 'V' else (i * n + j - 1, i * n + j)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            (V if kind == 'V' else H)[i, j] = False
    extra = rng.random(len(edges))
    for k, (kind, i, j) in enumerate(edges):
        W = V if kind == 'V' else H
        if W[i, j] and extra[k] < loop_p:
            W[i, j] = False
    return V, H


def wall_boxes(V, H):
    """[xmin, ymin, xmax, ymax] in cell units for every present wall."""
    h = T_WALL / 2
    out = [[i - h, j - h, i + h, j + 1 + h] for i, j in zip(*np.nonzero(V))]
    out += [[i - h, j - h, i + 1 + h, j + h] for i, j in zip(*np.nonzero(H))]
    return np.array(out, dtype=np.float64)


def write_obj(path, boxes, height=H_OBJ):
    """Each box extruded to z in [0, height], 8 vertices and 12 outward triangles, maze units."""
    lines = ['# maze-scale maze, cell units (jeffrey/maze_scale/maze.py)\n']
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        for z in (0.0, height):
            lines += [f'v {x0:.6f} {y0:.6f} {z:.6f}\n', f'v {x1:.6f} {y0:.6f} {z:.6f}\n', f'v {x1:.6f} {y1:.6f} {z:.6f}\n', f'v {x0:.6f} {y1:.6f} {z:.6f}\n']
        lines += [f'f {8 * k + a + 1} {8 * k + b + 1} {8 * k + c + 1}\n' for a, b, c in faces]
    with open(path, 'w') as f:
        f.writelines(lines)


def read_obj_vertices(path):
    return np.array([[float(x) for x in l.split()[1:4]] for l in open(path) if l.startswith('v ')])


def to_off_transform(vertices):
    """ntrl-demo to_off: centre = bounding-box centre, scale = 1 / largest extent."""
    hi, lo = vertices.max(0), vertices.min(0)
    return (hi + lo) / 2, float((hi - lo).max())


def rect_box_signed(p, th, centre, half, hl, hw):
    """Signed distance between rectangles (centre p (N, 2), heading th (N,), half extents hl, hw) and axis-aligned boxes
    (centre, half: (N, K, 2)) -> (N, K); negative = penetration depth."""
    c, s = torch.cos(th), torch.sin(th)
    u = torch.stack([c, s], -1)
    v = torch.stack([-s, c], -1)
    car = torch.stack([p + a * hl * u + b * hw * v for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))], 1)    # (N, 4, 2)
    e = (car[:, None] - centre[:, :, None]).abs() - half[:, :, None]                                     # (N, K, 4, 2)
    d1 = torch.sqrt((e.clamp(min=0) ** 2).sum(-1) + 1e-12).amin(-1)
    sign = torch.tensor([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=p.dtype, device=p.device)
    corners = centre[:, :, None] + half[:, :, None] * sign                                                # (N, K, 4, 2)
    r = corners - p[:, None, None]
    lu = (r * u[:, None, None]).sum(-1).abs() - hl
    lv = (r * v[:, None, None]).sum(-1).abs() - hw
    d2 = torch.sqrt(lu.clamp(min=0) ** 2 + lv.clamp(min=0) ** 2 + 1e-12).amin(-1)
    separated = torch.minimum(d1, d2)
    k = p[:, None] - centre                                                                                 # (N, K, 2)
    one, zero = torch.ones_like(c), torch.zeros_like(c)
    axes = torch.stack([torch.stack([one, zero], -1), torch.stack([zero, one], -1), u, v], 1)            # (N, 4, 2)
    dist = (k[:, :, None] * axes[:, None]).sum(-1).abs()                                                    # (N, K, 4)
    rc = hl * (axes * u[:, None]).sum(-1).abs() + hw * (axes * v[:, None]).sum(-1).abs()                    # (N, 4)
    rb = (half[:, :, None] * axes[:, None].abs()).sum(-1)                                                   # (N, K, 4)
    gap = (dist - rc[:, None] - rb).amax(-1)
    return torch.where(gap > 0, separated, gap)


class Maze:
    def __init__(self, n, device='cuda:0', dtype=torch.float32, seed=None):
        self.n = int(n)
        self.seed = maze_seed(n) if seed is None else int(seed)
        self.V, self.H = generate(self.n, self.seed)
        self.boxes = wall_boxes(self.V, self.H)
        self.E = self.n + T_WALL                          # largest extent of the mesh (cells)
        self.pitch = 1.0 / self.E                         # normalized cell pitch
        self.phi_max = RHO * (math.pi / 2) / self.E       # phi range [-phi_max, phi_max]
        self.margin = MARGIN_CELLS * self.pitch           # normalized (for reference; labels are computed in cells)
        self.offset = OFFSET_CELLS * self.pitch
        self.lo = torch.tensor([-0.5, -0.5, -self.phi_max], device=device, dtype=dtype)
        self.hi = torch.tensor([0.5, 0.5, self.phi_max], device=device, dtype=dtype)
        self.device, self.dtype = device, dtype
        self.Vt = torch.tensor(self.V, device=device)
        self.Ht = torch.tensor(self.H, device=device)

    # --- coordinates ---------------------------------------------------------------------------------------------
    def to_cells(self, q):
        """(x, y, theta) in cell units / radians from normalized q."""
        return q[:, 0] * self.E + self.n / 2, q[:, 1] * self.E + self.n / 2, q[:, 2] * self.E / RHO

    def from_cells(self, x, y, th):
        return torch.stack([(x - self.n / 2) / self.E, (y - self.n / 2) / self.E, th * RHO / self.E], -1)

    def planner_scale(self):
        """Pitch relative to the REF_N maze: MPPI step and tolerance scale with it, its temperature with 1 / it."""
        return self.pitch / (1.0 / (REF_N + T_WALL))

    # --- clearance -----------------------------------------------------------------------------------------------
    def candidates(self, x, y):
        """Centres and half extents (N, 24, 2) of the walls on the edges of the 3 x 3 cells around (x, y); absent
        walls are moved far away."""
        n, dev = self.n, x.device
        ci = torch.floor(x).long().clamp(0, n - 1)
        cj = torch.floor(y).long().clamp(0, n - 1)
        di = torch.arange(-1, 3, device=dev)
        dj = torch.arange(-1, 2, device=dev)
        vi = (ci[:, None, None] + di[None, :, None]).expand(-1, 4, 3).reshape(len(x), 12)
        vj = (cj[:, None, None] + dj[None, None, :]).expand(-1, 4, 3).reshape(len(x), 12)
        hi = (ci[:, None, None] + dj[None, :, None]).expand(-1, 3, 4).reshape(len(x), 12)
        hj = (cj[:, None, None] + di[None, None, :]).expand(-1, 3, 4).reshape(len(x), 12)
        vok = (vi >= 0) & (vi <= n) & (vj >= 0) & (vj <= n - 1)
        hok = (hi >= 0) & (hi <= n - 1) & (hj >= 0) & (hj <= n)
        vok = vok & self.Vt[vi.clamp(0, n), vj.clamp(0, n - 1)]
        hok = hok & self.Ht[hi.clamp(0, n - 1), hj.clamp(0, n)]
        h = T_WALL / 2
        vc = torch.stack([vi.to(x.dtype), vj.to(x.dtype) + 0.5], -1)
        hc = torch.stack([hi.to(x.dtype) + 0.5, hj.to(x.dtype)], -1)
        vh = torch.tensor([h, 0.5 + h], device=dev, dtype=x.dtype).expand(len(x), 12, 2)
        hh = torch.tensor([0.5 + h, h], device=dev, dtype=x.dtype).expand(len(x), 12, 2)
        centre = torch.cat([vc, hc], 1)
        half = torch.cat([vh, hh], 1)
        ok = torch.cat([vok, hok], 1)
        centre = torch.where(ok[..., None], centre, torch.full_like(centre, 1e4))
        return centre, half

    def signed_cells(self, q):
        """Signed clearance in cell units (N,) of normalized configurations q (N, 3)."""
        x, y, th = self.to_cells(q.to(self.dtype))
        centre, half = self.candidates(x.detach(), y.detach())
        d = rect_box_signed(torch.stack([x, y], -1), th, centre, half, ROBOT_L / 2, ROBOT_W / 2)
        return d.amin(1)

    def clearance(self, q, chunk=65536):
        """Signed clearance in cells (N,)."""
        with torch.no_grad():
            return torch.cat([self.signed_cells(q[i:i + chunk]) for i in range(0, len(q), chunk)]) if len(q) else q.new_zeros(0)

    def clearance_normal(self, q, chunk=65536):
        """Clearance (cells) and its unit gradient with respect to normalized q (ntrl-demo's normal label)."""
        ds, ns = [], []
        for i in range(0, len(q), chunk):
            x = q[i:i + chunk].detach().clone().to(self.dtype).requires_grad_(True)
            with torch.enable_grad():
                d = self.signed_cells(x)
                g, = torch.autograd.grad(d.sum(), x)
            ds.append(d.detach())
            ns.append(torch.nn.functional.normalize(g, dim=1))
        return torch.cat(ds), torch.cat(ns)

    @staticmethod
    def unit_speed(d_cells):
        """ntrl-demo label s = clip(d / margin, offset / margin, 1)."""
        return (d_cells / MARGIN_CELLS).clamp(OFFSET_CELLS / MARGIN_CELLS, 1.0)

    @staticmethod
    def train_speed(d_cells):
        s = Maze.unit_speed(d_cells)
        return ALPHA * s * s * (2 - s) * (2 - s) + 1 - ALPHA

    # --- sampling ------------------------------------------------------------------------------------------------
    def uniform(self, count, gen):
        return self.lo + (self.hi - self.lo) * torch.rand(count, 3, generator=gen, device=self.device, dtype=self.dtype)

    def free(self, count, gen, min_clear=0.0, batch=262144):
        """count configurations uniform in the C-space box with clearance >= min_clear (cells)."""
        out, have = [], 0
        while have < count:
            q = self.uniform(batch, gen)
            q = q[self.clearance(q) >= min_clear]
            out.append(q)
            have += len(q)
        return torch.cat(out)[:count]

    def inside(self, q):
        return ((q >= self.lo) & (q <= self.hi)).all(1)

    def describe(self):
        return dict(n=self.n, seed=self.seed, walls=int(len(self.boxes)), interior_walls_present=int(self.V[1:-1].sum() + self.H[:, 1:-1].sum()),
                    E_cells=self.E, pitch=self.pitch, phi_max=self.phi_max, margin=self.margin, offset=self.offset, robot=[ROBOT_L, ROBOT_W],
                    rho=RHO, t_wall=T_WALL, loop_p=LOOP_P, planner_scale=self.planner_scale())
