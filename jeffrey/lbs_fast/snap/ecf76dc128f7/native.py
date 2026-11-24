"""ctypes wrapper of native.cpp: the fast LBS planner's whole MPPI loop on the CPU, rows of each step split over threads.

    lib = build()                         compiles native.cpp next to itself into build/native_<hash>.so (g++ -O3 -march=native -fopenmp)
    NativePlanner(field, st, variant='goal' | 'meet', threads=8, samples=None, step_mult=1.0)
        .plan(start, goal, budget, seed) -> dict(path, steps, converged, how)      the interface of fast.Planner
    check(field, pln)                     clearance, features and travel times against the torch field

Supports the exact clearance of maze scenes (maze.Maze and the gap mazes) and of the 3-D houses (scenes3.House3); the random numbers are
the kernel's own (xoshiro256++ / Box-Muller), so paths are statistically, not numerically, those of the torch planner.
"""
import ctypes
import hashlib
import os
import subprocess
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FLAGS = ['-O3', '-march=native', '-mprefer-vector-width=512', '-ffast-math', '-fopenmp', '-shared', '-fPIC']
F32 = ctypes.POINTER(ctypes.c_float)


def build(src=None, cache=None):
    # the kernel's OpenMP runtime: one thread per physical core, spinning between the steps of a query (set before the library loads)
    os.environ.setdefault('OMP_PLACES', 'cores')
    os.environ.setdefault('OMP_PROC_BIND', 'spread')
    os.environ.setdefault('OMP_WAIT_POLICY', 'active')
    src = Path(src or HERE / 'native.cpp')
    cache = Path(cache or os.environ.get('LBS_FAST_BUILD', Path.home() / '.cache/lbs_fast'))
    cache.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(src.read_bytes() + ' '.join(FLAGS).encode()).hexdigest()[:12]
    so = cache / f'native_{tag}.so'
    if not so.exists():
        tmp = so.with_suffix(f'.{os.getpid()}.tmp')
        subprocess.run(['g++', *FLAGS, str(src), '-o', str(tmp)], check=True)
        tmp.replace(so)
    lib = ctypes.CDLL(str(so))
    lib.lbs_create.restype = ctypes.c_void_p
    lib.lbs_create.argtypes = [ctypes.c_int, F32, F32, F32, F32, ctypes.c_int]
    lib.lbs_set_threads.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.lbs_set_lift.argtypes = [ctypes.c_void_p, F32, ctypes.c_int] + [ctypes.c_float] * 6
    lib.lbs_set_maze.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_float] * 5 + [ctypes.POINTER(ctypes.c_uint8)] * 2
    lib.lbs_set_house.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, F32, F32, ctypes.c_float, F32, ctypes.c_float]
    lib.lbs_clearance.argtypes = [ctypes.c_void_p, F32, ctypes.c_int, F32]
    lib.lbs_encode.argtypes = [ctypes.c_void_p, F32, ctypes.c_int, F32]
    lib.lbs_travel.argtypes = [ctypes.c_void_p, F32, ctypes.c_int, F32]
    lib.lbs_plan.restype = ctypes.c_int
    lib.lbs_plan.argtypes = [ctypes.c_void_p, F32, F32, ctypes.c_int, ctypes.c_int, ctypes.c_int] + [ctypes.c_float] * 6 + [ctypes.c_int, ctypes.c_uint64, ctypes.c_float, F32,
                             ctypes.POINTER(ctypes.c_int)]
    return lib


def _f(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return a, a.ctypes.data_as(F32)


class NativeField:
    """The pieces of fast.Field handed to the kernel."""

    def __init__(self, field, threads=8, lib=None):
        from maze import MARGIN_CELLS, OFFSET_CELLS, RHO, ROBOT_L, ROBOT_W, T_WALL
        self.lib = lib or build()
        scene = field.scene
        W, wp = _f(field.W.cpu().numpy())
        ws, wsp = _f(np.stack([w.cpu().numpy() for w, _ in field.layers]))
        bs, bsp = _f(np.stack([b.cpu().numpy() for _, b in field.layers]))
        assert ws.shape == (9, 256, 256) and W.shape[1] == 128 and field.nl1 == 2, 'the kernel is built for ntrl-demo\'s metric network'
        g, gp = _f(np.array([float(x) for x in field.gates]))
        self.h = ctypes.c_void_p(self.lib.lbs_create(W.shape[0], wp, wsp, bsp, gp, int(threads)))
        if field.lift:
            G, Gp = _f(field.table.cpu().numpy())
            self.lib.lbs_set_lift(self.h, Gp, len(G), OFFSET_CELLS, MARGIN_CELLS, field.G_offset, field.floor_slope, field.depth, field.hscale)
            if getattr(scene, 'kind', '') == 'house3':
                t, tp = _f(scene.table.cpu().numpy())
                q0, q0p = _f(scene.q0.cpu().numpy())
                um, ump = _f(scene.umax.cpu().numpy())
                self.lib.lbs_set_house(self.h, *scene.shape, tp, q0p, float(scene.per_q), ump, float(scene.vpc))
            else:
                assert hasattr(scene, 'Vt') and hasattr(scene, 'Ht'), 'the kernel knows maze scenes and 3-D houses'
                V = np.ascontiguousarray(scene.Vt.cpu().numpy().astype(np.uint8))
                H = np.ascontiguousarray(scene.Ht.cpu().numpy().astype(np.uint8))
                u8 = ctypes.POINTER(ctypes.c_uint8)
                self.lib.lbs_set_maze(self.h, int(scene.n), float(scene.E), RHO, ROBOT_L / 2, ROBOT_W / 2, T_WALL, V.ctypes.data_as(u8), H.ctypes.data_as(u8))
        self.lo2, self.hi2 = field.lo2, field.hi2

    def clearance(self, q):
        q, qp = _f(q)
        out = np.zeros(len(q), np.float32)
        self.lib.lbs_clearance(self.h, qp, len(q), out.ctypes.data_as(F32))
        return out

    def encode(self, q):
        q, qp = _f(q)
        out = np.zeros((len(q), 256), np.float32)
        self.lib.lbs_encode(self.h, qp, len(q), out.ctypes.data_as(F32))
        return out

    def travel(self, pairs):
        p, pp = _f(pairs)
        out = np.zeros(len(p), np.float32)
        self.lib.lbs_travel(self.h, pp, len(p), out.ctypes.data_as(F32))
        return out


class NativePlanner:
    def __init__(self, field, st, variant='goal', threads=8, samples=None, horizon=None, step_mult=1.0, freeze_steps=3.0, native=None):
        assert variant in ('goal', 'meet')
        self.nf = native or NativeField(field, threads)
        self.nf.lib.lbs_set_threads(self.nf.h, int(threads))
        self.variant = variant
        self.S, self.H = int(samples or st['sample_num']), int(horizon or st['horizon'])
        self.step = float(st['step']) * float(step_mult)
        self.temperature = float(st['temperature']) / float(step_mult)
        self.tol = float(st['tol'])
        self.freeze = freeze_steps * self.step + self.tol
        self.A = 2 if variant == 'meet' else 1
        self.setup_seconds = 0.0
        self._buf = np.zeros((1024, self.A, 3), np.float32)

    def warm(self, start, goal, calls=3):
        for _ in range(calls):
            self.plan(start, goal, 16, 0)

    def plan(self, start, goal, budget, seed=0):
        s = np.ascontiguousarray(start.detach().cpu().numpy() if torch.is_tensor(start) else start, dtype=np.float32)
        g = np.ascontiguousarray(goal.detach().cpu().numpy() if torch.is_tensor(goal) else goal, dtype=np.float32)
        budget = int(budget)
        if len(self._buf) < budget:
            self._buf = np.zeros((budget, self.A, 3), np.float32)
        conv = ctypes.c_int(0)
        steps = self.nf.lib.lbs_plan(self.nf.h, s.ctypes.data_as(F32), g.ctypes.data_as(F32), int(self.variant == 'meet'), self.S, self.H, self.step, self.temperature,
                                     self.tol, self.freeze, self.nf.lo2, self.nf.hi2, budget, int(seed), 10.0, self._buf.ctypes.data_as(F32), ctypes.byref(conv))
        P = self._buf[:steps]
        if self.variant == 'goal':
            path = np.concatenate([s[None], P[:, 0], g[None]])
        else:
            path = np.concatenate([s[None], P[:, 0], P[::-1, 1], g[None]])
        return dict(path=torch.from_numpy(path.copy()), steps=int(steps), converged=bool(conv.value), how=('goal' if self.variant == 'goal' else 'meet') if conv.value else 'budget')


@torch.no_grad()
def check(field, nf, n=4096, seed=0):
    """Max differences between the kernel and the torch field on random configurations of the C-space box."""
    scene = field.scene
    gen = torch.Generator(scene.device).manual_seed(seed)
    q = scene.uniform(2 * n, gen)
    out = {}
    if field.lift:
        d_t = scene.signed_cells(q).cpu().numpy()
        d_n = nf.clearance(q.cpu().numpy())
        near = d_t < 0.68
        out['clearance'] = float(np.abs(d_t - d_n)[near].max())
    f_t = field.encode(q).cpu().numpy()
    f_n = nf.encode(q.cpu().numpy())
    out['features'] = float(np.abs(f_t - f_n).max())
    pairs = torch.cat([q[:n], q[n:]], 1)
    out['travel'] = float(np.abs(field.travel(pairs).cpu().numpy() - nf.travel(pairs.cpu().numpy())).max())
    out['travel_mean'] = float(field.travel(pairs).mean())
    return out
