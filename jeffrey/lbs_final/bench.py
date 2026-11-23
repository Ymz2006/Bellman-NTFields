"""CPU benchmark of one prepared scene of the LBS final benchmark: every planner solves the scene's 500 random queries one at a time on the
CPU (--threads torch threads; no batching), and each query's planning time, success and path length are recorded.

    python bench.py --prepared <prep run> --methods ntrl,lbs --field ntrl=<field run> --field lbs=<field run> --lbs-src <snapshot> --out <dir>
    python bench.py --prepared <prep run> --methods rrtc,lazyprm --lbs-src <snapshot> --out <dir>

Planners
  ntrl, lbs   ntrl-demo's MPPI (maze_scale/plan.py: plan.settings(scene), planning seed 0, the per-query step budget max(200, 3 x truth path
              C-space length / step)) on the learned travel-time field of a registered field run (the network of epoch 5,000). Time = wall
              time of plan.mppi for that one query (until it reaches the goal or the budget runs out).
  rrtc        OMPL's RRT-Connect (ompl 2.0.1 Python wheel, C++ planner), default settings (range 0.2 x the C-space box diagonal).
  lazyprm     LazyPRM, a Python port of OMPL's geometric/planners/prm/LazyPRM.cpp (not in the 2.0.1 bindings): uniform samples added without
              checks, k = 5 nearest within 0.2 x the diagonal, A* on the roadmap, then the path's vertices are checked (all invalid ones
              removed) and then its edges (the first invalid one removed), until a path survives. A fresh roadmap per query.
  Both sampling planners get --time-limit seconds per query; their path is then shortened with OMPL's PathSimplifier.simplifyMax (what
  SimpleSetup.simplifySolution() does), and time = solve + simplification. The unsimplified length is stored as well.
C-space: the scene's box (SE(2): x, y and the scaled heading phi, no wrap-around, as the networks and MPPI see it; 3-D houses: x, y, z),
Euclidean metric in normalized coordinates (MPPI's step norm). Collision checks are the scorer's own: a state is valid if the exact
clearance is >= 0, a motion if every configuration interpolated at <= 0.005 cells per coordinate is (plan.score's resolution), tested
coarse-to-fine (every 10th point first). Every returned path is scored by plan.score as the learned planners' paths are.
Path length: xy length in cells for SE(2) scenes (plan.score's length_cells), x-y-z length in cells for the 3-D houses (one cell 0.25 m).
"""
import argparse
import hashlib
import heapq
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_scene(prep_a, prep_b):
    """The scene geometry of two prepare runs is identical (maze walls / 3-D house table / 2-D house spec)."""
    a, b = json.load(open(Path(prep_a) / 'maze.json')), json.load(open(Path(prep_b) / 'maze.json'))
    if 'scene' in a or 'scene' in b:
        if a.get('scene', {}).get('kind') == 'house3':
            return sha(Path(prep_a) / 'house3_table.npy') == sha(Path(prep_b) / 'house3_table.npy')
        return a.get('scene') == b.get('scene')
    return a['n'] == b['n'] and a['V'] == b['V'] and a['H'] == b['H']


class Checker:
    """Validity of states and straight motions with the scorer's exact clearance and resolution."""
    def __init__(self, scene, torch, coarse=10):
        self.scene, self.torch, self.coarse = scene, torch, coarse
        self.res = 0.005 * scene.pitch
        self.states = self.motions = self.points = 0

    def clear(self, pts):
        self.points += len(pts)
        return self.scene.clearance(self.torch.as_tensor(pts, dtype=self.torch.float32)).numpy()

    def valid(self, x):
        self.states += 1
        return bool(self.clear(np.asarray(x, np.float64)[None])[0] >= 0)

    def motion(self, p, q):
        self.motions += 1
        p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
        k = max(1, int(math.ceil(np.abs(q - p).max() / self.res)))
        t = np.arange(k + 1) / k
        if k > 2 * self.coarse:
            tc = t[::self.coarse]
            if (self.clear(p + tc[:, None] * (q - p)) < 0).any():
                return False
        return bool((self.clear(p + t[:, None] * (q - p)) >= 0).all())


class LazyPRM:
    """OMPL LazyPRM (geometric/planners/prm/src/LazyPRM.cpp) in Python, single query, default parameters."""

    def __init__(self, lo, hi, checker, seed, k=5, max_distance=None):
        self.lo, self.hi, self.chk, self.k = np.asarray(lo, float), np.asarray(hi, float), checker, k
        self.d = len(lo)
        self.maxd = 0.2 * float(np.linalg.norm(self.hi - self.lo)) if max_distance is None else max_distance
        self.rng = np.random.default_rng(seed)

    def _reset(self):
        self.X = np.zeros((4096, self.d))
        self.alive = np.zeros(4096, bool)
        self.vok = np.zeros(4096, bool)
        self.comp = np.zeros(4096, np.int64)
        self.n, self.ncomp = 0, 0
        self.adj, self.members = [], {}

    def _grow(self):
        m = len(self.X)
        self.X = np.concatenate([self.X, np.zeros((m, self.d))])
        self.alive = np.concatenate([self.alive, np.zeros(m, bool)])
        self.vok = np.concatenate([self.vok, np.zeros(m, bool)])
        self.comp = np.concatenate([self.comp, np.zeros(m, np.int64)])

    def _new_comp(self):
        c = self.ncomp
        self.ncomp += 1
        self.members[c] = set()
        return c

    def _unite(self, a, b):
        ca, cb = int(self.comp[a]), int(self.comp[b])
        if ca == cb:
            return
        if len(self.members[ca]) < len(self.members[cb]):
            ca, cb = cb, ca
        for v in self.members[cb]:
            self.comp[v] = ca
        self.members[ca] |= self.members.pop(cb)

    def _mark(self, v, c):
        """markComponent: every vertex reachable from v gets component c."""
        seen, stack = {v}, [v]
        while stack:
            u = stack.pop()
            for w in self.adj[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        for u in seen:
            old = int(self.comp[u])
            if old in self.members:
                self.members[old].discard(u)
                if not self.members[old]:
                    del self.members[old]
            self.comp[u] = c
        self.members[c] = seen

    def _add(self, x, valid=False):
        if self.n == len(self.X):
            self._grow()
        m = self.n
        self.n += 1
        self.X[m], self.alive[m], self.vok[m] = x, True, valid
        self.adj.append({})
        c = self._new_comp()
        self.comp[m] = c
        self.members[c].add(m)
        if m:
            idx = np.nonzero(self.alive[:m])[0]
            d = np.linalg.norm(self.X[idx] - x, axis=1)
            sel = np.argpartition(d, self.k)[:self.k] if len(idx) > self.k else np.arange(len(idx))
            for j in sel:
                if d[j] <= self.maxd:
                    nb = int(idx[j])
                    e = [float(d[j]), False]                      # weight, checked valid
                    self.adj[m][nb] = e
                    self.adj[nb][m] = e
                    self._unite(m, nb)
        return m

    def _remove_vertex(self, v):
        for w in list(self.adj[v]):
            del self.adj[w][v]
        self.adj[v] = {}
        self.alive[v] = False
        c = int(self.comp[v])
        self.members[c].discard(v)
        if not self.members[c]:
            del self.members[c]

    def _astar(self, s, g):
        xg = self.X[g]
        dist, prev = {s: 0.0}, {s: s}
        heap = [(float(np.linalg.norm(self.X[s] - xg)), s)]
        done = set()
        while heap:
            _, u = heapq.heappop(heap)
            if u == g:
                return prev
            if u in done:
                continue
            done.add(u)
            for w, e in self.adj[u].items():
                nd = dist[u] + e[0]
                if nd < dist.get(w, math.inf):
                    dist[w], prev[w] = nd, u
                    heapq.heappush(heap, (nd + float(np.linalg.norm(self.X[w] - xg)), w))
        return None

    def _construct(self, s, g):
        prev = self._astar(s, g)
        if prev is None:                                          # stale labels (should not happen): relabel and go on
            self._mark(s, self._new_comp())
            self._mark(g, self._new_comp())
            return None
        path = [g]
        while path[-1] != s:
            path.append(prev[path[-1]])
        bad = []
        for u in path[1:-1]:                                      # vertices first; every invalid one is removed
            if not self.vok[u] and self.chk.valid(self.X[u]):
                self.vok[u] = True
            if not self.vok[u]:
                bad.append(u)
        if bad:
            c = int(self.comp[s])
            nbrs = set()
            for u in bad:
                nbrs |= set(self.adj[u])
            nbrs -= set(bad)
            for u in bad:
                self._remove_vertex(u)
            for w in nbrs:
                if int(self.comp[w]) == c:
                    self._mark(w, self._new_comp())
            return None
        for j in range(len(path) - 1):                            # then edges from the goal; the first invalid one is removed
            a, b = path[j + 1], path[j]
            e = self.adj[a][b]
            if not e[1] and self.chk.motion(self.X[a], self.X[b]):
                e[1] = True
            if not e[1]:
                del self.adj[a][b]
                del self.adj[b][a]
                self._mark(a, self._new_comp())                  # the goal side keeps the old label if the edge was a bridge
                return None
        return self.X[path[::-1]].copy()

    def solve(self, start, goal, limit):
        self._reset()
        t0 = time.perf_counter()
        s = self._add(np.asarray(start, float), True)
        g = self._add(np.asarray(goal, float), True)
        attempts = 0
        while time.perf_counter() - t0 < limit:
            if self.comp[s] == self.comp[g]:
                attempts += 1
                p = self._construct(s, g)
                if p is not None:
                    return p, dict(vertices=int(self.alive[:self.n].sum()), samples=self.n - 2, attempts=attempts)
                continue
            self._add(self.lo + (self.hi - self.lo) * self.rng.random(self.d))
        return None, dict(vertices=int(self.alive[:self.n].sum()), samples=self.n - 2, attempts=attempts)


def summarize(rows):
    ok = np.array([r['success'] for r in rows])
    out = dict(n=len(rows), success=int(ok.sum()), rate=float(ok.mean()) if len(rows) else None)
    for key in ('time', 'length_cells'):
        v = np.array([r[key] for r, o in zip(rows, ok) if o], float)
        out[key] = dict(mean=float(v.mean()), std=float(v.std()), median=float(np.median(v))) if len(v) else None
    t = np.array([r['time'] for r in rows], float)
    out['time_all'] = dict(mean=float(t.mean()), std=float(t.std()), median=float(np.median(t))) if len(t) else None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--methods', required=True)
    p.add_argument('--field', action='append', default=[], help='method=<registered field run directory>')
    p.add_argument('--lbs-src', required=True)
    p.add_argument('--time-limit', type=float, default=10.0)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--limit', type=int, default=0, help='only the first N queries (smoke)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, a.lbs_src)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    torch.set_num_threads(a.threads)
    import seeds as SD
    import scenes3 as S3
    import lift as LF
    import plan as P
    from common import Logger, save_paths
    log = Logger(a.out / 'log.txt')
    info = json.load(open(a.prepared / 'maze.json'))
    instance = SD.apply(info)
    scene, info = S3.load_scene(a.prepared, 'cpu')
    three_d = info.get('scene', {}).get('kind') == 'house3'
    L = 3 if three_d else 2
    Q = dict(np.load(a.prepared / 'queries.npz'))
    n = len(Q['start']) if a.limit <= 0 else min(a.limit, len(Q['start']))
    S = torch.tensor(Q['start'][:n])
    G = torch.tensor(Q['goal'][:n])
    lo, hi = scene.lo.numpy().astype(float), scene.hi.numpy().astype(float)
    st = P.settings(scene)
    budget = torch.tensor(np.maximum(st['min_steps'], np.ceil(st['budget_factor'] * Q['path_clen'][:n] / st['step'])))
    res = dict(stage='lbs-final-bench', prepared=str(a.prepared), scene_name=S3.scene_name(info), instance=instance, n_queries=n, threads=a.threads,
               time_limit=a.time_limit, planner=st, cpu=os.cpu_count(), torch=torch.__version__, truth=dict(path_cells=Q['path_cells'][:n].tolist(), d_cells=Q['d_cells'][:n].tolist()),
               methods={}, summary={})
    fields = dict(f.split('=', 1) for f in a.field)

    def length(path):
        x = np.asarray(path, float)
        return float(np.linalg.norm(np.diff(x[:, :L], axis=0), axis=1).sum() * scene.E)

    def score(path, converged=True):
        return P.score(scene, [torch.as_tensor(np.asarray(path), dtype=torch.float32)], [converged])[0]

    for m in a.methods.split(','):
        rows, paths = [], []
        t_m = time.time()
        if m in ('ntrl', 'lbs'):
            run = Path(fields[m])
            fres = json.load(open(run / 'results.json'))
            fprep = Path(fres['prepared'])
            if not fprep.is_absolute():
                fprep = Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield')) / fprep
            if fprep.resolve() != a.prepared.resolve():
                assert same_scene(fprep, a.prepared), f'{run} was trained on another scene'
            net = LF.load_field(run / 'training/checkpoint.pt', scene, 'cpu')
            cost = P.field_cost(net)
            res['methods'][m] = dict(field=str(run), field_prepared=str(fprep), arm=fres.get('arm'), extra=fres.get('args'))
            for i in range(n):
                t0 = time.perf_counter()
                pp, stop, conv = P.mppi(cost, S[i:i + 1], G[i:i + 1], 0, budget[i:i + 1], scene.lo, scene.hi, **st)
                dt = time.perf_counter() - t0
                path = pp[0]
                r = score(path, bool(conv[0]))
                rows.append(dict(success=bool(r['success']), converged=bool(r['converged']), collision=bool(r['collision']), time=dt, steps=int(stop[0]),
                                 budget=int(budget[i]), length_cells=length(path.numpy())))
                paths.append(path.numpy())
                if i % 50 == 0:
                    log(f'{m} {i}/{n}: {sum(x["success"] for x in rows)} solved, {dt:.2f}s')
        elif m in ('rrtc', 'lazyprm'):
            from ompl import base as ob, geometric as og, util as ou
            ou.setLogLevel(ou.LOG_WARN)
            chk = Checker(scene, torch)
            space = ob.RealVectorStateSpace(len(lo))
            b = ob.RealVectorBounds(len(lo))
            for k in range(len(lo)):
                b.setLow(k, float(lo[k]))
                b.setHigh(k, float(hi[k]))
            space.setBounds(b)
            ss = og.SimpleSetup(space)
            ss.setStateValidityChecker(lambda s_: chk.valid([s_[k] for k in range(len(lo))]))
            si = ss.getSpaceInformation()

            class MV(ob.MotionValidator):
                def checkMotion(self, s1, s2, *rest):
                    return chk.motion([s1[k] for k in range(len(lo))], [s2[k] for k in range(len(lo))])
            mv = MV(si)
            si.setMotionValidator(mv)

            def to_state(x):
                s_ = space.allocState()
                for k in range(len(lo)):
                    s_[k] = float(x[k])
                return s_

            def to_array(path):
                return np.array([[s_[k] for k in range(len(lo))] for s_ in path.getStates()])
            if m == 'rrtc':
                planner = og.RRTConnect(si)
                ss.setPlanner(planner)
                ss.setup()
                res['methods'][m] = dict(planner='ompl RRTConnect', range=planner.getRange(), ompl='2.0.1')
            else:
                si.setup()                                        # SimpleSetup.setup() without a planner and start states crashes
                lp = LazyPRM(lo, hi, chk, seed=0)
                res['methods'][m] = dict(planner='LazyPRM (Python port of OMPL LazyPRM.cpp)', k=lp.k, max_distance=lp.maxd)
            simp = og.PathSimplifier(si)
            for i in range(n):
                c0 = (chk.states, chk.motions, chk.points)
                s_np, g_np = Q['start'][i].astype(float), Q['goal'][i].astype(float)
                extra = {}
                if m == 'rrtc':
                    ss.clear()
                    ss.setStartAndGoalStates(to_state(s_np), to_state(g_np))
                    t0 = time.perf_counter()
                    ss.solve(a.time_limit)
                    t_solve = time.perf_counter() - t0
                    found = bool(ss.haveExactSolutionPath())
                    raw = to_array(ss.getSolutionPath()) if found else None
                    gp = og.PathGeometric(ss.getSolutionPath()) if found else None
                else:
                    t0 = time.perf_counter()
                    raw, extra = lp.solve(s_np, g_np, a.time_limit)
                    t_solve = time.perf_counter() - t0
                    found = raw is not None
                    gp = og.PathGeometric(si, [to_state(x) for x in raw]) if found else None
                t_simp, path = 0.0, None
                if found:
                    t1 = time.perf_counter()
                    simp.simplifyMax(gp)
                    t_simp = time.perf_counter() - t1
                    path = to_array(gp)
                    path[0], path[-1] = s_np, g_np
                    r = score(path)
                    ok = bool(r['success'])
                else:
                    ok = False
                c1 = (chk.states, chk.motions, chk.points)
                rows.append(dict(success=ok, found=found, time=t_solve + t_simp, solve_time=t_solve, simplify_time=t_simp,
                                 length_cells=length(path) if found else None, raw_length_cells=length(raw) if found else None,
                                 raw_waypoints=int(len(raw)) if found else 0, state_checks=c1[0] - c0[0], motion_checks=c1[1] - c0[1], points=c1[2] - c0[2], **extra))
                paths.append(path if found else np.zeros((0, len(lo))))
                if i % 50 == 0:
                    log(f'{m} {i}/{n}: {sum(x["success"] for x in rows)} solved, {t_solve:.2f}s + {t_simp:.2f}s')
        else:
            raise SystemExit(f'unknown method {m}')
        res['methods'][m]['rows'] = rows
        res['methods'][m]['seconds'] = time.time() - t_m
        res['summary'][m] = summarize(rows)
        if m in ('rrtc', 'lazyprm'):
            ok = [r for r in rows if r['success']]
            res['summary'][m]['raw_length_cells'] = dict(mean=float(np.mean([r['raw_length_cells'] for r in ok])), std=float(np.std([r['raw_length_cells'] for r in ok]))) if ok else None
            res['summary'][m]['solve_time'] = dict(mean=float(np.mean([r['solve_time'] for r in ok])), std=float(np.std([r['solve_time'] for r in ok]))) if ok else None
        save_paths(a.out / f'paths_{m}.npy', [torch.as_tensor(np.asarray(x, np.float32)) for x in paths])
        log(f'{m}: {json.dumps(res["summary"][m])} ({time.time() - t_m:.0f}s)')
        json.dump(res, open(a.out / 'bench.json', 'w'))
    res['complete'] = True
    json.dump(res, open(a.out / 'bench.json', 'w'))
    log('done')
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
