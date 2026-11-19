"""Where the failed runs end on the gap mazes (planning seed 0 paths): report/gap_stalls.json.

    for k in 1 2 4 8 12; do python jeffrey/embed_warp/gapdiag.py $k; done     # seed-87358 runs in report/gap_runs.json (gappage.py first);
                                                                             # one gap per process (gap.install is once per process)

For each query with a quadrant end that failed by never arriving (stall), the last MPPI position's region relative to the query:
'same' (a same-quadrant query), 'start quadrant' (never left it), 'band' (left it, stopped in the open band), 'goal quadrant' (reached it, stopped inside), 'other quadrant';
plus the distance to the goal (cells) and, for stalls in the start quadrant, how far from the start (cells). Runs that arrived through a
wall are counted by the same regions of their first collision-free -> collision step is not tracked here (MPPI appends the goal).
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = HERE / 'report'
sys.path.insert(0, str(HERE))
sys.path.append(str(ROOT / 'jeffrey/cross_warp'))


def main():
    import torch
    import scene as SC
    import gap as GP
    data = json.load(open(REPORT / 'gap_runs.json'))
    only = int(sys.argv[1])
    f = REPORT / 'gap_stalls.json'
    out = json.load(open(f)) if f.exists() else {}
    scenes = {}
    for r in data['rows']:
        if r['seed'] != 87358 or r['gap'] != only:
            continue
        prep = ROOT / r['prepared']
        if str(prep) not in scenes:
            scenes[str(prep)] = SC.load_scene(prep, 'cuda:0')[0]
        scene = scenes[str(prep)]
        q = np.load(prep / 'queries.npz')
        d = ROOT / 'results' / f'registered_{r["id"]}'
        rows = np.load(d / 'rows.npz')
        paths = np.load(d / 'paths_seed0.npy', allow_pickle=True)
        conv, ok = rows['converged'][0], rows['success'][0]
        n = scene.n
        cells = lambda x: np.array([float(x[0]) * scene.E + n / 2, float(x[1]) * scene.E + n / 2])
        cnt = dict(same=0, start=0, band=0, goal=0, other=0)
        dist_goal, dist_start = [], []
        for k in range(len(paths)):
            if ok[k] or conv[k] or q['category'][k] == 4:
                continue
            p = np.asarray(paths[k], np.float32)
            if len(p) < 2:
                continue
            end = p[-2]                                   # last MPPI position (the goal is appended)
            c = cells(end)
            reg = int(GP.region_of(n, c[0], c[1]))
            rs, rg = int(q['region_start'][k]), int(q['region_goal'][k])
            key = 'same' if q['category'][k] == 0 else 'start' if reg == rs and rs < 4 else 'goal' if reg == rg and rg < 4 else 'band' if reg == 4 else 'other'
            cnt[key] += 1
            g, s = cells(q['goal'][k]), cells(q['start'][k])
            dist_goal.append(float(np.hypot(*(c[:2] - g[:2]))))
            if key == 'start':
                dist_start.append(float(np.hypot(*(c[:2] - s[:2]))))
        tot = sum(cnt.values())
        out[r['id']] = dict(gap=r['gap'], label=r['label'], stalls=tot, share={k: v / max(tot, 1) for k, v in cnt.items()}, counts=cnt,
                            goal_dist_median=float(np.median(dist_goal)) if dist_goal else None, start_moved_median=float(np.median(dist_start)) if dist_start else None)
        print(f'x{r["gap"]:>2} {r["label"]:8s} stalls {tot:4d}  ' + '  '.join(f'{k} {100 * v / max(tot, 1):4.0f}%' for k, v in cnt.items()) +
              f'  goal dist {out[r["id"]]["goal_dist_median"]}  moved in start q {out[r["id"]]["start_moved_median"]}')
    json.dump(out, open(f, 'w'), indent=1)


if __name__ == '__main__':
    main()
