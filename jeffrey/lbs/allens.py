"""Allensville: which queries fail in the LBS and no-segment runs, and where their plans stop -> report/allensville.json.

Asked on 2026-11-20 ("I want a re-run ... make sure it is actually harmful"): LBS solved 92.5 % on Allensville against 100.0 % for the
no-segment run reused from jeffrey/pair_lift. This reads the registered runs only (rows.npz = success per planning seed and query,
paths_seed0.npy = the plans of planning seed 0, results.json = success during training).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/lbs/allens.py
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCENE = ROOT / 'results/registered_20261117T054449Z-0144646b'
ROOM = dict(x_max=-2.0, y_min=11.0)        # cells; the room every failing query of the same-seed pair starts in (found from the failures)


def in_room(p):
    return (p[:, 0] < ROOM['x_max']) & (p[:, 1] > ROOM['y_min'])


def main():
    E = json.load(open(SCENE / 'maze.json'))['E_cells']
    q = np.load(SCENE / 'queries.npz')
    start = q['start'] * E
    room = in_room(start)
    d = json.load(open(HERE / 'report/runs.json'))
    rows = [r for r in d['runs'] + d.get('extra', []) if r['scene'] == 'Allensville' and r['arm'] in ('lbs', 'noS')]
    out = dict(room=ROOM, queries=int(len(start)), queries_from_room=int(room.sum()), long_from_room=int((room & (q['bin'] == 5)).sum()), long_queries=int((q['bin'] == 5).sum()), runs=[])
    base = None
    for r in sorted(rows, key=lambda r: (r['seed'], r['arm'], r['source'])):
        rd = ROOT / f"results/registered_{r['id']}"
        s = np.load(rd / 'rows.npz')['success']
        bad = np.where(~s[0])[0]
        paths = np.load(rd / 'paths_seed0.npy', allow_pickle=True)
        ends = np.array([np.asarray(paths[i])[-2][:2] * E for i in bad]) if len(bad) else np.zeros((0, 2))
        course = [dict(epoch=c['epoch'], pct=c['pct'], fails=c['runs'] - c['success']) for c in r['course']]
        for ep in (1000, 2000, 3000, 4000):                                  # the earlier study kept the rows of its course points
            f = rd / f'course_{ep:05d}/rows.npz'
            if f.exists():
                cs = np.load(f)['success']
                cs = cs[0] if cs.ndim == 2 else cs
                for c in course:
                    if c['epoch'] == ep:
                        c['fails_from_room'] = int((~cs & room).sum())
        if not any(c['epoch'] == 5000 for c in course):
            course.append(dict(epoch=5000, pct=100 * float(s[0].mean()), fails=int((~s[0]).sum()), fails_from_room=int((~s[0] & room).sum())))
        row = dict(id=r['id'], arm=r['arm'], seed=r['seed'], source=r['source'], solved=r['solved'], fails_per_planning_seed=[int((~s[i]).sum()) for i in range(len(s))],
                   same_set_each_planning_seed=bool(all((s[i] == s[0]).all() for i in range(len(s)))), fails_from_room=int((~s[0] & room).sum()),
                   stall_end_median=[float(v) for v in np.median(ends, 0)] if len(ends) else None, stall_end_spread=float(np.abs(ends - np.median(ends, 0)).max()) if len(ends) else None,
                   stall_ends=[[round(float(a), 2), round(float(b), 2)] for a, b in ends], course=course, fails=[int(i) for i in bad])
        if r['arm'] == 'noS' and r['seed'] == 87358 and r['source'] == 'jeffrey/lbs':
            base = set(row['fails'])
        out['runs'].append(row)
    for row in out['runs']:
        row['shared_with_same_seed_no_segment'] = len(set(row['fails']) & base) if base is not None else None
    json.dump(out, open(HERE / 'report/allensville.json', 'w'), indent=1)
    print(f"room: {out['queries_from_room']} of {out['queries']} queries start there ({out['long_from_room']} of the {out['long_queries']} longest)")
    for row in out['runs']:
        print(row['arm'], row['seed'], row['source'], f"{row['solved']:.1f}", 'fails', row['fails_per_planning_seed'], 'from room', row['fails_from_room'], 'stall at', row['stall_end_median'],
              'course', [(c['epoch'], c['fails'], c.get('fails_from_room')) for c in row['course']])


if __name__ == '__main__':
    main()
