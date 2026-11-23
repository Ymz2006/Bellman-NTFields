"""Collect the LBS benchmark: report/runs.json and report/tables.md from the registered runs of queue/state.json plus the runs of earlier
studies that are the same arm on the same scene, queries and planner (REUSE).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/lbs/report.py
"""
import json
import re
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'report'
ARMS = ['lbs', 'noL', 'noB', 'noS', 'ntrl']
ARM_NAME = dict(lbs='LBS (lift + Bellman + segment)', noL='no lift (Bellman + segment)', noB='no Bellman (lift + segment)', noS='no segment (lift + Bellman)', ntrl='ntrl-demo')
G3 = ['Allensville3d', 'Mifflinburg3d', 'Noxapater3d', 'Markleeville3d', 'Lakeville3d']        # 3-D Gibson (asked 2026-11-20): point robot in R^3, LBS and ntrl-demo only
SCENES = ['maze10', 'maze13', 'maze18', 'gap1', 'gap2', 'gap4', 'Allensville', 'Lakeville'] + G3   # Klickitat was removed from the benchmark on 2026-11-20 (request); its runs stay in runs.json
SCENE_TITLE = dict(maze10='maze 10 x 10', maze13='maze 13 x 13', maze18='maze 18 x 18', gap1='gap maze x1 (10 x 10 base, 16 cells)', gap2='gap maze x2 (22 cells)',
                   gap4='gap maze x4 (34 cells)', Allensville='Gibson Allensville (43 cells)', Lakeville='Gibson Lakeville (83 cells)', Klickitat='Gibson Klickitat (96 cells)', **{s: f'Gibson {s[:-2]} in 3-D' for s in G3})
# earlier registered runs that are an arm of this benchmark on the same prepared scene (same queries, planner and scorer), training seed 87358
REUSE = {('Allensville', 'ntrl'): ('20261117T121605Z-9c558da3', 'jeffrey/lift_scale'), ('Lakeville', 'ntrl'): ('20261117T094420Z-31c51802', 'jeffrey/lift_scale'),
         ('Klickitat', 'ntrl'): ('20261117T104445Z-0c50416c', 'jeffrey/lift_scale'),
         ('Allensville', 'noB'): ('20261117T182927Z-0a0eca6d', 'jeffrey/lift_midpoint'), ('Lakeville', 'noB'): ('20261117T191038Z-55bb04e3', 'jeffrey/lift_midpoint'),
         ('Klickitat', 'noB'): ('20261117T200908Z-309ea4ed', 'jeffrey/lift_midpoint'),
         ('Allensville', 'noS'): ('20261119T111813Z-dfeaeac0', 'jeffrey/pair_lift'), ('Lakeville', 'noS'): ('20261119T105154Z-d5ee7cbe', 'jeffrey/pair_lift'),
         ('Klickitat', 'noS'): ('20261119T103714Z-b462ad74', 'jeffrey/pair_lift')}


def num(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v


def row_of(rid, scene, arm, seed, source):
    d = ROOT / f'results/registered_{rid}'
    if not (d / 'results.json').exists():
        return None
    r = json.load(open(d / 'results.json'))
    tot, fl = r['success_total'], r['failures']
    acc = r.get('accuracy', {})
    det = acc.get('detour', {})
    course = []
    for v in r.get('validate', []):
        course.append(dict(epoch=v['epoch'], pct=100 * v['success'] / v['runs'], success=v['success'], runs=v['runs'], minutes=num(v.get('train_seconds', float('nan')) / 60)))
    if not course:
        for e, v in sorted(r.get('course', {}).items(), key=lambda kv: int(kv[0])):
            if v.get('success'):
                course.append(dict(epoch=int(e), pct=100 * v['success'][0] / v['success'][1], success=v['success'][0], runs=v['success'][1], minutes=None))
    hist = d / 'training/history.json'
    train_min = None
    if hist.exists():
        h = json.load(open(hist))
        train_min = (h.get('seconds', 0) - h.get('validate_seconds', 0)) / 60 if h.get('seconds') else None
    rf = r.get('refresh') or []
    drives = dict(reached=100 * float(np.mean([x['reached'] / x['traversals'] for x in rf])), collided=100 * float(np.mean([x['collided'] / x['traversals'] for x in rf])),
                  stalled=100 * float(np.mean([x['stalled'] / x['traversals'] for x in rf])), pieces=float(np.mean([x['nodes'] for x in rf])), refreshes=len(rf),
                  traverse=r.get('config', {}).get('traverse'), nodes=r.get('config', {}).get('nodes'), leaf=r.get('config', {}).get('leaf')) if rf else None
    out = dict(drives=drives, id=rid, scene=scene, arm=arm, seed=int(seed), source=source, solved=100 * tot[0] / tot[1], success=tot,
                by_bin={k: 100 * v[0] / v[1] for k, v in r['success_by_bin'].items()}, wall=100 * fl['arrived_wall'] / fl['runs'], stall=100 * fl['not_arrived'] / fl['runs'],
                within10=num(100 * acc.get('within10', float('nan'))), ratio=num(acc.get('ratio_median')), seen_through=num(100 * det.get('seen_through', float('nan'))),
                rho=num(det.get('rho_median')), los_within10=num(100 * acc.get('los', {}).get('within10', float('nan'))), course=course,
                train_min=train_min, minutes=r.get('elapsed_seconds', 0) / 60, rollbacks=r.get('rollbacks'), bell=r.get('config', {}).get('bell'), t_max=num(acc.get('t_max')),
                t_star_max=num(acc.get('t_star_max')), by_distance={k: num(v.get('ratio_median')) for k, v in acc.get('by_distance', {}).items() if isinstance(v, dict)},
                course_bins=[dict(epoch=v['epoch'], **{k: x[0] for k, x in v.get('by_bin', {}).items()}) for v in r.get('validate', [])])
    return no_path_pairs(out, acc)


def no_path_pairs(out, acc):
    """3-D houses: the truth grid stores 1000 at nodes it cannot reach (prepare3.py), and the accuracy pairs of diag.truth_pairs keep them
    (the value is finite). Such a pair has no true travel time, and all of them fall in the last distance bin ('>128', median ratio ~0).
    They are taken out here: within10 is recomputed from the other bins (exact, the bins partition the pairs); the statistics that cannot
    be recomputed from the stored numbers (median ratio, detour statistics) are dropped when more than 1 % of the pairs are affected."""
    bd = {k: v for k, v in acc.get('by_distance', {}).items() if isinstance(v, dict)}
    bad = {k: v for k, v in bd.items() if (v.get('ratio_median') or 1) < 0.01}
    if not bad and (acc.get('t_star_max') or 0) < 999:
        return out
    good = {k: v for k, v in bd.items() if k not in bad}
    n_bad, n_good = sum(v['n'] for v in bad.values()), sum(v['n'] for v in good.values())
    out['no_path_share'] = 100 * n_bad / max(1, n_bad + n_good)
    out['t_star_max'] = None
    out['by_distance'] = {k: num(v.get('ratio_median')) for k, v in good.items()}
    if n_bad:
        out['within10'] = 100 * sum(v['n'] * v['within10'] for v in good.values()) / n_good
    if out['no_path_share'] > 1:
        out.update(ratio=None, rho=None, seen_through=None)
    return out


def scenes3d():
    """The prepare run of every 3-D house in the queue: size, queries and the two planner controls on the same queries."""
    sc = json.load(open(HERE / 'queue/scenes.json'))
    out = {}
    for s in G3:
        if s not in sc or not (ROOT / sc[s] / 'results.json').exists():
            continue
        r = json.load(open(ROOT / sc[s] / 'results.json'))
        h = r['house3']
        q = np.load(ROOT / sc[s] / 'queries.npz')
        n = int(len(q['start']))
        out[s] = dict(prepare=Path(sc[s]).name.replace('registered_', ''), name=h['name'], cells=r['maze']['E_cells'], z_cells=h['z_cells'], side_m=h['side_m'], slab=h['slab'], vox_m=h['build']['vox_m'],
                      free_share=100 * r['maze']['free_share'], queries=n, longest_path_cells=float(q['path_cells'].max()), straight=100 * r['straight']['success'] / n,
                      oracle=[100 * v['success'] / n for _, v in sorted(r.get('oracle', {}).items())], truth_grid=r['truth']['grid_shape'], minutes=r['elapsed_seconds'] / 60,
                      per_bin=r['queries']['per_bin'])
    return out


def progress(job):
    """Last training line of a job that is still running (epoch, minutes)."""
    f = HERE / 'queue/logs' / (''.join(c if c.isalnum() or c in '-_.' else '_' for c in job)[:180] + '.log')
    m = re.search(r'"id": "([^"]+)"', f.read_text()) if f.exists() else None
    if not m:
        return None
    log = ROOT / f'results/registered_{m.group(1)}/log.txt'
    if not log.exists():
        return dict(id=m.group(1), epoch=0)
    ep = re.findall(r'epoch (\d+): ntrl loss', log.read_text())
    return dict(id=m.group(1), epoch=int(ep[-1]) if ep else 0)


def main():
    OUT.mkdir(exist_ok=True)
    state = json.load(open(HERE / 'queue/state.json')) if (HERE / 'queue/state.json').exists() else dict(done={}, failed={})
    jobs = [l.strip() for l in open(HERE / 'queue/jobs.txt') if l.strip() and not l.startswith('#')]
    runs, pending, failed, extra, controls = [], [], [], [], []           # controls: jobs whose label is not one of the five arms
    for j in jobs:
        label, scene, seed = j.split('|')[:3]
        if j in state['done']:
            row = row_of(state['done'][j], scene, label, seed, 'jeffrey/lbs')
            if row:
                (runs if label in ARMS else controls).append(row)
        elif j in state['failed']:
            failed.append(dict(scene=scene, arm=label, seed=int(seed), id=state['failed'][j]))
        else:
            pr = progress(j)
            pending.append(dict(scene=scene, arm=label, seed=int(seed), **(pr or {})))
    have = {(r['scene'], r['arm'], r['seed']) for r in runs}
    for (scene, arm), (rid, src) in REUSE.items():
        row = row_of(rid, scene, arm, 87358, src)
        if row:
            (extra if (scene, arm, 87358) in have else runs).append(row)       # extra: the earlier study's run next to this benchmark's own run of the same arm
    json.dump(dict(runs=runs, extra=extra, controls=controls, pending=pending, failed=failed, arms=ARMS, arm_name=ARM_NAME, scenes=SCENES, scene_title=SCENE_TITLE, scenes3d=scenes3d()), open(OUT / 'runs.json', 'w'), indent=1)
    f = lambda v, d=1: '–' if v is None else f'{v:.{d}f}'  # noqa: E731
    lines = ['# LBS benchmark: registered runs', '', '| scene | arm | seed | solved % | through a wall % | stalled % | T within 10 % | median T / T* | detours seen through % | train min | run | from |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for s in SCENES:
        for a in ARMS:
            for r in sorted([r for r in runs if r['scene'] == s and r['arm'] == a], key=lambda r: r['seed']):
                lines.append(f"| {SCENE_TITLE[s]} | {ARM_NAME[a]} | {r['seed']} | {f(r['solved'])} ({r['success'][0]}/{r['success'][1]}) | {f(r['wall'])} | {f(r['stall'])} | {f(r['within10'])} | {f(r['ratio'], 3)} | "
                             f"{f(r['seen_through'])} | {f(r['train_min'], 0)} | {r['id']} | {r['source']} |")
    for r in controls:
        lines.append(f"| {SCENE_TITLE[r['scene']]} | control {r['arm']} | {r['seed']} | {f(r['solved'])} ({r['success'][0]}/{r['success'][1]}) | {f(r['wall'])} | {f(r['stall'])} | {f(r['within10'])} | {f(r['ratio'], 3)} | "
                     f"{f(r['seen_through'])} | {f(r['train_min'], 0)} | {r['id']} | {r['source']} |")
    (OUT / 'tables.md').write_text('\n'.join(lines) + '\n')
    print(f'{len(runs)} runs, {len(pending)} pending, {len(failed)} failed -> {OUT / "tables.md"}')


if __name__ == '__main__':
    main()
