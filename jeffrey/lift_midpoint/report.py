"""Tables of the lift + midpoint study from registered runs: report/runs.json (every number the page uses) and report/tables.md.

Arms per scene (seed 87358):
    ntrl      ntrl-demo (jeffrey/maze_scale ntrl stage on the 10/13/22 mazes, jeffrey/lift_scale '--lift none' elsewhere)
    lift      the travel-time lift alone, no push (jeffrey/lift_scale '--lift time')
    liftmid   lift + travel-time midpoint loss from epoch 2,000 (this study; identical to 'lift' up to epoch 2,000)
    mid       the midpoint loss without the lift (jeffrey/midpoint_dp MAIN on the mazes; this study '--lift none' elsewhere)
    liftpush  lift + push, the lift page's schedule (jeffrey/lift_scale; reference only)
Field measures of runs that did not compute them (maze_scale ntrl, midpoint_dp) come from posthoc.py on the same truth pairs.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCENES = ['maze10', 'maze13', 'maze22', 'plus22', 'plus30', 'plus40', 'Allensville', 'Lakeville', 'Klickitat']
TITLE = dict(maze10='maze 10 x 10', maze13='maze 13 x 13', maze22='maze 22 x 22', plus22='plus maze 22 x 22 (+ of 6)', plus30='plus maze 30 x 30 (+ of 8)',
             plus40='plus maze 40 x 40 (+ of 12)', Allensville='Gibson Allensville (43 cells)', Lakeville='Gibson Lakeville (83 cells)', Klickitat='Gibson Klickitat (96 cells)')
ARMS = ['ntrl', 'lift', 'liftmid', 'mid', 'liftpush', 'drive']
ARM_NAME = dict(ntrl='ntrl-demo', lift='lift alone', liftmid='lift + midpoint', mid='midpoint alone', liftpush='lift + push (reference)', drive='lift + midpoint, drives by the ntrl-demo field (control)')
FIXED = dict(
    maze10=dict(ntrl='20261102T032216Z-e76476bf', mid='20261102T081615Z-0adefad6'),
    maze13=dict(ntrl='20261101T203437Z-ebc95225', mid='20261102T081356Z-c2f195aa'),
    maze22=dict(ntrl='20261101T193703Z-dd44f02b', mid='20261102T084156Z-6809fed7'),
    plus22=dict(ntrl='20261117T054519Z-9a3bc556', lift='20261117T055249Z-d953394e', liftpush='20261117T061710Z-c95ca37c'),
    plus30=dict(ntrl='20261117T120340Z-057a1ff3', lift='20261117T120955Z-aa3b4619', liftpush='20261117T121530Z-ea7160ea'),
    plus40=dict(ntrl='20261117T070240Z-9926f070', lift='20261117T070630Z-a905919d', liftpush='20261117T071105Z-e272648c'),
    Allensville=dict(ntrl='20261117T121605Z-9c558da3', lift='20261117T122340Z-11cef54b', liftpush='20261117T122655Z-ff2c4b49'),
    Lakeville=dict(ntrl='20261117T094420Z-31c51802', lift='20261117T094615Z-a1c691d1', liftpush='20261117T104235Z-b6379951'),
    Klickitat=dict(ntrl='20261117T104445Z-0c50416c', lift='20261117T105600Z-740e99b6', liftpush='20261117T110015Z-d7107d88'))
MAZE_PREP = {'20261102T032150Z-b0baa8e9': 'maze10', '20261101T192504Z-13cf1d04': 'maze13', '20261101T193403Z-cea05796': 'maze22'}


def discovered():
    """Run ids of this study (queue/state.json) and of the lift-alone maze controls (work/logs)."""
    out = {}
    st = HERE / 'queue/state.json'
    if st.exists():
        for job, rid in json.load(open(st)).get('done', {}).items():
            label, key, seed, extra = job.split('|')
            arm = label if label in ('liftmid', 'mid', 'drive') else None
            if arm and seed == '87358':
                out.setdefault(key, {})[arm] = rid
    for log in sorted((HERE / 'work/logs').glob('lift_*.log')):
        m = re.search(r'"id": "([^"]+)"', log.read_text())
        prep = log.stem[len('lift_'):]
        if m and prep in MAZE_PREP and (ROOT / f'results/registered_{m.group(1)}/results.json').exists():
            out.setdefault(MAZE_PREP[prep], {})['lift'] = m.group(1)
    return out


def pct(a, b):
    return 100.0 * a / b if b else None


def summarize(rid, post):
    run = ROOT / f'results/registered_{rid}'
    r = json.load(open(run / 'results.json'))
    p = post.get(rid, {})
    acc = r.get('accuracy') or p.get('accuracy') or {}
    fail = r.get('failures') or p.get('failures') or {}
    wall = r.get('wall_fit') or p.get('wall_fit') or {}
    ok, runs = r['success_total']
    det = acc.get('detour', {})
    row = dict(run=rid, stage=r['stage'], solved=pct(ok, runs), success=[ok, runs], wall=pct(fail.get('arrived_wall', 0), runs) if fail else None,
               stall=pct(fail.get('not_arrived', 0), runs) if fail else None, by_bin={k: pct(v[0], v[1]) for k, v in r['success_by_bin'].items()},
               within10=acc.get('within10'), ratio=acc.get('ratio_median'), los_within10=acc.get('los', {}).get('within10'), los_ratio=acc.get('los', {}).get('ratio_median'),
               rho=det.get('rho_median'), seen_through=det.get('seen_through'), detour_within10=det.get('within10'),
               pairs_n=acc.get('n'), detour_n=det.get('n'), los_n=acc.get('los', {}).get('n'),
               detour_by_length={k: dict(true=v['true_cells'], learned=v['learned_cells'], rho=v['rho_median'], n=v['n']) for k, v in det.get('by_length', {}).items()},
               wall_innermost=wall.get('innermost'), descent_cos=r.get('field', {}).get('descent_cos_median'), t_over_tstar_queries=r.get('field', {}).get('t_over_tstar_median'),
               minutes=(r.get('elapsed_seconds') or 0) / 60, gpu=r.get('gpu'))
    q = runs // 3 if runs % 3 == 0 else runs
    p0 = r.get('planning', {}).get('0', {})
    row['seed0'] = pct(p0['success'], q) if 'success' in p0 else None
    curve = []
    for e, c in sorted(((int(k), v) for k, v in (r.get('course') or {}).items()), key=lambda t: t[0]):
        if isinstance(c.get('success'), list):
            curve.append([e, pct(c['success'][0], c['success'][1])])
    for v in r.get('validate', []) or []:
        curve.append([int(v['epoch']), pct(v['success'], v.get('runs', q))])
    if row['seed0'] is not None:
        curve.append([int(r.get('epochs', r.get('config', {}).get('epochs', 5000)) or 5000), row['seed0']])
    row['curve'] = sorted({e: v for e, v in curve}.items())
    if r['stage'] == 'lift-midpoint-field':
        row['refresh'] = [dict(epoch=s['epoch'], reached=s['reached'], stalled=s['stalled'], collided=s['collided'], traversals=s['traversals'],
                               detour_share=s.get('reached_long_detour_share'), path_over_line=s.get('reached_path_over_line_median'), seconds=s['seconds'],
                               steps_mean=s['steps_mean'], log_t_over_y=s['log_net_over_target_median']) for s in r.get('refresh', [])]
        row['validate'] = [dict(epoch=v['epoch'], success=v['success'], runs=v['runs'], wall=v['arrived_wall'], stall=v['not_arrived']) for v in r.get('validate', [])]
        row['reached'] = [[s['epoch'], pct(s['reached'], s['traversals'])] for s in r.get('refresh', [])]
        row['detour_drives'] = [[s['epoch'], pct(s['reached'] * (s.get('reached_long_detour_share') or 0.0), s['traversals'])] for s in r.get('refresh', [])]
        row['course'] = r.get('course', {})
        row['train_minutes'] = r['train_seconds'] / 60
    elif 'course' in r:
        row['course'] = r['course']
    return row


def main():
    post = json.load(open(HERE / 'report/posthoc.json')) if (HERE / 'report/posthoc.json').exists() else {}
    found = discovered()
    out = dict(scenes={}, order=SCENES, arms=ARMS, arm_name=ARM_NAME, title=TITLE)
    for sc in SCENES:
        ids = dict(FIXED.get(sc, {}))
        ids.update(found.get(sc, {}))
        arms = {}
        for arm in ARMS:
            rid = ids.get(arm)
            if rid and (ROOT / f'results/registered_{rid}/results.json').exists():
                arms[arm] = summarize(rid, post)
        if arms:
            out['scenes'][sc] = arms
    (HERE / 'report').mkdir(exist_ok=True)
    (HERE / 'report/runs.json').write_text(json.dumps(out, indent=1, default=float))
    f = lambda v, k=1: '' if v is None else (f'{v:.{k}f}' if isinstance(v, float) else str(v))
    lines = ['# lift_midpoint tables (report.py; every number from registered runs, seed 87358)', '',
             '| scene | arm | solved % | wall % | stall % | T within 10 % | median T/T* | line-of-sight within 10 % | LOS T/T* | detour index | seen through % | S x slope innermost | run |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for sc, arms in out['scenes'].items():
        for arm in ARMS:
            if arm not in arms:
                continue
            r = arms[arm]
            lines.append(f'| {TITLE[sc]} | {ARM_NAME[arm]} | {f(r["solved"])} | {f(r["wall"])} | {f(r["stall"])} | {f(r["within10"] and 100 * r["within10"])} | {f(r["ratio"], 3)} | '
                         f'{f(r["los_within10"] and 100 * r["los_within10"])} | {f(r["los_ratio"], 3)} | {f(r["rho"], 2)} | {f(r["seen_through"] and 100 * r["seen_through"])} | {f(r["wall_innermost"], 2)} | {r["run"]} |')
    lines += ['', '## Success by true distance (cells), % of runs', '']
    for sc, arms in out['scenes'].items():
        bins = sorted({b for r in arms.values() for b in r['by_bin']}, key=lambda s: float(s.split('-')[0]))
        lines += [f'### {TITLE[sc]}', '', '| arm | ' + ' | '.join(bins) + ' |', '|---|' + '---:|' * len(bins)]
        for arm in ARMS:
            if arm in arms:
                lines.append(f'| {ARM_NAME[arm]} | ' + ' | '.join(f(arms[arm]['by_bin'].get(b), 0) for b in bins) + ' |')
        lines.append('')
    lines += ['## Lift + midpoint during training (planning seed 0 on the prepared queries; refresh = 8,192 traversals of the current field)', '']
    for sc, arms in out['scenes'].items():
        r = arms.get('liftmid')
        if not r:
            continue
        v = ', '.join(f'{x["epoch"]}: {pct(x["success"], x["runs"]):.1f} % (stall {x["stall"]}, wall {x["wall"]})' for x in r.get('validate', []))
        rf = r.get('refresh', [])
        s = ', '.join(f'{x["epoch"]}: reached {x["reached"]}, detour share {f(x["detour_share"], 3)}' for x in rf[::5] + rf[-1:])
        lines += [f'* {TITLE[sc]}: validate {v}', f'  refresh {s}', f'  training {r.get("train_minutes", 0):.0f} min, whole run {r["minutes"]:.0f} min on {r["gpu"]}']
    (HERE / 'report/tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:60]))


if __name__ == '__main__':
    main()
