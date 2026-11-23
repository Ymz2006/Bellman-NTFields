"""Collect every finished pair-lift run of the queue (queue/state.json: job -> registered id) and the reference runs of jeffrey/lift_scale
into report/runs.json and report/tables.md.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/pair_lift/report.py
"""
import json
import re
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'report'
# jeffrey/lift_scale runs quoted as references (label -> run ids by scene; same pipeline, no pair coordinate)
REFERENCES = {'plus22': {'ntrl-demo': ['20261117T054519Z-9a3bc556'], 'lift': ['20261117T055249Z-d953394e'], 'lift + push': ['20261117T061710Z-c95ca37c'],
                         'lift + push + octaves': ['20261117T093110Z-3477083e'], 'lift + strong push + octaves': ['20261117T101750Z-b9e07f67']}}
SCENE_TITLE = dict(plus22='plus maze 22 x 22', plus30='plus maze 30 x 30', plus40='plus maze 40 x 40', apt32='apartment, 32 cells', apt64='apartment, 64 cells',
                   Allensville='house Allensville, 43 cells', Lakeville='house Lakeville, 83 cells', Klickitat='house Klickitat, 96 cells',
                   plus22in='plus maze 22 x 22, 20 % of the pairs with one end inside an obstacle',
                   plus22w='plus maze 22 x 22, whole-interior pairs (55 % of the first ends inside an obstacle)', plus22w3='plus maze 22 x 22, whole-interior pairs (a third of the first ends inside)',
                   plus40in='plus maze 40 x 40, 20 % of the pairs with one end inside an obstacle', apt64in='apartment, 64 cells, 20 % of the pairs with one end inside an obstacle')


def nan(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def row_of(rid, scene, label, seed, extra, kind):
    d = ROOT / f'results/registered_{rid}'
    if not (d / 'results.json').exists():
        return None
    r = json.load(open(d / 'results.json'))
    acc = r['accuracy']
    det = acc.get('detour', {})
    row = dict(id=rid, scene=scene, label=label, seed=int(seed), extra=extra, kind=kind, within10=100 * acc['within10'], ratio=acc['ratio_median'],
               los_within10=100 * acc.get('los', {}).get('within10', np.nan), los_ratio=acc.get('los', {}).get('ratio_median'), rho=det.get('rho_median'),
               seen_through=100 * det.get('seen_through', np.nan), blocked_within10=100 * acc.get('segment_blocked', {}).get('within10', np.nan),
               detour_by_length=det.get('by_length', {}), minutes=r['elapsed_seconds'] / 60)
    if 'success_total' in r:
        tot, f = r['success_total'], r['failures']
        row.update(solved=100 * tot[0] / tot[1], success=tot, wall=100 * f['arrived_wall'] / f['runs'], stall=100 * f['not_arrived'] / f['runs'],
                   by_bin={k: 100 * v[0] / v[1] for k, v in r['success_by_bin'].items()})
    if kind == 'fit':
        j = r['jumps']
        row.update(steps=r['steps'], final_loss=r['final_loss'], jump_big=(j.get('big') or {}).get('ratio'), jumps=j['by_true'], band=r['args'].get('band', 0.0))
    else:
        row.update(wall_innermost=r['wall_fit']['innermost'], rollbacks=r['rollbacks'], course=r.get('course', {}), t_blocked_free=r.get('t_blocked_free'),
                   tri_final=r.get('tri_final'), bumps=r['bumps']['bumps_per_path'], stalls=r['stalls'], train_min=r['train_seconds'] / 60)
    return row


def load():
    state = json.load(open(HERE / 'queue/state.json'))
    runs = []
    for job, rid in state['done'].items():
        label, key, seed, extra = job.split('|')
        row = row_of(rid, key, label, seed, extra, 'fit' if label.startswith('fit_') else 'field')
        if row:
            runs.append(row)
    direct = HERE / 'queue/direct.txt'
    for job in ([l.strip() for l in open(direct) if l.strip()] if direct.exists() else []):
        # runs launched outside the runner (queue/direct.py): the id is in the job's log; ';field=<id>' / ';init=<id>' name the frozen field / warm start
        f = HERE / 'queue/logs' / (''.join(c if c.isalnum() or c in '-_.' else '_' for c in job)[:180] + '.log')
        m = re.search(r'"id": "([^"]+)"', f.read_text()) if f.exists() else None
        label, key, seed, extra = job.split(';')[0].split('|')
        row = row_of(m.group(1), key, label, seed, extra, 'field') if m else None
        if row:
            row.update({kv.split('=')[0]: kv.split('=')[1] for kv in job.split(';')[1:]})
            runs.append(row)
    for scene, refs in REFERENCES.items():
        for label, ids in refs.items():
            for rid in ids:
                res = json.load(open(ROOT / f'results/registered_{rid}/results.json'))
                a = res['args']
                extra = ' '.join(f'--{k} {a[k]}' for k in ('lift', 'push', 'octaves') if a.get(k) and a[k] != 'none')
                row = row_of(rid, scene, label, res['seed'], extra, 'reference')
                runs.append(row)
    return runs, state


def fmt(v, d=1):
    return '–' if nan(v) else f'{v:.{d}f}'


def main():
    OUT.mkdir(exist_ok=True)
    runs, state = load()
    json.dump(dict(runs=runs, failed=state['failed']), open(OUT / 'runs.json', 'w'), indent=1, default=float)
    lines = ['# pair_lift tables (report.py; every number from registered runs)\n']
    for kind, title in (('reference', 'References (jeffrey/lift_scale, same pipeline)'), ('field', 'Fields trained without the truth'), ('fit', 'Supervised fits to the true travel time (representation test)')):
        rs = [r for r in runs if r['kind'] == kind]
        if not rs:
            continue
        lines.append(f'\n## {title}\n')
        if kind == 'fit':
            lines.append('| scene | arm | seed | id | steps | band share | solved % | wall % | stall % | T within 10 % | T/T* | detour index | seen through % | step across walls / true (true >= 3 cells) | arguments |')
            lines.append('|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
            for r in rs:
                lines.append(f"| {r['scene']} | {r['label']} | {r['seed']} | {r['id']} | {r['steps']} | {r['band']} | {fmt(r.get('solved'))} | {fmt(r.get('wall'))} | {fmt(r.get('stall'))} | {r['within10']:.1f} | "
                             f"{r['ratio']:.3f} | {fmt(r['rho'], 2)} | {fmt(r['seen_through'])} | {fmt(r['jump_big'], 2)} | `{r['extra']}` |")
        else:
            lines.append('| scene | arm | seed | id | solved % | wall % | stall % | T within 10 % | T/T* | LOS within 10 % | detour index | seen through % | S x slope innermost | rollbacks | minutes | arguments |')
            lines.append('|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
            for r in rs:
                lines.append(f"| {r['scene']} | {r['label']} | {r['seed']} | {r['id']} | {r['solved']:.1f} | {r['wall']:.1f} | {r['stall']:.1f} | {r['within10']:.1f} | {r['ratio']:.3f} | {fmt(r['los_within10'])} | "
                             f"{fmt(r['rho'], 2)} | {fmt(r['seen_through'])} | {fmt(r['wall_innermost'], 2)} | {r['rollbacks']} | {r['minutes']:.1f} | `{r['extra']}` |")
    if state['failed']:
        lines.append('\n## Jobs that did not finish\n')
        for j, rid in state['failed'].items():
            lines.append(f'* `{j}` -> {rid}')
    (OUT / 'tables.md').write_text('\n'.join(lines) + '\n')
    print(f'{len(runs)} runs -> {OUT}/tables.md')


if __name__ == '__main__':
    main()
