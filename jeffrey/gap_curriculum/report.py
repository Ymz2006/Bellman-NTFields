"""Tables for jeffrey/gap_curriculum/README.md from registered gap-curriculum runs (read-only).

    report.py --runs registered_<A>=B10 registered_<B>=C10 ... [--reference B10] [--diagnostics] > tables.md

Per-query success flags come from the evaluation's *.score.json files; gained / lost are paired against the
--reference run of the same scene (same queries, same planning seeds).
"""
import argparse, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[2]
SHORT = dict(free_to_free='f→f', free_to_compartment='f→c', compartment_to_free='c→f', compartment_to_compartment='c→c', same_compartment='same c', anywhere='any',
             cutting='cut', hammering='ham', screwdriver='scr', stirring='stir', wiping='wipe', ur5e_mixed_c30_p30_x40_certified='mixed')


def scene_of(status):
    settings = status.get('settings') or dict(scene=status.get('scene'))
    return 'cabinet' if settings.get('scene') else 'nowall'


def load(run):
    root = HERE/'results'/Path(run).name; status = json.loads((root/'status.json').read_text())
    spec = json.loads((root/'spec.json').read_text()) if (root/'spec.json').exists() else None
    config = json.loads((root/'training/config.json').read_text()); ev = json.loads((root/'evaluation/evaluation.json').read_text())
    flags = {}
    for path in sorted((root/'evaluation').glob('*.score.json')):
        doc = json.loads(path.read_text()); label = path.name[:-len('.score.json')]; row = doc[label]
        flags[label] = np.array([row['per_query'][str(i)]['success'] for i in range(row['n'])], bool)
    diag = json.loads((root/'diagnostics.json').read_text()) if (root/'diagnostics.json').exists() else None
    return dict(run=root.name, root=root, status=status, config=config, ev=ev, flags=flags, diag=diag)


def method_rows(r, method):
    return [x for x in r['ev']['scores'] if x['method'] == method]


def suites(r):
    return list(dict.fromkeys(x['task'] for x in r['ev']['scores']))


def combined(r, kind):
    keys = sorted(k for k in r['flags'] if ('_rk4_' in k) == (kind == 'rk4'))
    return keys, np.concatenate([r['flags'][k] for k in keys])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--runs', nargs='+', required=True); p.add_argument('--reference'); p.add_argument('--diagnostics', action='store_true')
    a = p.parse_args()
    named = [x.split('=', 1) if '=' in x else (x, x) for x in a.runs]; runs = [dict(load(run), name=name) for run, name in named]
    ref = next((r for r in runs if r['name'] == a.reference), None)
    print('|Run|Scene|Rounds × epochs|Goals|Radii|Seed|RK4 /600|Release MPPI /1800|Per seed|RK4 vs ref +/-|MPPI vs ref +/-|')
    print('|---|---|---|---|---|---:|---:|---:|---|---|---|')
    for r in runs:
        c = r['config']; t = r['status']['totals']; radii = c.get('radii') or []
        rad = '—' if not radii else ('mix of ' if c.get('mix') else '')+(f"{radii[0]:g}…{radii[-1]:g}" if len(radii) > 2 else ','.join(f'{x:g}' for x in radii))
        cells = []
        for kind in ('rk4', 'mppi'):
            if ref is None or ref is r: cells.append('—'); continue
            k1, f1 = combined(r, kind); k0, f0 = combined(ref, kind); assert k1 == k0
            cells.append(f'+{int((f1 & ~f0).sum())} / −{int((~f1 & f0).sum())}')
        print(f"|{r['name']} (`{r['run'][11:]}`)|{scene_of(r['status'])}|{c['rounds']} × {c['epochs_per_round']}|{c.get('goals', 'natural')}|{rad}|"
              f"{c.get('seed', 87358)}|{t['native_rk4']['success']}|{t['matched_mppi']['success']}|{t['matched_mppi']['per_seed']}|{cells[0]}|{cells[1]}|")
    for scene in ('nowall', 'cabinet'):
        group = [r for r in runs if scene_of(r['status']) == scene]
        if not group: continue
        names = suites(group[0])
        print(); print(f'{scene}: per suite, RK4 /100 and release MPPI /300'); print()
        print('|Run|'+'|'.join(f"{SHORT.get(s, s)} RK4|{SHORT.get(s, s)} MPPI" for s in names)+'|'); print('|---|'+'---:|'*(2*len(names)))
        for r in group:
            cells = []
            for s in names:
                cells.append(str(sum(x['success'] for x in method_rows(r, 'native_rk4') if x['task'] == s)))
                cells.append(str(sum(x['success'] for x in method_rows(r, 'matched_mppi') if x['task'] == s)))
            print(f"|{r['name']}|"+'|'.join(cells)+'|')
    if a.diagnostics:
        for r in runs:
            d = r['diag']
            if d is None: continue
            bins = d['bins']; heads = [f'{bins[i]:g}–{bins[i+1]:g}' for i in range(len(bins)-1)]
            for key, title in (('ratio', 'T / |q-g|'), ('abs_fixed', '|T - B T| / B T (self-consistency)'), ('abs_fit', '|T_k - B T_{k-1}| / B T_{k-1} (fit to own target)'), ('eikonal', 'eikonal residual')):
                print(); print(f"{r['name']} ({r['run'][11:]}): {title} by gap"); print()
                print('|Round|'+'|'.join(heads)+'|'); print('|---|'+'---:|'*len(heads))
                for row in d['rounds']:
                    print(f"|{row['checkpoint'][6:8]}|"+'|'.join('—' if v is None else f'{v:.3f}' for v in row[key])+'|')


if __name__ == '__main__': main()
