"""Collect the anisotropic-eikonal study's registered runs into report/summary.json and report/tables.md.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/aniso_eikonal/report.py [--epochs 4000] [--print]

Runs are read from the experiment registry (succeeded 'Aniso eikonal train' runs with results.json complete). One row per
run; pooled rows per (base, gap, variant) over training seeds. Variant = mode, time unit, warp and any other option
(variant_of). Also reads the verify runs and, for reference, the gap study's fields trained with ntrl-demo's full loss
(report/../cross_warp/report/zdraw10.json when present).
"""
import argparse
import json
import os
import sqlite3
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DB = Path(os.environ.get('NTFIELDS_REGISTRY', Path.home() / '.local/share/ntfields-experiments')) / 'registry.sqlite3'
CATS = ['same quadrant', 'adjacent quadrants', 'diagonal quadrants', 'quadrant - open', 'open - open']
BASES = {'cross22': 'c22', 'maze10': 'm10'}


def variant_of(r):
    """Short variant name of a run's results.json."""
    v = r['mode']
    if r['mode'] in ('zaniso', 'zsplit', 'zboth', 'zres'):
        u = r['unit_cells']
        v += '-U' + (u if u in ('box', 'quad') else f'{float(u):g}')
    if r['mode'] != 'id' and r.get('band_z_request'):
        v += f"-bz{float(r['band_z_request']):g}"
    elif r['mode'] != 'id' and abs(float(r['warp']['m_lo']) - 1.0) > 1e-9:
        drawn = float((r.get('pairs_drawn_on') or {}).get('m_lo', 1.0))
        # -sh: the gap study's shrunk-band warp with the pairs drawn on it; -lo<m>: another band density (pairs as drawn)
        v += '-sh' if abs(float(r['warp']['m_lo']) - drawn) < 1e-9 else f"-lo{float(r['warp']['m_lo']):g}"

    if abs(float(r.get('tau_weight', 0.5)) - 0.5) > 1e-12:
        v += f'-tw{float(r["tau_weight"]):g}'
    if r.get('head_scale', 1.0) == 1.0:
        r = dict(r, head_scale=0)
    if r.get('emb_scale', 1.0) == 1.0:
        r = dict(r, emb_scale=0)
    if r.get('messy'):
        v += f"-messy{float(r['messy']['amp']):g}"
    if r.get('open_pairs'):
        v += '-open'
    if r.get('extra_data'):
        v += '-mix'
    if r.get('group_scales'):
        v += '-gs' + r['group_scales'].replace(',', '+')
    if r.get('sum_groups'):
        v += f"-sum{r['sum_groups']}" + ('' if float(r.get('sum_scale', 1.0)) == 1.0 else f"x{float(r['sum_scale']):g}")
    if r.get('far'):
        v += '-far' + ('u' if r.get('far_kind') == 'unwarp' else '') + ('' if r['far'] == 'auto' else str(r['far']))
    if r.get('pair_weight', 'tau') != 'tau':
        v += '-pw' + r['pair_weight']
    for key, tag in (('head_scale', 'hs'), ('emb_scale', 'es'), ('lb_weight', 'lb'), ('td', 'td')):
        if r.get(key):
            v += f'-{tag}{r[key]:g}' if isinstance(r[key], (int, float)) else f'-{tag}'
    if r.get('swap_warp'):
        v += '-from' + str(r.get('swap_from_gap', 'x'))
    elif r.get('checkpoint_from') and r['mode'] in ('zsplit', 'zboth'):
        v += '-posthoc'
    if r.get('variant_tag'):
        v += '-' + r['variant_tag']
    if r.get('head_from') and r.get('stopped_at_epoch'):          # the first frozen-head runs: the loop's rollback rule stopped them at epoch 10
        v += '-aborted'
    return v


def registry_rows(prefix):
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    out = []
    for rid, name, state, data in con.execute("select id, name, state, data from runs where name like ? order by id", (prefix + '%',)):
        if state != 'succeeded' or name.endswith(' smoke'):
            continue
        d = json.loads(data)
        out.append((rid, name, d['spec']['config'], Path(d['output'])))
    return out


def collect(epochs=4000):
    runs = []
    for rid, name, cfg, out in registry_rows('Aniso eikonal train') + registry_rows('Aniso eikonal evalq'):
        f = out / 'results.json'
        if not f.exists():
            continue
        r = json.load(open(f))
        if not r.get('complete') or r.get('smoke') or r.get('epochs') != epochs:
            continue
        row = dict(id=rid, label=cfg.get('label'), base=BASES[r['gap_base']], k=r['gap_scale'], n=r['maze']['n'], seed=r['seed'], mode=r['mode'], variant=variant_of(r),
                   unit_cells=r['unit_cells'], c=r['c'], m_lo=r['warp']['m_lo'], final_loss=r.get('final_loss'), rollbacks=r.get('rollbacks'),
                   train_minutes=(r.get('train_seconds') or 0) / 60, plans={}, time_accuracy=r.get('time_accuracy'), direction=r.get('direction'),
                   speed_on_maze=r.get('speed_on_maze'), lower_bound=r.get('lower_bound'), stopped=r.get('stopped_at_epoch', 0), t_scale=r.get('t_scale'), checkpoint_from=r.get('checkpoint_from'))
        for pl, e in r['plans'].items():
            row['plans'][pl] = dict(success=e['success_total'][0], runs=e['success_total'][1], by_category={c: e['by_category'][c] for c in CATS if c in e['by_category']},
                                    arrived_wall=e['arrived_wall'], not_arrived=e['not_arrived'], time_ratio_median=e['time_ratio_median'], time_ratio_p90=e.get('time_ratio_p90'),
                                    time_ratio_by_category=e.get('time_ratio_by_category'), arrived_wall_by_category=e.get('arrived_wall_by_category'),
                                    not_arrived_by_category=e.get('not_arrived_by_category'))
        runs.append(row)
    return runs


def pct(a, b):
    return 100.0 * a / b if b else float('nan')


def pool(runs):
    """{(base, k, variant): pooled row over training seeds}."""
    out = {}
    for r in runs:
        out.setdefault((r['base'], r['k'], r['variant']), []).append(r)
    pooled = {}
    for key, rs in out.items():
        e = dict(base=key[0], k=key[1], variant=key[2], seeds=sorted({r['seed'] for r in rs}), ids=[r['id'] for r in rs], plans={})
        firsts = {}                                   # one row per training seed for the field's own diagnostics (evaluation-only runs repeat them)
        for r in rs:
            firsts.setdefault(r['seed'], r)
        own = list(firsts.values())
        for pl in ('x', 'z', 'zh'):
            have = [r['plans'][pl] for r in rs if pl in r['plans']]
            if not have:
                continue
            e['plans'][pl] = dict(pct=pct(sum(h['success'] for h in have), sum(h['runs'] for h in have)), per_seed=[pct(h['success'], h['runs']) for h in have],
                                  by_category={c: pct(sum(h['by_category'][c][0] for h in have), sum(h['by_category'][c][1] for h in have)) for c in CATS if all(c in h['by_category'] for h in have)},
                                  arrived_wall=sum(h['arrived_wall'] for h in have), not_arrived=sum(h['not_arrived'] for h in have), runs=sum(h['runs'] for h in have),
                                  time_ratio_median=float(np.mean([h['time_ratio_median'] for h in have if h['time_ratio_median'] is not None])) if any(h['time_ratio_median'] is not None for h in have) else None)
        ta = [r['time_accuracy'] for r in own if r.get('time_accuracy')]
        if ta:
            e['time_accuracy'] = dict(ratio_median=float(np.mean([t['ratio_median'] for t in ta])), abs_log_ratio_median=float(np.mean([t['abs_log_ratio_median'] for t in ta])),
                                      within_10pct=float(np.mean([t['within_10pct'] for t in ta])), within_25pct=float(np.mean([t['within_25pct'] for t in ta])),
                                      by_category={c: float(np.mean([t['by_category'][c]['ratio_median'] for t in ta])) for c in CATS if all(c in t['by_category'] for t in ta)})
        dr = [r['direction'] for r in own if r.get('direction')]
        if dr:
            e['direction_median_deg'] = float(np.mean([d['median_deg'] for d in dr]))
            e['direction_by_category'] = {c: float(np.mean([d['by_category'][c] for d in dr])) for c in CATS if all(c in d['by_category'] for d in dr)}
        sp = [r['speed_on_maze'] for r in own if r.get('speed_on_maze')]
        if sp:
            e['speed_on_maze'] = {reg: float(np.mean([s['by_region'][reg]['median'] for s in sp if reg in s['by_region']])) for reg in ('quadrants', 'band edge cells', 'arms', 'centre')
                                  if any(reg in s['by_region'] for s in sp)}
        lbs = [r['lower_bound'] for r in own if r.get('lower_bound')]
        if lbs:
            e['lower_bound'] = dict(share_below_0_9=float(np.mean([l['share_below_0_9'] for l in lbs])), band_share_below_0_9=float(np.mean([l['band_pairs']['share_below_0_9'] or 0 for l in lbs])),
                                    band_median=float(np.mean([l['band_pairs']['median'] or 0 for l in lbs])))
        e['final_loss'] = float(np.mean([r['final_loss'] for r in rs if r.get('final_loss') is not None])) if any(r.get('final_loss') is not None for r in rs) else None
        pooled[key] = e
    return pooled


def verify_runs():
    out = {}
    for rid, name, cfg, folder in registry_rows('Aniso eikonal verify'):
        f = folder / 'results.json'
        if not f.exists():
            continue
        r = json.load(open(f))
        if not r.get('complete') or r.get('smoke'):
            continue
        key = f'{BASES[r["gap_base"]]}_k{r["gap_scale"]}' + ('_sh' if abs(r['warp']['m_lo'] - 1) > 1e-9 else '')
        out[key] = dict(id=rid, base=BASES[r['gap_base']], k=r['gap_scale'], warp=r['warp'], metric_check=r['metric_check'], equations=r['equations'], price=r['price'])
    return out


def fmt(v, d=1):
    return '–' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{v:.{d}f}'


def table(pooled, base, k, planner_pref=('z', 'x')):
    lines = [f'### {base} gap ×{k}', '', '| variant | seeds | MPPI in z | MPPI in x | same q. | adjacent | diagonal | q.-open | open-open | never arrive / wall (z) | T/T* | within 10 % | path time/T* | speed centre | loss | band pairs below bound |', '|' + '---|' * 16]
    keys = sorted([key for key in pooled if key[0] == base and key[1] == k], key=lambda t: t[2])
    for key in keys:
        e = pooled[key]
        pl = next((p for p in planner_pref if p in e['plans']), None)
        z, x = e['plans'].get('z'), e['plans'].get('x')
        main = e['plans'][pl]
        ta = e.get('time_accuracy') or {}
        lines.append(f'| {e["variant"]} | {len(e["seeds"])} | {fmt(z["pct"]) if z else "–"} | {fmt(x["pct"]) if x else "–"} | ' + ' | '.join(fmt(main['by_category'].get(c)) for c in CATS)
                     + f' | {main["not_arrived"]} / {main["arrived_wall"]} of {main["runs"]} | {fmt(ta.get("ratio_median"), 2)} | {fmt(100 * ta.get("within_10pct", float("nan")), 0)} | '
                     f'{fmt(main["time_ratio_median"], 3)} | {fmt((e.get("speed_on_maze") or {}).get("centre"), 2)} | {("%.1e" % e["final_loss"]) if e["final_loss"] is not None else "–"} | {fmt(100 * (e.get("lower_bound") or {}).get("band_share_below_0_9", float("nan")), 0)} |')
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=4000)
    p.add_argument('--print', action='store_true')
    a = p.parse_args()
    runs = collect(a.epochs)
    pooled = pool(runs)
    ver = verify_runs()
    (HERE / 'report').mkdir(exist_ok=True)
    json.dump(dict(epochs=a.epochs, runs=runs, pooled=[v for _, v in sorted(pooled.items(), key=lambda t: (t[0][0], t[0][1], t[0][2]))], verify=ver),
              open(HERE / 'report/summary.json', 'w'), indent=1, default=float)
    lines = ['# Anisotropic eikonal: tables', '', f'{len(runs)} runs at {a.epochs} epochs. By-category columns: the z planner when it was run, else x. '
             'T/T*: median of the field\'s travel time over the true one on the 300 queries; path time/T*: median over solved runs; '
             'speed centre: the speed the field implies on the maze over the true speed, band centre.', '']
    for base in ('c22', 'm10'):
        for k in sorted({key[1] for key in pooled if key[0] == base}):
            lines += table(pooled, base, k) + ['']
    (HERE / 'report/tables.md').write_text('\n'.join(lines) + '\n')
    if a.print:
        print('\n'.join(lines))
    print(f'{len(runs)} runs, {len(pooled)} pooled rows, verify {sorted(ver)}')


if __name__ == '__main__':
    main()
