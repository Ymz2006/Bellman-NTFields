"""Tables from the registered embed-warp runs: report/runs.json (one row per run) and report/tables.md.

    python jeffrey/embed_warp/report.py            # reads the registry (name 'Embed warp ...', state succeeded)

Row: scene (plus / disks / gibson / plain / gap), n, seed, label, octaves, gate, success (total and %), by bin, by where (plus),
failures by arrival, speed fit, near-wall slope / cos / attract, bumps, isotropy (band / open stretch and anisotropy), gate mu.
"""
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HOME = Path(os.environ.get('NTFIELDS_REGISTRY', Path.home() / '.local/share/ntfields-experiments'))
REPORT = HERE / 'report'
BINS = ['0.5-2', '2-4', '4-8', '8-16', '16-32', '32-64', '64-128', '128-256']
ORDER = ['ntrl', 'oct', 'oct8', 'geo8', 'geo', 'geo0', 'geot', 'geow', 'geo8', 'geo24', 'geob', 'kde', 'diff', 'curv', 'curvr', 'curvg', 'learn', 'learnb', 'learnx']
NAMES = dict(ntrl='ntrl-demo (control)', oct='octaves always on (uniform multi-scale control)', geo='gate: clearance band 0.5-1.5 cells', geo0='geo, mu_min 0',
             geot='geo, tight band 0.25-0.75', geow='geo, wide band 1-3', geo8='geo, octaves 2,4,8', geo24='geo, 24 rows per octave', geob='geo, base rows x1.41',
             kde='gate: band-sample density', diff='gate: masked diffusion from walls', curv='gate: curvature of the ntrl field (Hessian)', curvr='gate: eikonal misfit of the ntrl field',
             curvg='gate: curvature of the geo field', learn='gate: learned MLP(u0), by the T loss', learnb='learned, budget 0.3', learnx='learned MLP(x)')


def registry_rows():
    con = sqlite3.connect(str(HOME / 'registry.sqlite3'))
    cols = [c[1] for c in con.execute('pragma table_info(runs)')]
    out = []
    for row in con.execute('select * from runs'):
        d = dict(zip(cols, row))
        if 'row' in d and isinstance(d['row'], str):
            try:
                d = json.loads(d['row'])
            except Exception:
                pass
        out.append(d)
    return out


def load_runs():
    rows = []
    for d in sorted((HOME / 'runs').iterdir()):
        spec_p, rec_p = d / 'spec.json', d / 'record.json'
        if not spec_p.exists():
            continue
        spec = json.load(open(spec_p))
        if not spec['name'].startswith('Embed warp') or spec['config'].get('smoke'):
            continue
        rec = json.load(open(rec_p)) if rec_p.exists() else {}
        if rec.get('state') != 'succeeded':
            continue
        out = Path(rec.get('output', ROOT / 'results' / f'registered_{d.name}'))
        rp = out / 'results.json'
        if not rp.exists():
            continue
        res = json.load(open(rp))
        if res.get('stage') != 'embed-warp' or not res.get('complete'):
            continue
        cfg = spec['config']
        sc = res['scene']
        scene = sc.get('variant', 'plain')
        if scene == 'gibson' and int(round(sc.get('E_cells', 64))) != 64:
            scene = f'gibson{int(round(sc["E_cells"]))}'
        if scene == 'gap':
            scene = f'gap{sc.get("gap_scale", "")}'
        succ = res['success_total']
        emb = res['embed']
        row = dict(id=d.name, label=cfg['label'], extra=cfg.get('extra', ''), scene=scene, n=sc['n'], seed=cfg['seed'], prepared=cfg.get('prepared'),
                   octaves=emb['octaves'], rows=emb['rows'], gate=emb['gate']['kind'], success=succ[0], runs=succ[1], pct=100.0 * succ[0] / succ[1],
                   by_bin={k: v for k, v in res['success_by_bin'].items()}, by_where=res.get('by_where'), by_category=res.get('by_category'),
                   failures=res['failures'], speed_corr=res['speed_fit']['log_speed_corr'], speed_floor=res['speed_fit'].get('speed_pred_median_at_floor'),
                   near_wall=res['near_wall']['all'], bumps=res['bumps'], t_ratio=res['field']['t_over_tstar_median'], spearman=res['field']['spearman'],
                   descent=res['field']['descent_cos_median'], final_loss=res['final_loss'], rollbacks=res['rollbacks'], train_min=res['train_seconds'] / 60,
                   gpu=res['gpu'], gate_info={k: v for k, v in res.get('gate_info', {}).items() if k != 'by_clearance'}, gate_by_clearance=res.get('gate_info', {}).get('by_clearance'),
                   gate_mu=res.get('gate_mu'), iso={k: dict(stretch=v['stretch']['median'], aniso=v['aniso']['median'], aniso_q90=v['aniso']['q90'], share=v['share']['median'])
                                                    for k, v in res.get('isotropy', {}).items()})
        rows.append(row)
    return rows


def load_evalq(rows):
    """Held-out query-set evaluations (stage evalq): per field run its success on that set; pooled per (scene, label, set)."""
    by_id = {r['id']: r for r in rows}
    out = []
    for d in sorted((HOME / 'runs').iterdir()):
        spec_p, rec_p = d / 'spec.json', d / 'record.json'
        if not spec_p.exists():
            continue
        spec = json.load(open(spec_p))
        if not spec['name'].startswith('Embed warp evalq') or spec['config'].get('smoke'):
            continue
        rec = json.load(open(rec_p)) if rec_p.exists() else {}
        if rec.get('state') != 'succeeded':
            continue
        rp = Path(rec['output']) / 'results.json'
        if not rp.exists():
            continue
        res = json.load(open(rp))
        qset = spec['config'].get('queries', '')
        for name, e in res['runs'].items():
            rid = name.replace('registered_', '')
            fr = by_id.get(rid)
            if fr is None:
                continue
            s = e['success_total']
            out.append(dict(evalq=d.name, field=rid, label=fr['label'], scene=fr['scene'], seed=fr['seed'], qset=qset, success=s[0], runs=s[1], pct=100.0 * s[0] / s[1],
                            by_bin=e['success_by_bin'], failures=e['failures']))
    pooled_q = defaultdict(lambda: dict(success=0, runs=0, seeds=[], per_seed={}, by_bin=defaultdict(lambda: [0, 0]), arrived_wall=0, not_arrived=0))
    seen = set()
    for e in out:
        key = (e['scene'], e['label'], e['qset'])
        if (key, e['seed']) in seen:
            continue
        seen.add((key, e['seed']))
        g = pooled_q[key]
        g['success'] += e['success']
        g['runs'] += e['runs']
        g['seeds'].append(e['seed'])
        g['per_seed'][str(e['seed'])] = round(e['pct'], 1)
        g['arrived_wall'] += e['failures']['arrived_wall']
        g['not_arrived'] += e['failures']['not_arrived']
        for b, v in e['by_bin'].items():
            g['by_bin'][b][0] += v[0]
            g['by_bin'][b][1] += v[1]
    pq = []
    for (scene, label, qset), g in pooled_q.items():
        pq.append(dict(scene=scene, label=label, qset=qset, name=NAMES.get(label, label), seeds=sorted(g['seeds']), per_seed=g['per_seed'], success=g['success'], runs=g['runs'],
                       pct=100.0 * g['success'] / g['runs'], by_bin=dict(g['by_bin']), arrived_wall=g['arrived_wall'], not_arrived=g['not_arrived']))
    key = lambda p: (p['scene'], p['qset'], ORDER.index(p['label']) if p['label'] in ORDER else 99, p['label'])
    return out, sorted(pq, key=key)


def pooled(rows):
    """Pool seeds per (scene, label): success %, per-seed %, by bin."""
    g = defaultdict(list)
    for r in rows:
        g[(r['scene'], r['label'])].append(r)
    out = []
    for (scene, label), rs in g.items():
        s = sum(r['success'] for r in rs)
        n = sum(r['runs'] for r in rs)
        bb = {}
        for b in BINS:
            vals = [r['by_bin'].get(b) for r in rs if r['by_bin'].get(b)]
            if vals:
                bb[b] = [sum(v[0] for v in vals), sum(v[1] for v in vals)]
        where = None
        if rs[0].get('by_where'):
            where = {k: [sum(r['by_where'][k]['success'] for r in rs), sum(r['by_where'][k]['runs'] for r in rs)] for k in rs[0]['by_where']}
        out.append(dict(scene=scene, label=label, name=NAMES.get(label, label), seeds=sorted(r['seed'] for r in rs), success=s, runs=n, pct=100.0 * s / n,
                        per_seed={str(r['seed']): round(r['pct'], 1) for r in sorted(rs, key=lambda r: r['seed'])}, by_bin=bb, by_where=where,
                        arrived_wall=sum(r['failures']['arrived_wall'] for r in rs), not_arrived=sum(r['failures']['not_arrived'] for r in rs),
                        speed_corr=float(np.mean([r['speed_corr'] for r in rs])), slope=float(np.mean([r['near_wall']['slope'] for r in rs])),
                        cos=float(np.mean([r['near_wall']['cos'] for r in rs])), attract=float(np.mean([r['near_wall']['attract'] for r in rs])),
                        bumps=float(np.mean([r['bumps']['bumps_per_path'] or 0 for r in rs])), t_ratio=float(np.mean([r['t_ratio'] for r in rs])),
                        iso_band=rs[0]['iso'].get('band'), iso_open=rs[0]['iso'].get('open'), ids=[r['id'] for r in rs]))
    key = lambda p: (p['scene'], ORDER.index(p['label']) if p['label'] in ORDER else 99, p['label'])
    return sorted(out, key=key)


def md_tables(pool):
    lines = []
    scenes = sorted(set(p['scene'] for p in pool))
    for scene in scenes:
        ps = [p for p in pool if p['scene'] == scene]
        bins = [b for b in BINS if any(b in p['by_bin'] for p in ps)]
        lines.append(f'\n### {scene}\n')
        lines.append('| method | seeds | success | % | per seed | ' + ' | '.join(bins) + ' | arrived+wall | not arrived | speed corr | wall slope | wall cos | attract | bumps/path | band stretch / aniso | open stretch / aniso |')
        lines.append('|---|---|---:|---:|---|' + '---:|' * len(bins) + '---:|---:|---:|---:|---:|---:|---:|---|---|')
        for p in ps:
            bb = ' | '.join(f'{p["by_bin"][b][0]}/{p["by_bin"][b][1]}' if b in p['by_bin'] else '' for b in bins)
            ib = p['iso_band'] or {}
            io = p['iso_open'] or {}
            lines.append(f'| {p["name"]} (`{p["label"]}`) | {",".join(map(str, p["seeds"]))} | {p["success"]}/{p["runs"]} | {p["pct"]:.1f} | {" / ".join(f"{v:.1f}" for v in p["per_seed"].values())} | {bb} | '
                         f'{p["arrived_wall"]} | {p["not_arrived"]} | {p["speed_corr"]:.2f} | {p["slope"]:.2f} | {p["cos"]:.2f} | {p["attract"]:.2f} | {p["bumps"]:.2f} | '
                         f'{ib.get("stretch", float("nan")):.2f} / {ib.get("aniso", float("nan")):.2f} | {io.get("stretch", float("nan")):.2f} / {io.get("aniso", float("nan")):.2f} |')
        if any(p['by_where'] for p in ps):
            lines.append(f'\nPlus maze, success by where the query ends lie ({scene}):\n')
            lines.append('| method | both in + | one in + | neither |')
            lines.append('|---|---:|---:|---:|')
            for p in ps:
                if p['by_where']:
                    w = p['by_where']
                    lines.append(f'| `{p["label"]}` | ' + ' | '.join(f'{w[k][0]}/{w[k][1]} ({100 * w[k][0] / max(w[k][1], 1):.0f} %)' for k in ('both', 'one', 'neither')) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    REPORT.mkdir(exist_ok=True)
    rows = load_runs()
    pool = pooled(rows)
    eq, pq = load_evalq(rows)
    json.dump(dict(rows=rows, pooled=pool, evalq=eq, evalq_pooled=pq), open(REPORT / 'runs.json', 'w'), indent=1, default=float)
    md = md_tables(pool)
    if pq:
        md += '\n### Held-out query sets (evalq)\n\n| scene | set | method | seeds | success | % | per seed | through wall | never arrived |\n|---|---|---|---|---:|---:|---|---:|---:|\n'
        for p in pq:
            md += f'| {p["scene"]} | {p["qset"].split("_")[-1]} | {p["name"]} (`{p["label"]}`) | {",".join(map(str, p["seeds"]))} | {p["success"]}/{p["runs"]} | {p["pct"]:.1f} | {" / ".join(f"{v:.1f}" for v in p["per_seed"].values())} | {p["arrived_wall"]} | {p["not_arrived"]} |\n'
    (REPORT / 'tables.md').write_text(md)
    print(md)
    print(f'{len(rows)} runs')


if __name__ == '__main__':
    main()
