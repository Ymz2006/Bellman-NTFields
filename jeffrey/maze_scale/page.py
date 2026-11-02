"""Build the findings page (report/maze_scale.html) from the registered runs and the report outputs.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/page.py

Reads report.collect() (+ zoom runs), report/wallfeel.json and report/posthoc.json (run wallfeel.py and posthoc.py
first), embeds the numbers as JSON into page_template.html and writes report/maze_scale.html; the figures
(report/*.png) are published next to it as supporting files.
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report as R  # noqa: E402
import extras as X  # noqa: E402
from common import bin_labels  # noqa: E402

SERIES = ['ntrl-demo', 'ntrl-demo, zoomed input', 'segment DP', 'segment DP + wall term (weight 0.3)']


def pooled(by_seed, labs):
    seeds = sorted(by_seed)
    out = dict(seeds=seeds, counts=[], pct=[], lo=[], hi=[], ids=[by_seed[s]['_id'] for s in seeds])
    for lab in labs:
        per = [by_seed[s]['success_by_bin'].get(lab) for s in seeds]
        per = [p for p in per if p]
        if not per:
            out['counts'].append(None)
            out['pct'].append(None)
            out['lo'].append(None)
            out['hi'].append(None)
            continue
        out['counts'].append([sum(p[0] for p in per), sum(p[1] for p in per)])
        rates = [100 * p[0] / p[1] for p in per]
        out['pct'].append(100 * sum(p[0] for p in per) / sum(p[1] for p in per))
        out['lo'].append(min(rates))
        out['hi'].append(max(rates))
    tot = [by_seed[s]['success_total'] for s in seeds]
    out['total'] = [sum(t[0] for t in tot), sum(t[1] for t in tot)]
    out['per_seed_total'] = {str(s): by_seed[s]['success_total'] for s in seeds}
    f = [by_seed[s]['field'] for s in seeds]
    out['field'] = dict(t_over_tstar=[float(np.mean([x['by_bin'].get(l, {}).get('t_over_tstar_median', np.nan) for x in f])) if any(l in x['by_bin'] for x in f) else None for l in labs],
                        cos=[float(np.mean([x['by_bin'].get(l, {}).get('descent_cos_median', np.nan) for x in f])) if any(l in x['by_bin'] for x in f) else None for l in labs])
    sp = [by_seed[s].get('speed_fit', {}).get('log_speed_corr') for s in seeds]
    sp = [x for x in sp if x is not None]
    out['speed_fit'] = float(np.mean(sp)) if sp else None
    return out


def main():
    prep, ntrl, seg = R.collect()
    zoom = {}
    for d in R.runs('zoom'):
        zoom.setdefault(d['maze']['n'], {})[d['seed']] = d
    groups = {'ntrl-demo': ntrl, 'ntrl-demo, zoomed input': zoom}
    groups.update(seg)
    labels = bin_labels()
    mazes = []
    for n in sorted(prep):
        p = prep[n]
        labs = [l for l in labels if p['queries']['per_bin'].get(l)]
        o = p['oracle']
        st = p['straight']['by_bin']
        m = dict(n=n, id=p['_id'], bins=labs, per_bin=[p['queries']['per_bin'][l] for l in labs],
                 oracle=[100 * sum(o[s]['by_bin'][l][0] for s in o) / sum(o[s]['by_bin'][l][1] for s in o) for l in labs],
                 straight=[100 * st[l][0] / st[l][1] for l in labs], fields={})
        for name, by_n in groups.items():
            if by_n.get(n):
                m['fields'][name] = pooled(by_n[n], labs)
        mazes.append(m)
    wall = json.loads((R.OUT / 'wallfeel.json').read_text()) if (R.OUT / 'wallfeel.json').exists() else {}
    post = json.loads((R.OUT / 'posthoc.json').read_text()) if (R.OUT / 'posthoc.json').exists() else {}
    contact = {}
    for key, v in post.items():
        name, n, s = key.split('|')
        if name == 'ntrl' and s == '87358':
            contact[n] = {lab: dict(cells=b['contact_cells_median'], within1=b['contact_within_1_cell'], collisions=b['collisions'], failures=b['failures'])
                          for lab, b in v['by_bin'].items()}
    labs_all = [l for l in labels if any(prep[n]['queries']['per_bin'].get(l) for n in prep)]
    dp_rows = []
    variants = ['segment DP', 'segment DP, checked pieces', 'segment DP, zoomed base', 'segment DP + wall term', 'segment DP + wall term (weight 0.3)']
    for n in sorted(prep):
        entries = []
        if 87358 in ntrl.get(n, {}):
            entries.append(('ntrl-demo', ntrl[n][87358]))
        if 87358 in zoom.get(n, {}):
            entries.append(('ntrl-demo, zoomed input', zoom[n][87358]))
        for v in variants:
            if 87358 in seg.get(v, {}).get(n, {}):
                entries.append((v, seg[v][n][87358]))
        for k, (name, d) in enumerate(entries):
            cells = [f"{100 * d['success_by_bin'][l][0] / d['success_by_bin'][l][1]:.0f}" if l in d['success_by_bin'] else '' for l in labs_all]
            t = d['success_total']
            dp_rows.append(dict(maze=f'{n}×{n}' if k == 0 else '', name=name, cells=cells, all=f'{100 * t[0] / t[1]:.0f} ({t[0]}/{t[1]})'))
    data = dict(mazes=mazes, series=SERIES, wallfeel=wall, contact=contact, dp=dict(bins=labs_all, rows=dp_rows), extras=X.collect())
    html = (HERE / 'page_template.html').read_text().replace('/*DATA*/null', json.dumps(data, default=float))
    seeds = R.OUT / 'seeds.md'
    html = html.replace('<!--SEEDS-->', seeds.read_text().strip() if seeds.exists() else '')
    for key in ('LEVELS_TEXT', 'CANDIDATES_TEXT', 'FOURIER_TEXT', 'SCALEDDP_TEXT', 'UTURN_TEXT', 'UTURN_FIG'):     # prose for the follow-up sections
        f = HERE / 'page_text' / f'{key.lower()}.html'
        html = html.replace(f'<!--{key}-->', f.read_text().strip() if f.exists() else '')
    (R.OUT / 'maze_scale.html').write_text(html)
    print('wrote', R.OUT / 'maze_scale.html', len(html) // 1024, 'KB')


if __name__ == '__main__':
    main()
