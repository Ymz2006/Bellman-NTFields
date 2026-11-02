"""Assemble README.md from README_src.md + README_setup.md and the report outputs (report.py, wallfeel.py first).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/assemble.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report as R  # noqa: E402
from common import bin_labels  # noqa: E402

VARIANTS = ['segment DP', 'segment DP, checked pieces', 'segment DP, zoomed base', 'segment DP + wall term', 'segment DP + wall term (weight 0.3)']


def pct(a, b):
    return f'{100 * a / b:.0f}' if b else '-'


def table1(prep, ntrl, labs):
    rows = ['| maze | field | ' + ' | '.join(labs) + ' | all |', '|' + '---|' * (len(labs) + 3)]
    for n in sorted(prep):
        o, st = prep[n]['oracle'], prep[n]['straight']['by_bin']
        cells = [pct(sum(o[s]['by_bin'][l][0] for s in o), sum(o[s]['by_bin'][l][1] for s in o)) if l in o['0']['by_bin'] else '' for l in labs]
        rows.append(f'| {n}×{n} | oracle | ' + ' | '.join(cells) + ' | 100 |')
        rows.append(f'| | straight line | ' + ' | '.join(pct(*st[l]) if l in st else '' for l in labs) + f' | {pct(prep[n]["straight"]["success"], sum(v[1] for v in st.values()))} |')
        by = ntrl.get(n, {})
        if by:
            cells = []
            for l in labs:
                per = [by[s]['success_by_bin'].get(l) for s in by]
                cells.append(pct(sum(p[0] for p in per), sum(p[1] for p in per)) if all(per) else '')
            tot = [by[s]['success_total'] for s in by]
            rows.append(f'| | **ntrl-demo** | ' + ' | '.join(f'**{c}**' if c else '' for c in cells) + f' | **{pct(sum(t[0] for t in tot), sum(t[1] for t in tot))}** |')
    return '\n'.join(rows)


def table3(prep, ntrl, seg, zoom, labs):
    rows = ['| maze | field (training seed) | ' + ' | '.join(labs) + ' | all |', '|' + '---|' * (len(labs) + 3)]
    for n in sorted(prep):
        entries = []
        for s, d in sorted(ntrl.get(n, {}).items(), key=lambda t: (t[0] != 87358, t[0])):
            if any(s in seg.get(v, {}).get(n, {}) for v in VARIANTS):
                entries.append((f'ntrl-demo ({s})', d))
        for s, d in sorted(zoom.get(n, {}).items()):
            entries.append((f'ntrl-demo, zoomed input ({s})', d))
        for v in VARIANTS:
            for s, d in sorted(seg.get(v, {}).get(n, {}).items(), key=lambda t: (t[0] != 87358, t[0])):
                entries.append((f'{v} ({s})', d))
        for k, (name, d) in enumerate(entries):
            cells = [pct(*d['success_by_bin'][l]) if l in d['success_by_bin'] else '' for l in labs]
            rows.append(f'| {f"{n}×{n}" if k == 0 else ""} | {name} | ' + ' | '.join(cells) + f' | {pct(*d["success_total"])} ({d["success_total"][0]}/{d["success_total"][1]}) |')
    return '\n'.join(rows)


def runs_table(prep, ntrl, seg, zoom):
    rows = ['| maze | stage | training seed | run id (results/registered_&lt;id&gt;) |', '|---|---|---|---|']
    for n in sorted(prep):
        rows.append(f'| {n}×{n} | prepare | - | {prep[n]["_id"]} |')
        for s, d in sorted(ntrl.get(n, {}).items()):
            rows.append(f'| | ntrl-demo | {s} | {d["_id"]} |')
        for s, d in sorted(zoom.get(n, {}).items()):
            rows.append(f'| | zoom x{d["zoom"]:.3f} | {s} | {d["_id"]} |')
        for v in VARIANTS:
            for s, d in sorted(seg.get(v, {}).get(n, {}).items()):
                rows.append(f'| | {v} | {s} | {d["_id"]} |')
    return '\n'.join(rows)


def main():
    prep, ntrl, seg = R.collect()
    zoom = {}
    for d in R.runs('zoom'):
        zoom.setdefault(d['maze']['n'], {})[d['seed']] = d
    labs = [l for l in bin_labels() if any(prep[n]['queries']['per_bin'].get(l) for n in prep)]
    src = (HERE / 'README_src.md').read_text()
    src = src.replace('<!--SETUP-->', (HERE / 'README_setup.md').read_text().strip())
    src = src.replace('<!--TABLE1-->', table1(prep, ntrl, labs))
    wall = (R.OUT / 'wallfeel.md').read_text().split('\n')
    src = src.replace('<!--TABLE2-->', '\n'.join(l for l in wall if l.startswith('|')))
    src = src.replace('<!--TABLE3-->', 'MPPI success (%) by true travel distance, per training seed (3 planning seeds each):\n\n' + table3(prep, ntrl, seg, zoom, labs))
    src = src.replace('<!--RUNS-->', runs_table(prep, ntrl, seg, zoom))
    seeds = (HERE / 'report' / 'seeds.md')
    src = src.replace(' <!--SEEDS-->', (' ' + seeds.read_text().strip()) if seeds.exists() else '')
    (HERE / 'README.md').write_text(src)
    print('README.md', len(src), 'chars')


if __name__ == '__main__':
    main()
