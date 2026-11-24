"""Rewrite the results tables of README.md (between the `<!-- table:start -->` and `<!-- table:end -->` markers) from
report/summary.json and report/depth.json, so the README's numbers are read from the registered runs.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/arm_placement/readme_table.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
C2C = 'compartment_to_compartment'
FIELDS = dict(ntrl05='--lift none --weight 0 --freq 0.5', lift1='--lift time --weight 0', lift025='--lift time --weight 0 --freq 0.25',
              lbs='--lift time --bell 1 --traversals 4096 --pair-max 0.85 --pair-data --freq 0.5')


def md(t):
    """The page's HTML fragments as Markdown."""
    t = re.sub(r'<span class="verdict[^>]*>(.*?)</span>', lambda m: m.group(1).capitalize() + '.', t)
    for a, b in (('<b>', '**'), ('</b>', '**'), ('<code>', '`'), ('</code>', '`'), ('<p>', ''), ('</p>', '\n\n'), ('<ul>', '\n'), ('</ul>', ''), ('<li>', '\n- '), ('</li>', ''), ('&lt;', '<'), ('&gt;', '>')):
        t = t.replace(a, b)
    return re.sub(r'<[^>]+>', '', t).strip()


def main():
    S = json.loads((HERE / 'report/summary.json').read_text())
    D = json.loads((HERE / 'report/depth.json').read_text()) if (HERE / 'report/depth.json').exists() else {}
    P = sorted(S['placements'], key=lambda p: (p['name'] != 'base', p['dx'], p['s'], p['h']))
    sid = lambda r: f'`{r[-8:]}`'  # noqa: E731

    def cell(pl, label, kind, steps):
        out = []
        for fr in sorted([x for x in pl['fields'] if x['label'] == label], key=lambda x: x['seed']):
            if kind == 'forward':
                out.append(f'{fr["success"][0]} ({fr["by_task"][C2C]}) {sid(fr["run"])}')
                continue
            q = next((q for q in fr.get('planners', []) if q['kind'] == kind and q['steps'] == steps and q['samples'] == 50 and q.get('suite') == 'placement'), None)
            if q:
                out.append(f'{q["success"][0]} ({q["by_task"][C2C]}) {sid(q["run"])}')
        return '; '.join(out) or '–'

    L = ['Solved runs of 900 (300 queries × 3 planning seeds; `base`: of 1,800), in brackets the zone → another zone runs solved of 150 (`base`: of 300), then the run id (last eight characters).',
         'Several entries in a cell = training seeds 87358; 87359. "two-ended" = `bidir.plan_bidir`, steps per end. Reach = share of the zones’ volume at least one arm / both arms reach. Depth = deepest grasp point',
         'behind the front face among the zone → zone test queries.', '',
         '| placement | dx, s, h (m) | reach % | depth max (cm) | straight line | Euclid (control) | lift, ntrl-demo planner | lift, two-ended 600 | lift, two-ended 2,400 | ntrl-demo, own planner | ntrl-demo, two-ended 600 |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for pl in P:
        if not pl['fields']:
            continue
        rc, pr, e = pl.get('reach'), pl.get('prepare'), pl.get('euclid')
        d = D.get(pl['name'])
        L.append(f'| `{pl["name"]}` | {pl["dx"]:.2f}, {pl["s"]:.2f}, {pl["h"]:.2f} | ' + (f'{100 * rc["union_mean"]:.0f} / {100 * rc["both_mean"]:.0f}' if rc else '–') + ' | '
                 + (f'{100 * d["zone_to_zone"]["max"]:.0f}' if d else '–') + ' | ' + (f'{pr["straight_line"]}/{pr["queries"]}' if pr else '–') + ' | ' + (f'{e["success"][0]} {sid(e["run"])}' if e else '–') + ' | '
                 + ' | '.join(cell(pl, lab, k, st) for lab, k, st in (('lift', 'forward', 600), ('lift', 'bidir', 600), ('lift', 'bidir', 2400), ('ntrl', 'forward', 600), ('ntrl', 'bidir', 600))) + ' |')
    L += ['', 'Other fields and planner settings (same notation):', '', '| placement | field | training seed | planner | steps | solved | run |', '|---|---|---|---|---|---|---|']
    for pl in P:
        for fr in sorted(pl['fields'], key=lambda x: (x['label'], x['seed'])):
            if fr['label'] not in ('lift', 'ntrl'):
                L.append(f'| `{pl["name"]}` | {fr["label"]} (`{FIELDS.get(fr["label"], "")}`) | {fr["seed"]} | ntrl-demo planner | 600 | {fr["success"][0]}/{fr["success"][1]} ({fr["by_task"][C2C]}) | {sid(fr["run"])} |')
            for q in sorted(fr.get('planners', []), key=lambda q: (q['kind'], q['steps'], q['samples'])):
                usual = fr['label'] in ('lift', 'ntrl') and q['kind'] == 'bidir' and q['steps'] in (600, 2400) and q['samples'] == 50
                if q.get('suite') != 'placement' or q['steps'] == 300 or (usual and not (fr['label'] == 'ntrl' and q['steps'] == 2400)):
                    continue
                L.append(f'| `{pl["name"]}` | {fr["label"]} | {fr["seed"]} | {q["kind"]}' + (f', {q["samples"]} samples' if q['samples'] != 50 else '') + f' | {q["steps"]} | {q["success"][0]}/{q["success"][1]} ({q["by_task"][C2C]}) | {sid(q["run"])} |')
    hard = [q for q in S['plans'] if q.get('suite') == 'hard']
    if hard:
        L += ['', 'Zone → another zone sets (200 queries × 3 planning seeds = 600 runs):', '', '| placement | field | training seed | planner | steps | solved of 600 | queries solved by at least one / by all three planning seeds | did not arrive | arrived in contact | run |', '|---|---|---|---|---|---|---|---|---|---|']
        for q in sorted(hard, key=lambda q: (q['placement'], q['field'], q['seed'], q['kind'], q['steps'])):
            L.append(f'| `{q["placement"]}` | {q["field"]} | {q["seed"]} | {q["kind"]} | {q["steps"]} | {q["success"][0]} | {q["any_seed"]} / {q["all_seeds"]} | {q["not_arrived"]} | {q["arrived_contact"]} | {sid(q["run"])} |')
    bins = [(n, d['by_depth']) for n, d in D.items() if d.get('by_depth')]
    if bins:
        labels = [b['label'] for b in bins[0][1]['bins']]
        L += ['', f'Lift field, two-ended planner, {bins[0][1]["steps"]:,} steps per end, by the depth of the query’s deepest hand (solved / runs; all training seeds evaluated at that budget):', '',
              '| placement | ' + ' | '.join(labels) + ' |', '|---|' + '---|' * len(labels)]
        for n, bd in sorted(bins, key=lambda x: (x[0] != 'base', D[x[0]]['dx'], x[0])):
            L.append(f'| `{n}` | ' + ' | '.join(f'{b["solved"]}/{b["runs"]}' + (f' = {100 * b["solved"] / b["runs"]:.0f} %' if b['runs'] else '') for b in bd['bins']) + ' |')
    f = HERE / 'README.md'
    s = f.read_text()
    a, b = s.index('<!-- table:start -->') + len('<!-- table:start -->'), s.index('<!-- table:end -->')
    s = s[:a] + '\n' + '\n'.join(L) + '\n' + s[b:]
    N = json.loads((HERE / 'report/notes.json').read_text())
    F = [f'{i + 1}. **{q}{"" if q.endswith("?") else "."}** {md(t)}' for i, (q, t) in enumerate(N['answers'])]
    for title, key in (('Free space and reach', 'ease_text'), ('Depth, and the control at equal depth', 'depth_text'), ('Which queries', 'kinds_text'), ('The two-ended planner', 'planner_text'),
                       ('Spacing and height', 'variants_text'), ('The ten configurations', 'demo_text'), ('Setup notes and controls', 'setup_text')):
        if N.get(key):
            F += ['', f'**{title}.** ' + md(N[key])]
    a, b = s.index('<!-- findings:start -->') + len('<!-- findings:start -->'), s.index('<!-- findings:end -->')
    f.write_text(s[:a] + '\n' + '\n'.join(F) + '\n' + s[b:])
    print(f'README tables: {len(L)} lines')


if __name__ == '__main__':
    main()
