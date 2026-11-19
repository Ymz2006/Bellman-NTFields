"""Findings page of the lift-scale study: report/lift_scale.html from report/runs.json (report.py), report/jumps.json, heading.json,
losscomp.json (forensics scripts), the figures of figs.py and the prose in pagetext.py.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/lift_scale/page.py

The page is published as an artifact with report/*.png as its files (img/<name>.png).
"""
import json
from pathlib import Path
import numpy as np

import pagelib as L
from pagelib import Data, f, hbar, line, table, details, chart, figure, esc, SCENES, TITLE, SHORT, ARM, SLOT

HERE = Path(__file__).resolve().parent
OUT = HERE / 'report'
CORE = ['ntrl', 'lift', 'liftpush', 'pushonly']
BIN_ORDER = ['0.5-2', '2-4', '4-8', '8-16', '16-32', '32-64', '64-128', '128-256']


def build_charts(D, extra_arms):
    C = {}
    scenes = [s for s in SCENES if D.has(s, 'ntrl')]
    arms = CORE + [a for a in extra_arms if a not in CORE]
    leg = [(ARM[a], SLOT.get(a, 5)) for a in arms if any(D.has(s, a) for s in scenes)]

    def groups(key, digits=1, labels=arms):
        out = []
        for s in scenes:
            rows = []
            for a in labels:
                p = D.pool(s, a)
                if p and p.get(key) is not None:
                    note = f"{p['seeds']} training seed{'s' if p['seeds'] > 1 else ''}" + (': ' + ' / '.join(f(v) for v in p['each']) if key == 'solved' and p['seeds'] > 1 else '')
                    rows.append((ARM[a], p[key], SLOT.get(a, 5), note))
            if rows:
                out.append((TITLE[s], rows))
        return out
    C['solved'] = hbar('Queries solved', 'Share of MPPI plans (planning seeds 0 / 1 / 2, pooled over training seeds) that arrive with the exact clearance never negative. Scenes from the smallest to the largest of each family.',
                       groups('solved'), legend=leg, value_name='solved')
    C['within'] = hbar('Travel times within 10 % of the truth', 'Random pairs (free configuration, goal) with clearance of at least 0.1 cells, against the grid solver’s travel time.',
                       groups('within10'), legend=leg, value_name='within 10 %')
    oarms = ['ntrl', 'ntrl_oct', 'liftpush', 'liftpush_oct', 'lift_c10_oct']
    oleg = [(ARM[a], SLOT.get(a, 5)) for a in oarms if any(D.has(s, a) for s in scenes)]
    C['solved_oct'] = hbar('Queries solved with finer position features', 'The octave arms next to ntrl-demo and lift + push on the scenes where they were run (octaves ×2, ×4, ×8 on the 22-cell maze, scaled with the box side elsewhere).',
                           [g for g in groups('solved', labels=oarms) if len(g[1]) > 2], legend=oleg, value_name='solved')
    C['within_oct'] = hbar('Travel times within 10 % with finer position features', 'Same arms.', [g for g in groups('within10', labels=oarms) if len(g[1]) > 2], legend=oleg, value_name='within 10 %')
    C['wall'] = hbar('Wall layer: S × |∇T| in the innermost band', 'Median over held-out band points with speed below 0.06 (the floor is 0.012). 1 means the eikonal equation holds there; below 1 the slope falls short.',
                     groups('wall_innermost', labels=['ntrl', 'lift', 'liftpush']), xmax=1.2, ticks=(0, 0.25, 0.5, 0.75, 1.0), digits=2, suffix='', value_name='S × slope', unit=True,
                     legend=[(ARM[a], SLOT[a]) for a in ('ntrl', 'lift', 'liftpush')])
    C['seen'] = hbar('Detour pairs seen through', 'Random pairs whose true path is at least 30 % longer than the straight segment: share whose learned detour is less than half the true one.',
                     groups('seen_through'), legend=leg, value_name='seen through')
    C['stall'] = hbar('Plans that stop short of the goal', 'Share of plans that did not arrive (stalls in a local minimum of the field or runs out of steps).', groups('stall'), legend=leg, value_name='stopped short')
    C['wallhit'] = hbar('Plans that arrive through an obstacle', 'Share of plans that reach the goal with negative clearance somewhere on the path.', groups('wall'), legend=leg, value_name='through a wall')
    C['los'] = hbar('Line-of-sight pairs within 10 %', 'Pairs whose true travel time is within 5 % of the straight C-space distance.', groups('los_within10'), legend=leg, value_name='within 10 %')

    # wall layer by speed bin, per scene
    sb = ['0-0.02', '0.02-0.06', '0.06-0.15', '0.15-0.4', '0.4-0.8', '0.8-1']
    for s in scenes:
        ser = []
        for a in ('ntrl', 'lift', 'liftpush'):
            r = D.one(s, a)
            if r:
                ser.append((ARM[a], SLOT[a], [(i, r['wall_by_speed'].get(b)) for i, b in enumerate(sb) if r['wall_by_speed'].get(b) is not None], ''))
        if ser:
            C[f'wallspeed_{s}'] = line(f'{TITLE[s]}: S × |∇T| by speed', 'Held-out band points by the speed there (floor 0.012 on the left, free space on the right); seed 87358.', ser,
                                       xlabel='speed at the point', ylabel='S × slope', xticks=list(range(len(sb))), xticklabels={str(i): b for i, b in enumerate(sb)}, yticks=[0, 0.25, 0.5, 0.75, 1.0, 1.25],
                                       ymin=0, ymax=1.4, refs=[(1.0, 'eikonal')], digits=2, xname='speed', height=240)
    # by distance bin, per scene
    for s in scenes:
        rs = [(a, D.pool(s, a)) for a in arms]
        rs = [(a, p) for a, p in rs if p]
        bins = [b for b in BIN_ORDER if b in rs[0][1]['by_bin']]
        C[f'bybin_{s}'] = line(f'{TITLE[s]}: solved by true travel distance', 'Queries binned by the true travel time in cells at full speed; 40 queries per bin where the scene has them.',
                               [(ARM[a], SLOT.get(a, 5), [(i, p['by_bin'][b]) for i, b in enumerate(bins)], '') for a, p in rs],
                               xlabel='true travel distance (cells)', ylabel='solved %', xticks=list(range(len(bins))), xticklabels={str(i): b for i, b in enumerate(bins)},
                               yticks=[0, 25, 50, 75, 100], ymin=0, ymax=100, digits=1, xname='bin', height=260)

    # training course of lift + push: detour index and planning
    def course_series(key, labels, scene_list, by='scene'):
        out = []
        k = 1
        for s in scene_list:
            for a in labels:
                r = D.one(s, a)
                if not r or not r['course']:
                    continue
                pts = [(int(e), c.get(key)) for e, c in sorted(r['course'].items(), key=lambda t: int(t[0]))]
                out.append(((SHORT[s] if by == 'scene' else ARM[a]), (k if by == 'scene' else SLOT.get(a, k)), pts, ''))
                k += 1
        return out
    ep_ticks = [100, 1000, 2000, 3000, 4000, 5000]
    C['course_rho'] = line('Lift + push: detour index during training', 'Median of (T − D) / (T* − D) on detour pairs at the saved epochs of the seed-87358 run; 1 is the truth, 0 the straight distance. The push is strong (λ = 1e-2) from epoch 600 to 2,000 and relaxed to 3e-3 by 3,000.',
                           course_series('rho', ['liftpush'], scenes), xlabel='epoch', ylabel='detour index', xticks=ep_ticks, yticks=[0, 0.5, 1, 1.5], refs=[(1.0, 'truth')], digits=2, xname='epoch')
    C['course_los'] = line('Lift + push: line-of-sight times during training', 'Median learned / true time on line-of-sight pairs.',
                           course_series('los_ratio', ['liftpush'], scenes), xlabel='epoch', ylabel='learned / true', xticks=ep_ticks, yticks=[0.9, 1.0, 1.1], refs=[(1.0, 'truth')], digits=3, xname='epoch')
    # planning at epochs 1000 / 2000 / 3000 / final (planning seed 0)
    ser = []
    for k, s in enumerate(scenes):
        r = D.one(s, 'liftpush')
        if not r or not r['course']:
            continue
        pts = [(int(e), 100 * c['success'][0] / c['success'][1]) for e, c in sorted(r['course'].items(), key=lambda t: int(t[0])) if 'success' in c]
        n = sum(v[1] for v in r['by_bin_n'].values()) / 3
        pts.append((5000, 100 * sum(v[0] for v in r['by_bin_n'].values()) / 3 / n))
        ser.append((SHORT[s], k + 1, pts, ''))
    C['course_solved'] = line('Lift + push: queries solved at saved epochs', 'Planning seed 0 with the checkpoints of epochs 1,000, 2,000 and 3,000; the last point is the final field (mean of the three planning seeds).',
                              ser, xlabel='epoch', ylabel='solved %', xticks=[1000, 2000, 3000, 5000], yticks=[0, 25, 50, 75, 100], ymin=0, ymax=100, digits=1, xname='epoch')
    # plus22: push variants
    var = [a for a in ('liftpush', 'lift_c10', 'lift_x', 'lift_x3', 'lift_r6', 'vecpush') if D.has('plus22', a)]
    ser = []
    for k, a in enumerate(var):
        r = D.one('plus22', a)
        ser.append((ARM.get(a, a), k + 1, [(int(e), c.get('rho')) for e, c in sorted(r['course'].items(), key=lambda t: int(t[0]))], ''))
    C['variants_rho'] = line('22 × 22 plus maze: detour index of the push variants during training', 'Same measure, one run per variant (seed 87358).', ser, xlabel='epoch', ylabel='detour index',
                             xticks=ep_ticks, yticks=[0, 0.5, 1, 1.5], refs=[(1.0, 'truth')], digits=2, xname='epoch')
    ser = []
    for k, a in enumerate(var):
        r = D.one('plus22', a)
        ser.append((ARM.get(a, a), k + 1, [(int(e), c.get('los_ratio')) for e, c in sorted(r['course'].items(), key=lambda t: int(t[0]))], ''))
    C['variants_los'] = line('22 × 22 plus maze: line-of-sight times of the push variants', 'Median learned / true time on line-of-sight pairs. A push that inflates the field shows here.', ser, xlabel='epoch', ylabel='learned / true',
                             xticks=ep_ticks, yticks=[0.9, 1.0, 1.1], refs=[(1.0, 'truth')], digits=3, xname='epoch')

    # scale: solved and seen-through against scene size
    fam = (('plus mazes', ['plus22', 'plus30', 'plus40']), ('floor plans', ['apt32', 'Allensville', 'apt64', 'Lakeville', 'Klickitat']))
    for key, title, sub, yt, ymax in (('solved', 'Queries solved against scene size', 'Box side in cells on the horizontal axis; one line per arm and scene family.', [0, 25, 50, 75, 100], 100),
                                      ('within10', 'Travel times within 10 % against scene size', 'Same axes.', [0, 25, 50, 75, 100], 100)):
        for fname, flist in fam:
            ser = []
            for a in ('ntrl', 'lift', 'liftpush'):
                pts = [(D.blocked[s]['E'], D.pool(s, a)[key]) for s in flist if D.pool(s, a) and s in D.blocked]
                if len(pts) > 1:
                    ser.append((ARM[a], SLOT[a], sorted(pts), ''))
            if not ser:
                continue
            C[f'scale_{key}_{"plus" if fname.startswith("plus") else "plan"}'] = line(f'{title}: {fname}', sub, ser, xlabel='box side (cells)', ylabel='%', yticks=yt, ymin=0, ymax=ymax,
                                                                                       xticks=[int(round(D.blocked[s]['E'])) for s in flist if s in D.blocked], digits=1, xname='cells', height=260)

    # heading slope
    hd = {(h['scene'], h['lift'], h['push'], h.get('zero_heading', False)): h for h in D.heading}
    for sname, key in (('plus maze 22 x 22', 'plus22'), ('apartment (32 cells)', 'apt32')):
        ser_r, ser_h = [], []
        for lab, slot, k in (('ntrl-demo', 1, (sname, 'none', 'none', False)), ('lift', 2, (sname, 'time', 'none', False)), ('lift + push', 3, (sname, 'time', 'page', False)),
                             ('lift, heading-blind network', 5, (sname, 'time', 'none', True))):
            if k in hd:
                items = list(hd[k]['by_distance'].items())
                ser_r.append((lab, slot, [(i, v['radial']) for i, (b, v) in enumerate(items)], ''))
                ser_h.append((lab, slot, [(i, v['heading']) for i, (b, v) in enumerate(items)], ''))
                labels = {str(i): b for i, (b, v) in enumerate(items)}
        if ser_r:
            C[f'heading_radial_{key}'] = line(f'{TITLE[key]}: radial slope on line-of-sight pairs', 'Median of ∇T · u, u pointing from the goal to the configuration: 1 for the true field.', ser_r,
                                              xlabel='straight distance (cells)', ylabel='radial slope', xticks=list(range(len(labels))), xticklabels=labels, yticks=[0.5, 0.75, 1.0], refs=[(1.0, 'truth')], digits=3, xname='distance', height=260)
            C[f'heading_phi_{key}'] = line(f'{TITLE[key]}: slope along the heading coordinate', 'Median |∂T/∂φ| on the same pairs (the heading coordinate spans 0.39 cells in all).', ser_h,
                                           xlabel='straight distance (cells)', ylabel='heading slope', xticks=list(range(len(labels))), xticklabels=labels, yticks=[0, 0.25, 0.5], digits=3, xname='distance', height=260)

    # steps across walls
    by_run = {}
    for j in D.jumps:
        by_run[(j['run'], j['epoch'])] = j
    ser = []
    cand = [('ntrl-demo', 1, D.one('plus22', 'ntrl'), None), ('lift', 2, D.one('plus22', 'lift'), None), ('lift + push, epoch 2,000 (strong push)', 4, D.one('plus22', 'liftpush'), 2000),
            ('lift + push, final', 3, D.one('plus22', 'liftpush'), None)]
    for lab, slot, r, ep in cand:
        if r and (r['id'], ep) in by_run:
            j = by_run[(r['id'], ep)]
            ser.append((lab, slot, [(v['true'], v['learned']) for v in j['by_true'].values()], ''))
    if ser:
        ser.append(('truth', 6, [(0, 0), (14, 14)], '4 3'))
        C['jumps'] = line('22 × 22 plus maze: the step in travel time a field holds across a wall', 'Mirror configurations on the two sides of a wall (0.8 cells apart): the true difference of their travel times to a goal on the horizontal axis, the field’s on the vertical (medians by class).',
                          ser, xlabel='true step across the wall (cells)', ylabel='learned step (cells)', xticks=[0, 2, 4, 6, 8, 10, 12, 14], yticks=[0, 5, 10, 15, 20], digits=2, xname='true step')
    return C


def main():
    import pagetext as T
    D = Data()
    C = build_charts(D, T.EXTRA_ARMS)
    body, ctx = T.body(D, C)
    body = T.finish(D, C, body, ctx)
    css = (HERE / 'page_assets/style.css').read_text()
    js = (HERE / 'page_assets/charts.js').read_text()
    page = (f'<title>{T.TITLE_TAG}</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">\n'
            f'<style>\n{css}\n{T.EXTRA_CSS}</style>\n<div class="wrap">\n{body}\n</div>\n<div class="tip" id="tip" hidden></div>\n'
            f'<script id="data" type="application/json">{json.dumps(dict(charts=C))}</script>\n<script>\n{js}\n</script>\n')
    (OUT / 'lift_scale.html').write_text(page)
    print(f'{OUT / "lift_scale.html"}: {len(page) / 1024:.0f} KB, {len(C)} charts')


if __name__ == '__main__':
    main()
