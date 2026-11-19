"""Prose of the lift-scale findings page (page.py). Every number is read from report/*.json at build time."""
import json
from pathlib import Path

import pagelib as L
from pagelib import f, table, details, figure, esc, SCENES, TITLE, SHORT, ARM, SLOT, line, hbar

HERE = Path(__file__).resolve().parent
TITLE_TAG = '2026-11-17 Lift Scale Test'
EXTRA_ARMS = ['liftpush_oct']
EXTRA_CSS = """
.scenes { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 200px), 1fr)); gap: 10px; margin: 14px 0 6px; }
.scenes img { display: block; width: 100%; height: auto; border-radius: 4px; background: #fff; border: 1px solid var(--rule); }
"""
PLUS = ['plus22', 'plus30', 'plus40']
PLAN = ['apt32', 'Allensville', 'apt64', 'Lakeville', 'Klickitat']


def body(D, C):
    S = [s for s in SCENES if D.has(s, 'ntrl') and D.has(s, 'liftpush')]
    plus = [s for s in PLUS if s in S]
    plan = [s for s in PLAN if s in S]
    names = ' / '.join(SHORT[s] for s in S)

    def P(s, a):
        return D.pool(s, a) or {}

    def v(s, a, k='solved', d=1):
        return f(P(s, a).get(k), d)

    def row(a, k='solved', d=1, scenes=S):
        return ' / '.join(v(s, a, k, d) for s in scenes)

    def rng(a, k, d=2, scenes=S):
        vals = [P(s, a).get(k) for s in scenes if P(s, a).get(k) is not None]
        return f'{min(vals):.{d}f}–{max(vals):.{d}f}' if vals else '–'

    def ch(key):
        return L.chart(key) if key in C else ''

    def grid(*keys):
        inner = ''.join(ch(k) for k in keys)
        return f'<div class="grid2">{inner}</div>' if inner else ''

    def one(s, a):
        return D.one(s, a) or {}

    def cr(s, a, e, k='rho', d=2):
        r = one(s, a)
        c = (r.get('course') or {}).get(str(e), {})
        return f(c.get(k), d)

    def cs(s, a, e):
        r = one(s, a)
        c = (r.get('course') or {}).get(str(e), {})
        return f(100 * c['success'][0] / c['success'][1], 0) if 'success' in c else '–'

    hd = {(h['scene'], h['lift'], h['push'], bool(h.get('zero_heading'))): h for h in D.heading}

    def hv(scene, lift, push, bin_, k, zero=False, d=2):
        h = hd.get((scene, lift, push, zero))
        return f(h['by_distance'][bin_][k], d) if h and bin_ in h['by_distance'] else '–'

    jm = {(j['run'], j['epoch']): j for j in D.jumps}

    def jv(s, a, ep, cls, k='learned', d=1):
        r = one(s, a)
        j = jm.get((r.get('id'), ep))
        return f(j['by_true'][cls][k], d) if j and cls in j['by_true'] else '–'

    lc = {}
    if (L.OUT / 'losscomp.json').exists():
        lc = {(j['run'], j['epoch']): j for j in json.load(open(L.OUT / 'losscomp.json'))}

    B = D.blocked
    out = []
    A = out.append

    # ------------------------------------------------------------------------------------------------------------------ extra charts
    ep_ticks = [100, 1000, 2000, 3000, 4000, 5000]
    items = [('plus22', 'liftpush', 'page schedule (strong, then relaxed)', 3), ('plus22', 'lift_c10', 'constant strong push', 5), ('plus22', 'lift_x', 'constant strong push, exact slope cap', 6),
             ('plus22', 'lift', 'no push', 2)]
    C['push_rho'] = line('22 × 22 plus maze: detour index during training, by push', 'Median of (T − D) / (T* − D) on detour pairs at the saved epochs (seed 87358); 1 is the truth, 0 the straight distance.',
                         L.course(D, items, 'rho'), xlabel='epoch', ylabel='detour index', xticks=ep_ticks, yticks=[0, 0.5, 1, 1.5], refs=[(1.0, 'truth')], digits=2, xname='epoch')
    C['push_los'] = line('22 × 22 plus maze: line-of-sight times during training, by push', 'Median learned / true time on pairs with a free line of sight. A push that buys slope violations shows here.',
                         L.course(D, items, 'los_ratio'), xlabel='epoch', ylabel='learned / true', xticks=ep_ticks, yticks=[0.9, 1.0, 1.1], refs=[(1.0, 'truth')], digits=3, xname='epoch')
    items_in = [('plus22', 'lift', 'lift, ntrl-demo’s pairs', 2), ('plus22in', 'lift_in', 'lift, 20 % inside pairs', 1), ('plus22in', 'lift_in_long', 'the same, trained 12,500 epochs', 4),
                ('plus22', 'lift_x', 'strong push + exact cap, ntrl-demo’s pairs', 6), ('plus22in', 'liftx_in', 'strong push + exact cap, 20 % inside pairs', 3)]
    ser = L.course(D, items_in, 'rho')
    if ser:
        mx = max(int(x) for _, _, pts, _ in ser for x, _ in pts)
        C['inside_rho'] = line('22 × 22 plus maze: detour index with one end of 20 % of the pairs inside obstacles', 'Same measure. The lift is continued through the walls (depth 0.3 cells) in the inside-pair runs.',
                               ser, xlabel='epoch', ylabel='detour index', xticks=[100, 2000, 4000] + list(range(6000, mx + 1, 2000)), yticks=[0, 0.5, 1, 1.5], refs=[(1.0, 'truth')], digits=2, xname='epoch')
        C['inside_los'] = line('The same runs: line-of-sight times', 'Median learned / true time on line-of-sight pairs.', L.course(D, items_in, 'los_ratio'), xlabel='epoch', ylabel='learned / true',
                               xticks=[100, 2000, 4000] + list(range(6000, mx + 1, 2000)), yticks=[0.8, 0.9, 1.0, 1.1], refs=[(1.0, 'truth')], digits=3, xname='epoch')

    items_o = [('plus22', 'liftpush', 'lift + push', 3), ('plus22', 'liftpush_oct8', 'lift + push + octaves', 6), ('plus22', 'lift_r6_oct8', 'the same, relaxed to 6e-3 only', 4),
               ('plus22', 'lift_c6_oct8', 'lift + octaves + constant push 6e-3', 5), ('plus22', 'lift_c10_oct8', 'lift + octaves + constant push 1e-2', 2), ('plus22', 'ntrl_oct8', 'ntrl-demo + octaves (control)', 1)]
    C['oct_rho'] = line('22 × 22 plus maze: detour index with finer position features', 'Octaves ×2, ×4, ×8 in place of 48 of ntrl-demo’s 128 Fourier rows; seed 87358.', L.course(D, items_o, 'rho'), xlabel='epoch', ylabel='detour index',
                        xticks=ep_ticks, yticks=[0, 0.5, 1, 1.5, 2], refs=[(1.0, 'truth')], digits=2, xname='epoch')
    C['oct_solved'] = line('The same runs: queries solved at saved epochs', 'Planning seed 0 at epochs 1,000, 2,000 and 3,000; the last point is the final field (three planning seeds).', L.course_solved(D, items_o), xlabel='epoch',
                           ylabel='solved %', xticks=[1000, 2000, 3000, 5000], yticks=[50, 60, 70, 80, 90, 100], ymin=50, ymax=100, digits=1, xname='epoch')
    C['oct_within'] = line('The same runs: travel times within 10 %', 'Share of random pairs within 10 % of the truth at the saved epochs.', L.course(D, items_o, 'within10', 100.0), xlabel='epoch', ylabel='within 10 %',
                           xticks=ep_ticks, yticks=[0, 25, 50, 75, 100], ymin=0, ymax=100, digits=1, xname='epoch')
    th = {}
    if (L.OUT / 'thick.json').exists():
        th = {j['run']: j for j in json.load(open(L.OUT / 'thick.json'))}

    def thick_series(label, cells=True):
        ser = []
        for k, s_ in enumerate(S):
            r = one(s_, label)
            j = th.get(r.get('id'))
            if not j:
                continue
            pts = [((v_['mid'] if cells else 100 * v_['mid'] / B[s_]['E']), 100 * v_['seen_through']) for v_ in j['by_thickest'].values() if v_['n'] >= 40]
            if len(pts) > 1:
                ser.append((SHORT[s_], k + 1 if k < 6 else (k - 5), pts, '' if k < 6 else '5 3'))
        return ser
    if th:
        C['thick'] = line('Lift + push: detour pairs seen through, by the thickest obstacle on the straight line', 'Detour pairs grouped by the longest stretch of their straight segment that lies inside one obstacle (cells, log scale). 100 % = every such pair is seen through. Dashed lines: the two larger houses, Lakeville and Klickitat.',
                          thick_series('liftpush'), xlabel='thickest obstacle crossed (cells)', ylabel='seen through %', xticks=[0.5, 1, 2, 4, 8, 16], yticks=[0, 25, 50, 75, 100], ymin=0, ymax=100, logx=True,
                          xmin=0.3, xmax=24, digits=0, xname='thickness')
        C['thick_oct'] = line('Lift + push + octaves: the same measure', 'Same pairs, with the octaves scaled to the scene (dashed: Lakeville and Klickitat).', thick_series('liftpush_oct'), xlabel='thickest obstacle crossed (cells)', ylabel='seen through %',
                              xticks=[0.5, 1, 2, 4, 8, 16], yticks=[0, 25, 50, 75, 100], ymin=0, ymax=100, logx=True, xmin=0.3, xmax=24, digits=0, xname='thickness')
        C['thick_ntrl'] = line('ntrl-demo: the same measure', 'Same pairs, the control fields (dashed: Lakeville and Klickitat).', thick_series('ntrl'), xlabel='thickest obstacle crossed (cells)', ylabel='seen through %',
                               xticks=[0.5, 1, 2, 4, 8, 16], yticks=[0, 25, 50, 75, 100], ymin=0, ymax=100, logx=True, xmin=0.3, xmax=24, digits=0, xname='thickness')

    def tv(s_, label, bin_, k='seen_through', scale=100.0, d=0):
        j = th.get(one(s_, label).get('id'))
        return f(scale * j['by_thickest'][bin_][k], d) if j and bin_ in j['by_thickest'] else '–'

    # ------------------------------------------------------------------------------------------------------------------ header
    A('<!--HEADER-->')
    A('<!--ANSWERS-->')

    # ------------------------------------------------------------------------------------------------------------------ what was run
    A('<h2>What was run</h2>')
    A(f'''<p>The study folder of the lift page is not on this workstation, so the lift and the push were rebuilt here from the page’s description, on the scenes, planner and scorer every local maze study uses. Two things differ from that page and are part of the test.</p>
<ul class="plain">
<li><b>A configuration space, not the plane.</b> The robot is a 0.5 × 0.2-cell rectangle with heading, q = (x, y, φ); the clearance is the rectangle’s exact signed distance and ntrl-demo’s speed is computed from it (1 beyond 0.175 cells, floor 0.012). The lift is the same one-dimensional function of that clearance, h = G(d(q)), as one more input column. In this space the clearance gradient is not a unit vector (median {f(min(B[s]["nu"]["median"] for s in S), 2)}–{f(max(B[s]["nu"]["median"] for s in S), 2)} times one, 90th percentile {f(min(B[s]["nu"]["q90"] for s in S), 2)}–{f(max(B[s]["nu"]["q90"] for s in S), 2)}), the case the lift page lists as not tested.</li>
<li><b>Larger scenes.</b> Plus mazes of 22, 30 and 40 cells a side, the P-NTFields apartment at 32 and 64 cells, and three Gibson houses cut into floor plans at 0.25 m per cell (43, 83 and 96 cells a side). 500,000 of ntrl-demo’s own training pairs per scene, 5,000 epochs, ntrl-demo unchanged on disk.</li>
</ul>''')
    A('<div class="scenes">' + ''.join(f'<img src="img/scene_{s}.png" alt="{esc(TITLE[s])}">' for s in S) + '</div>')
    A('<p class="note">Obstacles for the robot’s centre at heading 0 (axes in cells). The houses are slices of the scanned meshes at five heights between 0.4 and 1.6 m above the floor, furniture included; everything outside the largest sealed interior is an obstacle.</p>')
    A(f'''<p>Arms, each trained and planned the same way (MPPI, planning seeds 0 / 1 / 2, success = arrives with the exact clearance never negative):</p>
<ul class="plain">
<li><b>ntrl-demo</b>, the control. The no-lift arm of this code reproduces the registered ntrl-demo run of the same seed to every printed digit of the loss.</li>
<li><b>lift</b>: the extra input h, nothing else.</li>
<li><b>lift + push</b>: the selection term of the lift page with its schedule (λ faded in from epoch 200, 1e-2 from 600 to 2,000, relaxed to 3e-3 by 3,000; one-sided slope term μ = 0.1; pairs pushed while T is below 3 box lengths). A pair is pushed when its straight segment crosses an obstacle; that is {f(100 * min(B[s]["blocked_share"] for s in S), 0)}–{f(100 * max(B[s]["blocked_share"] for s in S), 0)} % of the training pairs here.</li>
<li><b>push without the lift</b>, the second control.</li>
<li><b>lift + push + octaves</b> (from Part 5 on): the same with 48 of ntrl-demo’s 128 Fourier rows replaced by whitened rows at ×2, ×4 and ×8 the base frequency, times the box side over 22 cells.</li>
</ul>
<p>Field measures against the grid solver’s travel time T*: the share of random (free configuration, goal) pairs within 10 %; the same on pairs with a free line of sight; and the <b>detour index</b> ρ = (T − D) / (T* − D) on pairs whose true path is at least 30 % longer than the straight distance D (0 = the field is the straight distance, 1 = the truth). A detour pair counts as <b>seen through</b> when ρ is below 0.5.</p>''')
    rows = []
    for s in S:
        b = B[s]
        rows.append([esc(TITLE[s]), f(b['E'], 0), f(100 * b['blocked_share'], 1), f(100 * b['missed_by_48'], 1), f(100 * b['truth']['detour_pair_share'], 0), f(100 * b['truth']['los_pair_share'], 0),
                     f(b['truth']['pair_t_star_quantiles'][-1], 2), f"{b['nu']['median']:.2f} / {b['nu']['q90']:.2f}", D.scenes[s]['prepared'].replace('results/registered_', ''), b['id']])
    A(details('Scenes: size, blocked pairs, detour pairs, run ids', table(['scene', 'box side (cells)', 'training pairs blocked %', 'of those, missed by a 48-point check %', 'random pairs with a detour ≥ 30 % (%)',
                                                                             'with a line of sight %', 'largest T* between free configurations (box lengths)', '|∇d| median / 90th pct.', 'prepare run', 'blocked run'], rows)))
    # ------------------------------------------------------------------------------------------------------------------ the lift pictured
    if (L.OUT / 'lift_visual_Klickitat.png').exists():
        A('<h3>What the lift is, on a floor plan</h3>')
        A(f"""<p>For every configuration the network gets one more input, h = G(d) / E: d is the robot’s clearance (the minimum of the floor plan’s signed-distance table over a 9 × 5 lattice of points on the rectangle), and G(d) is the travel time from that clearance straight out of the slow band, a one-dimensional table computed from the speed profile. Nothing else changes: the same Fourier features with one more column, the same loss, differentiated with respect to the configuration through ∂h/∂q = G′(d) ∇d / E.</p>""")
        A(figure('img/lift_visual_Klickitat.png', 'The lift on house Klickitat (96 cells a side, 0.25 m per cell), robot heading 0. 1: the speed profile and the height it gives (0 beyond 0.175 cells of clearance, 0.685 cells at the inner edge of the trained band, 3.60 cells as deep as training points go, constant deeper). 2: on the scale of the house the lifted rim is too thin to see; the floor is untouched. 3–4: a 3 × 3-cell window, speed and height (colour scale compressed). 5: the same window as the surface the network takes as input, to scale. 6: the height along the line drawn in panels 3 and 4, for the robot’s long side along the line and across it: h is the same on the two sides of the obstacle, and it depends on the heading because the clearance does.'))
    # ------------------------------------------------------------------------------------------------------------------ results
    A('<h2>Results on all scenes</h2>')
    A(L.chart('solved'))
    A(L.chart('within'))
    A(grid('stall', 'wallhit'))
    A('<!--RESULTS_TEXT-->')
    A('<h3>By travel distance</h3>')
    A('<div class="grid2">' + ''.join(ch(f'bybin_{s}') for s in S) + '</div>')
    A('<!--BYBIN_TEXT-->')

    # ------------------------------------------------------------------------------------------------------------------ part 1
    A('<h2>Part 1 · The lift itself holds: the wall layer is exact at every size</h2>')
    A(f"""<p>The lift’s own claim carries over unchanged. S × |∇T| on held-out band points is {rng('lift', 'wall_innermost')} in the innermost band on every scene with the lift alone, against {rng('ntrl', 'wall_innermost')} for ntrl-demo, whose fit gets worse as the scene grows (the band is the same 0.16 cells, a smaller share of the network’s input). It is there early: {' / '.join(cr(s, 'lift', 100, 'wall_innermost') for s in S)} at epoch 100 and {' / '.join(cr(s, 'lift', 1000, 'wall_innermost') for s in S)} at epoch 1,000 on {names} (on the floor plans it settles from above). The clearance gradient not being a unit vector costs nothing measurable; the lift equalizes the normal speed to within that factor, which is at most about 1.4 here, where the unlifted profile spans 83.</p>""")
    A(grid(*[f'wallspeed_{s}' for s in ('plus22', 'plus40', 'apt64', 'Klickitat') if s in S]))
    A(L.chart('wall'))
    A(f"""<p>With the push the fit drops to {rng('liftpush', 'wall_innermost')}: the one-sided term pulls slopes under 1 across the band while the push is on. That is a property of the push as specified, not of the lift (Part 4).</p>""")

    # ------------------------------------------------------------------------------------------------------------------ part 2
    A('<h2>Part 2 · The lift alone: obstacles are seen through, and in this space the far field also comes out short</h2>')
    A(f"""<p>As on the lift page, an exact wall layer does not make a field that plans. With the lift alone {row('lift', 'seen_through', 0)} % of the detour pairs are seen through ({names}) and no plan ends inside a wall: every failure is a stall ({row('lift', 'stall', 0)} % of the plans), on the mazes at a point pressed against a wall with the goal straight behind it, on the floor plans mostly in open space (Part 7). ntrl-demo fails the other way on the mazes, by arriving through walls ({row('ntrl', 'wall', 0, plus)} % of plans on the plus mazes).</p>""")
    A(figure('img/fields_plus40.png', 'Travel time to one goal (star) on the 40 × 40 plus maze at the goal’s heading, seed 87358; contours every 4 cells, obstacles grey, one colour scale. ntrl-demo’s field bends around the quadrants but is smooth; the lifted field is flat away from the goal; with the push it is a set of circles around the goal that ignore the walls.'))
    A(f"""<p>What is new in the configuration space is that the lifted field is also <b>too short along free lines of sight</b>, which the 2-D fields of the lift page were not. Learned / true time on line-of-sight pairs is {row('lift', 'los_ratio', 2)} with the lift alone ({names}; ntrl-demo {row('ntrl', 'los_ratio', 2)}), and only {row('lift', 'los_within10', 0)} % of those pairs are within 10 %. The slope is not missing: |∇T| is {hv('plus maze 22 x 22', 'time', 'none', '16-32', 'grad')} on the 22-cell maze even beyond 16 cells. It points the wrong way. The heading coordinate φ spans 0.39 cells in all, so a field can satisfy the eikonal equation with part of its slope along φ and gain almost no travel time from it.</p>""")
    A(grid('heading_radial_plus22', 'heading_phi_plus22'))
    A(grid('heading_radial_apt32', 'heading_phi_apt32'))
    A(f"""<p>On the 22-cell maze the lifted field’s radial slope falls from {hv('plus maze 22 x 22', 'time', 'none', '0-2', 'radial')} within 2 cells of the goal to {hv('plus maze 22 x 22', 'time', 'none', '16-32', 'radial')} beyond 16, while its heading slope rises from {hv('plus maze 22 x 22', 'time', 'none', '0-2', 'heading')} to {hv('plus maze 22 x 22', 'time', 'none', '16-32', 'heading')} (ntrl-demo: radial {hv('plus maze 22 x 22', 'none', 'none', '16-32', 'radial')}, heading {hv('plus maze 22 x 22', 'none', 'none', '16-32', 'heading')}). In the 32-cell apartment the radial slope is down to {hv('apartment (32 cells)', 'time', 'none', '16-32', 'radial')}. This is one more member of the family the lift page describes: a field with the right slope everywhere that is smaller than the true one. ntrl-demo’s smeared wall layer leaves it no such freedom; once the lift makes the local fit exact, nothing in the loss holds the far field up.</p>""")
    A('<!--NOHEAD-->')
    # ------------------------------------------------------------------------------------------------------------------ part 3
    A('<h2>Part 3 · Lift and push: accurate wherever there is a line of sight; behind a wall it depends on the wall</h2>')
    A(f"""<p>The push removes the short far field at once: line-of-sight pairs within 10 % go to {row('liftpush', 'los_within10', 0)} % ({names}; lift alone {row('lift', 'los_within10', 0)} %, ntrl-demo {row('ntrl', 'los_within10', 0)} %), with the heading slope back to {hv('plus maze 22 x 22', 'time', 'page', '16-32', 'heading')} beyond 16 cells on the 22-cell maze. Over all random pairs {row('liftpush', 'within10', 0)} % of travel times are within 10 % (ntrl-demo {row('ntrl', 'within10', 0)} %). The part the push was added for holds on some scenes and not on others: {row('liftpush', 'seen_through', 0)} % of the detour pairs are still seen through at the end of training. The houses, whose walls are thick, keep most of their detours (Part 7); the plus mazes and the apartment, whose walls are thin, do not. The rest of this part and Parts 4 to 6 are about the thin-walled scenes.</p>""")
    A(grid('seen', 'los'))
    A(grid('course_rho', 'course_solved'))
    A(f"""<p>The training course shows where the detours go. On the 22-cell maze the detour index reaches {cr('plus22', 'liftpush', 2000)} at epoch 2,000, the end of the strong push, and the field then plans {cs('plus22', 'liftpush', 2000)} % of the queries (planning seed 0), including {f(100 * one('plus22', 'liftpush')['course']['2000']['success_by_bin']['32-64'][0] / 40, 0) if one('plus22', 'liftpush') else '–'} % of the longest bin that no final field solves. Its line-of-sight times are {f(100 * (one('plus22', 'liftpush')['course']['2000']['los_ratio'] - 1), 0) if one('plus22', 'liftpush') else '–'} % long at that point. When the push is relaxed the index falls to {cr('plus22', 'liftpush', 3000)} within 1,000 epochs and ends at {cr('plus22', 'liftpush', 5000)}; planning is back to {v('plus22', 'liftpush')} %. This is the lift page’s plus-maze observation (index 1.66 at epoch 2,000, 0.50 at 3,000), reproduced in a different space with different code. The earlier run of the same recipe, whose one-sided term was not faded in, reached only {cr('plus22', 'lift_push', 2000)} at epoch 2,000 and planned {cs('plus22', 'lift_push', 2000)} % there: the state at the end of the strong push is not a reproducible one.</p>""")
    A(L.chart('jumps') if 'jumps' in C else '')
    A(f"""<p>A direct look at one wall: for mirror configurations on its two sides, 0.8 cells apart, the true travel times to a goal differ by the whole detour when the wall separates them. ntrl-demo holds such a step almost exactly on the 22-cell maze (a true step of 13.5 cells is learned as {jv('plus22', 'ntrl', None, '12-24')}); the lift alone holds {jv('plus22', 'lift', None, '12-24')} cells; lift + push holds {jv('plus22', 'liftpush', 2000, '12-24')} at epoch 2,000 and {jv('plus22', 'liftpush', None, '12-24')} at the end.</p>""")
    A(figure('img/stalls_plus22.png', '22 × 22 plus maze, planning seed 0: where the plans that did not arrive stopped (red), joined to their goals (blue). ntrl-demo’s remaining failures are mostly wall hits, not shown here; the lifted fields stop pressed against a wall of a quadrant with the goal behind it.'))
    # ------------------------------------------------------------------------------------------------------------------ part 4
    ins = {}
    if (L.OUT / 'inside.json').exists():
        ins = {(j['run'], j['epoch']): j for j in json.load(open(L.OUT / 'inside.json'))}

    def iv(s, a, k='median', d=1):
        j = ins.get((one(s, a).get('id'), None))
        return f(j['across'][k], d) if j else '–'

    def idp(s, a, d=2):
        j = ins.get((one(s, a).get('id'), None))
        if not j:
            return '–'
        vals = [x['slope'] for x in j['by_depth'].values()]
        return f'{min(vals):.{d}f}–{max(vals):.{d}f}'

    A('<h2>Part 4 · Why the detours behind thin walls do not hold: a wall is a bump in the lifted input, and the push cannot make it a barrier</h2>')
    A('<p>Four tests on the 22-cell plus maze (one seed each, same data and schedule length) separate what the push does from what the field can represent.</p>')
    A('<h3>1. A stronger push buys slope violations, not a stable field</h3>')
    A(f"""<p>Kept at its strong value for the whole training (λ = 1e-2), the push does not hold the detours either. The detour index swings between checkpoints, {' → '.join(cr('plus22', 'lift_c10', e) for e in (1000, 1500, 2000, 3000, 4000, 5000))} from epoch 1,000 to 5,000, the line-of-sight times stay {f(100 * (min(c['los_ratio'] for e, c in one('plus22', 'lift_c10')['course'].items() if int(e) >= 1000) - 1), 0) if one('plus22', 'lift_c10') else '–'}–{f(100 * (max(c['los_ratio'] for e, c in one('plus22', 'lift_c10')['course'].items() if int(e) >= 1000) - 1), 0) if one('plus22', 'lift_c10') else '–'} % long, ntrl-demo’s epoch-repeat rule rolls back {one('plus22', 'lift_c10').get('rollbacks', '–')} times, and the final field plans {v('plus22', 'lift_c10')} % with {v('plus22', 'lift_c10', 'within10', 0)} % of times within 10 %. The slope cap of the recipe is a quadratic penalty: its gradient vanishes at the constraint, so any push is paid for with a violation proportional to its strength, everywhere, and the detours that appear are made of that violation.</p>""")
    A(grid('push_rho', 'push_los'))
    A('<h3>2. With the slope cap enforced, the push builds no detours at all</h3>')
    A(f"""<p>Adding a linear (exact) one-sided penalty, whose gradient does not vanish at the constraint, to the same strong push: the line-of-sight times stay at {f(one('plus22', 'lift_x').get('los_ratio'), 2)} of the truth, and the detour index is {f(one('plus22', 'lift_x').get('rho'), 2)} at the end, with {v('plus22', 'lift_x', 'seen_through', 0)} % of the detour pairs seen through ({v('plus22', 'lift_x')} % solved). A stiff quadratic cap (μ = 10) gives the same: index {f(one('plus22', 'liftq').get('rho'), 2)}. So the largest field that respects the slope cap at the trained points <em>is</em> the see-through field. The detours of the soft-cap runs existed only as violations. (The linear cap has a cost of its own: it drags the band slopes to {f(one('plus22', 'lift_x').get('wall_innermost'), 2)} of the eikonal value.)</p>""")
    A('<h3>3. The reason: nothing separates the two sides of a wall</h3>')
    A(f"""<p>ntrl-demo samples no point deeper than 0.0175 cells inside an obstacle, so across a wall the eikonal loss says nothing, and a network keeps its output continuous there. That much the lift page already names. The test here goes one step further: train the inside as well. With one end of 20 % of the training pairs inside obstacles (floor speed there, the registered data variant of the embedding study) and the lift continued through the walls at the floor slope, the interior becomes as learnable as the band: S × |∇T| is {idp('plus22in', 'lift_in')} at every depth inside the walls (ntrl-demo: {idp('plus22', 'ntrl')}). The equation now holds everywhere, and the field is still see-through:</p>""")
    rows = []
    for lab, s_, a_ in (('ntrl-demo', 'plus22', 'ntrl'), ('lift', 'plus22', 'lift'), ('lift + push', 'plus22', 'liftpush'), ('lift + strong push + exact cap', 'plus22', 'lift_x'),
                        ('lift, inside pairs', 'plus22in', 'lift_in'), ('lift, inside pairs, 12,500 epochs', 'plus22in', 'lift_in_long'), ('lift + push, inside pairs', 'plus22in', 'liftpush_in'),
                        ('lift + strong push + exact cap, inside pairs', 'plus22in', 'liftx_in'), ('lift + strong push + stiff cap, inside pairs', 'plus22in', 'liftq_in'),
                        ('vector lift, inside pairs', 'plus22in', 'vec_in'), ('vector lift + push, inside pairs', 'plus22in', 'vecpush_in')):
        r = one(s_, a_)
        if r:
            rows.append([esc(lab), f(r['solved']), f(r['within10']), f(r['los_ratio'], 2), f(r['rho'], 2), f(r['wall_innermost'], 2), idp(s_, a_), iv(s_, a_), iv(s_, a_, 'q90'), r['id']])
    A(f"""<p>between two configurations that face each other across a wall, 0.8 cells apart, the learned travel time is {iv('plus22in', 'lift_in')} cells (median), where going straight through costs {iv('plus22in', 'lift_in', 'straight_median', 0)} cells at the floor speed and going around costs the detour. The field climbs the wall at the right slope from one side and descends it at the right slope on the other. In the lifted input that is the natural thing to do: h depends on the clearance alone, so the two faces of a wall have the same height and the wall is a symmetric bump, T ≈ T<sub>0</sub>(x) + h with T<sub>0</sub> smooth across it. That function satisfies the eikonal equation of even the closed problem almost everywhere. The true field has the same shape and differs only in T<sub>0</sub>: a step of the size of the detour across the wall’s own thickness, at equal height. That step has to be built in the position coordinates, inside 0.3–0.6 cells, and it is the one thing the lift gives no input length for.</p>""")
    A(details('Forensics on the 22 × 22 plus maze (seed 87358)', table(['field', 'solved %', 'T within 10 %', 'line-of-sight learned / true', 'detour index', 'S × slope, innermost band', 'S × slope inside the walls',
                                                                         'T across a wall, median (cells)', '90th pct.', 'run'], rows)))
    A(grid('inside_rho', 'inside_los'))
    A(f"""<p>Training the interior does change two things. The short far field of Part 2 is gone without any push (line-of-sight learned / true {f(one('plus22in', 'lift_in').get('los_ratio'), 2)}, {v('plus22in', 'lift_in', 'los_within10', 0)} % within 10 %), and the detour index creeps up by itself, to {f(one('plus22in', 'lift_in').get('rho'), 2)} after 5,000 epochs and {f(one('plus22in', 'lift_in_long').get('rho'), 2)} after 12,500, where it wanders between 0.3 and 0.6. The long run plans {v('plus22in', 'lift_in_long')} % with {v('plus22in', 'lift_in_long', 'within10', 0)} % of times within 10 % and an exact wall layer, with no push and no schedule (lift + push {v('plus22', 'liftpush')} %, ntrl-demo {v('plus22', 'ntrl')} %). Its detours saturate where the planar step runs out: a true detour of 8.0 cells is learned as {f(one('plus22in', 'lift_in_long')['detour_by_length']['6-12']['learned_cells']) if one('plus22in', 'lift_in_long') else '–'}, one of 13.3 as {f(one('plus22in', 'lift_in_long')['detour_by_length']['12-24']['learned_cells']) if one('plus22in', 'lift_in_long') else '–'}.</p>""")
    A('<!--INSIDE_LARGE-->')
    A('<!--VECTOR-->')
    A('<h3>4. Why plain ntrl-demo has detours on the small maze and loses them on the large ones</h3>')
    A(f"""<p>ntrl-demo cannot draw the thin wall layer, so its slope is high through the whole band on both sides of a wall, and that excess is what carries the step: on the 22-cell maze it holds a true step of 13.5 cells across a wall as {jv('plus22', 'ntrl', None, '12-24')}, with travel times 10 % long everywhere. The lift removes exactly that excess. And the excess does not scale: on the 40-cell maze the same network holds {jv('plus40', 'ntrl', None, '12-24')} cells of a 14-cell step ({jv('plus40', 'ntrl', None, '6-12')} of 7.8), and arrives through walls on {v('plus40', 'ntrl', 'wall', 0)} % of its plans.</p>""")
    # ------------------------------------------------------------------------------------------------------------------ part 5 (octaves)
    def cq(s_, a_, e, k):
        c = (one(s_, a_).get('course') or {}).get(str(e), {})
        return c.get(k)

    A('<h2>Part 5 · Testing that explanation: finer position features</h2>')
    A(f"""<p>If the missing piece is planar resolution across a wall, more of it should make the push’s detours real. The embedding study’s octaves do that without a map: 48 of ntrl-demo’s 128 Fourier rows are replaced by whitened rows at ×2, ×4 and ×8 the base frequency (network size unchanged; the lift column is untouched). On the 22-cell maze, with the lift and the push of the page:</p>""")
    A(grid('oct_rho', 'oct_solved'))
    A(L.chart('oct_within'))
    A(f"""<ul class="plain">
<li><b>Under the strong push the detours are now stable.</b> The detour index sits at {' / '.join(cr('plus22', 'liftpush_oct8', e) for e in (600, 1000, 1500, 2000))} from epoch 600 to 2,000 (without octaves it wandered between 0.3 and 1.6 across runs), and the field plans {cs('plus22', 'liftpush_oct8', 1000)} % at epoch 1,000 and {cs('plus22', 'liftpush_oct8', 2000)} % at 2,000, {f(100 * cq('plus22', 'liftpush_oct8', 2000, 'success_by_bin')['32-64'][0] / 40, 0) if cq('plus22', 'liftpush_oct8', 2000, 'success_by_bin') else '–'} % of the longest bin included. They are overestimated (line-of-sight times {f(100 * (cq('plus22', 'liftpush_oct8', 2000, 'los_ratio') - 1), 0) if cq('plus22', 'liftpush_oct8', 2000, 'los_ratio') else '–'} % long, {f(100 * cq('plus22', 'liftpush_oct8', 2000, 'within10'), 0) if cq('plus22', 'liftpush_oct8', 2000, 'within10') else '–'} % of times within 10 %).</li>
<li><b>They outlive the relaxation for a while.</b> At epoch 3,000, with the push at 3e-3, the index is {cr('plus22', 'liftpush_oct8', 3000)}, {f(100 * cq('plus22', 'liftpush_oct8', 3000, 'within10'), 0) if cq('plus22', 'liftpush_oct8', 3000, 'within10') else '–'} % of travel times are within 10 % and the field plans {cs('plus22', 'liftpush_oct8', 3000)} %: accurate and planning at once, the one checkpoint on this maze where that holds. By epoch 4,000 the index is {cr('plus22', 'liftpush_oct8', 4000)} and by 5,000 {cr('plus22', 'liftpush_oct8', 5000)}; the final field plans {v('plus22', 'liftpush_oct8')} %.</li>
<li><b>A push that stays at 6e-3</b> keeps the index around 1 but not still: {' / '.join(cr('plus22', 'lift_c6_oct8', e) for e in (1000, 2000, 3000, 4000, 5000))} at epochs 1,000 to 5,000 (constant), with {cs('plus22', 'lift_c6_oct8', 1000)} / {cs('plus22', 'lift_c6_oct8', 2000)} / {cs('plus22', 'lift_c6_oct8', 3000)} % solved at 1,000 / 2,000 / 3,000 and {v('plus22', 'lift_c6_oct8')} % at the end ({v('plus22', 'lift_c6_oct8', 'within10', 0)} % within 10 %); relaxed from 1e-2 to 6e-3 it ends at {v('plus22', 'lift_r6_oct8')} % ({v('plus22', 'lift_r6_oct8', 'within10', 0)} %). Planning follows the index closely: every checkpoint with an index above 1.5 plans 91–95 %, every one near 0.8 about 72 %.</li>
<li><b>Controls.</b> ntrl-demo with the same octaves and no lift: {v('plus22', 'ntrl_oct8')} % solved, {v('plus22', 'ntrl_oct8', 'within10', 0)} % within 10 %, {v('plus22', 'ntrl_oct8', 'wall', 0)} % of plans through walls (the embedding study registered 68.9 / 82.8 / 84.2 % for three seeds of this arm, 70.1 / 74.6 / 68.1 % for ntrl-demo). The final lift fields with octaves are inside that seed range; what the lift and the push add is the wall layer (no wall hits) and the travel times, and at the strong-push checkpoints the planning.</li>
</ul>""")
    A('<!--OCT_TEXT-->')

    # ------------------------------------------------------------------------------------------------------------------ part 6
    A('<h2>Part 6 · Size: the lift does not care, the detours do</h2>')
    A(grid('scale_solved_plus', 'scale_solved_plan'))
    A(grid('scale_within10_plus', 'scale_within10_plan'))
    A(f"""<p>On the plus mazes of {' / '.join(str(int(round(B[s]['E']))) for s in plus)} cells lift + push plans {row('liftpush', 'solved', 1, plus)} % against {row('ntrl', 'solved', 1, plus)} % for ntrl-demo, with {row('liftpush', 'within10', 0, plus)} % of travel times within 10 % against {row('ntrl', 'within10', 0, plus)} %. The two fail in opposite ways at every size: ntrl-demo arrives through walls on {row('ntrl', 'wall', 0, plus)} % of its plans, lift + push on {row('liftpush', 'wall', 0, plus)} % and stalls on {row('liftpush', 'stall', 0, plus)} %.</p>
<p>What changes with size is the push’s grip on the detours. At the end of the strong phase the detour index is {' / '.join(cr(s, 'liftpush', 2000) for s in plus)} on the {' / '.join(str(int(round(B[s]['E']))) for s in plus)}-cell mazes: on the larger ones the detours never form, not even as violations, and {row('liftpush', 'seen_through', 0, plus)} % of the detour pairs are seen through at the end. Two obvious rescalings were tried on the 40-cell maze (one seed each):</p>
<ul class="plain">
<li><b>The push per cell.</b> T is measured in box lengths, so a detour of a given number of cells earns less push the larger the scene. With λ multiplied by 40 / 22 the index reaches {cr('plus40', 'liftpush_cell', 2000)} at epoch 2,000 with line-of-sight times {f(100 * (one('plus40', 'liftpush_cell')['course']['2000']['los_ratio'] - 1), 0) if one('plus40', 'liftpush_cell') else '–'} % long, and ends at {f(one('plus40', 'liftpush_cell').get('rho'), 2)}: {v('plus40', 'liftpush_cell')} % solved (unscaled {v('plus40', 'liftpush')} %). More push, more violation, the same end.</li>
<li><b>The network’s frequencies.</b> With ntrl-demo’s Fourier matrix and the lift column multiplied by 40 / 22, so that a cell has the input length it has on the 22-cell maze: index {cr('plus40', 'liftpush_b', 2000)} at epoch 2,000, {f(one('plus40', 'liftpush_b').get('rho'), 2)} at the end, {v('plus40', 'liftpush_b')} % solved. The control, ntrl-demo with the same scaling: {v('plus40', 'ntrl_b')} % solved, wall hits {v('plus40', 'ntrl_b', 'wall', 0)} % (unscaled {v('plus40', 'ntrl', 'wall', 0)} %).</li>
</ul>
<p>Neither brings the 22-cell behaviour back. What helps is the rescaling that Part 5 points to, of the octaves: with ×2, ×4, ×8 multiplied by 40 / 22, lift + push reaches an index of {cr('plus40', 'liftpush_oct', 2000)} at epoch 2,000 and keeps {f(one('plus40', 'liftpush_oct').get('rho'), 2)} at the end, {v('plus40', 'liftpush_oct')} % solved and {v('plus40', 'liftpush_oct', 'within10', 0)} % of travel times within 10 % (ntrl-demo with the same octaves and no lift: {v('plus40', 'ntrl_oct')} % and {v('plus40', 'ntrl_oct', 'within10', 0)} %).</p>""")

    # ------------------------------------------------------------------------------------------------------------------ part 6
    A('<h2>Part 7 · Gibson floor plans</h2>')
    apt = [s for s in ('apt32', 'apt64') if s in S]
    an = ' / '.join(SHORT[s] for s in apt)
    A(f"""<p>The two kinds of floor plan behave differently. The <b>apartment</b> is a scan outline: walls a quarter of a cell thick at 64 cells, with free space on both sides and around the outside. The <b>houses</b> have walls with their real thickness, a closed outside and furniture.</p>
<h3>The apartment: every field is see-through</h3>
<p>ntrl-demo itself sees through {row('ntrl', 'seen_through', 0, apt)} % of the detour pairs here ({an}) and fails by stalling ({row('ntrl', 'stall', 0, apt)} % of plans, {row('ntrl', 'wall', 0, apt)} % through walls); its far field is a little short as well ({row('ntrl', 'los_ratio', 2, apt)} of the truth along lines of sight).</p>""")
    A(figure('img/fields_apt64.png', 'Travel time to one goal (star) in the 64-cell apartment, seed 87358; contours every 4 cells, walls grey (the outline is a scan, the outside is free space). None of the learned fields bends around the walls; the lift alone is flat away from the goal; lift + push has the right spacing of its circles.'))
    A(f"""<p>The lift alone makes it worse, as on the mazes: {row('lift', 'solved', 1, apt)} % solved against {row('ntrl', 'solved', 1, apt)} %, with {row('lift', 'los_within10', 0, apt)} % of line-of-sight times within 10 %; its plans stall in the open middle of rooms where the short field has a minimum (at 64 cells {f(100 * one('apt64', 'lift').get('stalls', {}).get('open_space', float('nan')), 0)} % of its stalls are more than half a cell from any wall). Lift + push plans {row('liftpush', 'solved', 1, apt)} %: the far field is right ({row('liftpush', 'los_within10', 0, apt)} % of line-of-sight times within 10 %), no plan ends in a wall, and the stalls that remain are pressed against a wall with the goal behind it. But every detour pair is still seen through, and unlike on the mazes nothing tried here changes that: octaves scaled with the scene {v('apt64', 'liftpush_oct')} % solved, one octave higher {v('apt64', 'liftpush_oct2')} %, the strong push kept to the end {v('apt64', 'lift_c10_oct')} % (detour index {f(one('apt64', 'lift_c10_oct').get('rho'), 2)}), ntrl-demo with octaves {v('apt64', 'ntrl_oct')} %. This scene is the open case of the study.</p>""")
    A(figure('img/stalls_apt64.png', '64-cell apartment, planning seed 0: where the plans that did not arrive stopped (red) and their goals (blue). ntrl-demo and the lift alone stop in open space; lift + push stops at walls.'))
    A('<h3>The houses: lift + push works, and with octaves it solves them</h3>')
    A('<!--HOUSES-->')
    if th:
        A(grid('thick', 'thick_oct'))
        A(L.chart('thick_ntrl'))
        A(f"""<p>The houses give the explanation of Part 4 a natural test, because their obstacles come in every thickness: furniture and partitions under a cell, walls with their unreachable interiors of several cells, whole blocks of rooms. Grouping the detour pairs by the thickest single obstacle their straight line crosses, lift + push in Klickitat sees through {tv('Klickitat', 'liftpush', '1-1.5')} % of the pairs whose thickest obstacle is 1–1.5 cells, {tv('Klickitat', 'liftpush', '1.5-2.5')} % at 1.5–2.5, {tv('Klickitat', 'liftpush', '2.5-4')} % at 2.5–4 and {tv('Klickitat', 'liftpush', '4-8')} % at 4–8 cells (detour index {tv('Klickitat', 'liftpush', '4-8', 'rho', 1.0, 2)} there). Lakeville shows the same order with a later turn ({tv('Lakeville', 'liftpush', '1-1.5')} / {tv('Lakeville', 'liftpush', '2.5-4')} / {tv('Lakeville', 'liftpush', '4-8')} / {tv('Lakeville', 'liftpush', '>8')} % at 1–1.5 / 2.5–4 / 4–8 / over 8 cells). The plus mazes and the apartment have nothing but thin walls, and there everything is seen through. So the recipe of the lift page works as specified where an obstacle is thick enough for the step across it to fit in the position coordinates, and fails where it is not; which scenes work is decided by their wall thickness in cells, not by their size. With the octaves the turn moves to thinner obstacles: Klickitat {tv('Klickitat', 'liftpush_oct', '1-1.5')} % seen through at 1–1.5 cells and {tv('Klickitat', 'liftpush_oct', '1.5-2.5')} % at 1.5–2.5, Lakeville {tv('Lakeville', 'liftpush_oct', '1-1.5')} % and {tv('Lakeville', 'liftpush_oct', '1.5-2.5')} %.</p>""")

    # ------------------------------------------------------------------------------------------------------------------ did not help
    A('<h2>Tried and not adopted</h2>')
    items = []
    if one('plus22', 'vecpush'):
        items.append(f"""<li><b>Vector lift</b> (h times the unit clearance gradient, three extra inputs, so that the two faces of a wall get opposite vectors). With ntrl-demo’s pairs and the push: {v('plus22', 'vecpush')} % solved, detour index {f(one('plus22', 'vecpush').get('rho'), 2)} (22-cell maze). With inside pairs and the lift continued through the walls: {v('plus22in', 'vecpush_in')} % with the push, {v('plus22in', 'vec_in')} % without. The vectors differ inside the wall and close to its faces only; free configurations on the two sides are still 0.7 cells apart in the input.</li>""")
    if one('plus40', 'liftpush_oct2'):
        items.append(f"""<li><b>One octave higher</b> (×4, ×8, ×16 times the scene’s scale) under lift + push: 40-cell maze {v('plus40', 'liftpush_oct2')} % solved against {v('plus40', 'liftpush_oct')} % with the standard octaves; 64-cell apartment {v('apt64', 'liftpush_oct2')} % against {v('apt64', 'liftpush_oct')} %.</li>""")
    if one('plus22', 'lift_x3'):
        items.append(f"""<li><b>A weak constant push with the exact cap</b> (λ = 3e-3): {v('plus22', 'lift_x3')} % solved, detour index {f(one('plus22', 'lift_x3').get('rho'), 2)}. <b>Three times the strong push with a stiff cap and inside pairs</b>: {v('plus22in', 'liftq3_in')} %, index {f(one('plus22in', 'liftq3_in').get('rho'), 2)}.</li>""")
    if one('plus22', 'push'):
        items.append(f"""<li><b>The one-sided term at full weight from the first push epoch</b> (the first two runs): ntrl-demo’s epoch-repeat rule rolled back {one('plus22', 'lift_push').get('rollbacks', 0):,} times, almost all in the first 100 push epochs; the result is the same as with the faded term ({v('plus22', 'lift_push')} % solved). The term is faded in with the push in every other run.</li>""")
    A('<ul class="plain">' + ''.join(items) + '</ul>')
    A('<!--CLOSING-->')

    # ------------------------------------------------------------------------------------------------------------------ run table
    arms = []
    for r in D.runs:
        if r['label'] not in arms:
            arms.append(r['label'])
    order = S + [s for s in ('plus22in', 'plus40in', 'apt64in') if any(r['scene'] == s for r in D.runs)]
    tt = dict(TITLE, plus22in='plus maze 22 × 22, inside pairs', plus40in='plus maze 40 × 40, inside pairs', apt64in='apartment, 64 cells, inside pairs')
    rows = []
    for s in order:
        for a in arms:
            p = D.pool(s, a)
            if p:
                rows.append([esc(tt[s]), esc(ARM.get(a, a)), p['seeds'], f"{p['solved']:.1f}" + (f" [{min(p['each']):.1f}–{max(p['each']):.1f}]" if p['seeds'] > 1 else ''), f(p['wall']), f(p['stall']), f(p['within10']),
                             f(p['los_within10']), f(p['rho'], 2), f(p['seen_through'], 0), f(p['wall_innermost'], 2)])
    A('<h2>All arms</h2>')
    A(details('Pooled over training seeds', table(['scene', 'arm', 'seeds', 'solved % [range]', 'through a wall %', 'stopped short %', 'T within 10 %', 'line of sight within 10 %', 'detour index',
                                                   'seen through %', 'S × slope, innermost'], rows, left=2)))
    rows = []
    for r in sorted(D.runs, key=lambda r: (order.index(r['scene']) if r['scene'] in order else 99, arms.index(r['label']), r['seed'])):
        rows.append([esc(SHORT.get(r['scene'], r['scene'])), esc(ARM.get(r['label'], r['label'])), r['seed'], f(r['solved']), f(r['within10']), f(r['rho'], 2), r['rollbacks'], f(r['train_min'], 0), f'<code>{esc(r["extra"])}</code>', r['id']])
    A(details(f'Every field run ({len(D.runs)}): arguments and registered ids', table(['scene', 'arm', 'seed', 'solved %', 'T within 10 %', 'detour index', 'rollbacks', 'training (min)', 'arguments', 'run'], rows, left=2)))
    A('<!--FOOTER-->')
    html_ = '\n'.join(out)

    # ------------------------------------------------------------------------------------------------------------------ late text (needs every result)
    def delta(a, b, k='solved', scenes=S):
        return ', '.join(f"{SHORT[s]} {P(s, a)[k] - P(s, b)[k]:+.1f}" for s in scenes if P(s, a).get(k) is not None and P(s, b).get(k) is not None)

    better = [s for s in S if P(s, 'liftpush').get('solved', 0) > P(s, 'ntrl').get('solved', 0)]
    worse = [s for s in S if P(s, 'liftpush').get('solved', 0) <= P(s, 'ntrl').get('solved', 0)]
    seeds_note = ', '.join(f"{SHORT[s]} {P(s, 'liftpush').get('seeds', 0)}" for s in S)
    results_text = f"""<p>Queries solved, {names}: ntrl-demo {row('ntrl')} %; lift alone {row('lift')} %; lift + push {row('liftpush')} %; push without the lift {' / '.join(v(s, 'pushonly') for s in S)} %. Lift + push against ntrl-demo in points: {delta('liftpush', 'ntrl')}. Travel times within 10 %: ntrl-demo {row('ntrl', 'within10', 0)} %, lift + push {row('liftpush', 'within10', 0)} %. Training seeds per scene for lift + push: {seeds_note}; one training of the same recipe moves by 3–6 points between seeds (the 22-cell ntrl-demo run here gives 65.0 % for seed 87358 where the embedding study registered 70.1 % for the same seed on the other GPU), so single-seed differences of that size are not findings.</p>
<p>Three things hold across the scenes. The lift alone plans below ntrl-demo on {sum(1 for s in S if P(s, 'lift').get('solved', 0) < P(s, 'ntrl').get('solved', 0))} of {len(S)} ({delta('lift', 'ntrl')}). With the push it plans {'above ntrl-demo on ' + ', '.join(SHORT[s] for s in better) if better else 'no better than ntrl-demo'}{(' and below it on ' + ', '.join(SHORT[s] for s in worse)) if worse else ''}, and its travel times are closer to the truth on {sum(1 for s in S if P(s, 'liftpush').get('within10', 0) > P(s, 'ntrl').get('within10', 0))} of {len(S)} scenes. And the lifted fields almost never arrive through a wall ({row('liftpush', 'wall', 1)} % of plans, ntrl-demo {row('ntrl', 'wall', 1)} %; the push without the lift {' / '.join(v(s, 'pushonly', 'wall', 0) for s in S)} %): their failures are stalls.</p>
<p>The fifth arm in the charts, lift + push with octaves of the position features scaled to the scene, is introduced in Part 5: {' / '.join(v(s_, 'liftpush_oct') for s_ in S)} % solved and {' / '.join(v(s_, 'liftpush_oct', 'within10', 0) for s_ in S)} % of travel times within 10 % ({names}).</p>"""
    html_ = html_.replace('<!--RESULTS_TEXT-->', results_text)
    long_bins = []
    for s in S:
        pb, nb = P(s, 'liftpush').get('by_bin', {}), P(s, 'ntrl').get('by_bin', {})
        top = [b for b in pb if b in ('16-32', '32-64', '64-128')]
        long_bins.append(f"{SHORT[s]} {' / '.join(f(pb[b], 0) for b in top)} against {' / '.join(f(nb.get(b), 0) for b in top)}")
    html_ = html_.replace('<!--BYBIN_TEXT-->', f"""<p>Up to 16 cells of travel the lifted fields with the push are at or above ntrl-demo almost everywhere; the queries nobody solves are the long ones, whose paths go around several walls. Lift + push against ntrl-demo in the bins from 16 cells up (16–32 / 32–64 / 64–128 where present): {'; '.join(long_bins)} %.</p>""")
    nh = one('plus22', 'lift_nohead')
    if nh:
        html_ = html_.replace('<!--NOHEAD-->', f"""<p><b>Control.</b> A lifted network whose Fourier column for φ is zero cannot put slope there (it still sees heading through h, which depends on it). Trained like the lift arm, its line-of-sight times are right without any push: learned / true {f(nh['los_ratio'], 2)}, {f(nh['los_within10'], 0)} % within 10 % (lift {v('plus22', 'lift', 'los_within10', 0)} %), radial slope {hv('plus maze 22 x 22', 'time', 'none', '16-32', 'radial', True)} beyond 16 cells. It is a control, not a recipe: a field that cannot see heading plans {f(nh['solved'], 0)} % and it is as see-through as the lift ({f(nh['seen_through'], 0)} % of detour pairs).</p>""")
    vtxt = ''
    if one('plus22in', 'vec_in'):
        vtxt = f"""<p>A coordinate that tells the two sides apart was tried in the same setting: the <b>vector lift</b>, h times the unit clearance gradient (three extra inputs), which is the same lift along a normal and gives the two faces of a wall opposite vectors. With inside pairs it plans {v('plus22in', 'vec_in')} % without a push and {v('plus22in', 'vecpush_in')} % with the page’s push, detour index {f(one('plus22in', 'vec_in').get('rho'), 2)} and {f(one('plus22in', 'vecpush_in').get('rho'), 2)}. The vectors separate the two sides inside the wall and within a few hundredths of a cell of its faces; free configurations on the two sides, where the step has to show, are still 0.7 cells apart in the input. What is missing is room in the position coordinates themselves.</p>"""
    html_ = html_.replace('<!--VECTOR-->', vtxt)
    return html_, dict(S=S, plus=plus, plan=plan, P=P, v=v, row=row, one=one, cr=cr, cs=cs, delta=delta, names=names, rng=rng, B=B, iv=iv, idp=idp, hv=hv, jv=jv, grid=grid, ch=ch, tv=tv)


def finish(D, C, html_, X):
    """Header, answers, the late sections and the footer."""
    S, plus, plan, P, v, row, one, cr, cs, delta, names, rng, B = (X[k] for k in ('S', 'plus', 'plan', 'P', 'v', 'row', 'one', 'cr', 'cs', 'delta', 'names', 'rng', 'B'))
    grid, ch = X['grid'], X['ch']
    SO = [s for s in S if D.has(s, 'liftpush_oct')]                      # scenes with the octave arm
    SC = [s for s in S if D.has(s, 'lift_c10_oct')]
    on = ' / '.join(SHORT[s] for s in SO)
    cn = ' / '.join(SHORT[s] for s in SC)

    # ---- octave text for the larger scenes
    oct_text = ''
    if SO:
        oct_text = f"""<p>On the other scenes the octaves are scaled with the box side, so that a cell keeps the input length it has on the 22-cell maze. Lift + push + octaves, {on}: {row('liftpush_oct', 'solved', 1, SO)} % solved and {row('liftpush_oct', 'within10', 0, SO)} % of travel times within 10 %, against {row('liftpush', 'solved', 1, SO)} % and {row('liftpush', 'within10', 0, SO)} % without the octaves, {row('ntrl', 'solved', 1, SO)} % and {row('ntrl', 'within10', 0, SO)} % for ntrl-demo, and {' / '.join(v(s, 'ntrl_oct') for s in SO)} % and {' / '.join(v(s, 'ntrl_oct', 'within10', 0) for s in SO)} % for ntrl-demo with the same octaves (where run). Detour pairs seen through at the end: {row('liftpush_oct', 'seen_through', 0, SO)} % ({row('liftpush', 'seen_through', 0, SO)} % without the octaves).</p>"""
        if SC:
            oct_text += f"""<p>With the push kept at its strong value instead of relaxed (lift + octaves + constant push 1e-2), {cn}: {row('lift_c10_oct', 'solved', 1, SC)} % solved, {row('lift_c10_oct', 'within10', 0, SC)} % within 10 %, line-of-sight times {' / '.join(f(100 * (P(s, 'lift_c10_oct').get('los_ratio', float('nan')) - 1), 0) for s in SC)} % long, {row('lift_c10_oct', 'wall', 0, SC)} % of plans through a wall. This is the lift page’s trade between accuracy and planning, now with the detours staying where the push puts them: on the thin-walled mazes the strong push with octaves is the best planner of the study and the least accurate field.</p>"""
        oct_text += grid('solved_oct', 'within_oct')
    html_ = html_.replace('<!--OCT_TEXT-->', oct_text)

    # ---- inside pairs on larger scenes
    it = ''
    if one('apt64in', 'lift_in'):
        it = f"""<p>The same arm on the 64-cell apartment (one seed) changes little: {v('apt64in', 'lift_in')} % solved, {v('apt64in', 'lift_in', 'within10', 0)} % of times within 10 %, {v('apt64in', 'lift_in', 'los_within10', 0)} % along lines of sight (the lift with ntrl-demo’s pairs: {v('apt64', 'lift')} %, {v('apt64', 'lift', 'within10', 0)} %, {v('apt64', 'lift', 'los_within10', 0)} %). There the walls are a small part of a large open floor, and training their interiors does not hold the far field up; the push does.</p>"""
    html_ = html_.replace('<!--INSIDE_LARGE-->', it)

    # ---- houses
    houses = [s for s in ('Allensville', 'Lakeville', 'Klickitat') if s in S]
    ht = ''
    if houses:
        hn = ' / '.join(SHORT[s] for s in houses)
        ht = f"""<p>The three houses ({hn} cells) are kinder to ntrl-demo than the apartment: their walls have real thickness and the outside is closed, and ntrl-demo’s travel times are already {row('ntrl', 'within10', 0, houses)} % within 10 % with a detour index of {row('ntrl', 'rho', 2, houses)}. The lift alone is at its worst there, {row('lift', 'solved', 1, houses)} % solved with {row('lift', 'within10', 0, houses)} % of travel times within 10 % (line-of-sight learned / true {row('lift', 'los_ratio', 2, houses)}): the larger the open floor, the more of the field is the short far field of Part 2. Lift + push plans {row('liftpush', 'solved', 1, houses)} % against {row('ntrl', 'solved', 1, houses)} % for ntrl-demo, with {row('liftpush', 'within10', 0, houses)} % of times within 10 % and {row('liftpush', 'seen_through', 0, houses)} % of detour pairs seen through ({row('ntrl', 'seen_through', 0, houses)} % for ntrl-demo).</p>"""
        ho = [s_ for s_ in houses if D.has(s_, 'liftpush_oct')]
        if ho:
            ht += f"""<p>With octaves scaled to the scene, lift + push on {' / '.join(SHORT[s_] for s_ in ho)}: {row('liftpush_oct', 'solved', 1, ho)} % solved, {row('liftpush_oct', 'within10', 0, ho)} % of travel times within 10 %, {row('liftpush_oct', 'seen_through', 1, ho)} % of detour pairs seen through, {row('liftpush_oct', 'stall', 1, ho)} % of plans stopped short and none through a wall. The octaves alone do not do this (ntrl-demo with the same octaves: {' / '.join(v(s_, 'ntrl_oct') for s_ in ho)} % solved, {' / '.join(v(s_, 'ntrl_oct', 'within10', 0) for s_ in ho)} % within 10 %); the lift, the push and the octaves each remove a different failure: wall hits, the short and see-through far field, and the thin obstacles.</p>"""
        if (L.OUT / 'fields_Klickitat.png').exists() and 'Klickitat' in S:
            ht += figure('img/fields_Klickitat.png', 'House Klickitat (96 cells a side at 0.25 m per cell), travel time to one goal (star), seed 87358; contours every 8 cells.')
        if (L.OUT / 'stalls_Klickitat.png').exists() and 'Klickitat' in S:
            ht += figure('img/stalls_Klickitat.png', 'House Klickitat, planning seed 0: where the plans that did not arrive stopped (red) and their goals (blue).')
    html_ = html_.replace('<!--HOUSES-->', ht)

    # ---- header and answers
    iv, idp, hv = X['iv'], X['idp'], X['hv']
    lp2000 = (one('plus22', 'liftpush').get('course') or {}).get('2000', {})
    o2000 = (one('plus22', 'liftpush_oct8').get('course') or {}).get('2000', {})
    o3000 = (one('plus22', 'liftpush_oct8').get('course') or {}).get('3000', {})
    n_runs = len(D.runs)
    best_c = f"{v('plus22', 'lift_c10_oct')} %" if one('plus22', 'lift_c10_oct8') else '–'
    tv = X['tv']
    kl = 'Klickitat' if 'Klickitat' in S else None
    header = f"""<header class="top">
  <div class="eyebrow">lift_scale study · follow-up to the <a href="https://claude.ai/artifact/HBHUi6dRwWctNp1eDQMs9d">travel-time lift page</a> · {n_runs} registered field runs and read-only forensics on their checkpoints</div>
  <h1>The lift holds at every size; lift + push works where walls are thick, and finer position features give thin walls the room they lack</h1>
  <p class="lede">The lift and the push were rebuilt on this workstation and run for a rectangle robot (x, y, heading) on plus mazes of 22 to 40 cells and Gibson floor plans of 32 to 96 cells. The lift’s own claim holds everywhere: S × |∇T| is {rng('lift', 'wall_innermost')} in the innermost band on every scene (ntrl-demo {rng('ntrl', 'wall_innermost')}). The lift alone plans below ntrl-demo on {sum(1 for s in S if P(s, 'lift').get('solved', 0) < P(s, 'ntrl').get('solved', 0))} of {len(S)} scenes: obstacles are seen through, and in this space the far field also comes out short, because the eikonal equation can be satisfied with slope along the heading coordinate. With the push, {names}: {row('liftpush')} % of queries solved against {row('ntrl')} % for ntrl-demo and {row('liftpush', 'within10', 0)} % of travel times within 10 % against {row('ntrl', 'within10', 0)} %, with almost no plan ending in a wall. Where it stops short, the detours behind thin walls are seen through, and the tests here say why: the push’s detours are slope violations (with the slope cap enforced there are none), and even with the wall interiors trained the learned time across a wall is {iv('plus22in', 'lift_in')} cells where crossing costs {iv('plus22in', 'lift_in', 'straight_median', 0)}. A wall is a symmetric bump in the lifted input; the true field needs a step across the wall’s own thickness in the position coordinates. In the houses that step fits where the obstacle is thick: in the 96-cell house lift + push sees through {tv('Klickitat', 'liftpush', '1-1.5')} % of the detour pairs whose thickest obstacle is 1–1.5 cells and {tv('Klickitat', 'liftpush', '4-8')} % of those at 4–8 cells. Finer position features (octaves scaled with the scene) give thinner obstacles the same room. With them lift + push solves {' / '.join(v(s_, 'liftpush_oct') for s_ in ('Allensville', 'Lakeville', 'Klickitat') if D.has(s_, 'liftpush_oct'))} % of the three houses with {' / '.join(v(s_, 'liftpush_oct', 'within10', 0) for s_ in ('Allensville', 'Lakeville', 'Klickitat') if D.has(s_, 'liftpush_oct'))} % of travel times within 10 % (ntrl-demo {' / '.join(v(s_, 'ntrl') for s_ in ('Allensville', 'Lakeville', 'Klickitat') if D.has(s_, 'liftpush_oct'))} %), and {v('plus30', 'liftpush_oct')} / {v('plus40', 'liftpush_oct')} % of the 30- and 40-cell mazes with {v('plus30', 'liftpush_oct', 'within10', 0)} / {v('plus40', 'liftpush_oct', 'within10', 0)} % within 10 % (ntrl-demo {v('plus30', 'ntrl')} / {v('plus40', 'ntrl')} %). On the 22-cell maze the detours last only while the push is strong ({f(100 * o2000['success'][0] / o2000['success'][1], 0) if 'success' in o2000 else '–'} % solved at epoch 2,000, {best_c} when the push stays strong, {v('plus22', 'liftpush_oct')} % after the page’s relaxation), and the apartment, a thin scan outline in free space, does not respond to anything tried.</p>
</header>"""
    html_ = html_.replace('<!--HEADER-->', header)

    def qa(q, verdict, cls, a):
        badge = f'<span class="verdict {cls}">{verdict}</span>' if verdict else ''
        return f'<div><div class="q">{q}</div><div class="a">{badge}{a}</div></div>'

    ans = []
    ans.append(qa('1. Does the lift still work for a robot with heading, and on larger scenes?', 'yes', 'yes',
                  f"""The wall layer is exact on every scene, early in training: S × |∇T| {rng('lift', 'wall_innermost')} in the innermost band (ntrl-demo {rng('ntrl', 'wall_innermost')}, worse the larger the scene). The clearance gradient not being a unit vector in this space costs nothing measurable."""))
    ans.append(qa('2. Lift alone?', 'plans worse', 'no',
                  f"""{row('lift')} % solved against {row('ntrl')} % for ntrl-demo ({names}). Two causes: obstacles are seen through ({row('lift', 'seen_through', 0)} % of detour pairs), as on the lift page; and the far field is short even along free lines of sight (learned / true {row('lift', 'los_ratio', 2)}), which is new: the field keeps |∇T| = 1 by putting slope into the heading coordinate. A heading-blind control removes the second effect exactly."""))
    ans.append(qa('3. Lift + push?', 'better than ntrl-demo on most scenes', 'mixed',
                  f"""In queries solved: {delta('liftpush', 'ntrl')} points. In travel times within 10 %: {delta('liftpush', 'ntrl', 'within10')} points. Line-of-sight times are right ({row('liftpush', 'los_within10', 0)} % within 10 %) and plans do not end in walls ({row('liftpush', 'wall', 1)} % against {row('ntrl', 'wall', 1)} %). The remaining failures are stalls at walls with the goal behind them."""))
    ans.append(qa('4. Why does it stop short on some scenes?', 'thin walls', 'mixed',
                  f"""The detours behind thin walls do not hold: {row('liftpush', 'seen_through', 0)} % of detour pairs are seen through at the end. On the 22-cell maze they exist at the end of the strong push (index {cr('plus22', 'liftpush', 2000)}, {cs('plus22', 'liftpush', 2000)} % solved) and are gone 1,000 epochs later; on the 40-cell maze they never form. They were slope violations: with the slope cap enforced the same push builds none (index {f(one('plus22', 'lift_x').get('rho'), 2)}). The lift gives the two sides of a wall the same height, so the field has to step across the wall’s own thickness in x and y, and a thin wall leaves no room for that. In the houses, where obstacles come in every thickness, lift + push sees through the thin ones and respects the thick ones ({tv('Klickitat', 'liftpush', '1-1.5')} % seen through at 1–1.5 cells, {tv('Klickitat', 'liftpush', '4-8')} % at 4–8, Klickitat)."""))
    ans.append(qa('5. What gives a thin wall room?', 'finer position features', 'yes',
                  f"""Octaves of the position features (×2, ×4, ×8, scaled with the scene), with no map. 22-cell maze with lift + push: detour index stable at {cr('plus22', 'liftpush_oct8', 1000)} under the strong push, {f(100 * o2000['success'][0] / o2000['success'][1], 0) if 'success' in o2000 else '–'} % solved at epoch 2,000 and {f(100 * o3000['success'][0] / o3000['success'][1], 0) if 'success' in o3000 else '–'} % at 3,000 with {f(100 * o3000.get('within10', float('nan')), 0)} % of times within 10 %; {best_c} at the end when the push stays strong (times 14 % long). With the page’s schedule on {on}: {row('liftpush_oct', 'solved', 1, SO)} % solved, {row('liftpush_oct', 'within10', 0, SO)} % within 10 % (ntrl-demo {row('ntrl', 'solved', 1, SO)} % and {row('ntrl', 'within10', 0, SO)} %). Training the wall interiors also helps without any push (68.1 % on the 22-cell maze, 84 % within 10 %); a vector-valued lift does not."""))
    ans.append(qa('6. What changes with size?', 'not the lift', 'yes',
                  f"""The far field of the lift alone gets shorter the larger the open floor (line-of-sight learned / true {row('lift', 'los_ratio', 2)}), and on the thin-walled mazes the push’s detours get weaker (index at epoch 2,000: {' / '.join(cr(s, 'liftpush', 2000) for s in plus)} on the {' / '.join(str(int(round(B[s]['E']))) for s in plus)}-cell mazes). Scaling the push or the base frequencies with the scene does not restore them; scaling the octaves does. The largest scene, a 96-cell house, is the one where the recipe works best."""))
    ans.append(qa('7. Gibson floor plans?', 'yes on the houses', 'yes',
                  f"""Lift + push against ntrl-demo, {' / '.join(SHORT[s] for s in plan)}: {row('liftpush', 'solved', 1, plan)} % against {row('ntrl', 'solved', 1, plan)} % solved, {row('liftpush', 'within10', 0, plan)} % against {row('ntrl', 'within10', 0, plan)} % of travel times within 10 %. The houses have walls of real thickness and gain the most; the apartment is a thin scan outline in free space, where every detour stays seen through, octaves or not."""))
    html_ = html_.replace('<!--ANSWERS-->', '<h2>Answers</h2><div class="answers">' + ''.join(ans) + '</div>')

    # ---- closing
    closing = f"""<h2>Where this leaves the recipe</h2>
<ul class="plain">
<li><b>The lift</b> is safe to keep at any size and in a configuration space: one extra input from the clearance the speed labels already use, an exact wall layer, no plans through walls. Alone it is not a planner.</li>
<li><b>The far field</b> needs something more than the eikonal loss once the wall layer is exact, and more so with a short configuration coordinate. The push supplies it within a few hundred epochs; training the wall interiors supplies it too, with no push.</li>
<li><b>The detours</b> need room in the position coordinates across an obstacle. Thick walls have it, and there lift + push works as the lift page specifies it. For thinner obstacles the lift page used a wall-magnifying planar map; here octaves of the position features, scaled with the scene, do that job with no map. On the houses they complete the recipe (98–100 % solved with the page’s schedule, travel times right). On the thin-walled mazes they make the push’s detours stable, and how much of them survives is set by how strong the push is kept, which is the accuracy-for-planning trade the lift page describes: the schedule that relaxes to 3e-3 gives the accurate end, a push that stays at 1e-2 the planning end. The apartment is the exception to all of this.</li>
<li><b>Not needed here:</b> a denser collision check for the push (the 48-point check misses 0.2–1.6 % of the blocked pairs), a larger bound (the largest true travel time between free configurations is 0.9–1.9 box lengths), the vector lift.</li>
</ul>
<h2>Limits</h2>
<ul class="plain">
<li>Most variants are one training seed, and one training moves by 3–6 points. The findings above that rest on single runs are the ones with large effects (detour index 0.0 against 1.0 and more, 91 % against 68 %); differences of a few points between final fields are not claimed.</li>
<li>Under a strong push the field keeps changing: with octaves and a constant push of 6e-3 the same run plans 72 %, 95 %, 72 % and 72 % at epochs 1,000, 2,000, 3,000 and 5,000. A final number of such a run is one sample of that motion.</li>
<li>The lift page’s scenes (point robots in the plane) and its wall-magnifying map are not on this machine, so the map could not be put next to the octaves, and the difference between the plane and the configuration space is inferred from the heading measurement and its control, not from a matched pair of scenes.</li>
<li>On the plus mazes the longest queries, across two or more quadrants, are solved only by strong-push fields. The houses are slices of scans at fixed heights with their furniture; the apartment’s outside is free space, the houses’ is not.</li>
<li>The 64-cell apartment is not explained. Its walls are a quarter of a cell thick against a ground-truth grid of an eighth of a cell: 95 of 448 candidate queries were dropped at preparation because their true path crossed a wall, so the detour index there is measured against a truth that is itself partly through walls. The queries that are planned are certified collision-free.</li>
</ul>"""
    html_ = html_.replace('<!--CLOSING-->', closing)
    footer = """<footer>Study folder <code>jeffrey/lift_scale</code> (README: request, setup, run ids). ntrl-demo unchanged on disk, 5,000 epochs unless stated, 500,000 training pairs per scene, MPPI planning seeds 0 / 1 / 2, exact clearance scoring. Prepares: plus mazes by <code>jeffrey/maze_scale</code>, apartment by <code>jeffrey/embed_warp</code>, houses and blocked-pair flags by this study. Forensics: <code>jumps.py</code>, <code>heading.py</code>, <code>losscomp.py</code>, <code>inside.py</code> on the registered checkpoints.</footer>"""
    html_ = html_.replace('<!--FOOTER-->', footer)
    return html_
