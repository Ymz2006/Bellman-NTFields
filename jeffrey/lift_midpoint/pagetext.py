"""Prose of the lift + midpoint findings page. Numbers come from report/runs.json (report.py), report/leafcheck.json (leafcheck.py); the sentences that interpret them were written after the runs."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def text(runs):
    S, T = runs['scenes'], runs['title']
    leaf = json.load(open(HERE / 'report/leafcheck.json')) if (HERE / 'report/leafcheck.json').exists() else {}
    prev = {}
    for s_ in runs['order']:
        for a_ in ('lift', 'ntrl', 'liftmid'):
            x_ = S.get(s_, {}).get(a_)
            if x_ and x_.get('pairs_n') and x_.get('detour_n'):
                prev[s_] = x_['detour_n'] / x_['pairs_n']
                break

    def r(scene, arm):
        return S.get(scene, {}).get(arm)

    def v(scene, arm, key, k=1, scale=1.0):
        x = r(scene, arm)
        if not x or x.get(key) is None:
            return '–'
        out = f'{scale * x[key]:.{k}f}'
        return out[1:] if out.startswith('-') and float(out) == 0 else out

    def lf(scene, arm, key, k=1, scale=100.0):
        x = r(scene, arm)
        y = leaf.get(x['run']) if x else None
        return '–' if not y or y.get(key) is None else f'{scale * y[key]:.{k}f}'

    def has(scene, arm):
        return r(scene, arm) is not None

    def gain(scene, a='liftmid', b='lift'):
        return S[scene][a]['solved'] - S[scene][b]['solved']

    done = [s for s in runs['order'] if has(s, 'liftmid') and has(s, 'lift')]
    houses = [s for s in done if s in ('Allensville', 'Lakeville', 'Klickitat')]
    pluses = [s for s in done if s.startswith('plus')]
    mazes = [s for s in done if s.startswith('maze')]
    open_done = [s for s in done if not s.startswith('maze')]
    los_lift = [S[s]['lift']['los_ratio'] for s in open_done if S[s]['lift'].get('los_ratio')]
    los_lm = [S[s]['liftmid']['los_ratio'] for s in open_done if S[s]['liftmid'].get('los_ratio')]
    span = lambda xs, k=2: f'{min(xs):.{k}f}–{max(xs):.{k}f}' if xs else '–'
    lst = lambda xs, fn, sep=' / ': sep.join(fn(s) for s in xs)

    # ---- where the lift alone struggles ---------------------------------------------------------------------------------------------
    struggle = [s for s in runs['order'] if has(s, 'lift') and has(s, 'ntrl') and S[s]['lift']['solved'] < S[s]['ntrl']['solved'] - 2]
    ahead = [s for s in runs['order'] if has(s, 'lift') and has(s, 'ntrl') and S[s]['lift']['solved'] > S[s]['ntrl']['solved'] + 2]

    # ---- controls --------------------------------------------------------------------------------------------------------------------
    ctl = []
    for s in ('plus22', 'Allensville'):
        if has(s, 'drive'):
            ctl.append(f'{T[s]}: drives by ntrl-demo {v(s, "drive", "solved")} % (detour index {v(s, "drive", "rho", 2)}) against '
                       f'{v(s, "liftmid", "solved")} % ({v(s, "liftmid", "rho", 2)}) when the lifted field drives itself')
    mid_ctl = []
    for s in ('plus22', 'Allensville', 'plus40', 'Lakeville', 'plus30'):
        if has(s, 'mid') and has(s, 'liftmid'):
            mid_ctl.append(f'{T[s]} {v(s, "mid", "solved")} % (lift + midpoint {v(s, "liftmid", "solved")} %)')
    for s in mazes:
        if has(s, 'mid'):
            mid_ctl.append(f'{T[s]} {v(s, "mid", "solved")} % (lift + midpoint {v(s, "liftmid", "solved")} %)')

    lede = (f'Added to the travel-time lift without the push, the recursive midpoint loss fixes the lift\'s short far field wherever it shows up, '
            f'and its see-through detours only where the lifted field can already drive around obstacles. On the plus mazes and the houses, travel times along free lines '
            f'of sight go from {span(los_lift)} of the truth to {span(los_lm)}. Detours are learned in '
            f'{T["Allensville"] if has("Allensville", "liftmid") else "the first house"}: {v("Allensville", "liftmid", "solved")} % of the queries solved '
            f'against {v("Allensville", "lift", "solved")} % for the lift alone and {v("Allensville", "ntrl", "solved")} % for ntrl-demo. On the plus '
            f'mazes the field keeps going through walls and gains {lst(pluses, lambda s: f"{gain(s):+.1f}")} points ({lst(pluses, lambda s: T[s].split(" (")[0])}). '
            'The loss learns only from drives that the current field makes. A field that sees through a wall drives into it and stalls, and its '
            'drives that do arrive are almost all straight lines.')

    answers = []
    answers.append(('Where does the lift alone fall behind ntrl-demo?',
                    f'<span class="verdict">not everywhere</span>On the open scenes: {", ".join(T[s].split(" (")[0] for s in struggle)} '
                    f'({lst(struggle, lambda s: v(s, "lift", "solved"))} % against {lst(struggle, lambda s: v(s, "ntrl", "solved"))} %). '
                    f'On the regular mazes it is ahead ({lst(ahead, lambda s: v(s, "lift", "solved"))} % against {lst(ahead, lambda s: v(s, "ntrl", "solved"))} % on '
                    f'{", ".join(T[s] for s in ahead)}), because ntrl-demo arrives through the thin walls there '
                    f'({lst(ahead, lambda s: v(s, "ntrl", "wall"))} % of its runs).'))
    answers.append(('Does the midpoint loss fix the short far field?',
                    f'<span class="verdict">yes</span>Line-of-sight travel times on the plus mazes and the houses, learned over true: {span(los_lift)} with the '
                    f'lift alone, {span(los_lm)} with the midpoint loss. In the configuration space the lifted field puts part of its slope into the heading '
                    'coordinate and comes out short far away; drives along free lines of sight are what the loss sees most, and it corrects them. '
                    f'On the mazes the lift alone is already right along lines of sight ({lst(mazes, lambda s: v(s, "lift", "los_ratio", 2))}).'))
    def moved(group):
        g = [s_ for s_ in group if has(s_, 'liftmid') and has(s_, 'lift')]
        return (f'{lst(g, lambda s_: v(s_, "lift", "rho", 2))} → {lst(g, lambda s_: v(s_, "liftmid", "rho", 2))}, queries solved '
                f'{lst(g, lambda s_: f"{gain(s_):+.1f}")} points')
    hs = [s_ for s_ in ('Allensville', 'Lakeville', 'Klickitat') if has(s_, 'liftmid')]
    ps = [s_ for s_ in ('plus22', 'plus30', 'plus40') if has(s_, 'liftmid')]
    answers.append(('Does it fix the see-through detours?',
                    '<span class="verdict">only where drives find them</span>Detour index (0 = straight through the obstacle, 1 = the true detour), lift '
                    f'alone → lift + midpoint. Houses ({lst(hs, lambda s_: T[s_].split(" (")[0].replace("Gibson ", ""))}): {moved(hs)}; '
                    f'{T["Allensville"].split(" (")[0].replace("Gibson ", "")} reaches ntrl-demo ({v("Allensville", "liftmid", "solved")} % against '
                    f'{v("Allensville", "ntrl", "solved")} %), the larger houses stay below it. Plus mazes (22 / 30 / 40 cells): {moved(ps)}. '
                    + (f'22-cell maze: {moved(["maze22"])}. ' if has('maze22', 'liftmid') else '')
                    + (f'The 13-cell maze is the contrast: the lift alone is not see-through there (detour index {v("maze13", "lift", "rho", 2)}), and '
                       f'lift + midpoint plans {v("maze13", "liftmid", "solved")} %, level with the midpoint alone ({v("maze13", "mid", "solved")} %).'
                       if has('maze13', 'liftmid') and has('maze13', 'mid') else '')))
    answers.append(('Why not on the plus mazes?',
                    '<span class="verdict">no detours in the drives</span>The loss trains on the drives the current field makes. On the 22-cell plus maze '
                    f'the lifted field\'s drives reach their goal {lf("plus22", "lift", "reached", 0)} % of the time, and {lf("plus22", "lift", "detour_share")} % of '
                    f'those are detours (more than 30 % longer than the straight line); ntrl-demo\'s drives: {lf("plus22", "ntrl", "reached", 0)} % and '
                    f'{lf("plus22", "ntrl", "detour_share")} %. In {T["Allensville"]} the lifted field\'s own drives already hold some detours '
                    f'({lf("Allensville", "lift", "detour_share")} % of the arrivals) and the loss grows them: {lf("Allensville", "liftmid", "reached", 0)} % of its '
                    f'drives arrive at the end, {lf("Allensville", "liftmid", "detour_share")} % of them detours.'
                    + (f' The same limit holds without the lift: on the 40-cell plus maze even ntrl-demo\'s drives arrive almost only along straight lines '
                       f'({lf("plus40", "ntrl", "detour_share")} % detours), and the midpoint loss on ntrl-demo\'s network stays at {v("plus40", "mid", "solved")} % '
                       f'(ntrl-demo {v("plus40", "ntrl", "solved")} %).' if has('plus40', 'mid') else '')))
    def det(scene, arm, key):
        d = S.get(scene, {}).get(arm, {}).get('detour_by_length', {}).get(key)
        return None if not d else d

    if has('plus22', 'drive') and has('Allensville', 'drive'):
        p8, a17 = det('plus22', 'drive', '6-12'), det('Allensville', 'drive', '12-24')
        p8s, a17s = det('plus22', 'liftmid', '6-12'), det('Allensville', 'liftmid', '12-24')
        p8m, a17m = det('plus22', 'mid', '6-12'), det('Allensville', 'mid', '12-24')
        answers.append(('And when the drives do hold detours?',
                        '<span class="verdict">in the house, mostly yes; on the plus maze, partly</span>Driver control: the same lift + midpoint training with '
                        'every drive made by the frozen ntrl-demo field of the scene, which arrives with detours '
                        f'({lf("plus22", "ntrl", "detour_share")} % of its arrivals on the 22-cell plus maze, {lf("Allensville", "ntrl", "detour_share")} % in '
                        f'{T["Allensville"].split(" (")[0]}). In {T["Allensville"].split(" (")[0]} the lifted network then learns the long detours almost fully: '
                        + (f'a true detour of {a17["true"]:.1f} cells is learned as {a17["learned"]:.1f} cells, against {a17s["learned"]:.1f} when the lifted field '
                           f'drives itself and {a17m["learned"]:.1f} for ntrl-demo\'s network with the midpoint loss; ' if a17 and a17s and a17m else '') +
                        f'detour index {v("Allensville", "drive", "rho", 2)} against {v("Allensville", "liftmid", "rho", 2)}. On the 22-cell plus maze it learns '
                        f'part of them: detour index {v("plus22", "drive", "rho", 2)} against {v("plus22", "liftmid", "rho", 2)} '
                        f'(ntrl-demo\'s network {v("plus22", "mid", "rho", 2)})'
                        + (f', {p8["true"]:.1f}-cell detours learned as {p8["learned"]:.1f} cells (ntrl-demo\'s network {p8m["learned"]:.1f})' if p8 and p8m else '') +
                        f'. Queries solved with ntrl-demo\'s drives: {v("Allensville", "drive", "solved")} % and {v("plus22", "drive", "solved")} % '
                        f'(self-driven {v("Allensville", "liftmid", "solved")} % and {v("plus22", "liftmid", "solved")} %). So in the house the missing '
                        'piece is the detours in the data; on the plus maze it is also the representation: the lift gives the two faces of a 0.1-cell wall '
                        'the same height, and the step in travel time across the wall has to come from the planar features alone (the lift-scale '
                        'study\'s finding).'))
    mid_rows = [s_ for s_ in runs['order'] if has(s_, 'mid') and has(s_, 'liftmid')]
    if mid_rows:
        ahead_m = [s_ for s_ in mid_rows if S[s_]['mid']['solved'] > S[s_]['liftmid']['solved'] + 5]
        level_m = [s_ for s_ in mid_rows if abs(S[s_]['mid']['solved'] - S[s_]['liftmid']['solved']) <= 5]
        why_level = dict(maze13='the lifted field is not see-through there', plus40='neither field\'s drives hold detours there')
        verdict = 'no' if not [s_ for s_ in mid_rows if S[s_]['liftmid']['solved'] > S[s_]['mid']['solved'] + 5] else 'mixed'
        answers.append(('Is the lift worth adding to the midpoint loss?',
                        f'<span class="verdict">{verdict}</span>Queries solved, midpoint alone against lift + midpoint: ' +
                        '; '.join(f'{T[s_].split(" (")[0]} {v(s_, "mid", "solved")} % against {v(s_, "liftmid", "solved")} %' for s_ in mid_rows) + '. '
                        + (f'Midpoint alone is ahead by more than 5 points on {", ".join(T[s_].split(" (")[0] for s_ in ahead_m)}' if ahead_m else '')
                        + (' and within 5 points on ' + ', '.join(T[s_].split(" (")[0] + (f' ({why_level[s_]})' if s_ in why_level else '') for s_ in level_m) if level_m else '') +
                        '. The lift removes ntrl-demo\'s wall crossings, but the midpoint loss needs a field whose drives go around walls, and '
                        'ntrl-demo\'s smeared walls do that where the lift sees through.'))
    answers.append(('What would make the combination work?',
                    'Both missing pieces are known from the lift-scale study: drives that hold detours from the start (a push, or a driver field that is not '
                    'see-through) and planar resolution at the cell scale, so a thin wall can carry a step (octaves of the position features scaled with '
                    'the scene took the houses to 98–100 % with lift + push). Neither was combined with the midpoint loss here.'))

    answers_html = ''.join(f'<div><div class="q">{q}</div><div class="a">{a}</div></div>' for q, a in answers)

    solved_intro = (f'<p>Every arm on every scene, training seed 87358, three planning seeds. The lift alone is below ntrl-demo on the open scenes and '
                    'ahead of it on the regular mazes; the midpoint loss on top of it ' +
                    ', '.join(f'{T[s].split(" (")[0]} {gain(s):+.1f}' for s in done) + ' points against the lift alone.</p>')

    field_intro = ('<p>The lifted field has two faults the lift-scale study measured: in the robot\'s configuration space it is short far away even where '
                   'nothing is in the way, and it treats obstacles as bumps it can see through. The first chart is the first fault, the second the second.</p>')
    field_after = (f'<p>The midpoint loss narrows the first fault on every open scene ({span(los_lift)} → {span(los_lm)} of the truth): drives along free lines of '
                   'sight are the drives a see-through field completes, and their travel times are what the loss pulls the field towards. The detour index '
                   'moves only where detours are among those drives.</p>')

    why_intro = (f'<p>Every 100 epochs the loss drives 8,192 random pairs with the current field and learns travel times only along those drives. '
                 'A pair whose straight line crosses a wall reaches its goal only if the field leads the drive around the wall. A field that sees '
                 'through the wall leads it into the wall, where the drive stalls; the part before the stall is kept, and it says nothing about '
                 'the way around. How many random pairs need a detour at all differs by scene: ' +
                 ', '.join(f'{T[s].split(" (")[0]} {100 * prev[s]:.0f} %' for s in runs['order'] if s in prev) + '.</p>')
    why_after = (f'<p>Per field, from its own drives (leafcheck.py, 4,096 drives of the final network): on the 22-cell plus maze the lifted field '
                 f'arrives {lf("plus22", "lift", "reached", 0)} % of the time with {lf("plus22", "lift", "detour_share")} % detours, ntrl-demo '
                 f'{lf("plus22", "ntrl", "reached", 0)} % with {lf("plus22", "ntrl", "detour_share")} %; on the 40-cell plus maze both fields\' arrivals are '
                 f'straight ({lf("plus40", "lift", "detour_share")} % and {lf("plus40", "ntrl", "detour_share")} % detours). In {T["Allensville"]} the lifted field '
                 f'arrives {lf("Allensville", "lift", "reached", 0)} % of the time with {lf("Allensville", "lift", "detour_share")} % detours and, after the '
                 f'midpoint loss, {lf("Allensville", "liftmid", "reached", 0)} % with {lf("Allensville", "liftmid", "detour_share")} %. '
                 + (f'On the 22-cell maze, where every corridor wall is seen through (detour index {v("maze22", "lift", "rho", 2)}), only '
                    f'{S["maze22"]["liftmid"]["refresh"][0]["reached"] / S["maze22"]["liftmid"]["refresh"][0]["traversals"] * 100:.0f} % of the first refresh\'s '
                    'drives arrive at all. ' if has('maze22', 'liftmid') and S['maze22']['liftmid'].get('refresh') else '') +
                 f'On the 13-cell maze the '
                 f'lifted field is not see-through at all (detour index {v("maze13", "lift", "rho", 2)}) and {lf("maze13", "lift", "detour_share", 0)} % of its '
                 'arrivals are detours.</p>'
                 '<p>A second, smaller effect is in the leaves. A leaf is worth min(network time, time the drive took); that minimum was built for '
                 'ntrl-demo, whose field is long at the one-cell scale, so the drive\'s time usually wins and anchors the targets. The lifted field in '
                 f'the configuration space is short locally on the 22-cell plus maze: the network\'s time over a leaf is {lf("plus22", "lift", "leaf_net_over_path_median", 2, 1)} '
                 f'of the drive\'s before the midpoint loss and {lf("plus22", "liftmid", "leaf_net_over_path_median", 2, 1)} after it (ntrl-demo '
                 f'{lf("plus22", "ntrl", "leaf_net_over_path_median", 2, 1)}), and the minimum takes the network\'s own value {lf("plus22", "lift", "min_takes_net", 0)} % '
                 f'and {lf("plus22", "liftmid", "min_takes_net", 0)} % of the time. The targets inherit that shortfall, so the field ends at '
                 f'{v("plus22", "liftmid", "los_ratio", 2)} of the truth along lines of sight, consistent but short. In {T["Allensville"]} it went the other way '
                 f'({lf("Allensville", "lift", "leaf_net_over_path_median", 2, 1)} → {lf("Allensville", "liftmid", "leaf_net_over_path_median", 2, 1)}), once the drives '
                 'there held enough detours.</p>')

    fails_intro = ('<p>The lift never arrives through a wall on any scene; its failures are stalls in front of walls with the goal behind them. '
                   'ntrl-demo fails mostly by driving through walls. The midpoint loss on the lift turns stalls into arrivals where it learns detours.</p>')
    course_intro = ('<p>Planning with seed 0 on the prepared queries at the saved epochs (the lift-alone curve is the registered run, identical to the lift + '
                    'midpoint run up to epoch 2,000).</p>')
    bins_intro = '<p>Where the gains are: by the true travel distance of the query.</p>'

    setup = f"""<ul class="tight">
<li><strong>Robot and scenes</strong>: the SE(2) rectangle robot of every local maze study (0.5 × 0.2 cells, configuration (x, y, heading)),
exact clearance, ntrl-demo's speed label (margin 0.175 cells, floor 0.012), in 2-D scenes: maze_scale's 10 × 10, 13 × 13 and 22 × 22 mazes,
plus mazes of 22, 30 and 40 cells with a centred + of 6, 8 and 12 cells, and three Gibson houses as floor plans (Allensville 43 cells,
Lakeville 83, Klickitat 96; 0.25 m per cell).</li>
<li><strong>Lift</strong>: lift_scale's rebuild of the lift page. ntrl-demo's network gets one more input, the travel-time height
h = G(d) / E of the configuration, G(d) = ∫<sub>d</sub><sup>margin</sup> √(1/S² − 1). No push, no octaves.</li>
<li><strong>Midpoint loss</strong>: midpoint_dp's main setting, unchanged. From epoch 2,000, every 100 epochs ntrl-demo's MPPI drives
8,192 random pairs on the current field. Training drives keep 0.02 cells off the walls (rollouts that would enter get +100); each drive
is cut where it collides or stalls. Each drive is split at travel-time midpoints down to one cell; a leaf is worth min(network time,
time the drive took), and every node and as many random sub-paths get the sum of their leaves as target. The term is
|log T − log target| on 8,192 nodes per update, weight ramped 0 → 1 over epochs 2,000–2,500, to epoch 5,000.</li>
<li><strong>Same run up to the fork</strong>: lift + midpoint is the lift-alone run of the same seed until epoch 2,000, call for call
(its loss equals the registered lift-alone run to every digit before the fork, and the 22-cell plus maze plans 129 / 240 at epoch 2,000
in both). The lifted network is evaluated in the batched drives as one matrix product on rows (x, y, heading, h); it matches ntrl-demo's
own forward pass to 4 × 10<sup>−7</sup>.</li>
<li><strong>Arms</strong>: ntrl-demo and lift alone are the registered runs of the earlier studies (new lift-alone runs on the three
mazes); midpoint alone is midpoint_dp's main run on the mazes and this study's <code>--lift none</code> run elsewhere; lift + push is
lift_scale's page schedule, shown for reference. Driver control: the lift + midpoint training with the drives made by the frozen
ntrl-demo field of the scene instead of the network being trained; leaves, targets and everything else unchanged.</li>
<li><strong>Scoring</strong>: the prepared queries of each scene (40 per true-distance bin, certified), ntrl-demo's MPPI with planning
seeds 0, 1, 2, success = arriving without touching a wall (exact clearance); field measures on 24,000 random (configuration, goal) pairs
against the prepared truth; training seed 87358 everywhere.</li>
</ul>"""

    caveats = """<ul class="tight">
<li>One training seed per arm. Between seeds the same recipe moves by 3–6 points on these scenes (lift_scale, midpoint_dp), so differences
below about 5 points are not resolved.</li>
<li>"2-D" was read as 2-D scenes with the robot every local study uses. With a point robot the lift is exact and the heading shortcut
(the reason the lifted field is short far away) does not exist.</li>
<li>The midpoint setting was chosen on ntrl-demo fields (midpoint_dp, 13 × 13 maze) and is used unchanged; nothing was tuned for the lift.</li>
<li>In Allensville the lift + midpoint field was still improving at epoch 5,000 (80.4 % at 4,500, 84.6 % at 5,000 with planning seed 0);
longer training was not run.</li>
<li>Field measures of the maze_scale ntrl-demo and midpoint_dp runs were computed afterwards on their final networks (posthoc.py), on the
same truth pairs as every other arm.</li>
</ul>"""

    planned = dict(liftmid=['maze13', 'maze22', 'plus22', 'plus30', 'plus40', 'Allensville', 'Lakeville', 'Klickitat'], mid=['plus22', 'plus40', 'Allensville'],
                   drive=['plus22', 'Allensville'])
    missing = [f'{T[s_].split(" (")[0]} ({runs["arm_name"][a_].split(" (")[0]})' for a_, ss in planned.items() for s_ in ss if not has(s_, a_)]
    status = ('' if not missing else '<p class="notep"><strong>Interim version.</strong> Still training: ' + ', '.join(missing) +
              '. Every number on the page comes from the runs that have finished; the page will be republished when the rest are in.</p>')
    return dict(
        h1='Does the midpoint loss fix what the lift alone cannot?', lede=lede, answers=answers_html, status=status,
        solved_intro=solved_intro, solved_sub='Bars: the arms run on each scene; ring and dotted line: lift + push from the lift-scale study, for reference.',
        field_intro=field_intro, los_sub='Random pairs whose true path is within 5 % of the straight line; 1 is the truth.',
        rho_sub='(learned − straight) / (true − straight): 0 means the field goes straight through the obstacle, 1 the true detour.', field_after=field_after,
        why_intro=why_intro, reached_sub='A detour: the drive is more than 30 % longer than the straight line between its ends. One point per refresh, epochs 2,001–4,901; each panel has its own scale.',
        why_after=why_after, fails_intro=fails_intro, fails_sub='Planning failures by arrival: a run either reaches the goal through a wall or stops short of it.',
        course_intro=course_intro, course_sub='Lift alone: the registered lift-alone run (epochs 1,000, 2,000, 3,000, 5,000). Lift + midpoint: the same run, forked at epoch 2,000. Midpoint alone and the driver control from epoch 2,000. Dashed: ntrl-demo at epoch 5,000.',
        bins_intro=bins_intro, setup=setup, caveats=caveats)
