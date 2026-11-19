"""Prose of the mobile-arm findings page. Numbers come from report/runs.json (report9.py) and report/posthoc_runs.json (posthoc9.py); the
sentences that interpret them were written after the runs (seed 87358 complete on every house, later seeds added as they finished)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOUSES = ['allensville', 'lakeville', 'klickitat']


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def stats(R):
    """Per house and field: pooled planning, field bounds, forensics, training drives."""
    ph = json.loads((HERE / 'report/posthoc_runs.json').read_text()) if (HERE / 'report/posthoc_runs.json').exists() else {}
    out = {}
    for h in HOUSES:
        paired = set(R.get('paired_seeds', {}).get(h, []))
        for v in ('ntrl', 'ctl', 'mid'):
            S = [s for sd, s in sorted(R['runs'].get(h, {}).get(v, {}).items()) if sd in paired]
            if not S:
                continue
            n = sum(s['solved_n'][1] for s in S)
            P = [ph[s['run']] for s in S if s['run'] in ph]
            x = dict(seeds=len(S), solved=100 * sum(s['solved_n'][0] for s in S) / n, per_seed=[s['solved'] for s in S],
                     through=mean([s['through'] for s in S]), stall=mean([s['stall'] for s in S]),
                     below=100 * mean([s['field']['below_lower_bound'] for s in S]), t_se2=mean([s['field']['t_over_se2_median'] for s in S]),
                     time_ref=mean([s['time_over_ref'] for s in S]),
                     depth=mean([p['contact_depth_median'] for p in P]), shallow=sum(p['shallow_contacts'] for p in P), contacts=sum(p['contacts'] for p in P),
                     arm_first=sum(p['arm_first'] for p in P), base_first=sum(p['base_first'] for p in P), internal=sum(p['internal'] for p in P),
                     tuck_near=mean([p['tucked_share_near_wall'] for p in P]), tuck_open=mean([p['tucked_share_open'] for p in P]),
                     arm_gap=mean([p['arm_first_base_clearance_median'] for p in P]), n_post=len(P))
            ref = [s['refresh'] for s in S if s.get('refresh')]
            if ref:
                x['reach_first'] = mean([r[0][1] for r in ref])
                x['reach_last'] = mean([r[-1][1] for r in ref])
                x['stall_first'] = mean([r[0][2] for r in ref])
                x['stall_last'] = mean([r[-1][2] for r in ref])
                x['refresh_s'] = mean([s['refresh_seconds_mean'] for s in S if s.get('refresh')])
                x['minutes'] = mean([s['minutes'] for s in S])
            else:
                x['minutes'] = mean([s['minutes'] for s in S])
            out[(h, v)] = x
    return out


def text(R):
    H, P = R['houses'], R['pooled']
    X = stats(R)
    name = lambda h: H[h]['name']
    done = [h for h in HOUSES if all((h, v) in X for v in ('ntrl', 'ctl', 'mid'))]
    seeds = {h: X[(h, 'mid')]['seeds'] for h in done}
    gain = {h: X[(h, 'mid')]['solved'] - X[(h, 'ctl')]['solved'] for h in done}
    gain_b = {h: X[(h, 'mid')]['solved'] - X[(h, 'ntrl')]['solved'] for h in done}
    f1 = lambda v: '–' if v is None else f'{v:.1f}'
    f0 = lambda v: '–' if v is None else f'{v:.0f}'
    f2 = lambda v: '–' if v is None else f'{v:.2f}'
    sd = lambda h, v: ', '.join(f'{x:.1f}' for x in X[(h, v)]['per_seed'])
    T = {}
    T['title'] = '2026-11-18 Mobile Arm Midpoint'
    T['h1'] = 'The travel-time midpoint loss on a 9-DOF mobile manipulator'
    if done:
        T['lede'] = ('The Gibson-house rectangle robot now carries a six-joint arm (a UR5e at one fifth of its size): 9 degrees of freedom, planned '
                     'through three scanned houses with an ntrl-demo travel-time field. Adding the recursive travel-time midpoint loss at epoch 2,000 '
                     'changes the share of solved queries by ' + ', '.join(f'<b>{gain[h]:+.1f}</b> points in {name(h)}' for h in done) +
                     ' against the paired control (the same resumed training without the term).')
    else:
        T['lede'] = 'Runs are in progress.'
    complete = len(done) == 3 and all(seeds[h] == 3 for h in done)
    T['status'] = '' if complete else ('<p class="notep">Interim: training seeds finished per house (midpoint runs) ' + ', '.join(f'{name(h)} {seeds[h]}' for h in done) +
                                       '. The queue keeps running the remaining seeds; this page is republished as they land.</p>')
    T['answers'] = '\n'.join(f'<div><div class="q">{q}</div><div class="a">{a}</div></div>' for q, a in answers(R, X, done, gain, gain_b))
    T['robot'] = ('<p>The base is the rectangle every Gibson-house study here uses: 0.5 × 0.2 cells, 12.5 × 5 cm at 0.25 m per cell, the size that '
                  'reaches every room of Allensville, Lakeville and Klickitat. A wheeled base on the floor moves in x, y and heading, so with the arm the '
                  'robot has 9 degrees of freedom. The arm is the repository\'s UR5e (its URDF and its 28 collision spheres) at one fifth of its size, '
                  'mounted on the centre of the base\'s top. It reaches up to 0.79 cells from the mount, so reaching sideways it sticks out more than '
                  'half a cell past the base. Folded straight up, every sphere sits inside the base\'s footprint, so the tucked robot fits wherever '
                  'the rectangle fits.</p><p>Walls are the houses\' floor plans extruded upward, so they stop the arm as well as the base. The arm can '
                  'also hit the floor, the base and itself (cuRobo\'s UR5e self-collision pairs). The speed is ntrl-demo\'s label of the robot\'s '
                  'clearance, with the arm\'s internal distances counted at a 3.5 times smaller margin. The arm\'s own parts sit closer than the '
                  'wall margin in every pose, so without that the robot would be slow everywhere.</p>')
    T['robot_cap'] = ('Plan view (top) and side view (bottom) of three poses; colours mark the six links. Left: the tucked pose used to certify every '
                      'query. Grid in cells (0.25 m).')
    T['solved_intro'] = ('<p>Every field plans the same certified queries with ntrl-demo\'s MPPI, three planning seeds each. A run counts if it arrives '
                         'and no configuration on its path touches a wall, the floor, the base or the arm itself. The midpoint run and the control both '
                         'continue the baseline\'s own epoch-2,000 network, so the three fields of a training seed share their first 2,000 epochs.</p>')
    T['solved_sub'] = 'Bars pool every training seed; dots are single training seeds. The straight segment from start to goal is the trivial control.'
    T['bins_intro'] = ('<p>Queries are binned by the base\'s true travel distance: the bare rectangle\'s optimal time, in cells at full speed, 40 '
                       'queries per bin. Short queries are mostly arm motion; long ones cross the house.</p>')
    T['bins_sub'] = 'Pooled over training seeds and planning seeds 0, 1, 2.'
    T['fails_intro'] = ('<p>The planner appends the goal to every path, so a run that stalls also ends with a jump to the goal. Runs are therefore split by '
                        'arrival: arrived cleanly, arrived through something (the part touched first is recorded), or never arrived.</p>')
    T['fails_sub'] = 'Pooled over training seeds and planning seeds. Hover a segment for counts.'
    T['contacts_intro'] = ('<p>For every run that arrived through something, the first configuration that touches anything is found on the dense '
                           'path, and the part that touches is recorded: the arm against a wall, the base against a wall, or the arm against the floor, '
                           'the base or itself.</p>')
    T['contacts_sub'] = 'Planning seed 0 paths of every training seed, pooled. Hover a segment for counts.'
    T['contacts_after'] = contacts_after(R, X, done)
    T['field_intro'] = ('<p>No 9-D ground truth exists, but the optimum is bracketed. The bare rectangle\'s time between the two base poses, '
                        'T*_SE2, is a lower bound: the 9-D robot\'s clearance never exceeds the rectangle\'s and its metric never charges less. The '
                        'reference path (tuck, drive the SE(2) optimum, untuck) is an upper bound. A field whose T(start, goal) falls below T*_SE2 '
                        'promises a trip that no path can make, which is what a field that does not see a wall does.</p>')
    T['field_sub'] = ('Mean over training seeds of the median per bin. Dashed: the reference path (above the frame on the shortest queries, where '
                      'tucking and untucking the arm dominate). Below the dotted line at 1, the field is shorter than any feasible path.')
    T['field_after'] = field_after(R, X, done)
    T['course_intro'] = ('<p>During training the current field plans the queries with planning seed 0 (diagnostics only). Epochs 1,000 and 2,000 are '
                         'the shared baseline.</p>')
    T['course_sub'] = 'Mean over training seeds. Epoch 5,000 is the final field (planning seed 0 of the evaluation).'
    T['refresh_sub'] = ('Every 100 epochs the current field drives 8,192 random pairs. The loss learns from the drives that exist: they are cut at the '
                        'first contact or stall and split at travel-time midpoints.')
    T['houses_intro'] = ('<p>The houses are the floor plans of the earlier house studies (cuts of the scanned mesh 0.4–1.6 m above the floor, '
                         '0.25 m per cell). Starts are random configurations; goals are the earlier studies\' 48 truth goal poses with random arms.</p>')
    T['setup'] = setup(R)
    T['caveats'] = caveats(R)
    T['footer'] = ('Study folder jeffrey/mobile_arm (README: request, setup, registered run ids; report/tables.md). Parent pages: '
                   '<a href="https://claude.ai/artifact/L7QtsFtB8Kk6nGv3mJ8d4U">Travel-Time Midpoint Loss</a>, '
                   '<a href="https://claude.ai/artifact/DdUvv75Ld972r86nabE8kT">2026-11-17 Lift + Midpoint</a>, '
                   '<a href="https://claude.ai/artifact/3NQ9XSVrVgRCMBjaugePSM">2026-11-17 Lift Scale Test</a>.')
    return T


def answers(R, X, done, gain, gain_b):
    H = R['houses']
    name = lambda h: H[h]['name']
    out = []
    if not done:
        return [('Status', 'Runs in progress.')]
    rows = '; '.join(f'{name(h)}: midpoint <b>{X[(h, "mid")]["solved"]:.1f} %</b>, control {X[(h, "ctl")]["solved"]:.1f} %, baseline {X[(h, "ntrl")]["solved"]:.1f} %'
                     f' ({X[(h, "mid")]["seeds"]} training seed{"s" if X[(h, "mid")]["seeds"] > 1 else ""})' for h in done)
    paired = {}
    for h in done:
        seeds = R.get('paired_seeds', {}).get(h, [])
        paired[h] = [R['runs'][h]['mid'][sd]['solved'] - R['runs'][h]['ctl'][sd]['solved'] for sd in seeds]
    steady = [h for h in done if paired[h] and all(g > 0 for g in paired[h])]
    mixed = [h for h in done if h not in steady]
    verdict = []
    if steady:
        verdict.append('It helps in every training seed in ' + ' and '.join(f'{name(h)} ({", ".join(f"{g:+.1f}" for g in paired[h])} points against the control)' for h in steady)
                       + (', the houses' if len(steady) > 1 else ', the house') + ' where the baseline field sees through walls')
    if mixed:
        verdict.append('in ' + ' and '.join(f'{name(h)}, where the baseline already plans well, it helps in {sum(g > 0 for g in paired[h])} of {len(paired[h])} seeds '
                                             f'({", ".join(f"{g:+.1f}" for g in paired[h])}; {gain[h]:+.1f} pooled)' for h in mixed))
    out.append(('Does the midpoint loss help on the 9-DOF task?', rows + '. ' + ('; '.join(verdict) + '.' if verdict else '')))
    st = '; '.join(f'{name(h)} {X[(h, "ctl")]["stall"]:.1f} → <b>{X[(h, "mid")]["stall"]:.1f} %</b>' for h in done)
    bl = '; '.join(f'{name(h)} {X[(h, "ctl")]["below"]:.0f} → <b>{X[(h, "mid")]["below"]:.0f} %</b>' for h in done)
    out.append(('What does it fix?', f'Stalls: runs that never arrive, control → midpoint, {st}. Many baseline stalls are short queries where only the arm has '
                f'to move (local minima of the field in joint space). And the short far field: queries whose learned T(start, goal) lies below the base\'s '
                f'physical lower bound, {bl}.'))
    deep = [h for h in done if (X[(h, 'mid')]['depth'] or 0) > 0.25]
    P = R['pooled']

    def long_gain(h):
        bins = [b for b in P[h]['mid']['by_bin'] if float(b.split('-')[0]) >= 32]
        return sum(P[h]['mid']['by_bin'][b] - P[h]['ctl']['by_bin'][b] for b in bins) / max(len(bins), 1), bins
    lg = {h: long_gain(h) for h in done}
    long_txt = '; '.join(f'{name(h)} {lg[h][0]:+.0f} points' for h in done if lg[h][1])
    out.append(('What does it leave?', 'Runs that drive through walls. ' + '; '.join(
        f'{name(h)}: {X[(h, "ctl")]["through"]:.1f} → {X[(h, "mid")]["through"]:.1f} % of runs, median penetration {X[(h, "ctl")]["depth"]:.2f} → {X[(h, "mid")]["depth"]:.2f} cells'
        for h in done if X[(h, 'mid')]['depth'] is not None and X[(h, 'ctl')]['depth'] is not None)
        + '. ' + ('In ' + ' and '.join(name(h) for h in deep) + ' they stay deep crossings of walls the field does not see. ' if deep else '')
        + (f'On the longest queries (32 cells and more) the change against the control is {long_txt}. ' if long_txt else '')
        + ('In Allensville the contacts stay shallow clips of corners (seed 87358 clipped one corner of the central block on most of its failed long queries).' if 'allensville' in done else '')))
    lk = [h for h in done if X[(h, 'mid')].get('reach_first') is not None]
    out.append(('Why some walls and not others?', 'The loss learns only from the drives the current field can make. A drive that heads into an '
                'unseen wall is cut at the wall or stalls in front of it, so no training pair ever holds the detour around that wall. Training drives '
                'that reach their goal, first → last refresh: ' + '; '.join(f'{name(h)} {X[(h, "mid")]["reach_first"]:.0f} → {X[(h, "mid")]["reach_last"]:.0f} %' for h in lk)
                + '. Where the drives learn to get around a wall, the field learns it (Klickitat\'s contacts get fewer and shallower); where a third of the drives still stall, deep crossings stay (Lakeville). This is the same limit the lift + midpoint study found in 2-D (on-policy data).'))
    arm = [h for h in done if X[(h, 'ntrl')]['contacts']]
    if arm:
        out.append(('Does the arm matter?', 'Yes. Of the runs that arrive through something, the arm touches first in ' + '; '.join(
            f'{name(h)} {100 * X[(h, "ntrl")]["arm_first"] / max(X[(h, "ntrl")]["contacts"], 1):.0f} %' for h in arm)
            + ' (baseline, planning seed 0), every time with the base itself clear of the wall. The baseline field does not fold the arm near walls: '
            'the share of tucked waypoints within half a cell of a wall minus the share in the open is ' + ', '.join(
                f'{X[(h, "ntrl")]["tuck_near"] - X[(h, "ntrl")]["tuck_open"]:+.2f}' for h in done if X[(h, 'ntrl')]['tuck_near'] is not None)
            + ' (' + ', '.join(name(h) for h in done) + '). With the midpoint loss it is ' + ', '.join(
                f'{X[(h, "mid")]["tuck_near"] - X[(h, "mid")]["tuck_open"]:+.2f}' for h in done if X[(h, 'mid')]['tuck_near'] is not None)
            + ' (control ' + ', '.join(f'{X[(h, "ctl")]["tuck_near"] - X[(h, "ctl")]["tuck_open"]:+.2f}' for h in done if X[(h, 'ctl')]['tuck_near'] is not None)
            + '): the arm learns to fold near walls only where the drives succeed almost always (Allensville).'))
    se2 = R.get('se2', {})
    if se2:
        out.append(('How much harder is it than the base alone?', 'The baseline on the bare SE(2) rectangle (the lift-scale study\'s ntrl-demo runs, same houses '
                    'and bins, their own queries) against the 9-DOF baseline: ' + '; '.join(f'{name(h)} {se2[h]["solved"]:.1f} → {X[(h, "ntrl")]["solved"]:.1f} %' for h in done)
                    + '. The arm costs most in the two large houses, where far more of the baseline\'s runs drive through walls.'))
    by_gpu = {}
    for h in done:
        for sd, s_ in R['runs'][h].get('mid', {}).items():
            if s_.get('refresh_seconds_mean'):
                by_gpu.setdefault(s_['gpu'].replace('NVIDIA GeForce ', ''), []).append((s_['refresh_seconds_mean'], s_['minutes']))
    gtxt = '; '.join(f'{g}: {min(a for a, _ in v):.0f}–{max(a for a, _ in v):.0f} s per refresh, {min(b for _, b in v) / 60:.1f}–{max(b for _, b in v) / 60:.1f} h per run'
                     for g, v in sorted(by_gpu.items()))
    out.append(('What does it cost?', 'A baseline field trains in 7–8 minutes. The midpoint run drives 8,192 pairs every 100 epochs and that dominates its cost ('
                + gtxt + ').'))
    return out


def contacts_after(R, X, done):
    if not done:
        return ''
    H = R['houses']
    parts = []
    for h in done:
        c, m = X[(h, 'ctl')], X[(h, 'mid')]
        if c['contacts'] and m['contacts']:
            parts.append(f'{H[h]["name"]}: control {c["contacts"]} contacts (median depth {c["depth"]:.2f} cells, {c["shallow"]} under 0.1 cells), '
                         f'midpoint {m["contacts"]} ({m["depth"]:.2f} cells, {m["shallow"]} shallow)')
    return ('<p>How deep the first contact goes separates two failures. In the large houses the baseline\'s contacts are deep: the path crosses a '
            'wall the field does not see. In Allensville they are shallow clips of corners. ' + '; '.join(parts) + '.</p>')


def field_after(R, X, done):
    if not done:
        return ''
    H = R['houses']
    lows = []
    for h in done:
        S = [s for sd, s in sorted(R['runs'][h]['ctl'].items()) if sd in set(R.get('paired_seeds', {}).get(h, []))]
        if S:
            bins = list(S[0]['field_by_bin'])
            lows.append(min(sum(s['field_by_bin'][b]['t_over_se2'] for s in S) / len(S) for b in bins))
    low_txt = f' down to {min(lows):.2f}' if lows else ''
    far = []
    for h in done:
        S = [s for sd, s in sorted(R['runs'][h]['ntrl'].items()) if sd in set(R.get('paired_seeds', {}).get(h, []))]
        bins = [b for b in S[0]['field_by_bin'] if float(b.split('-')[0]) >= 32] if S else []
        if bins:
            vals = [100 * sum(s['field_by_bin'][b]['below_lower'] for s in S) / len(S) for b in bins]
            far.append(f'{H[h]["name"]} {min(vals):.0f}–{max(vals):.0f} %' if len(vals) > 1 else f'{H[h]["name"]} {vals[0]:.0f} %')
    return ('<p>The baseline and the control are short in the far field: from 8 cells on, about half or more of their travel times sit below the '
            f'lower bound in every house (baseline, 32 cells and beyond: {", ".join(far)}; median T/T*_SE2{low_txt} in the longest bin). '
            'With the midpoint loss the median stays above the bound in every bin. ' + '; '.join(
                f'{H[h]["name"]}: median T/T*_SE2 control {X[(h, "ctl")]["t_se2"]:.2f}, midpoint {X[(h, "mid")]["t_se2"]:.2f}' for h in done)
            + '. A consistent far field is not a correct one, though. Travel times are pulled to the drives\' own times, and a wall no drive gets '
            'around stays see-through locally even when T(start, goal) is long enough.</p>')


def setup(R):
    H = R['houses']
    q = ', '.join(f'{H[h]["name"]} {H[h]["queries"]} ({H[h]["E"]:g} cells across)' for h in HOUSES if h in H)
    return ('<ul class="tight">'
            '<li><b>Configuration</b>: q = (x, y, φ, ψ₁…ψ₆). x, y span the house box. Heading φ = 0.25 θ / E, as in the house studies. Joint '
            'ψⱼ = ρⱼ qⱼ / E with ρⱼ = the joint\'s largest lever arm (0.76, 0.76, 0.42, 0.10, 0.06, 0.05 cells), so turning a joint costs like moving '
            'the farthest sphere it carries.</li>'
            '<li><b>Training data</b>: ntrl-demo\'s pair recipe in the 9-D box, 500,000 pairs. In Allensville, 34 % of the labels come from the base '
            'against a wall, 23 % from the arm against a wall and 43 % from the arm\'s internal constraints.</li>'
            f'<li><b>Queries</b>: {q}. Each is certified by a collision-free reference path: tuck, drive the SE(2) optimum, untuck.</li>'
            '<li><b>Baseline</b>: ntrl-demo\'s models/metric with dim 9, 5,000 epochs, its own loop (checked identical to the package\'s Model.train() '
            'over the first epochs), training seeds 87358, 1, 2.</li>'
            '<li><b>Midpoint</b>: the baseline\'s epoch-2,000 network and optimizer, then 3,000 epochs with jeffrey/midpoint_dp\'s main setting. Every 100 '
            'epochs, 8,192 safe MPPI drives between random free configurations, cut at the first contact or stall. Each drive is split at '
            'travel-time midpoints down to one cell of travel time, with the exact speed and min leaves; tree and sub-path nodes; '
            '|log T − log y|, weight ramped to 1 over 500 epochs.</li>'
            '<li><b>Control</b>: the same resumed loop without the term. A paired design, because two runs of one seed already differ by up to 12 '
            'points across the two GPUs; on one GPU a run repeats exactly.</li>'
            '<li><b>Planner</b>: ntrl-demo\'s MPPI (50 samples, horizon 5, the house studies\' step scaling) over all nine coordinates, with angles clamped to '
            'the box. Budget is 3 × the reference path\'s length. Planning seeds 0, 1, 2; exact scoring every 0.005 cells per coordinate.</li></ul>')


def caveats(R):
    return ('<ul class="tight">'
            '<li>Walls are prisms from the floor plan: furniture lower than 0.4 m is not in the plan, and the arm cannot reach over anything.</li>'
            '<li>The arm\'s internal distances use a 0.05-cell margin and the walls 0.175 cells. That is a modelling choice: the UR5e\'s own links sit '
            'closer than the wall margin in every pose.</li>'
            '<li>No 9-D optimum is known; times are bracketed between the bare rectangle\'s optimum and the reference path.</li>'
            '<li>The SE(2) comparison uses the lift-scale study\'s own queries (same houses and distance bins), not these.</li>'
            '<li>Runs of one training seed are paired; the three fields of a seed still ran on whichever GPU was free.</li></ul>')
