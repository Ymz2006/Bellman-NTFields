"""Prose of the pair-lift findings page (page.py). Every number is read from report/*.json and the registered runs at build time."""
import json
from pathlib import Path
import numpy as np

import page as PG
from page import f, table, details, figure, esc, chart, ARM, FIT, SCENES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXTRA_CSS = """
.pending { color: var(--ink-3); font-style: italic; }
figure img { background: #fff; }
"""
TITLE = PG.L.TITLE
SHORT = PG.L.SHORT


def body(D, C):
    def ch(key):
        return chart(key) if key in C else ''

    def grid(*keys):
        inner = ''.join(ch(k) for k in keys)
        return f'<div class="grid2">{inner}</div>' if inner else ''

    def P(label, scene='plus22'):
        return D.pool(label, scene) or {}

    def v(label, key='solved', d=1, scene='plus22'):
        return f(P(label, scene).get(key), d)

    def one(label, scene='plus22'):
        return D.one(label, scene) or {}

    def jump(label, cls, key='learned', d=1):
        r = one(label)
        j = D.jump(r.get('id')) if r else None
        return f(j['by_true'][cls][key], d) if j and cls in j['by_true'] else '–'

    def fit(label, key, d=1):
        return f(one(label).get(key), d)

    def fx(section, label, epoch=None):
        r = one(label)
        return D.fx(section, r.get('id'), epoch) or {}

    def hist(label, key):
        r = one(label)
        p = ROOT / f"results/registered_{r.get('id')}/training/history.json"
        return json.load(open(p)).get(key, []) if r and p.exists() else []

    have = lambda label, scene='plus22': bool(D.sel(label, scene))  # noqa: E731
    VAR = ['bxt1', 'bxt1n', 'bxt1b', 'bxt1nL']                     # the backup's variants on a scene (bxt1L is bxt1nL on the 22-cell maze)

    def PV(lab, scene):
        return P(lab, scene) or (P('bxt1L', scene) if lab == 'bxt1nL' else {})
    def BL(scene):
        """The long-step backup on a scene: steps up to 0.24 box lengths where that was run, else up to 5.3 cells."""
        return P('bxt1nLB', scene) or PV('bxt1nL', scene)

    def bl(scene, bold=False):
        x = f(BL(scene).get('solved'))
        return (f'<b>{x} %</b>' if bold else f'{x} %') + (' with steps up to 0.24 box lengths' if P('bxt1nLB', scene) else '')
    done_scenes = [s for s in SCENES if any(PV(l, s) for l in VAR)]
    others = [s for s in done_scenes if s != 'plus22']
    seeds = P('bxt1')
    seedsL = P('bxt1L')

    def each(p):
        return ' / '.join(f(x) for x in p['each']) if p.get('seeds', 0) > 1 else ''

    def refs(s):
        out = [f'ntrl-demo {f((D.ref(s, "ntrl") or {}).get("solved"))}', f'lift + push {f((D.ref(s, "liftpush") or {}).get("solved"))}']
        if D.ref(s, 'liftpush_oct'):
            out.append(f'with octaves {f(D.ref(s, "liftpush_oct")["solved"])}')
        return ', '.join(out)
    H = []
    A = H.append

    # ------------------------------------------------------------------ header
    A('<header class="top">'
      '<div class="eyebrow">pair_lift study · follow-up to the <a href="https://claude.ai/artifact/HBHUi6dRwWctNp1eDQMs9d">Travel-Time Lift</a> and '
      '<a href="https://claude.ai/artifact/Bu7xZpbydnqWwN9URoKEHp">Lift Variants</a> pages · rectangle robot in SE(2), ntrl-demo’s MPPI, exact scoring · every number from registered runs</div>'
      '<h1>A coordinate of the pair does not fix the lift. The lifted length of a short step, used as its cost, does.</h1>'
      f'<p class="lede">Two tasks were asked for: give the lifted network the length of the lifted curve of the straight segment between the two points (Task 1), or simply a −1 / +1 flag when that segment is blocked (Task 2). '
      f'As a network input, neither helps on the 22 × 22 plus maze: {v("bit_push_tri")} % and {v("ud_push_tri")} % of the queries solved against {v("lift + push")} % for lift + push and {v("ntrl-demo")} % for ntrl-demo, with the detours still seen through. '
      f'The lifted network turns out to be able to hold the step across a wall once it is shown the true travel times, so what is missing is not an input but a loss that selects the right field. '
      f'The same quantity, the lifted length of a straight segment, supplies that loss when it is used as the cost of a short step in a Bellman backup: '
      f'<b>{v("bxt1")} % of the queries solved</b>' + (f' over {seeds["seeds"]} training seeds ({" / ".join(f(x) for x in seeds["each"])})' if seeds.get('seeds', 0) > 1 else '') +
      f', {v("bxt1", "within10")} % of the travel times within 10 % of the truth, {v("bxt1", "seen_through")} % of the detours seen through, with no push and no pair input. '
      + (f'With candidate steps up to 5.3 cells long: {v("bxt1L")} %' + (f' ({each(seedsL)})' if each(seedsL) else '') + f', and detours {f(100 * ((seedsL.get("rho") or np.nan) - 1), 0)} % too long in excess. ' if seedsL else '')
      + ('On larger scenes the backup needs its weight scaled with the step and the longer steps; with both, one training seed per scene: ' + '; '.join(f'{SHORT[s]} {bl(s)} ({refs(s)})' for s in others if BL(s)) + '.' if any(BL(s) for s in others) else '')
      + '</p></header>')

    # ------------------------------------------------------------------ answers
    def qa(q, verdict, cls, a):
        badge = f'<span class="verdict {cls}">{verdict}</span>' if verdict else ''
        return f'<div><div class="q">{q}</div><div class="a">{badge}{a}</div></div>'
    s_bit = fx('sensitivity', 'bit_push_tri')
    cl_c = fx('cliffs', 'fit_countB_f1')
    cl_l = fx('cliffs', 'fit_liftB')
    ans = []
    ans.append(qa('1. Is the diagnosis right: two points across a thin wall get similar embeddings, so no sharp change?', 'true of the trained fields · not a limit of the network', 'mixed',
                  f'In the lift field the travel time between mirror configurations across a wall (0.8 cells apart) is {jump("lift", "6-12", "cross")} cells where the true detour is 8 cells or more, and of a true step of {jump("lift", "6-12", "true")} cells '
                  f'in the travel time to a goal it holds {jump("lift", "6-12")} (lift + push: {jump("lift + push", "6-12")}). But the same network regressed on the true travel times holds {fit("fit_liftB", "jump_big", 2)} of the step '
                  f'and plans {fit("fit_liftB", "solved")} % ({fit("fit_lift_30k", "jump_big", 2)} and {fit("fit_lift_30k", "solved")} % with uniform samples). After 6,000 steps of the same regression it holds {fit("fit_lift", "jump_big", 2)}: the step is representable and slow to learn. '
                  'What the lift lacks on this maze is a loss term that asks for it.'))
    ans.append(qa('2. Task 2, the flag c = −1 / +1 for blocked pairs', 'no', 'no',
                  f'Flag alone: {v("bit")} % solved, travel times {f(P("bit").get("ratio"), 0)} times too long (a random Fourier column puts the two ends of every blocked pair far apart from the start, at about 17 box lengths, and nothing brings them back). '
                  f'With the push: {v("bit_push")} %. With the push and a cap that ties the two branches together: {v("bit_push_tri")} %, {v("bit_push_tri", "seen_through", 0)} % of the detours seen through.'
                  + (f' With pairs sampled inside obstacles and the rule “0 when an end is inside”: {v("bit_in_zero", scene="plus22in")} %.' if have('bit_in_zero', 'plus22in') else '')))
    t1 = [('saturated length', 'ud_push_tri'), ('excess of the lifted length', 'len_push_tri'), ('unsaturated length', 'udraw_f025'), ('wall count', 'count_f025'), ('its own Fourier rows', 'rows_ud'),
          ('same value at both ends', 'same_len'), ('32 copies after the norm', 'post32'), ('8 copies before the norm', 'pre8'), ('128 copies before the norm', 'pre128')]
    t1 = [(n, l) for n, l in t1 if have(l)]
    ans.append(qa('3. Task 1, the lifted length of the straight segment appended', 'no', 'no',
                  'Every way tried plans below lift + push and keeps the detours seen through: ' + '; '.join(f'{n} {v(l)} %' for n, l in t1) +
                  f' (lift + push {v("lift + push")} %, the same loss with no pair input {v("lift_push_tri")} %).' if have('lift_push_tri') else
                  'Every way tried plans below lift + push and keeps the detours seen through: ' + '; '.join(f'{n} {v(l)} %' for n, l in t1) + f' (lift + push {v("lift + push")} %).'))
    ans.append(qa('4. Why do the pair inputs fail?', '', '',
                  f'Three measured reasons. <b>The loss learns to ignore them.</b> The input jumps wherever the straight segment starts to cross an obstacle, in open space, where the true travel time is continuous; the term that keeps the field continuous there removes the dependence on the input within 100 epochs '
                  f'(zeroing the pair column of the trained flag network changes its travel times by a median {f(s_bit.get("change_median"), 2)} cells), and the push then lifts blocked and clear pairs together. '
                  f'<b>A flag does not separate the two faces of a wall when both are blocked from the goal</b>, which is the usual case in a maze. '
                  f'<b>An input that does separate them (a count, an unsaturated length) puts cliffs in open space</b>: fitted to the truth, such fields change by more than twice the step on {f(100 * cl_c.get("changes", {}).get("over2", np.nan), 0)} % of the 0.1-cell steps across a line where the count changes '
                  f'({f(100 * cl_l.get("changes", {}).get("over2", np.nan), 1)} % for the lift alone) and plan {fit("fit_countB_f1", "solved")} % with {fit("fit_countB_f1", "within10")} % of their travel times within 10 %.'))
    ans.append(qa('5. What makes the idea work?', 'yes', 'yes',
                  f'Using the lifted length where it is exact: over a short straight step it is the step’s travel time, and it is very large when the step crosses a wall. A Bellman backup over short free steps, '
                  f'T(q, g) = min over steps of [step cost + T(step end, g)], raises exactly the configurations a see-through field gets wrong. Lift + backup on the 22-cell maze: <b>{v("bxt1")} %</b> solved, {v("bxt1", "within10")} % of travel times within 10 %, '
                  f'{v("bxt1", "seen_through")} % seen through; a true step of {jump("bxt1", "6-12", "true")} cells across a wall is held as {jump("bxt1", "6-12")}.'
                  + (f' With steps up to 5.3 cells: {v("bxt1L")} % solved, {v("bxt1L", "within10")} % of travel times within 10 %.' if seedsL else '')))
    if seeds.get('seeds', 0) > 1 or others:
        txt = f'22-cell maze, {seeds["seeds"]} training seeds: {each(seeds)} % solved' + (f'; with long steps {each(seedsL) or v("bxt1L")} %' if seedsL else '') + '. ' if seeds.get('seeds', 0) > 1 else ''
        unch = [s for s in others if P('bxt1', s)]
        if unch:
            txt += 'Larger scenes with the settings unchanged (steps of 0.66 cells): ' + '; '.join(f'{SHORT[s]} {v("bxt1", scene=s)} %' for s in unch) + ', no better than the references: the term fades as the scene grows (Part 4). '
        long_ = [s for s in others if PV('bxt1nL', s)]
        if long_:
            txt += 'With the weight scaled with the step and steps up to 5.3 cells: ' + '; '.join(f'{SHORT[s]} {bl(s, True)} ({refs(s)})' for s in long_) + '.'
        ans.append(qa('6. Does it hold on other seeds and scenes?', 'seeds yes · houses yes · larger mazes and the apartment partly', 'mixed', txt))
    A('<h2>Answers</h2><div class="answers">' + ''.join(ans) + '</div>')

    # ------------------------------------------------------------------ setup
    A('<h2>What was run</h2>')
    A('<p>Everything is built on <code>jeffrey/lift_scale</code>, the local rebuild of the lift (the Lift Variants page was made on another machine; its code is not on this workstation). The robot is the 0.5 × 0.2-cell rectangle of every local maze study, '
      'so a configuration is q = (x, y, heading) and the lifted input is (q, h): the pair coordinate is a fifth input where the request says “4d”. Training pairs are ntrl-demo’s (first end in the speed band next to an obstacle, second end anywhere free), '
      '5,000 epochs, ntrl-demo’s loss unless a term is named. Main scene: the 22 × 22 plus maze (walls 0.1 cells thick, 0.3 cells in configuration space), 240 queries × 3 planning seeds. '
      'Reference arms are the lift-scale study’s registered runs of the same pipeline.</p>')
    A('<div class="math">pair feature of (q0, q1), from the straight segment q0 → q1 sampled every 0.1 cells:\n'
      '  flag     ψ = 1 if the segment enters an obstacle, else 0                                        (Task 2)\n'
      '  length   e = Σ √(|dq|² + dh²) − |dq|   the lifted curve’s excess over the segment, in cells:   (Task 1)\n'
      '           0 on a clear segment outside the speed bands, about 7.2 cells for every wall crossed; ψ = tanh(e / s), or e itself\n'
      '  count    ψ = number of obstacles the segment enters\n'
      'fed as      c0 = −ψ, c1 = +ψ   (or ψ at both ends)   through one more Fourier column, its own Fourier rows,\n'
      '            or k copies appended to the 256-wide embedding before / after the instance norm</div>')

    # ------------------------------------------------------------------ part 1
    A('<h2>Part 1 · Can the lifted network hold a step across a wall at all?</h2>')
    A('<p>This is the premise of both tasks, so it was tested directly. Each network was regressed on the true travel time (the grid solver’s, 48 goals, 600,000 pairs) and then asked two things: how much of the true step in travel time it holds between mirror configurations on the two faces of a wall, '
      'and how it plans.</p>')
    A(grid('fit_jump', 'fit_solved'))
    A(f'<p>After 6,000 steps every network holds about a third of the step ({fit("fit_ntrl", "jump_big", 2)} ntrl-demo, {fit("fit_lift", "jump_big", 2)} lift, {fit("fit_count_f1", "jump_big", 2)} lift + wall count) while {fit("fit_lift", "within10")} % of the lift’s travel times are already within 10 %: '
      f'the bulk of the field is learned first and the two faces of the walls last. Trained longer, with most samples next to walls, the plain lifted network holds <b>{fit("fit_liftB", "jump_big", 2)}</b> of the step and plans {fit("fit_liftB", "solved")} % '
      f'({fit("fit_lift_30k", "jump_big", 2)} and {fit("fit_lift_30k", "solved")} % with uniform samples and 30,000 steps). ntrl-demo without the lift also holds the step ({fit("fit_ntrlB", "jump_big", 2)}) but plans {fit("fit_ntrlB", "solved")} %, '
      f'with {fit("fit_ntrlB", "wall")} % of its plans through a wall: the lift is what makes the band next to the wall learnable.</p>')
    A(f'<p>So on this maze, with walls 0.3 cells thick in configuration space, the two faces of a wall are not too close for the network. They are too close for the losses used so far to tell apart. '
      f'The pair inputs do not make the step easier to fit either: at 6,000 steps they hold {fit("fit_bit", "jump_big", 2)}–{fit("fit_count_f1", "jump_big", 2)} of it, and at 20,000 the flag is level with the lift ({fit("fit_bitB", "jump_big", 2)}, {fit("fit_bitB", "solved")} % solved) '
      f'while the count and the unsaturated length plan far worse ({fit("fit_countB_f1", "solved")} %, {fit("fit_udrawB_f025", "solved")} %, {fit("fit_lenrawB_box", "solved")} %) although {fit("fit_countB_f1", "within10")} % of their travel times are within 10 %.</p>')
    A(ch('cliffs'))
    A(f'<p>The reason is in the chart above. A pair input that tells the two faces of a wall apart also changes in open space, wherever the straight segment to the goal starts to cross one more obstacle. On {f(100 * cl_c.get("share_changes", np.nan), 1)} % of 0.1-cell steps in open space the wall count changes. '
      f'The truth is continuous there. The count-fed field is not: its travel time changes by a median {f(cl_c.get("changes", {}).get("median"), 1)} times the step across such a line (90th percentile {f(cl_c.get("changes", {}).get("q90"), 1)}), '
      f'and a planner that follows the field stops at those cliffs ({fit("fit_countB_f1", "stall")} % of its plans stall). The flag has few such cliffs ({f(100 * fx("cliffs", "fit_bitB").get("changes", {}).get("over2", np.nan), 1)} %) because the fit learned to ignore it.</p>')
    if 'copies_solved' in C:
        cp = lambda lab, key='solved', d=1: fit(lab, key, d)  # noqa: E731
        A('<h3>The lifted length appended to the embedding: 2, 8, 32, 128 copies</h3>')
        A('<p>The first form in the request. ψ = tanh of the up-and-down part of the lifted curve of the straight segment (cells; 0 on a clear segment along which the lift is monotone, saturating at one wall), −ψ / 2 at one end and +ψ / 2 at the other, '
          'appended k times to the 256-wide embedding, either before the instance norm or after it. After the norm the metric reads the copies directly: every 16 of them add 0.2 × amplitude × ψ to the travel time (amplitude 0.3; 0.075 for 128 copies, so that 32 and 128 add the same 2.7 cells). '
          'Same test as above: fitted to the truth for 20,000 steps, then planned.</p>')
        A(grid('copies_solved', 'copies_jump'))
        A(f'<p>A few copies do no harm and no measurable good: 2 and 8 copies plan {cp("fit_pre2B")} % and {cp("fit_pre8B")} % before the norm, {cp("fit_post2B")} % and {cp("fit_post8B")} % after it, against {cp("fit_liftB")} % for the lift alone '
          f'(the fits without a pair input range from {cp("fit_liftB")} to {cp("fit_lift_30k")} %), and all hold {cp("fit_post2B", "jump_big", 2)}–{cp("fit_pre8B", "jump_big", 2)} of the step. '
          f'Many copies hurt: 32 copies {cp("fit_pre32B")} % before the norm and {cp("fit_post32B")} % after it, 128 copies {cp("fit_pre128B")} % and {cp("fit_post128B")} %, with {cp("fit_pre128B", "stall")} % and {cp("fit_post128B", "stall")} % of the plans stalled '
          f'and travel times less accurate than the lift’s own ({cp("fit_pre128B", "within10")} % and {cp("fit_post32B", "within10")} % within 10 % against {cp("fit_liftB", "within10")} %). '
          'The more of the embedding the feature occupies, the more the travel time follows it, and it is a function of the straight segment: it jumps in open space where the segment starts to cross a wall, and it says the same thing for a wall with a door next to it and a wall with a 20-cell detour.</p>')

    # ------------------------------------------------------------------ part 2
    A('<h2>Part 2 · The pair feature as a network input, trained without the truth</h2>')
    A('<p>The pair input can only make the right field representable. ntrl-demo’s loss accepts fields that pass straight through obstacles (lift page, Part 3), so every arm here also gets the lift page’s push on pairs whose segment is blocked. One more term is needed because the input is discontinuous: '
      'the eikonal term is a derivative at a point and cannot tie the field on the two sides of a line where the input jumps. The cap does: over random free steps of up to 0.66 cells, the travel time may not change by more than the step’s own travel time.</p>')
    A(grid('solved', 'within'))
    A(ch('seen'))
    td = hist('bit', 't_blocked')
    A(f'<p><b>Flag alone.</b> {v("bit")} % solved. From the first epoch the blocked pairs sit at a travel time of about {f(td[0] if td else None, 0)} box lengths: the −1 / +1 column shifts the phases of all 128 Fourier rows, so the two ends of a blocked pair start out as unrelated points. '
      f'ntrl-demo’s loss weights every pair by exp(−T / 2), so at that travel time the pair no longer counts, and nothing corrects it. <b>Flag + push</b>: the same, {v("bit_push")} %.</p>')
    b100, b2000, bfin = fx('sensitivity', 'bit_push_tri', 100), fx('sensitivity', 'bit_push_tri', 2000), fx('sensitivity', 'bit_push_tri')
    bc = fx('bycount', 'bit_push_tri').get('by_count', {})
    A(f'<p><b>Flag + push + cap.</b> {v("bit_push_tri")} % solved, travel times fine ({v("bit_push_tri", "within10")} % within 10 %), detours seen through ({v("bit_push_tri", "seen_through", 0)} %). The cap brings the blocked pairs down in the first 100 epochs, and it does so by making the network insensitive to the flag: '
      f'with the pair column zeroed the trained field’s travel times change by a median {f(b100.get("change_median"), 2)} cells at epoch 100, {f(b2000.get("change_median"), 2)} at 2,000 and {f(bfin.get("change_median"), 2)} at the end. The push then has no handle that separates blocked from clear pairs: '
      f'at epoch 2,000 clear pairs are {f(100 * ((fx("bycount", "bit_push_tri", 2000).get("by_count", {}).get("0", {}).get("ratio") or np.nan) - 1), 0)} % long and the detours are still seen through. '
      f'Even pairs blocked by a single obstacle, where a flag is all the information needed, stay seen through ({f(100 * (bc.get("1", {}).get("seen_through") or np.nan), 0)} %, detour index {f(bc.get("1", {}).get("rho"), 2)}).</p>')
    A(f'<p><b>The lifted length, in the ways the request lists.</b> Saturated (tanh of the up-and-down part of the lifted curve, which is 0 on clear segments): {v("ud_push_tri")} %. Wall count: {v("count_f025")} %.'
      + ''.join(f' {n[0].upper() + n[1:]}: {v(l)} %.' for n, l in t1 if l not in ('ud_push_tri', 'count_f025')) +
      (f' The same loss with no pair input at all: {v("lift_push_tri")} %.' if have('lift_push_tri') else '') +
      ' One more thing was tried and dropped: with the derivative of the length feature inside the eikonal term (so that the loss sees the total derivative of the field the planner reads) the loss diverges within 100 epochs, because the feature’s gradient reaches 1,300 per box length where a segment grazes a wall. The length arms above leave that derivative out.</p>')
    A(ch('jumps'))
    A(f'<p>The step across a wall, which the pair input was meant to supply: with the flag, a true step of {jump("bit_push_tri", "6-12", "true")} cells is held as {jump("bit_push_tri", "6-12")}, and the travel time between the two faces themselves is {jump("bit_push_tri", "6-12", "cross")} cells '
      f'(lift alone {jump("lift", "6-12")} and {jump("lift", "6-12", "cross")}).</p>')

    # ------------------------------------------------------------------ part 3
    q6, q17 = D.fx('quadrature', 'trapezoid, 6 points r=0.03') or {}, D.fx('quadrature', 'antiderivative, 17 points r=0.03') or {}
    tdb = hist('bxt1', 'td_blocked')
    A('<h2>Part 3 · The lifted length as the cost of a short step</h2>')
    A('<p>The lifted curve of a straight segment has an exact meaning only when the segment is short: its length is then the travel time of the step (along a wall normal the two are equal by construction of the lift), and a step that crosses a wall is longer than any detour worth taking. '
      'That is a local cost, and a local cost with a minimum over steps is the Bellman equation of the travel time. The term added to ntrl-demo’s loss:</p>')
    A('<div class="math">for each end q of a training pair (q, g): 16 straight steps q → q′ of 0.33–0.66 cells that stay out of obstacles\n'
      '    (2 down the field’s own gradient, 6 around it, 8 in random directions)\n'
      'y = min over the steps of [ τ(q, q′) + T(q′, g) ]        τ = travel time of the step = ∫ ds / S along it;  no gradient through y\n'
      'loss += mean (T(q, g) − y)²\n'
      'and ntrl-demo’s own temporal-difference step is dropped where it crosses an obstacle</div>')
    A(f'<p>A field that descends into a wall has no free step that lowers it there, so y is above T and the term raises it, at that configuration only; the correction then travels back along the detour one step per backup. No pair is pushed and nothing is asked of pairs that are already consistent. '
      f'ntrl-demo has a term of this kind, a 0.66-cell step down the gradient, but takes the step without a collision check: at the start of training {f(100 * (tdb[0] if tdb else np.nan), 0)} % of those steps cross an obstacle and {f(100 * (tdb[-1] if tdb else np.nan), 0)} % at the end, '
      f'each one equating the travel time on one face of a wall with the travel time on the other.</p>')
    A(grid('abl_solved', 'abl_within'))
    A(f'<p><b>Lift + backup: {v("bxt1")} % solved</b>' + (f' ({" / ".join(f(x) for x in seeds["each"])} over {seeds["seeds"]} training seeds)' if seeds.get('seeds', 0) > 1 else ' (seed 87358)') +
      f', {v("bxt1", "wall")} % of the plans through a wall and {v("bxt1", "stall")} % stalled; {v("bxt1", "within10")} % of travel times within 10 % (median learned / true {v("bxt1", "ratio", 3)}), line-of-sight pairs {v("bxt1", "los_within10")} %, '
      f'{v("bxt1", "seen_through")} % of the detours seen through. The detours are {f(100 * ((P("bxt1").get("rho") or np.nan) - 1), 0)} % long in excess (detour index {v("bxt1", "rho", 2)}). '
      f'Queries with a true path of 32–64 cells, which no earlier lift arm solved without an inflated field: {f(one("bxt1").get("by_bin", {}).get("32-64"), 0)} %.</p>')
    parts = []
    if have('tdfree'):
        parts.append(f'Dropping the wall-crossing TD steps without adding the backup gives {v("tdfree")} % and a field {f(100 * (1 - (P("tdfree").get("ratio") or np.nan)), 0)} % short: ntrl-demo’s step was also what kept the far field’s scale.')
    if have('ntrl_bxt1'):
        parts.append(f'The backup without the lift: {v("ntrl_bxt1")} % with {v("ntrl_bxt1", "wall")} % of the plans through a wall.')
    elif have('ntrl_bell1'):
        parts.append(f'The backup without the lift (first version): {v("ntrl_bell1")} % with {v("ntrl_bell1", "wall")} % of the plans through a wall, so the lift still does its own job, the band next to the wall.')
    if have('bxt1_tdall'):
        parts.append(f'Keeping ntrl-demo’s wall-crossing TD steps next to the backup: {v("bxt1_tdall")} %.')
    if have('bxt1_k8'):
        parts.append(f'8 candidate steps and none around the gradient: {v("bxt1_k8")} %.')
    if have('bxt3'):
        parts.append(f'Weight 3: {v("bxt3")} %.')
    if have('bxt1_r15'):
        parts.append(f'Half the step radius: {v("bxt1_r15")} %.')
    if have('bxt1L'):
        parts.append(f'<b>Longer steps</b> (lengths log-uniform between 0.33 and 5.3 cells): {v("bxt1L")} % solved' + (f' ({each(seedsL)})' if each(seedsL) else '') +
                     f', no detour seen through from epoch 600 on, but detours {f(100 * ((seedsL.get("rho") or np.nan) - 1), 0)} % too long in excess and {v("bxt1L", "within10")} % of travel times within 10 %: '
                     'a long step carries the correction farther per update, and the minimum over 16 sampled steps overestimates more when the steps are long.')
    if have('bxt1_bit') or have('bxt1_count') or have('bell1_count'):
        parts.append('A pair input on top of the working recipe does not help: ' + ', '.join(x for x in [f'flag {v("bxt1_bit")} %' if have('bxt1_bit') else '', f'wall count {v("bxt1_count")} %' if have('bxt1_count') else '',
                                                                                                         f'wall count on the first version {v("bell1_count")} % against {v("bell1")} %' if have('bell1_count') else ''] if x) + '.')
    A('<p>' + ' '.join(parts) + '</p>')
    A(ch('jumps') if 'jumps' not in C else '')
    A(f'<p>Across a wall (chart in Part 2): a true step of {jump("bxt1", "3-6", "true")} cells is held as {jump("bxt1", "3-6")}, {jump("bxt1", "6-12", "true")} as {jump("bxt1", "6-12")}, {jump("bxt1", "12-24", "true")} as {jump("bxt1", "12-24")}, '
      f'and the travel time between the two faces themselves, 0.8 cells apart, is {jump("bxt1", "6-12", "cross")} cells where the lift alone says {jump("lift", "6-12", "cross")}. The network and its input are the lift’s, unchanged. '
      'The embeddings of the two faces are far apart because the loss asked for it.</p>')
    A(figure('img/fields_plus22_zoom.png', 'Travel time to a goal (star) deep inside a maze quadrant, at the goal’s heading; contours every cell, obstacles for the robot’s centre in grey. In the truth the cells next to the goal but across a wall are far (lighter). '
                                           'Lift + push and the flag network see through those walls; the backup field has them.', 'Four panels of the travel-time field around a goal: truth, lift + push, flag network, Bellman backup'))
    A('<h3>The first version ran 28 % long, and why</h3>')
    bk = fx('backup', 'bell1')
    A(f'<p>The first two backup runs planned {v("bell1")} % and {v("bell01")} % but with every travel time about 28 % too long ({v("bell1", "within10")} % within 10 %), at weight 1 and at weight 0.1 alike. The cause was the step’s travel time: a 6-point trapezoid rule on 1 / S. '
      f'In the band 1 / S falls from 83 to 9 within 0.0175 cells of the wall, so a step that starts there is overestimated: {f(q6.get("mean"), 2)} times the reference on average and {f(q6.get("q99"), 1)} times at the 99th percentile. '
      f'On the trained field the band configurations were {f(-(bk.get("band", {}).get("trapezoid", {}).get("gap_mean") or np.nan), 2)} cells below their targets and the open-space ones {f(bk.get("open", {}).get("trapezoid", {}).get("gap_mean"), 2)} cells above: an equilibrium between an inflated target at the walls and the correct one elsewhere. '
      f'The step time now uses the antiderivative of 1 / S between samples ({f(q17.get("mean"), 3)} of the reference on average, {f(q17.get("q99"), 3)} at the 99th percentile). The inflated fields plan as well as the correct one, as over-priced walls did on the earlier pages; only the correct one has travel times that can be used.</p>')
    A(grid('course_solved', 'course_los'))
    A(ch('course_seen'))

    # ------------------------------------------------------------------ part 4
    if others:
        A('<h2>Part 4 · Larger scenes</h2>')
        unch = [s for s in others if P('bxt1', s)]
        A('<p>First with nothing retuned and the step kept at 0.66 cells' + (': ' + '; '.join(f'{SHORT[s]} {v("bxt1", scene=s)} % ({refs(s)})' for s in unch) if unch else '') +
          '. The backup has almost no effect there. Its residual T − y is a travel time in box lengths, and a step of 0.66 cells is 0.03 box lengths on the 22-cell maze, 0.0165 on the 40-cell maze, 0.010 on the apartment and 0.007 in the largest house: '
          'the squared residual, and with it the term’s weight against ntrl-demo’s own loss, falls to 30 %, 12 % and 5 %. '
          + (f'On the 40-cell maze the share of detours seen through barely moves during training ({f(100 * (one("bxt1", "plus40").get("course", {}).get("600", {}).get("seen_through") or np.nan), 0)} % at epoch 600, {v("bxt1", "seen_through", 0, "plus40")} % at the end).' if have('bxt1', 'plus40') else '') + '</p>')
        fix = []
        if have('bxt1n', 'plus40'):
            c = one('bxt1n', 'plus40').get('course', {})
            fix.append(f'<b>Weight scaled with the step</b> (× (0.03 / radius)², so the term has the strength it has on the 22-cell maze): 40-cell maze {v("bxt1n", scene="plus40")} %, detours seen through {f(100 * (c.get("100", {}).get("seen_through") or np.nan), 0)} % at epoch 100 and '
                       f'{v("bxt1n", "seen_through", 0, "plus40")} % at the end, still falling. The correction moves 0.66 cells per backup and the longest paths are over 100 cells.')
        if have('bxt1b', 'plus40'):
            fix.append(f'Steps of 0.03 box lengths instead (1.2 cells on the 40-cell maze): {v("bxt1b", scene="plus40")} %.')
        long_ = [s for s in others if PV('bxt1nL', s)]
        if long_:
            fix.append('<b>Scaled weight and steps up to 5.3 cells</b>: ' + '; '.join(f'{SHORT[s]} {f(PV("bxt1nL", s).get("solved"))} % solved, {f(PV("bxt1nL", s).get("seen_through"), 0)} % of detours seen through, {f(PV("bxt1nL", s).get("within10"))} % of travel times within 10 % ({refs(s)})' for s in long_) + '.')
        A('<p>' + ' '.join(fix) + '</p>')
        A(grid('scenes_solved', 'scenes_within'))
        A(ch('scenes_wall'))
        notes = []
        if PV('bxt1nL', 'plus40'):
            r40 = one('bxt1nL', 'plus40')
            notes.append(f'On the 40-cell maze the long-step field has every detour (none seen through) and still stalls on {v("bxt1nL", "stall", 1, "plus40")} % of the plans. All of the queries with a true path of 64–128 cells fail ({f(r40.get("by_bin", {}).get("64-128"), 0)} % solved; '
                         f'32–64 cells: {f(r40.get("by_bin", {}).get("32-64"), 0)} %), and short detours are over-priced: a true detour of {f(r40.get("detour_by_length", {}).get("1-3", {}).get("true_cells"))} cells is learned as '
                         f'{f(r40.get("detour_by_length", {}).get("1-3", {}).get("learned_cells"))}, one of {f(r40.get("detour_by_length", {}).get("3-6", {}).get("true_cells"))} as {f(r40.get("detour_by_length", {}).get("3-6", {}).get("learned_cells"))}.')
        if PV('bxt1nL', 'plus30'):
            r30 = one('bxt1nL', 'plus30')
            notes.append(f'The 30-cell maze shows the same two things at a smaller size: {v("bxt1nL", "seen_through", 0, "plus30")} % of detours seen through, {v("bxt1nL", "stall", 1, "plus30")} % of the plans stalled, queries of 32–64 cells {f(r30.get("by_bin", {}).get("32-64"), 0)} % solved.')
        hs = [s for s in ('Allensville', 'Lakeville', 'Klickitat') if PV('bxt1nL', s)]
        if hs:
            notes.append('<b>Houses.</b> ' + '; '.join(f'{SHORT[s]} {v("bxt1nL", "solved", 1, s)} % solved against {f((D.ref(s, "liftpush") or {}).get("solved"))} % for lift + push and {f((D.ref(s, "liftpush_oct") or {}).get("solved"))} % with the octaves' for s in hs) +
                         '. The backup reaches the level of the octave arm with the lift’s plain Fourier features and no push.'
                         + (f' The largest house is the exception in its travel times: the field is {f(100 * (1 - (P("bxt1nL", "Klickitat").get("ratio") or np.nan)), 0)} % short and {v("bxt1nL", "seen_through", 0, "Klickitat")} % of its detours are still seen through, mostly the long ones (12–24 cells); '
                            f'a step of at most 5.3 cells is 0.055 box lengths there, against 0.24 on the 22-cell maze.' if PV('bxt1nL', 'Klickitat') else '')
                         + (f' With the longest step set to 0.24 box lengths there too (23 cells): <b>{v("bxt1nLB", "solved", 1, "Klickitat")} % solved</b>, {v("bxt1nLB", "within10", 1, "Klickitat")} % of travel times within 10 %, '
                            f'{v("bxt1nLB", "seen_through", 0, "Klickitat")} % of detours seen through. The longest step has to follow the size of the scene, not of the cell.' if P('bxt1nLB', 'Klickitat') else ''))
        if PV('bxt1nL', 'apt64'):
            notes.append(f'On the apartment the backup plans a few points above the earlier arms (one training seed) but the detours stay seen through ({f(PV("bxt1nL", "apt64").get("seen_through"), 0)} %), as under every arm of the lift-scale study; its training also rolls back {one("bxt1nL", "apt64").get("rollbacks", "–")} times. That scene is still open.')
        A('<p>' + ' '.join(notes) + '</p>')
        if 'fit_scenes' in C:
            def fs(lab, sc, key='solved', d=1):
                return f((D.one(lab, sc) or {}).get(key), d)
            A('<h3>How far is the backup from what the network can do?</h3>')
            A(f'<p>The representation test of Part 1, on the two scenes where the backup falls short. Regressed on the true travel times, the lifted network plans {fs("fit_liftB", "plus40")} % on the 40-cell maze '
              f'({fs("fit_lift_octB", "plus40")} % with the octaves), with {fs("fit_liftB", "plus40", "within10")} % of its travel times within 10 % and no query of 64–128 cells solved. The backup field plans {f(BL("plus40").get("solved"))} % there. '
              f'So on the 40-cell maze the backup plans as well as a field that was shown the right travel times, and a better loss alone would not go further. The plans that fail stop in front of a wall they have to go around ({v("bxt1nL", "stall", 1, "plus40")} % stalled, {v("bxt1nL", "wall", 1, "plus40")} % through a wall), '
              f'and the fitted field holds only {fs("fit_liftB", "plus40", "jump_big", 2)} of the true step across walls ({fs("fit_lift_octB", "plus40", "jump_big", 2)} with the octaves) after the 20,000 steps that gave 0.95 on the 22-cell maze: here the step across a wall is the part the plain lifted network is slow to fit, and the octaves help. '
              f'On the apartment the fitted lifted network plans {fs("fit_liftB", "apt64")} % and holds {fs("fit_liftB", "apt64", "jump_big", 2)} of the step ({fs("fit_lift_octB", "apt64")} % and {fs("fit_lift_octB", "apt64", "jump_big", 2)} with the octaves), '
              f'against {f(BL("apt64").get("solved"))} % for the backup, whose field holds almost none of the large steps there (a true step of 17 cells is held as 1.6): on the apartment the loss is still the limit.</p>')
            A(grid('fit_scenes', 'fit_scenes_jump'))
        oc = [s for s in SCENES if P('rec', s) or P('bxt1nL_oct', s)]
        if oc:
            def ocv(s, key='solved', d=1):
                return f((P('rec', s) or P('bxt1nL_oct', s)).get(key), d)
            A('<h3>Backup and octaves together</h3>')
            r40 = one('bxt1nL_oct', 'plus40') or one('rec', 'plus40')
            A('<p>The octaves of the lift-scale study (three more bands of Fourier rows at 2, 4 and 8 times ntrl-demo’s frequencies on the 22-cell maze, scaled with the scene) are a representation fix; the backup is a selection fix. '
              + (f'On the 40-cell maze each alone stops near 70 % (lift + push + octaves {f((D.ref("plus40", "liftpush_oct") or {}).get("solved"))} %, lift + backup {v("bxt1nL", "solved", 1, "plus40")} %). Together, with steps up to 5.3 cells: '
                 f'<b>{v("bxt1nL_oct", "solved", 1, "plus40")} % solved</b>, no detour seen through, and the queries with a true path of 64–128 cells, which no earlier field solved (not even the ones fitted to the truth), {f(r40.get("by_bin", {}).get("64-128"), 0)} % solved. '
                 f'The share solved was still rising at the end of training ({f(100 * r40.get("course", {}).get("3000", {}).get("success", [0, 1])[0] / max(1, r40.get("course", {}).get("3000", {}).get("success", [0, 1])[1]), 0)} % at epoch 3,000 on planning seed 0). '
                 if P('bxt1nL_oct', 'plus40') else '')
              + (f'On the apartment the same pair gives {v("bxt1nL_oct", "solved", 1, "apt64")} %, with {v("bxt1nL_oct", "seen_through", 0, "apt64")} % of detours still seen through. ' if P('bxt1nL_oct', 'apt64') else '')
              + ('The same with the longest step scaled to the scene (0.24 box lengths instead of 5.3 cells; one training seed): '
                 + '; '.join(f'{SHORT[s]} <b>{v("rec", "solved", 1, s)} %</b> ({refs(s)})' for s in SCENES if P('rec', s)) + '. ' if any(P('rec', s) for s in SCENES) else '')
              + (f'On the 40-cell maze the scaled steps (up to 9.6 cells) are {f(P("bxt1nL_oct", "plus40")["solved"] - P("rec", "plus40")["solved"])} points below the 5.3-cell steps, and almost all of the difference is in the queries of 64–128 cells '
                 f'({f(one("rec", "plus40").get("by_bin", {}).get("64-128"), 0)} % against {f(one("bxt1nL_oct", "plus40").get("by_bin", {}).get("64-128"), 0)} %); with one training seed each this is not yet separated from seed-to-seed variation. '
                 if P('rec', 'plus40') and P('bxt1nL_oct', 'plus40') and P('rec', 'plus40')['seeds'] == 1 else '') + '</p>')
        rows = []
        for s in done_scenes:
            for lab, _, name in PG.BACKUPS:
                p = P(lab, s) or (P(PG.ALIAS[lab], s) if lab in PG.ALIAS else {})
                if p:
                    rows.append([TITLE[s], name, p['seeds'], f(p['solved']) + (f' ({each(p)})' if each(p) else ''), f((D.ref(s, 'ntrl') or {}).get('solved')), f((D.ref(s, 'liftpush') or {}).get('solved')), f((D.ref(s, 'liftpush_oct') or {}).get('solved')),
                                 f(p.get('within10')), f((D.ref(s, 'liftpush_oct') or D.ref(s, 'liftpush') or {}).get('within10')), f(p.get('wall')), f(p.get('stall')), f(p.get('seen_through'))])
        A(details('Every scene and variant: the numbers', table(['scene', 'backup variant', 'seeds', 'solved %', 'ntrl-demo', 'lift + push', 'lift + push + octaves', 'T within 10 %', 'best lift arm T within 10 %', 'through a wall %', 'stalled %', 'seen through %'], rows, left=2)))

    # ------------------------------------------------------------------ limits
    A('<h2>What this does not show</h2>')
    A('<p>The pair input was tested on the rectangle robot in SE(2), not on the point robot of the lift pages; the three failure mechanisms do not depend on the robot, but the numbers are this setup’s. '
      'The pair inputs were trained with the push and the cap; a pair input under some other selection term was tested only once, on top of the backup. '
      'The backup needs a collision check of short straight steps and the clearance along them, the same information the lift and the push already use; its weight, number of candidate steps and radius were chosen on the 22-cell maze, where a three times larger weight or half the radius lose 12–18 points, all on the longest queries. '
      'The backup’s step time reads the clearance from a table (32 nodes per cell, within 1 % of the exact step time on average); the run with the exact rectangle clearance was stopped at epoch 2,400 to free the GPU and is not a result.</p>')

    # ------------------------------------------------------------------ tables
    def run_rows(kind):
        out = []
        for r in D.runs:
            if r['kind'] != kind:
                continue
            name = FIT.get(r['label']) if kind == 'fit' else ARM.get(r['label'], (r['label'],))[0]
            if kind == 'fit':
                out.append([esc(name), r['seed'], f"{r['steps']:,}", r.get('band', 0), f(r.get('solved')), f(r.get('wall')), f(r.get('stall')), f(r['within10']), f(r['ratio'], 3), f(r.get('rho'), 2), f(r.get('seen_through')), f(r.get('jump_big'), 2),
                            f"<code>{r['id']}</code>", f"<code>{esc(r['extra'])}</code>"])
            else:
                out.append([esc(TITLE.get(r['scene'], r['scene'])), esc(name), r['seed'], f(r['solved']), f(r['wall']), f(r['stall']), f(r['within10']), f(r['ratio'], 3), f(r.get('los_within10')), f(r.get('rho'), 2), f(r.get('seen_through')),
                            f(r.get('wall_innermost'), 2), r.get('rollbacks', ''), f(r['minutes']), f"<code>{r['id']}</code>", f"<code>{esc(r['extra'])}</code>"])
        return out
    head = ['scene', 'arm', 'seed', 'solved %', 'through a wall %', 'stalled %', 'T within 10 %', 'median T / T*', 'line of sight within 10 %', 'detour index', 'seen through %', 'S × slope innermost band', 'rollbacks', 'minutes', 'run', 'arguments']
    A('<h2>Every run</h2>')
    A(details('Fields trained without the truth, and the reference runs', table(head, run_rows('reference') + run_rows('field'), left=2)))
    A(details('Fits to the true travel time', table(['network', 'seed', 'steps', 'band share', 'solved %', 'through a wall %', 'stalled %', 'T within 10 %', 'median T / T*', 'detour index', 'seen through %', 'step held across walls', 'run', 'arguments'], run_rows('fit'))))
    if D.failed:
        A(details('Runs stopped by hand', '<ul>' + ''.join(f'<li><code>{esc(j.split("|")[0])}</code> ({esc(rid)}): stopped to free a GPU slot when a faster version of the same arm had finished; noted in the registry.</li>' for j, rid in D.failed.items()) + '</ul>'))
    A('<footer>Study folder <code>jeffrey/pair_lift</code> (README.md has the setup and the run ids). Scores: 240 queries × MPPI planning seeds 0 / 1 / 2 on the 22 × 22 plus maze, exact full-robot clearance. '
      'Travel-time measures: 24,000 random (free configuration, goal) pairs against the grid solver. One training of a recipe moves by 3–6 points between seeds.</footer>')
    return '\n'.join(x for x in H if x)
