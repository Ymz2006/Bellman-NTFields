"""Words and page shell of the sharp-feature findings page (page.py fills in the charts and numbers).

FINDINGS holds the per-section result text (edited as the runs finish); everything else is the fixed description."""
import html
from page_css import CSS, SCRIPT

esc = lambda s: html.escape(str(s), quote=True)  # noqa: E731

ANSWER = [
    "<b>No sharp feature beats the octave.</b> The best ones tie it, exactly as plain features of the same count do: on 22×22, over "
    "three training seeds, the triangle pair added to every row +1.6 ± 2.0 points, a copy of the 128 rows +3.0 ± 2.0, 128 new octave rows "
    "+2.2 ± 2.0. The gate as written, max(0, tanh(z − h)) · (8/π) arcsin(cos z) with two random offsets per row: −4.5 ± 2.0 (−6.7 against "
    "the 128 extra plain rows); on 13×13 −0.4 ± 1.5. No gate variant (offsets at grid vertices or on every wall line, slopes, amplitudes, "
    "the gate alone, the gate on sin / cos) came out ahead.",
    "<b>Even powers lose at every p and in every form.</b> Mixed p = 2…16 replacing half the rows: −14.0 ± 2.0 (22×22, 3 seeds), "
    "−22.2 ± 1.5 (13×13); added next to every row: −17.2 ± 2.0 and −18.8 ± 1.5, i.e. −19 / −21 against the same number of plain inputs; "
    "replacing every row: −40 (22×22, one seed per p).",
    "<b>The triangle as written loses through its shape, not its size.</b> Replacing every row: −7.6 ± 2.0 (22×22), −11.2 ± 1.5 "
    "(13×13); sin / cos scaled to the triangle's size: +1.1 (one seed).",
    "<b>Why.</b> The sharp features do what was hoped at the walls (with the powers added, runs through a wall fall from about 130 of "
    "720 to 27, the speed fit and the training loss improve), but the learned travel time folds into a sawtooth: up to 21 % of the "
    "true path runs uphill toward the goal (the octave under 1 %) and the planner stops in the dips. The eikonal loss only constrains "
    "the size of the gradient, which a sawtooth satisfies. Even powers make the fold cheap because they are mirror-symmetric about every "
    "crest of their row: ±cos 2z alone (= sin², cos²) folds and costs 23 points, the full pair sin 2z, cos 2z does not (−10, under 1 % "
    "uphill), and the sign-keeping power |s|<sup>4</sup> sgn(s) removes the local minima.",
    "<b>Putting the sharpness on the walls does not rescue it.</b> On rows along the grid axes, whose crests and gate edges lie on "
    "every wall line, powers still lose 10–13 points and gates 17–19; the slow band the loss must resolve lies 0.15–0.36 cell from the "
    "wall line and moves with the robot's heading.",
    "<b>The midpoint loss does not rescue them.</b> Forked at epoch 2,000 and trained with jeffrey/midpoint_dp's travel-time midpoint "
    "loss (22×22, three seeds), the exact octave goes from 46.3 to 54.9 % and 128 extra plain octave rows from 48.6 to 58.1 % (+3.2 ± 2.5 "
    "against the octave with the loss). The sharp features gain only 2–3 points from it and stay behind the octave with the loss: the gate as "
    "written −10.7 ± 2.5, the triangle as written −13.4 ± 2.5, the powers on half the rows −20.0 ± 2.5, the powers added −22.6 ± 2.5. Their "
    "travel time stays folded (the added powers keep 18 % of the true paths uphill) because the loss learns from the field's own "
    "traversals, and on a folded field almost all of them stall, so the long paths that would unfold the sawtooth never reach the loss.",
]

MIDPOINT = """<p>Each field is forked at epoch 2,000 of its run and trained to 5,000 with jeffrey/midpoint_dp's travel-time midpoint
loss (its best setting: weight 1 ramped over 500 epochs, min leaves of one cell, traversals that keep out of walls, tree and sub-path
nodes); the loss asks T(a, b) = T(a, c) + T(c, b) at travel-time midpoints c of the field's own MPPI traversals, so a sawtooth that
satisfies the eikonal loss no longer satisfies it. 22×22, training seeds 87358 / 1 / 2 (the grid-axis powers and the dense top-octave
gates: seed 87358 only). Change against the exact octave with the same loss on the same seed (± SE from one run's training noise, 3.0
points):</p>
<div class="scroll"><table><thead><tr><th>field</th><th>without → with the loss</th><th>with the loss vs octave + loss</th><th>share of the true paths uphill, without → with</th></tr></thead><tbody>
<tr><td>exact octave</td><td>46.3 → 54.9 %</td><td>–</td><td>2.2 → 0.5 %</td></tr>
<tr><td>128 extra plain octave rows (control)</td><td>48.6 → 58.1 %</td><td>+3.2 ± 2.5</td><td>1.4 → 0.5 %</td></tr>
<tr><td>gate × triangle, as written</td><td>41.9 → 44.2 %</td><td>−10.7 ± 2.5</td><td>1.3 → 1.1 %</td></tr>
<tr><td>triangle replacing every row, as written</td><td>38.8 → 41.5 %</td><td>−13.4 ± 2.5</td><td>2.6 → 1.8 %</td></tr>
<tr><td>sin<sup>p</sup>, cos<sup>p</sup> mixed, half the rows</td><td>32.4 → 35.0 %</td><td>−20.0 ± 2.5</td><td>13.7 → 12.7 %</td></tr>
<tr><td>sin<sup>p</sup>, cos<sup>p</sup> mixed, added</td><td>29.2 → 32.3 %</td><td>−22.6 ± 2.5</td><td>18.9 → 17.9 %</td></tr>
<tr><td>sin¹⁶, cos¹⁶ on the grid-axis rows (seed 87358)</td><td>39.2 → 44.6 %</td><td>−17.1</td><td>3.1 → 2.4 %</td></tr>
<tr><td>gate × triangle, top octave, 8 offsets (seed 87358)</td><td>32.4 → 35.7 %</td><td>−26.0</td><td>14.6 → 14.7 %</td></tr>
</tbody></table></div>
<p>The loss helps plain fields (octave +8.6, extra plain rows +9.5, both lose their small fold) and barely moves the sharp ones (+2 to +3):
their travel time stays folded. The loss learns from the field's own traversals, and on the folded fields they stall: with the powers
added 93–95 % of the 8,192 traversals of each refresh stall (octave 64 %), so the trees hold 20,000–34,000 nodes of short paths against
about 280,000 for the octave, and the long-range targets that would unfold the sawtooth never appear.</p>"""

FINDINGS = {
    'powers': """<p>22×22, seed 87358, one training per spec unless noted (the exact octave of that seed averages 51.9 %). Every even power
loses, at every p, in every form (three seeds: half the rows mixed −14.0 ± 2.0, added −17.2 ± 2.0). Half the rows replaced: 29–34 % for p = 2, 4, 8, 22 % for p = 16, 31 % for the mix; every row
replaced: 5–12 %. Added next to every row (512 inputs): 26–29 % for p = 2 … 16, 24 % with all four powers (1,280 inputs), 28 % with
the powers on the lower octaves only. The plain baselines of the same width: a copy of the rows 51.7 %, the rows at twice their
frequency (sin 2z and cos 2z) 42.2 %, 128 new octave rows 43.3 %, 512 new octave rows (1,280 inputs) 54.0 %.</p>
<p>Two controls locate the cause. sin² z and cos² z are both ½ ∓ ½ cos 2z: a smooth cosine at twice the frequency, nothing sharp.
Added next to the rows they still cost 23 points, while the full pair sin 2z, cos 2z (a superset of their information) costs 10, with
the travel time folded in the first case (18 % of the true path uphill) and not in the second (under 1 %). And the sign-keeping power
sgn(s) |s|<sup>4</sup>, as sharp as s<sup>4</sup> but odd, replacing half the rows: 41.9 % against 33.5 %, with the local minima gone.
So what hurts is that even powers are mirror-symmetric about every crest of their row; turned down to 0.3 or 0.1 of their size the
added powers fold much less (41 %, 43 %), and gain nothing over plain features.</p>""",
    'triangle': """<p>Three seeds (22×22): replacing every row −7.6 ± 2.0, added +1.6 ± 2.0; 13×13 (two seeds) −11.2 ± 1.5 and −3.3 ± 1.5.
Seed 87358 alone: the triangle as written replacing every row: 38.6 %, with 4 local minima per long path;
sin / cos scaled to the triangle's RMS (×3.27, the gain control): 53.1 %, so the loss comes from the triangle's shape, not its
amplitude. At amplitude 1 the triangle does not fold but its walls go soft (41.2 %, 197 runs through a wall against about 130).
Added next to every row: 43.6 %, level with 128 extra plain octave rows (43.3 %).</p>""",
    'gates': """<p>The gate as written over three seeds (22×22): −4.5 ± 2.0; 13×13 (two seeds) −0.4 ± 1.5. Seed 87358 alone: no gate variant beats the octave (51.9 %). The formula as written, max(0, tanh(z − h)) ·
(8/π) arcsin(cos z) with two random offsets per row, added next to every row: 44.4 %. Offsets at random grid vertices: 50.7 %; the gate
times sin / cos: 49.7 %; the gate times the unit triangle: 43.9 %; the gate times the triangle pair: 43.1 %; the gate alone: 42.6 %;
slope a = 3 or 0.3: 38.2 % / 41.5 %; on top of the triangle rows: 39.6 %. Dense gates fold the travel time: eight offsets on every
top-octave row give 32.4 % with 14 % of the true path uphill. Plain controls of the same width: 43.3 % (128 new rows), 51.7 % (a copy
of the rows).</p>""",
    'axis': """<p>Interim (22×22, seed 87358). Aligning the sharp features with the walls does not rescue them. Plain sin / cos on the
grid-axis rows: 52.8 %; the triangle on them: 52.9 % (a tie); sin⁴, cos⁴ on them: 41.4 % (7.5 % of the path uphill); sin¹⁶, cos¹⁶:
39.2 %; the gate with an edge on every wall line: 34.9 % alone, 32.6 % times the triangle (16 % of the path uphill). Before training
the reason is already visible: the slow band the eikonal loss must resolve lies 0.15–0.36 cell from the wall line (half the wall
plus the robot's half extent, which depends on its heading), so a crest or edge on the line itself sits in the wrong place
(near-wall arc-length ratio 0.97–1.02 for every axis feature).</p>""",
    'fail': """<p>The even powers move the failures from walls to stalls: with the mixed powers added, the runs that reach the goal only
through a wall fall from about 130 of 720 to 27, and the runs that never arrive rise from about 215 to 501.</p>""",
    'mechanism': """<p>The sharp features fit the training loss better and plan worse. With the powers added, the held-out speed correlation
rises from 0.80–0.82 to 0.85, the near-wall slope from 0.62–0.71 to 0.76 and the final loss falls; yet the learned travel time has 11
local minima per long true path (exact octave 0.3–0.5). Along the true paths the eikonal residual |S |∇T| − 1| drops from 0.44–0.51
to 0.16, while 21 % of the path length runs uphill toward the goal (exact octave under 1.1 %). The eikonal loss only constrains the
size of the gradient, so a travel time that zig-zags (a sawtooth) satisfies it as well as the true one. Two routes lead there: an
input that is itself folded (even powers replacing rows: the network cannot build an odd function of those rows) and an input with
kinks or narrow peaks (the triangle, the added powers), which make a sawtooth cheap to represent. The planner stops in its valleys.</p>""",
}


def pct(v, p=1):
    return '–' if v is None else f'{100 * v:.{p}f} %'


def pair(charts, key, title13, title22, note=''):
    return (f'<div class="pair"><div class="panel"><h3>{title13}</h3>{charts[key + "13"] if key + "13" in charts else charts[key + "_13"]}</div>'
            f'<div class="panel"><h3>{title22}</h3>{charts[key + "22"] if key + "22" in charts else charts[key + "_22"]}</div></div>'
            + (f'<p class="note">{note}</p>' if note else ''))


def render(S, charts, legends, tables, screen, usage, runs, status, groups, pm, spec, label):
    ref13 = S['sizes'].get('13', {}).get('reference', {})
    ref22 = S['sizes'].get('22', {}).get('reference', {})
    sd13 = (S['noise'].get('13', {}).get('success_rate') or {}).get('sd')
    sd22 = (S['noise'].get('22', {}).get('success_rate') or {}).get('sd')

    def oct_rate(ref):
        rates = [v['mean_rate'] for v in ref.values()]
        return sum(rates) / len(rates) if rates else None

    n_runs = sum(len(e['per_seed']) for n in ('13', '22') for k, e in (S['sizes'].get(n) or {}).get('specs', {}).items() if not k.startswith(('ctx:', 'ntrl')))
    answer = ''
    if ANSWER:
        answer = '<section class="answer"><h2>Answer</h2><ul>' + ''.join(f'<li>{b}</li>' for b in ANSWER) + '</ul></section>'
    status_html = f'<p class="status">{status}</p>' if status else ''
    shapes = ''.join(f'<div class="panel"><h3>{t}</h3>{s}</div>' for t, s in charts['shapes'])

    def section(i):
        return f'<div class="panel full">{charts[f"g{i}"]}</div>'

    return f"""<title>2026-11-18 Sharp Features</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&family=JetBrains+Mono:wght@400&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header class="stack">
  <span class="eyebrow">jeffrey/sharp_features · grid-sync octave embedding · 13×13 and 22×22 mazes · ntrl-demo</span>
  <h1>2026-11-18 Sharp Features</h1>
  <p class="lede">Can sharp input features on top of the octave sin / cos embedding make walls easier to learn? Three ideas, each in
  several forms: even powers sin<sup>p</sup>, cos<sup>p</sup> (p = 2, 4, 8, 16), replacing some rows or added next to them; the
  triangle wave (8/π) arcsin(cos z); and that triangle switched on past an offset h by max(0, tanh(z − h)). Every field is scored
  against the exact octave trained with the same seed.</p>
  {status_html}
</header>
{answer}

<h2>What was tried</h2>
<div class="stack">
<p>Each of the 128 octave rows has a phase z = 2π B·(x, y, φ); the network reads sin z and cos z of every row (256 inputs). The
features below act on that phase. Replacing keeps 256 inputs and ntrl-demo's network bit for bit (same initialization and data order
as the octave run of the seed). Adding appends 256 inputs (1,280 with every power); its baselines append the same number of plain
sin / cos: 128 new octave rows, a copy of the 128 rows, or the rows at twice their frequency (sin² z = (1 − cos 2z) / 2).</p>
<div class="shapes">{shapes}</div>
<p class="note">Reference: {pct(oct_rate(ref13))} on 13×13 and {pct(oct_rate(ref22))} on 22×22 for the exact octave (mean of the reference
trainings of seeds 87358 / 1 / 2). Training noise of one run: {'–' if sd13 is None else f'{100 * sd13:.2f}'} points on 13×13,
{'–' if sd22 is None else f'{100 * sd22:.2f}'} on 22×22; the shaded band on the charts is ±2 of that for one seed against its
reference mean. Whiskers: ±2 standard errors of the mean over seeds. Hover a row for every seed.</p>
{legends['roles']}
</div>

<h2>Idea 1: powers of sin and cos</h2>
<div class="stack">
{FINDINGS['powers']}
{legends['power']}
{pair(charts, 'power', 'Success change against the power, 13×13', 'Success change against the power, 22×22')}
<h3>Way 1: replace (256 inputs)</h3>
{section(0)}
<h3>Way 2: add (512 inputs), with its plain baselines</h3>
{section(1)}
</div>

<h2>Idea 2: the triangle (8/π) arcsin(cos z)</h2>
<div class="stack">
{FINDINGS['triangle']}
{section(2)}
</div>

<h2>Idea 3: the tanh gate</h2>
<div class="stack">
{FINDINGS['gates']}
{section(3)}
</div>

<h2>Best case: rows on the grid axes</h2>
<div class="stack">
<p>The octave rows keep ntrl-demo's random directions, so a crest of sin<sup>16</sup> or a kink of the triangle crosses the walls at an
angle almost everywhere. Rows on the x and y axes at each octave level are the only ones whose crests, kinks and gate edges lie on the
wall lines everywhere (a gate on them gets one edge on every grid line). Their plain sin / cos and gate-alone versions are the controls.</p>
{FINDINGS['axis']}
{section(4)}
</div>

<h2>How the failures change</h2>
<div class="stack">
<p>A failed run either reaches the goal only after crossing a wall (soft walls) or never reaches it (stopped against a wall or in a
local minimum). Filled dot: change in the share of runs through a wall; open dot: change in runs that never arrive.</p>
{FINDINGS['fail']}
{legends['fail']}
<div class="panel full">{charts['fail']}</div>
<p>Where the runs that never arrive stop (planning seed 0): moved less than 0.1 cell over the last 20 steps with no wall within 0.15
cell means the planner sits in a local minimum of the learned travel time in an open corridor. The exact octave stops there in
3–9 % of its runs.</p>
<div class="panel full">{charts['stall']}</div>
</div>

<h2>Mechanism</h2>
<div class="stack">
{FINDINGS['mechanism']}
{legends['mech']}
{pair(charts, 'mech', 'Eikonal fit against monotone travel time, 13×13', 'Eikonal fit against monotone travel time, 22×22',
      'Each dot is one training. Left = the field satisfies the eikonal equation better along the true paths; up = more of the true path runs uphill toward the goal (the true travel time falls all the way). Log scale.')}
{pair(charts, 'mechs', 'Success against uphill share, 13×13', 'Success against uphill share, 22×22')}
{pair(charts, 'trade', 'Near-wall slope against local minima, 13×13', 'Near-wall slope against local minima, 22×22',
      'Each dot is one spec (paired change against the exact octave). Right = steeper travel time at walls (walls that hold); up = more local minima of T along the true paths (ripples that stall the planner).')}
<details><summary>How much the trained network weighs the sharp columns (usage.py)</summary>
<p class="note">Share of the squared weight (after ntrl-demo's row normalization) of the three layers that read the input that sits on
the sharp columns, the share those columns have of the input, and for replace specs the share the exact octave of the same seed puts on
the same columns. A share near the column share means the network uses a sharp column like a plain one; it never ignores them, and it
weighs the grid-axis powers three times above their share.</p>
{usage}
</details>
<details><summary>Input mapping before training (screen.py, seed 87358)</summary>
<p class="note">arc: squared arc length of the embedding per unit of the plane at near-wall points (trained speed below 0.3) over the
same in the open (speed above 0.95); 1 means the feature puts no extra resolution at the walls. fold: embedding distance between
points 0.25 cell either side of a grid line over the same displacement inside a cell. Linear speed readout: ridge regression of log
speed on the features, held-out R².</p>
{screen}
</details>
</div>

<h2>With the midpoint loss</h2>
<div class="stack">
{MIDPOINT}
{legends['mid']}
<div class="pair"><div class="panel"><h3>Success, 22×22</h3>{charts['mid_rate']}</div>
<div class="panel"><h3>Share of the true paths running uphill, 22×22</h3>{charts['mid_uphill']}</div></div>
</div>

<h2>Setup</h2>
<details><summary>Mazes, training, scoring, pairing</summary>
<div class="stack">
<p>Mazes, data, network, losses, optimizer (AdamW, 5,000 epochs), MPPI planner and exact collision scorer are those of the earlier
maze studies, unchanged: 13×13 (200 queries × planning seeds 0/1/2 = 600 runs) and 22×22 (240 queries, 720 runs). Every spec starts
from the exact octave matrix of the same training seed. Reference: every earlier training of that maze and seed with the exact octave
embedding (the octave-detune study's octave and its ×1.0001 / ×0.9999 nulls, plus the grid-sync run where it ran on the other GPU).
Added features enter the three layers that read the input (widened to 256 + m inputs); the first block's residual carries the 256
octave features. A pipeline check reproduced a registered octave run loss for loss over 40 epochs. All runs are registered (registry
search "Sharp features"); modified-environment results on this workstation.</p>
</div>
</details>

<h2>Tables</h2>
<details><summary>Every spec: success, paired changes, failure split, field measures</summary>
{tables}
</details>
<details><summary>Run ids ({n_runs} trainings so far)</summary>
{runs}
</details>
</div>
<div id="tip" hidden></div>
<script>{SCRIPT}</script>
"""
