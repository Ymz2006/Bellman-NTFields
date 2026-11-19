"""Prose for the gap page (gappage.py), written from report/gap_tables.md after all 121 runs (3 training seeds; the warped-pair test on 87358).
Numbers that come from tables are read from P where they head a tile; the rest quote gap_tables.md / gap_stalls.json (seed 87358)."""


def tiles(P):
    t = []
    g, n, o = P.get((12, 'geoc')), P.get((12, 'ntrl')), P.get((12, 'octc'))
    if g and n and o:
        t.append((f'{g["pct"]:.1f} %', f'gap ×12, clearance gate on per-cell octaves (ntrl-demo {n["pct"]:.1f} %, same octaves always on {o["pct"]:.1f} %), 3 seeds'))
    a, n2 = P.get((2, 'oct8')), P.get((2, 'ntrl'))
    if a and n2:
        t.append((f'{a["pct"]:.1f} %', f'gap ×2, octaves ×2..×8 (ntrl-demo {n2["pct"]:.1f} %), 3 seeds'))
    if g:
        t.append((f'{100 * g["cats"]["open - open"][0] / g["cats"]["open - open"][1]:.0f} %', 'open–open queries kept at ×12 (the constructed C-space warp trained on the same maze: 47 %)'))
    if g:
        c = g['cats']
        adj = 100 * (c['adjacent quadrants'][0] + c['diagonal quadrants'][0]) / (c['adjacent quadrants'][1] + c['diagonal quadrants'][1])
        t.append((f'{adj:.0f} %', 'queries between two quadrants solved at ×12, every method (the part nothing here fixes)'))
    return t


TEXT = dict(
    lede=[
        'The plus maze of the main study, cut down to the 10 × 10 maze\'s four 5 × 5 quadrants and pushed apart by an open band of 6, 12, 24, 48 and 72 cells (×1 … ×12), '
        'with the three gates you asked for on the same embedding map as before: the <b>constructed gate</b> from the environment (clearance band, open within half a cell of a wall, shut beyond 1.5 cells), '
        'the <b>diffusion gate</b> (heat from the free/obstacle boundary through free space), and the <b>gate learned from T</b> (sigmoid MLP on ntrl-demo\'s own features, trained only by the travel-time loss). '
        'Each against ntrl-demo and against the same octaves always on, three training seeds, the same 300 queries per maze and the same training pairs as ntrl-demo.',
        '<b>What came out.</b> With the octaves scaled to the cell, every gate and the always-on control beat ntrl-demo at every gap over three seeds: by 8–9 points at ×1 (78–79 against 70 %), 13–20 at ×2, 10–12 at ×4, 4–6 at ×8 and 4–7 at ×12, '
        'and none of them gives up the open band, which is exactly what the C-space warps lost on these mazes. Two settings matter more than the choice of gate. '
        'The octaves have to follow the cell: ×2, ×4, ×8 at every gap fades to ntrl-demo\'s level at ×12 (26 %), the same octaves scaled by the maze side keep the gain (30–33 %). '
        'And the gate only pays once those octaves are high against the open band: at ×12 (top octave ×30) the clearance gate is ahead of the always-on control on each of the three seeds (33.3 against 30.2 %), '
        'the diffusion and learned gates by about a point; from ×1 to ×8 all four are within seed noise of each other. The learned gate again finds the geometry by itself. '
        'What none of them touches is the query between two quadrants (1–5 % from ×8 on): the field inside a quadrant does not know the way out to a far goal, and nine times more training pairs inside the quadrants did not change that either.',
    ],
    tiles=tiles,
    results=[
        'Two octave families, each with the three gates and the always-on control. <b>Fixed</b>: ×2, ×4, ×8 at every gap, the plus maze\'s best setting. <b>Per cell</b>: the same three octaves times E / 22.1 (E = maze side in cells), so that a quadrant cell meets the spectrum it had on the 22 × 22 plus maze; the factor is 1.54 / 2.63 / 3.71 at ×4 / ×8 / ×12 (top octave ×12.3 / ×21 / ×29.7), and is clamped to 1 at ×1 and ×2, where the two families are the same runs.',
        'At ×1 and ×2 the gain is large and the gate does not matter: always on 77.9 / 64.9 %, clearance 78.3 / 63.4, diffusion 78.7 / 60.4, learned 79.2 / 58.0, against ntrl-demo 70.2 / 45.1 (one training of the same recipe moves by up to 14 points at ×1, so these four are level). '
        'From ×4 the fixed octaves fall behind the per-cell ones (×8: 27–29 against 30–31 %; ×12: 26.2–26.6, ntrl-demo\'s level, against 30.2–33.3): the quadrant cells shrink in the encoder\'s box as the band grows, and an octave tied to the box no longer resolves them.',
    ],
    refs_note='The fields trained and planned on the warped maze with pairs drawn there (85 % at ×12 for the constructed warp, 93 % for the learned line warp) are a different problem and not drawn: they are not asked for the robot\'s real travel time across the band. ntrl-demo at ×2 is 45.1 % here (3 seeds on this machine\'s GPUs) and 52.0 % in the cross-warp run of seed 87358 on the other card; ×4 / ×8 / ×12 agree to the decimal.',
    cats=[
        'By query kind, the embeddings gain in the queries that stay in one quadrant or go between a quadrant and the band, and keep the band itself: at ×12 the clearance gate solves 35 % of same-quadrant queries (ntrl-demo 12 %) and 26 % of quadrant–open ones (20 %), and 99 % of open–open ones. '
        'The constructed C-space warp of the cross-warp study, trained on the same maze with the same pairs, reached the same-quadrant level (34 %) only by losing the band (open–open 47 %) and solved 17.9 % in total (training seed 87358).',
        'Queries between two quadrants are the wall nothing here passes: adjacent 3–5 %, diagonal 0–1 % at ×12, for every gate, for always on and for ntrl-demo. ntrl-demo "solves" none of its failures by stalling: at ×12 1,980 of its 1,986 failed runs reach the goal through walls. The embeddings stop at the walls instead (44–48 runs through a wall, 1,750–1,840 never arrive).',
    ],
    cat_gaps=[2, 12],
    cat_labels=['octc', 'geoc', 'diffc', 'learnc'],
    fail_gaps=[2, 12],
    gates=[
        'The gate maps (training seed 87358, per-cell octaves). The clearance and diffusion gates are built once from the maze, before training, and stay fixed; the learned gate starts at 0.5 everywhere and is shaped only by the travel-time loss. '
        'All three open over the quadrants and shut over the band; the learned one opens over the whole quadrant (μ 0.85 near walls at ×4) and leaves a soft halo into the band, the diffusion one decays more slowly than the clearance one.',
        'Isotropy, measured on the network\'s actual input: the clearance and diffusion gates keep the band anisotropy at 1.00 at every gap (stretch at the walls ×5.7 / ×9.7 / ×13.7 at ×4 / ×8 / ×12 for per-cell octaves, ×0.8 in the band). '
        'The learned gate does not stay isotropic on the wide mazes: its μ changes within a fraction of an octave wavelength near the walls, and the rank-one term m|∇μ|² of the induced metric grows: median band anisotropy 1.0 at ×4, 1.7–4.1 at ×12 (90th percentile 8–14), and 2–4.5 with fixed octaves at ×8 / ×12, where it also half-closes over the quadrants (μ 0.3–0.4) and lets the walls leak (1,510 runs through a wall at ×12 against 44–48 for the per-cell gates).',
    ],
    gates_caption='Gate μ at heading 0 (mean over the three octaves), training seed 87358, per-cell octaves; walls black. Top: the whole ×4 maze. Bottom: the lower-left quadrant of the ×12 maze and four cells of its band. Title: that run\'s success.',
    mechanism=[
        'Where the failed runs end (seed 87358, planning seed 0, queries with a quadrant end that never arrived): at ×12, 44–48 % of the octave fields\' stalls are queries that leave their start quadrant and never do; they stop within about half a cell of the start (median 0.4–0.5 cells moved). '
        '18–27 % stop in the band, 7–14 % inside the goal quadrant, and about a fifth are same-quadrant queries. ×8 is the same; at ×1 most stalls are already in the goal quadrant (41–54 %).',
        'Why they stop at the start: below, T(·, goal) over the start quadrant of a diagonal ×12 query, the goal 75 cells away in the opposite quadrant. The octave fields have steep barriers inside every wall (the crowded contours), but between the walls T simply falls in the straight-line direction of the far goal; the true path (dashed) first runs up a corridor to the quadrant\'s edge, and nothing in the field says so. '
        'MPPI follows the descent into the nearest wall and stops (red). ntrl-demo has no barrier, so it walks through the walls along the same straight line and arrives in collision.',
    ],
    far_caption='T(·, goal) at the start heading over the lower-left quadrant of the ×12 maze and four band cells, for a diagonal query (start: red dot; goal 75 cells up and right). Dashed: the true path; red: the MPPI plan, planning seed 0. Colors per panel from the 1st to the 99th percentile; white contours evenly spaced in T. Training seed 87358.',
    mechanism2=[
        'So the stretch did what it is for, locally: wall slope 0.22 → 0.53, speed fit 0.29 → 0.81, attraction into walls 0.16 → 0.007 at ×12, and almost no wall crossings. What is wrong is the far-field value inside a quadrant, the same failure the plus-maze stalls had: the embedding gives the network the resolution to put the barrier there, not the global information that the way to a goal across the band starts by going away from it.',
    ],
    zp=[
        'One explanation would be the data: ntrl-demo\'s pairs are drawn in units of the whole maze, so at ×12 only 2.9 % of them have both ends in one quadrant. The cross-warp study\'s pairs drawn on its warped maze put nine times as many there (27 %), on the same maze, with the same queries; here they are trained exactly like ntrl-demo\'s (loss and planning on the real maze, no warp anywhere). '
        'They help a little at ×4 (ntrl-demo 32.0 → 42.3 %, the gates +2 to +6) and nothing at ×12 (ntrl-demo 26.9 %, the four embeddings 30.7–33.8 against 30.4–35.7 % with their own pairs); between-quadrant queries stay at 0–5 %. '
        'More pairs inside the quadrants give the field more of its local structure, not the way out: as in the cross-warp study, those pairs only paid once the loss itself was taken in the warped coordinates.',
    ],
    verdict=[
        '<b>The embedding map works on the widening gap, and it does what the C-space warps could not:</b> it adds resolution at the quadrants\' walls without taking anything from the open band, measured isotropic for the two fixed gates, and with octaves scaled to the cell beats ntrl-demo at every gap over three seeds. '
        'The three gates you asked for all work as maps: the environment\'s clearance gate and the boundary diffusion gate are equivalent up to a point at most gaps, and the gate learned from T rediscovers the same picture with no geometry given. '
        'Which gate barely matters until the octaves are high against the open space; there the clearance gate is the safe choice (×12: 33.3 % against 30.2 % always on, on every seed), the learned gate loses its isotropy, and the octaves themselves must scale with the cell, which is the rule the main study left open.',
        '<b>What it does not do</b> is solve the wide-gap maze: queries between quadrants stay at 0–5 % from ×8 on for every field. That failure is the far-field value of T inside a quadrant, not resolution and not the number of local pairs (both tested). The fields that pass it (85–93 % at ×12) were trained in a warped space where the band is short, i.e. with a different target; on the real maze it needs a non-local term (the midpoint / segment-DP losses) on top of this embedding.',
    ],
)
