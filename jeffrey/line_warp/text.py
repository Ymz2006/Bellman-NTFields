"""The prose of the findings page and of README.md. Every number comes from report/summary.json (blocks(S))."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATE = '2026-11-06'
TITLE = f'{DATE} Line-Loss Warp'
H1 = 'A warp learned from the obstacles alone'


def q(S, name):
    v = S['q1200'].get(name)
    t = list(v.values())
    return 100 * sum(x[0] for x in t) / sum(x[1] for x in t), [x[0] for x in t]


def gap(S, base, k):
    return S['gap'][f'{base}/{k}']


def ctl(S, base, k, lab):
    return S['gap'][f'{base}/{k}']['controls'][lab]['pct']


def seeds(xs):
    return ' / '.join(f'{x:.1f}' for x in xs)


def rr(v):
    return f'{v:+.2f}'.replace('-', '−')


def tile(k, v, s):
    return f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>'


def blocks(S):
    P, C = S['plus'], S['plus_controls']
    rec, dil = P['recipe'], P['dil']
    q_rec, q_dil = q(S, 'recipe without dilation'), q(S, 'the recipe')
    q_id, q_u4, q_pm, q_la = q(S, 'ntrl-demo'), q(S, 'uniform B x 4'), q(S, 'constructed x 6 (pm6)'), q(S, 'earlier learned warp (la6b)')
    q_nt, q_b7, q_b2 = q(S, 'recipe without TV'), q(S, 'bound 0.70, no TV'), q(S, 'bound 2, no TV')
    c12, m12 = gap(S, 'c22', 12), gap(S, 'm10', 12)
    A = S['ablation']
    G, L, FZ, Z12, H, SH = A['grid'], A['links'], A['fieldz'], A['zones12'], A['hits'], A['shape']
    fit = {f['label']: f for f in S['fits']}
    stripes = fit['free ResNet, axis lines, final code']
    g = lambda b, c, k='pct': G[b][c][k]
    pair = lambda c, k='pct', f='{:.1f}': f"{f.format(g('m10', c, k))} and {f.format(g('c22', c, k))}"
    b = {}
    b['TITLE'], b['H1'], b['DATE'] = TITLE, H1, DATE
    b['LEDE'] = (f"You proposed a Fourier network that learns a diffeomorphism from one rule: along any line in the warped space, the free and obstacle runs should be the same length. It works, under three conditions. "
                 f"Fitted to the obstacles alone, the warp at the widest gap (×12) solves <strong>{m12['pct']:.1f} %</strong> of runs on the 10×10 quadrants and <strong>{c12['pct']:.1f} %</strong> on the 22×22 quadrants. "
                 f"ntrl-demo gets {ctl(S, 'm10', 12, 'id'):.1f} and {ctl(S, 'c22', 12, 'id'):.1f} %. The hand-built warp gets {ctl(S, 'm10', 12, 'pmzw'):.1f} and {ctl(S, 'c22', 12, 'pmzw'):.1f} %, or {ctl(S, 'm10', 12, 'pmzs'):.1f} and {ctl(S, 'c22', 12, 'pmzs'):.1f} % with its band also shrunk by hand, so the learned warp finds that shrinking without being told where the band is. "
                 f"On the plain plus maze it is {q_dil[0] - q_id[0]:.0f} points above ntrl-demo, about level with a uniform stretch, and {q_pm[0] - q_dil[0]:.1f} below the hand-built warp. "
                 f"The map has to be limited, because left free the loss makes zebra stripes that the field cannot learn. Its density needs a total variation penalty. And at wide gaps the field has to be trained on the warped maze. The value of the loss does not predict planning (r = {rr(L['r_ratio'])}). How well the field learns the warped maze does.")
    b['TILES'] = ''.join([
        tile('gap ×12, 10×10 quadrants', f"{m12['pct']:.1f} %", f"3 seeds {seeds(m12['per_seed'])}. ntrl-demo {ctl(S, 'm10', 12, 'id'):.1f}, hand-built {ctl(S, 'm10', 12, 'pmzw'):.1f}, band shrunk by hand {ctl(S, 'm10', 12, 'pmzs'):.1f}"),
        tile('gap ×12, 22×22 quadrants', f"{c12['pct']:.1f} %", f"3 seeds {seeds(c12['per_seed'])}. ntrl-demo {ctl(S, 'c22', 12, 'id'):.1f}, hand-built {ctl(S, 'c22', 12, 'pmzw'):.1f}, band shrunk by hand {ctl(S, 'c22', 12, 'pmzs'):.1f}"),
        tile('plus maze, held-out 1,200 queries', f"{q_dil[0]:.1f} %", f"ntrl-demo {q_id[0]:.1f}, uniform B×4 {q_u4[0]:.1f}, hand-built {q_pm[0]:.1f}"),
        tile('same warp, trained the old way', f"{A['xtrained_c22']:.1f} %", f"Gap ×12, 22×22 quadrants, loss in x on the maze’s own pairs, one seed. The hand-built warp gets {ctl(S, 'c22', 12, 'pm'):.1f} and the earlier learned warp 3.1. The jump to {c12['pct']:.0f} % also needs the gap study’s training on the warped maze."),
    ])
    b['LOSSTEXT'] = ('<p>Take a warp z = φ(x) and any line in z. Each point on the line is free (0) or obstacle (1), so the line gives a step function, and the steps are runs of lengths ℓ₁ … ℓᵣ. '
                     'The loss is the difference between the longest and the shortest run, in units of that line’s mean run:</p>'
                     '<div class="eq">L = ( max ℓ − min ℓ ) / mean ℓ <span class="c">averaged over lines; dividing by the mean keeps φ = constant from winning</span></div>'
                     '<p>Each row below is one line through the plus maze. On the left, in the maze, the + is a single huge free run. On the right, after the warp, it is still the longest run, but the ratio of the largest to the smallest has about halved.</p>')
    b['STEPCAP'] = 'Free and obstacle along three rows of the plus maze at heading 0: in the maze (blue) and in the learned warp (red), with the graph’s length scaled to 1.'
    b['LOSSTEXT2'] = ('<p><strong>How it is computed.</strong> Nothing has to be inverted or smoothed. The breakpoints of the step function are where the line crosses the image of the obstacle boundary, so the boundary polylines go through φ once and are intersected with the line. '
                      'Each crossing is a closed-form function of φ’s output, so the gradient is exact: it matches finite differences to 3 × 10⁻⁸. The one exception is on purpose: a crossing where the line runs nearly parallel to the boundary keeps its position but drops its gradient. Obstacle space is the robot’s C-space obstacle at 9 headings, as exact polygons. They agreed with the maze’s clearance function at every one of 200,000 random positions at each of 4 headings, outside a shell of 0.001 cells around the boundary.</p>'
                      '<p>φ is fitted first, from the geometry alone, in about a minute. It is then frozen and ntrl-demo trains through it unchanged. The network, loss and planner are the earlier studies’.</p>')
    st_rows = [r for k in ('stripes_final', 'stripes_axis', 'stripes_axis4', 'stripes_any', 'stripes_any4') for r in P[k]['rows']]
    span = lambda k, f='{:.2f}': f"{f.format(min(r[k] for r in st_rows))}–{f.format(max(r[k] for r in st_rows))}"
    avg = lambda k: sum(r[k] for r in dil['rows']) / len(dil['rows'])
    st_lo, st_hi = min(r['pct'] for r in st_rows), max(r['pct'] for r in st_rows)
    b['STRIPETEXT'] = (f"<p>A free Fourier ResNet lowers the loss by turning the maze into stripes of nearly one wavelength. The largest / smallest run falls from {stripes['identity_ratio']:.0f} to {stripes['ratio']:.0f}, lower than any map that plans well, "
                       f"but planning drops to <strong>{st_lo:.0f}–{st_hi:.0f} %</strong>, against {C['ntrl-demo']['pct']:.0f} % with no warp at all. That range covers five fields: axis and any-angle lines, quadrant stretch 4 and 6, and a rerun on the final code.</p>")
    b['GALLERYCAP'] = 'The same maze under six fitted warps, at heading 0. Top: residual networks. Bottom: maps that cannot shear far.'
    Z = A['z1']
    fp = FZ['plus']
    b['STRIPETEXT2'] = (f"<p><strong>Why stripes win the loss.</strong> Runs are compared only within a line. A line that runs along a stripe holds one long run, has nothing to compare, and costs nothing, while a line across the stripes sees a single wavelength. So stretching walls into long thin stripes is nearly free. It takes singular values from {stripes['sv'][0]:.3f} to {stripes['sv'][1]:.0f}.</p>"
                        "<p>The stripes are a stubborn attractor. They survived lines at any angle, a comparison of runs across lines, and a shear penalty that took the rotation from 24° to 1° (the stripes just lined up with an axis). A free triangular map found them too. Only a limit on the map itself removed them.</p>"
                        f"<p><strong>Why they plan badly: the field cannot learn the maze through them.</strong> Trained through the stripes, the field’s speed correlates with the true speed at only {span('speed_corr')}, against {avg('speed_corr'):.2f} for the recipe. At a wall it predicts a speed of {span('speed_wall')}, where the recipe’s field predicts {avg('speed_wall'):.2f}, and a long path has {span('bumps', '{:.0f}')} local minima against {avg('bumps'):.1f}. "
                        f"Training on the warped maze does not rescue it: the correlation, measured in z, is {fp['stripes']['corr']:.2f} against {fp['recipe']['corr']:.2f}, descent leads into the wall at {100 * fp['stripes']['into']:.0f} % of slow-zone poses against {100 * fp['recipe']['into']:.0f} %, and {Z['stripes']:.1f} % of runs succeed against {gap(S, 'c22', 1)['pct']:.1f} %.</p>")
    b['STRIPETEXT3'] = (f"<p><strong>A lower loss is not a better map.</strong> The first chart is the loss against planning on the plus maze: no relation across {len(L['loss'])} fits (r = {rr(L['r_ratio'])}). The second is what the field learned, over {len(L['fields'])} fields including the controls: the more local minima along long paths, the fewer runs solved (r = {rr(L['r_bumps'])}, or {rr(L['r_bumps_sep'])} within the {L['separable']} separable fields). "
                        f"Speed agrees (r = {rr(L['r_speed'])} with the correlation between learned and true log speed), and so does the direction of descent (r = {rr(L['r_cos'])}). Fields trained on the warped maze show the same with speed measured in z: r = {rr(L['r_z1'])} over {len(L['z1_points'])} fields at ×1, and {rr(L['r_z12_m10'])} and {rr(L['r_z12_c22'])} over the fits at ×12 on the two mazes. "
                        "The loss asks for even runs. Planning needs a maze the field can learn. Those are different goals, and the rest of the work was closing the difference.</p>")
    b['FIXTEXT'] = '<p>Four changes, each from a measured failure.</p>'
    b['DENSCAP'] = 'Stretch along x, scaled to × 6 in the quadrants. Dashed: the hand-built warp. The bottom panel’s flat density with sharp edges came out of the line loss with no knowledge of the band.'
    b['FIXTEXT2'] = (f"<p><strong>1. Limit the map.</strong> A separable map (F(x), G(y)), or a triangular one, cannot shear far, and its Jacobian is positive by construction, so it cannot fold. Its log-density is the Fourier network. This removes the stripes.</p>"
                     f"<p><strong>2. Penalize the total variation of the log-density.</strong> The loss needs the log-density to fall from a quadrant level to a band level. Any range beyond that is slack, and the fit spends slack on spikes and tilts that the loss does not charge for (top panel). "
                     f"Total variation charges for every rise and fall, so only the two levels the loss pays for remain. A plain bound can do the same job, but only when it is tight (second chart): {P['sep_h070']['pct']:.1f} % at 0.70 and {P['sep_h200']['pct']:.1f} % at 2.00, three seeds each. The loose bounds lose runs to stalls, not to walls.</p>")
    rng = lambda ks: ' / '.join(f"{SH[k]['range']:.2f}" for k in ks)
    k22, k10 = ['c22k1d', 'c22k2d', 'c22k4d', 'c22k8d', 'c22k12dil'], ['m10k1d', 'm10k2d', 'm10k4d', 'm10k8d', 'm10k12d']
    b['FIXTEXT3'] = (f"<p>The reason to prefer total variation is that it finds the level by itself. With one weight, 0.1, the log-density range grows with the gap: {rng(k22)} at ×1 / ×2 / ×4 / ×8 / ×12 on the 22×22 quadrants and {rng(k10)} on the 10×10 quadrants, with about two steps per axis every time. A bound does not adapt like that. "
                     f"On the plus maze the weight hardly matters above 0.05, because the shape of the map hardly matters there: at 0.40 the map is almost uniform (band / quadrant {A['variation']['tv40']['band_over_quadrant']:.2f}) and still solves {P['tv40']['pct']:.1f} %.</p>"
                     f"<p>With the loss in x, slack hurts through the field’s slope. That slope in z is (1/S)/F′, so every wiggle in F′ becomes a wiggle in the field: bound 1.25 has {P['sep_h125']['rows'][0]['bumps']:.2f} local minima per long path against {rec['rows'][0]['bumps']:.2f} with total variation, and {P['sep_h125']['rows'][0]['not_arrived']} of 720 runs never arrive while only {P['sep_h125']['rows'][0]['arrived_wall']} hit a wall. A field trained on the warped maze has no such term, and it fails differently (next section).</p>"
                     f"<p><strong>3. Draw lines through the boundary.</strong> Total variation is an L1 penalty, so at the identity it can absorb a weak gradient completely. On the emptiest maze (10×10 quadrants, ×12) lines drawn uniformly rarely met a wall, and the map never moved: a run that looked like a failure of the method ({A['stuck']:.1f} %) was just a uniform stretch. Lines through random boundary points always have runs to compare, and that fixed it. Starting the penalty late did not, because wiggles that form early stay. A fit that does not move now stops with an error.</p>"
                     f"<p><strong>4. Do not compress next to a wall.</strong> The loss treats a free run as one thing, so compression runs right up to the wall. A dilation of the log-density by one cell moves it away. At ×12 this takes total variation alone from {pair('tv')} % to {pair('both')} % (10×10 and 22×22 quadrants), with no overlap between seeds. "
                     f"Without it, on the 22×22 quadrants the wall’s slow zone on the band side is {A['zone_nodil']:.4f} z-units wide, {A['zone_nodil_share']:.0f} % of its width inside a quadrant and {A['step'] / A['zone_nodil']:.0f} × smaller than one planner step. With it, the zone is {A['zone_dil']:.4f} wide, twice a step. "
                     f"Queries that must cross that wall gain most (quadrant to open: {A['qo_nodil']:.0f} → {A['qo_dil']:.0f} %, one seed). Widths of 0 / 0.5 / 1 / 2 cells gave {A['width'][0]:.1f} / {A['width'][1]:.1f} / {A['width'][2]:.1f} / {A['width'][3]:.1f} % on one seed, so about a cell, within noise.</p>"
                     f"<p><strong>Also: use the C-space obstacle.</strong> With workspace walls as the obstacle the warp compresses too wide a span (x ≈ 6–17 and y ≈ 5–15, where the band is 8–14). {A['walls_text']} The same seed gets {P['recipe']['rows'][0]['pct']:.1f} % with the C-space obstacle.</p>"
                     f"<p><strong>Tried and dropped.</strong> A third level on the step function (a contour one cell inside free space, so that a wide run splits into edge zones and an interior) did not produce the same effect: the density shows no step inside the band, because a 1-cell edge zone is already longer than a typical run. Fit seeds do not matter: {S['fitseeds']}.</p>")
    b['LOOTEXT'] = ('<p>Each part taken away in turn, with three training seeds. The green row is a control: total variation replaced by a plain bound, set at the range total variation reached on that maze (1.775 and 1.875 at ×12, with dilation; 0.70 on the plus maze, without).</p>')
    d_tv = (g('m10', 'both') - g('m10', 'dil'), g('c22', 'both') - g('c22', 'dil'))
    d_dil = (g('m10', 'both') - g('m10', 'tv'), g('c22', 'both') - g('c22', 'tv'))
    bd = lambda base: ' / '.join(f"{x['pct']:.1f}" for x in A['bounds'][base])
    b['LOOTEXT2'] = (f"<p>At ×12 total variation is worth +{d_tv[0]:.1f} and +{d_tv[1]:.1f} points on top of dilation, and dilation +{d_dil[0]:.1f} and +{d_dil[1]:.1f} on top of total variation. On the plus maze total variation is again the part that matters: without it the field solves {q_nt[0]:.1f} % of the held-out queries, below ntrl-demo’s {q_id[0]:.1f}.</p>"
                     f"<p>The control shows what total variation is doing. A plain bound at the same range plans about as well ({pair('range')} %, against {pair('both')} %), so the penalty’s effect here is the range it sets. The chart sweeps that bound at ×12, at 0.70 / 1.25 / total variation’s range / 2.5 / 3: {bd('m10')} % on the 10×10 quadrants and {bd('c22')} % on the 22×22. So at one gap a bound has a window about a factor of two wide, and the log-density starts to tilt past its top end. "
                     f"None of the bounds I tried serves both ends: at ×1, trained on the warped maze and without dilation, 1.25 gets {Z['wiggly']:.1f} % and 0.70 gets {Z['tight']:.1f} %, against {Z['tv_nodil']:.1f} % with total variation, and 0.70 is far too tight at ×12. I did not try values between the two. One total variation weight served every gap and both mazes, with no knowledge of the band.</p>")
    b['FAILTEXT'] = '<p><strong>How each one fails.</strong> The success rate hides two opposite mistakes.</p>'
    hit = lambda base, c: H[base][c]
    n_hit = lambda base, c: sum(hit(base, c)[k] for k in ('last_hop', 'jump', 'walked_in'))
    outside = lambda base, c: round(n_hit(base, c) * (1 - hit(base, c)['contact_in_quadrant']))
    b['FAILTEXT1'] = (f"<p><strong>Too little range: runs stall.</strong> With a bound of 0.70 the band stays wide (band / quadrant {g('c22', 'tight', 'band_q'):.2f}), the box in z is about twice the recipe’s, and {pair('tight', 'stall', '{:.0f}')} % of runs never arrive (10×10 and 22×22 quadrants). This is the wide band that broke ntrl-demo in the gap study.</p>"
                      f"<p><strong>Too much range: runs walk into walls.</strong> With a bound of 3 and no total variation the density uses its slack to tilt across each quadrant (figure): it moves by {pair('dil', 'tilt', '{:.2f}')} in log-density, against {pair('both', 'tilt', '{:.2f}')} for the recipe. The map is scaled so the quadrant’s median stretch is m, so much of a quadrant sits well below m, and that is where runs hit walls: the median contact is at {hit('m10', 'dil')['stretch_at_contact_median']:.2f} and {hit('c22', 'dil')['stretch_at_contact_median']:.2f} m, against {hit('m10', 'both')['stretch_at_contact_median']:.2f} and {hit('c22', 'both')['stretch_at_contact_median']:.2f} m for the recipe.</p>")
    b['TILTCAP'] = 'Stretch across the first quadrant and the start of the band at gap ×12, as a multiple of the quadrant’s median stretch: a flat line at 1 is an even quadrant. Without total variation the slack goes into a tilt.'
    bw = lambda base, c: FZ[base][c]['by_width']
    pc = lambda v: f'{100 * v:.1f}'
    st = A['steps']
    sw = lambda base, c, f: st[base][c][f]['wall']
    b['FAILTEXT2'] = (f"<p>There the wall’s slow zone is thin: under two planner steps at {100 * g('m10', 'dil', 'two_in'):.0f} and {100 * g('c22', 'dil', 'two_in'):.0f} % of near-wall poses inside the quadrants, and at none of them for the recipe. (I chose two steps after seeing this. One step, which I tried first, missed it.) A thin zone costs twice.</p>"
                      '<ul class="tight">'
                      f"<li><strong>The field resolves it less well.</strong> Within one field, descent points into the wall at {pc(bw('m10', 'dil')['under 2']['into'])} % of slow-zone poses where the zone is under two steps, and {pc(bw('m10', 'dil')['over 3']['into'])} % where it is over three. On the 22×22 quadrants it is {pc(bw('c22', 'dil')['under 2']['into'])} and {pc(bw('c22', 'dil')['over 3']['into'])} %. Averaged over all poses the field looks fine (speed correlation {g('m10', 'dil', 'corr_z'):.2f} in z on the 10×10 quadrants, against {g('m10', 'both', 'corr_z'):.2f} for the recipe), so only this split shows it.</li>"
                      f"<li><strong>The planner gets less warning.</strong> Nearly every hit is an ordinary step landing inside the obstacle: {hit('m10', 'dil')['walked_in']} of {n_hit('m10', 'dil')}, with {hit('m10', 'dil')['jump']} jumps across a wall. On the same frozen 10×10 field, a quarter of the step cuts hits from {sw('m10', 'dil', '1.0')} to {sw('m10', 'dil', '0.25')} of 300, and from {sw('m10', 'tv', '1.0')} to {sw('m10', 'tv', '0.25')} with total variation alone, while the recipe stays at {sw('m10', 'both', '0.25')}. On the 22×22 quadrants the test shows little ({sw('c22', 'dil', '1.0')} → {sw('c22', 'dil', '0.25')}, {sw('c22', 'tv', '1.0')} → {sw('c22', 'tv', '0.25')}, and {sw('c22', 'both', '1.0')} → {sw('c22', 'both', '0.25')} for the recipe), so there the step is not the main limit.</li>"
                      '</ul>'
                      f"<p>The same happens next to the band without dilation, which is why dilation removes contacts outside the quadrants ({outside('m10', 'tv')} → {outside('m10', 'both')} on the 10×10 quadrants and {outside('c22', 'tv')} → {outside('c22', 'both')} on the 22×22, one seed).</p>"
                      f"<p>The 10×10 fit with neither part went wrong in its own way: it barely compressed (band / quadrant {g('m10', 'none', 'band_q'):.2f}), its box in z is {g('m10', 'none', 's') / g('m10', 'both', 's'):.1f} × the recipe’s, and its field did not learn the maze (speed correlation {g('m10', 'none', 'corr_z'):.2f} in z). The 22×22 quadrants still lose runs after all of this: {100 * g('c22', 'both', 'wall') / (100 - g('c22', 'both', 'stall')):.0f} % of arrivals hit a wall, mostly inside a quadrant.</p>")
    b['PLUSTEXT'] = (f"<p>Fields trained through the frozen warp at the same quadrant stretch as the hand-built warp (× 6), with the loss in x, on ntrl-demo’s own pairs and MPPI. The bars span the three training seeds. I chose the settings on the 240 queries, so the 1,200 are the fair comparison.</p>")
    b['PLUSTEXT2'] = (f"<p><strong>The learned warp is a clear gain over ntrl-demo ({q_id[0]:.1f} → {q_dil[0]:.1f} % held out), and no better than the simpler choices.</strong> Uniform B×4 gets {q_u4[0]:.1f}, the earlier learned warp {q_la[0]:.1f}, and the hand-built warp is still ahead at {q_pm[0]:.1f}. Most of the gain here is the stretch itself: the band is only a quarter of the width, so there is little to compress. "
                      f"On the gap study’s harder ×1 query set the same fields solve {A['region_x']:.1f} %, against {ctl(S, 'c22', 1, 'pm'):.1f} % for the hand-built warp.</p>"
                      f"<p>A tight bound is level with total variation on this maze ({P['sep_h070']['pct']:.1f} and {q_b7[0]:.1f} % at 0.70, against {dil['pct']:.1f} and {q_dil[0]:.1f} % for the recipe), because here it sets almost the same range ({SH['sep070']['range']:.2f} against {SH['dil']['range']:.2f}). The two query sets rank them differently, so I would not separate them. "
                      f"Dilation’s effect is small here and not settled: {q_rec[0]:.1f} → {q_dil[0]:.1f} % on the 1,200 queries (all three seeds up, 304 queries gained, 210 lost) but {rec['pct']:.1f} → {dil['pct']:.1f} % on the 240. Trained on the warped maze at ×1 it changes nothing ({Z['tv_nodil']:.1f} → {Z['recipe']:.1f} %). Its effect grows with the compression, which is why it shows at ×12.</p>")
    b['GAPTEXT'] = (f"<p>The gap mazes push the four quadrants apart with an open band of 6k cells. A wider band is what a warp should be good at, and it is what broke every earlier learned warp. The fit is the same at every gap, with the same settings. The field is trained on the warped maze, with pairs drawn on it and MPPI run in it, the setup the gap study found necessary. "
                    f"That setup is needed here too: with the loss in x at ×12 the same warp solves {A['xtrained_c22']:.1f} % (one seed), level with the hand-built warp’s {ctl(S, 'c22', 12, 'pm'):.1f} %.</p>")
    b['GAPSCAP'] = 'The real maze and its learned warp at three gaps. At ×12 the band is 72 cells wide; the warp shrinks it to a narrow cross and the four quadrants fill the box.'
    b['GAPTEXT2'] = S['gap_text']
    k22c = c12['by_category']
    b['GAPTEXT3'] = (f"<p>At ×12 the gain is in the queries that cross between quadrants, which are the hardest. {S['cat_text']}</p>"
                     f"<p>The failures left on the 22×22 quadrants are mostly those long routes through two 8 × 8 mazes: adjacent {k22c['adjacent quadrants']:.0f} %, diagonal {k22c['diagonal quadrants']:.0f} %. Every field here is weakest on them, and I did not test why.</p>")
    b['ARCHTEXT'] = S['arch_text']
    b['STARCAP'] = 'The residual network on the ×4 maze: a valid diffeomorphism (det J > 0 on the whole grid, outline a simple curve) with the + collapsed to a hub and the quadrants fanned out. At ×12 it did not stay valid.'
    three = lambda base: ', '.join(f'×{k}' for k in (1, 2, 4, 8, 12) if gap(S, base, k)['controls']['pmzw']['seeds'] > 1)
    b['LIMITS'] = ('<ul class="tight">'
                   '<li><strong>The best map is separable, and the plus is axis-aligned.</strong> A band at an angle, or rooms that do not line up, would need the triangular map or something else. The study does not show this recipe on such a scene, and your note that learned warps will be messy still stands.</li>'
                   '<li><strong>The ResNet you asked me to start with did not win.</strong> It gives valid maps at ×1 and ×4 but plans worse than the separable map at both, and I could not get a valid map from it at ×12 in two attempts.</li>'
                   '<li><strong>The scale is not learned.</strong> The quadrant stretch (6 E / 22.1) is copied from the hand-built warp. The fit supplies the shape; the size comes from that study.</li>'
                   f"<li><strong>The wide-gap result needs training on the warped maze.</strong> With the loss in x the same warp gets {A['xtrained_c22']:.1f} % at ×12 on one seed, like the hand-built warp.</li>"
                   '<li><strong>Travel times are in warped units.</strong> Fields trained on the warped maze give good paths, but T is the warped maze’s time, not the real one. The anisotropic eikonal study’s correction was not applied here.</li>'
                   '<li><strong>Settings came from these mazes.</strong> TV 0.1 and the one-cell dilation were set here. The 1,200 queries are held out only from the TV choice; I read the dilation result on them too.</li>'
                   '<li><strong>Heading is a constant factor.</strong> The loss runs on 9 heading slices, but the warp acts on (x, y) only.</li>'
                   f"<li><strong>Few seeds in places.</strong> One maze per base. A cell’s three seeds vary the field and share one fit, so a fit that went wrong, like the 10×10 one with neither part, is one draw. Ablation rows are single runs unless they say otherwise, and two trainings of one setting can differ by about 5 points. Hand-built controls have three seeds at {three('c22')} on the 22×22 quadrants and {three('m10')} on the 10×10 quadrants, one elsewhere; the band shrunk by hand has one.</li>"
                   '<li><strong>Correlations are not causes.</strong> The link from local minima and learned speed to planning is measured across fields, and only the bound sweep and the step test change one thing at a time. I did not find which limit of the field makes stripes unlearnable.</li>'
                   '</ul>')
    b['SETUP'] = ('<p><strong>Stage 1, fit.</strong> <code>fitrun.py</code>: the recipe is <code>--arch sep --kind hard --tv-weight 0.1 --dilate 1 --hmax 2</code> (3 on the gap mazes), 1,500 steps, lines through boundary points, 128 lines per axis on 3 of 9 heading slices per step. '
                  '<strong>Stage 2, field.</strong> <code>pipeline.py</code>: ntrl-demo through cross_warp’s trainer, 5,000 epochs, quadrant stretch <code>--m</code> 6 E / 22.1. '
                  'Gap mazes also use <code>zdraw.py</code> (pairs drawn on the warped maze) and <code>--z-space</code>, then <code>zq_net.py</code> for MPPI in z. Every run went through <code>run.py</code> and the registry. '
                  'Measurements on those runs: <code>mechanism.py</code> and <code>wallhits.py</code>. A z-trained field’s own <code>speed_fit</code> is in x units and scales with the local stretch, so <code>wallhits.py field</code> measures speed in z.</p>')
    return b


def main():
    S = json.load(open(HERE / 'report/summary.json'))
    print(len(blocks(S)))


if __name__ == '__main__':
    main()
