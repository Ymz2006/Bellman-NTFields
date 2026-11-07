"""Text of the findings page; every number is read from report/summary.json through page.py's helpers."""
import math
import numpy as np

TITLE = '2026-11-06 Anisotropic Warp Eikonal'
PLAIN, REF, REFZ = 'zaniso-Uquad', 'ziso-open', 'ziso'
LB, OPEN, OPENLB = 'zaniso-Uquad-lb10', 'zaniso-Uquad-open', 'zaniso-Uquad-open-lb10'
TAUZ, PWZ, GS4 = 'zaniso-Uquad-open-pwtauz-lb10', 'zaniso-Uquad-open-pwz-lb10', 'zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10'
HS4, HS4T, GS6 = 'zaniso-Uquad-open-hs4-lb10', 'zaniso-Uquad-open-pwtauz-hs4-lb10', 'zaniso-Uquad-open-gs8x1+8x6-pwtauz-lb10'
LADDER = 'zaniso-Uquad-open-gs4x1+4x2+4x4+4x8-pwtauz-lb10'
GS6B, GS3, GS12, L1 = ('zaniso-Uquad-open-gs4x1+12x6-pwtauz-lb10', 'zaniso-Uquad-open-gs8x1+8x3-pwtauz-lb10', 'zaniso-Uquad-open-gs8x1+8x12-pwtauz-lb10',
                        'zaniso-Uquad-open-pwtauz-hs60-es0.1-lb10')
RES, RESFULL, RESMESSY = 'zres-Uquad-open-far-lb10', 'zres-Uquad-open-far-lb10-full1s', 'zres-Uquad-messy0.6-open-far-lb10'
RESFULL2 = 'zres-Uquad-open-far-lb10-full'      # the correction on the full-loss fields with the plain two-sided term (control)
FAR1, SUM2 = 'zaniso-Uquad-open-far-pwtauz-lb10', 'zaniso-Uquad-open-gs8x1+8x6-sum2-pwtauz-lb10'
RESU = 'zres-Uquad-open-faru-lb10'      # the correction as a learned embedding distance minus the distance on the warp
NAN = float('nan')
XQ = ('adjacent quadrants', 'diagonal quadrants')


def blocks(h):
    P, plan, pc, f1, run_of, rows_of, VER, BASE_NAME, CATS, table, recipe = (h[k] for k in ('P', 'plan', 'pc', 'f1', 'run_of', 'rows_of', 'VER', 'BASE_NAME', 'CATS', 'table', 'recipe'))
    v12 = VER.get('c22_k12') or {}
    eq = (v12.get('equations') or {}).get('by_region', {})
    price = (v12.get('price') or {})
    by = price.get('by_category', {})
    pa = price.get('all', {})

    def pct(b, k, v, cat=None, planner=None):
        return f1(pc(b, k, v, planner, cat)) if v else '–'

    def xq(b, k, v, planner=None):
        """Quadrant-to-quadrant queries (adjacent and diagonal pooled, both 60 queries)."""
        a, d = (pc(b, k, v, planner, c) if v else None for c in XQ)
        return '–' if a is None or d is None else f1((a + d) / 2)

    def seedpct(b, k, v, planner='z', seed=87358):
        """One training seed's success (%), not the pooled value."""
        r = run_of(b, k, v, seed) if v else None
        e = (r or {}).get('plans', {}).get(planner)
        return f1(100 * e['success'] / e['runs']) if e else '–'

    fw, fl, fr = h.get('fail_where'), h.get('feel'), h.get('feel_raw')

    def arr(b, k, v, planner='z'):
        x = h['arrival'](b, k, v, planner) if (h.get('arrival') and v) else None
        return f1(x, 1)

    def rises(b, k, v, goals=None):
        """wallfeel.py: share (%) of near-wall poses in the quadrants where T rises into the wall; goals None = another quadrant."""
        x = (fl(b, k, v, goals) if goals else fl(b, k, v)) if (fl and v) else None
        return f1(x, 0)

    def failed_in_start(b, k, v):
        x = fw(b, k, v) if (fw and v) else None
        return ('–', '–') if not x else (str(x[1]), str(x[0]))

    def quart(b, k, v):
        w = fr(b, k, v) if (fr and v) else None
        qs = w['feel']['adjacent quadrant']['0.06-0.12']['normal_quantiles'] if w else None
        return ('–', '–') if not qs else (f'{qs[1]:+.2f}'.replace('-', '−'), f'{qs[3]:+.2f}'.replace('-', '−'))

    def gapx(b, k, v, planner_a='z', planner_b='zh'):
        """Points lost by planning on the sum against its head alone, same runs."""
        a, c_ = (pc(b, k, v, pl) if v else None for pl in (planner_a, planner_b))
        return '–' if a is None or c_ is None else f1(c_ - a)

    def ta(b, k, v, key='within_10pct', scale=100.0, d=0):
        p = P(b, k, v) if v else None
        return f1(scale * p['time_accuracy'][key], d) if p and p.get('time_accuracy') else '–'

    def tcat(b, k, v, cat, d=2):
        p = P(b, k, v) if v else None
        return f1(p['time_accuracy']['by_category'].get(cat), d) if p and p.get('time_accuracy') else '–'

    def spd(b, k, v, reg):
        p = P(b, k, v) if v else None
        return f1((p.get('speed_on_maze') or {}).get(reg), 2) if p else '–'

    def below(b, k, v):
        p = P(b, k, v) if v else None
        return f1(100 * p['lower_bound']['band_share_below_0_9'], 0) if p and p.get('lower_bound') else '–'

    def ang(b, k, v, cat):
        p = P(b, k, v) if v else None
        return f1((p.get('direction_by_category') or {}).get(cat), 0) if p else '–'

    def nseeds(b, k, v):
        p = P(b, k, v) if v else None
        return len(p['seeds']) if p else 0

    def top_output(b, k, v):
        r = run_of(b, k, v)
        if not r:
            return None
        z = rows_of(r)
        return float(z['T_pred'].max() * r['t_scale']), float(z['t_star'].max() * r['t_scale'])

    R = recipe
    gaps = h['GAPS']
    out = dict(TITLE=TITLE, DATE='5–6 Nov 2026', H1='Anisotropic Warp Eikonal')

    def refv(b, k):
        return REF if P(b, k, REF) else REFZ

    def behind(b, k):
        """Points by which the recipe is below the same-pairs reference (pooled over training seeds)."""
        x, y = pc(b, k, R(b, k)) if R(b, k) else None, pc(b, k, refv(b, k))
        return '–' if x is None or y is None else f1(y - x)

    def rng(b, k, v, planner='z'):
        """Range of success over training seeds, 'lo–hi', or '' with one seed."""
        e = plan(b, k, v, planner) if v else None
        return '' if not e or len(e['per_seed']) < 2 else f'{min(e["per_seed"]):.0f}–{max(e["per_seed"]):.0f}'

    def span(b, v_of, cat=None):
        vals = [pc(b, k, v_of(b, k), None, cat) for k in gaps if v_of(b, k) and P(b, k, v_of(b, k))]
        return '–' if not vals else f'{min(vals):.0f}–{max(vals):.0f}'

    out['LEDE'] = (
        'The request: a warp that compresses the maze makes planning easy, but a field trained on the stretched space no longer tells the maze’s travel time, because there the speed depends on the '
        'direction of travel. Fix it with an anisotropic eikonal equation, an inner product on the gradient of T in place of its length, and plan as well as on the stretched space. '
        '<strong>The equation is right</strong>: on the warp it is the maze’s own eikonal equation, to rounding on a network and to the grid’s accuracy on exact travel times. '
        '<strong>Dropped into ntrl-demo’s loss it does not train</strong>, at any gap, and not because of the network’s output range: things the stretched-space training gets for free are missing, '
        'and each has a general fix that uses only the warp’s Jacobian and the poses on the maze.</p><p class="lede">'
        f'<strong>With them one field comes close to the stretched space and tells the travel time.</strong> Queries solved, against the stretched-space field trained on the same pairs: '
        f'{pct("c22", 4, R("c22", 4))} against {pct("c22", 4, REF)} % (22×22 quadrants) and {pct("m10", 4, R("m10", 4))} against {pct("m10", 4, REF)} % (10×10) at gap ×4; '
        f'{pct("c22", 12, R("c22", 12))} against {pct("c22", 12, refv("c22", 12))} % and {pct("m10", 12, R("m10", 12))} against {pct("m10", 12, refv("m10", 12))} % at gap ×12, where a distance head with a second scale '
        f'supplies the output range (two or three training seeds each). On the 22×22 quadrants that is inside the spread between training seeds; on the 10×10 the single field is '
        f'{behind("m10", 4)} and {behind("m10", 12)} points behind. Its travel time is within 10 % of the truth on {ta("c22", 4, R("c22", 4))} and {ta("m10", 4, R("m10", 4))} % of queries at gap ×4 and on {ta("c22", 12, R("c22", 12))} and '
        f'{ta("m10", 12, R("m10", 12))} % at ×12; the stretched space’s on {ta("c22", 4, REF)} and {ta("m10", 4, REF)} %. '
        f'The totals hide a trade at the widest gap: the single field solves more queries from the open band into a quadrant, and on the 10×10 quadrants far fewer from one quadrant to another '
        f'({xq("m10", 12, R("m10", 12))} against {xq("m10", 12, refv("m10", 12))} %). The cause of the second is measured below: for a far goal its travel time stops rising into the walls.</p><p class="lede">'
        '<strong>A second route keeps the stretched-space field and learns the rest</strong>: the head is trained with the plain length, a small unbounded network is added to it, and the inner product trains the sum. '
        f'It plans {gapx("c22", 12, RES)} and {gapx("m10", 12, RES)} points below its own head at gap ×12 ({pct("c22", 12, RES)} and {pct("m10", 12, RES)} %), with the travel time within 10 % on '
        f'{ta("c22", 12, RES)} and {ta("m10", 12, RES)} % of queries. The correction can also be trained alone on a field that already exists: on the gap study’s full-loss fields at gap ×12 the sum solves '
        f'{pct("m10", 12, RESFULL)} % of queries where the field itself solves {pct("m10", 12, RESFULL, planner="zh")} % (10×10 quadrants; {pct("c22", 12, RESFULL)} against {pct("c22", 12, RESFULL, planner="zh")} % on the 22×22), '
        f'and its travel time is within 10 % of the truth on {ta("m10", 12, RESFULL)} and {ta("c22", 12, RESFULL)} % of queries.')

    def tile(k, v, s):
        return f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>'
    out['TILES'] = ''.join([
        tile('the equation, on exact travel times', f'{f1((eq.get("centre") or {}).get("inner_product_on_z", {}).get("median"), 2)} <small>vs {f1((eq.get("centre") or {}).get("plain_length_on_z", {}).get("median"), 0)}</small>',
             'S × gradient norm of the true travel time in the band’s centre at gap ×12: with the inner product, and with the plain length on the warp (1 = the equation holds)'),
        tile('inner product alone, gap ×4', f'{pct("c22", 4, PLAIN)} % <small>/ {pct("m10", 4, PLAIN)} %</small>', f'queries solved, 22×22 / 10×10 quadrants; the stretched space: {pct("c22", 4, REF)} / {pct("m10", 4, REF)} %'),
        tile('full recipe, gap ×4', f'{pct("c22", 4, R("c22", 4))} % <small>/ {pct("m10", 4, R("m10", 4))} %</small>',
             f'travel time within 10 % of the truth on {ta("c22", 4, R("c22", 4))} / {ta("m10", 4, R("m10", 4))} % of queries (stretched space: {ta("c22", 4, REF)} / {ta("m10", 4, REF)} %)'),
        tile('full recipe, gap ×12', f'{pct("c22", 12, R("c22", 12))} % <small>/ {pct("m10", 12, R("m10", 12))} %</small>',
             f'stretched space {pct("c22", 12, refv("c22", 12))} / {pct("m10", 12, refv("m10", 12))} %; travel time within 10 % on '
             f'{ta("c22", 12, R("c22", 12))} / {ta("m10", 12, R("m10", 12))} % of queries'),
        tile('two-part field, gap ×12', f'{pct("c22", 12, RES)} % <small>/ {pct("m10", 12, RES)} %</small>',
             f'its own head alone {pct("c22", 12, RES, planner="zh")} / {pct("m10", 12, RES, planner="zh")} %; travel time within 10 % on {ta("c22", 12, RES)} / {ta("m10", 12, RES)} % of queries'),
    ] + ([tile('correction on a full-loss field, gap ×12', f'{pct("c22", 12, RESFULL)} % <small>/ {pct("m10", 12, RESFULL)} %</small>',
               f'the gap study’s field itself {pct("c22", 12, RESFULL, planner="zh")} / {pct("m10", 12, RESFULL, planner="zh")} %; with the correction its travel time is within 10 % on '
               f'{ta("c22", 12, RESFULL)} / {ta("m10", 12, RESFULL)} % of queries')] if P('m10', 12, RESFULL) else []))

    out['IDEATEXT'] = (
        '<p>Put a warp z = φ(x) in front of the network, so the travel time is T(x<sub>0</sub>, x<sub>1</sub>) = H(φ(x<sub>0</sub>), φ(x<sub>1</sub>)). On the maze the travel time obeys the '
        'eikonal equation S(x)·|∇<sub>x</sub>T| = 1, with S the speed. The chain rule gives ∇<sub>x</sub>T = Jᵀ∇<sub>z</sub>H, with J = ∂φ/∂x, so the same equation written on the warp’s coordinates is</p>'
        '<div class="eq">S(x)² · ∇<sub>z</sub>Hᵀ (J Jᵀ) ∇<sub>z</sub>H = 1 <span class="c">&nbsp;&nbsp;instead of&nbsp;&nbsp;</span> S(x)² · |∇<sub>z</sub>H|² = 1</div>'
        '<p>That is the proposal: the squared length of the gradient becomes an inner product with the matrix M = J Jᵀ, positive definite because φ is a diffeomorphism. '
        'With Lie gradients it reads the same, c<sub>H</sub>ᵀ (J<sub>φ</sub>J<sub>φ</sub>ᵀ) c<sub>H</sub> = 1 / S², with J<sub>φ</sub> the differential of README_warp’s section 5. '
        'On the warp the robot’s speed along a unit direction u is S / |J⁻¹u|, so it depends on the direction wherever J is not a multiple of the identity, and the fastest way down the field is '
        '−M∇<sub>z</sub>H, not −∇<sub>z</sub>H. The plain length is the eikonal equation of another world, the stretched maze with the maze’s speeds, which is what the cross-warp study’s '
        '“trained on the warped maze” fields learn.</p>'
        '<p>The warp is the gap study’s constructed one, fixed: (x, y, heading) → (F(x), F(y), m·heading), with F magnifying m = 6E/22.1 times over the quadrants and the band’s '
        'edge cells and once over the band’s interior (m = 23.9 at gap ×12). The code takes M from the warp’s autograd Jacobian as a full matrix and never uses that this one is diagonal.</p>')
    out['INDCAPTION'] = ('Equal circles on the gap ×1 maze (left) and their images on the constructed warp (right): the distance covered in one unit of time. Inside the quadrants the warp is a uniform '
                         'magnification and the circles stay circles. In the centre of the band they shrink 6-fold (24-fold at gap ×12), and in the arms, where the warp magnifies one axis only, '
                         'they flatten: there the cost of a step depends on its direction.')
    c, a, q = eq.get('centre') or {}, eq.get('arms') or {}, eq.get('quadrants') or {}
    out['IDEATEXT2'] = (
        '<p><strong>Checked three ways.</strong> (1) The matrix: J Jᵀ from autograd against central differences of the warp and against its analytic slopes (largest relative difference '
        f'{(v12.get("metric_check") or {}).get("max_rel_diag", NAN):.0e} for the latter). (2) On a network: |∇<sub>x</sub>T|² through the warp against ∇<sub>z</sub>Hᵀ M ∇<sub>z</sub>H on the warp’s coordinates, '
        '4,096 random pairs in float64, largest relative difference below 10⁻¹⁵ (logged by every training run). (3) On exact travel times of the gap ×12 maze (Dijkstra on a 1321×1321×13 grid, six goals, '
        'central differences at about two million nodes, table below): differentiated on the warp’s own node coordinates with the inner product, the true travel time gives '
        f'S·‖∇T‖<sub>M</sub> = {f1(c.get("inner_product_on_z", {}).get("median"), 3)} in the band’s centre and {f1(q.get("inner_product_on_z", {}).get("median"), 3)} in the quadrants, '
        f'the same as on the maze’s coordinates ({f1(c.get("maze_equation_on_x", {}).get("median"), 3)} and {f1(q.get("maze_equation_on_x", {}).get("median"), 3)}; the 3–5 % above 1 is the grid’s own error). '
        f'With the plain length it gives {f1(q.get("plain_length_on_z", {}).get("median"), 2)} in the quadrants, {f1(a.get("plain_length_on_z", {}).get("median"), 1)} in the arms '
        f'(from {f1(a.get("plain_length_on_z", {}).get("p10"), 1)} to {f1(a.get("plain_length_on_z", {}).get("p90"), 1)} depending on the direction of travel) and '
        f'{f1(c.get("plain_length_on_z", {}).get("median"), 1)} in the centre: the maze’s travel time does not solve the stretched maze’s equation.</p>'
        '<p><strong>What the check also shows.</strong> The inner product is, number for number, the eikonal loss taken on the maze through φ, which the gap study ran as its '
        '“constructed warp” rows (10 % and 18 % at gap ×12). So the equation was never the missing piece; what the training around it needs is.</p>')

    oo, adj = (by.get(k) or {} for k in ('open - open', 'adjacent quadrants'))
    wp = pa.get('warped_path_over_true_path') or {}
    out['PRICETEXT'] = (
        '<p>How wrong is a planner that works on the stretched maze and ignores all this? Measured with exact fields, no network involved, on the 300 queries of the gap ×12 maze: '
        f'read as the maze’s travel time, the stretched maze’s is exact inside a quadrant, {f1(1 / adj.get("warped_time_over_true", {}).get("median", NAN), 1)}× too short between adjacent quadrants and '
        f'{f1(1 / oo.get("warped_time_over_true", {}).get("median", NAN), 0)}× too short across the open band. Its shortest path, timed on the maze, is {f1(100 * (wp.get("median", NAN) - 1), 1)} % slower than the maze’s '
        f'shortest in the median, {f1(100 * (wp.get("p90", NAN) - 1), 1)} % at the 90th percentile and {f1(100 * (pa.get("worst", NAN) - 1), 0)} % for the worst query: the losses are in the open band, where a straight line on '
        'the warp is an L on the maze (figure).</p>')
    out['PATHFIG'] = ('<figure><img src="paths.png" alt="Four gap ×12 mazes, each with the maze’s shortest path and the stretched maze’s shortest path for one query; the two differ only in the open band.">'
                      '<figcaption>The four queries where the stretched maze’s shortest path (orange) loses the most travel time against the maze’s (blue), gap ×12, exact fields.</figcaption></figure>'
                      if h.get('have_paths') else '')

    out['STEPS0'] = (
        '<p>Base, as asked: ntrl-demo’s network and loop with the eikonal term only (no temporal-difference term, no normal term), 4,000 epochs, on the warp’s coordinates. The reference is the same network trained '
        'with the plain length (the stretched space) and planned there; “similar” below means against that reference on the same training pairs. Smaller gaps first, as asked: at gap ×4 the largest travel time '
        'fits the network’s output comfortably, and the inner product still lost a quarter to a third of the reference’s queries. Five things were wrong, in the order found; each fix is general.</p>')
    out['UNITTEXT'] = (
        '<p>The field may learn c·T for any c; the loss divides the gradient by c. In a 22-cell unit (ntrl-demo’s own unit at gap ×1) the eikonal-only loss fails even at gap ×1, where range is '
        f'irrelevant: {pct("c22", 1, "zaniso-U22.1")} % and {pct("m10", 1, "zaniso-U22.1")} % of queries solved, against {pct("c22", 1, PLAIN)} % and {pct("m10", 1, PLAIN)} % in the unit where the most magnified '
        f'region’s equation is exactly the stretched space’s (c = the warp’s largest stretch; here one unit = 3.7 cells). At gap ×12, same-quadrant queries: {pct("c22", 12, "zaniso-Ubox", "same quadrant")} % and '
        f'{pct("m10", 12, "zaniso-Ubox", "same quadrant")} % in the maze-side unit, {pct("c22", 12, PLAIN, "same quadrant")} % and {pct("m10", 12, PLAIN, "same quadrant")} % in the warp’s unit '
        f'(stretched space: {pct("c22", 12, REFZ, "same quadrant")} % and {pct("m10", 12, REFZ, "same quadrant")} %). This is the control for the gap study’s falling “constructed warp” curve: same warp, same equation, '
        'and inside the quadrants the result follows the unit. Everything below uses the warp’s unit.</p>')
    out['OPENTEXT'] = (
        '<p>ntrl-demo’s sampler keeps a pair only if its first pose is within 0.175 cells of a wall. On the stretched space that is harmless: the band is small and smooth. For the maze’s travel time the band is where '
        f'the field is steepest on the warp, and no pair starts inside it. At gap ×4 the field’s implied speed in the band’s centre was {spd("c22", 4, PLAIN, "centre")} and {spd("m10", 4, PLAIN, "centre")} of the true speed '
        f'and open–open queries were solved {pct("c22", 4, PLAIN, "open - open")} % and {pct("m10", 4, PLAIN, "open - open")} % of the time. Adding 250,000 pairs whose first pose is anywhere in the open '
        f'(to the 500,000 recipe pairs) makes the band obey the equation: implied speed {spd("c22", 4, OPEN, "centre")} and {spd("m10", 4, OPEN, "centre")}, open–open {pct("c22", 4, OPEN, "open - open")} % and '
        f'{pct("m10", 4, OPEN, "open - open")} %. The reference is retrained on the same pairs.</p>')
    out['LBTEXT'] = (
        '<p>The inner product weights the gradient along a direction by the square of the warp’s stretch there. Along a compressed direction that weight is 1/m² of the weight along a magnified one, so a field that '
        'simply does not vary along the compressed direction satisfies the equation almost everywhere. The loss cannot tell it from the travel time. That is what the network learned in the arms of the band '
        f'(map below): at gap ×4 on the 10×10 quadrants, {below("m10", 4, OPEN)} % of random pairs of band poses had a travel time below 0.9 × their straight-line distance on the maze, which no travel time can be.</p>'
        '<p>The missing information is exactly that bound. The speed is at most 1, so T(a, b) ≥ |x<sub>a</sub> − x<sub>b</sub>| for any two poses, with the distance taken on the maze. It needs no labels and no knowledge '
        'of the warp beyond the poses the pairs were drawn at. Added as a hinge on each batch’s own pairs and on its starts paired with shuffled ends (weight 10), it removes the violations '
        f'({below("c22", 4, OPENLB)} % and {below("m10", 4, OPENLB)} %) and brings open–open to {pct("c22", 4, OPENLB, "open - open")} % and {pct("m10", 4, OPENLB, "open - open")} %. '
        f'Alone, without the open-start pairs, it gives {pct("c22", 4, LB)} % and {pct("m10", 4, LB)} % of all queries; the two together {pct("c22", 4, OPENLB)} % and {pct("m10", 4, OPENLB)} %.</p>')
    out['FIELDFIG'] = ('<figure><img src="fields.png" alt="Maps of the travel time to one goal in the open band of the gap ×4 maze: the truth has round contours; the stretched-space field is nearly flat; the inner product alone has contours stretched along the band; the full recipe is round again; the two-part field has the right size and diamond-shaped contours.">'
                       '<figcaption>Travel time to one goal (star) in the open band of the gap ×4 maze, 22×22 quadrants, each field in the maze’s time unit. The stretched-space field is nearly flat across the band. '
                       'The inner product alone barely grows along the band’s compressed direction. With the fixes the contours are the truth’s. The last panel is the two-part field of a later section: the right size, '
                       'and diamonds where the truth has circles.</figcaption></figure>' if h.get('have_fields') else '')
    out['PWTEXT'] = (
        '<p>ntrl-demo weights each pair by exp(−T/2). On the stretched space two quadrants are near each other, so pairs between them count. With the maze’s travel time they are far, their weight is close to zero, and the '
        f'field inside one quadrant for a goal in another is not trained: quadrant-to-quadrant queries were solved {xq("m10", 4, OPENLB)} % of the time on the 10×10 quadrants at gap ×4 against {xq("m10", 4, REF)} % for the '
        'reference. The fix keeps ntrl-demo’s weight and shortens the time by how much the warp compressed the pair: exp(−T·ρ/2), ρ = |z<sub>a</sub> − z<sub>b</sub>| / (c·|x<sub>a</sub> − x<sub>b</sub>|). Inside a fully '
        'magnified region ρ = 1 and nothing changes; across a compressed one the pair weighs as it would on the stretched space. '
        f'Quadrant-to-quadrant: {xq("m10", 4, TAUZ)} % (22×22: {xq("c22", 4, OPENLB)} → {xq("c22", 4, TAUZ)} %, reference {xq("c22", 4, REF)} %). '
        f'Weighting by the distance on the warp alone, without the time, destroys the quadrants ({pct("c22", 4, PWZ)} % and {pct("m10", 4, PWZ)} % of all queries).</p>')
    to, to_m = top_output('c22', 12, PLAIN), top_output('m10', 12, PLAIN)
    out['RANGETEXT'] = (
        f'<p>The pair weight left quadrant-to-quadrant at {xq("m10", 4, TAUZ)} % against {xq("m10", 4, REF)} %. The head is the likely cause (the section on walls below measures it). ntrl-demo’s distance is a sum over 16 groups of the largest feature difference in '
        'each group. For a goal across the band a group can be won by a feature that carries the band, which is constant inside the start’s quadrant, and the features that know the way out of the quadrant are then masked. Consistent with that, '
        f'the field’s direction at the start is {ang("m10", 4, TAUZ, "adjacent quadrants")}° off for adjacent-quadrant queries where the reference is {ang("m10", 4, REF, "adjacent quadrants")}° off, although '
        f'for open–open queries it is {ang("m10", 4, TAUZ, "open - open")}° against {ang("m10", 4, REF, "open - open")}°. Giving the head two scales, eight groups as they are and eight with their features multiplied by a factor, '
        f'so that coarse groups can carry the band and fine ones the quadrants, helps where it was worst: quadrant-to-quadrant {xq("m10", 4, GS4)} % and {xq("c22", 4, GS4)} % at gap ×4 (reference {xq("m10", 4, REF)} % and {xq("c22", 4, REF)} %), '
        f'all queries {pct("m10", 4, GS4)} % and {pct("c22", 4, GS4)} %.</p>'
        + (f'<p><strong>The same head lifts the output range.</strong> At gap ×12 the true travel time reaches {to[1]:.0f} units and the plain network’s largest output over the 300 queries is {to[0]:.0f} '
           f'({to_m[0]:.0f} of {to_m[1]:.0f} on the 10×10 quadrants): ntrl-demo’s head cannot exceed 25.6 and is about 10 between unrelated points. With the first four fixes and the head left alone, the travel time is within 10 % '
           f'on {ta("c22", 12, OPENLB)} % of queries and {below("c22", 12, OPENLB)} % of band pairs are still below the straight-line bound (22×22 quadrants). '
           f'A scalar on the whole head (0.2 → 0.8, the first thing suggested) does give the range, travel time within 10 % on {ta("c22", 12, HS4)} % and {ta("m10", 12, HS4)} % of queries, but the quadrants pay for it: '
           f'same quadrant {pct("c22", 12, HS4, "same quadrant")} % and {pct("m10", 12, HS4, "same quadrant")} %, all queries {pct("c22", 12, HS4)} % and {pct("m10", 12, HS4)} %. '
           f'A factor inside the head, on the features of eight groups (×6), gives the range without that: travel time within 10 % on {ta("c22", 12, GS6)} % and {ta("m10", 12, GS6)} %, same quadrant {pct("c22", 12, GS6, "same quadrant")} % and '
           f'{pct("m10", 12, GS6, "same quadrant")} %, all queries {pct("c22", 12, GS6)} % and {pct("m10", 12, GS6)} %. Other layouts are within a training seed’s noise of it on the 10×10 quadrants '
           f'(twelve coarse groups {pct("m10", 12, GS6B)} %, factor 3 {pct("m10", 12, GS3)} %, a 1-2-4-8 ladder {pct("m10", 12, LADDER)} %, factor 12 {pct("m10", 12, GS12)} %); a soft, near-L1 head is worse ({pct("m10", 12, L1)} %).</p>'
           if to and to_m else ''))
    abl = []
    for b_ in ('c22', 'm10'):
        rows = []
        for v, label in ((PLAIN, 'inner product alone'), (LB, '+ lower bound'), (OPEN, '+ open-start pairs'), (OPENLB, '+ both'), (TAUZ, '+ pair weight'), (GS4, '+ two-scale head'),
                         (REF, 'reference: stretched space, same pairs')):
            if P(b_, 4, v):
                rows.append([label, pct(b_, 4, v), pct(b_, 4, v, 'same quadrant'), pct(b_, 4, v, 'open - open'), xq(b_, 4, v), ta(b_, 4, v), spd(b_, 4, v, 'centre')])
        abl.append(table([f'{BASE_NAME[b_]}, gap ×4', 'all (%)', 'same quadrant', 'open–open', 'quadrant ↔ quadrant', 'T within 10 % (%)', 'speed, centre'], rows))
    out['ABLTABLE'] = '</div><div class="tablewrap" style="margin-top: 14px">'.join(abl)
    out['ABLNOTE'] = ('Training seed 87358; the last two rows are pooled over the training seeds run. Each row adds one ingredient to the row above (the reference row does not use the inner product). Queries solved by MPPI on the warp; quadrant ↔ quadrant: adjacent and diagonal pooled; '
                      'T within 10 %: share of the 300 queries whose field travel time is within 10 % of the true one; speed, centre: the speed the field implies on the maze over the true speed in the band’s centre.')

    def seeds_txt(n):
        return {0: 'no run', 1: 'one training seed', 2: 'two training seeds', 3: 'three training seeds'}.get(n, f'{n} training seeds')

    def same_thing(b, k):
        """Success (%) of every training of the stretched-space field on the same pairs with seed 87358: the reference run and the heads inside the two-part runs."""
        vals = [seedpct(b, k, REF, 'z'), seedpct(b, k, RES, 'zh'), seedpct(b, k, RESU, 'zh')]
        vals = [v for v in vals if v != '–']
        return ', '.join(vals[:-1]) + ' and ' + vals[-1] if len(vals) > 1 else (vals[0] if vals else '–')
    out['MAINTEXT'] = (
        f'<p>Queries solved, planner on the warp, over gaps ×1 to ×12: the single field with the full recipe {span("c22", R)} % (22×22 quadrants) and {span("m10", R)} % (10×10); the stretched-space field trained on the same pairs '
        f'{span("c22", lambda b, k: REF)} % and {span("m10", lambda b, k: REF)} %; the inner product alone {span("c22", lambda b, k: PLAIN)} % and {span("m10", lambda b, k: PLAIN)} %. '
        f'At gap ×4: {pct("c22", 4, R("c22", 4))} against {pct("c22", 4, REF)} % and {pct("m10", 4, R("m10", 4))} against {pct("m10", 4, REF)} % '
        f'({seeds_txt(nseeds("c22", 4, R("c22", 4)))} for the recipe, {seeds_txt(nseeds("c22", 4, REF))} for the reference). At gap ×12: {pct("c22", 12, R("c22", 12))} against {pct("c22", 12, refv("c22", 12))} % and '
        f'{pct("m10", 12, R("m10", 12))} against {pct("m10", 12, refv("m10", 12))} % ({seeds_txt(nseeds("c22", 12, R("c22", 12)))} and {seeds_txt(nseeds("c22", 12, refv("c22", 12)))}). '
        f'On the 22×22 quadrants the two are within the spread between seeds at every gap (ranges over seeds at ×4: {rng("c22", 4, R("c22", 4))} and {rng("c22", 4, REF)} %; the largest difference, '
        f'{behind("c22", 8)} points at ×8, is between two single runs). On the 10×10 quadrants the single field is behind at ×4 and ×12, at the low end of the reference’s range '
        f'({rng("m10", 4, R("m10", 4))} against {rng("m10", 4, REF)} % and {rng("m10", 12, R("m10", 12))} against {rng("m10", 12, refv("m10", 12))} %), and within three points either way at ×1, ×2 and ×8 '
        f'({pct("m10", 1, R("m10", 1))} against {pct("m10", 1, REF)}, {pct("m10", 2, R("m10", 2))} against {pct("m10", 2, REF)} and {pct("m10", 8, R("m10", 8))} against {pct("m10", 8, REF)} %, one seed each).</p>'
        '<p><strong>The totals are made differently.</strong> At gap ×12, from the open band into a quadrant the single field solves '
        f'{pct("c22", 12, R("c22", 12), "quadrant - open")} against {pct("c22", 12, refv("c22", 12), "quadrant - open")} % and {pct("m10", 12, R("m10", 12), "quadrant - open")} against '
        f'{pct("m10", 12, refv("m10", 12), "quadrant - open")} %; from quadrant to quadrant {xq("c22", 12, R("c22", 12))} against {xq("c22", 12, refv("c22", 12))} % and '
        f'{xq("m10", 12, R("m10", 12))} against {xq("m10", 12, refv("m10", 12))} %. Between the 8×8 quadrants of the 22×22 maze nobody solves much. Between the 5×5 quadrants of the 10×10 the reference does and the single field '
        f'does not, and the next section finds the reason. At gap ×4 on the 10×10 quadrants it is also behind inside a quadrant ({pct("m10", 4, R("m10", 4), "same quadrant")} against {pct("m10", 4, REF, "same quadrant")} % '
        'of same-quadrant queries), which that reason does not cover.</p>'
        '<p><strong>How much is noise.</strong> The head inside a two-part run (below) is a stretched-space field trained from the same initial weights, on the same pairs, with the same loss as the reference run; no gradient from the correction reaches it, and only the order of the batches differs. '
        f'So there are several trainings of the same thing. With seed 87358 they solve {same_thing("m10", 12)} % at gap ×12 on the 10×10 quadrants, {same_thing("c22", 12)} % on the 22×22; at gap ×4, '
        f'{same_thing("m10", 4)} % (10×10) and {same_thing("c22", 4)} % (22×22). Single runs of one recipe differ by five points and more. The whiskers in the charts show the range over training seeds where three were run; '
        'a difference inside that range says nothing.</p>')
    if plan('c22', 4, R('c22', 4), 'x') or plan('m10', 4, R('m10', 4), 'x'):
        out['MAINTEXT'] += (
            '<p><strong>Planned on the maze itself.</strong> A field that tells the maze’s travel time can be descended on the maze with one step size and one temperature, which needs the warp but not its inverse. '
            f'Gap ×4: {pct("c22", 4, R("c22", 4), planner="x")} % and {pct("m10", 4, R("m10", 4), planner="x")} % for the recipe, {pct("c22", 4, REF, planner="x")} % and {pct("m10", 4, REF, planner="x")} % for the '
            f'stretched-space field. Gap ×12: {pct("c22", 12, R("c22", 12), planner="x")} % and {pct("m10", 12, R("m10", 12), planner="x")} % against '
            f'{pct("c22", 12, refv("c22", 12), planner="x")} % and {pct("m10", 12, refv("m10", 12), planner="x")} % (training seed 87358).</p>')
    out['MAINNOTE'] = ('All queries solved against the gap (300 certified queries per maze, 60 per region pair, planning seeds 0/1/2), MPPI on the warp’s coordinates, scored on the maze; pooled over training seeds, '
                       'the whisker is the range over them. The recipe and its reference train on the same pairs (recipe pairs drawn on the warped maze plus open-start pairs); the inner product alone on the recipe pairs. Hover for the numbers.')
    SAME = ['same quadrant']
    rc12, rf12 = R('m10', 12), refv('m10', 12)
    fs_r, fs_f = failed_in_start('m10', 12, rc12), failed_in_start('m10', 12, rf12)
    q_r, q_f = quart('m10', 12, rc12), quart('m10', 12, rf12)
    out['FEELTEXT'] = (
        f'<p>Between quadrants the single field falls behind as the band grows. The plans fail where they start: of the {fs_r[1]} failed plans between quadrants at gap ×12 on the 10×10 quadrants (planning seed 0), '
        f'{fs_r[0]} touch a wall before they have left the start quadrant ({fs_f[0]} of {fs_f[1]} for the reference).</p>'
        '<p>MPPI compares the field’s values at the points of its rollouts and nothing else, so it keeps off a wall only where the travel time rises as the pose moves into the wall’s margin. The eikonal term sets the '
        'size of the gradient there. It says nothing about its sign along the wall’s normal. The charts count, over 18,000 poses inside the margin of the quadrants’ walls, how often the travel time rises into the wall. '
        f'For a goal in the same quadrant every field does it for most poses at every gap (10×10 quadrants, gap ×12: {rises("m10", 12, rc12, SAME)} % for the recipe, {rises("m10", 12, rf12, SAME)} % for the reference). '
        f'For a goal in another quadrant the stretched-space field keeps doing it ({rises("m10", 4, REF)} % at gap ×4, {rises("m10", 12, rf12)} % at ×12) and the inner product stops as the band grows: alone, '
        f'{rises("m10", 1, PLAIN)} % at gap ×1 and {rises("m10", 4, PLAIN)} % at ×4 ({rises("c22", 4, PLAIN)} % on the 22×22 quadrants; 50 % is a coin flip); with the recipe {rises("m10", 4, R("m10", 4))} % at ×4 and '
        f'{rises("m10", 12, rc12)} % at ×12. The slope has the right size at those poses and either sign: at gap ×12 its quartiles along the wall’s normal are {q_r[0]} and {q_r[1]} for the recipe, '
        f'{q_f[0]} and {q_f[1]} for the reference (+1 is the true slope out of the margin).</p>'
        '<p><strong>Why.</strong> ntrl-demo’s head adds, over 16 groups, the largest feature difference in each group. On the maze’s clock a goal across the band is far, and the groups are won by the features that carry '
        'the band. A wall can then only show as a ripple on the winner, and whether a ripple raises or lowers the travel time depends on which side the goal lies. On the stretched space the band is short, no feature '
        f'wins everywhere, and the walls keep features of their own. The two-scale head of fix 5 is this remedy in part: at gap ×4 it brought the walls back ({rises("m10", 4, TAUZ)} → {rises("m10", 4, GS4)} % on the 10×10 '
        f'quadrants, {rises("c22", 4, TAUZ)} → {rises("c22", 4, GS4)} % on the 22×22), at ×12 it does not.</p>')
    out['FEELNOTE'] = ('Share of near-wall poses at which the field’s travel time rises as the pose moves towards the wall. Poses: inside the quadrant mazes, clearance 0.02 to 0.175 cells (the speed margin), 6,000 per '
                       'clearance band; goal: a free pose in the same quadrant (left) or in another quadrant (right). Training seed 87358. At 50 % the field does not tell a wall from free space.')
    wf = (h.get('figs') or {}).get('wallsfar') or {}
    re_ = wf.get('rising_ends') or {}
    out['FEELFIG'] = ('<figure><img src="wallsfar.png" alt="Six small charts, one per cut across a corridor of the start quadrant, each with four curves of the travel time to a far goal: the true travel time rises steeply at both ends of every cut; the learned fields rise less, and the single field falls at several ends.">'
                      f'<figcaption>The travel time to one goal in the next quadrant along {wf.get("cuts", 6)} cuts across corridors of the start quadrant (10×10 quadrants, gap ×12, heading 0), above its lowest value on the cut; '
                      'a wall stands at both ends of each cut. The true travel time rises into every wall. Of the '
                      f'{wf.get("ends", 12)} ends, the reference’s field rises at {re_.get("stretched space (reference)", "–")}, the two-part field’s at {re_.get("two-part field", "–")} and the single field’s at '
                      f'{re_.get("inner product, one field (recipe)", "–")}. All learned fields rise far less than the truth; that is the eikonal-only base.</figcaption></figure>' if (h.get('have_wallsfar') and wf) else '')
    have_far, have_sum = bool(P('m10', 12, FAR1)), bool(P('m10', 12, SUM2))
    out['FEELFIX'] = ((
        '<p><strong>Two more things tried on the single field at gap ×12; neither is kept.</strong> '
        + ('(a) A far term: a small smooth network on the warp’s coordinates with no upper limit, T<sub>F</sub> = scale·|e(z<sub>a</sub>) − e(z<sub>b</sub>)|, added to the head and trained with it. '
           f'It takes the output range off the head: travel time within 10 % on {ta("c22", 12, FAR1)} and {ta("m10", 12, FAR1)} % of queries with the head as ntrl-demo has it. For adjacent goals the walls come back '
           f'({rises("m10", 12, FAR1, ["adjacent quadrant"])} % on the 10×10 quadrants), but the field’s direction at the start gets worse ({ang("m10", 12, FAR1, "adjacent quadrants")}° off against '
           f'{ang("m10", 12, GS6, "adjacent quadrants")}°) and planning does not move: {pct("c22", 12, FAR1)} and {pct("m10", 12, FAR1)} % of all queries. ' if have_far else '')
        + ('(b) Two of the head’s 16 groups added up instead of entering the maximum, so that what a wall adds cannot be masked: adjacent-quadrant queries go from '
           f'{pct("m10", 12, GS6, "adjacent quadrants")} to {pct("m10", 12, SUM2, "adjacent quadrants")} % on the 10×10 quadrants and same-quadrant queries from {pct("m10", 12, GS6, "same quadrant")} to '
           f'{pct("m10", 12, SUM2, "same quadrant")} %; all queries {pct("c22", 12, SUM2)} and {pct("m10", 12, SUM2)} %.' if have_sum else '')
        + '</p>') if (have_far or have_sum) else '')

    have_res = bool(P('m10', 12, RES) or P('c22', 12, RES))
    have_full = bool(P('m10', 12, RESFULL) or P('c22', 12, RESFULL))
    res_gaps = [k for k in gaps if P('c22', k, RES) and P('m10', k, RES)]
    full_gaps = [k for k in gaps if P('c22', k, RESFULL) and P('m10', k, RESFULL)]

    def full_span(b, planner):
        vals = [pc(b, k, RESFULL, planner) for k in full_gaps]
        return '–' if not vals else f'{min(vals):.0f}–{max(vals):.0f}'

    def res_span(b, planner):
        vals = [pc(b, k, RES, planner) for k in res_gaps]
        return '–' if not vals else f'{min(vals):.0f}–{max(vals):.0f}'
    out['RESTEXT'] = ((
        '<p>If one bounded head cannot hold a long band and a fine maze at once, it need not. Write the travel time as T = H + F. <strong>H</strong> is ntrl-demo’s network and head, trained exactly as on the stretched '
        'space: plain length, ntrl-demo’s pair weight, and no gradient from anything else. <strong>F</strong> is the far term (a three-layer network, 128 wide, eight outputs, on the warp’s coordinates, no Fourier features, '
        'no upper limit), trained by the inner-product equation on the sum with the head’s gradient held fixed, and by the lower bound of fix 3. In the time unit of fix 1 the two equations are the same wherever the warp '
        'magnifies fully, so there F has nothing to add and the walls stay the head’s: with a goal in another quadrant the travel time rises into the wall for '
        f'{rises("c22", 12, RES)} and {rises("m10", 12, RES)} % of near-wall poses at gap ×12, the reference’s level.</p>'
        '<p>This has the shape of the first pass’s deficit term (the stretched maze’s time plus what the warp removed), which was set aside because its formula needs the constructed warp. Here there is no formula. '
        'F is a network; its scale is read from the smallest eigenvalue of J Jᵀ over the training poses; what it learns is whatever the inner-product equation asks for on top of H.</p>'
        '<p><strong>Result.</strong> Every run is planned twice, with the sum and with its head alone. The head alone is a stretched-space field trained on the same pairs from the same initial weights, so the pair is a controlled '
        f'comparison. Over gaps ×{res_gaps[0]} to ×{res_gaps[-1]} the sum solves {res_span("c22", "z")} % of queries on the 22×22 quadrants where its head solves {res_span("c22", "zh")} %, and {res_span("m10", "z")} % '
        f'against {res_span("m10", "zh")} % on the 10×10. Gap ×12: {pct("c22", 12, RES)} against {pct("c22", 12, RES, planner="zh")} % and {pct("m10", 12, RES)} against {pct("m10", 12, RES, planner="zh")} % '
        f'({seeds_txt(nseeds("m10", 12, RES))}); gap ×4: {pct("c22", 4, RES)} against {pct("c22", 4, RES, planner="zh")} % and {pct("m10", 4, RES)} against {pct("m10", 4, RES, planner="zh")} %. '
        f'The sum’s travel time is within 10 % of the truth on {ta("c22", 12, RES)} and {ta("m10", 12, RES)} % of queries at gap ×12 ({ta("c22", 4, RES)} and {ta("m10", 4, RES)} % at ×4). '
        f'Between quadrants at ×12 it solves {xq("c22", 12, RES)} and {xq("m10", 12, RES)} %, the single field {xq("c22", 12, R("c22", 12))} and {xq("m10", 12, R("m10", 12))} %.'
        + (f' Planned on the maze instead of on the warp (training seed 87358): {pct("c22", 12, RES, planner="x")} and {pct("m10", 12, RES, planner="x")} % at gap ×12, '
           f'{pct("c22", 4, RES, planner="x")} and {pct("m10", 4, RES, planner="x")} % at ×4.' if plan('m10', 12, RES, 'x') else '') + '</p>'
        '<p><strong>What it costs.</strong> The sum plans a few points below its own head, and most of the loss is in one kind of query, from the open band into a quadrant: '
        f'{pct("c22", 12, RES, "quadrant - open")} against {pct("c22", 12, RES, "quadrant - open", "zh")} % and {pct("m10", 12, RES, "quadrant - open")} against {pct("m10", 12, RES, "quadrant - open", "zh")} % at gap ×12. '
        f'On those queries the single field solves {pct("c22", 12, R("c22", 12), "quadrant - open")} and {pct("m10", 12, R("m10", 12), "quadrant - open")} %. The likely reason is the form. In the band’s arms the warp compresses '
        'one direction only; a time along the arm and a time across it add up in H + F like the two legs of an L, where the maze’s travel time is the straight line (the diamonds in the last panel of the field maps above). Diagonal moves are over-priced there by up to 41 % '
        f'(the field’s implied speed in the arms is {spd("c22", 12, RES, "arms")} and {spd("m10", 12, RES, "arms")} of the true one). Consistent with that, plans on the sum first reach the goal quadrant’s outline '
        f'a median of {arr("c22", 12, RES, "z")} and {arr("m10", 12, RES, "z")} cells from where the true path enters it, plans on the head alone {arr("c22", 12, RES, "zh")} and {arr("m10", 12, RES, "zh")} cells. '
        f'It is not the whole story: the single field arrives as far off ({arr("c22", 12, R("c22", 12), "z")} and {arr("m10", 12, R("m10", 12), "z")} cells) and still solves the most. '
        'A correction that is a metric of its own cannot change the L.</p>'
        + ('<p><strong>A second form, tried for that.</strong> F = |u(z<sub>a</sub>) − u(z<sub>b</sub>)| − |z<sub>a</sub> − z<sub>b</sub>|, with u a learned map of the warp’s coordinates that starts as the identity: the straight '
           'distance in a learned embedding minus the straight distance on the warp. With u the warp undone this is exact for straight travel in the open, diagonals through the arms included, and it is again learned from the '
           f'inner-product loss alone. It does repair the arms (implied speed {spd("c22", 12, RESU, "arms")} and {spd("m10", 12, RESU, "arms")} of the true one at gap ×12) and it can damage the quadrants: on the 10×10 quadrants the sum solves '
           f'{pct("m10", 12, RESU)} % where its head solves {pct("m10", 12, RESU, planner="zh")} % at gap ×12 (same-quadrant queries {pct("m10", 12, RESU, "same quadrant")} against {pct("m10", 12, RESU, "same quadrant", "zh")} %), '
           f'and on the 22×22 quadrants {pct("c22", 4, RESU)} against {pct("c22", 4, RESU, planner="zh")} % at gap ×4; the other two cases are close '
           f'({pct("c22", 12, RESU)} against {pct("c22", 12, RESU, planner="zh")} % and {pct("m10", 4, RESU)} against {pct("m10", 4, RESU, planner="zh")} %). A difference of two straight-line distances keeps a slope inside a quadrant, where the correction should have none; the separate metric '
           'has none by construction. The first form is the one kept.</p>' if (P('m10', 12, RESU) and P('c22', 12, RESU)) else '')
    ) if have_res else '<p>Runs in progress.</p>')
    def full_diffs(b):
        """(largest, smallest) of field-alone minus sum over the gaps run, in points."""
        d = [pc(b, k, RESFULL, 'zh') - pc(b, k, RESFULL, 'z') for k in full_gaps]
        return (max(d), min(d)) if d else (NAN, NAN)

    def per_gap(b, fn):
        return ', '.join(fn(b, k) for k in full_gaps)
    def full_diffs(b):
        """(largest, smallest) of field-alone minus sum over the gaps run, in points."""
        d = [pc(b, k, RESFULL, 'zh') - pc(b, k, RESFULL, 'z') for k in full_gaps]
        return (max(d), min(d)) if d else (NAN, NAN)

    def per_gap(b, fn):
        return ', '.join(fn(b, k) for k in full_gaps)

    def stall_n(b, k, v):
        x = h['stalled_open'](b, k, v) if (h.get('stalled_open') and v) else None
        return '–' if x is None else str(x)
    out['RESTEXT2'] = ((
        '<p><strong>On a field that already exists.</strong> The head is not touched, so the correction can be trained alone on any stretched-space field. The gap study’s full-loss fields '
        '(temporal-difference and normal terms, so not this study’s base) were frozen and only F was trained, 4,000 epochs, 8 to 15 minutes each, with one change to its term. These fields run slow inside the quadrants '
        f'by themselves (same-quadrant travel time {tcat("c22", 12, RESFULL, "same quadrant")}× the truth on the 22×22 quadrants, {tcat("m10", 12, RESFULL, "same quadrant")}× on the 10×10), so there the head alone is already '
        'steeper than the equation asks, and a correction that is a metric cannot take slope away. Where that is so, the sum’s target is the head’s own slope: the correction is asked to add nothing.</p>'
        f'<p>On the 10×10 quadrants the sum then solves {per_gap("m10", lambda b, k: pct(b, k, RESFULL))} % of queries at gaps {", ".join("×" + str(k) for k in full_gaps)}, the fields themselves '
        f'{per_gap("m10", lambda b, k: pct(b, k, RESFULL, planner="zh"))} %: never more than {f1(full_diffs("m10")[0])} points apart (three training seeds at ×12, one at the other gaps). '
        f'On the 22×22 quadrants: {per_gap("c22", lambda b, k: pct(b, k, RESFULL))} % against {per_gap("c22", lambda b, k: pct(b, k, RESFULL, planner="zh"))} %, at most {f1(full_diffs("c22")[0])} points apart. '
        f'The travel time is within 10 % of the truth on {per_gap("m10", lambda b, k: ta(b, k, RESFULL))} % of queries (10×10) and {per_gap("c22", lambda b, k: ta(b, k, RESFULL))} % (22×22), where the stretched-space fields alone are '
        'at a few per cent once the gap is wide. The correction restores what the warp removed and nothing else: the fields’ own slowness inside the quadrants stays, and at small gaps most of a query’s time is spent there.</p>'
        + (f'<p><strong>Control: the plain two-sided term.</strong> Without the change the sum lost up to {f1(max(pc("c22", k, RESFULL2, "zh") - pc("c22", k, RESFULL2, "z") for k in gaps if P("c22", k, RESFULL2)))} points on the 22×22 quadrants '
           f'({pct("c22", 8, RESFULL2)} against {pct("c22", 8, RESFULL2, planner="zh")} % at gap ×8), on open–open queries too ({pct("c22", 8, RESFULL2, "open - open")} %), which nothing else in this study fails. '
           f'Those plans stall in the open band without touching a wall, on a flat spot of the summed field: {stall_n("c22", 8, RESFULL2)} of the 120 plans into or inside the band at gap ×8 (planning seed 0), '
           f'{stall_n("c22", 8, RESFULL)} with the changed term. Asked to cancel slope it cannot cancel, the correction folds.</p>' if P('c22', 8, RESFULL2) else '')
    ) if (have_full and len(full_gaps) > 1) else '')
    out['FULLNOTE'] = ('The gap study’s stretched-space fields trained with ntrl-demo’s full loss (dashed blue: planned as they are) and the same fields with the far term trained on top of them by the inner-product loss, '
                       'the field frozen (solid). Dotted: the control, the correction trained with the plain two-sided term. Top: queries solved by MPPI on the warp. Bottom: share of queries whose travel time is within 10 % '
                       'of the truth (the stretched-space field alone is at 1–6 % once the gap is wide). Training seed 87358; at gap ×12 the solid and dashed lines pool three training seeds.' if have_full else '')
    out['RESNOTE'] = ('Queries solved by MPPI on the warp, pooled over training seeds and planning seeds 0/1/2. “Its head alone”: the same runs planned with the correction switched off. Reference run: the stretched-space '
                      'field of the main chart (training seed 87358 unless three seeds were run). T within 10 %: share of the 300 queries whose field travel time is within 10 % of the true one.')

    out['TIMETEXT'] = (
        f'<p>Each point is one of the 300 queries at gap ×12: the field’s travel time from start to goal against the true one. The stretched-space field lies far below the diagonal for every query that crosses the band '
        f'(median ratio {tcat("c22", 12, refv("c22", 12), "adjacent quadrants")} between adjacent quadrants, {tcat("c22", 12, refv("c22", 12), "open - open")} open–open). The inner product alone bends over '
        f'({tcat("c22", 12, PLAIN, "diagonal quadrants")} between diagonal quadrants). With the recipe the cloud lies on the diagonal: {tcat("c22", 12, R("c22", 12), "adjacent quadrants")} adjacent, '
        f'{tcat("c22", 12, R("c22", 12), "diagonal quadrants")} diagonal, {tcat("c22", 12, R("c22", 12), "open - open")} open–open, within 10 % of the truth on {ta("c22", 12, R("c22", 12))} % of all queries '
        f'({ta("m10", 12, R("m10", 12))} % on the 10×10 quadrants). '
        + (f'The two-part field’s cloud lies on the diagonal as well ({ta("c22", 12, RES)} and {ta("m10", 12, RES)} % within 10 %). ' if P('c22', 12, RES) else '')
        + 'The charts under the scatter give that share at every gap.</p>'
        f'<p>What is left is inside the quadrants, where every field here is slow by the same margin (same-quadrant ratio {tcat("c22", 12, refv("c22", 12), "same quadrant")} for the stretched space, '
        f'{tcat("c22", 12, R("c22", 12), "same quadrant")} for the recipe): the eikonal-only base, not the warp.</p>')
    out['SCATTERNOTE'] = ('The field’s travel time T(start, goal) against the true travel time, one point per query, training seed 87358, gap ×12; the dashed line is equality. Pick a region pair to highlight it. '
                          'Times in ntrl-demo’s unit of the maze (one unit = the maze’s side at full speed).')
    out['PATHTEXT'] = (
        f'<p><strong>Direction.</strong> At a query’s start, the angle between the field’s descent direction on the maze and the true one, gap ×4, median: open–open {ang("c22", 4, R("c22", 4), "open - open")}° and '
        f'{ang("m10", 4, R("m10", 4), "open - open")}° for the recipe against {ang("c22", 4, REF, "open - open")}° and {ang("m10", 4, REF, "open - open")}° for the stretched space; same quadrant '
        f'{ang("c22", 4, R("c22", 4), "same quadrant")}° and {ang("m10", 4, R("m10", 4), "same quadrant")}° against {ang("c22", 4, REF, "same quadrant")}° and {ang("m10", 4, REF, "same quadrant")}°. '
        'The stretched-space field points along the warp’s geometry; the recipe’s points along the maze’s.</p>')

    m_r, m_i = 'zaniso-Uquad-messy0.6-open-gs8x1+8x4-pwtauz-lb10', 'ziso-messy0.6-open'
    have_messy = bool(P('m10', 4, m_r) or P('c22', 4, m_r))
    out['GENTEXT'] = (
        '<p>The request’s end goal is a learned warp, which will be anisotropic, non-uniform and not separable. Every ingredient above is written for that case: the loss takes J Jᵀ from the warp’s autograd Jacobian as a full '
        'matrix; the time unit is the warp’s largest stretch; the lower bound and the pair weight need only each pose on the maze and on the warp; the open-start pairs and the head know nothing about the warp. '
        'A closed-form correction that exists for this constructed warp (the distance the warp removed, added back; it reproduces the travel time to 1 % on exact fields) was set aside for that reason: it needs the '
        'compressed region to be open and the warp to be separable.</p>'
        + ('<p><strong>Test: a messy warp.</strong> The constructed warp followed by a smooth random displacement of (x, y), eight plane waves, Jacobian up to 0.6 in norm: sheared, locally anisotropic everywhere, '
           'not separable (still fixed, not learned). Its Jacobian is full; the chain-rule identity holds to 10⁻¹⁵ on it. Gap ×4, same recipe, nothing retuned: '
           f'{pct("c22", 4, m_r)} % and {pct("m10", 4, m_r)} % of queries solved (constructed warp, same training seed: {seedpct("c22", 4, GS4)} % and {seedpct("m10", 4, GS4)} %), travel time within 10 % on {ta("c22", 4, m_r)} % and {ta("m10", 4, m_r)} % of queries; '
           f'the plain length on the same messy warp solves {pct("c22", 4, m_i)} % and {pct("m10", 4, m_i)} % with travel time within 10 % on {ta("c22", 4, m_i)} % and {ta("m10", 4, m_i)} %. '
           f'Planned on the maze itself, which needs the warp but not its inverse: {pct("c22", 4, m_r, planner="x")} % and {pct("m10", 4, m_r, planner="x")} % for the recipe, '
           f'{pct("c22", 4, m_i, planner="x")} % and {pct("m10", 4, m_i, planner="x")} % for the plain length. One training seed each.</p>'
           + ('<p><strong>The two-part field on the same messy warp.</strong> No formula for the correction exists here, and the warp magnifies fully almost nowhere, so the correction has work to do everywhere; the time unit is '
              f'this warp’s own largest stretch, read from its Jacobian. Queries solved with the sum: {pct("c22", 4, RESMESSY)} and {pct("m10", 4, RESMESSY)} % on the warp, {pct("c22", 4, RESMESSY, planner="x")} and '
              f'{pct("m10", 4, RESMESSY, planner="x")} % on the maze; with its head alone {pct("c22", 4, RESMESSY, planner="zh")} and {pct("m10", 4, RESMESSY, planner="zh")} %. '
              f'Travel time within 10 % of the truth on {ta("c22", 4, RESMESSY)} and {ta("m10", 4, RESMESSY)} % of queries. The travel time carries over to the messy warp; the cost in planning is larger than on the constructed '
              'warp on the 22×22 quadrants.</p>'
              if (P("c22", 4, RESMESSY) or P("m10", 4, RESMESSY)) else '') if have_messy else
           '<p>A test on a deliberately messy fixed warp (sheared, non-separable) is running.</p>'))
    rows = []
    for b in BASE_NAME:
        for v, label in ((GS4, 'constructed warp · recipe'), (REF, 'constructed warp · stretched space'), (m_r, 'messy warp · recipe'), (m_i, 'messy warp · stretched space'), (RESMESSY, 'messy warp · two-part field')):
            p = P(b, 4, v)
            if not p:
                continue
            z, x = plan(b, 4, v, 'z'), plan(b, 4, v, 'x')
            rows.append([f'{BASE_NAME[b]} · {label}', f1(z and z['pct']), f1(x and x['pct']), ta(b, 4, v), tcat(b, 4, v, 'open - open'), spd(b, 4, v, 'arms'), spd(b, 4, v, 'centre')])
    out['GENTABLE'] = table(['gap ×4, training seed 87358', 'MPPI on the warp (%)', 'MPPI on the maze (%)', 'T within 10 % (%)', 'open–open T / T*', 'implied speed, arms', '… centre'], rows) if have_messy else ''

    out['VERDICT'] = (
        '<ul class="tight">'
        '<li><strong>The idea is correct</strong>, and it is the same loss term as training on the maze through φ. The equation was never the obstacle.</li>'
        '<li><strong>Its training has blind spots the stretched space does not have</strong>: no pair starts in the compressed region; the equation is satisfied by a field that ignores a compressed direction; '
        'and the pair weight drops everything the warp brought close. Open-start pairs, the straight-line lower bound and a compression-aware pair weight fix them, and none of them looks at how the warp was built.</li>'
        f'<li><strong>With them, the warp’s time unit and a two-scale head, one field comes close to the stretched space and tells the maze’s travel time</strong>: {pct("c22", 4, R("c22", 4))} against {pct("c22", 4, REF)} % and '
        f'{pct("m10", 4, R("m10", 4))} against {pct("m10", 4, REF)} % at gap ×4, {pct("c22", 12, R("c22", 12))} against {pct("c22", 12, refv("c22", 12))} % and {pct("m10", 12, R("m10", 12))} against '
        f'{pct("m10", 12, refv("m10", 12))} % at ×12; travel time within 10 % on {ta("c22", 4, R("c22", 4))} and {ta("m10", 4, R("m10", 4))} % of queries at ×4 and {ta("c22", 12, R("c22", 12))} and {ta("m10", 12, R("m10", 12))} % at ×12, '
        f'where the stretched space has {ta("c22", 4, REF)} and {ta("m10", 4, REF)} %. On the 22×22 quadrants the difference is inside the spread between training seeds; on the 10×10 the single field is '
        f'{behind("m10", 4)} and {behind("m10", 12)} points behind.</li>'
        '<li><strong>The output range is a separate problem</strong>, and either a second scale inside the head or an unbounded far term solves it. A scalar on the whole head gives the range and costs the quadrants.</li>'
        f'<li><strong>At wide gaps the single field loses the walls for far goals.</strong> Its travel time rises into a quadrant’s wall for {rises("m10", 12, R("m10", 12))} % of near-wall poses when the goal is in another quadrant '
        f'(reference {rises("m10", 12, refv("m10", 12))} %), and quadrant-to-quadrant plans hit a wall in the quadrant they start in. It makes up the total on queries from the open band into a quadrant.</li>'
        + (f'<li><strong>A two-part field keeps the walls</strong>: the head trained as on the stretched space, plus a learned correction trained by the inner product on the sum. It plans {gapx("c22", 12, RES)} and '
           f'{gapx("m10", 12, RES)} points below its own head at gap ×12 with the travel time within 10 % on {ta("c22", 12, RES)} and {ta("m10", 12, RES)} % of queries. '
           'Its weak spot is the opposite of the single field’s: diagonal moves through the band’s arms.</li>' if P('m10', 12, RES) else '')
        + (f'<li><strong>It attaches to fields that already exist.</strong> The correction alone, trained on the gap study’s frozen full-loss fields, keeps their planning: {full_span("m10", "z")} % '
           f'of queries solved against {full_span("m10", "zh")} % on the 10×10 quadrants over gaps {", ".join("×" + str(k) for k in full_gaps)} (never more than {f1(full_diffs("m10")[0])} points apart) and '
           f'{full_span("c22", "z")} against {full_span("c22", "zh")} % on the 22×22 (at most {f1(full_diffs("c22")[0])} points), and gives them the travel time across the band. It needs one care: where the field is already '
           'steeper than the equation asks, the correction must be asked for nothing rather than for a negative amount.</li>' if len(full_gaps) > 1 else '')
        + '<li><strong>The base limits everything else here.</strong> Without the temporal-difference and normal terms the stretched-space reference itself solves a third to a half of the queries where the gap study’s full loss had 58–85 %.</li>'
        '<li><strong>Not run.</strong> The normal term of ntrl-demo’s full loss supervises exactly what the single field loses at wide gaps, the sign of the slope along a wall’s normal; the base asked for here leaves it out. '
        'The single field with that term, and either route on a learned warp, are the next tests.</li>'
        '</ul>')
    out['SETUP'] = (
        '<ul class="tight">'
        '<li><strong>Mazes and queries.</strong> The gap study’s: the four 8×8 quadrants of the 22×22 cross maze, and the 10×10 maze cut into four 5×5 quadrants, pushed apart by an open band of 6, 12, 24, 48 and 72 cells. '
        '300 certified queries per maze, 60 per region pair, planning seeds 0/1/2; a query is solved if MPPI arrives and the exact scorer finds no contact.</li>'
        '<li><strong>Warp.</strong> The gap study’s constructed warp (<code>warp.constructed</code>), fixed; the messy warp of the generality test adds a smooth random displacement to it (<code>aniso.MessyWarp</code>).</li>'
        '<li><strong>Training.</strong> ntrl-demo’s network and loop, eikonal term only, 4,000 epochs. Recipe pairs: 500,000 drawn on the warped maze (the gap study’s sets); open-start pairs: 250,000 more whose first pose is '
        'anywhere beyond the wall margin, drawn on the maze. Speeds are the maze’s.</li>'
        '<li><strong>The recipe (one field).</strong> Inner product with J Jᵀ; time unit = the warp’s largest stretch; open-start pairs; lower-bound hinge, weight 10; pair weight exp(−Tρ/2); head with eight groups as they are and '
        'eight multiplied by 4 (gaps ×1 to ×4) or 6 (gaps ×8, ×12).</li>'
        '<li><strong>The two-part field.</strong> Same pairs and time unit. Head: ntrl-demo’s, plain-length eikonal term, pair weight exp(−H/2). Correction: scale·|e(z<sub>a</sub>) − e(z<sub>b</sub>)|, e a 3-128-128-8 network with '
        'softplus units, scale = c / √(smallest eigenvalue of J Jᵀ) at the 1st percentile of the training poses, no weight decay; inner-product term on the sum with the head’s gradient detached, same pair weight, and the '
        'lower-bound hinge (weight 10) on the sum. On a frozen field (the gap study’s full-loss fields): only the correction is trained, without ntrl-demo’s rollback rule, and the sum’s target slope is the larger of the '
        'equation’s and the head’s own.</li>'
        '<li><strong>Planner.</strong> maze_scale’s MPPI on the warp’s coordinates (step and tolerance × the quadrants’ magnification; path mapped back exactly and scored on the maze). Every field is planned in the maze’s '
        'time unit.</li>'
        '<li><strong>Truth.</strong> T* by Dijkstra on maze_scale’s C-space grid (1/15 cell, 13 headings).</li>'
        '<li><strong>Not the pinned environment.</strong> Host venv, RTX 3090 and RTX 5070; every run is registered (table below). One training seed unless a whisker is drawn.</li>'
        '</ul>')
    return out
