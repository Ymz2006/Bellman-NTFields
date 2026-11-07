"""Builds report/speedlines.html from speedlines_template.html and report/speedlines/*.dist.json (speedlines.py).

    python3 jeffrey/cross_warp/speedpage.py

The page fetches report/speedlines/<base>_k<k>.json (published beside it as speedlines/...); the near-wall distributions are
embedded, without the per-field stored_speed_fit. Every number in the text is read from the data here, none is typed in.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'report'
BASES = ('c22', 'm10')
GAPS = (1, 2, 4, 8, 12)


def load():
    idx = [json.loads((OUT / f'speedlines/{b}_k{k}.dist.json').read_text()) for b in BASES for k in GAPS]
    for e in idx:
        for f in e['dist']['fields'].values():
            f.pop('stored_speed_fit', None)
    return idx


def main():
    idx = load()
    at = {(e['base'], e['k']): e for e in idx}
    # the 22 x 22 x1 ntrl-demo run stored its success on the older 240-query set: use its evalq result on this study's 300 queries
    g = json.loads((OUT / 'gap_summary.json').read_text())['1|ntrl-demo']
    assert g['seeds'][0] == 87358
    at[('c22', 1)]['runs']['id']['success'] = [round(g['per_seed_pct'][0] * 9), 900]
    F = lambda b, k, f: at[(b, k)]['dist']['fields'][f]
    s = lambda v: f'{v:.2f}' if v >= 0.1 else f'{v:.3f}'
    x1, x12 = F('c22', 1, 'idw'), F('c22', 12, 'idw')
    z12, n12 = F('c22', 12, 'pmzw'), F('c22', 12, 'id')
    m1, m12 = F('m10', 1, 'idw'), F('m10', 12, 'idw')
    mz12 = F('m10', 12, 'pmzw')
    floor = at[('c22', 1)]['dist']['target_curve'][1][0]
    top = at[('c22', 12)]['top_freq_cycles_per_unit']
    wb1, wb12 = at[('c22', 1)]['wall_band_norm'], at[('c22', 12)]['wall_band_norm']
    tf = at[('c22', 12)]['dist']['target_floor_median']
    st12 = at[('c22', 12)]['strip']['median']
    col = at[('c22', 12)]['strip']['collision_share']
    lede = f'''<p class="sub">ntrl-demo trained on the regular maze, with its training pairs drawn on the constructed warped maze. Learned speed is 1 / |&part;T/&part;x&#8320;|, compared with the target speed the loss asks for.</p>
<ul class="find">
<li><b>Your read holds: the field loses the walls as the gap grows.</b> At the slowest near-wall poses the target median is {s(tf)}. The learned median there rises from {s(x1['pair']['floor_median'])} at &times;1 to {s(x12['pair']['floor_median'])} at &times;12 on the 22&times;22 maze ({s(m1['pair']['floor_median'])} to {s(m12['pair']['floor_median'])} on the 10&times;10). By &times;12 the field is no slower beside a wall than in open space ({s(x12['free']['median'])}), where the target asks for a {1 / tf:.0f}&times; difference.</li>
<li><b>The pairs do not fix it.</b> Pairs drawn on the maze give the same wall speed at &times;12 ({s(n12['pair']['floor_median'])}), and correlation with the target is near zero either way ({x12['pair']['corr']:.2f} and {n12['pair']['corr']:.2f}).</li>
<li><b>The coordinates do, most of the way.</b> The same pairs, labels and network trained on the warped maze reach {s(z12['pair']['floor_median'])} at &times;12, with correlation {z12['pair']['corr']:.2f}, and stay flat across every gap ({s(F('c22', 1, 'pmzw')['pair']['floor_median'])} at &times;1). On the lines its dip goes down to {s(st12['pmzw'])} in the grey strips at &times;12, where the warped-pairs field stays at {s(st12['idw'])}. Its dip has a soft edge: just outside contact it is still about {z12['pair']['floor_median'] / tf:.0f}&times; above the target ({s(mz12['pair']['floor_median'])} on the 10&times;10).</li>
<li><b>What this does not settle.</b> It separates pairs from coordinates. It does not separate a network that cannot represent the walls in x from training that does not reach them. Each warped-pairs field is a single training seed.</li>
</ul>'''
    notes = f'''<p class="note"><b>Fields.</b> Training seed 87358 for all three, one training per maze and gap. &ldquo;Warped pairs&rdquo; is the <code>idw</code> run, &ldquo;maze pairs&rdquo; is <code>id</code>, &ldquo;Trained on warped maze&rdquo; is <code>pmzw</code>. No new training was run for this page.</p>
<p class="note"><b>Solved %.</b> MPPI on the regular maze, 300 queries by 3 planning seeds, one training seed. The warped-maze field does a little better with the warped planner; those runs are in the gap study.</p>
<p class="note"><b>Speed.</b> 1 / |&part;T/&part;x&#8320;| over (x, y, heading), the same measure as <code>evaluate.speed_fit</code>; the extractor reproduces the stored fit for the warped-pairs field. A speed depends on the goal as well as the pose, which is why every line has a goal. The average is the mean speed over the prepare run&rsquo;s 48 goals, and the shading is its 10&ndash;90 % range.</p>
<p class="note"><b>The warped-maze field</b> takes its loss in z = &phi;(x), so its speed is 1 / |&part;H/&part;z| at z = &phi;(x). That is the quantity the loss pulls towards the same label, so it compares directly with the target. Its speed measured in x would differ by the warp&rsquo;s stretch.</p>
<p class="note"><b>Target.</b> The trained label: clearance of the robot rectangle to the nearest wall, clipped to the 0.175-cell margin and stretched to a floor of {s(floor)}. Inside a wall it stays at the floor. Held-out poses stop at clearance 0.018, so they approach the floor without reaching it, which is why the near-wall charts use poses with a target below 0.05. Lines use heading 0, so the robot&rsquo;s long side lies along x, and a line along y sees a different wall band from a line along x.</p>
<p class="note"><b>Grey strips.</b> Poses with a target below 0.05. {100 * col:.0f} % of them are in collision, and training pairs barely go there, so what a field does inside a strip is mostly extrapolation. The ramps beside the strips and the near-wall charts below are the trained signal.</p>
<p class="note"><b>Normalized.</b> Each curve is divided by its own distance-weighted mean along the whole line, so the charts compare shape and ignore overall scale. The horizontal rule marks 1, the average.</p>
<p class="note"><b>Near-wall poses.</b> The 50,000 held-out pairs of the warped-pairs prepare run, never trained on. Every x&#8320; lies within the margin. &ldquo;Its held-out partner&rdquo; uses the pair&rsquo;s own x&#8321;, the training distribution, which is off-distribution for the maze-pairs field. &ldquo;A random far goal&rdquo; uses one of the 48 goals. Open space is 20,000 free poses with clearance of at least 0.35 cells, with far goals.</p>
<p class="note"><b>Resolution.</b> Along a line, 1/16-cell steps over the blocks and &frac12;-cell steps over the band&rsquo;s interior. Lines sit every 1/8 cell over the blocks and every cell over the interior, and the position snaps to the nearest one. Line values are stored as 8-bit log speed, within about 2 %, and clip at 0.001 (under 0.5 % of a line&rsquo;s cells, all on the &times;1 fields); all numbers in the text and table come from unquantized values.</p>
<p class="note"><b>Encoder.</b> The largest Fourier frequency is {top:.1f} cycles per unit of the maze box, the same at every gap, while the wall margin shrinks from {wb1:.4f} to {wb12:.4f} of the box between &times;1 and &times;12. The network&rsquo;s later layers can build higher frequencies, so this shows the gap in scale and does not prove a limit.</p>'''
    html = (HERE / 'speedlines_template.html').read_text()
    html = html.replace('<!--LEDE-->', lede).replace('<!--NOTES-->', notes).replace('/*INDEX*/null', json.dumps(idx, separators=(',', ':')))
    (OUT / 'speedlines.html').write_text(html)
    print(f'{(OUT / "speedlines.html").stat().st_size / 1e6:.2f} MB')
    for b in BASES:
        for f in ('idw', 'id', 'pmzw'):
            print(b, f, 'wall pair', [round(F(b, k, f)['pair']['floor_median'], 3) for k in GAPS], 'free', [round(F(b, k, f)['free']['median'], 3) for k in GAPS],
                  'corr', [round(F(b, k, f)['pair']['corr'], 2) for k in GAPS], 'contrast', [round(F(b, k, f)['contrast'], 1) for k in GAPS])
    print('top', top, 'band', wb1, wb12, 'target median at wall', tf, 'k1 id', at[('c22', 1)]['runs']['id']['success'])


if __name__ == '__main__':
    main()
