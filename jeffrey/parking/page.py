"""Build the findings page (report/parking_findings.html) from report.py / posthoc.py outputs.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/parking/page.py [--report jeffrey/parking/report]

Reads summary.json (one row per variant x car width x wall, merged over training seeds), posthoc.json and the PNG
figures; charts are drawn in the page as theme-aware SVG with a hover readout and a table view each.
"""
import argparse
import base64
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def png(path):
    return 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode()


def pct(v, n=300):
    return round(100.0 * v / n, 1)


def build(rep):
    rows = json.loads((rep / 'summary.json').read_text())
    walls = sorted([r for r in rows if r['variant'] == 'metric' and r['car_w'] == 0.13], key=lambda r: r['wall'])
    arm = sorted([r for r in rows if r['variant'] == 'metric_arm' and r['car_w'] == 0.13], key=lambda r: r['wall'])
    gap = sorted([r for r in rows if r['variant'] == 'metric' and (r['car_w'] == 0.13 or r['wall'] == 0.005)], key=lambda r: r['cspace_gap'])

    def pts(rs, key='default', x='wall'):
        out = []
        for r in rs:
            vals = [pct(v) for v in r[key]]
            out.append(dict(x=r[x], y=round(sum(vals) / len(vals), 1), lo=min(vals), hi=max(vals), n=len(vals)))
        return out

    seeds = ['87358', '1', '2']
    marks = {'87358': 'circle', '1': 'square', '2': 'diamond'}

    def seed_series(rs, key, scale, x='cspace_gap'):
        out = []
        for sd in seeds:
            pts = [dict(x=r[x], y=round(scale(r['per_seed'][sd][key]), 2)) for r in rs if sd in r['per_seed']]
            if pts:
                out.append(dict(name=f'metric, training seed {sd}', slot=1, marker=marks[sd], points=pts))
        return out

    oracle_lo = min(r['oracle'] for r in gap + walls)
    wpath = rep / 'widen_summary.json'                                  # widening-car sweep (widen.py)
    wrows = sorted(json.loads(wpath.read_text()), key=lambda r: r['side_gap']) if wpath.exists() else []
    charts = {
        'walls': dict(xlabel='Divider thickness w (lot is 1 x 1; car 0.30 x 0.13)', ylabel='Bay-to-bay success, %', ymax=100,
                      refs=[dict(y=pct(oracle_lo), label=f'oracle (same MPPI on the true travel time): {oracle_lo}-300 of 300'),
                            dict(y=0, label='straight line: 0 of 100 (always crosses the divider)')],
                      series=[dict(name='ntrl-demo metric (mean of training seeds, range)', slot=1, marker='circle', points=pts(walls)),
                              dict(name='ntrl-demo metric_arm (1 training seed)', slot=2, marker='circle', points=pts(arm))]),
        'gap': dict(xlabel='C-space wall thickness = divider + car width', ylabel='Bay-to-bay success, %', ymax=100,
                    refs=[dict(y=pct(oracle_lo), label=f'oracle: {oracle_lo}-300 of 300 at every wall')],
                    series=seed_series(gap, 'default', lambda v: pct(v))),
        'sharp': dict(xlabel='C-space wall thickness = divider + car width', ylabel='T(start, goal) / true T*', ymax=1.0,
                      refs=[], series=seed_series(gap, 't_over_tstar_median', lambda v: v)),
    }
    if wrows:
        series = [dict(name=f'small-step MPPI, training seed {sd}', slot=1, marker=marks[sd],
                       points=[dict(x=round(r['side_gap'], 4), y=pct(r['per_seed'][sd]['fine'])) for r in wrows if sd in r['per_seed']]) for sd in seeds]
        dpts = []
        for r in wrows:
            vals = [pct(r['per_seed'][sd]['default']) for sd in r['seeds']]
            dpts.append(dict(x=round(r['side_gap'], 4), y=round(sum(vals) / len(vals), 1), lo=min(vals), hi=max(vals), n=len(vals)))
        series.append(dict(name="ntrl-demo's own MPPI (mean of training seeds, range)", slot=2, marker='circle', points=dpts))
        series.append(dict(name="ntrl-demo's own MPPI on the true travel time", slot=4, marker='square',
                           points=[dict(x=round(r['side_gap'], 4), y=pct(r['oracle']['default'])) for r in wrows]))
        charts['widen'] = dict(xlabel='Side gap = free space beside the parked car (bays fixed at 0.23)', ylabel='Bay-to-bay success, %', ymax=100,
                               refs=[dict(y=100, label='small-step MPPI on the true travel time: 300 of 300 at every gap')], series=series)
    return rows, walls, arm, gap, charts, wrows



def make_figures(rep):
    """Page-sized figures (7.6 in wide, 9 pt type): scene, divider profiles for six C-space walls, field slices."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle, FancyArrowPatch
    import sys
    sys.path.insert(0, str(HERE))
    import scene as SC
    from report import find_runs, collect
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9aa0a6', 'axes.linewidth': 0.8})
    ink, blue, red = '#15181b', '#2a78d6', '#cf222e'
    groups = collect(find_runs([]))
    # scene
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.75))
    for ax, w in zip(axes, [0.30, 0.05, 0.005]):
        for b in SC.boxes(w):
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], color='#4a4f55', lw=0))
        ax.add_patch(Rectangle((SC.X_D, -w / 2 - SC.CAR_W / 2), 0.48 - SC.X_D, w + SC.CAR_W, color=red, alpha=0.22, lw=0))
        for bay, colour, label in (('A', '#1baf7a', 'start'), ('B', blue, 'goal')):
            y = SC.bay_y(w, bay)
            ax.add_patch(Rectangle((SC.PARK_X - SC.CAR_L / 2, y - SC.CAR_W / 2), SC.CAR_L, SC.CAR_W, color=colour, alpha=0.9, lw=0))
            ax.text(SC.PARK_X, y, label, ha='center', va='center', fontsize=7.5, color='white')
        ya, yb = SC.bay_y(w, 'A'), SC.bay_y(w, 'B')
        for a, b in (((SC.PARK_X, ya), (SC.X_BACK, ya)), ((SC.X_BACK, ya), (SC.X_BACK, yb)), ((SC.X_BACK, yb), (SC.PARK_X, yb))):
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle='->', mutation_scale=9, color='#b07d00', lw=1.3))
        ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f'divider {w:g}: C-space wall {w + SC.CAR_W:.3f}')
    fig.tight_layout(pad=0.4)
    fig.savefig(rep / 'page_scene.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    import page_figs
    page_figs.make_all(rep)                 # page_far_side, page_narrow_car, page_widen_paths, page_widen_profiles
    # slices with paths
    grid = np.linspace(-0.5, 0.5, 201)
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.2))
    for col, W in enumerate([0.01, 0.05, 0.13]):
        g = groups[('metric', W, 0.005)]
        t = np.load(Path(g['dirs'][0]) / 'truth.npz')
        sd = '87358'
        f = np.load(Path(g['seeds'][sd]['dir']) / 'field.npz')
        paths = np.load(Path(g['seeds'][sd]['dir']) / 'paths.npz', allow_pickle=True)['default_spot_to_spot_0']
        ok = set(g['seeds'][sd]['planning']['default/spot_to_spot/0']['success_ids'])
        for row, img in enumerate((f['slice_pred'], t['slice_star'])):
            ax = axes[row, col]
            im = ax.contourf(grid, grid, np.clip(img.T, 0, 2.0), levels=np.linspace(0, 2.0, 21), cmap='viridis')
            for b in SC.boxes(0.005, W):
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], color='#4a4f55', lw=0, zorder=3))
            if row == 0:
                for i, pth in enumerate(paths[:50]):
                    ax.plot(pth[:, 0], pth[:, 1], color='#2da44e' if i in ok else red, lw=0.7, zorder=4)
            ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{'ntrl-demo' if row == 0 else 'truth'}, C-space wall {0.005 + W:.3f}")
    fig.subplots_adjust(left=0.01, right=0.88, top=0.95, bottom=0.02, wspace=0.06, hspace=0.14)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cax, label='travel time from bay A (θ = 0 slice)')
    fig.savefig(rep / 'page_slices.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

TEMPLATE = Path(__file__).with_name('page_template.html')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, default=HERE / 'report')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    rows, walls, arm, gap, charts, wrows = build(a.report)
    make_figures(a.report)
    figs = {k: png(a.report / f) for k, f in [('scene', 'page_scene.png'), ('profiles', 'page_far_side.png'), ('slices', 'page_slices.png'),
                                               ('narrow', 'page_narrow_car.png'), ('widen_paths', 'page_widen_paths.png'), ('widen_profiles', 'page_widen_profiles.png')]}
    text = TEMPLATE.read_text()
    text = text.replace('/*DATA*/null', json.dumps(dict(charts=charts, rows=rows, wrows=wrows)))
    for k, v in figs.items():
        text = text.replace(f'{{{{fig:{k}}}}}', v)
    prose = json.loads((a.report / 'page_text.json').read_text())     # lede, tiles, walls_text, why_text, gap_text, mech_text, setup, caveats (HTML)
    for k, v in prose.items():
        text = text.replace(f'{{{{{k}}}}}', v)
    missing = [k for k in ['lede', 'tiles', 'walls_text', 'why_text', 'gap_text', 'mech_text', 'widen_text', 'setup', 'caveats', 'fig:scene', 'fig:profiles',
                           'fig:slices', 'fig:narrow', 'fig:widen_paths', 'fig:widen_profiles']
               if '{{' + k + '}}' in text]
    assert not missing, f'unfilled placeholders: {missing}'
    (a.out or a.report / 'parking_findings.html').write_text(text)
    print(a.out or a.report / 'parking_findings.html', len(text) // 1024, 'KB')


if __name__ == '__main__':
    main()
