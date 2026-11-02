"""Tables and figures for the parking thin-wall runs (registered results/registered_<id>/results.json).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/parking/report.py [--runs id ...] [--out jeffrey/parking/report]

Without --runs, every succeeded registry run whose config action is 'parking-thin-wall' (smoke runs excluded) is
used; runs with the same (variant, car width, wall) are merged by training seed. Writes tables.md, summary.json and
PNG figures: success and sharpness vs wall thickness per variant, success vs C-space gap (wall + car width) for
both sweeps, travel-time profiles across the divider, and theta = 0 field slices with MPPI paths.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene as SC  # noqa: E402

REGISTRY = Path('~/.local/share/ntfields-experiments/runs').expanduser()
PS = ['0', '1', '2']
DEFAULT_W = 0.13


def find_runs(ids):
    rows = []
    for rec in sorted(REGISTRY.glob('*/record.json')):
        r = json.loads(rec.read_text())
        cfg = r.get('spec', {}).get('config', {})
        if cfg.get('action') != 'parking-thin-wall' or cfg.get('smoke'):
            continue
        if ids and not any(r['id'] == i or r['id'].endswith(i) for i in ids):
            continue
        if r.get('state') != 'succeeded':
            continue
        out = Path(r['output'])
        if (out / 'results.json').exists():
            rows.append((r['id'], out, json.loads((out / 'results.json').read_text())))
    return rows


def pct(k, n):
    return 100.0 * k / n if n else float('nan')


def collect(runs, fixed_bays=False):
    """{(variant, car_w, wall): merged record} over the training seeds of every matching run. Runs with a fixed bay
    width (run.py --bay-width, the widening-car sweep; widen.py) are kept apart: fixed_bays selects them."""
    groups = {}
    for rid, out, res in runs:
        if bool(res.get('bay_width_fixed')) != fixed_bays:
            continue
        key = (res['variant'], round(res.get('car_width', DEFAULT_W), 6), res['wall'])
        g = groups.setdefault(key, dict(variant=key[0], car_w=key[1], wall=key[2], cspace_gap=key[1] + key[2], runs=[], dirs=[], seeds={},
                                        truth=res['truth'], straight=res['straight'], oracle=res['oracle']))
        g['runs'].append(rid)
        g['dirs'].append(str(out))
        for seed, sr in res.get('seeds_results', {}).items():
            g['seeds'][seed] = dict(sr, dir=str(out / f'seed{seed}'), run=rid)
    return groups


def rate(g, setting, suite):
    rows = [g['seeds'][s]['planning'][f'{setting}/{suite}/{p}'] for s in sorted(g['seeds']) for p in PS]
    return sum(r['success'] for r in rows), sum(r['n'] for r in rows), rows


SEED_ORDER = ['87358', '1', '2']


def ordered(g):
    return sorted(g['seeds'], key=lambda s: SEED_ORDER.index(s) if s in SEED_ORDER else 99)


def per_seed(g, setting, suite):
    return [sum(g['seeds'][s]['planning'][f'{setting}/{suite}/{p}']['success'] for p in PS) for s in ordered(g)]


def seed_cell(g, setting, suite):
    return ' · '.join(f'{s}: {v}' for s, v in zip(ordered(g), per_seed(g, setting, suite)))


def fmean(g, key):
    vals = [g['seeds'][s]['field'][key] for s in g['seeds'] if g['seeds'][s]['field'][key] is not None]
    return float(np.mean(vals)) if vals else float('nan')


def table(groups, title, rows_key, label):
    lines = [f'### {title}', '',
             f'|{label}|C-space gap|T*/Euclid|straight /100|oracle /300|ntrl MPPI default /300, training seed: successes|default mean|wide MPPI mean|aisle control mean|',
             '|---:|---:|---:|---:|---:|---|---:|---:|---:|']
    for g in rows_key:
        s, n, _ = rate(g, 'default', 'spot_to_spot')
        sw, nw, _ = rate(g, 'wide', 'spot_to_spot')
        sc, nc, _ = rate(g, 'default', 'aisle_to_spot')
        orc = sum(g['oracle'][k]['success'] for k in PS)
        lines.append(f"|{g[label]:g}|{g['cspace_gap']:.3f}|{g['truth']['detour_ratio_median']:.2f}|{g['straight']['spot_to_spot']['success']}|{orc}|"
                     f"{seed_cell(g, 'default', 'spot_to_spot')}|{pct(s, n):.1f}%|{pct(sw, nw):.1f}%|{pct(sc, nc):.1f}%|")
    lines += ['', f'|{label}|T/T* median|T/Euclid median|jump across divider / true jump|start descent cos vs truth|starts pointing through the wall /100|'
                  'divider contact / failures|not converged|speed corr all / near divider|', '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in rows_key:
        _, _, rows = rate(g, 'default', 'spot_to_spot')
        nfail = sum(r['n'] - r['success'] for r in rows)
        lines.append(f"|{g[label]:g}|{fmean(g, 't_over_tstar_median'):.2f}|{fmean(g, 't_over_euclid_median'):.2f}|{fmean(g, 'jump_ratio'):.2f}|"
                     f"{fmean(g, 'descent_cos_median'):.2f}|{fmean(g, 'start_points_through_wall'):.1f}|{sum(r['divider_contact'] for r in rows)}/{nfail}|"
                     f"{sum(r['n'] - r['converged'] for r in rows)}|{fmean(g, 'speed_corr'):.2f} / {fmean(g, 'speed_corr_divider'):.2f}|")
    return lines + ['']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='*', default=[])
    p.add_argument('--out', type=Path, default=ROOT / 'jeffrey/parking/report')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    groups = collect(find_runs(a.runs))
    lines = ['Success = converged and collision-free (exact clearance, <= 0.001 interpolation per coordinate). Bay-to-bay suite: 100 certified queries '
             '(50 A->B, 50 B->A) x planning seeds 0/1/2 per training seed. Oracle = the same MPPI on the true T* grid; straight = the straight segment.', '']
    summary = []
    for variant in sorted({k[0] for k in groups}):
        walls = sorted([g for k, g in groups.items() if k[0] == variant and k[1] == DEFAULT_W], key=lambda g: -g['wall'])
        if walls:
            lines += table(groups, f'ntrl-demo `{variant}`, car 0.30 x 0.13: divider thickness sweep', walls, 'wall')
        widths = sorted([g for k, g in groups.items() if k[0] == variant and k[2] == 0.005], key=lambda g: -g['car_w'])
        if len({g['car_w'] for g in widths}) > 1:
            lines += table(groups, f'ntrl-demo `{variant}`, divider 0.005: car width sweep (C-space gap = 0.005 + W)', widths, 'car_w')
    for g in groups.values():
        s, n, _ = rate(g, 'default', 'spot_to_spot')
        detail = {sd: dict(default=sum(g['seeds'][sd]['planning'][f'default/spot_to_spot/{k}']['success'] for k in PS),
                           wide=sum(g['seeds'][sd]['planning'][f'wide/spot_to_spot/{k}']['success'] for k in PS),
                           control=sum(g['seeds'][sd]['planning'][f'default/aisle_to_spot/{k}']['success'] for k in PS),
                           **{k: g['seeds'][sd]['field'][k] for k in ['t_over_tstar_median', 'jump_ratio', 'descent_cos_median', 'start_points_through_wall']},
                           run=g['seeds'][sd]['run']) for sd in ordered(g)}
        summary.append(dict(variant=g['variant'], car_w=g['car_w'], wall=g['wall'], cspace_gap=g['cspace_gap'], seeds=ordered(g), runs=g['runs'], per_seed=detail,
                            default=per_seed(g, 'default', 'spot_to_spot'), wide=per_seed(g, 'wide', 'spot_to_spot'), control=per_seed(g, 'default', 'aisle_to_spot'),
                            oracle=sum(g['oracle'][k]['success'] for k in PS), straight=g['straight']['spot_to_spot']['success'], success_pct=pct(s, n),
                            **{k: fmean(g, k) for k in ['t_over_tstar_median', 'jump_ratio', 'descent_cos_median', 'start_points_through_wall', 'speed_corr', 'speed_corr_divider']}))
    (a.out / 'tables.md').write_text('\n'.join(lines) + '\n')
    json.dump(sorted(summary, key=lambda r: (r['variant'], r['car_w'], -r['wall'])), open(a.out / 'summary.json', 'w'), indent=2)
    print('\n'.join(lines))
    figures(groups, a.out)


def scene_figure(out):
    """The lot for three divider thicknesses: walls, the car parked in both bays, the route, the divider's C-space band."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, w in zip(axes, [0.30, 0.05, 0.005]):
        for b in SC.boxes(w):
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], color='#444'))
        ax.add_patch(Rectangle((SC.X_D, -w / 2 - SC.CAR_W / 2), 0.48 - SC.X_D, w + SC.CAR_W, color='#cf222e', alpha=0.18, lw=0))
        for bay, colour in (('A', '#2da44e'), ('B', '#1f6feb')):
            y = SC.bay_y(w, bay)
            x0, y0 = SC.PARK_X - SC.CAR_L / 2, y - SC.CAR_W / 2
            ax.add_patch(Polygon([[x0, y0], [x0 + SC.CAR_L, y0], [x0 + SC.CAR_L, y0 + SC.CAR_W], [x0, y0 + SC.CAR_W]], color=colour, alpha=0.8))
            ax.text(SC.PARK_X, y, 'start' if bay == 'A' else 'goal', ha='center', va='center', fontsize=8, color='w')
        ya, yb = SC.bay_y(w, 'A'), SC.bay_y(w, 'B')
        for (a, b) in (((SC.PARK_X, ya), (SC.X_BACK, ya)), ((SC.X_BACK, ya), (SC.X_BACK, yb)), ((SC.X_BACK, yb), (SC.PARK_X, yb))):
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle='->', mutation_scale=12, color='#bf8700', lw=1.5))
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')
        ax.set_title(f'divider w = {w:g}; C-space gap w + W = {w + SC.CAR_W:.3f}', fontsize=9)
    fig.suptitle('Parking lot (1 x 1, ~3.3 car lengths), car 0.30 x 0.13; route: back out (-x), down (-y), drive in (+x). '
                 'Red band: positions of the car centre (theta = 0) where it overlaps the divider.', fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / 'scene.png', dpi=130)
    plt.close(fig)


def figures(groups, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    scene_figure(out)
    colours = {'metric': '#1f6feb', 'metric_arm': '#bf3989'}
    # success vs C-space gap, per training seed (metric: both sweeps merged by C-space gap)
    from matplotlib.ticker import FixedLocator, NullLocator
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    styles = {'87358': ('-', 'o'), '1': ('--', 's'), '2': (':', 'D')}
    for variant, colour in (('metric', '#2a78d6'), ('metric_arm', '#eb6834')):
        gs = [g for g in groups.values() if g['variant'] == variant and (g['car_w'] == DEFAULT_W or g['wall'] == 0.005)]
        for sd in SEED_ORDER:
            pts = sorted((g['cspace_gap'], pct(sum(g['seeds'][sd]['planning'][f'default/spot_to_spot/{k}']['success'] for k in PS), 300)) for g in gs if sd in g['seeds'])
            if pts:
                ls, mk = styles[sd]
                ax.plot(*zip(*pts), ls=ls, marker=mk, ms=4, color=colour, lw=1.4, label=f'{variant}, training seed {sd}')
    orc = sorted((g['cspace_gap'], pct(sum(g['oracle'][k]['success'] for k in PS), 300)) for g in groups.values() if g['variant'] == 'metric' and (g['car_w'] == DEFAULT_W or g['wall'] == 0.005))
    ax.plot(*zip(*orc), color='#4a5159', lw=1.2, label='oracle: same MPPI on true T*')
    ax.set_xscale('log')
    ticks = [0.015, 0.02, 0.03, 0.05, 0.1, 0.2, 0.4]
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_minor_locator(NullLocator()); ax.set_xticklabels([f'{t:g}' for t in ticks])
    ax.set_xlabel('C-space wall = divider + car width (divider sweep at car 0.13; car-width sweep at divider 0.005)')
    ax.set_ylabel('bay-to-bay success, ntrl-demo MPPI (%)')
    ax.set_ylim(-3, 103)
    ax.legend(fontsize=7, loc='center', bbox_to_anchor=(0.66, 0.42), frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'success_vs_gap.png', dpi=150)
    plt.close(fig)
    for variant in sorted({k[0] for k in groups}):
        for sweep, sel, xkey, xlabel in (('walls', lambda g: g['car_w'] == DEFAULT_W, 'wall', 'divider thickness w (lot 1 x 1, car 0.30 x 0.13)'),
                                         ('widths', lambda g: g['wall'] == 0.005, 'car_w', 'car width W (divider 0.005)')):
            gs = sorted([g for g in groups.values() if g['variant'] == variant and sel(g)], key=lambda g: g[xkey])
            if len(gs) < 2:
                continue
            xs = [g[xkey] for g in gs]
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            for setting, colour, label in (('default', '#1f6feb', 'ntrl-demo MPPI (defaults)'), ('wide', '#8250df', 'MPPI 200 samples, horizon 10, 400 steps')):
                ax[0].plot(xs, [pct(*rate(g, setting, 'spot_to_spot')[:2]) for g in gs], 'o-', color=colour, label=label)
                for x, g in zip(xs, gs):
                    ys = [pct(v, 300) for v in per_seed(g, setting, 'spot_to_spot')]
                    ax[0].scatter([x] * len(ys), ys, color=colour, s=10, alpha=0.4)
            ax[0].plot(xs, [pct(*rate(g, 'default', 'aisle_to_spot')[:2]) for g in gs], 's--', color='#57606a', label='aisle <-> bay control (defaults)')
            ax[0].plot(xs, [pct(sum(g['oracle'][k]['success'] for k in PS), 300) for g in gs], 'k:', label='oracle: same MPPI on true T*')
            ax[0].plot(xs, [g['straight']['spot_to_spot']['success'] for g in gs], 'x:', color='#cf222e', label='straight line')
            ax[0].set_xscale('log')
            ax[0].set_xlabel(xlabel)
            ax[0].set_ylabel('bay-to-bay success (%)')
            ax[0].set_ylim(-3, 103)
            ax[0].legend(fontsize=7)
            ax[0].set_title(f'ntrl-demo {variant}: planning success between the bays', fontsize=10)
            for key, colour, label in (('jump_ratio', '#1f6feb', 'T jump across divider / true jump'), ('t_over_tstar_median', '#bf8700', 'T(start, goal) / T*'),
                                       ('descent_cos_median', '#2da44e', 'cos(start descent, true descent)')):
                ax[1].plot(xs, [fmean(g, key) for g in gs], 'o-', color=colour, label=label)
            ax[1].axhline(1, color='k', lw=0.5)
            ax[1].set_xscale('log')
            ax[1].set_xlabel(xlabel)
            ax[1].set_title('field sharpness (1 = matches the true travel time)', fontsize=10)
            ax[1].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out / f'success_{variant}_{sweep}.png', dpi=130)
            plt.close(fig)
            # profiles across the divider
            ncol = (len(gs) + 1) // 2
            fig, axes = plt.subplots(2, ncol, figsize=(3 * ncol, 6), sharey=True, squeeze=False)
            for ax_, g in zip(axes.flat, sorted(gs, key=lambda g: -g[xkey])):
                t = np.load(Path(g['dirs'][0]) / 'truth.npz')
                ax_.plot(t['profile_y'], t['profile_star'], 'k', lw=1.2, label='T* (truth)')
                for s in sorted(g['seeds']):
                    f = np.load(Path(g['seeds'][s]['dir']) / 'field.npz')
                    ax_.plot(t['profile_y'], f['profile_pred'], lw=0.9, alpha=0.8, label=f'ntrl-demo seed {s}')
                half = g['wall'] / 2
                ax_.axvspan(-half - g['car_w'] / 2, half + g['car_w'] / 2, color='#cf222e', alpha=0.12)
                ax_.axvspan(-half, half, color='#cf222e', alpha=0.4)
                ax_.set_title(f'{xkey} = {g[xkey]:g}', fontsize=9)
                ax_.set_ylim(0, 3.0)
                ax_.set_xlabel('car y (x = 0.29, theta = 0)', fontsize=7)
            axes.flat[0].set_ylabel('T(bay-A pose, car at y)')
            axes.flat[0].legend(fontsize=6)
            fig.suptitle(f'{variant}: travel time from the bay-A pose along a vertical line through both bays (dark: divider; light: C-space obstacle)', fontsize=9)
            fig.tight_layout()
            fig.savefig(out / f'profiles_{variant}_{sweep}.png', dpi=130)
            plt.close(fig)
            # field slices with paths
            pick = [gs[0], gs[len(gs) // 2], gs[-1]]
            fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
            grid = np.linspace(-0.5, 0.5, 201)
            for col, g in enumerate(pick):
                s0 = sorted(g['seeds'])[0]
                t = np.load(Path(g['dirs'][0]) / 'truth.npz')
                f = np.load(Path(g['seeds'][s0]['dir']) / 'field.npz')
                paths = np.load(Path(g['seeds'][s0]['dir']) / 'paths.npz', allow_pickle=True)['default_spot_to_spot_0']
                ok = set(g['seeds'][s0]['planning']['default/spot_to_spot/0']['success_ids'])
                for row, (img, title) in enumerate(((f['slice_pred'], 'ntrl-demo T(., A)'), (t['slice_star'], 'true T*(., A)'))):
                    ax_ = axes[row, col]
                    im = ax_.contourf(grid, grid, np.clip(img.T, 0, 2.0), levels=np.linspace(0, 2.0, 21), cmap='viridis')
                    ax_.contour(grid, grid, np.clip(img.T, 0, 2.0), levels=np.linspace(0, 2.0, 21), colors='w', linewidths=0.3)
                    for b in SC.boxes(g['wall'], g['car_w']):
                        ax_.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], color='#444', zorder=3))
                    if row == 0:
                        for i, pth in enumerate(paths[:50]):
                            ax_.plot(pth[:, 0], pth[:, 1], color='#2da44e' if i in ok else '#cf222e', lw=0.6, zorder=4)
                    ax_.set_aspect('equal')
                    ax_.set_xlim(-0.5, 0.5)
                    ax_.set_ylim(-0.5, 0.5)
                    ax_.set_title(f'{"w" if xkey == "wall" else "W"} = {g[xkey]:g}: {title}', fontsize=8)
            fig.colorbar(im, ax=axes, shrink=0.6, label='travel time (theta = 0 slice; car centre)')
            fig.savefig(out / f'slices_{variant}_{sweep}.png', dpi=120)
            plt.close(fig)


if __name__ == '__main__':
    main()
