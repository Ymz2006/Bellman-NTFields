"""Extra findings-page figures (called by page.py), page-sized (7.6 in wide, 8-9 pt type) like page.py's own:
far-side levels across the divider, narrow-car paths, and the widening-car paths and profiles. Each writes
report/page_<name>.png."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plan as P  # noqa: E402
import scene as SC  # noqa: E402
from report import PS, collect, find_runs  # noqa: E402

INK, PASS, FAIL, START, WALLC, GREEN, GREY = '#15181b', '#2a78d6', '#cf222e', '#1baf7a', '#4a4f55', '#1a7f37', '#57606a'
BAY = 0.23                      # widening sweep: fixed bay width, divider 0.01
GX = np.linspace(-0.5, 0.5, 201)


def _style():
    plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                         'legend.fontsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9aa0a6', 'axes.linewidth': 0.8})


def _wkey(g, gap):
    """Widening-sweep group whose side gap (BAY - car width) / 2 is closest to gap."""
    return min(g, key=lambda k: abs((BAY - k[1]) / 2 - gap))


def _succ(planning, key):
    return sum(planning[f'{key}/{k}']['success'] for k in PS)


def _lot(ax, wall, car_w, bay_w=None, route=False):
    for b in SC.boxes(wall, car_w, bay_w):
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], color=WALLC, lw=0, zorder=1))
    half = wall / 2 + car_w / 2
    ax.add_patch(Rectangle((SC.X_D - SC.CAR_L / 2, -half), 0.48 - SC.X_D + SC.CAR_L / 2, 2 * half, color=FAIL, alpha=0.22, lw=0, zorder=2))
    ya, yb = SC.bay_y(wall, 'A', car_w, bay_w), SC.bay_y(wall, 'B', car_w, bay_w)
    ax.add_patch(Rectangle((SC.PARK_X - SC.CAR_L / 2, ya - car_w / 2), SC.CAR_L, car_w, color=START, alpha=0.9, lw=0, zorder=3))
    ax.add_patch(Rectangle((SC.PARK_X - SC.CAR_L / 2, yb - car_w / 2), SC.CAR_L, car_w, fill=False, ec=PASS, lw=1.2, zorder=3))
    if route:
        for p0, p1 in (((SC.PARK_X, ya), (SC.X_BACK, ya)), ((SC.X_BACK, ya), (SC.X_BACK, yb)), ((SC.X_BACK, yb), (SC.PARK_X, yb))):
            ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='->', mutation_scale=9, color='#b07d00', lw=1.3, zorder=4))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    for s in ax.spines.values():
        s.set_visible(False)
    return ya, yb, 2 * half


def _paths(ax, paths, ok, n=50):
    """Car-centre paths; stuck paths (not converged) lose the appended goal and get an x where the car stopped."""
    stuck = 0
    for i in range(n):
        p = paths[i]
        if np.linalg.norm(p[-2] - p[-1]) > 0.011:
            p = p[:-1]
            ax.plot(p[-1, 0], p[-1, 1], 'x', color=FAIL, ms=4.5, mew=1.1, zorder=6)
            stuck += 1
        ax.plot(p[:, 0], p[:, 1], color=GREEN if i in ok else FAIL, lw=0.7, alpha=0.75, zorder=5)
    return stuck


def far_side(rep):
    """T(bay-A pose, car) along x = 0.29 through both bays: truth vs field for thick, threshold and thinnest C-space walls."""
    _style()
    g = collect(find_runs([]))
    ylim = 1.5

    def load(key, sd):
        info = g[key]['seeds'][sd]
        d = Path(info['dir'])
        tr, fl = np.load(d.parent / 'truth.npz'), np.load(d / 'field.npz')
        return tr['profile_y'], tr['profile_star'], fl['profile_pred'], _succ(info['planning'], 'default/spot_to_spot')

    def panel(ax, key, seeds, title, numbered=False):
        _, car_w, wall = key
        C, yA = wall + car_w, SC.bay_y(wall, 'A', car_w)
        yb, lo, hi = -yA, -yA - 0.035, yA + 0.035
        ax.axvspan(-C / 2, C / 2, color=FAIL, alpha=0.13, lw=0)
        tx = ax.get_xaxis_transform()
        for xpos, text, colour in (((C / 2 + hi) / 2, 'bay A', INK), (0, 'divider', FAIL), ((-C / 2 + lo) / 2, 'bay B', INK)):
            ax.text(xpos, 1.01, text, transform=tx, ha='center', va='bottom', fontsize=8, color=colour)
        y, ts, _, _ = load(key, seeds[0][0])
        inside = np.abs(y) <= C / 2
        tb = np.interp(yb, y, ts)
        ax.plot(y, np.minimum(ts, ylim), color=INK, lw=1.8, label=f'truth: bay B at {tb:.2f}')
        ax.plot([yb], [tb], 'o', color=INK, ms=5, zorder=5)
        off = [f'truth {ts[inside].max():.1f}'] if ts[inside].max() > ylim else []
        peaks = {}
        for sd, colour in seeds:
            y, _, tf, succ = load(key, sd)
            fb = np.interp(yb, y, tf)
            ax.plot(y, np.minimum(tf, ylim), color=colour, lw=1.5, label=f'seed {sd} ({succ}/300): bay B at {fb:.2f}')
            ax.plot([yb], [fb], 'o', color=colour, ms=5, zorder=5)
            k = np.argmax(np.where(inside, tf, -np.inf))
            peaks[sd] = (y[k], tf[k])
            if tf[k] > ylim:
                off.append(f'field {tf[k]:.1f}')
        if off:
            ax.text(0, ylim * 0.95, 'peak off scale:\n' + '\n'.join(off), ha='center', va='top', fontsize=7,
                    bbox=dict(fc='white', ec='none', alpha=0.9), zorder=6)
        ax.axvline(yA, color=START, lw=0.9, ls=':')
        ax.text(yA, 0.12, 'start ', color=START, fontsize=7, va='bottom', ha='right')
        if numbered:
            kw = dict(fontsize=7.5, arrowprops=dict(arrowstyle='->', lw=0.7, color='#555'), color='#333')
            ax.annotate('1 just outside\nthe wall', xy=(C / 2 + 0.002, float(np.interp(C / 2 + 0.002, y, ts))), xytext=(C / 2 + 0.052, 0.62), ha='center', **kw)
            py, pv = peaks['2']
            ax.annotate('2 peak inside\nthe wall', xy=(py, pv), xytext=(-C / 2 - 0.045, 1.22), ha='center', **kw)
            ax.annotate('3 bay B centre', xy=(yb, 0.37), xytext=(yb + 0.012, 0.13), ha='center', **kw)
        ax.set_xlim(hi, lo)
        ax.set_ylim(0, ylim)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_title(title, pad=14)
        ax.set_xlabel("car's y on the line")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.6), gridspec_kw=dict(hspace=0.75, wspace=0.28))
    ax = axes[0, 0]
    _lot(ax, 0.005, 0.03, route=True)
    ya, yb = SC.bay_y(0.005, 'A', 0.03), SC.bay_y(0.005, 'B', 0.03)
    ax.plot([SC.PARK_X, SC.PARK_X], [ya + 0.06, yb - 0.06], ls='--', color=FAIL, lw=1.3, zorder=6)
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5)
    ax.set_title('The scene (C-space wall 0.035)', pad=14)
    ax.set_xlabel("green: start, blue outline: bay B, orange: real route;\nred dashed: line the car's centre slides along", fontsize=7.5)
    panel(axes[0, 1], ('metric', 0.13, 0.005), [('87358', PASS)], 'Thick wall (0.135)')
    panel(axes[1, 0], ('metric', 0.03, 0.005), [('87358', PASS), ('2', FAIL)], 'Same wall (0.035), two seeds', numbered=True)
    panel(axes[1, 1], ('metric', 0.01, 0.005), [('87358', FAIL)], 'Thinnest wall (0.015)')
    for ax in (axes[0, 1], axes[1, 0]):
        ax.set_ylabel('T from the start pose')
    fig.savefig(rep / 'page_far_side.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def narrow_car(rep):
    """Planned paths at four car widths (divider 0.005, training seed 87358, planning seed 0, bay A -> bay B)."""
    _style()
    g = collect(find_runs([]))
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.6), gridspec_kw=dict(hspace=0.3, wspace=0.05))
    for ax, W in zip(axes.flat, [0.13, 0.03, 0.02, 0.01]):
        info = g[('metric', W, 0.005)]['seeds']['87358']
        _, _, band = _lot(ax, 0.005, W)
        ok = set(info['planning']['default/spot_to_spot/0']['success_ids'])
        stuck = _paths(ax, np.load(Path(info['dir']) / 'paths.npz', allow_pickle=True)['default_spot_to_spot_0'], ok)
        ax.set_xlim(-0.3, 0.5); ax.set_ylim(-0.27, 0.27)
        ax.set_title(f"car width {W:g}: {_succ(info['planning'], 'default/spot_to_spot')}/300 succeed\nno-go band {band:.3f}; "
                     f'{50 - len(ok & set(range(50)))} of the 50 shown fail ({stuck} stuck)')
    fig.savefig(rep / 'page_narrow_car.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def widen_paths(rep, seed='1', gaps=(0.05, 0.021, 0.0125)):
    """Fixed 0.23 bays, wider car: fine-MPPI paths in the lot (left) and in the bay's own frame (right), one row per gap."""
    _style()
    g = collect(find_runs([]), fixed_bays=True)
    fig, axes = plt.subplots(len(gaps), 2, figsize=(7.6, 2.75 * len(gaps)), gridspec_kw=dict(width_ratios=[1, 1.15], hspace=0.45, wspace=0.38))
    for row, gap in enumerate(gaps):
        key = _wkey(g, gap)
        W, gap = key[1], (BAY - key[1]) / 2
        info = g[key]['seeds'][seed]
        ok = set(info['planning']['fine/spot_to_spot/0']['success_ids'])
        paths = np.load(Path(info['dir']) / 'paths.npz', allow_pickle=True)['fine_spot_to_spot_0']
        ax = axes[row, 0]
        _lot(ax, 0.01, W, BAY)
        _paths(ax, paths, ok)
        ax.set_xlim(-0.3, 0.5); ax.set_ylim(-0.27, 0.27)
        ax.set_title(f"side gap {gap:.3g} (car {W:g}): {_succ(info['planning'], 'fine/spot_to_spot')}/300 succeed")
        sc = SC.Scene(0.01, 'cpu', torch.float64, W, BAY)
        bx = axes[row, 1]
        for i in range(50):
            p = paths[i]
            if np.linalg.norm(p[-2] - p[-1]) > 0.011:
                p = p[:-1]
            x = P.densify(torch.tensor(p, dtype=torch.float64), 0.001)
            d, _ = sc.clearance(x)
            hit = torch.nonzero(d < 0)
            x = x.numpy()
            back, side = x[0, 0] - x[:, 0], x[0, 1] - x[:, 1]           # bay A start: the divider is at lower y
            keep = back <= 0.45
            bx.plot(back[keep], side[keep], color=GREEN if i in ok else FAIL, lw=0.7, alpha=0.7)
            if len(hit) and back[int(hit[0])] <= 0.45:
                bx.plot(back[int(hit[0])], side[int(hit[0])], 'x', color=FAIL, ms=4.5, mew=1.1, zorder=5)
        for yv, text in ((gap, 'touches the divider'), (-gap, 'touches the outer wall')):
            bx.axhline(yv, color=WALLC, lw=0.9, ls='--')
            bx.text(0.2, yv, text, ha='center', va='bottom' if yv > 0 else 'top', fontsize=7, color=GREY)
        bx.axvline(0.34, color='#8c959f', lw=0.7)
        bx.text(0.345, -1.35 * gap, 'out of\nthe bay', ha='left', va='bottom', fontsize=7, color=GREY)
        bx.set_xlim(-0.01, 0.45); bx.set_ylim(-1.4 * gap, 1.4 * gap)
        bx.set_ylabel('slid toward the divider')
        bx.set_title('in the bay\'s frame: x = first contact')
        if row == len(gaps) - 1:
            bx.set_xlabel('backed out toward the aisle')
    fig.savefig(rep / 'page_widen_paths.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def widen_profiles(rep, seed='1', gaps=(0.025, 0.0125, 0.005)):
    """Fixed bays: T(bay-A pose, car) across both bays (x = 0.29) and along bay B's centre line, truth vs field."""
    _style()
    g = collect(find_runs([]), fixed_bays=True)
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.6), gridspec_kw=dict(hspace=0.62, wspace=0.32))
    for col, gap in enumerate(gaps):
        key = _wkey(g, gap)
        W, gap = key[1], (BAY - key[1]) / 2
        info = g[key]['seeds'][seed]
        d = Path(info['dir'])
        tr, fl = np.load(d.parent / 'truth.npz'), np.load(d / 'field.npz')
        succ = _succ(info['planning'], 'fine/spot_to_spot')
        colour = PASS if succ >= 150 else FAIL
        C, yA = 0.01 + W, SC.bay_y(0.01, 'A', W, BAY)
        y, ts, tf = tr['profile_y'], tr['profile_star'], fl['profile_pred']
        tb, fb = np.interp(-yA, y, ts), np.interp(-yA, y, tf)
        ylim = 1.6 * max(tb, fb)
        ax = axes[0, col]
        ax.axvspan(-C / 2, C / 2, color=FAIL, alpha=0.13, lw=0)
        ax.plot(y, np.minimum(ts, ylim), color=INK, lw=1.8)
        ax.plot(y, np.minimum(tf, ylim), color=colour, lw=1.5)
        ax.plot([-yA], [tb], 'o', color=INK, ms=5); ax.plot([-yA], [fb], 'o', color=colour, ms=5)
        ax.set_xlim(yA + 0.04, -yA - 0.04); ax.set_ylim(0, ylim)
        ax.set_xlabel("car's y at x = 0.29\n(bay A | divider | bay B)", fontsize=7.5)
        ax.set_title(f'side gap {gap:.3g}: {succ}/300\nbay B: truth {tb:.2f}, field {fb:.2f}')
        jB = int(round((-yA + 0.5) / 0.005))
        xs = (GX >= 0.0) & (GX <= 0.25)
        st, sf = np.polyfit(GX[xs], tr['slice_star'][xs, jB], 1)[0], np.polyfit(GX[xs], fl['slice_pred'][xs, jB], 1)[0]
        bx = axes[1, col]
        span = (GX >= -0.25) & (GX <= 0.33)
        bx.axvspan(SC.X_D - SC.CAR_L / 2, 0.33, color='#8c959f', alpha=0.10, lw=0)
        bx.plot(GX[span], tr['slice_star'][span, jB], color=INK, lw=1.8)
        bx.plot(GX[span], fl['slice_pred'][span, jB], color=colour, lw=1.5)
        bx.set_xlabel("car's x along bay B\n(aisle -> parked; grey = in the bay)", fontsize=7.5)
        bx.set_title(f'slope going in: truth {st:+.2f}\nfield {sf:+.2f}')
    axes[0, 0].set_ylabel('T across both bays')
    axes[1, 0].set_ylabel('T along bay B')
    handles = [plt.Line2D([], [], color=INK, lw=1.8), plt.Line2D([], [], color=PASS, lw=1.5), plt.Line2D([], [], color=FAIL, lw=1.5)]
    fig.legend(handles, ['truth', f'field, training seed {seed}, plans at least half', f'field, training seed {seed}, fails'], loc='lower center',
               ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.savefig(rep / 'page_widen_profiles.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def make_all(rep):
    far_side(rep)
    narrow_car(rep)
    widen_paths(rep)
    widen_profiles(rep)


if __name__ == '__main__':
    make_all(HERE / 'report')
