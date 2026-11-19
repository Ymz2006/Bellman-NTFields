"""Figures for the embed-warp report (report/*.png), from registered runs (report/runs.json) and prepare runs.

    gates_<scene>.png     the gate mu(x) at heading 0 over the scene, one panel per gate kind (seed 87358), obstacles in black
    slices_<scene>.png    T(., goal 0) at heading 0 for ntrl / oct / geo (seed 87358): where the field has structure
    scenes.png            the three scenes (robot-centre occupancy at heading 0)
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = HERE / 'report'
sys.path.insert(0, str(HERE))

INK = '#1f2430'
GATES = ['geo', 'kde', 'diff', 'curv', 'learn', 'learnb', 'geot', 'geow']
TITLES = dict(geo='clearance band', kde='band-sample density', diff='masked diffusion', curv='curvature of ntrl field', learn='learned MLP(u0)', learnb='learned, budget', geot='tight band', geow='wide band')


def occupancy(prepared, res):
    """Robot-centre occupancy at heading 0 (res x res) from the prepare run (clearance_heading0.npy) or the maze."""
    p = Path(prepared)
    f = p / 'clearance_heading0.npy'
    if f.exists():
        d = np.load(f).astype(np.float32)
    else:
        import torch
        import scene as SC
        scene, _ = SC.load_scene(p, 'cuda:0' if torch.cuda.is_available() else 'cpu')
        t = torch.linspace(-0.5, 0.5, 1024, device=scene.device)
        X, Y = torch.meshgrid(t, t, indexing='ij')
        q = torch.stack([X.reshape(-1), Y.reshape(-1), torch.zeros(1024 * 1024, device=scene.device)], 1)
        d = scene.clearance(q).reshape(1024, 1024).cpu().numpy()
        np.save(f, d.astype(np.float16))
    return d < 0


def runs_by(rows, scene, seed=87358):
    out = {}
    for r in rows:
        if r['scene'] == scene and r['seed'] == seed:
            out[r['label']] = r
    return out


def gates_figure(rows, scene, prepared, out):
    by = runs_by(rows, scene)
    labels = [g for g in GATES if g in by and (ROOT / 'results' / f'registered_{by[g]["id"]}' / 'gate_slice.npy').exists()]
    if not labels:
        return None
    occ = occupancy(prepared, 1024)
    n = len(labels)
    cols = min(4, n)
    rws = (n + cols - 1) // cols
    fig, axes = plt.subplots(rws, cols, figsize=(3.2 * cols, 3.3 * rws), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, lab in zip(axes.ravel(), labels):
        mu = np.load(ROOT / 'results' / f'registered_{by[lab]["id"]}' / 'gate_slice.npy').astype(np.float32)
        im = ax.imshow(mu.T, origin='lower', cmap='magma', vmin=0, vmax=1, extent=[-0.5, 0.5, -0.5, 0.5])
        ax.imshow(np.ma.masked_where(~occ.T, occ.T), origin='lower', cmap='gray_r', vmin=0, vmax=1, extent=[-0.5, 0.5, -0.5, 0.5], alpha=0.9)
        pct = by[lab]['pct']
        ax.set_title(f'{TITLES.get(lab, lab)} ({pct:.0f} %)', fontsize=10, color=INK, loc='left')
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01, label='gate mu at heading 0 (mean over octaves)')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return labels


def slices_figure(rows, scene, prepared, out, labels=('ntrl', 'oct', 'geo')):
    by = runs_by(rows, scene)
    labels = [l for l in labels if l in by]
    if not labels:
        return None
    occ = occupancy(prepared, 1024)
    goal = np.load(Path(prepared) / 'goals.npy')[0]
    fig, axes = plt.subplots(1, len(labels), figsize=(3.6 * len(labels), 3.6), squeeze=False)
    vmax = None
    for ax, lab in zip(axes[0], labels):
        T = np.load(ROOT / 'results' / f'registered_{by[lab]["id"]}' / 'slice_goal0.npy').astype(np.float32)
        vmax = vmax or float(np.quantile(T, 0.98))
        ax.imshow(T.T, origin='lower', cmap='viridis', vmin=0, vmax=vmax, extent=[-0.5, 0.5, -0.5, 0.5])
        ax.contour(np.linspace(-0.5, 0.5, T.shape[0]), np.linspace(-0.5, 0.5, T.shape[1]), T.T, levels=24, colors='white', linewidths=0.3, alpha=0.6)
        ax.imshow(np.ma.masked_where(~occ.T, occ.T), origin='lower', cmap='gray_r', vmin=0, vmax=1, extent=[-0.5, 0.5, -0.5, 0.5])
        ax.plot(goal[0], goal[1], 'o', color='#ff5c5c', ms=5)
        ax.set_axis_off()
        ax.set_title(f'{lab} ({by[lab]["pct"]:.0f} %)', fontsize=10, color=INK, loc='left')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return labels


def scenes_figure(prepares, out):
    fig, axes = plt.subplots(1, len(prepares), figsize=(3.4 * len(prepares), 3.4), squeeze=False)
    for ax, (name, p) in zip(axes[0], prepares.items()):
        occ = occupancy(p, 1024)
        ax.imshow(occ.T, origin='lower', cmap='Greys', extent=[-0.5, 0.5, -0.5, 0.5])
        ax.set_axis_off()
        ax.set_title(name, fontsize=10, color=INK, loc='left')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)


def main():
    REPORT.mkdir(exist_ok=True)
    data = json.load(open(REPORT / 'runs.json'))
    rows = data['rows']
    prepares = {}
    for r in rows:
        prepares.setdefault(r['scene'], ROOT / r['prepared'])
    made = {}
    for scene, prep in prepares.items():
        if scene.startswith('gap'):
            continue                                    # gap mazes: gapfigs.py
        made[f'gates_{scene}'] = gates_figure(rows, scene, prep, REPORT / f'gates_{scene}.png')
        made[f'slices_{scene}'] = slices_figure(rows, scene, prep, REPORT / f'slices_{scene}.png')
    scenes_figure({k: v for k, v in prepares.items() if not k.startswith('gap')}, REPORT / 'scenes.png')
    json.dump({k: v for k, v in made.items()}, open(REPORT / 'figs.json', 'w'), indent=1)
    print(made)


if __name__ == '__main__':
    main()
