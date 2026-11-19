"""Figures for the gap-maze section (report/gap_*.png), from registered runs (report/gap_runs.json, gappage.py) on the 10 x 10 base.

    gap_gates.png    the gate mu(x) at heading 0 (mean over octaves) for the clearance / diffusion / learned gates, per-cell octaves,
                     training seed 87358: the whole x4 maze, and the lower-left quadrant of the x12 maze with 4 cells of the band
    gap_slices.png   T(., goal 0) at heading 0 at x12 for ntrl-demo, octaves always on and the three gates (seed 87358)
    gap_mazes10.png  copied from jeffrey/cross_warp/report (the five mazes with one true path each)

Occupancy is computed from the scene on the slice grid (8 nodes per cell) and cached in report/cache/ (never written into a prepare run).
"""
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = HERE / 'report'
CACHE = REPORT / 'cache'
sys.path.insert(0, str(HERE))
INK = '#1f2430'
_CHILD = len(sys.argv) > 1 and sys.argv[1] == 'occ'
TITLES = dict(ntrl='ntrl-demo', octc='octaves always on', geoc='clearance gate', diffc='diffusion gate', learnc='learned gate',
              oct8='octaves always on', geo8='clearance gate', diff8='diffusion gate', learn8='learned gate')


def occupancy(prepared, res):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f'occ_{Path(prepared).name}_{res}.npy'
    if f.exists():
        return np.load(f)
    if not _CHILD:                                      # gap.install is once per process: compute each maze in its own process
        import subprocess
        subprocess.run([sys.executable, __file__, 'occ', str(prepared), str(res)], check=True)
        return np.load(f)
    import torch
    import scene as SC
    scene, _ = SC.load_scene(prepared, 'cuda:0' if torch.cuda.is_available() else 'cpu')
    t = torch.linspace(-0.5, 0.5, res, device=scene.device)
    X, Y = torch.meshgrid(t, t, indexing='ij')
    q = torch.stack([X.reshape(-1), Y.reshape(-1), torch.zeros(res * res, device=scene.device)], 1)
    occ = (scene.clearance(q).reshape(res, res) < 0).cpu().numpy()
    np.save(f, occ)
    return occ


FAR_Q = 158          # x12 diagonal query from deep in the lower-left quadrant to the top-right one: every octave field stalls inside the start quadrant, ntrl-demo walks through walls
FAR_LABELS = ('ntrl', 'octc', 'geoc', 'learnc')
WIN = 9.0            # cells: the 5-cell quadrant and 4 cells of band


def far_fields(prepared, ids):
    import torch
    import scene as SC
    import features as FE
    sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
    import plan as P
    scene, _ = SC.load_scene(prepared, 'cuda:0')
    q = np.load(prepared / 'queries.npz')
    goal = torch.tensor(q['goal'][FAR_Q], device='cuda:0', dtype=torch.float32)
    res = int(WIN * 16) + 1
    xs = torch.linspace(0, WIN, res, device='cuda:0')
    X, Y = torch.meshgrid(xs, xs, indexing='ij')
    pts = scene.from_cells(X.reshape(-1), Y.reshape(-1), torch.zeros(res * res, device='cuda:0'))
    pts[:, 2] = float(q['start'][FAR_Q][2])
    out = {}
    for lab, rid in ids.items():
        net = FE.load_field(run_dir(rid) / 'training/checkpoint.pt', 'cuda:0')
        cost = P.field_cost(net)
        out[lab] = cost(torch.cat([pts, goal[None].expand(len(pts), 3)], 1)).reshape(res, res).cpu().numpy()
        del net
    np.savez(CACHE / 'far_fields.npz', **out)


def run_dir(rid):
    return ROOT / 'results' / f'registered_{rid}'


def pick(rows, k, label, seed=87358):
    for r in rows:
        if r['gap'] == k and r['label'] == label and r['seed'] == seed:
            return r
    return None


def gates_figure(rows, out):
    panels = []
    for k in (4, 12):
        for lab in ('geoc', 'diffc', 'learnc'):
            r = pick(rows, k, lab)
            if r and (run_dir(r['id']) / 'gate_slice.npy').exists():
                panels.append((k, lab, r))
    if not panels:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.9), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    im = None
    for (k, lab, r) in panels:
        ax = axes[(0 if k == 4 else 1), ('geoc', 'diffc', 'learnc').index(lab)]
        mu = np.load(run_dir(r['id']) / 'gate_slice.npy').astype(np.float32)
        occ = occupancy(ROOT / r['prepared'], mu.shape[0])
        n = r['n']
        if k == 12:                                   # lower-left quadrant (5 cells) + 4 cells of band
            res = mu.shape[0]
            m = int(round((5 + 4) / (n + 0.1) * (res - 1))) + 1
            mu, occ = mu[:m, :m], occ[:m, :m]
            ext = [-0.5, -0.5 + (m - 1) / (res - 1), -0.5, -0.5 + (m - 1) / (res - 1)]
        else:
            ext = [-0.5, 0.5, -0.5, 0.5]
        im = ax.imshow(mu.T, origin='lower', cmap='magma', vmin=0, vmax=1, extent=ext)
        ax.imshow(np.ma.masked_where(~occ.T, occ.T), origin='lower', cmap='gray_r', vmin=0, vmax=1, extent=ext, alpha=0.9)
        where = 'whole maze' if k == 4 else 'one quadrant + 4 band cells'
        ax.set_title(f'x{k} {TITLES[lab]} ({r["pct"]:.0f} %)\n{where}', fontsize=9.5, color=INK, loc='left')
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01, label='gate mu at heading 0 (mean over octaves)')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return [f'x{k} {lab}' for k, lab, _ in panels]


def slices_figure(rows, out, k=12, labels=('ntrl', 'octc', 'geoc', 'diffc', 'learnc')):
    have = [(lab, pick(rows, k, lab)) for lab in labels]
    have = [(lab, r) for lab, r in have if r]
    if not have:
        return None
    prepared = ROOT / have[0][1]['prepared']
    goal = np.load(prepared / 'goals.npy')[0]
    fig, axes = plt.subplots(1, len(have), figsize=(3.3 * len(have), 3.5), squeeze=False)
    vmax = None
    for ax, (lab, r) in zip(axes[0], have):
        T = np.load(run_dir(r['id']) / 'slice_goal0.npy').astype(np.float32)
        occ = occupancy(prepared, T.shape[0])
        vmax = vmax or float(np.quantile(T, 0.98))
        g = np.linspace(-0.5, 0.5, T.shape[0])
        ax.imshow(T.T, origin='lower', cmap='viridis', vmin=0, vmax=vmax, extent=[-0.5, 0.5, -0.5, 0.5])
        ax.contour(g, g, T.T, levels=24, colors='white', linewidths=0.3, alpha=0.6)
        ax.imshow(np.ma.masked_where(~occ.T, occ.T), origin='lower', cmap='gray_r', vmin=0, vmax=1, extent=[-0.5, 0.5, -0.5, 0.5])
        ax.plot(goal[0], goal[1], '*', color='#ff5c5c', ms=9)
        ax.set_axis_off()
        ax.set_title(f'{TITLES[lab]} ({r["pct"]:.0f} %)', fontsize=10, color=INK, loc='left')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return [lab for lab, _ in have]


def far_figure(rows, out):
    import subprocess
    ids = {lab: pick(rows, 12, lab)['id'] for lab in FAR_LABELS if pick(rows, 12, lab)}
    if len(ids) < len(FAR_LABELS):
        return None
    prepared = ROOT / pick(rows, 12, 'ntrl')['prepared']
    subprocess.run([sys.executable, __file__, 'far', str(prepared), json.dumps(ids)], check=True)
    F = np.load(CACHE / 'far_fields.npz')
    q = np.load(prepared / 'queries.npz')
    info = json.load(open(prepared / 'maze.json'))
    n, E = info['n'], info['E_cells']
    cells = lambda p: (np.asarray(p)[:, 0] * E + n / 2, np.asarray(p)[:, 1] * E + n / 2)
    occ = occupancy(prepared, 8 * n + 1)
    m = int(round(WIN / (n + 0.1) * (8 * n))) + 1                       # same window on the slice grid (8 nodes per cell over E cells)
    tp = np.load(prepared / 'truth_paths.npy', allow_pickle=True)[FAR_Q]
    fig, axes = plt.subplots(1, len(ids), figsize=(3.4 * len(ids), 3.6), squeeze=False)
    for ax, lab in zip(axes[0], ids):
        T = F[lab]
        g = np.linspace(0, WIN, T.shape[0])
        lo, hi = np.quantile(T, 0.01), np.quantile(T, 0.99)
        ax.imshow(T.T, origin='lower', cmap='viridis', vmin=lo, vmax=hi, extent=[0, WIN, 0, WIN])
        ax.contour(g, g, T.T, levels=np.linspace(lo, hi, 20), colors='white', linewidths=0.4, alpha=0.7)
        oc = occ[:m, :m]
        ext = [0, (m - 1) / (8 * n) * E, 0, (m - 1) / (8 * n) * E]
        ax.imshow(np.ma.masked_where(~oc.T, oc.T), origin='lower', cmap='gray_r', vmin=0, vmax=1, extent=ext)
        tx, ty = cells(tp)
        ax.plot(tx, ty, '--', color='#9ad1ff', lw=1.2)
        path = np.load(run_dir(ids[lab]) / 'paths_seed0.npy', allow_pickle=True)[FAR_Q]
        px, py = cells(np.asarray(path)[:-1])
        ax.plot(px, py, '-', color='#ff5c5c', lw=1.6)
        sx, sy = cells(q['start'][FAR_Q][None])
        ax.plot(sx, sy, 'o', color='#ff5c5c', ms=5)
        ax.set_xlim(0, WIN)
        ax.set_ylim(0, WIN)
        ax.set_axis_off()
        r = pick(rows, 12, lab)
        ok = np.load(run_dir(ids[lab]) / 'rows.npz')
        end = 'through walls' if ok['converged'][0, FAR_Q] else 'stalled'
        ax.set_title(f'{TITLES[lab]}: {end}', fontsize=9.5, color=INK, loc='left')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return list(ids)


def main():
    rows = json.load(open(REPORT / 'gap_runs.json'))['rows']
    made = dict(gates=gates_figure(rows, REPORT / 'gap_gates.png'), far=far_figure(rows, REPORT / 'gap_far.png'))
    src = ROOT / 'jeffrey/cross_warp/report/gap_mazes10.png'
    if src.exists():
        shutil.copyfile(src, REPORT / 'gap_mazes10.png')
    print(made)


if __name__ == '__main__':
    if _CHILD:
        occupancy(Path(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) > 1 and sys.argv[1] == 'far':
        far_fields(Path(sys.argv[2]), json.loads(sys.argv[3]))
    else:
        main()
