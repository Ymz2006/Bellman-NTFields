"""Eikonal-only training (P-NTFields recipe) with free-space-averaged embeddings as extra input, and planning evaluation.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/gibson_embed/eikonal/pipeline.py --scene 0 --out <dir>

Steps: occupancy grids (embedding/geodesic grid at --res, collision grid at --coll-res) -> embeddings on the free cells
(averaged coordinates, averaged Fourier, random coarse, unmasked-blur control) -> start/goal pairs with geodesic truth
(uniform and wall-separated) -> for each input variant: P-NTFields Eikonal training on the P-NTFields samples
(sampled_points.npy / speed.npy produced by their sampler, their B matrix) -> bidirectional gradient planning, collision
check on the fine grid, path length against the geodesic. The pretrained P-NTFields Gib_multi checkpoint is evaluated
with the same planner as a reference.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
import grid as G  # noqa: E402
import diffuse  # noqa: E402
import truth  # noqa: E402
import model as M  # noqa: E402
import training  # noqa: E402
import plan as P  # noqa: E402

ROOT = HERE.parents[3]
EMBEDDINGS = {
    'avgcoords': dict(kind='coords'),
    'avgfourier': dict(kind='fourier', channels=32),
    'random': dict(kind='random', snapshots=(512, 2048, 8192)),
    'blur': dict(kind='random', snapshots=(512, 2048, 8192), masked=False),
}
VARIANTS = {
    # name: (use P-NTFields Fourier mapping, embedding names to concatenate)
    'fourier': (True, []),
    'fourier+avgcoords': (True, ['avgcoords']),
    'avgfourier': (False, ['avgfourier']),
    'fourier+avgfourier': (True, ['avgfourier']),
    'fourier+random': (True, ['random']),
    'fourier+blur': (True, ['blur']),
}
DEFAULT_VARIANTS = 'fourier,avgfourier,fourier+avgcoords,fourier+random,fourier+blur'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', default='0')
    p.add_argument('--out', required=True)
    p.add_argument('--mesh', default='')
    p.add_argument('--data', default='baselines/P-NTFields/datasets_retrain/gibson', help='folder with <scene>/sampled_points.npy, speed.npy')
    p.add_argument('--bdir', default='baselines/P-NTFields/datasets/gibson', help='folder with <scene>/B.npy')
    p.add_argument('--pretrained', default='baselines/P-NTFields/Experiments/Gib_multi/Model_Epoch_10000_ValLoss_1.221157e-01.pt')
    p.add_argument('--res', type=int, default=128)
    p.add_argument('--coll-res', type=int, default=256)
    p.add_argument('--snapshots', default='8,32,128,512,2048,8192')
    p.add_argument('--channels', type=int, default=16)
    p.add_argument('--epochs', type=int, default=10000)
    p.add_argument('--gamma', type=float, default=1e-3, help='P-NTFields viscosity coefficient (0 disables the Laplacian term)')
    p.add_argument('--mode', choices=['jet', 'fd', 'autograd'], default='jet', help='derivative computation in the loss (jet: forward-mode, one pass; fd: finite-difference Laplacian)')
    p.add_argument('--fd-cells', type=float, default=2.0, help='finite-difference step for --mode fd, in cells of --res')
    p.add_argument('--variants', default=DEFAULT_VARIANTS)
    p.add_argument('--n-starts', type=int, default=200)
    p.add_argument('--margin', type=float, default=0.5 / 12, help='P-NTFields speed margin (units); pair end points keep this clearance')
    p.add_argument('--step', type=float, default=0.03)
    p.add_argument('--tol', type=float, default=0.06)
    p.add_argument('--max-iter', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'log.txt', 'a')

    def log(msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    dev = torch.device('cuda')
    torch.manual_seed(a.seed)
    log(f'args {vars(a)}')
    mesh_path = a.mesh or f'baselines/P-NTFields/datasets/gibson/{a.scene}/mesh_z_up_scaled.off'

    # 1. grids
    t0 = time.time()
    mesh = G.load_mesh(str(ROOT / mesh_path))
    occ_g, _ = G.build(mesh, 3, a.res, 0.0)
    occ_c, _ = G.build(mesh, 3, a.coll_res, 0.0)
    free_g_np, free_c_np = ~occ_g, ~occ_c
    clear_c = G.clearance_cells(free_c_np)
    labels_c, _ = G.components(free_c_np)
    info = dict(geodesic_grid=G.summary(occ_g), collision_grid=G.summary(occ_c), mesh=mesh_path, seconds=time.time() - t0)
    info['leak_check'] = G.leak_check(free_g_np, a.res, mesh=mesh, segs=None, count=1000, seed=a.seed)
    (out / 'grid.json').write_text(json.dumps(info, indent=2))
    np.savez_compressed(out / 'grid.npz', occupancy_geodesic=occ_g, occupancy_collision=occ_c)
    log(f'grids: {info}')
    free_g = torch.from_numpy(free_g_np).to(dev)
    occ_c_t = torch.from_numpy(occ_c).to(dev)

    # 2. embeddings
    snapshots = tuple(int(s) for s in a.snapshots.split(','))
    variants = a.variants.split(',')
    needed = sorted({e for v in variants for e in VARIANTS[v][1]})
    feats, meta = {}, {}
    for name in needed:
        spec = dict(EMBEDDINGS[name])
        kind = spec.pop('kind')
        snaps = spec.pop('snapshots', snapshots)
        chans = spec.pop('channels', a.channels)
        t0 = time.time()
        feat, keys = diffuse.embed(free_g, kind=kind, r=1.0, snapshots=snaps, channels=chans, seed=a.seed, log=log, **spec)
        feats[name] = feat.half()
        meta[name] = dict(kind=kind, channels=int(feat.shape[0]), snapshots=keys, seconds=time.time() - t0, **spec)
        log(f'embedding {name}: {meta[name]}')
        mid = a.res // 2
        np.savez_compressed(out / f'embedding_{name}_slice.npz', feat=feat[:, :, :, mid].half().cpu().numpy(), snapshots=np.array(keys))
        del feat
        torch.cuda.empty_cache()
    (out / 'embeddings.json').write_text(json.dumps(meta, indent=2))

    # 3. pairs with geodesic truth
    t0 = time.time()
    dist_fn = lambda cells: truth.geodesic(free_g, cells.to(dev), log=log)  # noqa: E731
    pairs = P.choose_pairs(free_c_np, clear_c, labels_c, free_g_np, dist_fn, a.n_starts, a.margin * a.coll_res, a.seed, log=log)
    np.savez_compressed(out / 'pairs.npz', **pairs)
    log(f'pairs: {len(pairs["kind"])} in {time.time() - t0:.0f}s; wall pairs geodesic/euclid median '
        f'{np.median(pairs["geodesic"][pairs["kind"] == "wall"] / pairs["euclid"][pairs["kind"] == "wall"]):.2f}')
    torch.cuda.empty_cache()

    # 4. P-NTFields samples and B
    data_dir = ROOT / a.data / a.scene
    points = torch.tensor(np.load(data_dir / 'sampled_points.npy').astype(np.float32), device=dev)
    speed = torch.tensor(np.load(data_dir / 'speed.npy').astype(np.float32), device=dev)
    B = np.load(ROOT / a.bdir / a.scene / 'B.npy')
    log(f'samples {tuple(points.shape)} speed quantiles {np.quantile(speed.cpu().numpy(), [0, 0.5, 1]).round(3).tolist()}, B {B.shape}')

    results = {}

    def record(key, res):
        results[key] = res
        (out / 'results.json').write_text(json.dumps(results, indent=2))

    # 5. pretrained reference (their planner, and their weights through the autograd planner)
    pre = ROOT / a.pretrained
    if a.pretrained and pre.exists():
        sys.modules['pickle5'] = pickle
        sys.path.insert(0, str(ROOT / 'baselines/P-NTFields'))
        from models import model_res_sigmoid_multi as md
        ck = torch.load(pre, map_location='cpu', weights_only=False)
        holder = md.Model('x', 'x', 3, 2, device='cuda')
        holder.network = md.NN('cuda', 3)
        holder.network.load_state_dict(ck['model_state_dict'])
        holder.network.to(dev).eval()
        Bt = torch.tensor(B, dtype=torch.float32, device=dev)
        t0 = time.time()
        res = P.evaluate(lambda X: holder.Gradient(X, Bt), pairs, occ_c_t, dev, a.step, a.tol, a.max_iter, log, label='pretrained')
        record('pretrained', dict(planning=res, seconds=time.time() - t0, checkpoint=a.pretrained))
        net = M.EikNet(3, [M.FourierB(B)]).to(dev)
        M.load_pntfields_state(net, ck['model_state_dict'])
        net.eval()
        res = P.evaluate(lambda X: M.descent_direction(net, X), pairs, occ_c_t, dev, a.step, a.tol, a.max_iter, log, label='pretrained-autograd')
        record('pretrained-autograd', dict(planning=res, checkpoint=a.pretrained))
        del holder, net
        torch.cuda.empty_cache()

    # 6. variants
    for v in variants:
        use_fourier, emb = VARIANTS[v]
        torch.manual_seed(a.seed)
        features = ([M.FourierB(B)] if use_fourier else []) + [M.MaskedGridFeature(feats[e].float(), free_g) for e in emb]
        net = M.EikNet(3, features).to(dev)
        n_params = sum(p.numel() for p in net.parameters())
        t0 = time.time()
        hist = training.train(net, points, speed, a.epochs, log, gamma=a.gamma, seed=a.seed, laplacian=a.gamma > 0, label=v, mode=a.mode,
                              fd_h=a.fd_cells / a.res)
        train_s = time.time() - t0
        torch.save({k: t for k, t in net.state_dict().items() if not k.startswith('features.')}, out / f'model_{v}.pt')
        net.eval()
        t0 = time.time()
        res = P.evaluate(lambda X: M.descent_direction(net, X), pairs, occ_c_t, dev, a.step, a.tol, a.max_iter, log, label=v)
        record(v, dict(parameters=n_params, gamma=a.gamma, mode=a.mode, fd_cells=a.fd_cells, inputs=dict(fourier=use_fourier, embeddings=emb), train_seconds=train_s,
                       plan_seconds=time.time() - t0, history=hist, planning=res))
        del net, features
        torch.cuda.empty_cache()
    log('done')


if __name__ == '__main__':
    main()
