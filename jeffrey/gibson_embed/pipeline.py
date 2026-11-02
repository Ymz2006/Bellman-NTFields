"""Free-space ball-average embedding test on a Gibson scene.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/gibson_embed/pipeline.py --mesh baselines/P-NTFields/datasets/gibson/0/mesh_z_up_scaled.off \
        --dim 2 --res 512 --out <dir>

Steps: occupancy grid (+ leak check) -> sources and geodesic truth -> embeddings (masked ball averaging and
controls) -> embedding probe -> pair-regression networks per input variant -> evaluation on held-out sources.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid as G  # noqa: E402
import diffuse  # noqa: E402
import truth  # noqa: E402
import probe as P  # noqa: E402
import train as T  # noqa: E402

EMBEDDINGS = {
    # name: (init kind, averaging options)
    'diff-random': dict(kind='random'),
    'diff-random-coarse': dict(kind='random', snapshots=(512, 2048, 8192)),  # drops the noise-like fine scales
    'blur-random': dict(kind='random', masked=False),
    'diff-random-leaky': dict(kind='random', r='leak'),
    'diff-random-los': dict(kind='random', r='leak', los=True),
    'diff-fourier': dict(kind='fourier', channels=32),
    'diff-coords': dict(kind='coords'),
    'spectral': dict(kind='random', channels=32, spectral=True),
}
VARIANTS_2D = ['coords', 'fourier', 'fourier-hi', 'diff-random', 'diff-random-coarse', 'blur-random', 'diff-random-leaky', 'diff-random-los',
               'diff-fourier', 'diff-coords', 'spectral', 'learned-grid', 'learned-grid-init', 'fourier+diff']
VARIANTS_3D = ['coords', 'fourier', 'fourier-hi', 'diff-random', 'diff-random-coarse', 'blur-random', 'diff-random-los', 'diff-fourier', 'diff-coords',
               'learned-grid', 'learned-grid-init', 'fourier+diff']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mesh', required=True)
    p.add_argument('--dim', type=int, choices=[2, 3], default=2)
    p.add_argument('--res', type=int, default=512)
    p.add_argument('--z', type=float, default=0.0)
    p.add_argument('--out', required=True)
    p.add_argument('--sources', type=int, default=96)
    p.add_argument('--test-sources', type=int, default=24)
    p.add_argument('--min-clearance', type=float, default=3.0, help='cells')
    p.add_argument('--max-clearance', type=float, default=0.05, help='units; sources are drawn near walls (0 = anywhere)')
    p.add_argument('--snapshots', default='8,32,128,512,2048,8192')
    p.add_argument('--channels', type=int, default=16)
    p.add_argument('--leak-radius', type=float, default=0.0, help='ball radius (cells) for the leaky/los variants; 0 = median wall thickness + 1')
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--batch', type=int, default=8192)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--grid-lr', type=float, default=5e-3)
    p.add_argument('--grid-channels', type=int, default=32)
    p.add_argument('--local-radius', type=float, default=0.15)
    p.add_argument('--probe-radius', type=float, default=0.06)
    p.add_argument('--eval-pairs', type=int, default=200000)
    p.add_argument('--variants', default='')
    p.add_argument('--train-counts', default='', help='extra training-source counts for a subset of variants, e.g. 24,8')
    p.add_argument('--sweep-variants', default='fourier,diff-random,diff-random-coarse,diff-random-los,blur-random,diff-fourier,diff-coords,spectral,learned-grid,fourier+diff')
    p.add_argument('--embeddings', default='')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save-embeddings', action='store_true')
    p.add_argument('--render-sources', type=int, default=2)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'log.txt', 'a')

    def log(msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed)
    log(f'args {vars(a)}')

    # 1. occupancy grid
    t0 = time.time()
    mesh = G.load_mesh(a.mesh)
    occ, segs = G.build(mesh, a.dim, a.res, a.z)
    free_np = ~occ
    clearance_np = G.clearance_cells(free_np)
    labels_np, sizes = G.components(free_np)
    info = G.summary(occ)
    info['leak_check'] = G.leak_check(free_np, a.res, mesh=mesh, segs=segs, count=3000 if a.dim == 2 else 1500, seed=a.seed)
    info['mesh'] = a.mesh
    info['z'] = a.z
    info['seconds'] = time.time() - t0
    (out / 'grid.json').write_text(json.dumps(info, indent=2))
    np.savez_compressed(out / 'grid.npz', occupancy=occ, clearance=clearance_np.astype(np.float32), labels=labels_np.astype(np.int32))
    log(f'grid: {info}')
    free = torch.from_numpy(free_np).to(dev)
    clearance = torch.from_numpy(clearance_np).float().to(dev)

    # 2. sources and geodesic truth
    t0 = time.time()
    max_clear = a.max_clearance * a.res if a.max_clearance else float('inf')
    src_np = truth.choose_sources(free_np, clearance_np, labels_np, a.sources, a.min_clearance, a.seed, max_clearance=max_clear)
    sources = torch.from_numpy(src_np).to(dev)
    dist = truth.geodesic(free, sources, log=log)
    finite = torch.isfinite(dist).float().mean().item()
    log(f'truth: {a.sources} sources (clearance {a.min_clearance:g}-{max_clear:g} cells), reachable fraction {finite:.3f}, {time.time() - t0:.0f}s')
    np.savez_compressed(out / 'truth.npz', sources=src_np, dist=dist.half().cpu().numpy())
    perm = torch.randperm(a.sources, generator=torch.Generator().manual_seed(a.seed + 1)).to(dev)
    test_ids, train_ids = perm[:a.test_sources], perm[a.test_sources:]

    # 3. embeddings
    thickness = info['wall_thickness_cells']['median']
    leak_r = a.leak_radius or math.floor(thickness) + 1
    snapshots = tuple(int(s) for s in a.snapshots.split(','))
    names = a.embeddings.split(',') if a.embeddings else list(EMBEDDINGS)
    if a.dim == 3 and not a.embeddings:
        names = ['diff-random', 'diff-random-coarse', 'blur-random', 'diff-random-los', 'diff-fourier', 'diff-coords']
    feats, meta = {}, {}
    for name in names:
        spec = dict(EMBEDDINGS[name])
        kind = spec.pop('kind')
        r = spec.pop('r', 1.0)
        r = leak_r if r == 'leak' else r
        chans = spec.pop('channels', a.channels)
        snaps = spec.pop('snapshots', snapshots)
        t0 = time.time()
        feat, keys = diffuse.embed(free, kind=kind, r=r, snapshots=snaps, channels=chans, seed=a.seed, log=log, **spec)
        feats[name] = feat.half()
        meta[name] = dict(kind=kind, r=r, channels=int(feat.shape[0]), snapshots=keys, seconds=time.time() - t0, **spec)
        log(f'embedding {name}: {meta[name]}')
        if a.save_embeddings or (a.dim == 2 and name == 'diff-random'):
            np.savez_compressed(out / f'embedding_{name}.npz', feat=feat.half().cpu().numpy(), snapshots=np.array(keys))
        elif a.dim == 3:
            mid = a.res // 2
            np.savez_compressed(out / f'embedding_{name}_slice.npz', feat=feat[:, :, :, mid].half().cpu().numpy(), snapshots=np.array(keys))
        del feat
        torch.cuda.empty_cache()
    (out / 'embeddings.json').write_text(json.dumps(meta, indent=2))

    # 4. probe (embedding only)
    probe_feats = dict(coords=diffuse.cell_centres(free.shape, dev) * free, fourier=diffuse.initial('fourier', free, 128, a.seed, sigma=0.5),
                       fourier_hi=diffuse.initial('fourier', free, 128, a.seed, sigma=4.0), **feats)
    pr = P.probe(probe_feats, free, sources, dist, radius_cells=max(2, int(a.probe_radius * a.res)), seed=a.seed)
    (out / 'probe.json').write_text(json.dumps(pr, indent=2))
    log('probe: ' + json.dumps({k: round(v['auc_stratified'], 3) for k, v in pr['features'].items()}))
    del probe_feats

    # 5. networks
    local_cells = int(a.local_radius * a.res)
    sampler = T.Sampler(free, clearance, dist, sources, local_cells, a.seed)
    variants = a.variants.split(',') if a.variants else (VARIANTS_2D if a.dim == 2 else VARIANTS_3D)
    variants = [v for v in variants if v.replace('fourier+', '') in feats or v in ('coords', 'fourier', 'fourier-hi', 'learned-grid', 'learned-grid-init', 'fourier+diff')]
    results = {}
    renders = {}
    counts = [len(train_ids)] + [int(c) for c in a.train_counts.split(',') if c]
    sweep = a.sweep_variants.split(',')
    jobs = [(v, n) for n in counts for v in variants if n == len(train_ids) or v in sweep]
    for v, n in jobs:
        key = v if n == len(train_ids) else f'{v}@{n}'
        ids = train_ids[:n]
        torch.manual_seed(a.seed)
        features = []
        if v == 'fourier':
            features = [T.FourierFeature(a.dim, 64, 0.5, a.seed)]
        elif v == 'fourier-hi':
            features = [T.FourierFeature(a.dim, 64, 4.0, a.seed)]
        elif v == 'learned-grid':
            g = torch.Generator().manual_seed(a.seed)
            features = [T.MaskedGridFeature(0.1 * torch.randn((a.grid_channels,) + tuple(free.shape), generator=g).to(dev), free, learnable=True)]
        elif v == 'learned-grid-init':
            # coarsest snapshots: initialising from the fine scales (unit-variance, noise-like) let the net memorise cells
            init = feats['diff-random'][-a.grid_channels:].float()
            features = [T.MaskedGridFeature(init, free, learnable=True)]
        elif v == 'fourier+diff':
            features = [T.FourierFeature(a.dim, 64, 0.5, a.seed), T.MaskedGridFeature(feats['diff-random'], free)]
        elif v != 'coords':
            features = [T.MaskedGridFeature(feats[v], free)]
        net = T.PairNet(a.dim, features).to(dev)
        n_params = sum(p.numel() for p in net.parameters())
        t0 = time.time()
        hist = T.train(net, sampler, ids, a.steps, a.batch, a.lr, a.grid_lr, log, label=key)
        res = dict(parameters=n_params, train_sources_count=int(n), seconds=time.time() - t0, history=hist,
                   test_sources=T.evaluate(net, sampler, test_ids, a.eval_pairs, a.batch, a.local_radius),
                   train_sources=T.evaluate(net, sampler, ids, a.eval_pairs, a.batch, a.local_radius))
        results[key] = res
        ts, tr = res['test_sources'], res['train_sources']
        log(f"result {key}: test all {ts['all']['mean_abs_log']:.4f} wall {ts.get('wall', {}).get('mean_abs_log', float('nan')):.4f} "
            f"detour {ts.get('detour', {}).get('mean_abs_log', float('nan')):.4f} corridor {ts.get('corridor', {}).get('mean_abs_log', float('nan')):.4f} "
            f"recall@2 {ts.get('detect_wall', {}).get('recall_at_log2', float('nan')):.3f} | train all {tr['all']['mean_abs_log']:.4f} wall {tr.get('wall', {}).get('mean_abs_log', float('nan')):.4f}")
        (out / 'results.json').write_text(json.dumps(results, indent=2))
        if a.dim == 2 and n == len(train_ids):
            renders[v] = np.stack([T.render(net, sampler, int(test_ids[i])).cpu().numpy() for i in range(a.render_sources)])
            np.savez_compressed(out / 'renders.npz', sources=test_ids[:a.render_sources].cpu().numpy(), **renders)
        del net, features
        torch.cuda.empty_cache()
    log('done')


if __name__ == '__main__':
    main()
