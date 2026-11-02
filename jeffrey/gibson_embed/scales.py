"""Per-scale probe of a saved multi-scale embedding: AUC (wall vs corridor pairs) for each snapshot K separately."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe as P  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('--name', default='diff-random')
    p.add_argument('--radius', type=float, default=0.06)
    a = p.parse_args()
    out = Path(a.out)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    occ = np.load(out / 'grid.npz')['occupancy']
    free = torch.from_numpy(~occ).to(dev)
    tr = np.load(out / 'truth.npz')
    sources = torch.from_numpy(tr['sources']).to(dev)
    dist = torch.from_numpy(tr['dist'].astype(np.float32)).to(dev)
    e = np.load(out / f'embedding_{a.name}.npz')
    feat, snaps = torch.from_numpy(e['feat']).to(dev), [int(k) for k in e['snapshots']]
    per = feat.shape[0] // len(snaps)
    feats = {f'K={k}': feat[i * per:(i + 1) * per] for i, k in enumerate(snaps)}
    feats['all scales'] = feat
    res = P.probe(feats, free, sources, dist, radius_cells=max(2, int(a.radius * occ.shape[0])))
    table = {k: dict(auc=round(v['auc_stratified'], 4), wall_over_corridor=round(v['wall_over_corridor_median'], 3), spearman_geo=round(v['spearman_geodesic'], 3))
             for k, v in res['features'].items()}
    (out / f'scales_{a.name}.json').write_text(json.dumps(dict(pairs={k: v for k, v in res.items() if k != 'features'}, scales=table), indent=2))
    for k, v in table.items():
        print(f'{k:>12}  AUC {v["auc"]:.3f}  wall/corridor feature-distance ratio {v["wall_over_corridor"]:.2f}  Spearman vs geodesic {v["spearman_geo"]:.3f}')


if __name__ == '__main__':
    main()
