"""Data for the warp-space viewer (report/warpspaces/): every gap study field's x, y map phi sampled on a grid of the
maze, so the viewer can draw the maze as ntrl-demo's encoder receives it. Inference only.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/cross_warp/warpspaces.py

Per gap scale k (one process per k: gap.install patches the maze module once) and method (gapreport.collect(),
training seed 87358): phi = warp(x, y, heading 0) at the vertices of a grid of h cells (h = 0.25 up to 28 x 28, 0.5
above) over [0, n] x [0, n] cells, in the encoder's input units (ntrl-demo itself: the identity, a 1 x 1 square).
report/warpspaces/k<k>.json: n, E, h, the wall boxes (cells), and per method its run id, heading factor, success
(pooled over the training seeds available) and u, v (float32, base64, row-major over x then y).
"""
import base64
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(HERE))
OUT = HERE / 'report/warpspaces'
METHODS = ['ntrl-demo', 'constructed', 'learned', 'README warp', 'warp + scaling', 'uniform']


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def compute(k):
    import gapreport as GR
    import gap
    gap.install(k)
    import warp as W
    data, preps = GR.collect()
    p = preps[k]
    info = json.loads((p['_dir'] / 'maze.json').read_text())
    n = info['n']
    E = n + 0.1
    h = 0.25 if n <= 28 else 0.5
    c = np.arange(0, n + 1e-9, h)
    X, Y = np.meshgrid(c, c, indexing='ij')
    q = torch.tensor(np.stack([(X.ravel() - n / 2) / E, (Y.ravel() - n / 2) / E, np.zeros(X.size)], 1), dtype=torch.float32, device='cuda:0')
    out = dict(k=k, n=n, E=E, h=h, nx=len(c), boxes=info['boxes_cells'], band=n - 16, methods=[])
    for m in METHODS:
        by = data[k].get(m, {})
        e = by.get(87358)
        if e is None:
            continue
        tot = [sum(by[s]['success_total'][i] for s in by) for i in (0, 1)]
        cats = {cname: [sum(by[s]['by_category'][cname][i] for s in by if cname in by[s]['by_category']) for i in (0, 1)] for cname in GR.CATS}
        entry = dict(name=m, run=e['_id'], seeds=len(by), success=tot, by_category=cats)
        if m == 'ntrl-demo':
            uv = q[:, :2].cpu().numpy()
            entry.update(heading=1.0, s=1.0, kind='identity')
        else:
            w = W.load_field(ROOT / 'results' / f'registered_{e["_id"]}' / 'training/checkpoint.pt').warp
            with torch.no_grad():
                uv = torch.cat([w(q[i:i + 65536]) for i in range(0, len(q), 65536)])[:, :2].cpu().numpy()
            entry.update(heading=float(w.s_p), s=float(w.s), kind='coupling' if w.xy else 'separable' if w.sep else 'uniform', learned=bool(w.cfg.get('learn')),
                         cfg={key: w.cfg.get(key) for key in ('s', 's_p', 'xy', 'sep', 'sep_norm', 'learn', 'k_sep', 'sep_scale', 'nodes', 'sep_nodes')})
            del w
        entry['u'], entry['v'] = b64(uv[:, 0]), b64(uv[:, 1])
        entry['extent'] = [float(uv[:, 0].max() - uv[:, 0].min()), float(uv[:, 1].max() - uv[:, 1].min())]
        out['methods'].append(entry)
        torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'k{k}.json').write_text(json.dumps(out))
    print(f'k={k}: n={n}, grid {len(c)}^2, methods {[m["name"] for m in out["methods"]]}, {(OUT / f"k{k}.json").stat().st_size / 1e6:.2f} MB')


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if '--child' in sys.argv:
        compute(int(args[0]))
        return
    for k in [int(a) for a in args] or [1, 2, 4, 8, 12]:
        subprocess.run([sys.executable, __file__, str(k), '--child'], check=True)


if __name__ == '__main__':
    main()
