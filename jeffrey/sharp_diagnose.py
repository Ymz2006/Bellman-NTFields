"""Bellman-sharp diagnostic (read-only): does the Bellman update raise T where P-NTFields tunnels through obstacles?

Probe pairs (q, g) of bank anchors are split by whether the straight joint-space segment q -> g is collision-free
(exact certification, guard thresholds 5 mm / 2 mm) or not. For every generation checkpoint: T(q, g) relative to
generation 0, and the one-step Bellman gap (backup - T) / T, per group. If the update fixes tunnelling, T must rise
more on colliding pairs than on free ones.

    sharp_diagnose.py --run DIR(with gen_XX.pt, local_cost_pnt.npy) --base base.pt --bank BANK --scene S --output out.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT/'method'):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
from jeffrey.geometry import ArmClearance
from jeffrey.bellman_sharp import Travel, pnt_module


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); p.add_argument('--base', type=Path, required=True); p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True); p.add_argument('--output', type=Path, required=True); p.add_argument('--pairs', type=int, default=8192); a = p.parse_args()
    dev = 'cuda:0'; rng = np.random.default_rng(1)
    source = np.load(a.bank/'source_q.npy', mmap_mode='r'); succ = np.load(a.bank/'successor_q.npy', mmap_mode='r'); safe = np.load(a.bank/'safe.npy', mmap_mode='r')
    cost = np.load(a.run/'local_cost_pnt.npy', mmap_mode='r'); n = len(cost)
    rows = np.sort(rng.choice(n, a.pairs, replace=False)); goals = rng.choice(n, a.pairs, replace=False)
    q = torch.as_tensor(np.asarray(source[rows]), device=dev); g = torch.as_tensor(np.asarray(source[goals]), device=dev)
    z = torch.as_tensor(np.asarray(succ[rows]), device=dev); c = torch.as_tensor(np.asarray(cost[rows]), device=dev); s = torch.as_tensor(np.asarray(safe[rows]), device=dev).bool()
    clear = ArmClearance(a.scene, None, 0., dev); free, _ = clear.certify(q, g)
    md = pnt_module().pnt_module(); out = dict(pairs=a.pairs, free_pairs=int(free.sum()), colliding_pairs=int((~free).sum()), generations=[])
    T0 = None
    for path in [a.base]+sorted(a.run.glob('gen_*.pt')):
        payload = torch.load(path, map_location=dev, weights_only=False); net = md.NN(dev, 6, payload['B_state_dict']); net.load_state_dict(payload['model_state_dict']); net.to(dev).float().eval()
        T = Travel(net, dev)
        with torch.no_grad():
            v = torch.cat([T(q[i:i+1024], g[i:i+1024]) for i in range(0, len(q), 1024)])
            tail = torch.cat([T(z[i:i+256].reshape(-1, 6), g[i:i+256, None].expand(-1, z.shape[1], -1).reshape(-1, 6)) for i in range(0, len(q), 256)]).reshape(c.shape)
            b = (c+tail).masked_fill(~s, torch.inf).amin(1)
        if T0 is None: T0 = v
        row = dict(checkpoint=path.name)
        for name, m in [('free', free), ('colliding', ~free)]:
            ok = m & torch.isfinite(b) & (v > 1e-2)
            row[name] = dict(T_over_T0_median=float((v[ok]/T0[ok]).median()), T_over_T0_mean=float((v[ok]/T0[ok]).mean()),
                             gap_median=float(((b[ok]-v[ok])/v[ok]).median()), gap_p95=float(((b[ok]-v[ok])/v[ok]).quantile(.95)), mean_T=float(v[ok].mean()))
        out['generations'].append(row); print(json.dumps(row), flush=True)
    a.output.write_text(json.dumps(out, indent=1)+'\n')


if __name__ == '__main__': main()
