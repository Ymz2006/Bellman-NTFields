"""Why does the DP chain lose to its own eikonal specialists? Implied speed 1/|grad_q T(q, goal)| of a segment field
along the frozen suites' certified witness paths (collision-free), against the frozen speed network S(q) the
specialists were trained on, split by the distance to the goal (which model of the field answers). Host venv:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/easy_cabinet/chain_vs_es.py \
      --scene cabinet_w20 --prepared results/registered_.../prepared label=results/registered_.../training/field.pt ...
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'method'), str(ROOT/'runtime/method'), str(ROOT/'jeffrey/segment_dp')]
import common; common.RELEASE = ROOT/'runtime'   # the release is mounted at /repo in the containers
from speed import load_speed
from fields import load_field
NORM = np.load(ROOT/'runtime/configs/joint_normalization.npz'); CENTER, RANGE = NORM['q_center'].astype(np.float32), NORM['q_range'].astype(np.float32)
SUITES = ['free_to_free', 'free_to_compartment', 'compartment_to_free', 'compartment_to_compartment', 'same_compartment', 'anywhere']


def densify(q, step):
    out = [q[0]]
    for i in range(1, len(q)):
        n = max(1, int(np.ceil(np.abs(q[i]-q[i-1]).max()/step))); out.extend(q[i-1]+(q[i]-q[i-1])*(k/n) for k in range(1, n+1))
    return np.asarray(out, np.float32)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--scene', default='cabinet_w20'); p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('fields', nargs='+', help='label=path/to/field.pt'); p.add_argument('--step', type=float, default=0.02); p.add_argument('--out', type=Path)
    a = p.parse_args(); dev = 'cuda:0'; tests = ROOT/'experiments/cabinet'/('tests'+a.scene[len('cabinet'):])
    speed = load_speed(a.prepared/'speed.pt', dev); fields = {k: load_field(Path(v), dev) for k, v in (f.split('=', 1) for f in a.fields)}
    bins = [0, 0.1, 0.2, 0.4, 0.8, 1.6, 9]; report = {}
    for suite in SUITES:
        z = np.load(tests/'suites'/f'{a.scene}_{suite}_100.npz', allow_pickle=True); qs, gs = [], []
        for i, w in enumerate(z['witness_trajectories']):
            d = densify(np.asarray(w, np.float32), a.step); qs.append(d); gs.append(np.repeat(z['goals_rad'][i][None], len(d), 0))
        q = torch.tensor((np.concatenate(qs)-CENTER)/RANGE, device=dev); g = torch.tensor((np.concatenate(gs)-CENTER)/RANGE, device=dev)
        dist = torch.linalg.vector_norm(q-g, dim=1).cpu().numpy()
        with torch.no_grad(): s = torch.cat([speed(q[i:i+8192]).reshape(-1) for i in range(0, len(q), 8192)]).cpu().numpy()
        far = dist > 0.02; row = dict(points=int(far.sum()), speed_net_mean=float(s[far].mean()))
        for name, field in fields.items():
            sp, tv = [], []
            for i in range(0, len(q), 8192):
                x = q[i:i+8192].clone().requires_grad_(True)
                with torch.enable_grad(): t = field(x, g[i:i+8192]); grad = torch.autograd.grad(t.sum(), x)[0]
                sp.append((1/grad.norm(dim=1).clamp_min(1e-8)).detach().cpu().numpy()); tv.append(t.detach().reshape(-1).cpu().numpy())
            sp = np.concatenate(sp); tv = np.concatenate(tv); lr = np.log(sp/np.maximum(s, 1e-6))
            # T along each path should fall monotonically towards the goal: share of steps where it rises
            rises, n = 0, 0; k = 0
            for d in qs:
                seg = tv[k:k+len(d)]; k += len(d); rises += int((np.diff(seg) > 0).sum()); n += len(seg)-1
            row[name] = dict(log_ratio_mean=float(lr[far].mean()), log_ratio_rmse=float(np.sqrt((lr[far]**2).mean())), optimistic_share=float((lr[far] > np.log(2)).mean()),
                             rising_step_share=rises/max(1, n), T_start_mean=float(np.mean([tv[sum(len(x) for x in qs[:j])] for j in range(len(qs))])),
                             by_distance={f'{lo}-{hi}': dict(n=int(m.sum()), log_ratio_mean=float(lr[m].mean()), optimistic_share=float((lr[m] > np.log(2)).mean())) for lo, hi in zip(bins[:-1], bins[1:]) for m in [far & (dist >= lo) & (dist < hi)] if m.any()})
        report[suite] = row
        print(suite, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'by_distance'}) for k, v in row.items()}), flush=True)
        for name in fields: print('   ', name, {k: (v['n'], round(v['log_ratio_mean'], 2), round(v['optimistic_share'], 2)) for k, v in row[name]['by_distance'].items()})
    if a.out: a.out.write_text(json.dumps(report, indent=1))


if __name__ == '__main__': main()
