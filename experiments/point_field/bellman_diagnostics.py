"""Why does an exact speed plan worse? (1) speed shape vs true clearance, (2) path clearance and failures per run."""
import sys, json, glob, os
sys.path.insert(0, '/clean'); sys.path.insert(0, '/clean/method')
import numpy as np, torch
from experiments.point_field.point_field import Hierarchical, speed_of
from experiments.face_speed.speed_net import load_piece_speed
from experiments.face_speed.speed_accuracy import densify
scene = '/clean/experiments/cabinet/scenes/cabinet.yml'; dev = 'cuda:0'
H = Hierarchical(scene, None, dev); robot = H.robot
runs = {'baseline (release net)': 'registered_20261025T070429Z-b9a2383c', 'split separate': 'registered_20261025T175901Z-4ef50717',
        'exact label': 'registered_20261026T010842Z-face045d', 'hierarchical learned': 'registered_20261026T012043Z-4ecda217'}
out = {}
# 1. Speed shape on the bank source states (shared by every run): mean speed by true body clearance.
src = np.load('/clean/results/registered_20261026T010842Z-face045d/prepared/bank/source_q.npy', mmap_mode='r')
idx = np.random.default_rng(0).choice(len(src), 100000, replace=False); q = torch.tensor(np.asarray(src[np.sort(idx)]), device=dev)
with torch.no_grad():
    d = torch.cat([H.clearances(q[s:s+8192])[2] for s in range(0, len(q), 8192)]).cpu().numpy()
    preds = {}
    for name in ['baseline (release net)', 'split separate']:
        net = load_piece_speed(f'/clean/results/{runs[name]}/prepared/speed.pt', dev); preds[name] = torch.cat([net(q[s:s+8192]) for s in range(0, len(q), 8192)]).cpu().numpy()
    preds['exact label'] = speed_of(torch.as_tensor(d)).numpy()
bins = [(-1, .005), (.005, .02), (.02, .035), (.035, .05), (.05, .075), (.075, .1), (.1, .15), (.15, .25), (.25, 9)]
shape = []
for lo, hi in bins:
    m = (d >= lo) & (d < hi); shape.append(dict(lo_mm=lo*1000, hi_mm=hi*1000, n=int(m.sum()), **{k: float(v[m].mean()) for k, v in preds.items()}))
out['speed_by_true_clearance'] = shape
# 2. Paths: min body clearance along each RK4 / MPPI trajectory (densified), per run and suite; failures.
center, scale = robot.q_center.cpu().numpy(), robot.q_range.cpu().numpy()
def path_min(traj):
    traj = np.asarray(traj, np.float32)
    qn = (densify(traj, .01)-center)/scale
    with torch.no_grad(): return float(torch.cat([H.clearances(torch.tensor(qn[s:s+8192], device=dev))[2] for s in range(0, len(qn), 8192)]).min())
paths = {}
for name, run in runs.items():
    ev = f'/clean/results/{run}/evaluation'; row = {}
    for planner in ['rk4', 'mppi']:
        files = sorted(glob.glob(f'{ev}/bellman_{planner}_*.npz')); mins, succ, reasons, solved = [], [], {}, set()
        for f in files:
            z = np.load(f, allow_pickle=True); score = json.load(open(f.replace('.npz', '.score.json'))); key = list(score)[0]; pq = score[key]['per_query']
            for t, qid in zip(z['trajectories'], z['query_ids'] if 'query_ids' in z.files else range(len(z['trajectories']))):
                r = pq[str(int(qid))]; mins.append(path_min(t)); succ.append(r['success'])
                for reason in r['reasons']: reasons[reason] = reasons.get(reason, 0)+1
                if r['success']: solved.add(os.path.basename(f).replace('.npz', '')+':'+str(int(qid)))
        mins, succ = np.array(mins), np.array(succ)
        row[planner] = dict(paths=len(mins), success=int(succ.sum()), median_min_clearance_mm=float(np.median(mins)*1000), share_min_below_5mm=float((mins < .005).mean()),
                            share_min_below_20mm=float((mins < .02).mean()), median_min_clearance_success_mm=float(np.median(mins[succ])*1000) if succ.any() else None, reasons=reasons)
        paths.setdefault(name, {})[planner] = solved
    out.setdefault('paths', {})[name] = row; print(name, json.dumps(row), flush=True)
pair = {}
for planner in ['rk4', 'mppi']:
    b = paths['baseline (release net)'][planner]
    pair[planner] = {n: dict(gained=len(paths[n][planner]-b), lost=len(b-paths[n][planner])) for n in runs if n != 'baseline (release net)'}
out['paired_vs_baseline'] = pair
json.dump(out, open('/outputs/diag.json', 'w'), indent=1); print(json.dumps(out['speed_by_true_clearance'], indent=0)); print(pair)
