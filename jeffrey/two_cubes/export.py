"""Page data for the two-cube planning viewer (jeffrey/two_cubes/README.md): the frozen suites of cabinet_twocubes, and
every planner's paths and release-scorer verdicts from one registered ntrl-cabinet run. Read-only on both. Writes data.json
(pairs, verdicts, totals) and paths_<group>.json (packed paths; offsets in data.json) beside it.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/two_cubes/export.py \
        --run results/registered_<ntrl-cabinet run> --output jeffrey/two_cubes/viewer/data.json

Per query and planner run: the path densified every --check rad and resampled evenly to at most --keep points for display
(ends kept), the gripper (grasp frame) position at each, and per displayed segment the smallest obstacle clearance of the padded
arm spheres over its dense samples (the release sphere model, shoulder spheres 0/1 ignored as in
the label code). The success verdicts and failure reasons are the release scorer's (<label>.score.json), not this check.
"""
import argparse, base64, json, sys
from pathlib import Path
import numpy as np
import torch
import yaml
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from experiments.face_speed import robot as R

TESTS = ROOT/'experiments/cabinet/tests_twocubes'; SCENE = ROOT/'experiments/cabinet/scenes/cabinet_twocubes.yml'
SUITES = ['free_to_free', 'free_to_compartment', 'compartment_to_free', 'compartment_to_compartment', 'same_compartment', 'anywhere']
PLANNERS = [('rk4', 'metric_rk4_{s}', [None]), ('ntrl', 'metric_ntrl_mppi_{s}_seed{k}', [0, 1, 2]), ('mppi', 'metric_mppi_{s}_seed{k}', [0, 1, 2])]
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


class Clearance:
    def __init__(self, dev):
        self.robot = R.Robot().to(dev); doc = yaml.safe_load(SCENE.read_text()); self.names = [c['name'] for c in doc['cuboids']]
        self.cen = torch.tensor([c['pose'][:3] for c in doc['cuboids']], dtype=torch.float32, device=dev)
        self.half = torch.tensor([c['dims'] for c in doc['cuboids']], dtype=torch.float32, device=dev)/2
        self.keep = torch.ones(len(self.robot.sphere_radii), dtype=torch.bool, device=dev); self.keep[list(R.IGNORED_SPHERES)] = False; self.dev = dev

    @torch.no_grad()
    def __call__(self, q, chunk=50000):
        """(N,) smallest obstacle clearance, (N,) index of the nearest cuboid, (N, 3) gripper position."""
        d, w, g = [], [], []
        for s in range(0, len(q), chunk):
            x = torch.as_tensor(q[s:s+chunk], dtype=torch.float32, device=self.dev); t = self.robot.transforms(x)
            sp = self.robot.link_spheres(x, t)[:, self.keep]
            delta = sp[:, :, None, :3]-self.cen[None, None]; out = (delta.abs()-self.half).clamp_min(0).norm(dim=-1)
            inside = (delta.abs()-self.half).max(-1).values.clamp_max(0)            # signed: negative depth inside a box
            signed = out+inside-sp[:, :, 3:4]; per_box = signed.min(1).values      # (B, boxes)
            m, i = per_box.min(1); d.append(m.cpu()); w.append(i.cpu()); g.append(self.robot.ee_position(x, t).cpu())
        return torch.cat(d).numpy(), torch.cat(w).numpy(), torch.cat(g).numpy()


def densify(path, step):
    """Joint-space polyline sampled every <= step rad (max norm), and the segment index of each sample."""
    qs, seg = [path[:1]], [np.zeros(1, int)]
    for i in range(len(path)-1):
        n = max(1, int(np.ceil(np.abs(path[i+1]-path[i]).max()/step))); t = np.arange(1, n+1)[:, None]/n
        qs.append(path[i]+(path[i+1]-path[i])*t); seg.append(np.full(n, i))
    return np.concatenate(qs), np.concatenate(seg)


def pack(path, clear, keep, step):
    """Display points: the path densified every <= step rad (max norm), resampled evenly to at most keep points, so the
    gripper curve between waypoints is drawn; per displayed segment the min clearance and nearest cuboid over its dense samples."""
    dense, _ = densify(np.asarray(path, np.float64), step); d, w, g = clear(dense)
    idx = np.unique(np.round(np.linspace(0, len(dense)-1, min(keep, len(dense)))).astype(int)); sd, sw = [], []
    for a_, b_ in zip(idx[:-1], idx[1:]):
        j = a_+int(np.argmin(d[a_:b_+1])); sd.append(d[j]); sw.append(w[j])
    return dict(q=dense[idx].astype(np.float32), g=g[idx].astype(np.float32), d=np.array(sd, np.float32), w=np.array(sw, np.int8), dmin=float(d.min()), wmin=int(w[np.argmin(d)]))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--keep', type=int, default=150); p.add_argument('--check', type=float, default=0.005); p.add_argument('--path-seeds', type=int, nargs='+', default=[0, 1, 2]); a = p.parse_args()
    dev = 'cuda:0'; clear = Clearance(dev); ev = a.run/'evaluation'; manifest = json.loads((TESTS/'suite_manifest.json').read_text())
    doc = yaml.safe_load(SCENE.read_text()); status = json.loads((a.run/'status.json').read_text())
    Q, G, Dd, W = [], [], [], []; off = [0, 0]; sizes = {}
    def put(pk):
        o = dict(o=off[0], n=len(pk['q']), so=off[1]); Q.append(pk['q']); G.append(pk['g']); Dd.append(pk['d']); W.append(pk['w'])
        off[0] += len(pk['q']); off[1] += len(pk['d']); return dict(o, dmin=round(pk['dmin'], 4), wmin=pk['wmin'])
    suites = []
    for item in manifest['suites']:
        s = item['task']; z = np.load(TESTS/'suites'/item['file'], allow_pickle=True); n = int(item['n']); queries = []
        Q, G, Dd, W = [], [], [], []; off[:] = [0, 0]
        for i in range(n):
            wit = put(pack(np.asarray(z['witness_trajectories'][i], np.float64), clear, a.keep, a.check))
            queries.append(dict(start=z['starts_rad'][i].round(5).tolist(), goal=z['goals_rad'][i].round(5).tolist(), sr=str(z['start_region'][i]), gr=str(z['goal_region'][i]),
                                direct=bool(z['direct_line_strict_feasible'][i]), src=str(z['witness_source'][i]), wcs=float(z['witness_cspace_length'][i]), wee=float(z['witness_ee_length'][i]), witness=wit, runs={}))
        for key, pattern, seeds in PLANNERS:
            for k in seeds:
                label = pattern.format(s=s, k=k); sc = json.loads((ev/f'{label}.score.json').read_text())[label]['per_query']
                tr = np.load(ev/f'{label}.npz', allow_pickle=True); traj = tr['trajectories']; ids = tr['query_ids'].tolist() if 'query_ids' in tr.files else list(range(len(traj)))
                fb = tr['used_fallback'] if 'used_fallback' in tr.files else None
                conv = tr['converged'] if 'converged' in tr.files else None
                for j, i in enumerate(ids):
                    r = sc[str(i)]; row = dict(ok=bool(r['success']), why=r['reasons'], ee=r.get('ee_len'), cs=r.get('cs_len'), conv=None if conv is None else bool(conv[j]), fb=None if fb is None else bool(fb[j]))
                    if k is None or k in a.path_seeds: row['path'] = put(pack(np.asarray(traj[j], np.float64), clear, a.keep, a.check))
                    queries[i]['runs'][f'{key}{"" if k is None else k}'] = row
            print(s, key, flush=True)
        suites.append(dict(task=s, n=n, queries=queries, paths=f'paths_{s}.json'))
        a.output.parent.mkdir(parents=True, exist_ok=True); f = a.output.parent/f'paths_{s}.json'
        f.write_text(json.dumps(dict(Q=b64(np.concatenate(Q)), G=b64(np.concatenate(G)), D=b64(np.concatenate(Dd)), W=b64(np.concatenate(W))), separators=(',', ':')))
        sizes[f.name] = round(f.stat().st_size/1e6, 2)
    e = json.loads((ev/'evaluation.json').read_text())
    out = dict(run=a.run.name, scene=SCENE.name, cuboids=doc['cuboids'], compartments=doc['compartments'], focus=doc['focus'], sampling={k: doc['sampling'][k] for k in 'xyz'},
               box_names=clear.names, totals=e['totals'], suites=suites, keep=a.keep, check_step=a.check,
               training=status.get('timing'))
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, separators=(',', ':')))
    print(a.output, round(a.output.stat().st_size/1e6, 2), 'MB', sizes)


if __name__ == '__main__': main()
