"""Task 2: accuracy of speed networks against ground truth on the cabinet task (read-only; no training, no planning).

For every run given (results/registered_<id> with prepared/speed.pt: the release two-output SpeedField of a
cabinet run, or a per-piece network of a face-speed run) the predicted body speed is compared with

  release truth   the release clearance speed, min(clip(d_obstacle), clip(0.5 d_self)) from the spheres, the
                  quantity the scorer's safety rule is built on, and
  cover truth     the network's own label: the box-face speed for cover=faces, the release speed otherwise,

on five sample sets: uniform joint draws, uniform draws the release gate accepts, the held-out original and
augmentation endpoints of the shared FK pool (pair index modulo 100 == 0, never fitted), and the densified
witness paths of the frozen cabinet suites split into points inside and outside the cupboard. Per-piece
networks also report the error of each output against its piece and how often the limiting piece is right.
Metrics as experiments/speed_test: MAE, RMSE, p95, max, bias, Pearson, log-RMSE and MAE by true-speed bin.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean')
from common import sha, write
from experiments.face_speed.clearance import PieceClearance
from experiments.face_speed.speed_net import build, components

SAMPLES, SEED = 20000, 20261018
BINS = [(0., .2), (.2, .5), (.5, .9), (.9, 1.)]


def metrics(true, pred):
    if not len(true): return dict(n=0)
    e = pred-true; a = np.abs(e)
    row = dict(n=int(len(a)), mae=float(a.mean()), rmse=float(np.sqrt((e**2).mean())), bias=float(e.mean()), p50=float(np.median(a)), p95=float(np.quantile(a, .95)),
               maximum=float(a.max()), log_rmse=float(np.sqrt(((np.log(pred)-np.log(np.maximum(true, .1)))**2).mean())),
               pearson=float(np.corrcoef(true, pred)[0, 1]) if true.std() > 0 and pred.std() > 0 else None, by_true_speed=[])
    for lo, hi in BINS:
        m = (true >= lo) & ((true < hi) if hi < 1 else (true <= hi+1e-5))
        row['by_true_speed'].append(dict(low=lo, high=hi, n=int(m.sum()), mae=float(a[m].mean()) if m.any() else None, bias=float(e[m].mean()) if m.any() else None))
    floor, free = true <= .1+1e-4, true >= .9
    row['floor_detection'] = dict(true_at_floor=int(floor.sum()), predicted_below_half=float((pred[floor] < .5).mean()) if floor.any() else None,
                                  true_free=int(free.sum()), predicted_at_least_half=float((pred[free] >= .5).mean()) if free.any() else None)
    return row


def densify(q, step=.02):
    out = [q[0]]
    for i in range(1, len(q)):
        n = max(1, int(np.ceil(np.abs(q[i]-q[i-1]).max()/step)))
        out.extend(q[i-1]+(q[i]-q[i-1])*(k/n) for k in range(1, n+1))
    return np.asarray(out, np.float32)


@torch.no_grad()
def predict(net, q, chunk=4096):
    body, parts = [], []
    for s in range(0, len(q), chunk):
        x = torch.as_tensor(q[s:s+chunk], device='cuda:0'); body.append(net(x).cpu().numpy()); parts.append(components(net, x).cpu().numpy())
    return np.concatenate(body), np.concatenate(parts)


@torch.no_grad()
def truth(clearances, q, chunk=4096):
    out = {name: [] for name in clearances}; valid = []; pos = []
    for s in range(0, len(q), chunk):
        x = torch.as_tensor(q[s:s+chunk], device='cuda:0')
        for name, c in clearances.items(): out[name].append(c.speeds(x).cpu().numpy())
        o, sd = clearances['release'].release_components(x); valid.append(((o >= .005) & (sd >= .002) & (x.abs() <= .5).all(1)).cpu().numpy())
        pos.append(clearances['release'].ee_position(x).cpu().numpy())
    if not len(q): return {k: np.zeros((0, c.n_pieces), np.float32) for k, c in clearances.items()}, np.zeros(0, bool), np.zeros((0, 3), np.float32)
    return {k: np.concatenate(v) for k, v in out.items()}, np.concatenate(valid), np.concatenate(pos)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--scene', type=Path, required=True); p.add_argument('--tests', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--runs', type=Path, nargs='+', required=True); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False); started = time.monotonic(); rng = np.random.default_rng(SEED); count = 512 if a.smoke else SAMPLES
    nets = {}
    for run in a.runs:
        payload = torch.load(run/'prepared/speed.pt', map_location='cpu', weights_only=False); contract = dict(payload['training_contract'])
        contract.setdefault('cover', 'release'); contract.setdefault('n_pieces', 2)   # the release SpeedField of a cabinet run predates the contract fields
        net = build(contract).to('cuda:0'); net.load_state_dict(payload['model_state_dict'], strict=True); net.eval().requires_grad_(False)
        nets[run.name] = dict(net=net, contract=contract, sha256=sha(run/'prepared/speed.pt'),
                              sampled=Path(contract.get('sampled', '')) if contract.get('sampled') else None)
    covers = {'release': PieceClearance(a.scene, 'spheres', 'cuda:0')}
    for info in nets.values():
        c = info['contract']['cover']
        if c not in covers and c != 'release': covers[c] = PieceClearance(a.scene, c, 'cuda:0')
    # Sample sets.
    sets = {'uniform': rng.uniform(-.5, .5, (count, 6)).astype(np.float32)}
    sampled = next((i['sampled'] for i in nets.values() if i['sampled'] and (i['sampled']/'inputs').exists()), None)
    if sampled is None:
        cabinet = [r for r in a.runs if (r/'sampled/inputs').exists()]; sampled = cabinet[0]/'sampled' if cabinet else None
    if sampled is not None:
        for kind in ['original', 'augmentation']:
            points = np.load(sampled/'inputs'/kind/'sampled_points.npy', mmap_mode='r'); ids = np.arange(len(points)); ids = ids[ids % 100 == 0]
            sets['heldout_'+kind] = np.asarray(points[ids[:256] if a.smoke else ids], dtype=np.float32).reshape(-1, 6)
    manifest = json.loads((a.tests/'suite_manifest.json').read_text()); robot = covers['release'].robot
    center, scale = robot.q_center.cpu().numpy(), robot.q_range.cpu().numpy(); witness = []
    for item in manifest['suites']:
        z = np.load(a.tests/'suites'/item['file'], allow_pickle=True)
        for w in z['witness_trajectories'][:2 if a.smoke else None]: witness.append((densify(np.asarray(w, np.float32))-center)/scale)
    witness = np.concatenate(witness).astype(np.float32)
    truths, valid, pos = truth(covers, witness); import yaml
    doc = yaml.safe_load(a.scene.read_text()); panels = [c for c in doc['cuboids'] if c['name'].startswith('shelf_')]
    lo = np.min([np.subtract(c['pose'][:3], np.divide(c['dims'], 2)) for c in panels], 0); hi = np.max([np.add(c['pose'][:3], np.divide(c['dims'], 2)) for c in panels], 0)
    inside = ((pos >= lo) & (pos <= hi)).all(1)
    sets['witness_inside_cupboard'] = witness[inside]; sets['witness_outside_cupboard'] = witness[~inside]
    # Uniform draws the release gate accepts.
    kept = []; total = 0
    while total < count:
        q = rng.uniform(-.5, .5, (65536, 6)).astype(np.float32); _, ok, _ = truth({'release': covers['release']}, q); kept.append(q[ok]); total += int(ok.sum())
    sets['uniform_valid'] = np.concatenate(kept)[:count]
    sets = {k: v for k, v in sets.items() if len(v)}
    report = dict(scene=str(a.scene), scene_sha256=sha(a.scene), tests_manifest_sha256=sha(a.tests/'suite_manifest.json'), seed=SEED, smoke=a.smoke, sets={}, networks={})
    truth_cache = {name: truth(covers, q) for name, q in sets.items()}
    for name, q in sets.items():
        t, ok, _ = truth_cache[name]; rel = t['release'].min(1)
        report['sets'][name] = dict(n=int(len(q)), share_release_valid=float(ok.mean()), share_release_at_floor=float((rel <= .1+1e-4).mean()), mean_release_speed=float(rel.mean()),
                                    **{f'mean_{c}_speed': float(t[c].min(1).mean()) for c in covers if c != 'release'})
    for run, info in nets.items():
        net, contract = info['net'], info['contract']; cover = contract['cover']; rows = {}
        for name, q in sets.items():
            t, ok, _ = truth_cache[name]; body, parts = predict(net, q); rel = t['release'].min(1)
            row = dict(vs_release=metrics(rel, body))
            if cover != 'release':
                own = t[cover]; row['vs_cover'] = metrics(own.min(1), body)
                row['pieces'] = dict(mae=float(np.abs(parts-own).mean()), p95=float(np.quantile(np.abs(parts-own), .95)),
                                     limiting_piece_accuracy=float((parts.argmin(1) == own.argmin(1)).mean()),
                                     limiting_piece_tie_aware=float((np.take_along_axis(own, parts.argmin(1)[:, None], 1)[:, 0] <= own.min(1)+.02).mean()))
            rows[name] = row
        report['networks'][run] = dict(cover=cover, pieces=contract.get('n_pieces'), speed_sha256=info['sha256'], sets=rows)
        print('ACCURACY', run, cover, {k: round(v['vs_release']['mae'], 4) for k, v in rows.items()}, flush=True)
    report['elapsed_seconds'] = time.monotonic()-started
    write(a.output/'accuracy.json', report); (a.output/'REPORT.md').write_text(report_md(report))
    np.savez_compressed(a.output/'samples.npz', **{k: v for k, v in sets.items()})
    print('ACCURACY_COMPLETE', flush=True)


def report_md(report):
    f = lambda v, d=4: '–' if v is None else f'{v:.{d}f}'
    lines = ['# Speed network accuracy on the cabinet task', '', ('SMOKE ONLY. ' if report['smoke'] else '')+'Ground truths: the release clearance speed (spheres; the scorer\'s rule) and each network\'s own cover label. Speeds lie in [0.1, 1].', '',
             '|Set|Points|Release-valid|Release speed at floor|Mean release speed|', '|---|---:|---:|---:|---:|']
    for name, s in report['sets'].items(): lines.append(f"|{name}|{s['n']}|{100*s['share_release_valid']:.1f}%|{100*s['share_release_at_floor']:.1f}%|{f(s['mean_release_speed'],3)}|")
    for run, net in report['networks'].items():
        lines += ['', f"## {run}: cover `{net['cover']}`, {net['pieces']} outputs", '', '|Set|MAE vs release|RMSE|p95|log-RMSE|Pearson|MAE vs own cover|Piece MAE|Limiting piece right|',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name, row in net['sets'].items():
            r = row['vs_release']; c = row.get('vs_cover', {}); pz = row.get('pieces', {})
            lines.append(f"|{name}|{f(r['mae'])}|{f(r['rmse'])}|{f(r['p95'])}|{f(r['log_rmse'])}|{f(r['pearson'],3)}|{f(c.get('mae'))}|{f(pz.get('mae'))}|{'–' if not pz else f'{100*pz['limiting_piece_tie_aware']:.1f}%'}|")
        lines += ['', 'MAE vs release by true release speed:', '', '|Set|'+'|'.join(f'{max(lo,.1)}–{hi}' for lo, hi in BINS)+'|', '|---|'+'---:|'*len(BINS)]
        for name, row in net['sets'].items(): lines.append(f'|{name}|'+'|'.join(f"{f(b['mae'])} (n={b['n']})" for b in row['vs_release']['by_true_speed'])+'|')
    return '\n'.join(lines)+'\n'


if __name__ == '__main__': main()
