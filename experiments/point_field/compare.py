"""Hierarchical point-field speed against the trained speed networks: (1) distance, (2) speed.

Models
  hierarchical:<name>  a PointField checkpoint (train.py) through Hierarchical: FK sphere centers, learned D(c_i) - r_i,
                       min over the spheres = environment clearance; exact 0.5 d_self; body = min; speed = release formula.
  hierarchical:exact   the same with the exact scene distance (sanity check: equals the release label).
  hierarchical:<name>+learned-self  (--self-net <run>) the learned environment clearance with the self piece of a split
                       run's network instead of the exact self term: every part learned, as the two-model split.
  <run>                prepared/speed.pt of a cabinet or face-speed run (release two-output SpeedField, per-sphere,
                       per-link, per-face, split shared/separate). Networks predict speed only; their distance is
                       0.05 s, defined inside the label's band [5, 50] mm. Networks with pieces also give an
                       environment speed (min over their obstacle pieces) and a self speed (the 'self' piece).

Ground truth: the release label from the spheres (Hierarchical with the exact distance, = PieceClearance('spheres')).
Sample sets: those of experiments/face_speed/speed_accuracy.py, same seed and order, so its numbers line up.

Distance metrics, mm: band MAE |clip(d_hat) - clip(d)| with clip to [5, 50] mm (the only range a speed network
encodes), for the environment clearance and for the body clearance; for the hierarchical models also the
unclipped MAE and the MAE where the true clearance is within 0-150 mm. Speed metrics: speed_accuracy.metrics.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT/'method'):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
from common import sha, write
from experiments.face_speed.speed_accuracy import metrics, densify, SEED, SAMPLES
from experiments.face_speed.speed_net import build, components
from experiments.point_field.point_field import Hierarchical, load_field, speed_of

MM = 1000.


def sample_sets(scene, tests, sampled, hier, count, smoke):
    """The sets of speed_accuracy.py, in its order and with its generator, so the draws are identical."""
    rng = np.random.default_rng(SEED); sets = {'uniform': rng.uniform(-.5, .5, (count, 6)).astype(np.float32)}
    for kind in ['original', 'augmentation']:
        points = np.load(sampled/'inputs'/kind/'sampled_points.npy', mmap_mode='r'); ids = np.arange(len(points)); ids = ids[ids % 100 == 0]
        sets['heldout_'+kind] = np.asarray(points[ids[:256] if smoke else ids], dtype=np.float32).reshape(-1, 6)
    manifest = json.loads((tests/'suite_manifest.json').read_text()); robot = hier.robot
    center, scale = robot.q_center.cpu().numpy(), robot.q_range.cpu().numpy(); witness = []
    for item in manifest['suites']:
        z = np.load(tests/'suites'/item['file'], allow_pickle=True)
        for w in z['witness_trajectories'][:2 if smoke else None]: witness.append((densify(np.asarray(w, np.float32))-center)/scale)
    witness = np.concatenate(witness).astype(np.float32)
    import yaml
    doc = yaml.safe_load(scene.read_text()); panels = [c for c in doc['cuboids'] if c['name'].startswith('shelf_')]
    lo = np.min([np.subtract(c['pose'][:3], np.divide(c['dims'], 2)) for c in panels], 0); hi = np.max([np.add(c['pose'][:3], np.divide(c['dims'], 2)) for c in panels], 0)
    with torch.no_grad(): pos = np.concatenate([robot.ee_position(robot.radians(torch.as_tensor(witness[s:s+8192], device='cuda:0'))).cpu().numpy() for s in range(0, len(witness), 8192)])
    inside = ((pos >= lo) & (pos <= hi)).all(1)
    sets['witness_inside_cupboard'] = witness[inside]; sets['witness_outside_cupboard'] = witness[~inside]
    kept = []; total = 0
    while total < count:
        q = rng.uniform(-.5, .5, (65536, 6)).astype(np.float32); ok = valid(hier, q); kept.append(q[ok]); total += int(ok.sum())
    sets['uniform_valid'] = np.concatenate(kept)[:count]
    return {k: v for k, v in sets.items() if len(v)}


@torch.no_grad()
def valid(hier, q, chunk=8192):
    """The release gate of speed_accuracy.truth: obstacle clearance >= 5 mm, self distance >= 2 mm, inside the box."""
    out = []
    for s in range(0, len(q), chunk):
        x = torch.as_tensor(q[s:s+chunk], device='cuda:0'); env, d_self, _ = hier.clearances(x)
        out.append(((env >= .005) & (d_self/.5 >= .002) & (x.abs() <= .5).all(1)).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def hierarchical(model, q, chunk=4096):
    env, sf, body = [], [], []
    for s in range(0, len(q), chunk):
        e, d, b = model.clearances(torch.as_tensor(q[s:s+chunk], device='cuda:0')); env.append(e.cpu().numpy()); sf.append(d.cpu().numpy()); body.append(b.cpu().numpy())
    return dict(env=np.concatenate(env), self=np.concatenate(sf), body=np.concatenate(body))


@torch.no_grad()
def network(net, names, q, chunk=4096):
    body, parts = [], []
    for s in range(0, len(q), chunk):
        x = torch.as_tensor(q[s:s+chunk], device='cuda:0'); body.append(net(x).cpu().numpy()); parts.append(components(net, x).cpu().numpy())
    body, parts = np.concatenate(body), np.concatenate(parts); out = dict(body_speed=body)
    if names and names[-1] == 'self' and len(names) > 1: out.update(env_speed=parts[:, :-1].min(1), self_speed=parts[:, -1])
    return out


def band(d): return np.clip(d, .005, .05)


def distance_row(true, pred, unclipped=False):
    e = np.abs(band(pred)-band(true))*MM; row = dict(band_mae_mm=float(e.mean()), band_p95_mm=float(np.quantile(e, .95)), band_bias_mm=float(((band(pred)-band(true))*MM).mean()))
    near = (true > .005) & (true < .05)
    row['inside_band_mae_mm'] = float(e[near].mean()) if near.any() else None; row['inside_band_n'] = int(near.sum())
    if unclipped:
        u = np.abs(pred-true)*MM; m = (true >= 0) & (true <= .15)
        row.update(mae_mm=float(u.mean()), p95_mm=float(np.quantile(u, .95)), bias_mm=float(((pred-true)*MM).mean()), mae_0_150mm=float(u[m].mean()) if m.any() else None, n_0_150mm=int(m.sum()))
    return row


@torch.no_grad()
def throughput(fn, dev='cuda:0', batch=4096, reps=50):
    q = torch.rand(batch, 6, device=dev)-.5
    for _ in range(5): fn(q)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps): fn(q)
    torch.cuda.synchronize(); return batch*reps/(time.perf_counter()-t)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--scene', type=Path, required=True); p.add_argument('--tests', type=Path, required=True)
    p.add_argument('--sampled', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fields', nargs='+', default=[], help='name=path/to/field.pt'); p.add_argument('--runs', type=Path, nargs='+', default=[])
    p.add_argument('--self-net', type=Path, help='a split run whose self piece replaces the exact self term (extra hierarchical models)')
    p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False); started = time.monotonic(); dev = 'cuda:0'
    exact = Hierarchical(a.scene, None, dev)
    models = {'hierarchical:exact': dict(kind='hierarchical', model=exact, parameters=0)}
    for spec in a.fields:
        name, path = spec.split('=', 1); field = load_field(path, dev); payload = torch.load(path, map_location='cpu', weights_only=False)
        models['hierarchical:'+name] = dict(kind='hierarchical', model=Hierarchical(a.scene, field, dev), parameters=sum(v.numel() for v in field.parameters()),
                                            sha256=sha(path), sources=payload.get('sources'), steps=payload.get('steps'))
    if a.self_net:
        payload = torch.load(a.self_net/'prepared/speed.pt', map_location='cpu', weights_only=False); contract = dict(payload['training_contract'])
        if contract.get('cover') != 'split': raise SystemExit(f'{a.self_net.name} is not a split run')
        self_net = build(contract).to(dev); self_net.load_state_dict(payload['model_state_dict'], strict=True); self_net.eval().requires_grad_(False)
        for name in [m for m in models if m != 'hierarchical:exact']:
            models[name+'+learned-self'] = dict(models[name], self_net=self_net, self_run=a.self_net.name)
    for run in a.runs:
        payload = torch.load(run/'prepared/speed.pt', map_location='cpu', weights_only=False); contract = dict(payload['training_contract'])
        contract.setdefault('cover', 'release'); contract.setdefault('n_pieces', 2)
        net = build(contract).to(dev); net.load_state_dict(payload['model_state_dict'], strict=True); net.eval().requires_grad_(False)
        label = f"{contract['cover']}" + (f"/{contract.get('speed_model')}" if contract['cover'] == 'split' else '')
        models[run.name] = dict(kind='network', model=net, names=contract.get('pieces') if contract['cover'] != 'release' else None, cover=label,
                                parameters=sum(v.numel() for v in net.parameters()), sha256=sha(run/'prepared/speed.pt'))
    sets = sample_sets(a.scene, a.tests, a.sampled, exact, 512 if a.smoke else SAMPLES, a.smoke)
    report = dict(scene=str(a.scene), scene_sha256=sha(a.scene), seed=SEED, smoke=a.smoke, sets={}, models={})
    truths = {name: hierarchical(exact, q) for name, q in sets.items()}
    for name, q in sets.items():
        t = truths[name]; report['sets'][name] = dict(n=int(len(q)), share_env_in_band=float(((t['env'] > .005) & (t['env'] < .05)).mean()),
                                                      share_body_at_floor=float((t['body'] <= .005).mean()), mean_release_speed=float(speed_of(torch.as_tensor(t['body'])).mean()))
    for mname, info in models.items():
        rows = {}
        for name, q in sets.items():
            t = truths[name]; true_speed = speed_of(torch.as_tensor(t['body'])).numpy()
            if info['kind'] == 'hierarchical':
                pr = hierarchical(info['model'], q)
                if 'self_net' in info:
                    s_self = network(info['self_net'], ['environment', 'self'], q)['self_speed']; pr['body'] = np.minimum(pr['env'], .05*s_self)
                    pred_speed = np.minimum(speed_of(torch.as_tensor(pr['env'])).numpy(), s_self)
                else: pred_speed = speed_of(torch.as_tensor(pr['body'])).numpy()
                row = dict(env=distance_row(t['env'], pr['env'], True), body=distance_row(t['body'], pr['body'], 'self_net' not in info))
            else:
                pr = network(info['model'], info['names'], q); pred_speed = pr['body_speed']
                row = dict(body=distance_row(t['body'], .05*pred_speed))
                if 'env_speed' in pr: row['env'] = distance_row(t['env'], .05*pr['env_speed'])
            row['speed'] = metrics(true_speed, pred_speed); rows[name] = row
        info['throughput_configs_per_s'] = throughput(info['model']) if 'self_net' not in info else throughput(lambda q: torch.minimum(speed_of(info['model'].components(q)[0].amin(1)), info['self_net'].pieces(q)[:, 1]))
        report['models'][mname] = dict(kind=info['kind'], cover=info.get('cover', 'point field'), parameters=info['parameters'], sha256=info.get('sha256'),
                                       sources=info.get('sources'), self_run=info.get('self_run'), throughput_configs_per_s=info['throughput_configs_per_s'], sets=rows)
        print('COMPARE', mname, {k: (round(v['speed']['mae'], 4), round(v['env']['band_mae_mm'], 2) if 'env' in v else None) for k, v in rows.items()}, flush=True)
    report['elapsed_seconds'] = time.monotonic()-started
    write(a.output/'comparison.json', report); (a.output/'REPORT.md').write_text(report_md(report)); print('COMPARE_COMPLETE', flush=True)


def report_md(r):
    f = lambda v, d=3: '–' if v is None else f'{v:.{d}f}'
    sets = list(r['sets'])
    L = ['# Hierarchical point-field speed vs speed networks (cabinet)', '', ('SMOKE ONLY. ' if r['smoke'] else '')+
         'Truth: the release label from the spheres. Distances in mm; band = clipped to [5, 50] mm, the range a speed label encodes.', '',
         '|Set|Points|Env clearance in band|Body at floor|Mean release speed|', '|---|---:|---:|---:|---:|']
    for n, s in r['sets'].items(): L.append(f"|{n}|{s['n']}|{100*s['share_env_in_band']:.1f}%|{100*s['share_body_at_floor']:.1f}%|{s['mean_release_speed']:.3f}|")
    for title, key, sub in [('Environment clearance, band MAE (mm)', 'env', 'band_mae_mm'), ('Environment clearance, MAE where 5 < d < 50 mm (mm)', 'env', 'inside_band_mae_mm'),
                            ('Body clearance, band MAE (mm)', 'body', 'band_mae_mm'), ('Body speed MAE vs release', 'speed', 'mae'), ('Body speed p95 abs error', 'speed', 'p95'),
                            ('Body speed bias', 'speed', 'bias')]:
        L += ['', f'## {title}', '', '|Model|Cover|Params|'+'|'.join(sets)+'|', '|---|---|---:|'+'---:|'*len(sets)]
        for m, info in r['models'].items():
            L.append(f"|{m}|{info['cover']}|{info['parameters']:,}|"+'|'.join(f(info['sets'][s].get(key, {}).get(sub)) if key in info['sets'][s] else '–' for s in sets)+'|')
    L += ['', '## Hierarchical models: unclipped environment clearance (mm)', '', '|Model|Set|MAE|p95|bias|MAE where 0-150 mm|', '|---|---|---:|---:|---:|---:|']
    for m, info in r['models'].items():
        if info['kind'] != 'hierarchical': continue
        for s in sets: e = info['sets'][s]['env']; L.append(f"|{m}|{s}|{f(e['mae_mm'])}|{f(e['p95_mm'])}|{f(e['bias_mm'])}|{f(e['mae_0_150mm'])}|")
    L += ['', '## Throughput (configurations per second, batch 4096)', '', '|Model|Configs/s|', '|---|---:|']
    for m, info in r['models'].items(): L.append(f"|{m}|{info['throughput_configs_per_s']:,.0f}|")
    return '\n'.join(L)+'\n'


if __name__ == '__main__': main()
