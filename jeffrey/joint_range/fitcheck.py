"""Read-only speed-fit check of the in-cabinet mixture sweep (joint-range test 3): does more in-cabinet training data
make the ntrl-demo field see the cupboard?

The trainer fits |d tau / d x| at both pair endpoints to 1 / s' with s' = alpha s^2 (2 - s)^2 + 1 - alpha, alpha 1.025
(baselines/ntrl-demo/models/metric/model_train_metric.py and model_function_metric.py), so the field's speed at an
endpoint is 1 / |d tau / d x|. Three fixed point sets, the same for every checkpoint: the first 20,000 in-cabinet pairs
(in the training data of every fraction >= 0.1), the last 20,000 in-cabinet pairs (held out for every fraction <= 0.9)
and the last 20,000 in-area pairs (held out for every mixture). Both endpoints of each pair are used, split by whether
the grasp point is in the cupboard's outer box (URDF forward kinematics, as mix.py) and by label (contact = the speed
floor 0.1, clear = label >= 0.5 i.e. clearance >= 2.5 cm, free = 1). Reports the correlation of field speed with s'
(Pearson as diagnose.py, and rank: Pearson is dominated by a few near-zero gradients), the median field speed at contact,
clear and free labels, and the share of contact labels the field reads faster than 0.5.

Host only (no cuRobo): ~/.venvs/bellman-ntfield/bin/python jeffrey/joint_range/fitcheck.py --runs <mix run dirs> --output <dir>
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT/'method', ROOT/'experiments/cabinet'): sys.path.insert(0, str(p))
import common
common.RELEASE = ROOT/'runtime'
from geometry import scene_boxes
from experiments.face_speed import robot as R
ALPHA, FLOOR, SETS = 1.025, 0.1001, dict(cab_seen=('cabinet', 0, 20000), cab_held=('cabinet', 180000, 200000), area_held=('data', 380000, 400000))


def host(path): return ROOT/Path(path).relative_to('/clean')


def load_field_module():
    spec = importlib.util.spec_from_file_location('ntrl_cabinet_field', ROOT/'experiments/ntrl_cabinet/field.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.NTRL, m.NORM = ROOT/'baselines/ntrl-demo', ROOT/'runtime/configs/joint_normalization.npz'; return m


def field_speeds(net, x):
    out = []
    for i in range(0, len(x), 5000):
        with torch.enable_grad():
            tau, _, xp = net.out(x[i:i+5000]); g = torch.autograd.grad(tau.sum(), xp)[0]
        out.append(torch.stack([1/g[:, :6].norm(dim=1).clamp_min(1e-9), 1/g[:, 6:].norm(dim=1).clamp_min(1e-9)], 1))
    return torch.cat(out).cpu().numpy().reshape(-1)


def rank(v): r = np.empty(len(v)); r[np.argsort(v, kind='stable')] = np.arange(len(v)); return r


def summary(label, pred):
    target = ALPHA*label**2*(2-label)**2+1-ALPHA; contact, free, clear = label <= FLOOR, label >= .9999, label >= .5
    return dict(n=int(len(label)), contact_share=float(contact.mean()), free_share=float(free.mean()), clear_share=float(clear.mean()), pearson=float(np.corrcoef(target, pred)[0, 1]),
                field_median_clear=float(np.median(pred[clear])) if clear.any() else None, target_median_clear=float(np.median(target[clear])) if clear.any() else None, field_quantiles=[float(v) for v in np.quantile(pred, [.5, .99, 1])],
                rank=float(np.corrcoef(rank(target), rank(pred))[0, 1]), target_median_contact=float(np.median(target[contact])) if contact.any() else None,
                field_median_contact=float(np.median(pred[contact])) if contact.any() else None, field_median_free=float(np.median(pred[free])) if free.any() else None,
                contact_read_fast=float((pred[contact] > .5).mean()) if contact.any() else None)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--runs', type=Path, nargs='+', required=True); p.add_argument('--output', type=Path, required=True); p.add_argument('--device', default='cuda:0'); a = p.parse_args()
    dev = a.device; FIELD = load_field_module(); status = [json.loads((r/'status.json').read_text()) for r in a.runs]
    for st in status: assert st['complete'] and st['mode'] == 'mix' and not st['settings']['smoke'] and st['settings']['pairs'] == 200000, st['settings']
    src = {k: {st['settings'][k] for st in status} for k in ('cabinet', 'data', 'scene')}; assert all(len(v) == 1 for v in src.values()), src
    src = {k: host(v.pop()) for k, v in src.items()}; meta = json.loads((src['cabinet']/'complete.json').read_text()); unit = torch.tensor(meta['radians_per_unit'], device=dev)
    assert meta['coords'] == 'full' and meta['pairs'] == 200000 and json.loads((src['data']/'complete.json').read_text())['pairs'] == 400000
    _, cupboard, _ = scene_boxes(src['scene']); kin = R.Kinematics().to(dev); points = {}
    for name, (which, lo, hi) in SETS.items():
        x = np.ascontiguousarray(np.load(src[which]/'sampled_points.npy', mmap_mode='r')[lo:hi]); s = np.ascontiguousarray(np.load(src[which]/'speed.npy', mmap_mode='r')[lo:hi])
        xt = torch.tensor(x, device=dev)
        with torch.no_grad():
            pos = torch.stack([kin.transforms(xt[:, c:c+6]*unit, [R.EE_LINK])[R.EE_LINK][:, :3, 3] for c in (0, 6)], 1).cpu().numpy().reshape(-1, 3)
        points[name] = (xt, s.reshape(-1), ((pos >= cupboard[:, 0]) & (pos <= cupboard[:, 1])).all(1))
    rows = []
    for run, st in zip(a.runs, status):
        f = st['settings']['fraction']; n_cab = int(round(f*200000)); net = FIELD.load_metric(run/'training/checkpoint.pt', dev, 'metric', 'release').net
        row = dict(run=run.name, fraction=f, cabinet_pairs_trained=n_cab, cab_held_trained=n_cab > 180000, cab_seen_trained=n_cab >= 20000)
        for name, (xt, label, cup) in points.items():
            pred = field_speeds(net, xt); row[name] = dict(cupboard=summary(label[cup], pred[cup]), outside=summary(label[~cup], pred[~cup]))
        rows.append(row); del net; torch.cuda.empty_cache()
    rows.sort(key=lambda r: r['fraction']); a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'fitcheck.json').write_text(json.dumps(dict(sources={k: str(v) for k, v in src.items()}, sets=SETS, rows=rows), indent=1))
    print(f"{'frac':>5} | cupboard points of held-out cabinet pairs: rank, field median at contact / clear, contact>0.5 | same, trained cabinet pairs | outside points of held-out area pairs: rank, contact / clear, contact>0.5")
    for r in rows:
        h, s_, o = r['cab_held']['cupboard'], r['cab_seen']['cupboard'], r['area_held']['outside']
        print(f"{r['fraction']:5.1f} | " + ' | '.join(f"{g['rank']:6.3f} {g['field_median_contact']:6.3f} / {g['field_median_clear']:6.3f} {g['contact_read_fast']:6.3f}" for g in (h, s_, o)))
    for name in SETS:
        for group in ('cupboard', 'outside'):
            g = rows[0][name][group]; print('labels', name, group, 'n', g['n'], 'contact share', round(g['contact_share'], 3), 'clear share', round(g['clear_share'], 3), 'free share', round(g['free_share'], 3), 'target median at contact', g['target_median_contact'])


if __name__ == '__main__': main()
