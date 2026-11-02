"""Read-only speed-fit check of Test 4 (jeffrey/joint_range/README.md): do collision pairs and thicker panels make the
ntrl-demo field slow where the arm collides?

The field's speed at a pair end is 1 / |d tau / d x| (the trainer fits it to s' = alpha s^2 (2 - s)^2 + 1 - alpha, alpha
1.025; fitcheck.py). Every run is checked on the held-out pairs of its own scene's data run (heldout/: mixed and
collision pairs that neither training set contains), both ends of every pair, split by stream (mixed / collision) and by
whether the grasp point is in the cupboard's outer box. Reported per group: rank correlation of field speed with s', the
median field speed at contact (label at the floor, clearance <= 5 mm), clear (clearance >= 2.5 cm) and free (>= 5 cm)
ends, the wall contrast median(clear) / median(contact) (the labels ask for s'(0.5) / s'(0.1) = 63x), and the share of
contact ends the field reads faster than 0.5.

Test 3 mixture runs (mode mix) may be added for reference: they are checked on the held-out pairs of the first Test 4 run
of their scene.

Host only (no cuRobo): ~/.venvs/bellman-ntfield/bin/python jeffrey/joint_range/fitcheck4.py --runs <train run dirs> --output <dir>
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import fitcheck as F
from fitcheck import ROOT, R, scene_boxes, field_speeds, summary


def main():
    p = argparse.ArgumentParser(); p.add_argument('--runs', type=Path, nargs='+', required=True); p.add_argument('--output', type=Path, required=True); p.add_argument('--device', default='cuda:0'); a = p.parse_args()
    dev = a.device; FIELD = F.load_field_module(); kin = R.Kinematics().to(dev); held = {}; rows = []; statuses = {r: json.loads((r/'status.json').read_text()) for r in a.runs}
    data_of = {}
    for r, st in statuses.items():
        if st['mode'] == 'train': data_of.setdefault(F.host(st['settings']['scene']), F.host(st['settings']['data']))
    for run in a.runs:
        st = statuses[run]; assert st['complete'] and st['mode'] in ('train', 'mix') and not st['settings']['smoke'], run
        scene = F.host(st['settings']['scene']); data = F.host(st['settings']['data']) if st['mode'] == 'train' else data_of[scene]
        if data not in held:
            h = data/'heldout'; meta = json.loads((h/'complete.json').read_text()); unit = torch.tensor(meta['radians_per_unit'], device=dev)
            x = torch.tensor(np.load(h/'sampled_points.npy'), device=dev); s = np.load(h/'speed.npy').reshape(-1); src = np.repeat(np.load(h/'source.npy'), 2)
            with torch.no_grad(): pos = torch.stack([kin.transforms(x[:, c:c+6]*unit, [R.EE_LINK])[R.EE_LINK][:, :3, 3] for c in (0, 6)], 1).cpu().numpy().reshape(-1, 3)
            _, cupboard, _ = scene_boxes(scene); cup = ((pos >= cupboard[:, 0]) & (pos <= cupboard[:, 1])).all(1)
            held[data] = (x, s, src, cup, meta)
        x, s, src, cup, meta = held[data]
        net = FIELD.load_metric(run/'training/checkpoint.pt', dev, 'metric', 'release').net; pred = field_speeds(net, x)
        row = dict(run=run.name, scene=scene.stem, set=st['settings'].get('set') or f"test3 f{st['settings']['fraction']:g}", data=str(data.relative_to(ROOT)), groups={})
        for gname, m in dict(all=np.ones(len(s), bool), mixed=src == 0, collision=src == 1, cupboard=cup, outside=~cup, mixed_cupboard=(src == 0) & cup, collision_cupboard=(src == 1) & cup).items():
            g = summary(s[m], pred[m]); g['contrast'] = g['field_median_clear']/g['field_median_contact'] if g['field_median_clear'] and g['field_median_contact'] else None; row['groups'][gname] = g
        rows.append(row); del net; torch.cuda.empty_cache()
    a.output.mkdir(parents=True, exist_ok=True); (a.output/'fitcheck4.json').write_text(json.dumps(dict(rows=rows), indent=1))
    print(f"{'scene':15s} {'set':8s} | group: rank, field median contact / clear, contrast, contact read > 0.5")
    for r in rows:
        for gname in ('all', 'cupboard', 'outside', 'collision'):
            g = r['groups'][gname]
            print(f"{r['scene']:15s} {r['set']:8s} | {gname:10s} n {g['n']:6d} contact {g['contact_share']:.2f} | rank {g['rank']:.3f}  {g['field_median_contact']:.3f} / {g['field_median_clear'] or float('nan'):.3f}  {g['contrast'] or float('nan'):5.2f}x  {g['contact_read_fast']:.3f}")


if __name__ == '__main__': main()
