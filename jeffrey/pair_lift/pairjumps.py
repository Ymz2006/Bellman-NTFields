"""The step in travel time a field holds across a wall (jeffrey/lift_scale/jumps.py's measurement, for pair-lift and lift-scale fields).

For band configurations a, the mirror configuration b on the other side of the nearest wall (same heading, 0.2 - 1.6 cells away, the
segment a-b blocked) and goals g of the prepare run: the truth's step D* = |T*(a, g) - T*(b, g)| against the field's D = |T(a, g) - T(b, g)|,
by D*. Also the field's own estimate of the crossing T(a, b) against the smaller of the two true detours it must be at least as long as
(|D*| is a lower bound of T*(a, b)).

    CUDA_VISIBLE_DEVICES=1 python jeffrey/pair_lift/pairjumps.py --out jeffrey/pair_lift/report/jumps.json <run dir>[@epoch] ...
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pairnet as PN  # noqa: E402
import scenes as SN  # noqa: E402

_spec = importlib.util.spec_from_file_location('lift_scale_jumps', PN.ROOT / 'jeffrey/lift_scale/jumps.py')
JP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(JP)
_scenes = {}


def analyse(run, epoch=None):
    res = json.load(open(run / 'results.json'))
    prepared = Path(res['prepared'])
    if str(prepared) not in _scenes:
        _scenes[str(prepared)] = SN.load_scene(prepared)
    scene, info = _scenes[str(prepared)]
    W = JP.wall_pairs(scene, prepared)
    ck = run / 'training/checkpoint.pt' if epoch is None else run / f'training/checkpoints/epoch_{epoch:05d}.pt'
    net = PN.load_field(ck, scene)
    with torch.no_grad():
        ta = net.out(torch.cat([W['A'], W['goal']], 1))[0][:, 0]
        tb = net.out(torch.cat([W['B'], W['goal']], 1))[0][:, 0]
        tab = net.out(torch.cat([W['A'], W['B']], 1))[0][:, 0] / scene.pitch
    step = (ta - tb).abs() / scene.pitch
    signed_ok = torch.sign(ta - tb)
    st = W['step_true']
    a = res['args']
    out = dict(run=run.name.replace('registered_', ''), epoch=epoch, scene=res['scene_name'], lift=a['lift'], pair=a.get('pair', 'none'), push=a['push'],
               pairs=int(len(st)), planar_gap_median=float(W['planar_gap'].median()), cross_median=float(tab.median()), cross_q10=float(tab.quantile(0.1)), by_true={})
    for i, lab in enumerate(JP.labels()):
        m = (st >= JP.BINS[i]) & (st < JP.BINS[i + 1])
        if int(m.sum()) > 15:
            out['by_true'][lab] = dict(n=int(m.sum()), true=float(st[m].median()), learned=float(step[m].median()), learned_q10=float(step[m].quantile(0.1)),
                                       learned_q90=float(step[m].quantile(0.9)), cross=float(tab[m].median()),
                                       ratio=float((step[m] / st[m].clamp(min=1e-6)).median()) if JP.BINS[i] > 0 else None)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--out', type=Path, default=None)
    a = p.parse_args()
    rows = []
    for spec in a.runs:
        path, _, ep = spec.partition('@')
        r = analyse(Path(path), int(ep) if ep else None)
        rows.append(r)
        print(f"{r['run']} | lift {r['lift']} pair {r['pair']} push {r['push']} | epoch {r['epoch']} | {r['pairs']} pairs {r['planar_gap_median']:.2f} cells apart | T(a, b) median {r['cross_median']:.1f} cells")
        print('   true step -> learned step (T(a, b)): ' + '; '.join(f"{k}: {v['true']:.1f} -> {v['learned']:.1f} ({v['cross']:.1f})" for k, v in r['by_true'].items()))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        prev = json.load(open(a.out)) if a.out.exists() else []
        keys = {(r['run'], r['epoch']) for r in rows}
        json.dump([r for r in prev if (r['run'], r['epoch']) not in keys] + rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
