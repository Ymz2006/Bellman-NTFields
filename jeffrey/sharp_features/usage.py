"""Does the trained network use the sharp columns?  -> report/usage.json (inference only).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/sharp_features/usage.py [--plan] [--ids a,b]

Per run (this study's runs and the exact octave reference of the same maze and seed):
  weight   share of the squared weight norm of the three layers that read the input (pe_gate[0], pe_gate[1], encoder[1],
           after ntrl-demo's lip_norm) that sits on the sharp columns (replaced or appended); for the octave reference, on the
           same input columns as the spec it is compared with (the same rows), so a share equal to the reference's means the
           network weighs a sharp column like a plain one; for an added block, the share it would have if every column
           weighed the same is its width / total width
  pinned   with --plan: success (planning seed 0) when the sharp columns are held at their mean over the maze (the network
           sees a constant there); for the octave reference, the same columns pinned. Lost share = 1 - pinned / unpinned.
"""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
import features as FT  # noqa: E402
import sharpnet as SN  # noqa: E402
import maze as MZ  # noqa: E402
import plan as P  # noqa: E402
import evaluate as EV  # noqa: E402
from common import Logger  # noqa: E402
from sharp import sharp_columns  # noqa: E402

OUT = HERE / 'report' / 'usage.json'
PREP = {13: ROOT / 'results/registered_20261101T192504Z-13cf1d04', 22: ROOT / 'results/registered_20261101T193403Z-cea05796'}


def weight_share(net, cols):
    out = {}
    for name, lin in (('pe_gate0', net.pe_gate[0]), ('pe_gate1', net.pe_gate[1]), ('encoder1', net.encoder[1])):
        w = net.lip_norm(lin.weight.detach())
        tot = float((w ** 2).sum())
        out[name] = float((w[:, cols] ** 2).sum()) / tot
    out['mean'] = float(np.mean(list(out.values())))
    out['uniform'] = float(cols.float().mean())
    return out


class Pinned(torch.nn.Module):
    """Wraps a trained net: the input mapping with the given columns replaced by constants."""

    def __init__(self, net, cols, values):
        super().__init__()
        self.net, self.cols, self.values = net, cols, values
        self.orig = net.input_mapping
        net.input_mapping = self.mapping

    def mapping(self, x):
        f = self.orig(x)
        return torch.where(self.cols, self.values.expand_as(f), f)

    def restore(self):
        self.net.input_mapping = self.orig


def pinned_success(maze, prepared, net, cols, log):
    hx = torch.tensor(np.load(prepared / 'heldout/sampled_points.npy')[:20000], device=maze.device)
    with torch.no_grad():
        mean = net.input_mapping(hx[:, :3].contiguous()).mean(0)
    pin = Pinned(net, cols, mean)
    try:
        cost, grad = P.field_cost(net), (lambda pairs: P.field_grad(net, pairs))
        out = Path(os.environ.get('SHARP_SCRATCH', str(HERE / 'report' / 'usage_eval')))
        out.mkdir(parents=True, exist_ok=True)
        res = EV.evaluate(maze, prepared, cost, grad, out, log, label='pinned', plan_seeds=[0])
    finally:
        pin.restore()
    return res['success_total']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--ids', default='')
    a = ap.parse_args()
    dev = 'cuda:0'
    state = json.load(open(HERE / 'queue/state.json'))['done']
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    want = [x for x in a.ids.split(',') if x]
    log = Logger(HERE / 'report' / 'usage_log.txt')
    refs = json.loads((HERE / 'report/summary.json').read_text())['sizes']
    for job, rid in state.items():
        if not rid or (want and rid not in want):
            continue
        _, n, seed, spec = job.split('|')
        if spec == 'octave':
            continue
        key = rid + ('|plan' if a.plan else '')
        if key in old:
            continue
        n, seed = int(n), int(seed)
        d = ROOT / f'results/registered_{rid}'
        r = json.loads((d / 'results.json').read_text())
        maze = MZ.Maze(n, dev)
        net = SN.load_field(d / 'training/checkpoint.pt', spec, maze.E, n, seed, dev)
        pl = FT.plan(spec, FT.octave(FT.draw(seed), maze.E), maze.E, n, seed)
        cols = torch.tensor(sharp_columns(pl), device=dev)
        rec = dict(spec=spec, n=n, seed=seed, weight=weight_share(net, cols))
        # the octave reference of the seed, the same base columns (replace specs) or nothing comparable (add specs)
        ref_id = (refs[str(n)]['reference'].get(str(seed)) or {}).get('runs', [None])[0]
        if ref_id and not pl['add']:
            rnet = SN.load_field(ROOT / f'results/registered_{ref_id}/training/checkpoint.pt', 'octave', maze.E, n, seed, dev)
            rec['ref_id'] = ref_id
            rec['ref_weight'] = weight_share(rnet, cols[:256])
        if a.plan:
            rec['success_seed0'] = int(np.load(d / 'rows.npz')['success'][0].sum())
            rec['pinned_seed0'] = pinned_success(maze, PREP[n], net, cols, log)
            if ref_id and not pl['add']:
                rec['ref_success_seed0'] = int(np.load(ROOT / f'results/registered_{ref_id}/rows.npz')['success'][0].sum())
                rec['ref_pinned_seed0'] = pinned_success(maze, PREP[n], rnet, cols[:256], log)
        old[key] = rec
        OUT.write_text(json.dumps(old, indent=1))
        print(job, rid, json.dumps({k: v for k, v in rec.items() if k not in ('spec', 'n', 'seed')}), flush=True)


if __name__ == '__main__':
    main()
