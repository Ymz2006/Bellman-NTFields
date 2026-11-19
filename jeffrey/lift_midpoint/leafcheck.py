"""Read-only forensics: what the midpoint loss's training drives look like for a registered field, and where the 'min' leaf takes the
network's value instead of the drive's own travel time.

For each run: load its final network, drive N random pairs (clearance >= 0.1 cells) exactly as a refresh does (liftfast.mppi, 'safe',
budget 2 T(a, b) / step, stall stop), cut at collisions / stalls, split at travel-time midpoints down to 1-cell leaves, then report
    drives reaching their goal, the detour share of the reached drives (path / straight > 1.3), and their path / straight ratio;
    per leaf: T_net(leaf ends) / travel time along the leaf, and the share of leaves whose min() takes the network's value;
    the same leaf ratio against the truth where a leaf end is one of the prepare run's 48 goals is not available, so the leaf's own
    travel time stands in for T* (leaves are <= 1 cell of travel time and nearly straight: path / straight 1.00-1.03).
With --driver the drives follow that frozen field instead (the driver control), the leaves are still valued with the run's network.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/lift_midpoint/leafcheck.py --runs <id> [<id> ...] --out jeffrey/lift_midpoint/report/leafcheck.json
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import trainer as TRN  # noqa: E402
import liftfast as LFT  # noqa: E402
import lift as LF  # noqa: E402
import scenes as SN  # noqa: E402
import traverse as TV  # noqa: E402
import tree as TR  # noqa: E402


def load(ck, scene, dev):
    import plan as P
    payload = torch.load(ck, map_location=dev, weights_only=False)
    if 'embed_config' in payload:
        return LF.load_field(ck, scene, dev)
    return P.load_field(ck, 'metric', dev)


def final_ck(run):
    ck = run / 'training/checkpoint.pt'
    if ck.exists():
        return ck
    found = sorted((run / 'training/model').glob('*/Model_Epoch_05000_*.pt'))
    return found[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--n', type=int, default=4096)
    p.add_argument('--seed', type=int, default=11)
    a = p.parse_args()
    import plan as P
    dev = 'cuda:0'
    out = json.load(open(a.out)) if a.out.exists() else {}
    for rid in a.runs:
        run = ROOT / 'results' / f'registered_{rid}'
        res = json.load(open(run / 'results.json'))
        prepared = Path(res['prepared'])
        scene, info = SN.load_scene(prepared, dev)
        net = load(final_ck(run), scene, dev)
        ev = LFT.Compiled(net, scene, chunk=1 << 19, compile=False)
        driver = (res.get('config') or {}).get('driver')
        drive = LFT.Compiled(load(driver, scene, dev), scene, chunk=1 << 19, compile=False) if driver else ev
        st = P.settings(scene)
        gen = torch.Generator(dev).manual_seed(a.seed)
        A = scene.free(a.n, gen, 0.1)
        B = scene.free(a.n, gen, 0.1)
        tab = drive.travel(torch.cat([A, B], 1))
        max_steps = int(math.ceil(5 * scene.n * scene.pitch / st['step']))
        budget = torch.clamp(torch.ceil(2.0 * tab / st['step']), st['min_steps'], max_steps)
        paths, stop, conv, stalled = LFT.mppi(drive, A, B, gen, budget, scene.lo, scene.hi, stall_window=60, stall_drop=0.05 * scene.pitch, safe=True,
                                              safe_margin=0.02, safe_mode='first', **st)
        d, first = TV.collisions(scene, paths, stop, clear=ev.clearance)
        collided = first <= stop
        end = torch.where(collided, (first - 1).clamp(min=0), stop)
        reached = conv & ~collided
        cum = TR.path_time(scene, paths, d)
        leaves, inner = TR.split_tree(cum, end, scene.pitch)
        la, lb = paths[leaves[:, 0], leaves[:, 1]], paths[leaves[:, 0], leaves[:, 2]]
        along = cum[leaves[:, 0], leaves[:, 2]] - cum[leaves[:, 0], leaves[:, 1]]
        tn = ev.travel(torch.cat([la, lb], 1))
        ok = along > 1e-6
        r = (tn[ok] / along[ok])
        dl = d[leaves[:, 0], leaves[:, 1]][ok]
        near = dl < 0.175
        L_path = torch.linalg.vector_norm(paths[:, 1:, :2] - paths[:, :-1, :2], dim=2).sum(1)
        L_line = torch.linalg.vector_norm(paths[torch.arange(len(paths)), end, :2] - A[:, :2], dim=1)
        ratio = L_path[reached] / L_line[reached].clamp(min=1e-6)
        row = dict(run=rid, stage=res['stage'], lift=(res.get('config') or {}).get('lift', 'none' if res['stage'] in ('ntrl', 'midpoint') else None), driver=driver,
                   drives=a.n, reached=float(reached.float().mean()), collided=float(collided.float().mean()), stalled=float((stalled & ~collided).float().mean()),
                   detour_share=float((ratio > 1.3).float().mean()) if len(ratio) else None, path_over_line_median=float(ratio.median()) if len(ratio) else None,
                   leaves=int(ok.sum()), leaf_net_over_path_median=float(r.median()), leaf_net_over_path_q10=float(r.quantile(0.1)), leaf_net_over_path_q90=float(r.quantile(0.9)),
                   min_takes_net=float((r < 1).float().mean()), min_takes_net_near_wall=float((r[near] < 1).float().mean()) if bool(near.any()) else None,
                   leaf_ratio_near_wall=float(r[near].median()) if bool(near.any()) else None, leaf_ratio_free=float(r[~near].median()) if bool((~near).any()) else None)
        out[rid] = row
        print(json.dumps(row), flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
        del net, ev, drive, paths, d, cum
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
