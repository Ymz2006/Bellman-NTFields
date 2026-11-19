"""Read-only forensics on registered fields: the lift-scale field measures (lift_scale/diag.py) for runs that did not compute them
(jeffrey/maze_scale ntrl runs and jeffrey/midpoint_dp runs on the 10/13/22 mazes), on the same 24,000 truth pairs (DG.truth_pairs, seed 5)
that every lift_scale / lift_midpoint field run scores, plus the failure split by arrival from each run's rows.npz.

    CUDA_VISIBLE_DEVICES=0 python jeffrey/lift_midpoint/posthoc.py --runs <run id> [<run id> ...] --out jeffrey/lift_midpoint/report/posthoc.json

Nothing is trained or planned; results are merged into --out by run id.
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
import lift as LF  # noqa: E402
import scenes as SN  # noqa: E402
import diag as DG  # noqa: E402
import diagnostics as EDG  # noqa: E402


def load_any(run, scene, dev):
    import plan as P
    res = json.load(open(run / 'results.json'))
    ck = run / 'training/checkpoint.pt'
    if not ck.exists():
        found = sorted((run / 'training/model').glob('*/Model_Epoch_05000_*.pt'))
        assert len(found) == 1, f'no final checkpoint in {run}'
        ck = found[0]
    payload = torch.load(ck, map_location=dev, weights_only=False)
    if 'embed_config' in payload:
        return LF.load_field(ck, scene, dev), res, str(ck)
    return P.load_field(ck, 'metric', dev), res, str(ck)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    import plan as P
    dev = 'cuda:0'
    out = json.load(open(a.out)) if a.out.exists() else {}
    cache = {}
    for rid in a.runs:
        run = ROOT / 'results' / f'registered_{rid}'
        res = json.load(open(run / 'results.json'))
        prepared = Path(res['prepared'])
        if str(prepared) not in cache:
            for k in list(cache):
                del cache[k]
            torch.cuda.empty_cache()
            scene, info = SN.load_scene(prepared, dev)
            truth = DG.Truth(scene, prepared)
            pairs, gi, t_star, D = DG.truth_pairs(scene, truth, 24000)
            blk = LF.segment_blocked(scene, pairs)
            cache[str(prepared)] = (scene, truth, pairs, t_star, D, blk)
        scene, truth, pairs, t_star, D, blk = cache[str(prepared)]
        net, res, ck = load_any(run, scene, dev)
        cost, grad = P.field_cost(net), lambda pr: P.field_grad(net, pr)
        rows = np.load(run / 'rows.npz')
        row = dict(run=rid, stage=res['stage'], checkpoint=ck, prepared=str(prepared), success_total=res['success_total'], success_by_bin=res['success_by_bin'],
                   failures=EDG.failures_by_arrival(rows), accuracy=DG.accuracy(scene, cost, pairs, t_star, D, blk),
                   wall_fit=DG.wall_fit(scene, prepared, grad))
        out[rid] = row
        acc = row['accuracy']
        print(rid, res['stage'], res['success_total'], row['failures'], 'within10 %.3f' % acc['within10'], 'los', acc.get('los'), 'rho',
              acc.get('detour', {}).get('rho_median'), 'seen', acc.get('detour', {}).get('seen_through'), 'wall', row['wall_fit']['innermost'], flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = a.out.with_suffix('.tmp')
        tmp.write_text(json.dumps(out, indent=1, default=float))
        tmp.replace(a.out)
        del net


if __name__ == '__main__':
    main()
