"""Load one saved field of this study and plan with it (no registry, no training): the way to use models/ on another machine.

    P=~/.venvs/bellman-ntfield/bin/python
    CUDA_VISIBLE_DEVICES=0 $P jeffrey/arm_placement/use_model.py --model jeffrey/arm_placement/models/p_d100_s30_h60/lift_s87358.pt \
        --demo jeffrey/arm_placement/report/demo.npz --out /tmp/paths.npz

    --model    a file written by finalize.py (weights, B matrix, embedding settings). Its folder holds the placement's scene.json,
               robot_model.npz and queries.npz; give another folder with --scene-dir.
    --demo     an .npz with start_rad and goal_rad, (N, 12) in radians, [left arm 6, right arm 6] (report/demo.npz). Without it the
               placement's own test queries (queries.npz) are planned.
    --planner  bidir (two heads, --steps per head; bidir.plan_bidir) or forward (ntrl-demo's MPPI; core.plan)

Every plan is checked by core.score at 0.001 rad per joint (5 mm from the scene, 5 mm between the arms, 2 mm self-clearance), the rule
that certified the test queries. Needs the repository (jeffrey/lbs_arm) and baselines/ntrl-demo (the network class is ntrl-demo's).
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'lbs_arm'))
import arm12  # noqa: E402
sys.modules['arm'] = arm12
import arm as A  # noqa: E402
import core as C  # noqa: E402
import bidir as BD  # noqa: E402
import trainer6 as TRN  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--scene-dir', type=Path)
    p.add_argument('--demo', type=Path)
    p.add_argument('--planner', default='bidir', choices=['bidir', 'forward'])
    p.add_argument('--steps', type=int, default=2400)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    dev = 'cuda:0'
    folder = a.scene_dir or a.model.parent
    scene, _ = A.load_scene(folder, dev)
    rng = scene.q_range.cpu().numpy()
    if a.demo:
        d = np.load(a.demo, allow_pickle=True)
        start, goal = d['start_rad'] / rng, d['goal_rad'] / rng
    else:
        q = np.load(folder / 'queries.npz')
        start, goal = q['start'], q['goal']
    if a.limit:
        start, goal = start[:a.limit], goal[:a.limit]
    qs, qg = torch.tensor(start, device=dev, dtype=torch.float32), torch.tensor(goal, device=dev, dtype=torch.float32)
    net = C.load_field(TRN.ntrl_modules()[2], a.model, scene, dev)
    ev = C.Compiled(net, scene, C.Clearance(scene, compile=False), compile=False)
    planner = C.planner_settings(scene, sample_num=50, horizon=5)
    if a.planner == 'bidir':
        paths, conv, _ = BD.plan_bidir(ev, qs, qg, a.seed, a.steps, **planner)
    else:
        paths, conv = C.plan(ev, qs, qg, a.seed, a.steps, **planner)
    rows = C.score(scene, paths, conv.cpu().numpy(), step_rad=1e-3)
    ok = np.array([bool(r['success']) for r in rows])
    for i, (pth, r) in enumerate(zip(paths, rows)):
        print(f'query {i}: {"solved" if ok[i] else "FAILED"}, {len(pth)} waypoints')
    print(f'{int(ok.sum())} of {len(ok)} solved ({a.planner}, {a.steps} steps, planning seed {a.seed})')
    if a.out:
        arr = np.empty(len(paths), dtype=object)
        arr[:] = [(pth * scene.q_range).cpu().numpy().astype(np.float32) for pth in paths]
        np.savez(a.out, paths_rad=arr, solved=ok, start_rad=start * rng, goal_rad=goal * rng)
        print('wrote', a.out, '(paths_rad: object array of (waypoints, 12) arrays, radians)')


if __name__ == '__main__':
    main()
