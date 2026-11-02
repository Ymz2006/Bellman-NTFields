"""Cabinet task with per-piece speed: per-piece labels, the per-piece speed network and the offline bank.

experiments/cabinet/prepare.py with the speed coefficient replaced:

- Inputs: the sampled/ directory of a finished cabinet run (FK pool pairs and coverage pairs with the release
  clearance labels, hashes checked against its source_manifest.json). The poses are reused unchanged, so every
  variant trains on the same configurations as the baseline.
- Labels: experiments/face_speed/clearance.py PieceClearance speeds per endpoint, (N, 2, P), written as memmaps
  under labels/. Covers: faces (box-face formulation; the min is <= the release label), spheres or links
  (exact decompositions; the min equals the release label, checked), release (the release two-output
  SpeedField trained exactly as cabinet/prepare.py: the same-environment baseline).
- Speed network: the release trunk with P outputs (speed_net.PieceSpeedField), each output fitted to its piece
  with the release log-MSE, seed 87358, 20,000 AdamW steps of 512 original + 512 augmentation rows, the same
  schedule and random stream as the release two-output training.
- Bank: experiments/from_source/bank.py with experiments/cabinet/geometry.py ClearanceGate (cuRobo) unchanged:
  the same seed states, walkers and exact veto as the baseline; only local_cost (the integral of 1/speed)
  depends on the speed, so every variant shares the baseline's transitions.
"""
import sys
from pathlib import Path
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/experiments/from_source'); sys.path.insert(0, '/clean/experiments/cabinet'); sys.path.insert(0, '/clean')
from common import sha, write
from speed import SpeedField
from geometry import ClearanceGate, ClearanceSpeed
from bank import build
from experiments.face_speed.clearance import PieceClearance
from experiments.face_speed.speed_net import PieceSpeedField, SeparateSpeedField, components
import argparse, json, time
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(); p.add_argument('--sampled', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True); p.add_argument('--cover', choices=['faces', 'spheres', 'links', 'split', 'environment', 'release'], required=True)
    p.add_argument('--speed-model', choices=['shared', 'separate'], default='shared', help='one trunk with an output per piece, or one network per piece')
    p.add_argument('--speed-steps', type=int, default=20000); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    inputs, scene = a.sampled/'inputs', a.scene
    out = a.output; out.mkdir(exist_ok=False); (out/'labels').mkdir(); started = time.monotonic()
    source_manifest = json.loads((inputs/'source_manifest.json').read_text())
    for name, digest in source_manifest['files'].items(): assert sha(inputs/name) == digest, name
    device = 'cuda:0'
    # 1. Per-piece labels of every endpoint, memory-mapped (N, 2, P); the release cover keeps the stored labels.
    clearance = PieceClearance(scene, a.cover if a.cover != 'release' else 'spheres', device)
    names = ['head_0', 'head_1'] if a.cover == 'release' else clearance.names; P = len(names)
    arrays = []; label_audit = []; label_started = time.monotonic()
    for kind in ['original', 'augmentation']:
        points = np.load(inputs/kind/'sampled_points.npy', mmap_mode='r'); release = np.load(inputs/kind/'task_speed_physical.npy', mmap_mode='r')
        assert points.shape[1] == 12 and release.shape == (len(points), 2, 2)
        n = min(4096, len(points)) if a.smoke else len(points)
        if a.cover == 'release':
            labels = release; audit = dict(source=kind, pairs=n, note='release labels used directly')
        else:
            labels = np.lib.format.open_memmap(out/'labels'/(kind+'.npy'), mode='w+', dtype=np.float32, shape=(n, 2, P))
            worst_min_vs_release, max_over_release, floor = 0., -1., 0
            with torch.no_grad():
                for begin in range(0, n, 4096):
                    end = min(begin+4096, n); q = torch.tensor(np.asarray(points[begin:end]).reshape(-1, 6), device=device)
                    s = clearance.speeds(q).reshape(end-begin, 2, P); labels[begin:end] = s.cpu().numpy()
                    m = s.amin(2).cpu().numpy(); r = np.asarray(release[begin:end, :, 0])
                    worst_min_vs_release = max(worst_min_vs_release, float(np.abs(m-r).max())); max_over_release = max(max_over_release, float((m-r).max())); floor += int((m <= .1+1e-6).sum())
            labels.flush()
            # Speed units: the kinematics agree with cuRobo to 1e-6 m, which is 2e-5 after the /0.05 of the label; allow 1e-4.
            if a.cover in ('spheres', 'links', 'split'): assert worst_min_vs_release <= 1e-4, ('exact cover must reproduce the release label', kind, worst_min_vs_release)
            elif a.cover == 'environment': pass   # no self term: the label is >= the release label by construction
            else: assert max_over_release <= 1e-4, ('face speed must not exceed the release speed', kind, max_over_release)
            audit = dict(source=kind, pairs=n, pieces=P, max_abs_min_vs_release=worst_min_vs_release, max_min_minus_release=max_over_release, share_min_at_floor=floor/(2*n))
        label_audit.append(audit); print('LABELS', audit, flush=True)
        ids = np.arange(n); arrays.append((points, labels, ids[ids % 100 != 0], ids[ids % 100 == 0]))
    write(out/'piece_label_audit.json', dict(cover=a.cover, pieces=names, sources=label_audit, seconds=time.monotonic()-label_started, evaluation_rows_read=False))
    # 2. Speed network: the release recipe with P outputs (two outputs for the release cover).
    torch.manual_seed(87358); rng = np.random.default_rng(87358)
    separate = a.speed_model == 'separate' and a.cover != 'release'
    speed = (SpeedField() if a.cover == 'release' else SeparateSpeedField(P, names) if separate else PieceSpeedField(P, names)).cuda()
    # Separate models: every network gets the release initialisation, its own AdamW and its own gradient clip, and all
    # see the same batches, so each is trained exactly as a stand-alone one-output speed network on its piece.
    groups = list(speed.nets) if separate else [speed]
    for g in groups: torch.nn.init.normal_(g.head.weight, std=.005); torch.nn.init.constant_(g.head.bias, -1.15)
    opts = [torch.optim.AdamW(g.parameters(), lr=.0003, weight_decay=1e-6) for g in groups]
    steps = 12 if a.smoke else a.speed_steps; history = []; speed_started = time.monotonic()
    for step in range(1, steps+1):
        qs = []; ss = []
        for q, s, ids, _ in arrays:
            rows = rng.choice(ids, 512); qs.append(q[rows].copy().reshape(-1, 6)); ss.append(np.asarray(s[rows]).reshape(-1, P))
        qt = torch.as_tensor(np.concatenate(qs), device='cuda:0'); target = torch.as_tensor(np.concatenate(ss), device='cuda:0')
        pred = components(speed, qt); loss = (pred.log()-target.clamp_min(.1).log()).square().mean()
        if not bool(torch.isfinite(loss)): raise FloatingPointError('Speed loss is nonfinite')
        for opt in opts: opt.zero_grad(set_to_none=True)
        # Mean over pieces: with separate models each network's gradient is its own loss / P; scale it back.
        (loss*len(groups)).backward() if separate else loss.backward()
        for g, opt in zip(groups, opts): torch.nn.utils.clip_grad_norm_(g.parameters(), 2); opt.step()
        if step % 250 == 0 or step == steps:
            row = dict(step=step, log_mse=float(loss), seconds=time.monotonic()-speed_started); history.append(row)
            write(out/'speed_progress.json', history); print('SPEED', row, flush=True)
    speed.eval().requires_grad_(False); validation = []
    with torch.no_grad():
        for kind, (q, s, _, ids) in zip(['original', 'augmentation'], arrays):
            release = np.load(inputs/kind/'task_speed_physical.npy', mmap_mode='r')
            piece_err, min_err, min_logs, rel_err = [], [], [], []
            for chunk in np.array_split(ids, max(1, len(ids)//1024)):
                qt = torch.tensor(q[chunk].copy().reshape(-1, 6), device='cuda:0'); target = torch.tensor(np.asarray(s[chunk]).reshape(-1, P), device='cuda:0')
                pred = components(speed, qt); piece_err.append((pred-target).abs().cpu().numpy())
                pm, tm = pred.amin(1), target.amin(1); min_err.append((pm-tm).abs().cpu().numpy()); min_logs.append((pm.log()-tm.log()).square().cpu().numpy())
                rel_err.append((pm-torch.tensor(np.asarray(release[chunk, :, 0]).reshape(-1), device='cuda:0')).abs().cpu().numpy())
            pe, me, rl = np.concatenate(piece_err), np.concatenate(min_err), np.concatenate(rel_err)
            validation.append(dict(source=kind, endpoint_count=len(me), mae=float(pe.mean()), p95=float(np.quantile(pe, .95)), log_rmse=None,
                                   min_speed=dict(mae=float(me.mean()), p95=float(np.quantile(me, .95)), log_rmse=float(np.sqrt(np.concatenate(min_logs).mean()))),
                                   min_speed_vs_release=dict(mae=float(rl.mean()), p95=float(np.quantile(rl, .95)))))
    contract = dict(steps=steps, seed=87358, scene_sha256=sha(scene), cover=a.cover, n_pieces=P, pieces=names, speed_model='separate' if separate else 'shared',
                    parameters=sum(x.numel() for x in speed.parameters()),
                    architecture=(f'{P}_separate_networks_Fourier128_harmonics4_width512_blocks6_1_output_min' if separate else f'Fourier128_harmonics4_width512_blocks6_{P}_outputs_min') if a.cover != 'release' else 'Fourier128_harmonics4_width512_blocks6_two_training_speed_outputs',
                    label_sources=f'cabinet original and augmentation per-piece clearance speeds ({a.cover}); 50/50 sampling', validation_split='pair index modulo100==0',
                    scalar_composition='minimum over the piece outputs' if a.cover != 'release' else 'FK-Y smoothstep of two outputs trained on the same label',
                    orientation='none', sampled=str(a.sampled), evaluation_rows_read=False, smoke=a.smoke)
    speed_path = out/'speed.pt'; torch.save(dict(model_state_dict=speed.state_dict(), training_contract=contract, validation=validation), speed_path)
    speed_seconds = time.monotonic()-speed_started
    write(out/'speed_audit.json', dict(training=contract, validation=validation, seconds=speed_seconds)); print('SPEED_VALIDATION', json.dumps(validation), flush=True)
    # 3. Seed states and the bank: cabinet/prepare.py unchanged (ClearanceGate is the release geometry through cuRobo).
    gate = ClearanceGate(scene); physical = ClearanceSpeed(gate)
    selected = []; count = 1024 if a.smoke else 100000; attempts = 0
    while sum(map(len, selected)) < count:
        q, _, ids, _ = arrays[attempts % 2]; rows = rng.choice(ids, min(8192, count)); candidates = q[rows].copy().reshape(-1, 6); valid = []
        for begin in range(0, len(candidates), 4096): valid.append(gate.states(torch.tensor(candidates[begin:begin+4096], device='cuda:0')).cpu().numpy())
        selected.append(candidates[np.concatenate(valid)]); attempts += 1
        if attempts > 1000: raise RuntimeError('Insufficient safe training poses')
    seeds = np.concatenate(selected)[:count]; np.save(out/'seed_states.npy', seeds)
    manifest = build(seeds, speed, gate, out/'bank', speed_path, scene, a.smoke)
    manifest['exact_veto'] = dict(densify_step_rad=.02, collision_margin=.005, self_collision_margin=.002, tilt_deg=None, overlap_logic=None, workspace=False,
                                  orientation='none (cabinet task)', gate='experiments/cabinet/geometry.py ClearanceGate')
    manifest['speed_cover'] = a.cover; manifest['speed_model'] = 'separate' if separate else 'shared'
    write(out/'bank/manifest.json', manifest); write(out/'bank/progress.json', dict(complete=True, **manifest))
    source = np.load(out/'bank/source_q.npy', mmap_mode='r'); ids = rng.choice(len(source), min(4096, len(source)), replace=False); vs_release, vs_cover = [], []
    with torch.no_grad():
        for part in np.array_split(ids, max(1, len(ids)//512)):
            q = torch.tensor(source[part].copy(), device='cuda:0'); pred = speed(q)
            vs_release.append((pred-physical(q)).abs().cpu().numpy()); vs_cover.append((pred-clearance.speed(q)).abs().cpu().numpy())
    vs_release, vs_cover = np.concatenate(vs_release), np.concatenate(vs_cover)
    write(out/'bank_speed_audit.json', dict(rows=len(vs_release), cover=a.cover, vs_release=dict(mae=float(vs_release.mean()), p95=float(np.quantile(vs_release, .95)), maximum=float(vs_release.max())),
                                            vs_cover_truth=dict(mae=float(vs_cover.mean()), p95=float(np.quantile(vs_cover, .95)), maximum=float(vs_cover.max())), evaluation_rows_read=False))
    write(out/'complete.json', dict(complete=True, smoke=a.smoke, cover=a.cover, speed_model='separate' if separate else 'shared', pieces=P, speed_steps=steps, bank_anchors=manifest['anchors'], elapsed_seconds=time.monotonic()-started,
                                    stages=dict(label_seconds=time.monotonic()-label_started-speed_seconds-manifest['elapsed_seconds'], speed_seconds=speed_seconds, bank_seconds=manifest['elapsed_seconds']),
                                    sampled=str(a.sampled), evaluation_rows_read=False))
    print('PREPARATION_COMPLETE', time.monotonic()-started, flush=True)


if __name__ == '__main__': main()
