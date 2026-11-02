"""The hierarchical point-field speed as the frozen coefficient of the cabinet Bellman pipeline.

Same recipe as experiments/face_speed (Task 3): the poses, the seed states, the bank walk and its transitions, the
338,177-parameter TravelCost, losses 6/2/2, seed 87358, 5 rounds and the evaluation are those of the baseline
rerun (`face-speed --method bellman --cover release`); only the speed module changes:

  learned  Hierarchical with a trained PointField (point_field.py): FK sphere centres, learned D(c_i) - r_i, min;
           exact 0.5 d_self; release speed formula.
  exact    Hierarchical with the exact scene distance: the release clearance label itself (the oracle speed).

prepare  replays the random stream of face_speed/prepare_pieces.py (its 20,000 speed-training draws consume the
         generator before the seed states are drawn) without training a network, so the seed states and the bank
         transitions are bitwise those of the baseline; checked against --reference when given. Only local_cost
         (the integral of 1/speed) depends on the speed.
train    face_speed/training_pieces.py unchanged, with its speed loader pointed at this module.
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/experiments/from_source'); sys.path.insert(0, '/clean/experiments/cabinet'); sys.path.insert(0, '/clean')
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from common import sha, write
from experiments.point_field.point_field import Hierarchical, load_field, speed_of

CHUNK = 16384   # configurations per forward chunk: 50 sphere queries each through the point MLP


class HierarchicalSpeed(nn.Module):
    """forward(q_norm) -> body speed (B,), evaluated in chunks so a bank call of 278k configurations fits on the GPU."""
    def __init__(self, scene, field_path=None, device='cuda:0'):
        super().__init__()
        self.hierarchy = Hierarchical(scene, load_field(field_path, device) if field_path else None, device)

    def forward(self, q):
        with torch.no_grad():
            return torch.cat([speed_of(self.hierarchy.clearances(q[s:s+CHUNK])[2]) for s in range(0, len(q), CHUNK)])


def load_speed(path, device):
    payload = torch.load(path, map_location='cpu', weights_only=False); c = payload['training_contract']
    return HierarchicalSpeed(c['scene'], c.get('field'), device).eval().requires_grad_(False)


def prepare(a):
    from geometry import ClearanceGate, ClearanceSpeed
    from bank import build
    inputs, out = a.sampled/'inputs', a.output; out.mkdir(exist_ok=False); started = time.monotonic()
    source_manifest = json.loads((inputs/'source_manifest.json').read_text())
    for name, digest in source_manifest['files'].items(): assert sha(inputs/name) == digest, name
    arrays = []
    for kind in ['original', 'augmentation']:
        points = np.load(inputs/kind/'sampled_points.npy', mmap_mode='r'); n = min(4096, len(points)) if a.smoke else len(points)
        ids = np.arange(n); arrays.append((points, ids[ids % 100 != 0]))
    # The baseline's generator: prepare_pieces.py draws rng.choice(ids, 512) per source per speed step before the seed states.
    torch.manual_seed(87358); rng = np.random.default_rng(87358); steps = 12 if a.smoke else 20000
    for _ in range(steps):
        for _, ids in arrays: rng.choice(ids, 512)
    field = str(a.field) if a.cover == 'learned' else None
    speed_path = out/'speed.pt'
    contract = dict(cover='point-field-'+a.cover, n_pieces=1, field=field, field_sha256=sha(a.field) if field else None, scene=str(a.scene), scene_sha256=sha(a.scene),
                    composition='min over FK spheres of D(c_i) - r_i, min with exact 0.5 d_self, clip(d, 0.005, 0.05)/0.05', sampled=str(a.sampled), evaluation_rows_read=False, smoke=a.smoke)
    torch.save(dict(training_contract=contract), speed_path)
    speed = load_speed(speed_path, 'cuda:0')
    gate = ClearanceGate(a.scene); physical = ClearanceSpeed(gate)
    selected = []; count = 1024 if a.smoke else 100000; attempts = 0
    while sum(map(len, selected)) < count:
        q, ids = arrays[attempts % 2]; rows = rng.choice(ids, min(8192, count)); candidates = q[rows].copy().reshape(-1, 6); valid = []
        for begin in range(0, len(candidates), 4096): valid.append(gate.states(torch.tensor(candidates[begin:begin+4096], device='cuda:0')).cpu().numpy())
        selected.append(candidates[np.concatenate(valid)]); attempts += 1
        if attempts > 1000: raise RuntimeError('Insufficient safe training poses')
    seeds = np.concatenate(selected)[:count]; np.save(out/'seed_states.npy', seeds)
    if a.reference: assert np.array_equal(seeds, np.load(a.reference/'seed_states.npy')), 'seed states differ from the reference run'
    manifest = build(seeds, speed, gate, out/'bank', speed_path, a.scene, a.smoke)
    manifest['exact_veto'] = dict(densify_step_rad=.02, collision_margin=.005, self_collision_margin=.002, tilt_deg=None, overlap_logic=None, workspace=False,
                                  orientation='none (cabinet task)', gate='experiments/cabinet/geometry.py ClearanceGate')
    manifest['speed_cover'] = contract['cover']
    same = {}
    if a.reference:
        ref = json.loads((a.reference/'bank/manifest.json').read_text())['file_sha256']
        same = {name: manifest['file_sha256'][name] == ref.get(name) for name in manifest['file_sha256']}
        for name in ('source_q.npy', 'successor_q.npy', 'safe.npy'): assert same[name], f'bank {name} differs from the reference run'
        manifest['reference_bank'] = dict(run=str(a.reference), identical_files=same)
    write(out/'bank/manifest.json', manifest); write(out/'bank/progress.json', dict(complete=True, **manifest))
    source = np.load(out/'bank/source_q.npy', mmap_mode='r'); ids = rng.choice(len(source), min(4096, len(source)), replace=False); err = []
    with torch.no_grad():
        for part in np.array_split(ids, max(1, len(ids)//512)):
            q = torch.tensor(source[part].copy(), device='cuda:0'); err.append((speed(q)-physical(q)).abs().cpu().numpy())
    err = np.concatenate(err)
    write(out/'bank_speed_audit.json', dict(rows=len(err), cover=contract['cover'], vs_release=dict(mae=float(err.mean()), p95=float(np.quantile(err, .95)), maximum=float(err.max())), evaluation_rows_read=False))
    write(out/'complete.json', dict(complete=True, smoke=a.smoke, cover=contract['cover'], bank_anchors=manifest['anchors'], elapsed_seconds=time.monotonic()-started,
                                    bank_seconds=manifest['elapsed_seconds'], reference_identical=same, sampled=str(a.sampled), evaluation_rows_read=False))
    print('PREPARATION_COMPLETE', time.monotonic()-started, flush=True)


def train(a):
    import experiments.face_speed.training_pieces as T
    T.load_piece_speed = load_speed
    sys.argv = [sys.argv[0], '--output', str(a.output), '--prepared', str(a.prepared)]+(['--smoke'] if a.smoke else [])
    T.main()


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'train']); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sampled', type=Path); p.add_argument('--scene', type=Path); p.add_argument('--cover', choices=['learned', 'exact'])
    p.add_argument('--field', type=Path); p.add_argument('--reference', type=Path, help='prepared/ of the baseline rerun: seed states and transitions must match')
    p.add_argument('--prepared', type=Path); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    if a.stage == 'prepare':
        if a.cover == 'learned' and not a.field: raise SystemExit('--cover learned needs --field')
        prepare(a)
    else: train(a)


if __name__ == '__main__': main()
