"""Region-restricted pool: the configurations of a cabinet pipeline run whose grasp point lies inside the scene's
compartments ("cabinet only"), written as a prepared/ directory the bank-free trainers read (load_pool).

Sources, both from the pipeline run and neither reading a test suite:
  pool      prepared/bank/source_q.npy (normalized), kept where FK(q) is inside a compartment box
  coverage  sampled/coverage/train/configurations_rad.npy (the coverage strata, half of their bases inside the
            cupboard), kept where FK(q) is inside a compartment box and the configuration passes the bank's state
            gate (geometry.ClearanceGate.states: obstacle >= 0.005 m, self >= 0.002 m, |q_norm| <= 0.5)
The speed network is the run's, copied unchanged (prepared/speed.pt), so the speed of a configuration is the same
as in every other cabinet run; only where pairs and split points are drawn changes.
"""
import sys
sys.path.insert(0, '/clean/method')
import argparse, json, shutil
from pathlib import Path
import numpy as np
import torch
import yaml
from common import RELEASE, sha, write
from fk import PositionFK


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prepared', type=Path, required=True); p.add_argument('--coverage', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    (a.output/'bank').mkdir(parents=True, exist_ok=False)
    norm = np.load(RELEASE/'configs/joint_normalization.npz'); center, scale = norm['q_center'], norm['q_range']
    boxes = {c['name']: np.array([c[k] for k in 'xyz'], np.float32) for c in yaml.safe_load(a.scene.read_text())['compartments']}
    fk = PositionFK().cuda()

    def grasp(rad):
        return torch.cat([fk(torch.as_tensor(rad[i:i+200000], device='cuda')) for i in range(0, len(rad), 200000)]).cpu().numpy()

    def region(xyz):
        inside = {k: ((xyz >= b[:, 0]) & (xyz <= b[:, 1])).all(1) for k, b in boxes.items()}
        return np.any(list(inside.values()), axis=0), {k: int(v.sum()) for k, v in inside.items()}

    pool = np.load(a.prepared/'bank/source_q.npy'); keep_pool, pool_counts = region(grasp(pool*scale+center))
    rad = np.load(a.coverage/'configurations_rad.npy'); d_obstacle = np.load(a.coverage/'d_obstacle.npy'); d_self = np.load(a.coverage/'d_self.npy')
    assert len(rad) == len(d_obstacle) == len(d_self)
    cov = ((rad-center)/scale).astype(np.float32); in_region, cov_counts = region(grasp(rad))
    gate = (d_obstacle >= .005) & (d_self >= .002) & (np.abs(cov) <= .5).all(1); keep_cov = in_region & gate
    q = np.concatenate([pool[keep_pool], cov[keep_cov]]).astype(np.float32)
    np.save(a.output/'bank/source_q.npy', q); shutil.copyfile(a.prepared/'speed.pt', a.output/'speed.pt')
    source = json.loads((a.prepared/'bank/manifest.json').read_text())
    assert sha(a.output/'speed.pt') == source['speed_checkpoint_sha256'] and source['scene_sha256'] == sha(a.scene)
    write(a.output/'bank/manifest.json', dict(
        kind='region-restricted pool (bank-free trainers only: no transitions)', region='compartments', compartments={k: v.tolist() for k, v in boxes.items()},
        file_sha256={'source_q.npy': sha(a.output/'bank/source_q.npy')}, speed_checkpoint_sha256=source['speed_checkpoint_sha256'], scene_sha256=source['scene_sha256'],
        sources=dict(pool=str(a.prepared/'bank/source_q.npy'), pool_sha256=sha(a.prepared/'bank/source_q.npy'), coverage=str(a.coverage/'configurations_rad.npy'),
                     coverage_sha256=sha(a.coverage/'configurations_rad.npy')),
        counts=dict(pool=len(pool), pool_in_region=int(keep_pool.sum()), pool_by_compartment=pool_counts, coverage=len(rad),
                    coverage_in_region=int(in_region.sum()), coverage_in_region_gate_valid=int(keep_cov.sum()), coverage_by_compartment=cov_counts, total=len(q)),
        information_boundary=dict(evaluation_rows_read=False, goals_read=False, paths_read=False)))
    write(a.output/'complete.json', dict(complete=True, total=len(q)))
    print(json.dumps(json.loads((a.output/'bank/manifest.json').read_text())['counts']))


if __name__ == '__main__': main()
