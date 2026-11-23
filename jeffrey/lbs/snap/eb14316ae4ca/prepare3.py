"""Prepare a 3-D Gibson house (scenes3.House3, a point robot in R^3) for the LBS benchmark: jeffrey/embed_warp/prepare.py (training
pairs, ground truth, distance-binned certified queries, straight control) run unchanged on the House3, as jeffrey/lift_scale/house.py
does for the 2-D houses. In this process only: embed_warp's Gibson class is replaced by a factory that returns the House3; the truth
grid gets z nodes at the spacing of its x, y nodes (the maze grid has 13 heading nodes); path lengths count all three coordinates.
Afterwards the scene's signed-distance table is stored next to the prepared files (house3_table.npy), so that field runs load it
instead of voxelizing the mesh again, and euclid_cells in queries.npz is replaced by the 3-D distance.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/lbs/prepare3.py --name Allensville --out <dir>
"""
import argparse
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scenes3 as S3  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--cell-m', type=float, default=0.25)
    p.add_argument('--vox-per-cell', type=int, default=10)
    p.add_argument('--slab', type=float, nargs=2, default=[0.15, 2.15], help='metres above the floor')
    p.add_argument('--h-cells', type=float, default=0.2, help='truth grid step (cells), all three directions')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location('embed_prepare', S3.ROOT / 'jeffrey/embed_warp/prepare.py')
    EP = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(EP)
    made = {}

    def factory(idx, res, E, z, dev):
        made['scene'] = S3.House3(a.name, a.cell_m, a.vox_per_cell, a.slab, device=dev)
        return made['scene']

    Grid = EP.TR.Grid

    def grid3(scene, h_cells=a.h_cells):
        return Grid(scene, h_cells=h_cells, n_theta=int(round(scene.z_cells / h_cells)) + 1)

    def certify(scene, grid, fields, gi, start, goal):
        paths = grid.descend(fields, gi, start, goal)
        ok, plen, clen = [], [], []
        for q in paths:
            dense = EP.P.densify(q, 0.005 * scene.pitch)
            ok.append(bool((scene.clearance(dense) >= 0).all()))
            L = float(torch.linalg.vector_norm(q[1:] - q[:-1], dim=1).sum())
            plen.append(L / scene.pitch)
            clen.append(L)
        return paths, np.array(ok), np.array(plen), np.array(clen)

    EP.SC.Gibson = factory
    EP.TR.Grid = grid3
    EP.certify = certify
    sys.argv = [sys.argv[0], '--scene', 'gibson', '--h-cells', str(a.h_cells), '--out', str(a.out)] + (['--smoke'] if a.smoke else [])
    EP.main()
    scene = made['scene']
    np.save(a.out / S3.TABLE, scene.table.reshape(scene.shape).cpu().numpy())
    q = dict(np.load(a.out / 'queries.npz'))
    q['euclid_cells'] = np.linalg.norm(q['start'] - q['goal'], axis=1) / scene.pitch
    np.savez(a.out / 'queries.npz', **q)
    check, _ = S3.load_scene(a.out, scene.device)
    x = scene.uniform(100_000, torch.Generator(scene.device).manual_seed(1))
    assert torch.equal(check.clearance(x), scene.clearance(x)), 'the stored table does not rebuild the scene'
    print(f'house3 {a.name}: table {scene.shape} stored, {len(q["start"])} queries, build {scene.build_stats}')


if __name__ == '__main__':
    main()
