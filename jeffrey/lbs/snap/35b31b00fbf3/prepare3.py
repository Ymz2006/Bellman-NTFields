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
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
UNREACHED = 1000.0                    # travel time (normalized units) stored for truth-grid nodes that no free edge reaches
SPACING, THRESH = 0.07, 0.02          # cells: sample spacing along a truth-grid edge, clearance below which a sample is in an obstacle
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

    def blocked(scene, p, step, length_cells, chunk=4_000_000):
        """Does the straight segment from each point p (N, 3) to p + step touch an obstacle? Samples at most SPACING cells apart, ends
        included; a sample counts below THRESH cells of clearance (the thinnest wall is one voxel, 1 / vox_per_cell cells)."""
        m = max(1, int(np.ceil(length_cells / SPACING)))
        out = torch.zeros(len(p), dtype=torch.bool, device=p.device)
        for i in range(0, len(p), chunk):
            for j in range(m + 1):
                out[i:i + chunk] |= scene.clearance(p[i:i + chunk] + (j / m) * (step if step.dim() == 1 else step[i:i + chunk])) < THRESH
        return out

    def grid3(scene, h_cells=a.h_cells):
        """maze_scale's truth grid with z nodes at the x, y spacing, and with every edge that touches an obstacle removed (the maze
        grid only slows such edges down through the speed of their end nodes, which a wall thinner than the node spacing escapes)."""
        g = Grid(scene, h_cells=h_cells, n_theta=int(round(scene.z_cells / h_cells)) + 1)
        nodes = torch.stack(torch.meshgrid(g.xs, g.xs, g.ts, indexing='ij'), -1).reshape(-1, 3)
        removed = 0
        for k, (i, j, l) in enumerate(g.offsets):
            step = torch.tensor([i * g.hx, j * g.hx, l * g.ht], device=nodes.device)
            b = blocked(scene, nodes, step, g.lengths[k] / scene.pitch)
            g.costs[k].view(-1)[b] = float('inf')
            removed += int(b.sum())
        g.cost_flat = torch.stack([c.reshape(-1) for c in g.costs])
        g.edges_removed_share = removed / (len(g.offsets) * len(nodes))
        seed = g.seed_block

        def seed_block(src):
            kc, cost = seed(src)
            tgt = g.node(kc)                                                     # (B, 64, 3)
            B, K = cost.shape
            for s in range(0, B, 50_000):
                p = src[s:s + 50_000, None].expand(-1, K, -1).reshape(-1, 3)
                step = (tgt[s:s + 50_000] - src[s:s + 50_000, None]).reshape(-1, 3)
                b = blocked(scene, p, step, 4 * 1.7321 * h_cells)
                cost[s:s + 50_000][b.reshape(-1, K)] = float('inf')
            return kc, cost
        g.seed_block = seed_block
        solve = g.solve

        def solve_finite(sources, **kw):
            # nodes no free edge reaches (inside obstacles, sealed pockets) get UNREACHED instead of inf: the planner control interpolates
            # the field, and 0 x inf is not a number; UNREACHED / pitch is far beyond the last distance bin, so no query uses such a node
            f, it = solve(sources, **kw)
            made['unreached'] = made.get('unreached', []) + [float((~torch.isfinite(f)).float().mean())]
            f[~torch.isfinite(f)] = UNREACHED
            return f, it
        g.solve = solve_finite
        made['grid'] = g
        return g

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
    res = json.load(open(a.out / 'results.json'))
    res['truth'].update(edges_removed_share=made['grid'].edges_removed_share, edge_sample_spacing_cells=SPACING, edge_clearance_threshold_cells=THRESH, unreached_value=UNREACHED, unreached_share=float(np.mean(made['unreached'])), z_nodes=made['grid'].shape[2])
    res['house3'] = scene.spec()
    json.dump(res, open(a.out / 'results.json', 'w'), indent=1)
    print(f'house3 {a.name}: table {scene.shape} stored, {len(q["start"])} queries, build {scene.build_stats}')


if __name__ == '__main__':
    main()
