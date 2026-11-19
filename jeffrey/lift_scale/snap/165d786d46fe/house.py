"""Stage 'house': jeffrey/embed_warp/prepare.py (training pairs, ground truth, distance-binned certified queries, straight and oracle
controls) on a lift_scale House scene (scenes.py). embed_warp's prepare builds its scene by class; here its Gibson class is replaced, in
this process only, by a factory that returns the House, and the rest of its main() runs unchanged.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/lift_scale/house.py --name Allensville --cell-m 0.25 --out <dir>
"""
import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scenes as SN  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--cell-m', type=float, default=0.25)
    p.add_argument('--pad', type=float, default=0.5)
    p.add_argument('--seal-m', type=float, default=0.15)
    p.add_argument('--h-cells', type=float, default=0.125)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    zs = SN.house_slices(a.name)
    spec = importlib.util.spec_from_file_location('embed_prepare', SN.ROOT / 'jeffrey/embed_warp/prepare.py')
    EP = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(EP)
    EP.SC.Gibson = lambda idx, res, E, z, dev: SN.House(a.name, a.cell_m, zs, None, a.pad, a.seal_m, dev)
    sys.argv = [sys.argv[0], '--scene', 'gibson', '--h-cells', str(a.h_cells), '--out', str(a.out)] + (['--smoke'] if a.smoke else [])
    EP.main()


if __name__ == '__main__':
    main()
