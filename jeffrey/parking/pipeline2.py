"""pipeline.py with truth2's ground truth (edge costs integrated along each edge), for C-space walls below ~0.015.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/parking/pipeline2.py --wall 0.0025 --car-width 0.0025 --seeds 87358 --out <dir>

Identical to pipeline.py (same arguments and outputs) except that T* (oracle MPPI, T/T*, jump, descent direction)
comes from truth2.Grid; results.json records truth_version 2.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline  # noqa: E402
import truth2  # noqa: E402

pipeline.TR = truth2

if __name__ == '__main__':
    pipeline.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1])
    res = json.loads((out / 'results.json').read_text())
    res['truth_version'] = 2
    (out / 'results.json').write_text(json.dumps(res, indent=2))
