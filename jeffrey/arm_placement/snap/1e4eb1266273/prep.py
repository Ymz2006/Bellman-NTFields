"""jeffrey/lbs_arm/prepare12.py, unchanged, with larger GPU batches for this machine's 48 GB cards: 2^20 candidate configurations per
draw of the pair sampler (prepare12: 2^18) and 2 times larger clearance pieces (4 times in the first prepare runs: 27 GB, too much next to a dozen trainings). The pairs come from the same recipe and seeds; only
the order in which the length bins fill differs with the batch size.

    CUDA_VISIBLE_DEVICES=0 python prep.py --export arm_export.npz --scene scenes/<name>.yml --tests <tests run>/tests --out <dir>
"""
import functools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LBS = HERE if (HERE / 'prepare12.py').is_file() else HERE.parent / 'lbs_arm'
sys.path.insert(0, str(LBS))
import arm12 as A  # noqa: E402
import prepare12 as P  # noqa: E402

BATCH = 1 << 20


class Big(A.ArmSet):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.inter_chunk = 1 << 16

    def clearance(self, x, chunk=1 << 16):
        return super().clearance(x, chunk)

    def valid_cells(self, x, chunk=1 << 16):
        return super().valid_cells(x, chunk)

    def clearance_normal(self, x, chunk=1 << 15):
        return super().clearance_normal(x, chunk)

    def gradient(self, x, chunk=1 << 15):
        return super().gradient(x, chunk)


if __name__ == '__main__':
    batch = int(sys.argv.pop(sys.argv.index('--pair-batch') + 1)) if '--pair-batch' in sys.argv else BATCH
    if '--pair-batch' in sys.argv:
        sys.argv.remove('--pair-batch')
    if batch != 1 << 18:
        A.ArmSet = Big
        P.sample_pairs = functools.partial(P.sample_pairs, batch=batch)
    P.main()
