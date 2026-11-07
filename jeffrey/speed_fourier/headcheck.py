"""How large is the head term of part 4 (embed.install_head) against the true travel time, before any training? report/headcheck.json.

    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= ~/.venvs/bellman-ntfield/bin/python jeffrey/speed_fourier/headcheck.py

Per maze and norm, over the prepared planning queries: term = scale * N(Z(start) - Z(goal)) with the 16 exact-integral rows Z and the run's scale
(speedfourier.head_scale). above[w] = share of queries where w * term exceeds T*: T_net >= 0 cannot undo that.
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(ROOT / 'jeffrey/grid_sync'))
import embed as EM  # noqa: E402
import speedfourier as SF  # noqa: E402
import maze as MZ  # noqa: E402
from rayscreen import PREPARED  # noqa: E402


def main():
    res = {}
    for n, rid in PREPARED.items():
        prepared = ROOT / f'results/registered_{rid}'
        emb = SF.build(MZ.Maze(n, 'cpu'), 87358, 'top', 'integral', 0, 'true', target='slow')[0]
        q = np.load(prepared / 'queries.npz')
        Z = torch.as_tensor(emb.raw, dtype=torch.float32).reshape(emb.raw.shape[0] ** 2, -1)
        cpu = EM.Embed(None, emb.raw, 'cpu')
        d = (cpu.speed_rows(torch.tensor(q['start'][:, :2]), Z) - cpu.speed_rows(torch.tensor(q['goal'][:, :2]), Z)).abs()
        res[str(n)] = {}
        for kind in ('l1', 'linf'):
            term = SF.head_scale(emb, kind, prepared) * (d.mean(1) if kind == 'l1' else d.amax(1)).numpy()
            res[str(n)][kind] = dict(above={str(w): float((w * term > q['t_star']).mean()) for w in (1.0, 0.2)}, spearman=float(spearmanr(term, q['t_star']).statistic),
                                     ratio_q95={str(w): float(np.quantile(w * term / q['t_star'], 0.95)) for w in (1.0, 0.2)})
            print(n, kind, res[str(n)][kind], flush=True)
    (HERE / 'report/headcheck.json').write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
