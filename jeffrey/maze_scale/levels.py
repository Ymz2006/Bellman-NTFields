"""Experiment 1 (fewer networks): segment DP + wall term (weight 0.3) on 13 x 13 with fewer, wider DP levels.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/levels.py

Same ntrl-demo base network per training seed, same first edge (0.5 cells) and per-level budget; only the geometric
ratio of the edges changes. On 13 x 13 the largest gap is 15.5-15.7 cells, so r = 2^(5/L) gives L levels whose next
edge would be 16 cells: L = 5 (r = 2, doubling), 7 (1.641), 9 (1.470); the reference is the existing r = 1.35 runs
(L = 12). Networks = L + 1 (the base). Writes report/levels.md and report/levels.json.
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report as R  # noqa: E402
from common import bin_labels  # noqa: E402

N = 13
BASE = 'segment DP + wall term (weight 0.3)'


def main():
    prep, ntrl, seg = R.collect()
    labs = [l for l in bin_labels() if prep[N]['queries']['per_bin'].get(l)]
    rows = [('ntrl-demo (1 network)', 1, ntrl[N])]
    for v, by_n in seg.items():
        if v.startswith(BASE) and N in by_n:
            d0 = next(iter(by_n[N].values()))
            rows.append((v, d0['levels'] + 1, by_n[N]))
    rows.sort(key=lambda r: r[1])
    out, md = [], []
    md.append(f'| networks | variant | runs solved per seed (87358 / 1 / 2) | all seeds | ' + ' | '.join(f'{l} cells' for l in labs)
              + ' | T/T* (16-32) | DP minutes per seed |')
    md.append('|' + '---|' * (6 + len(labs)))
    for name, nets, by in rows:
        seeds = sorted(by, key=lambda s: (s != 87358, s))
        tot = [by[s]['success_total'] for s in seeds]
        per_bin = []
        for l in labs:
            p = [by[s]['success_by_bin'][l] for s in seeds if l in by[s]['success_by_bin']]
            per_bin.append([sum(x[0] for x in p), sum(x[1] for x in p)])
        tt = [by[s]['field']['by_bin'].get('16-32', {}).get('t_over_tstar_median') for s in seeds]
        tt = [x for x in tt if x is not None]
        mins = [by[s].get('elapsed_seconds', np.nan) / 60 for s in seeds] if nets > 1 else []
        rec = dict(name=name, networks=nets, seeds=seeds, per_seed=[t[0] for t in tot], total=[sum(t[0] for t in tot), sum(t[1] for t in tot)],
                   per_bin=dict(zip(labs, per_bin)), t_over_tstar_16_32=float(np.mean(tt)) if tt else None,
                   minutes=[float(m) for m in mins], ids=[by[s]['_id'] for s in seeds])
        out.append(rec)
        md.append(f'| {nets} | {name} | {" / ".join(map(str, rec["per_seed"]))} | {100 * rec["total"][0] / rec["total"][1]:.0f} % ({rec["total"][0]}/{rec["total"][1]}) | '
                  + ' | '.join(f'{100 * a / b:.0f}' for a, b in per_bin)
                  + f' | {rec["t_over_tstar_16_32"]:.2f} | ' + (' / '.join(f'{m:.0f}' for m in mins) if mins else '-') + ' |')
    (R.OUT / 'levels.json').write_text(json.dumps(out, indent=1))
    (R.OUT / 'levels.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
