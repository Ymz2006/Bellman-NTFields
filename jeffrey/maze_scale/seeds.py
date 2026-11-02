"""The 13 x 13 seed comparison (ntrl-demo vs segment DP vs segment DP + wall term, same training seeds) as one sentence
for README.md and the page (report/seeds.md, report/seeds.json).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/seeds.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report as R  # noqa: E402

N = 13
NAMES = [('ntrl-demo', None), ('segment DP', 'segment DP'), ('segment DP + wall term', 'segment DP + wall term'),
         ('segment DP + wall term (weight 0.3)', 'segment DP + wall term (weight 0.3)')]


def main():
    prep, ntrl, seg = R.collect()
    rows = {}
    for label, key in NAMES:
        by = ntrl.get(N, {}) if key is None else seg.get(key, {}).get(N, {})
        rows[label] = {s: d for s, d in by.items()}
    out = {}
    for label, by in rows.items():
        seeds = sorted(by, key=lambda x: (x != 87358, x))
        out[label] = {str(sd): dict(total=by[sd]['success_total'][0], mid=by[sd]['success_by_bin'].get('4-8', [0, 0])[0],
                                    long=by[sd]['success_by_bin'].get('16-32', [0, 0])[0]) for sd in seeds}
    (R.OUT / 'seeds.json').write_text(json.dumps(out, indent=1))
    parts = []
    for label, by in out.items():
        if not by:
            continue
        ks = list(by)
        parts.append(f'{label} {" / ".join(str(by[k]["total"]) for k in ks)} (4–8 cells {" / ".join(str(by[k]["mid"]) for k in ks)}; '
                     f'16–32 cells {" / ".join(str(by[k]["long"]) for k in ks)}; training seeds {" / ".join(ks)})')
    text = 'On 13×13, runs solved of 600 per training seed: ' + '; '.join(parts) + '.'
    (R.OUT / 'seeds.md').write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
