"""Write a one-line conclusion into the registry for every run of this study that has none yet.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/aniso_eikonal/notes.py [--dry]

`registry.py note` appends, so runs that already carry notes are skipped.
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DB = Path(os.environ.get('NTFIELDS_REGISTRY', Path.home() / '.local/share/ntfields-experiments')) / 'registry.sqlite3'
WHAT = {'id': 'no warp', 'ziso': 'trained on the stretched space (plain length)', 'zaniso': 'inner-product eikonal', 'zsplit': 'first pass: inner product on network + deficit term',
        'zboth': 'first pass: split head, both equations', 'zres': 'two-part field: head on the plain length + learned far term on the inner product of the sum'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry', action='store_true')
    a = p.parse_args()
    S = json.load(open(HERE / 'report/summary.json'))
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    have = {rid: bool(json.loads(data).get('notes')) for rid, data in con.execute("select id, data from runs where name like 'Aniso eikonal%'")}
    con.close()
    todo = []
    for r in S['runs']:
        z, x = r['plans'].get('z'), r['plans'].get('x')
        ta = r.get('time_accuracy') or {}
        kind = 'first pass: stretched-space field + closed-form deficit term, not retrained' if '-posthoc' in r['variant'] else WHAT.get(r['mode'], r['mode'])
        txt = (f'aniso_eikonal {r["base"]} gap x{r["k"]} seed {r["seed"]} [{r["variant"]}] {kind}: '
               + ', '.join(f'MPPI on the {"warp" if pl == "z" else "warp with the head alone" if pl == "zh" else "maze"} {100 * e["success"] / e["runs"]:.1f} %' for pl, e in (('z', z), ('zh', r['plans'].get('zh')), ('x', x)) if e)
               + (f'; T within 10 % of T* on {100 * ta["within_10pct"]:.0f} % of queries (median ratio {ta["ratio_median"]:.2f})' if ta else '')
               + ('; training stopped by rollbacks' if r.get('stopped') else '') + '. See jeffrey/aniso_eikonal/README.md.')
        todo.append((r['id'], txt))
    for key, v in S['verify'].items():
        pa = v['price']['all']
        sp = pa.get('split_time_over_true')
        txt = (f'aniso_eikonal ground-truth check {key}: inner product on the warp = the maze equation; warped-maze shortest path costs +{100 * (pa["warped_path_over_true_path"]["median"] - 1):.1f} % travel time (median)'
               + (f'; (Tw / m + |du|) / T* median {sp["median"]:.3f} [{sp["p10"]:.3f}, {sp["p90"]:.3f}]' if sp else '') + '. See jeffrey/aniso_eikonal/README.md.')
        todo.append((v['id'], txt))
    n = 0
    for rid, txt in todo:
        if have.get(rid):
            continue
        n += 1
        if a.dry:
            print(rid, txt)
        else:
            subprocess.run([sys.executable, str(ROOT / 'tools/experiment_registry/registry.py'), 'note', rid, txt], check=True, capture_output=True)
    print(f'{n} notes {"to write" if a.dry else "written"}, {len(todo) - n} runs already noted')


if __name__ == '__main__':
    main()
