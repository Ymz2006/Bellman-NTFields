"""Markdown tables for jeffrey/easy_cabinet/README.md (host): representability sweeps and registered runs.
  python tables.py lines  <results.json>...            one row per variant (file label = folder name)
  python tables.py runs   <registered_id>=<label>...   totals and per-suite successes of cabinet runs
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
SUITES = ['free_to_free', 'free_to_compartment', 'compartment_to_free', 'compartment_to_compartment', 'same_compartment', 'anywhere']


def lines(paths):
    print('| data | variant | T at s_end (truth) | T at s_hit (truth) | max slope (truth 83) | median slope in band | speed at contact (target) | speed in band (target) | speed free | RMS log T over pairs |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for p in paths:
        r = json.loads(Path(p).read_text()); tag = Path(p).parent.name
        for name, v in r['variants'].items():
            print(f"| {tag} | {name} | {v['T_end_learned']:.2f} ({v['T_end_truth']:.2f}) | {v['T_hit_learned']:.2f} ({v['T_hit_truth']:.2f}) | {v['max_slope_learned']:.0f} | {v['slope_in_band_median']:.1f} | {v['speed_contact_median']:.4f} ({v['speed_contact_target']:.3f}) | {v['speed_band_median']:.2f} ({v['speed_band_target']:.2f}) | {v['speed_free_median']:.2f} | {v['pairs_rms_log']:.2f} |")


def runs(specs):
    print('| run | label | scene | RK4 /600 | MPPI /1800 | '+' | '.join(s.replace('_', ' ') for s in SUITES)+' (MPPI, per 100 x 3 seeds) |')
    print('|---|---|---|---|---|'+'---|'*len(SUITES))
    for spec in specs:
        rid, _, label = spec.partition('='); run = ROOT/'results'/('registered_'+rid if not rid.startswith('registered_') else rid)
        s = json.loads((run/'status.json').read_text()); e = json.loads((run/'evaluation/evaluation.json').read_text())
        scene = str(s.get('scene') or s.get('settings', {}).get('scene', '')).split('/')[-1].replace('.yml', '')
        per = [sum(x['success'] for x in e['scores'] if x['method'] == 'matched_mppi' and x.get('suite', SUITES[i % 6] if 'suite' in x else None) == SUITES[i]) for i in range(6)] if all('suite' in x for x in e['scores']) else None
        if per is None:   # scores are listed suite by suite, RK4 first then MPPI per seed
            m = [x['success'] for x in e['scores'] if x['method'] == 'matched_mppi']; per = [sum(m[i::6]) for i in range(6)]
        t = s['totals']; print(f"| {run.name[11:]} | {label or run.name[11:]} | {scene} | {t['native_rk4']['success']} | {t['matched_mppi']['success']} | "+' | '.join(map(str, per))+' |')


if __name__ == '__main__':
    (lines if sys.argv[1] == 'lines' else runs)(sys.argv[2:])
