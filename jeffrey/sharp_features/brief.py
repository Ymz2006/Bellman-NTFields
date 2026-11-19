"""One line per finished run of this study: success, change against the seed's octave reference, field measures, ripple."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = {13: {}, 22: {}}
summ = json.loads((HERE / 'report/summary.json').read_text())
for n, v in summ['sizes'].items():
    for s, r in v['reference'].items():
        REF[int(n)][int(s)] = r['mean_rate']
rip = json.loads((HERE / 'report/ripple.json').read_text()) if (HERE / 'report/ripple.json').exists() else {}
ids = sys.argv[1:]
for rid in ids:
    r = json.loads((ROOT / f'results/registered_{rid}/results.json').read_text())
    n, s = r['maze']['n'], r['seed']
    k, N = r['success_total']
    w = r['wall'].get('2-8', {})
    rp = rip.get(rid, {})
    ref = REF[n].get(s)
    print(f"{n}x{n} s{s} {r['spec']:22s} {100 * k / N:5.1f} %  d {100 * (k / N - ref) if ref else float('nan'):+5.1f}  speed {r['speed_fit']['log_speed_corr']:.3f} "
          f"slope {w.get('slope', 0):.2f} bumps {r['bumps']['bumps_per_path']:5.2f} never {r['failures']['never_arrived']:3d} wall {r['failures']['arrived_after_wall']:3d} "
          f"loss {r['final_loss']:.3f} logerr {r['accuracy']['log_err_median']:.3f}"
          + (f" | uphill {rp['uphill_share']:.3f} eik {rp['eikonal_residual_median']:.3f}" if rp else ''))
