"""Status of the pair-lift queue: every job of queue/jobs.txt with its registered id and, when finished, the headline numbers of its
results.json (solved %, failures by arrival, travel times within 10 %, line of sight, detour index, see-through share), otherwise the last
training line of its log.   python jeffrey/pair_lift/status.py [-v]
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run_id(job):
    f = HERE / 'queue/logs' / (''.join(c if c.isalnum() or c in '-_.' else '_' for c in job)[:180] + '.log')
    if not f.exists():
        return None
    m = re.search(r'"id": "([^"]+)"', f.read_text())
    return m.group(1) if m else None


def headline(res):
    s, n = res.get('success_total', [float('nan'), 1])
    f = res.get('failures', dict(arrived_wall=float('nan'), not_arrived=float('nan'), runs=1))
    acc = res['accuracy']
    det = acc.get('detour', {})
    return dict(solved=100 * s / n, wall=100 * f['arrived_wall'] / f['runs'], stall=100 * f['not_arrived'] / f['runs'], t10=100 * acc['within10'], ratio=acc['ratio_median'],
                los10=100 * acc.get('los', {}).get('within10', float('nan')), rho=det.get('rho_median', float('nan')), seen=100 * det.get('seen_through', float('nan')),
                blk10=100 * acc.get('segment_blocked', {}).get('within10', float('nan')), wallfit=res.get('wall_fit', {}).get('innermost', float('nan')), rollbacks=res.get('rollbacks', 0),
                minutes=res['elapsed_seconds'] / 60, jump=(res.get('jumps', {}).get('big') or {}).get('ratio', float('nan')))


def main():
    verbose = '-v' in sys.argv
    jobs = [l.strip() for l in open(HERE / 'queue/jobs.txt') if l.strip() and not l.startswith('#')]
    print(f'{"label":22s} {"scene":8s} {"seed":>5s} {"id":28s} solved  wall stall   T10  T/T*  LOS10   rho  seen blk10  fit   rb   min  jump')
    for j in jobs:
        label, key, seed, extra = j.split('|')
        rid = run_id(j)
        line = f'{label:22s} {key:8s} {seed:>5s} {str(rid):28s} '
        out = ROOT / 'results' / f'registered_{rid}' if rid else None
        if out and (out / 'results.json').exists():
            h = headline(json.load(open(out / 'results.json')))
            line += (f'{h["solved"]:5.1f} {h["wall"]:5.1f} {h["stall"]:5.1f} {h["t10"]:5.1f} {h["ratio"]:5.3f} {h["los10"]:5.1f} {h["rho"]:5.2f} {h["seen"]:5.1f} {h["blk10"]:5.1f} '
                     f'{h["wallfit"]:4.2f} {h["rollbacks"]:4d} {h["minutes"]:5.1f} {h["jump"]:5.2f}')
        elif out and (out / 'log.txt').exists():
            lines = [l for l in (out / 'log.txt').read_text().splitlines() if 'epoch' in l or 'MPPI' in l]
            line += (lines[-1][:230] if verbose else lines[-1][11:150]) if lines else 'starting'
        else:
            line += 'queued'
        print(line)
        if verbose:
            print('    ' + extra)


if __name__ == '__main__':
    main()
