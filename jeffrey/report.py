"""Tables for jeffrey/README.md from registered jeffrey-guard / jeffrey-sharp runs (read-only).

    report.py --guard results/registered_<A> results/registered_<B> ... [--sharp results/registered_<S>] > tables.md
"""
import argparse, json
from pathlib import Path
import numpy as np

SUITES = ['free_to_free', 'free_to_compartment', 'compartment_to_free', 'compartment_to_compartment', 'same_compartment', 'anywhere']
SHORT = dict(free_to_free='f→f', free_to_compartment='f→c', compartment_to_free='c→f', compartment_to_compartment='c→c', same_compartment='same c', anywhere='any')


def guard_rows(run):
    status = json.loads((run/'status.json').read_text()); rows = []
    for name in status['results']:
        ev = json.loads((run/name/'evaluation.json').read_text()); t = ev['totals']; sc = ev['scores']
        conn_fail = sum(sum(r['connected_failures'].values()) for r in sc)
        cs = np.mean([r['cs_len_mean'] for r in sc if r['success']]); ee = np.mean([r['ee_len_mean'] for r in sc if r['success']])
        rows.append(dict(run=run.name, name=name, field=ev['field'], distance=ev['distance']['mode'], eps=ev['eps']*1000, planner=ev['planner'], success=t['success'], n=t['n'],
                         per_seed=t['per_seed'], connected=t['connected'], conn_fail=conn_fail, per_suite=t['per_suite'], seconds_per_query=t['seconds']/t['n'], cs=cs, ee=ee,
                         depart_failed=sum(r.get('unconnected_departure_failed', 0) for r in sc)))
    return rows


def main():
    p = argparse.ArgumentParser(); p.add_argument('--guard', nargs='*', type=Path, default=[]); p.add_argument('--sharp', nargs='*', type=Path, default=[]); a = p.parse_args()
    rows = [r for run in a.guard for r in guard_rows(run)]
    if rows:
        print('|Variant (run)|Field|Distance, eps|Success /1800|Per seed|Connected|Failed certified paths|Unconnected after failed departure|s/query|Path rad / m|')
        print('|---|---|---|---:|---|---:|---:|---:|---:|---|')
        for r in rows:
            print(f"|`{r['name']}` ({r['run'][11:]})|{r['field']}|{r['distance']}, {r['eps']:.1f} mm|{r['success']} ({100*r['success']/r['n']:.1f}%)|{r['per_seed']}|{r['connected']}|{r['conn_fail']}|{r['depart_failed']}|"
                  f"{r['seconds_per_query']:.2f}|{r['cs']:.2f} / {r['ee']:.2f}|")
        print(); print('|Variant|'+'|'.join(SHORT[s] for s in SUITES)+'|'); print('|---|'+'---:|'*len(SUITES))
        for r in rows: print(f"|`{r['name']}`|"+'|'.join(f"{r['per_suite'].get(s, 0)}/300" for s in SUITES)+'|')
    for run in a.sharp:
        status = json.loads((run/'status.json').read_text())
        print(); print(f'Bellman-sharp run {run.name}:'); print(); print('|Generation|'+'|'.join(['RK4 /600', 'P-NTFields planner /600', 'ntrl MPPI /1800', 'Release MPPI /1800'])+'|'); print('|---|---:|---:|---:|---:|')
        for g, t in status['results'].items():
            cell = lambda k: f"{t[k]['success']}" if k in t else '–'
            print(f"|{g}|{cell('native_rk4')}|{cell('pnt_plan')}|{cell('ntrl_mppi')}|{cell('matched_mppi')}|")
        diag = run/'sharp_diagnostic.json'
        if diag.exists():
            d = json.loads(diag.read_text()); print(); print(f"Diagnostic ({d['free_pairs']} collision-free / {d['colliding_pairs']} colliding straight segments):"); print()
            print('|Checkpoint|T/T0 free|T/T0 colliding|Bellman gap free|Bellman gap colliding|'); print('|---|---:|---:|---:|---:|')
            for r in d['generations']:
                print(f"|{r['checkpoint']}|{r['free']['T_over_T0_median']:.4f}|{r['colliding']['T_over_T0_median']:.4f}|{100*r['free']['gap_median']:.2f}%|{100*r['colliding']['gap_median']:.2f}%|")


if __name__ == '__main__': main()
