"""Planning success rate per start-goal distance bin on the cabinet, from registered evaluation outputs (no planning).

Queries are binned by |q - g| in normalized joint units (the networks' coordinates). For every field:
  MPPI   frozen suites, the benchmark planner, 3 planning seeds per query  -> rate over 3 x queries in the bin
  RK4    frozen suites, one run per query                                  -> rate over queries in the bin
  RK4    the 400 long queries (gap 0.8-1.6) of a long.py run, if the field is in it
Success is the scorer's per-query flag (score.json per_query / solved_ids), so bin counts add up to the published totals.
Intervals are Wilson 95 %. Writes success_bins.json, success_bins.md and success_bins.png to --output.

usage (from the repo root):
  python jeffrey/segment_dp/success_bins.py --output OUT [--bin 0.08] [--long RUN] [--field LABEL=RUN[:PREFIX]] ...
PREFIX is the evaluation file prefix (bellman for the release evaluator, metric for experiments/ntrl_cabinet).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
STRATA = ['free_to_free', 'free_to_compartment', 'compartment_to_free', 'compartment_to_compartment', 'same_compartment', 'anywhere']
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#7c5cd6', '#8c8c8c']
DEFAULT_FIELDS = ['B5=20261027T044847Z-c42c6e8f', 'ES=20261027T175328Z-b7ab5a49', 'DP=20261027T184605Z-f57e1332',
                  'DP+TD+BELL=20261030T051716Z-502d7e3a', 'ntrl-demo=20261026T041234Z-664b5d50:metric']
DEFAULT_LONG = '20261030T065351Z-87bf2327'


def evaluation_dir(run):
    path = ROOT/'results'/f'registered_{run}'
    return path/'evaluation' if (path/'evaluation').exists() else path/'derived/evaluation'


def score(path):
    return next(iter(json.loads(path.read_text()).values()))


def frozen_gaps(tests):
    manifest = json.loads((tests/'suite_manifest.json').read_text())
    files = {s['file'].removeprefix('cabinet_').removesuffix('_100.npz'): s['file'] for s in manifest['suites']}
    gaps = []
    for stratum in STRATA:
        suite = np.load(tests/'suites'/files[stratum], allow_pickle=True)
        gaps.append(np.linalg.norm(suite['starts_norm']-suite['goals_norm'], axis=1))
    return np.concatenate(gaps)


def frozen_successes(run, prefix):
    """Per-query success counts over the 600 frozen queries (stratum-major, suite order): MPPI 0..3, RK4 0..1."""
    e = evaluation_dir(run); mppi = np.zeros(600); rk4 = np.zeros(600)
    for s, stratum in enumerate(STRATA):
        for seed in range(3):
            for q, v in score(e/f'{prefix}_mppi_{stratum}_seed{seed}.score.json')['per_query'].items():
                mppi[s*100+int(q)] += v['success']
        for q, v in score(e/f'{prefix}_rk4_{stratum}.score.json')['per_query'].items():
            rk4[s*100+int(q)] += v['success']
    return mppi, rk4


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p, max(c-h, 0.0), min(c+h, 1.0)


def binned(gaps, wins, trials_per_query, edges):
    idx = np.clip(np.digitize(gaps, edges)-1, 0, len(edges)-2)
    k = np.bincount(idx, weights=wins, minlength=len(edges)-1).astype(int)
    n = np.bincount(idx, minlength=len(edges)-1)*trials_per_query
    return k.tolist(), n.tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--bin', type=float, default=0.08)
    p.add_argument('--top', type=float, default=1.6, help='last bin edge; farther queries go to the last bin')
    p.add_argument('--tests', type=Path, default=ROOT/'experiments/cabinet/tests')
    p.add_argument('--long', default=DEFAULT_LONG, help='long.py run id (its long/ directory); "" for none')
    p.add_argument('--field', action='append', help='LABEL=RUN[:PREFIX]; default: B5, ES, DP, DP+TD+BELL, ntrl-demo')
    p.add_argument('--min-n', type=int, default=20, help='bins with fewer queries are drawn hollow')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)

    edges = np.round(np.arange(0, a.top+a.bin/2, a.bin), 6)
    gaps = frozen_gaps(a.tests)
    long_dir = ROOT/'results'/f'registered_{a.long}'/'long' if a.long else None
    long_gaps = np.load(long_dir/'long_queries.npz')['gap'] if long_dir else None
    fields = []
    for spec in a.field or DEFAULT_FIELDS:
        label, rest = spec.split('=', 1); run, _, prefix = rest.partition(':')
        fields.append((label, run, prefix or 'bellman'))

    out = dict(edges=edges.tolist(), tests=str(a.tests), long_run=a.long or None, fields={})
    for label, run, prefix in fields:
        mppi, rk4 = frozen_successes(run, prefix)
        row = dict(run=run, prefix=prefix, totals=dict(mppi=int(mppi.sum()), rk4=int(rk4.sum())))
        row['mppi'] = binned(gaps, mppi, 3, edges); row['rk4'] = binned(gaps, rk4, 1, edges)
        lscore = long_dir/f'rk4_{label}.score.json' if long_dir else None
        if lscore and lscore.exists():
            ok = np.zeros(len(long_gaps)); ok[json.loads(lscore.read_text())[label]['solved_ids']] = 1
            row['rk4_long'] = binned(long_gaps, ok, 1, edges); row['totals']['long'] = int(ok.sum())
        out['fields'][label] = row
    (a.output/'success_bins.json').write_text(json.dumps(out, indent=1))

    # markdown table: rate (successes / trials) per bin
    lines = [f'Success rate per |q-g| bin of {a.bin} (cabinet). MPPI: frozen, 3 planning seeds; RK4: frozen / long.', '',
             '|bin|frozen queries|long queries|' + '|'.join(f'MPPI {l}' for l in out['fields']) + '|' +
             '|'.join(f'RK4 {l}' for l in out['fields']) + '|', '|' + '---|'*(3+2*len(fields))]
    first = next(iter(out['fields'].values())); longn = next((r['rk4_long'][1] for r in out['fields'].values() if 'rk4_long' in r), None)
    for b in range(len(edges)-1):
        nf = first['rk4'][1][b]; nl = longn[b] if longn else 0
        if not nf and not nl:
            continue
        cell = lambda k, n: f'{100*k/n:.0f} % ({k}/{n})' if n else '·'
        mp = [cell(r['mppi'][0][b], r['mppi'][1][b]) for r in out['fields'].values()]
        rk = [cell(r['rk4'][0][b], r['rk4'][1][b]) if nf else (cell(r['rk4_long'][0][b], r['rk4_long'][1][b]) if 'rk4_long' in r else '·')
              for r in out['fields'].values()]
        lines.append(f'|{edges[b]:.2f}–{edges[b+1]:.2f}|{nf}|{nl}|' + '|'.join(mp) + '|' + '|'.join(rk) + '|')
    (a.output/'success_bins.md').write_text('\n'.join(lines)+'\n')

    # figure: two rate panels on one distance axis, then the query counts per bin
    centres = (edges[:-1]+edges[1:])/2; dodge = a.bin*0.09*(np.arange(len(fields))-(len(fields)-1)/2)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 3, 1.3], hspace=0.28))
    panels = [('MPPI on the frozen queries (3 planning seeds)', ['mppi']), ('RK4 on the frozen and the long queries', ['rk4', 'rk4_long'])]
    for ax, (title, keys) in zip(axes[:2], panels):
        for i, (label, row) in enumerate(out['fields'].items()):
            color = SERIES[i % len(SERIES)]
            for j, key in enumerate(keys):
                if key not in row:
                    continue
                k, n = map(np.array, row[key]); m = n > 0
                if not m.any():
                    continue
                r = np.array([wilson(kk, nn) for kk, nn in zip(k, n)])
                x = centres+dodge[i]; per_query = 3 if key == 'mppi' else 1
                ax.plot(np.where(m, x, np.nan), np.where(m, 100*r[:, 0], np.nan), color=color, lw=1.8, label=label if j == 0 else None)
                ax.errorbar(x[m], 100*r[m, 0], yerr=[100*(r[m, 0]-r[m, 1]), 100*(r[m, 2]-r[m, 0])], fmt='none', ecolor=color, elinewidth=1, alpha=0.55, capsize=0)
                small = m & (n/per_query < a.min_n)
                ax.scatter(x[m & ~small], 100*r[m & ~small, 0], s=26, color=color, edgecolor='white', linewidth=1, zorder=3)
                ax.scatter(x[small], 100*r[small, 0], s=26, facecolor='white', edgecolor=color, linewidth=1.4, zorder=3)
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold'); ax.set_ylim(-2, 102); ax.set_ylabel('success rate (%)')
        ax.grid(axis='y', color='#e6e6e3', lw=0.8); ax.set_axisbelow(True)
    axes[0].legend(ncol=len(fields), frameon=False, loc='upper right', fontsize=9)
    nf = np.array(first['rk4'][1]); nl = np.array(longn) if longn else np.zeros_like(nf)
    axes[2].bar(edges[:-1], nf, width=a.bin*0.92, align='edge', color='#8a9aa6', label='frozen queries')
    axes[2].bar(edges[:-1], nl, width=a.bin*0.92, align='edge', bottom=nf, color='#c9d2d8', label='long queries')
    axes[2].set_ylabel('queries'); axes[2].legend(frameon=False, fontsize=9, loc='upper right'); axes[2].grid(axis='y', color='#e6e6e3', lw=0.8); axes[2].set_axisbelow(True)
    axes[2].set_xlabel('|q − g|, normalized joint units'); axes[2].set_xticks(edges[::2]); axes[2].set_xlim(0, edges[-1])
    for ax in axes:
        for e in edges[1:-1]:
            ax.axvline(e, color='#d9dcdc', lw=0.6, ls=(0, (2, 3)), zorder=0)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    fig.text(0.01, 0.005, 'Bars: Wilson 95 % intervals. Hollow points: fewer than '
             f'{a.min_n} queries in the bin. Same success flags as the scorer, so bins add up to the published totals.', fontsize=8, color='#5f6367')
    fig.savefig(a.output/'success_bins.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    print(json.dumps({l: r['totals'] for l, r in out['fields'].items()}))
    print('wrote', a.output/'success_bins.png', a.output/'success_bins.md', a.output/'success_bins.json')


if __name__ == '__main__':
    main()
