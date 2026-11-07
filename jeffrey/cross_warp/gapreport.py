"""Tables and figure for the gap study (the cross maze's quadrants pushed apart): report/gap_tables.md, gap_summary.json,
gap.png.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/cross_warp/gapreport.py

Per gap scale k and method, MPPI success (planning seeds 0/1/2 pooled over training seeds) on the k's region-stratified
query set (gapprepare.py), overall and by region-pair category (same quadrant, adjacent quadrants, diagonal quadrants,
quadrant - open, open - open). k = 1 is the 22 x 22 cross maze: its fields are the cross-warp study's runs, planned on
the k = 1 region query set by evalq.py (planning seeds 0/1/2); k >= 2 are this study's training runs, planned on their
own prepared queries.

Methods (magnifications scale with the maze side E = n + 0.1 so the quadrants look the same to the encoder at every k):
  ntrl-demo        ntrl-demo (k = 1: the registered maze_scale plus-ntrl runs; k >= 2: the identity warp, the same code)
  uniform          uniform stretch x 4 E / 22.1 (ntrl-demo's Fourier matrix multiplied)
  constructed      separable warp: maze (quadrant columns / rows and the band's edge cells) x 6 E / 22.1, open band x 1
  learned          separable warp learned from the field's loss + free-space and wall-normal terms, bounded x1 .. x 6 E / 22.1
  warp + scaling   the README warp (learned, box-preserving coupling, free space pulled to a quarter) times a uniform x 4
  README warp      the same README warp alone (no scaling): it only moves resolution from the open band to the quadrants
"""
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'report'
REG = Path(os.environ.get('NTFIELDS_REGISTRY', Path.home() / '.local/share/ntfields-experiments')) / 'runs'
CATS = ['same quadrant', 'adjacent quadrants', 'diagonal quadrants', 'quadrant - open', 'open - open']
METHODS = ['ntrl-demo', 'uniform', 'constructed', 'learned', 'warp + scaling', 'README warp']
K1_LABELS = {'ntrl-demo': 'ntrl-demo', 'u4': 'uniform', 'pm6': 'constructed', 'la6b': 'learned', 'ws': 'warp + scaling', 'lf': 'README warp'}
KN_LABELS = {'id': 'ntrl-demo', 'u': 'uniform', 'pm': 'constructed', 'la': 'learned', 'ws': 'warp + scaling', 'lf': 'README warp', 'lap': 'learned, constructed start',
             'pmh1': 'constructed, heading ×1', 'idh': 'ntrl-demo, heading ×m', 'pmlb': 'constructed + lower bound', 'idlb': 'ntrl-demo + lower bound',
             'pmlos': 'constructed + line of sight', 'idlos': 'ntrl-demo + line of sight', 'wslos': 'warp + scaling + line of sight',
             'pmh1los': 'constructed, heading ×1 + line of sight', 'idtd': 'ntrl-demo, TD step 0.66 cells', 'pmtd': 'constructed, TD step 0.66 cells',
             'wsfix': 'warp + scaling + line of sight + TD step', 'idfix': 'ntrl-demo + line of sight + TD step',
             'pmbest': 'constructed, heading ×1 + line of sight + TD step'}
EXTRA = ['learned, constructed start', 'constructed, heading ×1', 'ntrl-demo, heading ×m', 'constructed + lower bound', 'ntrl-demo + lower bound',
         'constructed + line of sight', 'ntrl-demo + line of sight', 'warp + scaling + line of sight', 'constructed, heading ×1 + line of sight',
         'ntrl-demo, TD step 0.66 cells', 'constructed, TD step 0.66 cells', 'warp + scaling + line of sight + TD step', 'ntrl-demo + line of sight + TD step',
         'constructed, heading ×1 + line of sight + TD step']          # controls: in the tables, not in the chart
PLUS22 = 'registered_20261102T042454Z-62864e61'


def label_of(rid):
    spec = REG / rid / 'spec.json'
    return json.loads(spec.read_text())['config'].get('label') if spec.exists() else None


def load_all():
    runs = {}
    for r in sorted(ROOT.glob('results/registered_*/results.json')):
        try:
            d = json.loads(r.read_text())
        except Exception:
            continue
        if d.get('smoke') or not d.get('complete'):
            continue
        d['_id'], d['_dir'] = r.parent.name.replace('registered_', ''), r.parent
        runs[d['_id']] = d
    return runs


OPEN = ' + open pairs'
LOCAL = ' + local pairs'
OPEN_LOCAL = ' + open + local pairs'
VARIANTS = [OPEN, LOCAL, OPEN_LOCAL]


def data_variant(d):
    """The suffix of a data-variant prepare (gapprepare --queries-from with --open-pairs / --local-pairs)."""
    o, l = 'open' in d.get('data', {}), 'local' in d.get('data', {})
    return OPEN_LOCAL if o and l else OPEN if o else LOCAL if l else None


def cross22(d):
    """A gapprepare run on the gap study's own base (the 22 x 22 cross maze), not another base of gap.py (e.g. maze10)."""
    return ((d.get('maze') or {}).get('base') or {}).get('name', 'cross22') == 'cross22'


def collect():
    """{k: {method: {seed: entry with by_category, success_total, d_cells...}}} and the prepare runs {k: results}.
    Runs trained on a data-variant prepare (gapprepare --open-pairs / --local-pairs --queries-from the main prepare of
    k: the same queries) are reported as method + the variant's suffix (OPEN, LOCAL, OPEN_LOCAL)."""
    runs = load_all()
    preps, variants = {}, {}
    for rid, d in runs.items():
        if d.get('stage') == 'gapprepare' and not d.get('queries_from') and cross22(d):
            k = d['scale']
            if k not in preps or rid > preps[k]['_id']:
                preps[k] = d
    for rid, d in runs.items():
        if d.get('stage') == 'gapprepare' and d.get('queries_from') and data_variant(d):
            k = d['scale']
            if k in preps and Path(d['queries_from']).name == preps[k]['_dir'].name:
                variants[d['_dir'].name] = (k, data_variant(d))
    out = defaultdict(lambda: defaultdict(dict))
    # k >= 2: training runs on the gap prepare runs (and on the open-pair prepares)
    for rid, d in runs.items():
        if d.get('stage') != 'cross-warp' or not d.get('gap_scale'):
            continue
        k = d['gap_scale']
        name = Path(d['prepared']).name
        if k in preps and name == preps[k]['_dir'].name:
            suffix = ''
        elif name in variants and variants[name][0] == k:
            suffix = variants[name][1]
        else:
            continue
        lab = label_of(rid)
        if lab in KN_LABELS:
            out[k][KN_LABELS[lab] + suffix][d['seed']] = dict(by_category={c: [v['success'], v['runs']] for c, v in d.get('by_category', {}).items()},
                                                              success_total=d['success_total'], _id=rid)
    # k = 1: evalq runs on the k = 1 region query set
    if 1 in preps:
        qname = preps[1]['_dir'].name
        for rid, d in runs.items():
            if d.get('stage') != 'evalq' or Path(d['queries']).name != qname:
                continue
            for frid, e in d['runs'].items():
                fd = runs.get(frid)
                if fd is None or Path(fd.get('prepared', '')).name != PLUS22:
                    continue
                lab = 'ntrl-demo' if fd['stage'] == 'plus-ntrl' else label_of(frid)
                if lab in K1_LABELS:
                    out[1][K1_LABELS[lab]][fd['seed']] = dict(by_category=e['by_category'], success_total=e['success_total'], _id=frid, eval=rid)
    return out, preps


def pooled(by_seed):
    seeds = sorted(by_seed, key=lambda s: (s != 87358, s))
    row = dict(seeds=seeds)
    tot = [by_seed[s]['success_total'] for s in seeds]
    row['total'] = 100 * sum(t[0] for t in tot) / sum(t[1] for t in tot)
    row['per_seed_pct'] = [100 * t[0] / t[1] for t in tot]
    for c in CATS:
        v = [by_seed[s]['by_category'].get(c) for s in seeds]
        v = [x for x in v if x]
        row[c] = 100 * sum(x[0] for x in v) / sum(x[1] for x in v) if v else None
        row[c + ' range'] = (min(100 * x[0] / x[1] for x in v), max(100 * x[0] / x[1] for x in v)) if v else None
    return row


def main():
    data, preps = collect()
    OUT.mkdir(exist_ok=True)
    lines = ['# Gap study: success by region pair (generated by gapreport.py)', '']
    summary = {}
    for k in sorted(data):
        p = preps.get(k)
        n = p['maze']['n'] if p else None
        lines += [f'## Gap ×{k} (band {6 * k} cells, maze {n} × {n})', '']
        if p:
            o, st = p.get('oracle', {}), p.get('straight', {})
            lines.append('Oracle (planning seed 0): ' + ', '.join(f'{c} {v[0]}/{v[1]}' for c, v in o.get('by_category', {}).items()) +
                         '; straight line: ' + ', '.join(f'{c} {v[0]}/{v[1]}' for c, v in st.get('by_category', {}).items()))
            dq = p['queries'].get('d_cells_by_category', {})
            lines.append('True travel distance (cells, 10 / 50 / 90 %): ' + ', '.join(f'{c} {v[0]:.0f}/{v[1]:.0f}/{v[2]:.0f}' for c, v in dq.items()))
            lines.append('')
        lines += ['| method | seeds | total % | ' + ' | '.join(CATS) + ' |', '|' + '---|' * (3 + len(CATS))]
        for m in METHODS + EXTRA + [m + v for v in VARIANTS for m in METHODS + EXTRA]:
            if m not in data[k]:
                continue
            r = pooled(data[k][m])
            summary[f'{k}|{m}'] = r
            lines.append(f'| {m} | {len(r["seeds"])} | {r["total"]:.1f} | ' + ' | '.join('–' if r[c] is None else f'{r[c]:.0f}' for c in CATS) + ' |')
        lines.append('')
    (OUT / 'gap_tables.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'gap_summary.json').write_text(json.dumps(summary, indent=1))
    print('\n'.join(lines))
    figure(data)
    fig_mazes(preps)


def figure(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ks = sorted(data)
    if not ks:
        return
    cols = {'ntrl-demo': '#2a78d6', 'uniform': '#6250d6', 'constructed': '#eb6834', 'learned': '#1baf7a', 'warp + scaling': '#c98500', 'README warp': '#b8418e'}
    fig, axes = plt.subplots(1, len(CATS) + 1, figsize=(3.0 * (len(CATS) + 1), 3.2), sharey=True)
    for ax, c in zip(axes, ['total'] + CATS):
        for m in METHODS:
            xs, ys, lo, hi = [], [], [], []
            for k in ks:
                if m in data[k]:
                    r = pooled(data[k][m])
                    v = r['total'] if c == 'total' else r[c]
                    if v is None:
                        continue
                    xs.append(k)
                    ys.append(v)
                    rg = (min(r['per_seed_pct']), max(r['per_seed_pct'])) if c == 'total' else r[c + ' range']
                    lo.append(rg[0])
                    hi.append(rg[1])
            if xs:
                ax.plot(xs, ys, '-o', color=cols[m], lw=2 if m in ('constructed', 'learned') else 1.5, ms=4, label=m)
                ax.vlines(xs, lo, hi, color=cols[m], alpha=0.4, lw=1)
        ax.set_xscale('log', base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([f'×{k}' for k in ks], fontsize=8)
        ax.set_title(c, fontsize=9)
        ax.set_ylim(0, 102)
        ax.grid(axis='y', color='#e6e5e0')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('MPPI success (%)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc='upper center', ncol=len(l), bbox_to_anchor=(0.5, 1.0))
    fig.supxlabel('gap between the quadrants (× the 22 × 22 cross maze\'s 6 cells)', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / 'gap.png', dpi=120)
    plt.close(fig)



def fig_mazes(preps):
    """The gap mazes at their true relative size (cells), quadrant blocks shaded, with one diagonal query's truth path."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ks = sorted(preps)
    if not ks:
        return
    nmax = max(preps[k]['maze']['n'] for k in ks)
    fig, axes = plt.subplots(1, len(ks), figsize=(2.9 * len(ks), 3.3), gridspec_kw=dict(width_ratios=[1] * len(ks)))
    for ax, k in zip(np.atleast_1d(axes), ks):
        d = preps[k]['_dir']
        info = json.load(open(d / 'maze.json'))
        n = info['n']
        G = n - 16
        for (x0, y0) in ((0, 0), (8 + G, 0), (0, 8 + G), (8 + G, 8 + G)):
            ax.add_patch(plt.Rectangle((x0, y0), 8, 8, color='#eb6834', alpha=0.10, lw=0))
        for x0, y0, x1, y1 in info['boxes_cells']:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color='#3b3a37', lw=0))
        ax.add_patch(plt.Rectangle((0, 0), n, n, fill=False, ec='#3b3a37', lw=0.9))
        q = np.load(d / 'queries.npz')
        tp = np.load(d / 'truth_paths.npy', allow_pickle=True)
        idx = np.nonzero(q['category'] == 2)[0]
        if len(idx):
            i = idx[np.argmax(q['d_cells'][idx])]
            p = np.asarray(tp[i])
            E = n + 0.1
            ax.plot(p[:, 0] * E + n / 2, p[:, 1] * E + n / 2, color='#2a78d6', lw=1.2)
        pad = (nmax - n) / 2
        ax.set_xlim(-pad - 0.5, n + pad + 0.5)
        ax.set_ylim(-pad - 0.5, n + pad + 0.5)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f'gap ×{k}: {n} × {n} cells', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'gap_mazes.png', dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
