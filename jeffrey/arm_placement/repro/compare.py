"""Side-by-side table of the p_d055_s46_h60 re-creation: original runs (export registry console logs), the exported checkpoints
re-planned here (control), the full re-creation (new suite, chain_state.json) and the retraining on the original inputs
(chain_frozen_state.json). Solved runs of 900, zone -> another zone runs of 150 in brackets.     python repro/compare.py > repro/compare.md"""
import ast
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'import/arm_placement_export_2026-11-23/registry/runs'
NEW = Path.home() / '.local/share/ntfields-experiments/runs'
EVAL = re.compile(r'(\w+) (forward|reverse|bidir) steps (\d+) samples 50 horizon 5: (\d+)/(\d+) by task (\{.*?\})')
SEED = re.compile(r'\] planning seed \d: (\d+)/300; by task (\{.*?\})')
EUC = re.compile(r'\] steps 600 samples 50 seed \d: (\d+)/300')


def parse(folder, rid):
    text = (folder / rid / 'console.log').read_text()
    out = {}
    for m in EVAL.finditer(text):
        out[f'{m[2]} {m[3]}'] = (int(m[4]), ast.literal_eval(m[6])['compartment_to_compartment'])
    seeds = SEED.findall(text)
    if seeds:
        out['forward 600'] = (sum(int(s) for s, _ in seeds), sum(ast.literal_eval(t)['compartment_to_compartment'] for _, t in seeds))
    euc = EUC.findall(text)
    if euc:
        out['euclid'] = (sum(map(int, euc)), None)
    return out


def merge(folder, ids):
    out = {}
    for rid in ids:
        if rid and (folder / rid / 'console.log').exists():
            out.update(parse(folder, rid))
    return out


ORIGINAL = dict(euclid=['20261123T075650Z-68c1178b'],
                lift=['20261123T075650Z-1a7d6e92', '20261123T091548Z-cf129558', '20261123T115509Z-51d21287'],
                ntrl=['20261123T075650Z-b8d04579', '20261123T085243Z-e687f234', '20261123T114956Z-dd185a72'],
                lift2=[], ntrl2=['20261123T075650Z-6020e2ee'])
CONTROL = dict(lift=['20261123T191653Z-d5ddd746', '20261123T191759Z-a2a5b797', '20261123T192314Z-b72137ec'],
               ntrl=['20261123T193108Z-d21e7b39', '20261123T193202Z-1c5a71c0', '20261123T193451Z-b58e5c53'])


def from_state(name):
    p = HERE / name
    s = json.loads(p.read_text()) if p.exists() else {}
    return dict(euclid=[s.get('euclid')], lift=[s.get('lift_s87358'), s.get('plan_lift_s87358'), s.get('budget_lift_s87358')],
                ntrl=[s.get('ntrl_s87358'), s.get('plan_ntrl_s87358'), s.get('budget_ntrl_s87358')],
                lift2=[s.get('lift_s87359'), s.get('budget_lift_s87359')], ntrl2=[s.get('ntrl_s87359')]), s


ROWS = [('Euclid (control), ntrl-demo planner 600', 'euclid', 'euclid'),
        ('lift, ntrl-demo planner 600', 'lift', 'forward 600'), ('lift, reverse 600', 'lift', 'reverse 600'),
        ('lift, two-ended 300', 'lift', 'bidir 300'), ('lift, two-ended 600', 'lift', 'bidir 600'),
        ('lift, two-ended 1,200', 'lift', 'bidir 1200'), ('**lift, two-ended 2,400**', 'lift', 'bidir 2400'),
        ('ntrl-demo, own planner 600', 'ntrl', 'forward 600'), ('ntrl-demo, reverse 600', 'ntrl', 'reverse 600'),
        ('ntrl-demo, two-ended 600', 'ntrl', 'bidir 600'), ('ntrl-demo, two-ended 2,400', 'ntrl', 'bidir 2400'),
        ('lift seed 87359, ntrl-demo planner 600', 'lift2', 'forward 600'), ('lift seed 87359, two-ended 2,400', 'lift2', 'bidir 2400'),
        ('ntrl-demo seed 87359, own planner 600', 'ntrl2', 'forward 600')]


def main():
    frozen, fs = from_state('chain_frozen_state.json')
    chain, cs = from_state('chain_state.json')
    cols = [('original (4090, original queries)', OLD, ORIGINAL), ('original checkpoints re-planned here', NEW, CONTROL),
            ('retrained here, original data + queries', NEW, frozen), ('re-created from scratch here (new suite)', NEW, chain)]
    data = {c: {k: merge(f, v) for k, v in d.items()} for c, f, d in cols}
    cell = lambda v: '' if v is None else (f'{v[0]}' + (f' ({v[1]})' if v[1] is not None else ''))  # noqa: E731
    print('| p_d055_s46_h60 (0.55 m, ±0.46 m, 0.60 m high) | ' + ' | '.join(c for c, _, _ in cols) + ' |')
    print('|---' * (len(cols) + 1) + '|')
    for label, f, k in ROWS:
        print(f'| {label} | ' + ' | '.join(cell(data[c].get(f, {}).get(k)) for c, _, _ in cols) + ' |')
    print('\nRuns (re-created):', json.dumps(cs), '\nRuns (frozen):', json.dumps(fs))
    (HERE / 'compare.json').write_text(json.dumps({c: {f: {k: v for k, v in d.items()} for f, d in x.items()} for c, x in data.items()}, indent=1) + '\n')


if __name__ == '__main__':
    main()
