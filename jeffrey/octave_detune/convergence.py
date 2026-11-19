"""Convergence record (inference only): every kept checkpoint of a run, evaluated with one evaluator, plus the per-epoch
training loss and the epoch rollbacks from the registry console log -> report/conv/<run id>.json.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/octave_detune/convergence.py [--loop] [--old]

Runs: this study's complete octdetune runs (checkpoints every 500 epochs: ntrl-demo saves every 100 and the pipeline keeps
the multiples of 500), and with --old the
earlier fields on the same 13x13 / 22x22 mazes (jeffrey/grid_sync octave / octaveperm / octavecircle, jeffrey/maze_scale
ntrl-demo and B x s; checkpoints every 1,000 epochs). Each checkpoint: octdetune.measure with MPPI planning seed 0 only
(exact scorer, field vs truth, held-out speed fit, near-wall slope / cos, local minima along true paths, grid-line
profiles, failures by arrival). The final checkpoint is evaluated the same way, so a run's curve comes from one
evaluator on one GPU (the 5,000-epoch success of a curve is planning seed 0's, not the 3-seed total of results.json).
Order (TIERS): epochs 1,000 / 2,000 / 5,000 of every new run first, then the earlier fields and epochs 500 / 3,000,
then 1,500 / 4,000 (--max-tier 2, default); --max-tier 3 adds 2,500 / 3,500 / 4,500. One checkpoint per
pass, then a rescan, so a newly finished run is picked up at once.
--loop: repeat every 2 minutes until queue/STOP_CONV exists or the queue is finished and everything is evaluated.
It yields the shared GPU: no evaluation while more than --max-foreign other-session python processes use it.
"""
import argparse
import json
import os
import subprocess
import re
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import octdetune as OD  # noqa: E402

OUT = HERE / 'report' / 'conv'
REG = Path.home() / '.local/share/ntfields-experiments/runs'
PREPARED = {13: '20261101T192504Z-13cf1d04', 22: '20261101T193403Z-cea05796'}
OLD_STAGES = ('gridsync', 'ntrl', 'fourier')
OLD_MODES = ('octave', 'octaveperm', 'octavecircle')
EPOCH_RE = re.compile(r'^Epoch = (\d+) -- Loss = ([0-9.eE+-]+)')
REPEAT_RE = re.compile(r'^RepeatEpoch = (\d+) -- Loss')


def field_key(d):
    if d['stage'] == 'octdetune':
        return d['spec']
    if d['stage'] == 'gridsync':
        return d['mode']
    if d['stage'] == 'ntrl':
        return 'ntrl-demo'
    return f'B x {d["b_scale"]:g}'


def candidates(old):
    want = {str((ROOT / f'results/registered_{v}').resolve()): n for n, v in PREPARED.items()}
    out = []
    for r in sorted(ROOT.glob('results/registered_*/results.json')):
        try:
            d = json.loads(r.read_text())
        except Exception:
            continue
        if not d.get('complete') or d.get('smoke') or d.get('epochs') != 5000 or d.get('variant', 'metric') != 'metric':
            continue
        n = want.get(str(Path(d.get('prepared', '')).resolve()))
        if n is None:
            continue
        st = d.get('stage')
        if st == 'octdetune':
            pri = 0
        elif old and st == 'gridsync' and d.get('mode') in OLD_MODES:
            pri = 1
        elif old and (st == 'ntrl' or (st == 'fourier' and d.get('b_scale') in (3.0, 6.0))):
            pri = 2
        else:
            continue
        d['_id'], d['_dir'], d['_n'] = r.parent.name.replace('registered_', ''), r.parent, n
        out.append((pri, d))
    return [d for _, d in sorted(out, key=lambda x: (x[0], x[1]['_id']))]


def checkpoints(d):
    folders = [q for q in (d['_dir'] / 'training/model').iterdir() if q.is_dir()]
    assert len(folders) == 1, folders
    out = {}
    for c in folders[0].glob('Model_Epoch_*.pt'):
        out[int(c.name.split('_')[2])] = c
    return dict(sorted(out.items()))


def console(rid):
    loss, repeats = {}, {}
    path = REG / rid / 'console.log'
    if not path.exists():
        return None, None
    for line in open(path, errors='replace'):
        m = EPOCH_RE.match(line)
        if m:
            loss[int(m.group(1))] = float(m.group(2))
            continue
        m = REPEAT_RE.match(line)
        if m:
            e = int(m.group(1))
            repeats[e] = repeats.get(e, 0) + 1
    return loss, repeats


def slim(r):
    r = dict(r)
    pr = dict(r.get('profiles', {}))
    pr.pop('profile_doorway_median', None)
    pr.pop('profile_wall_median', None)
    r['profiles'] = pr
    return r


TIERS = [(0, 'new', {1000, 2000, 5000}), (1, 'old', None), (1, 'new', {500, 3000}), (2, 'new', {1500, 4000}),
         (3, 'new', None)]


def tier(d, e):
    """Evaluation order: a coarse curve for every run first, the earlier fields next, finer epochs last."""
    kind = 'new' if d['stage'] == 'octdetune' else 'old'
    for t, k, epochs in TIERS:
        if k == kind and (epochs is None or e in epochs):
            return t
    return 9


def record(d):
    rid = d['_id']
    path = OUT / f'{rid}.json'
    if path.exists():
        return path, json.loads(path.read_text())
    loss, repeats = console(rid)
    rec = dict(id=rid, stage=d['stage'], field=field_key(d), n=d['_n'], seed=d['seed'], train_gpu=d.get('gpu'), eval_gpu=torch.cuda.get_device_name(0),
               loss=[loss.get(e) for e in range(1, 5001)] if loss else None,
               repeats={str(k): v for k, v in sorted(repeats.items())} if repeats is not None else None, epochs={})
    path.write_text(json.dumps(rec, default=float))
    return path, rec


def evaluate_one(d, e, c, path, rec, log):
    maze = OD.MZ.Maze(d['_n'], 'cuda:0')
    net = OD.P.load_field(c, 'metric', 'cuda:0')
    tmp = OUT / d['_id'] / f'e{e:05d}'
    r = OD.measure(maze, Path(d['prepared']), net, tmp, log, f'conv {rec["field"]} n={d["_n"]} s{d["seed"]} e{e}', [0])
    (tmp / 'paths_seed0.npy').unlink(missing_ok=True)
    r = slim(r)
    r['checkpoint_loss'] = float(c.name.split('ValLoss_')[1][:-3])
    rec['epochs'][str(e)] = r
    del net
    torch.cuda.empty_cache()
    tmpf = path.with_suffix('.tmp')
    tmpf.write_text(json.dumps(rec, default=float))
    tmpf.replace(path)


def next_item(old, max_tier):
    items = []
    for d in candidates(old):
        path, rec = record(d)
        for e, c in checkpoints(d).items():
            t = tier(d, e)
            if str(e) not in rec['epochs'] and t <= max_tier:
                items.append((t, d['_id'], e, d, c, path, rec))
    if not items:
        return None
    return min(items, key=lambda x: x[:3])


def foreign(gpu):
    """Other sessions' python compute processes on this GPU (as queue/runner.py counts them)."""
    try:
        bus = subprocess.run(['nvidia-smi', '--query-gpu=pci.bus_id', '--format=csv,noheader', '-i', gpu], capture_output=True, text=True, timeout=30).stdout.strip()
        rows = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_bus_id,pid', '--format=csv,noheader'], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return 0
    count = 0
    for row in rows.strip().splitlines():
        b, pid = [x.strip() for x in row.split(',')]
        if b != bus:
            continue
        try:
            cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace')
        except Exception:
            continue
        if 'python' in cmd and 'octave_detune' not in cmd:
            count += 1
    return count


def queue_finished():
    q = HERE / 'queue'
    try:
        st = json.load(open(q / 'state.json'))
    except Exception:
        return False
    jobs = [l.strip() for l in open(q / 'jobs.txt') if l.strip() and not l.startswith('#')]
    return all(j in st['done'] or j in st['failed'] for j in jobs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--loop', action='store_true')
    p.add_argument('--old', action='store_true')
    p.add_argument('--max-foreign', type=int, default=2, help='wait while more other-session python processes share the GPU')
    p.add_argument('--max-tier', type=int, default=2, help='0 = epochs 1000/2000/5000 of the new runs; 1 = + earlier fields, epochs 500/3000; '
                   '2 = + 1500/4000; 3 = every kept checkpoint')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = OD.Logger(OUT / 'convergence.log')
    (OUT / 'convergence.pid').write_text(str(os.getpid()))
    while True:
        if (HERE / 'queue' / 'STOP_CONV').exists():
            return
        gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
        if foreign(gpu) > a.max_foreign:
            time.sleep(60)
            continue
        item = next_item(a.old, a.max_tier)
        if item is not None:
            t, rid, e, d, c, path, rec = item
            evaluate_one(d, e, c, path, rec, log)
            continue
        if not a.loop or queue_finished():
            return
        time.sleep(120)


if __name__ == '__main__':
    main()
