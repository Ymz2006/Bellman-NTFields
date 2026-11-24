"""More 3-D Gibson houses, LBS only (user request 2026-11-24 02:20, 10 hours): find houses where LBS has a high success rate and add them to
the Gibson 3-D table. ntrl-demo is trained and benched on a house only if LBS solves more than 80 % there.

Per house (key <House>3d):
  prep:<key>    jeffrey/lbs_final/run.py prep --kind house3 --name <House>             (GPU; same script and settings as the ten table houses)
  lbs:<key>     run.py field ... --label lbs  --extra LBS                              (GPU training, same recipe as the table's LBS fields)
  fast:<key>    jeffrey/lbs_fast/run.py bench --scene <key> --device cuda --gpu 0 ...  (the LBS row: fast planner on the RTX 5070, GPU 0 otherwise idle)
  ntrl:<key>    run.py field ... --label ntrl --extra NTRL                             (only if LBS > 80 %)
  nbench:<key>  run.py bench --methods ntrl --field ntrl=<run>                         (CPU, 2 threads, as the table's ntrl-demo row)

Scheduling. Preps go smallest house first (box side from scenes3.House3's footprint; the ten table houses were the smallest). LBS trainings go by the success predicted from the prepared queries' detour
mix (true path / straight line < 1.1, 1.1-1.5, >= 1.5, weighted by LBS's pooled success on the ten table houses: 93.2 / 51.8 / 24.3 %,
minus 1 point per cell of mean true path above 20), so
the likely high-success houses train first; houses predicted below min_pred are not trained, and preps pause while prep_ahead
prepared houses wait for a training. ntrl trainings go before new LBS trainings. The gate uses the LBS training's own 3-seed check
(the same 500 queries; the fast bench agreed with it within 0.6 points on every table scene). Fast benches run on GPU 0 only while nothing
else of this queue is on GPU 0 (no new GPU 0 launches while one is pending), so their times are like the table's.
control.json (re-read every loop): GPUs used, trainings and preps per GPU, GPUs that take one job at a time (exclusive), total trainings, deadline. `touch PAUSEMORE` holds new launches.
statemore.json (job -> run id), runner_more.log.
"""
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PY = os.path.expanduser('~/.venvs/bellman-ntfield/bin/python')
OMPL_PY = os.path.expanduser('~/.venvs/ompl-bench/bin/python')
RUN = str(ROOT / 'jeffrey/lbs_final/run.py')
FAST_RUN = str(ROOT / 'jeffrey/lbs_fast/run.py')
FAST_SCENES = ROOT / 'jeffrey/lbs_fast/scenes.json'
VARIANT = 'fast|c=ro,k=8,n=bulk'
LBS = '--lift time --bell 1 --bell-long 8 --lean --validate-every 1000'
NTRL = '--lift none --weight 0 --lean --validate-every 1000'
GATE = 0.80
POOLED = (0.932, 0.518, 0.243)
SIDE = {'Merom': 55, 'Ihlen': 55, 'Hiteman': 56, 'Wiconisco': 61, 'Uvalda': 62, 'Newfields': 64, 'Shelbyville': 64, 'Pomaria': 65, 'Tolstoy': 65, 'Cosmos': 66, 'Leonardo': 67, 'Hanson': 70, 'Beechwood': 72, 'Stockman': 73, 'McDade': 82, 'Darden': 85, 'Woodbine': 85, 'Wainscott': 86, 'Corozal': 87, 'Marstons': 87, 'Ranchester': 93, 'Klickitat': 96, 'Pinesdale': 99, 'Muleshoe': 103, 'Lindenwood': 136}   # box side in cells (0.25 m), from scenes3.House3's footprint
HOUSES = sorted(SIDE, key=lambda h: (SIDE[h], h))   # preps smallest first (the ten table houses were the smallest)
BIG = 80                     # preps of houses this big run on the 3090 only (Wainscott, 86, ran out of the 5070's 12 GB)
MIN_RAM = 6.0
state_f = HERE / 'statemore.json'
state = json.load(open(state_f)) if state_f.exists() else dict(done={}, failed={}, info={})
running = {}                                                      # job -> (Popen, gpu or 'cpu', file, kind)


def control():
    c = dict(train_per_gpu={'0': 1, '1': 1}, train_total=2, prep_per_gpu={'0': 1, '1': 1}, exclusive=['0'], cpu_slots=3, min_pred=0.6, prep_ahead=3, deadline=None, lbs_minutes=80,
             ntrl_minutes=15)
    f = HERE / 'control.json'
    if f.exists():
        c.update(json.load(open(f)))
    return c


def avail_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'):
            return int(line.split()[1]) / 1e6
    return 0


def stamp(msg):
    with open(HERE / 'runner_more.log', 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')


def save():
    tmp = state_f.with_suffix('.tmpmore')
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, state_f)


def rdir(rid):
    return ROOT / 'results' / f'registered_{rid}'


def log_path(name):
    (HERE / 'logs').mkdir(exist_ok=True)
    return HERE / 'logs' / ('more_' + ''.join(c if c.isalnum() or c in '-_.' else '_' for c in name) + '.log')


def predicted(prep_rid):
    q = np.load(rdir(prep_rid) / 'queries.npz')
    r = q['path_cells'] / np.maximum(q['euclid_cells'], 1e-9)
    f = [float(np.mean(r < 1.1)), float(np.mean((r >= 1.1) & (r < 1.5))), float(np.mean(r >= 1.5))]
    # long queries fail more often than the detour mix says (Lakeville: mix 0.75, LBS 0.40, mean true path 40 cells); ranking only
    return float(np.dot(f, POOLED) - 0.01 * max(0.0, float(q['path_cells'].mean()) - 20)), f


def lbs_rate(rid):
    s = json.load(open(rdir(rid) / 'results.json'))['success_total']
    return s[0] / s[1]


def launch(name, kind, gpu, cmd, env_extra):
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', **env_extra)
    f = open(log_path(name), 'w')
    running[name] = (subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env), gpu, f, kind)
    stamp(f'start {name} on {"GPU " + gpu if gpu != "cpu" else "CPU"} (RAM {avail_gb():.1f} GB)')


def add_fast_scene(key):
    sc = json.load(open(FAST_SCENES))
    if key not in sc:
        sc[key] = dict(prep=state['done'][f'prep:{key}'], field=state['done'][f'lbs:{key}'], note='lbs_final more houses (runner_more.py)')
        tmp = FAST_SCENES.with_suffix('.tmpmore')
        tmp.write_text(json.dumps(sc, indent=1))
        os.replace(tmp, FAST_SCENES)


def finish(name):
    p, gpu, f, kind = running.pop(name)
    f.close()
    rid = None
    for line in log_path(name).read_text().splitlines():
        if line.startswith('{"id"') and '"succeeded"' in line:
            rid = json.loads(line)['id']
    if p.returncode == 0 and rid:
        state['done'][name] = rid
    else:
        state['failed'][name] = f'exit {p.returncode}'
    key = name.split(':', 1)[1]
    info = state['info'].setdefault(key, {})
    if rid and kind == 'prep':
        info['predicted'], info['detour_mix'] = predicted(rid)
    if rid and kind == 'lbs':
        info['lbs_check'] = lbs_rate(rid)
    save()
    extra = ''
    if rid and kind == 'prep':
        extra = f' predicted LBS {info["predicted"]:.3f} mix {np.round(info["detour_mix"], 3).tolist()}'
    if rid and kind == 'lbs':
        extra = f' LBS check {info["lbs_check"]:.3f}' + (' -> ntrl-demo' if info['lbs_check'] > GATE else ' (<= 80 %, no ntrl-demo)')
    stamp(f'{"done" if rid else "FAILED"} {name} -> {rid} (exit {p.returncode}){extra}')


class Adopted:
    """A job a previous copy of this runner started whose registered run is still alive (the runner can be restarted)."""
    def __init__(self, rid):
        self.rid, self.returncode, self.gone = rid, None, 0

    def poll(self):
        if subprocess.run(['pgrep', '-f', f'registered_{self.rid}'], capture_output=True).returncode == 0:
            self.gone = 0
            return None
        self.gone += 1                                             # the registry writes its last log line after the run exits
        if self.gone < 4:
            return None
        self.returncode = 0
        return 0


def adopt():
    starts = {}
    if (HERE / 'runner_more.log').exists():
        for line in (HERE / 'runner_more.log').read_text().splitlines():
            w = line.split()
            if len(w) > 5 and w[2] == 'start':
                starts[w[3]] = w[6].rstrip(')') if w[5] == 'GPU' else 'cpu'
    for name, gpu in starts.items():
        if name in state['done'] or name in state['failed'] or not log_path(name).exists():
            continue
        rid = None
        for line in log_path(name).read_text().splitlines():
            if line.startswith('{"id"'):
                rid = json.loads(line)['id']
        if rid:
            running[name] = (Adopted(rid), gpu, open(os.devnull, 'w'), name.split(':')[0])
            stamp(f'adopt {name} ({rid}) on {gpu}')


t0 = time.time()
stamp('runner_more start')
adopt()
while True:
    c = control()
    for name in [n for n, (p, *_) in running.items() if p.poll() is not None]:
        finish(name)
    done, failed = state['done'], state['failed']
    finished = set(done) | set(failed)
    keys = [h + '3d' for h in HOUSES]
    left = max(0.0, (c['deadline'] - time.time()) / 60) if c['deadline'] else 1e9
    jobs = []                                              # (name, kind) ready to launch, in priority order
    for k in keys:
        if f'prep:{k}' not in finished:
            jobs.append((f'prep:{k}', 'prep'))
    fast = [(f'fast:{k}', 'fast') for k in keys if f'lbs:{k}' in done and f'fast:{k}' not in finished]
    ntrl = [(f'ntrl:{k}', 'ntrl') for k in keys if f'lbs:{k}' in done and state['info'][k].get('lbs_check', 0) > GATE and f'ntrl:{k}' not in finished]
    nb = [(f'nbench:{k}', 'nbench') for k in keys if f'ntrl:{k}' in done and f'nbench:{k}' not in finished]
    lbs = sorted([k for k in keys if f'prep:{k}' in done and f'lbs:{k}' not in finished and state['info'][k]['predicted'] >= c['min_pred']],
                 key=lambda k: -state['info'][k]['predicted'])
    if len(lbs) >= c['prep_ahead']:
        jobs = []                                          # enough prepared houses waiting for a training: preps cost GPU time too
    lbs = [(f'lbs:{k}', 'lbs') for k in lbs]
    if not running and not fast and not ntrl and not nb and (not lbs or left < c['lbs_minutes']) and (not jobs or left < 10):
        stamp('nothing left to launch before the deadline' if left < 1e8 else 'queue empty')
        break
    if not (HERE / 'PAUSEMORE').exists():
        on = lambda g, kinds: sum(1 for (_, gg, _, kd) in running.values() if gg == g and kd in kinds)
        trains = sum(1 for (_, _, _, kd) in running.values() if kd in ('lbs', 'ntrl'))
        # fast benches: GPU 0 with nothing else of this queue on it
        if fast and not any(gg == '0' for (_, gg, _, _) in running.values()) and fast[0][0] not in running:
            name = fast[0][0]
            add_fast_scene(name.split(':', 1)[1])
            launch(name, 'fast', '0', [PY, FAST_RUN, 'bench', '--scene', name.split(':', 1)[1], '--device', 'cuda', '--gpu', '0', '--threads', '2',
                                       '--variant', VARIANT, '--label', 'lbsfinal'], {})
        gpus = [g for g in c.get('gpus', ['1', '0']) if not (fast and g == '0')]   # a pending fast bench holds new GPU 0 launches
        for name, kind in nb:
            if sum(1 for (_, gg, _, _) in running.values() if gg == 'cpu') >= c['cpu_slots'] or avail_gb() < MIN_RAM:
                break
            if name in running:
                continue
            k = name.split(':', 1)[1]
            launch(name, 'nbench', 'cpu', [OMPL_PY, RUN, 'bench', '--prepared', str(rdir(done[f'prep:{k}'])), '--methods', 'ntrl', '--field',
                                           f'ntrl={rdir(done[f"ntrl:{k}"])}', '--threads', '2'], dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'))
        for g in gpus:
            if avail_gb() < MIN_RAM:
                break
            busy = lambda: g in c['exclusive'] and any(gg == g for (_, gg, _, _) in running.values())     # 5070: prep alone needs ~9 of 12 GB
            for name, kind in [j for j in ntrl if left >= c['ntrl_minutes']] + [j for j in lbs if left >= c['lbs_minutes']]:
                if name in running or busy() or on(g, ('lbs', 'ntrl')) >= c['train_per_gpu'][g] or trains >= c['train_total']:
                    continue
                k = name.split(':', 1)[1]
                extra = LBS if kind == 'lbs' else NTRL
                launch(name, kind, g, [PY, RUN, 'field', '--prepared', str(rdir(done[f'prep:{k}'])), '--label', kind, '--extra', extra],
                       dict(CUDA_VISIBLE_DEVICES=g))
                trains += 1
                time.sleep(20)
            for name, kind in jobs:
                if name in running or busy() or on(g, ('prep',)) >= c['prep_per_gpu'][g] or avail_gb() < MIN_RAM or (g == '0' and SIDE[name[5:-2]] >= BIG):
                    continue
                launch(name, 'prep', g, [PY, RUN, 'prep', '--kind', 'house3', '--name', name.split(':', 1)[1][:-2]], dict(CUDA_VISIBLE_DEVICES=g))
                time.sleep(20)
                break
    time.sleep(30)
