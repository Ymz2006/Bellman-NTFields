"""Register the post-hoc diagnostic of the direction sweep: diag2.py over every finished field of one scene.

    BELLMAN600_DERIVED_IMAGE=bellman600-env:local python3 jeffrey/direction_sweep/posthoc.py --scene cabinet --ext-run <extend run> --gpu 0

Collects the complete, non-smoke train runs (run.py and run2.py) on --ext-run and labels them D<pairs>s<seed> or
B<b>D<d>s<seed>; output diag2.json (the labels report.py reads with --posthoc).
"""
import argparse, hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(Path(__file__).parent))
from run import REGISTRY, RELEASE, TRUTH, EXT, environment
import subprocess


def fields(ext_run):
    out = {}
    for st in sorted(HERE.glob('results/registered_*/status.json')):
        try: s = json.loads(st.read_text())
        except ValueError: continue
        cfg = s.get('settings', {})
        if not s.get('complete') or cfg.get('stage') != 'train' or cfg.get('smoke') or not cfg.get('ext', '').rstrip('/').endswith(ext_run): continue
        label = (f'D{cfg["pairs"]}' if 'pairs' in cfg else f'B{cfg["bellman_pairs"]}D{cfg["direction_pairs"]}') + f's{cfg["seed"]}'
        out[label] = st.parent/'training/checkpoint.pt'
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--scene', choices=['nowall', 'cabinet'], required=True); p.add_argument('--ext-run', required=True)
    p.add_argument('--prepared-run', default='registered_20261019T034213Z-f9c501f9'); p.add_argument('--gpu'); p.add_argument('--prepare-only', action='store_true'); a = p.parse_args()
    gpus = 'all' if a.gpu is None else f'device={a.gpu}'
    prepared = HERE/'assets' if a.scene == 'nowall' else HERE/'results'/a.prepared_run/'prepared'; ext = HERE/'results'/a.ext_run
    truth = HERE/'results'/TRUTH[a.scene]/'truth/truth.npz'; F = fields(a.ext_run); assert F, 'no finished fields'
    inside = lambda path: f'/clean/{Path(path).relative_to(HERE)}'
    inputs = dict(runtime=str(RELEASE), registry=str(REGISTRY), diag2=str(HERE/'jeffrey/direction_sweep/diag2.py'), posthoc=str(HERE/'jeffrey/direction_sweep/posthoc.py'),
                  fit_fields=str(HERE/'jeffrey/segment_dp/fields.py'), launcher=str(HERE/'jeffrey/segment_dp/launch.py'), method=str(HERE/'method'),
                  speed=str(prepared/'speed.pt'), source_q=str(prepared/'bank/source_q.npy'), truth=str(truth),
                  **{'ext_'+f: str(ext/'ext'/f) for f in EXT}, **{'field_'+k: str(v) for k, v in F.items()})
    config = dict(action='direction-sweep-posthoc', scene=a.scene, ext_run=a.ext_run, fields=sorted(F), gpus=gpus, samples=16384, pairs=[1, 2, 4, 8, 16, 32])
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    spec = dict(name=f'Clean Bellman600 direction-sweep {a.scene} posthoc {len(F)} fields {tag} full', kind='pipeline', root=str(HERE), config=config,
                environment=environment(), inputs=inputs,
                command=[sys.executable, str(HERE/'jeffrey/segment_dp/launch.py'), '--output', '{output}', '--gpus', gpus, '--', '/clean/jeffrey/direction_sweep/diag2.py',
                         '--checkpoints', *[inside(v) for v in F.values()], '--labels', *F.keys(), '--prepared', inside(prepared), '--ext', inside(ext),
                         '--truth', inside(truth), '--output', '/outputs/diag2.json'],
                required_outputs=['diag2.json'], validation='files')
    directory = HERE/'specs'; path = directory/f'direction-sweep-posthoc_{a.scene}_{tag}.json'; path.write_text(json.dumps(spec, indent=2)+'\n')
    if a.prepare_only: print(path); return
    check = subprocess.run([sys.executable, str(REGISTRY), 'check', str(path)], capture_output=True, text=True); (directory/(path.stem+'.check.json')).write_text(check.stdout)
    if check.returncode: raise SystemExit(check.stderr or 'Duplicate experiment.')
    subprocess.run([sys.executable, str(REGISTRY), 'run', str(path)], check=True)


if __name__ == '__main__': main()
