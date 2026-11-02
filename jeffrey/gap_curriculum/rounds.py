"""Native RK4 of every round checkpoint of a finished gap-curriculum run (diagnostic only: the evaluated model is
always the last round; this shows how planning changes while the table fills and never selects a round).

no-wall: the frozen 600 queries (release scene, scorer tilt 20); cabinet: the frozen cabinet suites (tilt gate off).
Same native RK4 (experiments/transition_wall/evaluate.py) and unchanged release scorer as the final evaluation.
"""
import sys
sys.path.insert(0, '/clean/method')
import argparse, importlib.util, json, runpy
from pathlib import Path
import numpy as np
import torch
from common import RELEASE, sha, write
_spec = importlib.util.spec_from_file_location('transition_wall_evaluate', '/clean/experiments/transition_wall/evaluate.py')
SHARED = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(SHARED)
sys.path.insert(0, '/workspace')
NORM = RELEASE/'configs/joint_normalization.npz'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scene', type=Path, help='cabinet scene; omitted: the no-wall frozen 600 queries'); p.add_argument('--tests', type=Path)
    p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    if a.scene:
        tests, scene, tilt = a.tests, a.scene, '180'
    else:
        tests, scene, tilt = RELEASE/'tests', RELEASE/'configs/scene.yml', '20'
    manifest = json.loads((tests/'suite_manifest.json').read_text()); suites = manifest['suites'][:1] if a.smoke else manifest['suites']
    count = 2 if a.smoke else 100; a.output.mkdir(parents=True, exist_ok=False)
    report = dict(run=str(a.run), scene=str(scene), tilt_deg=float(tilt), rounds=[])

    def invoke(script, args):
        sys.argv = [str(script), *map(str, args)]
        runpy.run_path(str(script), run_name='__main__')

    for checkpoint in sorted((a.run/'training').glob('round_*.pt')):
        net = SHARED.load_scalar(checkpoint, 'cuda:0'); row = dict(checkpoint=checkpoint.name, sha256=sha(checkpoint), suites={}, converged={})
        for item in suites:
            suite = tests/'suites'/item['file']; paths, reached = SHARED.native(net, np.load(suite, allow_pickle=True), count)
            label = f"rk4_{checkpoint.stem}_{item['task']}"; path = a.output/(label+'.npz')
            np.savez(path, trajectories=paths, query_ids=np.arange(count), converged=reached)
            invoke('/workspace/score_cross_suite.py', ['--suite', suite, '--scene', scene, '--norm_file', NORM, '--trajectories', path, '--labels', label,
                                                       '--json_out', a.output/(label+'.score.json'), '--tilt_deg', tilt])
            result = json.loads((a.output/(label+'.score.json')).read_text())[label]; assert result['n'] == count
            row['suites'][item['task']] = result['success']; row['converged'][item['task']] = int(reached.sum())
        row['success'] = sum(row['suites'].values()); row['n'] = count*len(suites); report['rounds'].append(row)
        print('ROUND_RK4', json.dumps(row), flush=True); write(a.output/'rounds.json', report)
        del net; torch.cuda.empty_cache()
    report['complete'] = True; write(a.output/'rounds.json', report)


if __name__ == '__main__': main()
