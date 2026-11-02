"""Control field for the DP chain: the first K models of a trained segment field, the K-th used for every longer gap.

The DP chain starts from the eikonal specialists of segments 1..K and builds every longer segment by the min-plus
transition. This control skips the transition: model K (trained on [e_(K-1), e_K]) answers every gap beyond e_K,
so it measures what the anchored short-range models give by extrapolation alone. Writes field.pt (a segment field
of K models, edges e_0..e_(K-1) and the original last edge) and, unless --no-evaluate, the frozen-benchmark
evaluation of it with evaluate.py (same evaluators and planners as every run) into evaluation/.

--splice DONOR --segments A-B instead: every model of the source, with those of segments A..B (1-based) taken from
the donor segment field (same edges, blend and architecture) -- which segments' models decide a difference in planning.
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/jeffrey/segment_dp')
import argparse, json, subprocess
from pathlib import Path
import torch
from common import sha, write
from fields import save_field


def main():
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True); p.add_argument('--first', type=int, default=2)
    p.add_argument('--splice', type=Path, help='donor segment field'); p.add_argument('--segments', help='A-B: the segments whose models come from the donor')
    p.add_argument('--output', type=Path, required=True); p.add_argument('--scene', type=Path); p.add_argument('--tests', type=Path)
    p.add_argument('--no-evaluate', action='store_true'); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = torch.load(a.source, map_location='cpu', weights_only=False); assert source['kind'] == 'segment_field'
    if a.splice:
        donor = torch.load(a.splice, map_location='cpu', weights_only=False); lo, hi = map(int, a.segments.split('-'))
        assert donor['kind'] == 'segment_field' and donor['edges'] == source['edges'] and donor['blend'] == source['blend'] and 1 <= lo <= hi <= len(source['edges'])-1
        archs = lambda f: f.get('archs') or [f['arch']]*len(f['model_state_dicts']); spliced = list(range(lo-1, hi))
        states = [(donor if i in spliced else source)['model_state_dicts'][i] for i in range(len(source['model_state_dicts']))]
        arch = [archs(donor if i in spliced else source)[i] for i in range(len(states))]; edges = source['edges']
        save_field(a.output/'field.pt', source['arch'], edges, source['blend'], states, archs=arch, derived_from=str(a.source),
                   derived_from_sha256=sha(a.source), spliced_from=str(a.splice), spliced_from_sha256=sha(a.splice), spliced_segments=[lo, hi])
        report = dict(source=str(a.source), source_sha256=sha(a.source), splice=str(a.splice), splice_sha256=sha(a.splice), segments=[lo, hi], edges=edges, field_sha256=sha(a.output/'field.pt'))
    else:
        edges = source['edges'][:a.first]+[source['edges'][-1]]
        save_field(a.output/'field.pt', source['arch'], edges, source['blend'], source['model_state_dicts'][:a.first], derived_from=str(a.source),
                   derived_from_sha256=sha(a.source), first_models=a.first)
        report = dict(source=str(a.source), source_sha256=sha(a.source), first_models=a.first, edges=edges, field_sha256=sha(a.output/'field.pt'))
    if not a.no_evaluate:
        smoke = ['--smoke'] if a.smoke else []
        if a.scene:
            cmd = ['/clean/jeffrey/segment_dp/evaluate.py', 'cabinet', '--method', 'bellman', '--checkpoint', a.output/'field.pt', '--scene', a.scene, '--tests', a.tests, '--output', a.output/'evaluation', *smoke]
        else:
            cmd = ['/clean/jeffrey/segment_dp/evaluate.py', 'nowall', '--checkpoint', a.output/'field.pt', '--output', a.output/'evaluation', *smoke]
        with (a.output/'evaluate.log').open('w') as log:
            subprocess.run([sys.executable, '-u', *map(str, cmd)], stdout=log, stderr=subprocess.STDOUT, check=True)
        ev = json.loads((a.output/'evaluation/evaluation.json').read_text())
        report['totals'] = ev.get('totals') or dict(native_rk4=dict(success=ev['native_success']), matched_mppi=dict(success=sum(ev['mppi_successes']), per_seed=ev['mppi_successes']))
    report['complete'] = True; write(a.output/'derived.json', report)


if __name__ == '__main__': main()
