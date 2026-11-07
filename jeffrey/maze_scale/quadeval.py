"""Plan the quadrant queries (quadrant.py) of one maze variant with registered ntrl-demo fields: uturneval.py unchanged,
with two adaptations in this process only.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/maze_scale/quadeval.py --prepared <quadrant dir>/plus --plus-width 6 --runs <plus-ntrl run>,... --out <dir>
    CUDA_VISIBLE_DEVICES=1 python jeffrey/maze_scale/quadeval.py --prepared <quadrant dir>/orig --runs <ntrl run>,... --out <dir>

--plus-width installs the plus variant (plus.py) so the maze rebuilt by uturneval matches <quadrant dir>/plus/maze.json;
runs of stage 'plus-ntrl' load as ntrl-demo checkpoints. results.json is re-labelled stage 'quadeval'.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plus  # noqa: E402


def main():
    if '--plus-width' in sys.argv:
        k = sys.argv.index('--plus-width')
        plus.install(int(sys.argv[k + 1]))
        del sys.argv[k:k + 2]
    import plan as P
    import uturneval as U

    load = U.load_run_field

    def load_run_field(run_dir, dev):
        d = json.load(open(run_dir / 'results.json'))
        if d['stage'] == 'plus-ntrl':
            net = P.load_field(run_dir / 'training/checkpoint.pt', 'metric', dev)
            return P.field_cost(net), dict(stage='plus-ntrl', seed=d['seed'], n=d['maze']['n'], b_scale=1.0)
        return load(run_dir, dev)

    U.load_run_field = load_run_field
    U.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1])
    res = json.loads((out / 'results.json').read_text())
    res['stage'] = 'quadeval'
    (out / 'results.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
