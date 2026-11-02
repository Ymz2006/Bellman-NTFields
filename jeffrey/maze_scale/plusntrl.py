"""Stage 'ntrl' on the plus variant of a maze (plus.py): the maze generator is patched in this process to remove every
wall inside a centred + (width --plus-width, default 6 cells), then ntrl.py's main() runs unchanged. results.json is
re-labelled stage 'plus-ntrl' (maze.variant = 'plus'), so the unmodified maze's runs and queue placeholders never
pick it up.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/maze_scale/plusntrl.py --plus-width 6 [ntrl.py arguments]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plus  # noqa: E402


def main():
    width = 6
    if '--plus-width' in sys.argv:
        k = sys.argv.index('--plus-width')
        width = int(sys.argv[k + 1])
        del sys.argv[k:k + 2]
    plus.install(width)
    import ntrl
    ntrl.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1])
    res = json.loads((out / 'results.json').read_text())
    assert res['maze'].get('variant') == 'plus', 'plus variant not applied'
    res['stage'] = 'plus-ntrl'
    (out / 'results.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
