"""cross_warp/evalq.py on fields with a fitted warp (netwarp.install), unchanged otherwise."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = next(q for q in HERE.parents if (q / 'baselines/ntrl-demo').is_dir())
sys.path[:0] = [str(HERE), str(ROOT / 'jeffrey/cross_warp'), str(ROOT / 'jeffrey/maze_scale')]
import warp as W  # noqa: E402
import plan as P  # noqa: E402
import netwarp as NW  # noqa: E402
import evalq  # noqa: E402

NW.install(W, P)
evalq.main()
