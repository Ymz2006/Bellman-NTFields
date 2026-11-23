"""Train one field of the LBS final benchmark: jeffrey/lbs/pipeline.py (from a frozen snapshot, --lbs-src) run unchanged in this process, after
seeds.apply() has set the maze generator to the instance stored in the prepared maze.json (instances 2 and 3 of every maze size).

    python field.py --lbs-src <jeffrey/lbs snapshot> --prepared <prep run> --seed 87358 --lift time --bell 1 --bell-long 1 --lean --out <dir>
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    argv = sys.argv[1:]
    i = argv.index('--lbs-src')
    src = argv[i + 1]
    rest = argv[:i] + argv[i + 2:]
    prepared = Path(rest[rest.index('--prepared') + 1])
    sys.path.insert(0, src)
    import pipeline as PL                                  # imports trainer -> lift -> scenes -> embed_warp/scene (maze_scale on sys.path)
    sys.path.insert(0, str(HERE))
    import seeds as SD
    info = json.load(open(prepared / 'maze.json'))
    what = SD.apply(info)
    print(f'scene instance: {what}', flush=True)
    sys.argv = [str(Path(src) / 'pipeline.py')] + rest
    PL.main()
    out = Path(rest[rest.index('--out') + 1])
    res = json.load(open(out / 'results.json'))
    res['instance'] = what
    res['lbs_src'] = src
    json.dump(res, open(out / 'results.json', 'w'), indent=1)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
