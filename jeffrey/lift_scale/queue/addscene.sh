#!/bin/bash
# addscene.sh <key> <run id>: wait for the prepare run to finish, then add it to scenes.json
cd "$(dirname "$0")"
until [ -f ../../../results/registered_$2/results.json ]; do sleep 10; done
~/.venvs/bellman-ntfield/bin/python - "$1" "$2" <<'PY'
import json, sys, os
s = json.load(open('scenes.json'))
s.setdefault(sys.argv[1], {})['prepared'] = f'results/registered_{sys.argv[2]}'
open('scenes.tmp2', 'w').write(json.dumps(s, indent=1)); os.replace('scenes.tmp2', 'scenes.json')
PY
