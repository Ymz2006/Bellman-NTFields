#!/bin/bash
# Rebuild everything the page needs from the registered runs: tables, thickness forensics per scene, figures, page, layout check.
# usage: finalize.sh [gpu]   (run from anywhere)
cd "$(dirname "$0")"
PY=~/.venvs/bellman-ntfield/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${1:-0}
$PY report.py
$PY - <<'PYEOF' > report/.thick_jobs.txt
import json
d = json.load(open('report/runs.json'))
done = {j['run'] for j in json.load(open('report/thick.json'))} if __import__('os').path.exists('report/thick.json') else set()
alias = {'liftpush_oct8': 'liftpush_oct'}
for s in ('plus22', 'plus30', 'plus40', 'apt32', 'Allensville', 'apt64', 'Lakeville', 'Klickitat'):
    ids = ['../../results/registered_' + r['id'] for r in d['runs'] if r['scene'] == s and alias.get(r['label'], r['label']) in ('ntrl', 'liftpush', 'liftpush_oct') and r['seed'] == 87358 and r['id'] not in done]
    if ids:
        print(' '.join(ids))
PYEOF
while read line; do $PY thick.py --out report/thick.json $line 2>&1 | grep "detour pairs"; done < report/.thick_jobs.txt
$PY figs.py 2>&1 | grep -v "Size\|warn\|torch.Size" | tail -n 3
$PY page.py | tail -n 1
