import json, subprocess, re
from pathlib import Path
Q = Path(__file__).resolve().parent
st = json.loads((Q / 'state.json').read_text())
ps = subprocess.run(['ps', '-ww', '-eo', 'pid,args'], capture_output=True, text=True).stdout.splitlines()
for line in ps:
    m = re.match(r'\s*(\d+) .*registry.py run (\S+)', line)
    if not m:
        continue
    if not Path(m.group(2)).is_file():
        continue
    spec = json.loads(Path(m.group(2)).read_text())
    act, label = spec['config']['action'], spec['config']['label']
    jid = 'euclid_base' if act == 'armplace-euclid' else label.replace('_base', '_base_s87358') if act == 'armplace-field' else None
    print(m.group(1), act, label, '->', jid)
    if jid in st:
        st[jid].update(status='running', pid=int(m.group(1)))
for k in list(st):
    if st[k]['status'] in ('blocked', 'lost') or k.startswith('tests_'):
        del st[k]
(Q / 'state.json').write_text(json.dumps(st, indent=1))
lines = []
for line in (Q / 'jobs.jsonl').read_text().splitlines():
    j = json.loads(line)
    if j['mode'] == 'tests':
        j['args'][-1] = '--per 50 --iterations 3000'
    lines.append(json.dumps(j))
(Q / 'jobs.jsonl').write_text('\n'.join(lines) + '\n')
s = (Q / 'add.py').read_text()
if 'a.iterations' not in s:
    s = s.replace("f'--per {a.per}'", "f'--per {a.per} --iterations {a.iterations}'").replace("    p.add_argument('--per', type=int, default=50)\n", "    p.add_argument('--per', type=int, default=50)\n    p.add_argument('--iterations', type=int, default=3000)\n")
    (Q / 'add.py').write_text(s)
s = (Q / 'sched.py').read_text()
if 'TORCHINDUCTOR' not in s:
    (Q / 'sched.py').write_text(s.replace("OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')", "OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TORCHINDUCTOR_COMPILE_THREADS='1')"))
print({k: v['status'] for k, v in st.items()})
