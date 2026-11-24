import json
from pathlib import Path
Q = Path(__file__).resolve().parent
p = Q / 'sched.py'; s = p.read_text()
if 'LIMIT_MIB' not in s:
    s = s.replace("NEED_MIB = dict(tests=9000, prepare=8000, field=3500, evalfield=7000, demo=6000)       # free VRAM a job must find on its GPU",
                  "NEED_MIB = dict(tests=8000, prepare=6000, field=4000, evalfield=5500, demo=6000, euclid=1500, reach=3000)       # VRAM reserved per job (its peak), and free VRAM it must find\nLIMIT_MIB = 45000")
    s = s.replace("""        free = free_mib()
        last = getattr(main, 'last', {})""", """        free = free_mib()
        last = getattr(main, 'last', {})
        reserved = {g: 0 for g in caps}
        for jid in procs:
            reserved[state[jid]['gpu']] += NEED_MIB.get(state[jid].get('mode'), 3000)""")
    s = s.replace("            if free.get(gpu, 0) < need or time.time() - last.get(gpu, 0) < 20:", "            if free.get(gpu, 0) < need or reserved[gpu] + need > LIMIT_MIB or time.time() - last.get(gpu, 0) < 5:")
    s = s.replace("            free[gpu] -= need", "            free[gpu] -= need\n            reserved[gpu] += need")
    p.write_text(s)
st = json.loads((Q / 'state.json').read_text())
dead_pl = ['p_d105_s30_h60', 'p_d105_s20_h80']
out = []
for l in (Q / 'jobs.jsonl').read_text().splitlines():
    j = json.loads(l)
    if any(pl in j['id'] for pl in dead_pl) and not j['id'].startswith('tests_'):
        st.pop(j['id'], None)
        continue
    if j['id'] in ('plan_lift_p_d055_s46_h60_s87359', 'plan_lift_base_s87359'):
        st.pop(j['id'], None)
        continue
    if j['id'].startswith('budget_') and '--repeat-reason' not in j['args']:
        j['args'] += ['--repeat-reason', 'the first attempt died from CUDA out of memory (the card was full)']
        st.pop(j['id'], None)
    out.append(json.dumps(j))
(Q / 'jobs.jsonl').write_text('\n'.join(out) + '\n')
for k in [k for k, v in st.items() if v['status'] == 'blocked']:
    del st[k]
(Q / 'state.json').write_text(json.dumps(st, indent=1))
print({s_: sum(1 for v in st.values() if v['status'] == s_) for s_ in ('done', 'running', 'failed')})
