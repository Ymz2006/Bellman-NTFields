"""README.md of the mobile-arm study = README_setup.md + findings (the page's answers and mechanism paragraphs, pagetext9.py, as plain text)
+ registered runs + report/tables.md (report9.py).

    python3 jeffrey/mobile_arm/report9.py && python3 jeffrey/mobile_arm/readme9.py
"""
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pagetext9 as PT  # noqa: E402


def plain(s):
    s = re.sub(r'<(br|/p|/li|/div)>', '\n', s)
    s = re.sub(r'<[^>]+>', '', s)
    return html.unescape(re.sub(r'\n{3,}', '\n\n', s)).strip()


def main():
    R = json.loads((HERE / 'report/runs.json').read_text())
    T = PT.text(R)
    parts = [(HERE / 'README_setup.md').read_text().rstrip()]
    find = ['## Findings', '', plain(T['lede']), '']
    if T.get('status'):
        find += [plain(T['status']), '']
    for q, a in re.findall(r'<div class="q">(.*?)</div><div class="a">(.*?)</div>', T['answers'], re.S):
        find += [f'* **{plain(q)}** {plain(a)}']
    find += ['', '### Mechanism', '', plain(T['contacts_after']), '', plain(T['field_after'])]
    page = (HERE / 'report/artifact.txt').read_text().strip() if (HERE / 'report/artifact.txt').exists() else None
    if page:
        find += ['', f'Findings page: {page} (`report/mobile_arm.html`, built by `page9.py` + `pagetext9.py`; republish the same path).']
    parts.append('\n'.join(find))
    state = json.loads((HERE / 'queue/state.json').read_text())
    runs = ['## Registered runs', '', '| job | run id |', '|---|---|']
    houses = json.loads((HERE / 'queue/houses.json').read_text())
    runs += [f'| prepare {k} | {v.replace("results/registered_", "")} |' for k, v in houses.items()]
    runs += [f'| {j.split(" --resume")[0].replace("|", " ")} | {rid} |' for j, rid in state['done'].items()]
    runs += [f'| superseded: {j} | {rid} |' for j, rid in state.get('superseded', {}).items()]
    runs += [f'| failed: {j} | {rid} |' for j, rid in state.get('failed', {}).items()]
    parts.append('\n'.join(runs))
    if (HERE / 'report/tables.md').exists():
        t = (HERE / 'report/tables.md').read_text().split('\n', 1)[1]
        parts.append('# Tables (report/tables.md)\n' + t.rstrip())
    (HERE / 'README.md').write_text('\n\n'.join(parts) + '\n')
    print('wrote', HERE / 'README.md')


if __name__ == '__main__':
    main()
