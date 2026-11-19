"""Assemble README.md: README_setup.md (request, setup, files) + findings (the page's text, HTML stripped) + report/tables.md + page link.

    python3 jeffrey/lift_midpoint/report.py && python3 jeffrey/lift_midpoint/page.py && python3 jeffrey/lift_midpoint/readme.py
"""
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pagetext as PT  # noqa: E402

PAGE = 'https://claude.ai/artifact/DdUvv75Ld972r86nabE8kT'


def plain(s):
    s = re.sub(r'<span class="verdict">(.*?)</span>', r'**\1.** ', s)
    s = re.sub(r'<sub>(.*?)</sub>', r'_\1', s)
    s = re.sub(r'<sup>(.*?)</sup>', r'^\1', s)
    s = re.sub(r'</p>\s*<p[^>]*>', '\n\n', s)
    s = re.sub(r'<li>', '* ', s)
    s = re.sub(r'</li>', '\n', s)
    s = re.sub(r'<strong>(.*?)</strong>', r'**\1**', s)
    s = re.sub(r'<code>(.*?)</code>', r'`\1`', s)
    s = re.sub(r'<[^>]+>', '', s)
    return html.unescape(s).strip()


def main():
    runs = json.load(open(HERE / 'report/runs.json'))
    T = PT.text(runs)
    answers = re.findall(r'<div><div class="q">(.*?)</div><div class="a">(.*?)</div></div>', T['answers'], re.S)
    out = [(HERE / 'README_setup.md').read_text().rstrip(), '', '## Findings', '',
           f'Findings page: {PAGE} (`report/lift_midpoint.html`, built by `page.py` from `report/runs.json`). `report/tables.md` (`report.py`) is regenerated '
           'from the registry and is authoritative for every number. Training seed 87358 everywhere; one training of the same recipe moves by 3-6 points between seeds.', '']
    if T.get('status'):
        out += [plain(T['status']), '']
    out += [plain(T['lede']), '']
    for i, (q, a) in enumerate(answers, 1):
        out.append(f'{i}. **{plain(q)}** {plain(a)}')
    out += ['', '### Mechanism', '', plain(T['why_intro']), '', plain(T['why_after']), '', plain(T['field_after']), '', '### Caveats', '', plain(T['caveats']), '',
            '## Tables', '', (HERE / 'report/tables.md').read_text().split('\n', 1)[1].strip(), '']
    (HERE / 'README.md').write_text('\n'.join(out) + '\n')
    print('wrote README.md', sum(len(x) for x in out), 'chars')


if __name__ == '__main__':
    main()
