"""README.md = README_setup.md + README_findings.md (with the artifact link) + the scene, pooled and run tables of report/tables.md.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/lift_scale/assemble.py [artifact url]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else '(not published yet)'
    tables = (HERE / 'report/tables.md').read_text()
    body = tables.split('\n', 1)[1]
    text = (HERE / 'README_setup.md').read_text().rstrip() + '\n\n' + (HERE / 'README_findings.md').read_text().replace('ARTIFACT_URL', url).rstrip() + \
        '\n\n# Tables (report/tables.md)\n' + body
    (HERE / 'README.md').write_text(text)
    print(f'README.md: {len(text.splitlines())} lines')


if __name__ == '__main__':
    main()
