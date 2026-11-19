"""README.md from README_src.md and the page's numbers."""
import json
from pathlib import Path
import page
import page_values as PV

HERE = Path(__file__).resolve().parent
ARTIFACT = 'artifact https://claude.ai/artifact/8brW4DDNpffLm9w3ixLe45'


def main():
    v = PV.values(page.S, page.T10, page.T12, page.SC)
    t = (HERE / 'README_src.md').read_text().replace('ARTIFACT_LINK', ARTIFACT)
    for k, x in v.items():
        t = t.replace('{{' + k + '}}', str(x))
    assert '{{' not in t, t[t.index('{{'):t.index('{{') + 40]
    (HERE / 'README.md').write_text(t)
    print('README', len(t))


if __name__ == '__main__':
    main()
