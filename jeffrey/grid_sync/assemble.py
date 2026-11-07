"""Build README.md from README_src.md: <!--SETUP--> <- README_setup.md, <!--TABLES--> <- report/tables.md (without its
title), <!--MECHANISM--> <- report/mechanism.md, <!--BUMPPHASE--> <- report/bumpphase.md; {{kind:field:n}} tokens are
filled from the registered runs exactly as on the findings page (page.fill).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/grid_sync/assemble.py
"""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('grid_sync_page', HERE / 'page.py')
PG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PG)


def body(path, drop_title=True):
    text = path.read_text()
    lines = text.splitlines()
    if drop_title and lines and lines[0].startswith('# '):
        lines = lines[1:]
    return '\n'.join(lines).strip() + '\n'


def main():
    full = PG.GR.collect()
    for n in PG.SIZES:
        for ctrl in ('octaveperm', 'octavecircle'):
            st = PG.lock_stats(full, n, ctrl)
            if st:
                PG.LOCK[(n, ctrl)] = st
    PG.STRICT = True
    data = PG.table_data()
    mech_path = HERE / 'report/mechanism.json'
    mech = json.loads(mech_path.read_text()) if mech_path.exists() else {}
    src = (HERE / 'README_src.md').read_text()
    src = src.replace('<!--SETUP-->', (HERE / 'README_setup.md').read_text())
    for hook, f in (('<!--TABLES-->', 'report/tables.md'), ('<!--MECHANISM-->', 'report/mechanism.md'), ('<!--BUMPPHASE-->', 'report/bumpphase.md')):
        if hook in src:
            src = src.replace(hook, body(HERE / f).replace('\n## ', '\n#### ') if (HERE / f).exists() else '(not generated yet)\n')
    out = PG.fill(src, data, mech)
    (HERE / 'README.md').write_text(out)
    print('wrote', HERE / 'README.md', len(out))


if __name__ == '__main__':
    main()
