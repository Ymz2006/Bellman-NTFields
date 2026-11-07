"""Write page_text/explorer.html (the explorer section page.py inserts at <!--EXPLORER-->) from
explorer_template.html and report/explorer/meta.json (build.py). The data files report/explorer/*.b64.txt are
published next to the page under explorer/."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
meta = json.loads((HERE.parent / 'report/explorer/meta.json').read_text())
html = (HERE / 'explorer_template.html').read_text().replace('/*EXMETA*/null', json.dumps(meta, separators=(',', ':')))
out = HERE.parent / 'page_text/explorer.html'
out.write_text(html)
print('wrote', out, len(html) // 1024, 'KB', 'mazes', sorted(meta, key=int))
