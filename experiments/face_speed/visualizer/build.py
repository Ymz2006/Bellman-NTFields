"""Inline geometry.json and geometry.js into template.html -> collision_viewer.html (one self-contained page).

  ~/.venvs/bellman-ntfield/bin/python experiments/face_speed/visualizer/export.py
  python3 experiments/face_speed/visualizer/build.py
Open collision_viewer.html in a browser (three.js loads from cdn.jsdelivr.net).
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
page = (HERE/'template.html').read_text()
data = (HERE/'geometry.json').read_text().replace('</', '<\\/')
code = (HERE/'geometry.js').read_text()
assert page.count('/*DATA*/') == 1 and page.count('/*GEOMETRY_JS*/') == 1
out = HERE/'collision_viewer.html'
out.write_text(page.replace('/*GEOMETRY_JS*/', code).replace('/*DATA*/', data))
print(f'wrote {out} ({out.stat().st_size/1e6:.2f} MB)')
