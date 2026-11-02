"""Builds the Test 4 time-field page from test4_field.py's output: test4_template.html with both scenes' meta.json
inlined (minus the validation values, which timefield_check.js uses) plus each scene's line report -> test4_viewer.html.
The arrays stay in test4_viewer/<scene>/ and are published next to the page as base64 text (artifacts serve no raw binary),
test4_viewer/publish/<scene>__<name>.b64.txt: nodes, edges, lines and the two networks of each scene.

  ~/.venvs/bellman-ntfield/bin/python jeffrey/joint_range/build_test4_viewer.py
"""
import base64, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
SCENES = ['cabinet', 'cabinet_strong']


def main():
    folder = HERE/'test4_viewer'; scenes = {}
    for s in SCENES:
        meta = json.loads((folder/s/'meta.json').read_text()); meta.pop('validation'); [c.pop('validation') for c in meta['checkpoints'].values()]
        lines = json.loads((folder/s/'lines.json').read_text()); meta['lines'] = dict(sub=lines['sub'], report=lines['report']); scenes[s] = meta
    first = scenes[SCENES[0]]
    for s in SCENES[1:]: assert scenes[s]['arm_spheres'] == first['arm_spheres'] and scenes[s]['chain'] == first['chain'], 'arm model differs between scenes'
    page = dict(scenes=scenes, arm_spheres=first['arm_spheres'], chain=first['chain'])
    html = (HERE/'test4_template.html').read_text(); assert html.count('/*META*/null') == 1
    out = HERE/'test4_viewer.html'; out.write_text(html.replace('/*META*/null', json.dumps(page, separators=(',', ':'))))
    pub = folder/'publish'; pub.mkdir(exist_ok=True)
    for f in pub.glob('*.b64.txt'): f.unlink()
    for s in SCENES:
        for name in ['nodes', 'edges', 'lines']+[f'net_{k}' for k in scenes[s]['checkpoints']]:
            (pub/f'{s}__{name}.b64.txt').write_text(base64.b64encode((folder/s/(name+'.bin')).read_bytes()).decode())
    files = sorted(pub.glob('*.b64.txt'))
    print(out, out.stat().st_size//1024, 'KB;', ', '.join(f'{f.name} {f.stat().st_size/2**20:.1f} MB' for f in files), '; total', round(sum(f.stat().st_size for f in files)/2**20, 1), 'MB')


if __name__ == '__main__': main()
