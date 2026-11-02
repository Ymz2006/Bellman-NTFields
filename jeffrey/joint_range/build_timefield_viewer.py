"""Builds the time-field page from the output of timefield.py and timefield_lines.py: timefield_template.html with
meta.json inlined (minus the validation values, which timefield_check.js uses) and the line report of lines.json
-> timefield_viewer.html. The arrays the page loads stay in timefield_viewer/ and are published next to the page as
base64 text, publish/<name>.b64.txt (artifacts serve no raw binary): nodes, edges, lines and one network per model.
(The slice version of the page, timefield_slices_template.html, also used the emb_<model>.bin embedding tables.)

  ~/.venvs/bellman-ntfield/bin/python jeffrey/joint_range/build_timefield_viewer.py
"""
import base64, json
from pathlib import Path
HERE = Path(__file__).resolve().parent


def main():
    folder = HERE/'timefield_viewer'; meta = json.loads((folder/'meta.json').read_text())
    meta.pop('validation'); [c.pop('validation') for c in meta['checkpoints'].values()]
    lines = json.loads((folder/'lines.json').read_text()); meta['lines'] = dict(sub=lines['sub'], report=lines['report'])
    html = (HERE/'timefield_template.html').read_text(); assert html.count('/*META*/null') == 1
    out = HERE/'timefield_viewer.html'; out.write_text(html.replace('/*META*/null', json.dumps(meta, separators=(',', ':'))))
    pub = folder/'publish'; pub.mkdir(exist_ok=True)
    for f in pub.glob('*.b64.txt'): f.unlink()                              # publish only what the page loads
    for name in ['nodes', 'edges', 'lines']+[f'net_{k}' for k in meta['checkpoints']]:
        (pub/(name+'.b64.txt')).write_text(base64.b64encode((folder/(name+'.bin')).read_bytes()).decode())
    files = sorted(pub.glob('*.b64.txt'))
    print(out, out.stat().st_size//1024, 'KB;', ', '.join(f'{f.name} {f.stat().st_size/2**20:.1f} MB' for f in files), '; total', round(sum(f.stat().st_size for f in files)/2**20, 1), 'MB')


if __name__ == '__main__': main()
