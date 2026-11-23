"""Render report/viewer/arm_paths.html in headless Chrome (software WebGL) and save screenshots; prints the kinematics check and console errors.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/lbs_arm/viewercheck.py <out dir> ['<js to run before the shots>']
"""
import asyncio
import base64
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIEW = HERE / 'report/viewer'


async def main(out, js=None):
    import websockets
    out = Path(out)
    site = out / 'site'
    if site.exists():
        shutil.rmtree(site)
    site.mkdir(parents=True)
    (site / 'index.html').write_text('<!doctype html><html><head><meta charset=utf8><meta name=viewport content="width=device-width,initial-scale=1"><style>body{margin:0}</style></head><body>'
                                     + (VIEW / 'arm_paths.html').read_text() + '</body></html>')
    for f in VIEW.glob('scene_*.json'):
        shutil.copyfile(f, site / f.name)
    port, dport = 8813, 9373
    srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(port), '--directory', str(site)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    chrome = subprocess.Popen(['/opt/google/chrome/chrome', '--headless=new', f'--remote-debugging-port={dport}', f'--user-data-dir={out / "chrome"}', '--no-first-run',
                               '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--hide-scrollbars', 'about:blank'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f'http://127.0.0.1:{dport}/json'))
                break
            except Exception:
                time.sleep(0.2)
        ws_url = [t for t in tabs if t['type'] == 'page'][0]['webSocketDebuggerUrl']
        async with websockets.connect(ws_url, max_size=None) as ws:
            n = 0
            errors = []

            async def call(method, **params):
                nonlocal n
                n += 1
                await ws.send(json.dumps(dict(id=n, method=method, params=params)))
                while True:
                    m = json.loads(await ws.recv())
                    if m.get('id') == n:
                        return m.get('result', {})
                    if m.get('method') in ('Runtime.exceptionThrown', 'Log.entryAdded', 'Runtime.consoleAPICalled'):
                        errors.append(json.dumps(m.get('params'))[:500])
            await call('Runtime.enable')
            await call('Log.enable')
            await call('Page.enable')
            for name, width, height, dark in (('light', 1280, 900, False), ('dark', 1280, 900, True), ('phone', 390, 844, False)):
                await call('Emulation.setDeviceMetricsOverride', width=width, height=height, deviceScaleFactor=1, mobile=width < 500)
                await call('Emulation.setEmulatedMedia', features=[dict(name='prefers-color-scheme', value='dark' if dark else 'light')])
                await call('Page.navigate', url=f'http://127.0.0.1:{port}/index.html')
                await asyncio.sleep(6)
                if js:
                    await call('Runtime.evaluate', expression=js)
                    await asyncio.sleep(1.5)
                r = await call('Runtime.evaluate', expression='JSON.stringify({h: document.documentElement.scrollHeight, w: document.documentElement.scrollWidth, fk: window.__fk, msg: document.getElementById("msg").hidden ? "" : document.getElementById("msg").textContent, count: document.getElementById("count").textContent, cap: document.getElementById("cap").textContent, canvas: !!document.querySelector("#vp canvas")})')
                info = json.loads(r['result']['value'])
                print(name, info)
                shot = await call('Page.captureScreenshot', format='png', captureBeyondViewport=True, clip=dict(x=0, y=0, width=width, height=min(info['h'], 4000), scale=1))
                (out / f'{name}.png').write_bytes(base64.b64decode(shot['data']))
            print('console:', errors[:8])
    finally:
        chrome.terminate()
        srv.terminate()


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
