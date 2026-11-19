"""Render report/lift_midpoint.html in headless Chrome and save screenshots (light, dark, phone width) to a folder, with console errors.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/lift_midpoint/pagecheck.py <out dir>
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
REPORT = HERE / 'report'


async def main(out):
    import websockets
    out = Path(out)
    site = out / 'site'
    if site.exists():
        shutil.rmtree(site)
    (site / 'img').mkdir(parents=True)
    page = (REPORT / 'lift_midpoint.html').read_text()
    (site / 'index.html').write_text('<!doctype html><html><head><meta charset=utf8><meta name=viewport content="width=device-width,initial-scale=1"></head><body>' + page + '</body></html>')
    for f in REPORT.glob('*.png'):
        shutil.copyfile(f, site / 'img' / f.name)
    port, dport = 8791, 9351
    srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(port), '--directory', str(site)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    chrome = subprocess.Popen(['/opt/google/chrome/chrome', '--headless=new', f'--remote-debugging-port={dport}', f'--user-data-dir={out / "chrome"}', '--no-first-run', '--disable-gpu',
                               '--hide-scrollbars', 'about:blank'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
                        errors.append(json.dumps(m.get('params'))[:400])
            await call('Runtime.enable')
            await call('Log.enable')
            await call('Page.enable')
            for name, width, dark in (('light', 1200, False), ('dark', 1200, True), ('phone', 390, False)):
                await call('Emulation.setDeviceMetricsOverride', width=width, height=900, deviceScaleFactor=1, mobile=width < 500)
                await call('Emulation.setEmulatedMedia', features=[dict(name='prefers-color-scheme', value='dark' if dark else 'light')])
                await call('Page.navigate', url=f'http://127.0.0.1:{port}/index.html')
                await asyncio.sleep(2.5)
                r = await call('Runtime.evaluate', expression='JSON.stringify({h: document.documentElement.scrollHeight, w: document.documentElement.scrollWidth, charts: document.querySelectorAll(".chart svg").length, empty: Array.from(document.querySelectorAll(".chart")).filter(c => !c.querySelector("svg")).map(c => c.dataset.chart), broken: Array.from(document.images).filter(i => !i.naturalWidth).map(i => i.getAttribute("src"))})')
                info = json.loads(r['result']['value'])
                print(name, info)
                shot = await call('Page.captureScreenshot', format='png', captureBeyondViewport=True, clip=dict(x=0, y=0, width=width, height=min(info['h'], 16000), scale=1))
                (out / f'{name}.png').write_bytes(base64.b64decode(shot['data']))
                if name in ('light', 'dark') and len(sys.argv) > 2:
                    for key in sys.argv[2:]:
                        r = await call('Runtime.evaluate', expression=f'(function(){{var e=document.querySelector(\'[data-chart="{key}"]\'); if(!e) return -1; var b=e.getBoundingClientRect(); return b.top + window.scrollY;}})()')
                        y = r['result']['value']
                        if y < 0:
                            continue
                        shot = await call('Page.captureScreenshot', format='png', captureBeyondViewport=True, clip=dict(x=0, y=max(y - 20, 0), width=width, height=900, scale=0.75))
                        (out / f'chart_{key}{"" if name == "light" else "_dark"}.png').write_bytes(base64.b64decode(shot['data']))
            print('console:', errors[:10])
    finally:
        chrome.terminate()
        srv.terminate()


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1]))
