import sys, time, os
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import trainer as TRN
import liftfast as LFT
import lift as LF
import scenes as SN
R = Path('/home/jeffrey/bellman-ntfield/results')
for prep, ck in [('20261102T042454Z-62864e61', '20261117T055249Z-d953394e'), ('20261117T054449Z-0144646b', '20261117T122340Z-11cef54b')]:
    scene, info = SN.load_scene(R / f'registered_{prep}', 'cuda:0')
    net = LF.load_field(R / f'registered_{ck}/training/checkpoint.pt', scene)
    for comp in (False, True):
        t = time.time()
        print(prep, 'compile', comp, LFT.check(net, scene, compile=comp), f'{time.time() - t:.1f}s', flush=True)
