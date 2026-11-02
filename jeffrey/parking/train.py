"""Train ntrl-demo (baselines/ntrl-demo, unchanged on disk) on a parking data set: its Model.train() with dim = 3.

--variant metric      models/metric, what ntrl-demo trains its 2-D maze (train_maze.py) and 3-D Gibson scenes with
                      (B ~ N(0, 1) truncated at 2, speed stretch alpha 1.025, TD step 0.03 / s, metric 0.2 x sum)
--variant metric_arm  models/metric_arm, its UR5 package (B ~ 0.2 N(0, 1), alpha 1.0, TD step 0.02 / s, 0.1 x sum)
Everything else is the package's own loop: AdamW lr 5e-4 weight decay 0.5, 5 minibatches of 2,000 per epoch,
epoch rollback, checkpoints every 100 epochs; --epochs overrides 'Number of Epochs' (default 5,000) and the final
epoch is the model. Shims as experiments/ntrl_cabinet/train.py: pickle5 -> pickle, igl -> an OBJ writer for the
progress plots. The progress-plot source is the bay-A parked pose.
"""
import argparse
import hashlib
import json
import os
import pickle
import random
import shutil
import sys
import time
import types
import warnings
from pathlib import Path
import numpy as np
import torch

NTRL = Path(__file__).resolve().parents[2] / 'baselines/ntrl-demo'


def write_triangle_mesh(path, v, f):
    with open(path, 'w') as out:
        out.writelines(f'v {a} {b} {c}\n' for a, b, c in np.asarray(v, float))
        out.writelines(f'f {a+1} {b+1} {c+1}\n' for a, b, c in np.asarray(f, int))
    return True


def provenance(variant):
    files = [f'models/{variant}/model_train_metric.py', f'models/{variant}/model_network_metric.py', f'models/{variant}/model_function_metric.py', 'models/data_mlp.py']
    head = (NTRL / '.git/HEAD').read_text().strip()
    ref = NTRL / '.git' / head[5:]
    commit = ref.read_text().strip() if head.startswith('ref: ') and ref.exists() else head
    return dict(path=str(NTRL), git_commit=commit, files={f: hashlib.sha256((NTRL / f).read_bytes()).hexdigest() for f in files})


def train(data, out, variant='metric', epochs=5000, seed=87358, source=(0.0, 0.0, 0.0), keep_every=1000):
    """Returns the final checkpoint path (out/checkpoint.pt). data: folder with sampled_points/speed/normal.npy."""
    data, out = Path(data).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    warnings.filterwarnings('ignore', message="input's size at dim=0 does not match num_features")
    warnings.filterwarnings('ignore', message='You are using `torch.load` with `weights_only=False`')
    sys.modules['pickle5'] = pickle
    igl = types.ModuleType('igl')
    igl.write_triangle_mesh = write_triangle_mesh
    sys.modules['igl'] = igl
    cwd = os.getcwd()
    os.chdir(NTRL)
    if str(NTRL) not in sys.path:
        sys.path.insert(0, str(NTRL))
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if variant == 'metric':
            from models.metric import model_train_metric as md
        else:
            from models.metric_arm import model_train_metric as md
        assert Path(md.__file__).resolve().parent == NTRL / 'models' / variant, 'ntrl-demo models package shadowed'
        model = md.Model(str(out / 'model'), str(data) + '/', 3, list(map(float, source)), device='cuda:0')
        model.Params['Training']['Number of Epochs'] = epochs
        (out / 'model').mkdir()
        json.dump(dict(params=model.Params, data=str(data), epochs=epochs, seed=seed, variant=variant, source=list(map(float, source)),
                       ntrl_demo=provenance(variant)), open(out / 'config.json', 'w'), indent=2, default=str)
        model.train()
    finally:
        os.chdir(cwd)
    folder = Path(model.folder)
    final = sorted(folder.glob(f'Model_Epoch_{str(epochs).zfill(5)}_*.pt'))
    assert len(final) == 1, f'no final-epoch checkpoint in {folder}'
    shutil.copyfile(final[0], out / 'checkpoint.pt')
    payload = torch.load(out / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert payload['epoch'] == epochs
    for c in folder.glob('Model_Epoch_*.pt'):
        e = int(c.name.split('_')[2])
        if e % keep_every and e != epochs:
            c.unlink()
    for c in folder.glob('*.obj'):
        c.unlink()
    losses = [float(x) for x in payload['train_loss']]
    json.dump(dict(complete=True, epochs=epochs, variant=variant, seed=seed, final_checkpoint=str(final[0].relative_to(out)),
                   final_loss=losses[-1], elapsed_seconds=time.time() - started), open(out / 'progress.json', 'w'), indent=2)
    np.save(out / 'train_loss.npy', np.array(losses, dtype=np.float32))
    return out / 'checkpoint.pt'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--variant', choices=['metric', 'metric_arm'], default='metric')
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--seed', type=int, default=87358)
    a = p.parse_args()
    train(a.data, a.out, a.variant, a.epochs, a.seed)


if __name__ == '__main__':
    main()
