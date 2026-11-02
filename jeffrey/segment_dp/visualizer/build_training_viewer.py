"""Training-data viewer for the cabinet scene: the frozen bank's anchor poses as end-effector points.

The cabinet training data (prepared run registered_20261019T034213Z-f9c501f9, bank/) is 399,581 collision-checked
anchor configurations, each with 32 transitions; the anchors are also the segment-DP pool. This exports a fixed random
subset of --count anchors (seed --seed; the page shows the first N of that order, so a larger N only adds points):
grasp-frame position (forward kinematics of the release URDF), the arm's joint origins (stick figure for a clicked
point), the frozen speed network's speed at the pose, and the roll-forward generation that produced the anchor. Also the
1,200 endpoints of the frozen test suites (starts and goals) for comparison, and the scene's cuboids.
Read-only on results/; writes training_viewer.html next to this script (or --output).

  ~/.venvs/bellman-ntfield/bin/python jeffrey/segment_dp/visualizer/build_training_viewer.py
"""
import argparse, base64, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'method'), str(ROOT/'runtime/method'), str(ROOT/'runtime/vendor/ntrl'), str(ROOT/'runtime/vendor/planner')]
from experiments.face_speed import robot as R

PREPARED = ROOT/'results/registered_20261019T034213Z-f9c501f9/prepared'
SCENE = ROOT/'experiments/cabinet/scenes/cabinet.yml'
TESTS = ROOT/'experiments/cabinet/tests'
ARM = ['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', R.EE_LINK]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--count', type=int, default=5000); p.add_argument('--seed', type=int, default=20261031)
    p.add_argument('--output', type=Path, default=HERE/'training_viewer.html'); a = p.parse_args()
    bank = PREPARED/'bank'; manifest = json.loads((bank/'manifest.json').read_text())
    assert sha(bank/'source_q.npy') == manifest['file_sha256']['source_q.npy']
    source = np.load(bank/'source_q.npy', mmap_mode='r'); generation = np.load(bank/'rollforward_generation.npy', mmap_mode='r')
    rng = np.random.default_rng(a.seed); pick = rng.choice(len(source), a.count, replace=False)
    q = np.asarray(source[pick], dtype=np.float32); gen = np.asarray(generation[pick]).astype(np.int32)
    norm = np.load(R.NORMALIZATION); to_rad = lambda x: x*norm['q_range']+norm['q_center']
    kin = R.Kinematics()

    def frames(rad):
        with torch.no_grad():
            t = kin.transforms(torch.as_tensor(rad, dtype=torch.float64), ARM)
        return np.stack([t[l][:, :3, 3].numpy() for l in ARM], 1)        # [n, links, 3]

    arm = frames(to_rad(q))
    import common; common.RELEASE = ROOT/'runtime'          # the release mount (/repo inside the container)
    from speed import load_speed
    net = load_speed(PREPARED/'speed.pt', 'cpu')
    with torch.no_grad():
        s = torch.cat([net(torch.as_tensor(q[i:i+2048])).reshape(-1) for i in range(0, len(q), 2048)]).numpy()
    tm = json.loads((TESTS/'suite_manifest.json').read_text()); ends = []
    for suite in tm['suites']:
        z = np.load(TESTS/'suites'/suite['file'], allow_pickle=True)
        ends.append(np.concatenate([z['starts_rad'], z['goals_rad']]))
    test = frames(np.concatenate(ends))[:, -1]
    scene = yaml.safe_load(SCENE.read_text())
    boxes = [dict(name=c['name'], center=c['pose'][:3], dims=c['dims']) for c in scene['cuboids']]
    data = dict(count=a.count, total=len(source), seed=a.seed, links=ARM, ee=b64(arm[:, -1]), arm=b64(arm), q_rad=b64(to_rad(q)), index=pick.tolist(), speed=b64(s), generation=gen.tolist(),
                generations=int(generation.max())+1, test=b64(test), test_count=len(test), boxes=boxes, sampling=scene['sampling'],
                anchors=manifest['anchors'], actions_per_anchor=manifest['actions_per_anchor'], step_sizes=manifest['step_sizes'],
                source_sha256=manifest['file_sha256']['source_q.npy'][:12], speed_quantiles=np.quantile(s, [0, .1, .5, .9, 1]).tolist())
    html = (HERE/'training_template.html').read_text().replace('/*DATA*/null', json.dumps(data, separators=(',', ':')))
    a.output.write_text(html); print(a.output, len(html)//1024, 'KB', 'speed quantiles', np.round(data['speed_quantiles'], 3), 'generations', data['generations'])


if __name__ == '__main__': main()
