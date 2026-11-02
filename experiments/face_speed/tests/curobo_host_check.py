"""Check the host venv's cuRobo (copied from bellman600-env:local) against the release labels, on one GPU.

Run through the host backend layout (bwrap, see launch.py host_command), e.g. from the repository root:
    BELLMAN600_BACKEND=host ~/.venvs/bellman-ntfield/bin/python launch.py ... is not needed; this script only
    needs /workspace and /opt/ur5e_description, so call it with the bwrap command of launch.host_command.

1. CudaRobotModel.get_state on the cuRobo-made coverage/validation configurations of a cabinet run reproduces
   d_obstacle, d_self and the grasp point (the arrays were made by the pinned image's cuRobo on an RTX 3090).
2. The same for the pure-torch kinematics of experiments/face_speed/robot.py against this cuRobo.
Writes curobo_host_check.json into --output.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, yaml
sys.path.insert(0, '/clean'); sys.path.insert(0, '/workspace')
from experiments.face_speed import robot as R

p = argparse.ArgumentParser(); p.add_argument('--run', default='/clean/results/registered_20261019T034213Z-f9c501f9'); p.add_argument('--scene', default='/clean/experiments/cabinet/scenes/cabinet.yml')
p.add_argument('--output', type=Path, required=True); p.add_argument('--n', type=int, default=50000); a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
import ur5e_plan_eval_multi as E
from curobo.geom.types import WorldConfig
from collision_utils import load_cuboid_geometry, build_self_collision_mask, sphere_cuboid_distance, sphere_sphere_distance
import physical_geometry as geom
dev = 'cuda:0'; started = time.time()
_, rm = E.setup_robot_world('/workspace/ur5e_arm.yml', WorldConfig.from_dict(yaml.safe_load(open(a.scene))), dev)
cc, ch = load_cuboid_geometry(a.scene, dev); rcfg = yaml.safe_load(open('/workspace/ur5e_arm.yml'))
mask, _ = build_self_collision_mask(E.SPHERE_YML_PATH, rcfg['robot_cfg']['kinematics'].get('self_collision_ignore', {}), E.COLLISION_LINK_NAMES, device=dev)
val = Path(a.run)/'sampled/coverage/validation'; q = np.load(val/'configurations_rad.npy', mmap_mode='r'); rng = np.random.default_rng(1)
idx = np.sort(rng.choice(len(q), min(a.n, len(q)), replace=False)); q = np.asarray(q[idx]); do = np.asarray(np.load(val/'d_obstacle.npy', mmap_mode='r')[idx])
ds = np.asarray(np.load(val/'d_self.npy', mmap_mode='r')[idx]); ee = np.asarray(np.load(val/'ee_positions.npy', mmap_mode='r')[idx])[:, :3]
robot = R.Robot().to(dev); rows = []
err = dict(curobo_d_obstacle=0., curobo_d_self=0., curobo_ee=0., torch_vs_curobo_spheres=0., torch_vs_curobo_ee=0.)
with torch.no_grad():
    for s in range(0, len(q), 8192):
        qt = torch.tensor(q[s:s+8192], device=dev); st = rm.get_state(qt); sp = st.link_spheres_tensor
        d_o = geom.sphere_cuboid_distance(sp[:, :, :3], sp[:, :, 3], cc, ch, ignore_sphere_indices=geom.OBSTACLE_IGNORE_SPHERE_INDICES).cpu().numpy()
        d_s = sphere_sphere_distance(sp[:, :, :3], sp[:, :, 3], mask).cpu().numpy(); pos = st.ee_pose.position.cpu().numpy()
        err['curobo_d_obstacle'] = max(err['curobo_d_obstacle'], float(np.abs(d_o-do[s:s+8192]).max())); err['curobo_d_self'] = max(err['curobo_d_self'], float(np.abs(d_s-ds[s:s+8192]).max()))
        err['curobo_ee'] = max(err['curobo_ee'], float(np.abs(pos-ee[s:s+8192]).max()))
        mine = robot.link_spheres(qt); err['torch_vs_curobo_spheres'] = max(err['torch_vs_curobo_spheres'], float((mine-sp).abs().max()))
        err['torch_vs_curobo_ee'] = max(err['torch_vs_curobo_ee'], float((robot.ee_position(qt)-st.ee_pose.position).abs().max()))
report = dict(n=int(len(q)), gpu=torch.cuda.get_device_name(0), torch=torch.__version__, curobo=__import__('curobo').__version__, max_abs_errors=err, seconds=time.time()-started,
              passed=all(v < 2e-5 for v in err.values()))
(a.output/'curobo_host_check.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report, indent=1))
assert report['passed']
