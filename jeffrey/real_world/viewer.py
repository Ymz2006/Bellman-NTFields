"""Viser viewer of the real-world cupboard scenes (scenes/real_world_6dof.yml, scenes/real_world_12dof.yml). Read-only; no planning.

Shows the measured cupboard, floor and pedestals, the arm(s) with the release URDF meshes, joint sliders per arm, the live validity
of the configuration (robot.py: obstacle, self and inter-arm clearance), the collision spheres (red where a sphere is within the
threshold of an obstacle), the reach of the chosen placement (make_scenes.py) and example valid configurations reaching into each zone.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/real_world/viewer.py --port 8910        # then open http://localhost:8910
"""
import argparse
import math
import threading
import time
from pathlib import Path
import numpy as np
import torch
import trimesh
import viser
import yaml
import yourdfpy
from robot import Arms, HOME, JOINT_NAMES, OBS_VALID_M, SELF_VALID_M, INTER_VALID_M

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
URDF = ROOT / 'runtime/assets/runtime/robot/ur5e_robotiq_2f_140.urdf'
SCENES = ('real_world_6dof', 'real_world_12dof')
ZONES = ('bottom', 'middle', 'top')
ARM_COLORS = {'arm': (55, 185, 255), 'left': (55, 185, 255), 'right': (255, 150, 60)}
JOINT_LABELS = ('pan', 'shoulder lift', 'elbow', 'wrist 1', 'wrist 2', 'wrist 3')


def load_urdf():
    return yourdfpy.URDF.load(str(URDF), build_collision_scene_graph=False, load_collision_meshes=False)


def set_cfg(urdf, q):
    cfg = urdf.zero_cfg.copy()
    for name, v in zip(JOINT_NAMES, q):
        cfg[urdf.actuated_joint_names.index(name)] = float(v)
    urdf.update_cfg(cfg)


class Robot:
    """URDF meshes of one arm under a frame at its base pose."""

    def __init__(self, server, name, base):
        self.urdf = load_urdf()
        self.frame = server.scene.add_frame(f'/world/arm_{name}', position=tuple(base[:3]), wxyz=tuple(base[3:7]), show_axes=False)
        self.meshes = {}
        for node in self.urdf.scene.graph.nodes_geometry:
            T, geom = self.urdf.scene.graph.get(node)
            mesh = self.urdf.scene.geometry[geom]
            if not isinstance(mesh, trimesh.Trimesh):
                try:
                    mesh = mesh.to_mesh()
                except Exception:
                    continue
            mesh = mesh.copy()
            if float(mesh.extents.max()) > 10.0:
                mesh.apply_scale(0.001)
            self.meshes[node] = server.scene.add_mesh_trimesh(f'/world/arm_{name}/{node}', mesh=mesh, position=tuple(T[:3, 3]),
                                                              wxyz=tuple(trimesh.transformations.quaternion_from_matrix(T)))

    def update(self, q):
        set_cfg(self.urdf, q)
        for node, h in self.meshes.items():
            T, _ = self.urdf.scene.graph.get(node)
            h.position = tuple(T[:3, 3])
            h.wxyz = tuple(trimesh.transformations.quaternion_from_matrix(T))


def per_sphere_obstacle(model, p):
    """(A, S) distance of every sphere to the nearest cuboid (moving spheres only; the two shoulder spheres get +inf)."""
    delta = (p[:, :, None, :] - model.cc[None, None]).abs() - model.ch[None, None]
    d = (delta.clamp(min=0).norm(dim=-1) + delta.amax(-1).clamp(max=0)).amin(2) - model.radius
    keep = torch.zeros(model.S, dtype=torch.bool)
    keep[model.moving] = True
    return torch.where(keep, d, torch.full_like(d, math.inf))


def describe(doc, name):
    cub = {c['name']: c for c in doc['cuboids']}
    floor = cub['floor']['pose'][2] + cub['floor']['dims'][2] / 2
    front = cub['shelf_bottom']['pose'][0] - cub['shelf_bottom']['dims'][0] / 2
    zs = {z['name']: z for z in doc['interiors']}
    zone_h = zs['bottom']['z'][1] - zs['bottom']['z'][0]
    bases = ', '.join(f"{r['name']} at y = {r['base'][1]:+.2f}" for r in doc['robots'])
    lines = [f'### {name}', '',
             'Cupboard 1.82 m tall × 0.56 m deep × 0.92 m wide (outer). Top/bottom 8 cm, sides 3 cm, back 3 cm, '
             f'two 3 cm dividers (thickness assumed). Three zones {zone_h:.3f} m tall × 0.53 m deep × 0.86 m wide inside.', '',
             f'Base {-floor:.2f} m above the floor on a 0.25 m pedestal; cupboard front face {front:.2f} m in front of the base axis. '
             f'Robots: {bases}.', '',
             '|zone|floor height inside (m)|',
             '|---|---|']
    lines += [f"|{k}|{zs[k]['z'][0] - floor:.3f} – {zs[k]['z'][1] - floor:.3f}|" for k in ZONES]
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8910)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--scene', choices=SCENES, default='real_world_6dof')
    ap.add_argument('--reach', action='store_true', help='start with the reached voxels shown')
    ap.add_argument('--missed', action='store_true', help='start with the unreached voxels shown')
    ap.add_argument('--spheres', action='store_true', help='start with the collision spheres shown')
    ap.add_argument('--example', help='start on an example: a zone (6 DOF) or left|right zones (12 DOF), e.g. "top" or "bottom|top"')
    ap.add_argument('--index', type=int, default=0)
    a = ap.parse_args()
    samples = np.load(HERE / 'report/samples.npz') if (HERE / 'report/samples.npz').exists() else None
    server = viser.ViserServer(host=a.host, port=a.port, label='Real-world cupboard')
    server.scene.set_up_direction('+z')
    server.scene.add_light_ambient('/ambient', intensity=0.7)
    server.scene.add_light_directional('/key', intensity=1.3, wxyz=(0.85, 0.35, 0.35, 0.15))
    lock = threading.RLock()
    state = dict(nodes=[], robots={}, model=None, doc=None, spheres=[], key=None)

    gui_scene = server.gui.add_dropdown('Scene', options=SCENES, initial_value=a.scene)
    info = server.gui.add_markdown('')
    status = server.gui.add_markdown('')
    with server.gui.add_folder('Display'):
        show_cab = server.gui.add_slider('Cupboard opacity', min=0.1, max=1.0, step=0.05, initial_value=0.55)
        show_spheres = server.gui.add_checkbox('Collision spheres', initial_value=a.spheres)
        show_reach = server.gui.add_checkbox('Reached voxels', initial_value=a.reach)
        show_missed = server.gui.add_checkbox('Unreached voxels (red)', initial_value=a.missed)
    with server.gui.add_folder('Examples'):
        ex_zone = server.gui.add_dropdown('Reaching into', options=('…',), initial_value='…')
        ex_index = server.gui.add_slider('Example', min=0, max=15, step=1, initial_value=0)
        ex_show = server.gui.add_button('Show example')
        home_btn = server.gui.add_button('Home (arms straight up)')
    sliders = []
    folders = []
    for arm in range(2):
        f = server.gui.add_folder(f'Arm {arm + 1} joints (rad)')
        folders.append(f)
        with f:
            for j in range(6):
                lim = math.pi
                sliders.append(server.gui.add_slider(JOINT_LABELS[j], min=-lim, max=lim, step=0.01, initial_value=HOME[j]))

    def aim(client):
        mid = state.get('mid', (0.6, 0, 0.3))
        client.camera.position = (mid[0] - 2.9, -1.5, mid[2] + 0.7)
        client.camera.look_at = mid
        client.camera.up = (0, 0, 1)

    def q_now():
        n = 6 * len(state['doc']['robots'])
        return np.array([s.value for s in sliders[:n]], np.float32)

    def evaluate():
        with lock:
            doc, model = state['doc'], state['model']
            q = q_now()
            for i, r in enumerate(doc['robots']):
                state['robots'][r['name']].update(q[6 * i:6 * i + 6])
            P = model.parts(torch.tensor(q)[None])
            ok = bool(P['valid'][0])
            rows = ['|arm|obstacle (m)|self (m)|grasp point (x, y, z)|', '|---|---|---|---|']
            for i, r in enumerate(doc['robots']):
                e = P['ee'][0, i]
                rows.append(f"|{r['name']}|{P['obstacle'][0, i]:+.3f}|{P['self'][0, i]:+.3f}|{e[0]:.2f}, {e[1]:+.2f}, {e[2]:+.2f}|")
            inter = '' if len(doc['robots']) == 1 else f"\n\nArm–arm clearance **{float(P['inter'][0]):+.3f} m** (valid ≥ {INTER_VALID_M})"
            status.content = (f"**{'VALID' if ok else 'IN COLLISION'}** (obstacle ≥ {OBS_VALID_M} m, self ≥ {SELF_VALID_M} m){inter}\n\n" + '\n'.join(rows))
            if show_spheres.value:
                d = per_sphere_obstacle(model, P['spheres'][0])
                p = P['spheres'][0]
                k = 0
                for i, r in enumerate(doc['robots']):
                    for s in range(model.S):
                        h = state['spheres'][k]
                        h.position = tuple(p[i, s].tolist())
                        h.color = (230, 40, 40) if d[i, s] < OBS_VALID_M else ARM_COLORS[r['name']]
                        k += 1

    def build(_=None):
        with lock:
            for h in state['nodes'] + state['spheres'] + [r.frame for r in state['robots'].values()]:
                h.remove()
            state['nodes'], state['spheres'], state['robots'] = [], [], {}
            name = gui_scene.value
            doc = yaml.safe_load((HERE / 'scenes' / f'{name}.yml').read_text())
            state['doc'] = doc
            state['model'] = Arms.from_scene(HERE / 'scenes' / f'{name}.yml')
            for c in doc['cuboids']:
                cab = c['name'].startswith('shelf_')
                h = server.scene.add_box(f"/world/scene/{c['name']}", dimensions=tuple(c['dims']), color=tuple(c.get('color', (200, 80, 50))),
                                         opacity=show_cab.value if cab else (0.5 if c['name'] == 'floor' else 1.0),
                                         position=tuple(c['pose'][:3]), wxyz=tuple(c['pose'][3:]))
                state['nodes'].append(h)
            for z in doc['interiors']:
                state['nodes'].append(server.scene.add_label(f"/world/labels/{z['name']}", text=f"{z['name']} zone",
                                                             position=(z['x'][0], z['y'][1] - 0.05, z['z'][1] - 0.04)))
            for r in doc['robots']:
                state['robots'][r['name']] = Robot(server, r['name'], r['base'])
                if len(doc['robots']) > 1:
                    state['nodes'].append(server.scene.add_label(f"/world/labels/arm_{r['name']}", text=f"{r['name']} arm",
                                                                 position=(r['base'][0] - 0.2, r['base'][1], r['base'][2] - 0.1)))
            model = state['model']
            if show_spheres.value:
                for i, r in enumerate(doc['robots']):
                    for s in range(model.S):
                        state['spheres'].append(server.scene.add_icosphere(f"/world/spheres/{r['name']}_{s}", radius=float(model.radius[s]),
                                                                           color=ARM_COLORS[r['name']], opacity=0.45))
            key = '6' if name.endswith('6dof') else '12'
            state['key'] = key
            if samples is not None:
                if show_reach.value:
                    if key == '6':
                        state['nodes'].append(server.scene.add_point_cloud('/world/reach', points=samples['6/reached'], colors=(60, 200, 90), point_size=0.012))
                    else:
                        L = {tuple(np.round(p, 4)) for p in samples['12/reached_left']}
                        R = {tuple(np.round(p, 4)) for p in samples['12/reached_right']}
                        pts = samples['12/reached']
                        col = np.array([(60, 200, 90) if (tuple(np.round(p, 4)) in L and tuple(np.round(p, 4)) in R) else
                                        ARM_COLORS['left'] if tuple(np.round(p, 4)) in L else ARM_COLORS['right'] for p in pts], np.uint8)
                        state['nodes'].append(server.scene.add_point_cloud('/world/reach', points=pts, colors=col, point_size=0.012))
                if show_missed.value:
                    state['nodes'].append(server.scene.add_point_cloud('/world/missed', points=samples[f'{key}/missed'], colors=(230, 40, 40), point_size=0.012))
                opts = tuple(k.split('/', 2)[2] for k in samples.files if k.startswith(f'{key}/examples/') and len(samples[k]))
                ex_zone.options = opts or ('…',)
                ex_zone.value = ex_zone.options[0]
            for i, f in enumerate(folders):
                f.visible = i < len(doc['robots'])
            info.content = describe(doc, name) + (
                '\n\nReach (green; 12 DOF: blue = left only, orange = right only, green = both): voxels of the zone interiors reached by a valid '
                'configuration, any gripper orientation (make_scenes.py).' if samples is not None else '')
            cub = {c['name']: c for c in doc['cuboids']}
            mid = (cub['shelf_back']['pose'][0] - 0.3, 0.0, cub['shelf_back']['pose'][2])
            state['mid'] = mid
            for client in server.get_clients().values():
                aim(client)
        evaluate()

    def show_example(_=None):
        if samples is None:
            return
        k = f"{state['key']}/examples/{ex_zone.value}"
        if k not in samples.files:
            return
        q = samples[k][min(int(ex_index.value), len(samples[k]) - 1)]
        with lock:
            for s, v in zip(sliders, q):
                s.value = float(np.arctan2(np.sin(v), np.cos(v)))
        evaluate()

    def home(_=None):
        with lock:
            for i, s in enumerate(sliders):
                s.value = HOME[i % 6]
        evaluate()

    gui_scene.on_update(build)
    for g in (show_spheres, show_reach, show_missed):
        g.on_update(build)
    show_cab.on_update(lambda _: [setattr(h, 'opacity', show_cab.value) for h in state['nodes'] if '/scene/shelf_' in h.name])
    for s in sliders:
        s.on_update(lambda _: evaluate())
    ex_show.on_click(show_example)
    ex_index.on_update(show_example)
    home_btn.on_click(home)

    @server.on_client_connect
    def connected(client):
        aim(client)

    build()
    if a.example and a.example in ex_zone.options:
        ex_zone.value = a.example
        ex_index.value = a.index
        show_example()
    print(f'READY http://localhost:{a.port}', flush=True)
    while True:
        time.sleep(10)


if __name__ == '__main__':
    main()
