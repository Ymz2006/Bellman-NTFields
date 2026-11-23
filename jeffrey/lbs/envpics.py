"""Pictures of the nine environments -> report/env/<scene>.png and report/env.json (asked 2026-11-20: "add pictures of the environments").

Each picture is the scene's own geometry seen from above, the whole C-space box [-0.5, 0.5]^2 (x to the right, y up): obstacles filled,
free space transparent (so the page's theme shows through), and the true path of the longest test query (grid solver, truth_paths.npy)
from its start (dot) to its goal (ring). Mazes are drawn from the wall arrays of maze.json (maze_scale's 0.1-cell walls), houses from the
point signed distance of the House scene (rebuilt on the CPU). Allensville gets a second picture with the starts of the queries the
same-seed pair fails on and the spots their plans stop at (report/allensville.json).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/lbs/envpics.py
"""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
OUT = HERE / 'report/env'
PX, SS = 640, 3                                   # output pixels and supersampling
WALL = (124, 134, 150, 255)                       # reads on a light and on a dark page
PATH = (47, 125, 225, 255)
LBS_C, NOS_C = (240, 165, 0, 255), (232, 121, 160, 255)
T_WALL = 0.1


def to_px(xy, W):
    """Normalized box coordinates (x right, y up) -> pixel coordinates."""
    xy = np.asarray(xy, dtype=float)
    return np.stack([(xy[..., 0] + 0.5) * W, (0.5 - xy[..., 1]) * W], -1)


def maze_layer(info, W):
    n, E = info['n'], info['E_cells']
    V, H = np.array(info['V']).astype(bool), np.array(info['H']).astype(bool)
    im = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    h = T_WALL / 2
    boxes = [(i - h, j - h, i + h, j + 1 + h) for i, j in zip(*np.nonzero(V))] + [(i - h, j - h, i + 1 + h, j + h) for i, j in zip(*np.nonzero(H))]
    for x0, y0, x1, y1 in boxes:
        a = to_px([(x0 - n / 2) / E, (y1 - n / 2) / E], W)
        b = to_px([(x1 - n / 2) / E, (y0 - n / 2) / E], W)
        d.rectangle([a[0], a[1], max(b[0], a[0] + SS), max(b[1], a[1] + SS)], fill=WALL)      # at least one output pixel thick
    return im


def house_layer(prepared, W):
    import torch
    import scenes as SL
    scene, _ = SL.load_scene(prepared, 'cpu')
    xs = (np.arange(W) + 0.5) / W - 0.5
    X, Y = np.meshgrid(xs, -xs)                                    # row 0 is y = +0.5
    out = np.zeros((W, W), bool)
    for r in range(0, W, 256):
        xy = torch.tensor(np.stack([X[r:r + 256].ravel(), Y[r:r + 256].ravel()], -1), dtype=scene.dtype)
        out[r:r + 256] = (scene.point_sdf_cells(xy) < 0).reshape(-1, W).numpy()
    rgba = np.zeros((W, W, 4), np.uint8)
    rgba[out] = WALL
    return Image.fromarray(rgba, 'RGBA')


def house3_layer(prepared, info, W):
    """A 3-D house from above: the obstacle colour, as opaque as the share of the slab above each spot that is not free air."""
    tab = np.load(prepared / 'house3_table.npy', mmap_mode='r')
    share = (np.asarray(tab[:, :, 4:-4]) > 0).mean(2)                          # [x, y]; 4 = scenes3.House3.PAD_Z
    a = Image.fromarray(np.round(255 * (1 - share)).astype(np.uint8).T[::-1]).resize((W, W), Image.BILINEAR)
    rgba = np.zeros((W, W, 4), np.uint8)
    rgba[..., :3] = WALL[:3]
    rgba[..., 3] = np.asarray(a)
    return Image.fromarray(rgba, 'RGBA')


def draw_path(im, path, W):
    d = ImageDraw.Draw(im)
    p = to_px(np.asarray(path)[:, :2], W)
    d.line([tuple(v) for v in p], fill=PATH, width=max(2 * SS, W // 260), joint='curve')
    r = W / 90
    d.ellipse([p[0, 0] - r, p[0, 1] - r, p[0, 0] + r, p[0, 1] + r], fill=PATH)
    d.ellipse([p[-1, 0] - 1.3 * r, p[-1, 1] - 1.3 * r, p[-1, 0] + 1.3 * r, p[-1, 1] + 1.3 * r], outline=PATH, width=max(2 * SS, W // 240))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.load(open(HERE / 'queue/scenes.json'))
    W = PX * SS
    meta = {}
    for name, prepared in scenes.items():
        prepared = ROOT / prepared
        info = json.load(open(prepared / 'maze.json'))
        kind = info.get('scene', {}).get('kind')
        house = kind in ('house', 'house3')
        base = house3_layer(prepared, info, W) if kind == 'house3' else house_layer(prepared, W) if house else maze_layer(info, W)
        q = np.load(prepared / 'queries.npz')
        paths = np.load(prepared / 'truth_paths.npy', allow_pickle=True)
        k = int(np.argmax(q['path_cells']))
        im = base.copy()
        draw_path(im, paths[k], W)
        im.resize((PX, PX), Image.LANCZOS).save(OUT / f'{name}.png', optimize=True)
        meta[name] = dict(cells=info['E_cells'], queries=int(len(q['start'])), longest_path_cells=float(q['path_cells'][k]), free_share=info.get('free_share'), house=house)
        if name == 'Allensville' and (HERE / 'report/allensville.json').exists():
            al = json.load(open(HERE / 'report/allensville.json'))
            E = info['E_cells']
            im = base.copy()
            d = ImageDraw.Draw(im)
            rm = al['room']
            a, b = to_px([-0.5, 0.5], W), to_px([rm['x_max'] / E, rm['y_min'] / E], W)
            d.rectangle([a[0] + 2 * SS, a[1] + 2 * SS, b[0], b[1]], outline=PATH, width=2 * SS)
            for run in al['runs']:
                if run['seed'] != 87358 or run['source'] != 'jeffrey/lbs':
                    continue
                col = LBS_C if run['arm'] == 'lbs' else NOS_C
                st = to_px(q['start'][run['fails'], :2], W)
                en = to_px(np.array(run['stall_ends']) / E, W)
                r = W / 130
                for s in st:
                    d.ellipse([s[0] - r, s[1] - r, s[0] + r, s[1] + r], outline=col, width=2 * SS)
                for e in en:
                    d.line([e[0] - 1.4 * r, e[1] - 1.4 * r, e[0] + 1.4 * r, e[1] + 1.4 * r], fill=col, width=2 * SS)
                    d.line([e[0] - 1.4 * r, e[1] + 1.4 * r, e[0] + 1.4 * r, e[1] - 1.4 * r], fill=col, width=2 * SS)
            im.crop((0, 0, int(0.55 * W), int(0.55 * W))).resize((PX, PX), Image.LANCZOS).save(OUT / 'Allensville_room.png', optimize=True)      # the top-left 55 % of the box
        print(name, meta[name], (OUT / f'{name}.png').stat().st_size // 1024, 'KB')
    json.dump(meta, open(HERE / 'report/env.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
