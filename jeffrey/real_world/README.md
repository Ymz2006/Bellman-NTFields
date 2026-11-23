# Real-world cupboard scenes: `real_world_6dof` and `real_world_12dof`

## Request (2026-11-21)

> I want you to add a new scenario for arm. I have real life measurements for a cupboard. So, this cupboard is a hollow rectangle of
> 182cm tall 56cm depth and 92cm wide. The top and bottom is made of metal with thickness 8cm. The sides have 3cm thickness, and
> there are 2 dividers evenly placed inside to divide this cupboard into 3 zones differing on the height. The back metal is also 3cm
> thick. The thickness is included in the rectangle measurements. You will 1) re-create this in a viser simulation 2) add in the
> regular ur5e robot arm centered in an appropriate position such that the cabinets are reachable (you can do an approximation). So
> this is scenario 1. scenario 2 is 12dof, 2 ur5e robots and they are placed side by side symmetrically with respect to the width
> center of the cabinet. Create these 2 scenarios. You can call them real_world_6dof and real_world_12dof.

This folder holds the scenes and a viewer only. Nothing is trained or evaluated yet, and there are no test suites.

## Files

| file | what |
|---|---|
| `scenes/real_world_6dof.yml`, `scenes/real_world_12dof.yml` | the two scenes (written by `make_scenes.py`; do not edit by hand) |
| `make_scenes.py` | builds the cupboard from the measurements, chooses the base placement by reachability, writes the scenes and `report/` |
| `robot.py` | collision model for one or two arms (`Arms.from_scene(path)`; `parts(q)` gives obstacle, self and inter-arm clearance and validity) |
| `robot_geometry.npz` | the release arm's collision spheres, taken from the lbs_arm export `91c5e157` (`python robot.py --extract`) |
| `viewer.py` | viser viewer |
| `report/placement.json` | full placement grids, chosen placements, flood-fill convergence history, two-arm example rates |
| `report/samples.npz` | reached and unreached voxel centres, example valid configurations for the viewer |

## Cupboard (measured)

Outer box 1.82 m tall × 0.56 m deep × 0.92 m wide, panels included. It stands on the floor and opens towards the robot (−x).

| part | size |
|---|---|
| top, bottom (metal) | 0.08 m, full outer width and depth |
| sides | 0.03 m |
| back | 0.03 m |
| two dividers | **0.03 m: assumed**, the thickness was not given. Same as the sides and back. Change `DIVIDER_T` in `make_scenes.py` if it differs |
| three zones (inside) | 0.5333 m tall × 0.53 m deep × 0.86 m wide each, evenly spaced |

Height of each zone above the floor: bottom 0.080–0.613 m, middle 0.643–1.177 m, top 1.207–1.740 m.

The panels are named `shelf_*`, and the scene files also have `sampling:` and `compartments:` blocks in the cabinet scenes' format, so
`experiments/cabinet/geometry.py:scene_boxes` reads them. `interiors:` holds the full zone interiors. Each `compartments:` box starts
0.04 m inside the opening, the cabinet scenes' convention.

## Robots

The arm is the regular release UR5e with the Robotiq 2F-140 (`runtime/assets/runtime/robot/ur5e_robotiq_2f_140.urdf`). Collisions use
cuRobo's 53 spheres. `robot.py` FK matches the URDF's grasp frame to 1e-8 m on 64 random configurations. The sphere fit matches the
export to 5e-7 m.

Each base sits on a 0.25 × 0.25 m pedestal (an assumed mount; the UR5e base is 0.149 m across) that runs from the floor to the base. The
base faces the cupboard (+x).

Validity rules: obstacle clearance ≥ 0.005 m and self clearance ≥ 0.002 m (the release scorer's thresholds). The two shoulder spheres
never move and are excluded from the obstacle check, as in the release label. For 12 DOF the two arms must also stay ≥ 0.005 m apart,
checked on all sphere pairs, shoulders included.

| scene | frame origin | bases | base height above floor | cupboard front face from the base axis |
|---|---|---|---|---|
| `real_world_6dof` | the robot base | one, at (0, 0, 0) | **0.70 m** | **0.45 m** (back face 1.01 m) |
| `real_world_12dof` | midway between the bases | left y = **+0.30**, right y = **−0.30**, both facing +x | **0.60 m** | **0.25 m** (back face 0.81 m) |

In 12 DOF, q[0:6] is the left arm and q[6:12] the right.

## How the placement was chosen (`make_scenes.py`)

Reach: the zone interiors are cut into 4 cm voxels. A voxel counts as reached when some valid configuration puts the grasp point in it,
with any gripper orientation. For 12 DOF the other arm is held at home (straight up) during this check.

Uniform joint samples alone undercount reach. With 0.26 M samples they gave 7–46 % per zone, which measures where random joint vectors
land, not what the arm can reach. So 1 M uniform samples seed one configuration per voxel. Then a flood fill perturbs every stored
configuration (normal noise 0.03 / 0.1 / 0.3 rad) until a round adds fewer than 0.1 % of the voxels.

Convergence check: each chosen placement was rerun with 4 M seeds and up to 80 rounds. The results agree with the search values to
within 0.3 points:

| placement | reach in the search | reach in the rerun |
|---|---|---|
| 6 DOF | 79.6 / 85.3 / 73.4 % | 79.4 / 85.0 / 73.1 % |
| 12 DOF | 83.3 / 85.3 / 77.6 % | 83.4 / 85.5 / 77.7 % |

Grids searched:
- 6 DOF: base height 0.6–1.1 m in 0.1 m steps × front-face distance 0.15–0.75 m in 0.05 m steps.
- 12 DOF: the same heights × distance in 0.1 m steps × half spacing 0.20, 0.25, 0.30, 0.35, 0.40, 0.46 m.

The rule was the largest mean reach over the three zones. Every stored configuration was re-checked in the scene model (`robot.py`,
scene frame) and all were valid.

**6 DOF**, best front-face distance for each base height:

| base height (m) | best distance (m) | bottom | middle | top | mean |
|---|---|---|---|---|---|
| 0.6 | 0.35 | 81 % | 86 % | 70 % | 79.1 % |
| **0.7** | **0.45** | **80 %** | **85 %** | **73 %** | **79.4 %** |
| 0.8 | 0.45 | 70 % | 86 % | 82 % | 79.0 % |
| 0.9 | 0.30 | 58 % | 85 % | 85 % | 75.8 % |
| 1.0 | 0.30 | 47 % | 84 % | 85 % | 71.9 % |
| 1.1 | 0.40 | 23 % | 84 % | 84 % | 64.0 % |

- Base heights 0.6–0.8 m are within 0.4 points of each other. They trade the bottom zone against the top one.
- At 0.7 m, reach peaks at a distance of 0.45 m: 66 % at 0.15 m, 79 % at 0.45 m, 45 % at 0.75 m. Too close, the arm has to fold to get
  into the cupboard; too far, it cannot get deep.

**12 DOF**, union of both arms, best placement for each spacing:

| half spacing (m) | height / distance (m) | bottom | middle | top | mean |
|---|---|---|---|---|---|
| 0.20 | 0.7 / 0.45 | 82 % | 85 % | 77 % | 81.2 % |
| 0.25 | 0.6 / 0.25 | 82 % | 85 % | 79 % | 81.9 % |
| **0.30** | **0.6 / 0.25** | **83 %** | **85 %** | **78 %** | **82.1 %** |
| 0.35 | 0.6 / 0.25 | 83 % | 85 % | 76 % | 81.4 % |
| 0.40 | 0.6 / 0.25 | 82 % | 85 % | 74 % | 80.5 % |
| 0.46 | 0.6 / 0.15 | 82 % | 85 % | 71 % | 79.1 % |

- The spacing matters little between 0.20 and 0.40 m.
- 0.6 m is the bottom of the height grid. I also tried 0.5 m (distance 0.15–0.45 m, half spacing 0.25–0.35 m) and none of those beat it.
- Each arm alone reaches 71 / 85 / 68 % (bottom / middle / top). Two arms add the side the other arm cannot reach well, so the bottom and
  top zones gain the most.

**What stays unreached** (21 % of the voxels at 6 DOF, 18 % at 12 DOF):
- 66 % of the unreached voxels are within 6 cm of a panel; of the reached ones, 29 % are.
- Their median depth from the opening is 0.46 m, against 0.22 m for the reached ones.
- Cause: with the 2F-140 and its wrist, the grasp point cannot be put flush against the back panel or into the corners while keeping
  5 mm clearance. That is a limit of the geometry, not of the sampling.

**Both arms in zones at once (12 DOF).** I paired a left-arm configuration reaching zone i with a right-arm configuration reaching zone
j and checked them together. Share of pairs that are valid:

| left \ right | bottom | middle | top |
|---|---|---|---|
| bottom | 2 % | 81 % | 93 % |
| middle | 68 % | 36 % | 77 % |
| top | 91 % | 86 % | 48 % |

Both arms in the same zone collide with each other most of the time, the bottom zone worst (2 %). The 12-DOF scene therefore has real
inter-arm coupling.

## Viewer

```bash
~/.venvs/bellman-ntfield/bin/python jeffrey/real_world/viewer.py --port 8910                     # http://localhost:8910
~/.venvs/bellman-ntfield/bin/python jeffrey/real_world/viewer.py --scene real_world_12dof --reach --example 'bottom|top'
```

What it has:
- Scene dropdown.
- Joint sliders for each arm.
- Live validity: obstacle and self clearance per arm, the arm–arm clearance, the grasp point.
- Collision spheres, red where a sphere is within the obstacle threshold.
- Reached voxels (green; 12 DOF: blue = left only, orange = right only, green = both) and unreached voxels (red).
- Example valid configurations per zone, or per left|right zone pair for 12 DOF.
- Home button and a cupboard opacity slider.

It runs on the CPU and needs only `report/` and `scenes/`.

## Not registered

`make_scenes.py` is scene design, like `jeffrey/joint_range/make_strong.py`, not a training or evaluation run, so it is not in the
registry. It is deterministic (seed 20261121) and took 22 min on the 3090.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID ~/.venvs/bellman-ntfield/bin/python make_scenes.py --device cuda:1
```
