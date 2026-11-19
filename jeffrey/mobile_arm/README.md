# Mobile arm: the travel-time midpoint loss on a 9-DOF wheeled robot with an arm, in Gibson houses

Study of 2026-11-17. Request:

> I have a test for the midpoint travel time loss. I [want] you to introduce a new kind of test, this one is a wheeled robot and it has an
> arm on top of it, so the total is 12 degrees of freedom because the robot has 6 and the arm has 6. Run this in the gibson environment.
> Make the robot small enough to fit through everything, I remember there are rectangles that can do this so find those. Then, attach a
> robot arm on top of it. Set up this 12 degree of freedom task and train the baseline and also the baseline + midpoint loss on it.

A wheeled base on the floor has 3 degrees of freedom (x, y, heading), not 6, so I asked; the answer was **9 DOF, planar base**: the
rectangle robot of the Gibson-house studies (x, y, heading) plus a 6-joint arm.

## Robot

* **Base**: the rectangle of jeffrey/maze_scale, jeffrey/lift_scale and jeffrey/lift_midpoint, 0.5 x 0.2 cells = 12.5 x 5 cm at the
  houses' 0.25 m per cell (the rectangles that fit everywhere in Allensville, Lakeville and Klickitat), as a box 0.1 cells high on the floor.
* **Arm**: the repository's UR5e (runtime/assets/runtime/robot/ur5e.urdf joint origins; the 28 collision spheres of its six links from
  runtime/vendor/planner/spheres_ur5e_arm.yml, gripper left off) at 1/5 size (0.8 cells per UR5e metre: reach up to 0.79 cells, 20 cm),
  mounted at the centre of the base's top face, base_link x along the heading. `robot.py` FK equals cuRobo's on 2,048 random joint vectors
  to 4e-7 m. Joint box: every joint [-pi, pi] except the shoulder lift [-pi, 0] (upper arm above the shoulder's horizontal plane).
* **Collisions** (`robot.py`, `scene9.py`; one function for labels, planner, scorer): the floor plan's walls are vertical prisms (the
  plan is the house studies' cut of the mesh at 0.4-1.6 m): the base by the house studies' 9 x 5 lattice, every arm sphere by the plan's
  signed distance at its centre minus its radius; internally the arm against the floor, against the base box (the shoulder link, mounted
  on it, exempt) and against itself on cuRobo's UR5e self-collision pairs (ur5e_arm.yml self_collision_ignore).
* **Tucked pose** TUCK = (pi/2, -pi/2, 0, -pi/2, 0, 0): the arm straight up with the upper arm's offset along the heading. Every sphere's
  footprint is inside the base rectangle (0.02 cells to spare at the shoulder), so the tucked robot has exactly the bare rectangle's walls:
  on 400,000 random base poses its speed equals the rectangle's everywhere and the collision sign differs on 1 pose. **The robot fits
  through everything the rectangle fits through when it tucks**; reaching out, it needs room.

## Configuration space and speed

q = (x, y, phi, psi_1 .. psi_6), normalized as in the house studies: x, y in [-0.5, 0.5] (E cells across), phi = 0.25 theta / E
(theta in [-pi/2, pi/2]), psi_j = rho_j q_j / E. rho_j = the largest lever arm of joint j (distance from its axis to the farthest sphere
centre it moves, over a million random joint vectors): 0.76, 0.76, 0.42, 0.10, 0.06 and (wrist 3, whose only sphere sits on its axis)
the floor 0.05 cells, so turning a joint by dq moves no sphere centre farther than rho_j dq cells: every joint is charged like the base.

Speed = ntrl-demo's label clip(d / 0.175, 0.1, 1) (stretched 1.025 s^2 (2 - s)^2 - 0.025 in training) of the clearance
d = min(walls, 3.5 x internal). Internal distances count with a 3.5x smaller margin (0.05 cells) because the arm's own parts sit closer
than 0.175 cells in every pose (the upper arm's shoulder end is 0.067 cells above the base box): with the wall margin the robot would be
slow everywhere. 77 % of the joint box is internally free; 28 % of the 9-D box is collision-free in Allensville.

## Data, queries, controls (`prepare.py`, stage `prepare`)

* ntrl-demo's point recipe in the 9-D box (`data9.py`, as maze_scale/data.py: applied in the unit cube the box maps to, r_max = sqrt(9)):
  500,000 pairs + 50,000 held out. The label of x0 is set by the base against a wall for 34 %, the arm against a wall for 23 % and the
  arm's internal constraints for 43 % (Allensville).
* Queries: start = a random configuration with clearance >= 0.1 cells; goal = the base pose of one of the house run's 48 SE(2) truth
  goals with its own random arm (clearance >= 0.1). Bins by D = T*_SE2(start base, goal base) / pitch, the bare rectangle's true travel
  distance in cells (= the tucked robot's), 40 per bin as in the house studies: Allensville 240 (0.5-64 cells), Lakeville and Klickitat 280
  (0.5-128). Every query is certified by a reference path: tuck with the base standing (straight in joint space; else first shoulder lift,
  elbow and wrists, then pan and wrist 3), drive the SE(2) truth path tucked, untuck at the goal; dense check (<= 0.005 cells per
  coordinate). Its travel time T_ref is an upper bound on the 9-D optimum; T*_SE2 is a lower bound (the 9-D clearance never exceeds the
  rectangle's and the 9-D metric never charges less), so T*_SE2 <= T* <= T_ref.
* Controls: the straight segment (scored like the planner); no oracle field exists in 9-D.

## Fields (`pipeline9.py`, stage `field`)

* **ntrl (baseline)**: ntrl-demo's models/metric with dim 9, 5,000 epochs, training seeds 87358 / 1 / 2. `trainer9.py` is its
  Model.train() call for call (check_ntrl(): the first epochs' losses equal the package's own loop to the last printed digit).
* **mid**: the baseline run's epoch-2,000 checkpoint (network, AdamW state, B) + 3,000 epochs with jeffrey/midpoint_dp's MAIN term (safe
  MPPI traversals of 8,192 random pairs every 100 epochs, cut at the first contact or stall, travel-time midpoint trees down to 1 cell of
  travel time with the exact speed, min leaves, tree + sub-path nodes, weight 1 ramped over 500 epochs, |log T - log y|).
* **ctl (control)**: the same resumed loop without the term. Paired design because two runs of one seed differ by up to 12 points on
  the two GPUs (epoch 1,000, Allensville: 180 vs 210 of 240).
* Planning: ntrl-demo's MPPI in 9-D (`plan9.py`, maze_scale's planner and step scaling, angles clamped to the box), planning seeds 0 / 1
  / 2, budget 3 x the reference path's C-space length / step; success = arrived and every configuration at <= 0.005 cells per coordinate
  has clearance >= 0. Failures split by arrival (the planner appends the goal): arrived through something (part touched first) or did not
  arrive. Field: T(start, goal) against T*_SE2 and T_ref; speed fit on held-out pairs by the constraint that sets the label.

`run.py` registers both stages (sources frozen in `snap/<hash>/`); `queue/runner.py` runs `queue/jobs.txt` (one run per GPU);
`report9.py` -> `report/runs.json`, `report/tables.md`; `figs9.py` -> `report/robot.png`, `report/house_<name>.png`.

## Findings

The Gibson-house rectangle robot now carries a six-joint arm (a UR5e at one fifth of its size): 9 degrees of freedom, planned through three scanned houses with an ntrl-demo travel-time field. Adding the recursive travel-time midpoint loss at epoch 2,000 changes the share of solved queries by +4.1 points in Allensville, +12.7 points in Lakeville, +13.9 points in Klickitat against the paired control (the same resumed training without the term).

* **Does the midpoint loss help on the 9-DOF task?** Allensville: midpoint 89.4 %, control 85.4 %, baseline 83.7 % (3 training seeds); Lakeville: midpoint 63.4 %, control 50.8 %, baseline 50.1 % (3 training seeds); Klickitat: midpoint 71.3 %, control 57.4 %, baseline 55.7 % (3 training seeds). It helps in every training seed in Lakeville (+13.3, +14.6, +10.0 points against the control) and Klickitat (+13.8, +11.8, +16.1 points against the control), the houses where the baseline field sees through walls; in Allensville, where the baseline already plans well, it helps in 2 of 3 seeds (+7.1, +7.8, -2.6; +4.1 pooled).
* **What does it fix?** Stalls: runs that never arrive, control → midpoint, Allensville 6.7 → 0.6 %; Lakeville 21.5 → 11.4 %; Klickitat 16.9 → 8.7 %. Many baseline stalls are short queries where only the arm has to move (local minima of the field in joint space). And the short far field: queries whose learned T(start, goal) lies below the base's physical lower bound, Allensville 29 → 2 %; Lakeville 46 → 7 %; Klickitat 39 → 7 %.
* **What does it leave?** Runs that drive through walls. Allensville: 8.0 → 10.0 % of runs, median penetration 0.09 → 0.09 cells; Lakeville: 27.8 → 25.2 % of runs, median penetration 0.37 → 0.36 cells; Klickitat: 25.7 → 20.0 % of runs, median penetration 0.41 → 0.19 cells. In Lakeville they stay deep crossings of walls the field does not see. On the longest queries (32 cells and more) the change against the control is Allensville -2 points; Lakeville +5 points; Klickitat +15 points. In Allensville the contacts stay shallow clips of corners (seed 87358 clipped one corner of the central block on most of its failed long queries).
* **Why some walls and not others?** The loss learns only from the drives the current field can make. A drive that heads into an unseen wall is cut at the wall or stalls in front of it, so no training pair ever holds the detour around that wall. Training drives that reach their goal, first → last refresh: Allensville 82 → 99 %; Lakeville 44 → 70 %; Klickitat 55 → 77 %. Where the drives learn to get around a wall, the field learns it (Klickitat's contacts get fewer and shallower); where a third of the drives still stall, deep crossings stay (Lakeville). This is the same limit the lift + midpoint study found in 2-D (on-policy data).
* **Does the arm matter?** Yes. Of the runs that arrive through something, the arm touches first in Allensville 39 %; Lakeville 30 %; Klickitat 39 % (baseline, planning seed 0), every time with the base itself clear of the wall. The baseline field does not fold the arm near walls: the share of tucked waypoints within half a cell of a wall minus the share in the open is -0.03, -0.02, -0.01 (Allensville, Lakeville, Klickitat). With the midpoint loss it is +0.13, +0.02, -0.02 (control +0.06, -0.02, -0.05): the arm learns to fold near walls only where the drives succeed almost always (Allensville).
* **How much harder is it than the base alone?** The baseline on the bare SE(2) rectangle (the lift-scale study's ntrl-demo runs, same houses and bins, their own queries) against the 9-DOF baseline: Allensville 90.3 → 83.7 %; Lakeville 69.1 → 50.1 %; Klickitat 71.2 → 55.7 %. The arm costs most in the two large houses, where far more of the baseline's runs drive through walls.
* **What does it cost?** A baseline field trains in 7–8 minutes. The midpoint run drives 8,192 pairs every 100 epochs and that dominates its cost (RTX 3090: 178–269 s per refresh, 1.6–2.4 h per run; RTX 5070: 180–277 s per refresh, 1.6–2.4 h per run).

### Mechanism

How deep the first contact goes separates two failures. In the large houses the baseline's contacts are deep: the path crosses a wall the field does not see. In Allensville they are shallow clips of corners. Allensville: control 57 contacts (median depth 0.09 cells, 31 under 0.1 cells), midpoint 72 (0.09 cells, 42 shallow); Lakeville: control 232 contacts (median depth 0.37 cells, 24 under 0.1 cells), midpoint 209 (0.36 cells, 33 shallow); Klickitat: control 218 contacts (median depth 0.41 cells, 16 under 0.1 cells), midpoint 168 (0.19 cells, 33 shallow).

The baseline and the control are short in the far field: from 8 cells on, about half or more of their travel times sit below the lower bound in every house (baseline, 32 cells and beyond: Allensville 43 %, Lakeville 98–100 %, Klickitat 63–88 %; median T/T*_SE2 down to 0.82 in the longest bin). With the midpoint loss the median stays above the bound in every bin. Allensville: median T/T*_SE2 control 1.07, midpoint 1.10; Lakeville: median T/T*_SE2 control 1.02, midpoint 1.11; Klickitat: median T/T*_SE2 control 1.03, midpoint 1.09. A consistent far field is not a correct one, though. Travel times are pulled to the drives' own times, and a wall no drive gets around stays see-through locally even when T(start, goal) is long enough.

Findings page: https://claude.ai/artifact/GBNHtUw9NkkVxkEnLqVXHv (`report/mobile_arm.html`, built by `page9.py` + `pagetext9.py`; republish the same path).

## Registered runs

| job | run id |
|---|---|
| prepare allensville | 20261118T022119Z-c3fbc4f5 |
| prepare lakeville | 20261118T022122Z-73647bdc |
| prepare klickitat | 20261118T022149Z-87016e35 |
| ntrl allensville 87358 --weight 0 | 20261118T023748Z-5b9f5719 |
| ntrl lakeville 87358 --weight 0 | 20261118T023808Z-c65a276c |
| ntrl klickitat 87358 --weight 0 | 20261118T024648Z-59b46a8b |
| ctl allensville 87358 --weight 0 | 20261118T025508Z-4ca3fda9 |
| mid allensville 87358 --weight 1 | 20261118T024528Z-6a280004 |
| ctl lakeville 87358 --weight 0 | 20261118T042328Z-71cf25be |
| ctl klickitat 87358 --weight 0 | 20261118T042848Z-1e865f66 |
| mid lakeville 87358 --weight 1 | 20261118T025938Z-a3044b56 |
| ntrl allensville 1 --weight 0 | 20261118T043408Z-594e05ea |
| ntrl lakeville 1 --weight 0 | 20261118T044118Z-692bee68 |
| ntrl klickitat 1 --weight 0 | 20261118T044938Z-a96de18c |
| ctl allensville 1 --weight 0 | 20261118T045728Z-cb467132 |
| mid klickitat 87358 --weight 1 | 20261118T043658Z-3d6e309d |
| mid allensville 1 --weight 1 | 20261118T050218Z-6070b066 |
| ctl lakeville 1 --weight 0 | 20261118T063958Z-2c01ced5 |
| ctl klickitat 1 --weight 0 | 20261118T064519Z-9465989f |
| ntrl allensville 2 --weight 0 | 20261118T065038Z-c602da9c |
| ntrl lakeville 2 --weight 0 | 20261118T065758Z-9b98b0f6 |
| ntrl klickitat 2 --weight 0 | 20261118T070619Z-691cf175 |
| ctl allensville 2 --weight 0 | 20261118T071408Z-071b4529 |
| mid lakeville 1 --weight 1 | 20261118T062108Z-b63265db |
| mid allensville 2 --weight 1 | 20261118T071859Z-d5ba84a8 |
| ctl lakeville 2 --weight 0 | 20261118T085608Z-7c0f3ddb |
| mid klickitat 1 --weight 1 | 20261118T081548Z-5fe9a90a |
| mid lakeville 2 --weight 1 | 20261118T090129Z-4fb5a605 |
| ctl klickitat 2 --weight 0 | 20261118T112809Z-78b0cb56 |
| mid klickitat 2 --weight 1 | 20261118T104218Z-4537e4df |
| superseded: ntrl|allensville|87358|--weight 0 (snapshot path bug) | 20261118T022244Z-b9a618a7 |
| superseded: ntrl|lakeville|87358|--weight 0 (snapshot path bug) | 20261118T022706Z-6bfab72d |
| superseded: mid|allensville|87358|--weight 1 (from scratch, stopped) | 20261118T022304Z-f6aab8af |
| superseded: mid|allensville|87358 resume (stopped, path bug) | 20261118T022726Z-1aac4109 |
| superseded: ntrl|klickitat|87358|--weight 0 (stopped, path bug) | 20261118T023516Z-7a05f1fe |

# Tables (report/tables.md)

Generated by report9.py from the registered runs in queue/state.json.

## Houses and queries

| house | E (cells) | prepare run | house prepare (SE(2) truth) | queries | straight segment solved % | T_ref / T*_SE2 (median) | arm share of T_ref (median) |
|---|---|---|---|---|---|---|---|
| Allensville | 43 | 20261118T022119Z-c3fbc4f5 | 20261117T054449Z-0144646b | 240 | 48.8 | 1.44 | 0.31 |
| Lakeville | 83 | 20261118T022122Z-73647bdc | 20261117T054504Z-e60b96b5 | 280 | 46.1 | 1.30 | 0.23 |
| Klickitat | 96 | 20261118T022149Z-87016e35 | 20261117T063709Z-61a4f754 | 280 | 56.1 | 1.29 | 0.22 |

## Planning (MPPI planning seeds 0/1/2, exact scoring)

| house | variant | seed | solved % | per planning seed | arrived through % | did not arrive % | first contact (base / arm-wall / floor / base box / self) | path time / T_ref | T/T_ref | T/T*_SE2 | T below T*_SE2 | Spearman(T, T_ref) | log-speed corr | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Allensville | ntrl | 1 | 84.4 | 201/203/204 of 240 | 8.1 | 7.5 | 29 / 23 / 3 / 1 / 2 | 0.839 | 0.708 | 1.090 | 0.271 | 0.970 | 0.210 | 20261118T043408Z-594e05ea |
| Allensville | ntrl | 2 | 81.5 | 197/197/193 of 240 | 12.4 | 6.1 | 39 / 30 / 3 / 8 / 9 | 0.853 | 0.695 | 1.053 | 0.296 | 0.970 | 0.190 | 20261118T065038Z-c602da9c |
| Allensville | ntrl | 87358 | 85.1 | 204/206/203 of 240 | 9.3 | 5.6 | 22 / 31 / 3 / 2 / 9 | 0.847 | 0.729 | 1.081 | 0.254 | 0.968 | 0.187 | 20261118T023748Z-5b9f5719 |
| Allensville | ctl | 1 | 83.3 | 202/200/198 of 240 | 8.6 | 8.1 | 27 / 26 / 3 / 0 / 6 | 0.85 | 0.714 | 1.076 | 0.271 | 0.970 | 0.210 | 20261118T045728Z-cb467132 |
| Allensville | ctl | 2 | 84.6 | 204/204/201 of 240 | 9.4 | 6.0 | 21 / 33 / 3 / 4 / 7 | 0.866 | 0.693 | 1.044 | 0.342 | 0.969 | 0.194 | 20261118T071408Z-071b4529 |
| Allensville | ctl | 87358 | 88.2 | 209/214/212 of 240 | 5.8 | 6.0 | 11 / 17 / 3 / 3 / 8 | 0.848 | 0.716 | 1.096 | 0.250 | 0.970 | 0.194 | 20261118T025508Z-4ca3fda9 |
| Allensville | mid | 1 | 90.4 | 218/217/216 of 240 | 9.6 | 0.0 | 26 / 33 / 3 / 1 / 6 | 0.8 | 0.791 | 1.111 | 0.008 | 0.971 | 0.222 | 20261118T050218Z-6070b066 |
| Allensville | mid | 2 | 92.4 | 223/218/224 of 240 | 7.6 | 0.0 | 19 / 30 / 3 / 2 / 1 | 0.812 | 0.765 | 1.097 | 0.021 | 0.971 | 0.205 | 20261118T071859Z-d5ba84a8 |
| Allensville | mid | 87358 | 85.6 | 203/204/209 of 240 | 12.8 | 1.7 | 51 / 33 / 3 / 1 / 4 | 0.789 | 0.775 | 1.104 | 0.021 | 0.971 | 0.193 | 20261118T024528Z-6a280004 |
| Lakeville | ntrl | 1 | 50.5 | 141/142/141 of 280 | 28.6 | 21.0 | 157 / 72 / 3 / 0 / 8 | 0.819 | 0.730 | 0.994 | 0.514 | 0.986 | 0.117 | 20261118T044118Z-692bee68 |
| Lakeville | ntrl | 2 | 51.3 | 143/143/145 of 280 | 27.1 | 21.5 | 138 / 72 / 4 / 3 / 11 | 0.822 | 0.725 | 0.994 | 0.507 | 0.985 | 0.098 | 20261118T065758Z-9b98b0f6 |
| Lakeville | ntrl | 87358 | 48.6 | 137/136/135 of 280 | 29.6 | 21.8 | 159 / 73 / 3 / 0 / 14 | 0.816 | 0.750 | 1.005 | 0.493 | 0.985 | 0.098 | 20261118T023808Z-c65a276c |
| Lakeville | ctl | 1 | 48.6 | 135/138/135 of 280 | 26.9 | 24.5 | 147 / 76 / 3 / 0 / 0 | 0.822 | 0.743 | 1.019 | 0.454 | 0.985 | 0.125 | 20261118T063958Z-2c01ced5 |
| Lakeville | ctl | 2 | 52.0 | 145/147/145 of 280 | 26.7 | 21.3 | 141 / 75 / 4 / 0 / 4 | 0.832 | 0.735 | 1.022 | 0.443 | 0.985 | 0.108 | 20261118T085608Z-7c0f3ddb |
| Lakeville | ctl | 87358 | 51.7 | 144/145/145 of 280 | 29.8 | 18.6 | 157 / 79 / 5 / 3 / 6 | 0.832 | 0.747 | 1.011 | 0.475 | 0.985 | 0.097 | 20261118T042328Z-71cf25be |
| Lakeville | mid | 1 | 61.9 | 172/173/175 of 280 | 25.4 | 12.7 | 129 / 62 / 4 / 3 / 15 | 0.816 | 0.841 | 1.119 | 0.068 | 0.984 | 0.108 | 20261118T062108Z-b63265db |
| Lakeville | mid | 2 | 66.7 | 188/185/187 of 280 | 23.8 | 9.5 | 106 / 82 / 8 / 3 / 1 | 0.802 | 0.851 | 1.110 | 0.050 | 0.983 | 0.110 | 20261118T090129Z-4fb5a605 |
| Lakeville | mid | 87358 | 61.7 | 172/172/174 of 280 | 26.3 | 12.0 | 114 / 89 / 4 / 1 / 13 | 0.79 | 0.844 | 1.113 | 0.079 | 0.981 | 0.091 | 20261118T025938Z-a3044b56 |
| Klickitat | ntrl | 1 | 54.0 | 153/150/151 of 280 | 29.0 | 16.9 | 115 / 126 / 0 / 2 / 1 | 0.842 | 0.758 | 1.022 | 0.429 | 0.982 | 0.131 | 20261118T044938Z-a96de18c |
| Klickitat | ntrl | 2 | 57.3 | 159/163/159 of 280 | 23.6 | 19.2 | 116 / 79 / 0 / 0 / 3 | 0.866 | 0.768 | 1.040 | 0.361 | 0.982 | 0.118 | 20261118T070619Z-691cf175 |
| Klickitat | ntrl | 87358 | 55.7 | 156/154/158 of 280 | 28.1 | 16.2 | 155 / 74 / 1 / 1 / 5 | 0.843 | 0.764 | 1.028 | 0.364 | 0.982 | 0.115 | 20261118T024648Z-59b46a8b |
| Klickitat | ctl | 1 | 57.5 | 159/161/163 of 280 | 27.3 | 15.2 | 91 / 135 / 0 / 3 / 0 | 0.854 | 0.770 | 1.030 | 0.400 | 0.983 | 0.138 | 20261118T064519Z-9465989f |
| Klickitat | ctl | 2 | 57.3 | 160/164/157 of 280 | 22.9 | 19.9 | 106 / 68 / 3 / 0 / 15 | 0.858 | 0.760 | 1.028 | 0.375 | 0.982 | 0.111 | 20261118T112809Z-78b0cb56 |
| Klickitat | ctl | 87358 | 57.5 | 161/163/159 of 280 | 27.0 | 15.5 | 138 / 83 / 1 / 2 / 3 | 0.849 | 0.757 | 1.024 | 0.407 | 0.982 | 0.117 | 20261118T042848Z-1e865f66 |
| Klickitat | mid | 1 | 71.3 | 201/199/199 of 280 | 22.4 | 6.3 | 80 / 65 / 2 / 2 / 39 | 0.811 | 0.820 | 1.100 | 0.061 | 0.979 | 0.138 | 20261118T081548Z-5fe9a90a |
| Klickitat | mid | 2 | 69.0 | 191/194/195 of 280 | 19.0 | 11.9 | 101 / 52 / 0 / 1 / 6 | 0.835 | 0.834 | 1.094 | 0.061 | 0.979 | 0.119 | 20261118T104218Z-4537e4df |
| Klickitat | mid | 87358 | 73.6 | 206/207/205 of 280 | 18.6 | 7.9 | 86 / 60 / 3 / 0 / 7 | 0.837 | 0.819 | 1.079 | 0.079 | 0.979 | 0.118 | 20261118T043658Z-3d6e309d |

## Pooled over seeds

| house | variant | seeds | solved % | per seed | arrived through % | did not arrive % |
|---|---|---|---|---|---|---|
| Allensville | ntrl | 3 | 83.7 | 84.4, 81.5, 85.1 | 9.9 | 6.4 |
| Allensville | ctl | 3 | 85.4 | 83.3, 84.6, 88.2 | 8.0 | 6.7 |
| Allensville | mid | 3 | 89.4 | 90.4, 92.4, 85.6 | 10.0 | 0.6 |
| Lakeville | ntrl | 3 | 50.1 | 50.5, 51.3, 48.6 | 28.5 | 21.4 |
| Lakeville | ctl | 3 | 50.8 | 48.6, 52.0, 51.7 | 27.8 | 21.5 |
| Lakeville | mid | 3 | 63.4 | 61.9, 66.7, 61.7 | 25.2 | 11.4 |
| Klickitat | ntrl | 3 | 55.7 | 54.0, 57.3, 55.7 | 26.9 | 17.4 |
| Klickitat | ctl | 3 | 57.4 | 57.5, 57.3, 57.5 | 25.7 | 16.9 |
| Klickitat | mid | 3 | 71.3 | 71.3, 69.0, 73.6 | 20.0 | 8.7 |

## Context: the bare SE(2) rectangle (3 DOF), ntrl-demo, same houses (jeffrey/lift_scale runs, their own queries)

| house | solved % (mean over seeds) | per seed (run) |
|---|---|---|
| Allensville | 90.3 | 87358: 83.9 (20261117T121605Z-9c558da3), 1: 96.8 (20261117T143525Z-fdfc256c) |
| Lakeville | 69.1 | 87358: 69.4 (20261117T094420Z-31c51802), 1: 69.0 (20261117T140830Z-1e68d6a5), 2: 68.8 (20261117T153034Z-153e533a) |
| Klickitat | 71.2 | 87358: 72.5 (20261117T104445Z-0c50416c), 1: 69.3 (20261117T134051Z-0075f71c), 2: 71.7 (20261117T153019Z-18f6316c) |

## Solved % by base distance bin (cells), pooled over seeds

### Allensville

| variant | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 |
|---|---|---|---|---|---|---|
| straight segment | 100 | 88 | 78 | 20 | 8 | 0 |
| ntrl | 90 | 88 | 86 | 80 | 81 | 77 |
| ctl | 88 | 93 | 89 | 81 | 81 | 82 |
| mid | 97 | 96 | 91 | 89 | 83 | 80 |

### Lakeville

| variant | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 | 64-128 |
|---|---|---|---|---|---|---|---|
| straight segment | 100 | 90 | 80 | 50 | 2 | 0 | 0 |
| ntrl | 82 | 67 | 77 | 64 | 28 | 27 | 6 |
| ctl | 85 | 67 | 73 | 59 | 30 | 33 | 9 |
| mid | 91 | 80 | 89 | 82 | 49 | 41 | 11 |

### Klickitat

| variant | 0.5-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 | 64-128 |
|---|---|---|---|---|---|---|---|
| straight segment | 100 | 100 | 95 | 70 | 22 | 2 | 2 |
| ntrl | 81 | 78 | 82 | 64 | 36 | 28 | 22 |
| ctl | 78 | 84 | 82 | 67 | 35 | 34 | 22 |
| mid | 92 | 90 | 95 | 77 | 59 | 54 | 33 |

## Speed fit on held-out pairs by the constraint that sets the label

| house | variant | seed | base vs walls | arm vs walls | internal (floor / base / self) |
|---|---|---|---|---|---|
| Allensville | ntrl | 1 | corr 0.27, pred/true 0.98 | corr 0.21, pred/true 1.19 | corr 0.21, pred/true 1.09 |
| Allensville | ntrl | 2 | corr 0.26, pred/true 1.02 | corr 0.21, pred/true 1.23 | corr 0.17, pred/true 1.13 |
| Allensville | ntrl | 87358 | corr 0.25, pred/true 1.00 | corr 0.20, pred/true 1.22 | corr 0.17, pred/true 1.10 |
| Allensville | ctl | 1 | corr 0.28, pred/true 1.00 | corr 0.22, pred/true 1.20 | corr 0.19, pred/true 1.10 |
| Allensville | ctl | 2 | corr 0.26, pred/true 1.03 | corr 0.21, pred/true 1.24 | corr 0.17, pred/true 1.13 |
| Allensville | ctl | 87358 | corr 0.26, pred/true 0.96 | corr 0.21, pred/true 1.17 | corr 0.18, pred/true 1.10 |
| Allensville | mid | 1 | corr 0.29, pred/true 0.99 | corr 0.22, pred/true 1.19 | corr 0.21, pred/true 1.08 |
| Allensville | mid | 2 | corr 0.27, pred/true 1.01 | corr 0.21, pred/true 1.22 | corr 0.19, pred/true 1.11 |
| Allensville | mid | 87358 | corr 0.28, pred/true 0.97 | corr 0.22, pred/true 1.19 | corr 0.17, pred/true 1.14 |
| Lakeville | ntrl | 1 | corr 0.09, pred/true 1.10 | corr 0.08, pred/true 1.24 | corr 0.14, pred/true 1.13 |
| Lakeville | ntrl | 2 | corr 0.09, pred/true 1.12 | corr 0.08, pred/true 1.26 | corr 0.11, pred/true 1.15 |
| Lakeville | ntrl | 87358 | corr 0.08, pred/true 1.09 | corr 0.07, pred/true 1.22 | corr 0.12, pred/true 1.13 |
| Lakeville | ctl | 1 | corr 0.10, pred/true 1.09 | corr 0.08, pred/true 1.22 | corr 0.16, pred/true 1.10 |
| Lakeville | ctl | 2 | corr 0.09, pred/true 1.10 | corr 0.07, pred/true 1.23 | corr 0.13, pred/true 1.10 |
| Lakeville | ctl | 87358 | corr 0.08, pred/true 1.10 | corr 0.07, pred/true 1.23 | corr 0.11, pred/true 1.13 |
| Lakeville | mid | 1 | corr 0.09, pred/true 1.05 | corr 0.07, pred/true 1.18 | corr 0.13, pred/true 1.11 |
| Lakeville | mid | 2 | corr 0.11, pred/true 1.05 | corr 0.08, pred/true 1.19 | corr 0.13, pred/true 1.11 |
| Lakeville | mid | 87358 | corr 0.08, pred/true 1.06 | corr 0.07, pred/true 1.18 | corr 0.11, pred/true 1.13 |
| Klickitat | ntrl | 1 | corr 0.09, pred/true 1.10 | corr 0.10, pred/true 1.21 | corr 0.15, pred/true 1.12 |
| Klickitat | ntrl | 2 | corr 0.10, pred/true 1.08 | corr 0.09, pred/true 1.20 | corr 0.13, pred/true 1.09 |
| Klickitat | ntrl | 87358 | corr 0.09, pred/true 1.10 | corr 0.09, pred/true 1.20 | corr 0.13, pred/true 1.11 |
| Klickitat | ctl | 1 | corr 0.10, pred/true 1.08 | corr 0.10, pred/true 1.19 | corr 0.16, pred/true 1.09 |
| Klickitat | ctl | 2 | corr 0.09, pred/true 1.09 | corr 0.10, pred/true 1.20 | corr 0.12, pred/true 1.11 |
| Klickitat | ctl | 87358 | corr 0.09, pred/true 1.09 | corr 0.09, pred/true 1.20 | corr 0.13, pred/true 1.09 |
| Klickitat | mid | 1 | corr 0.10, pred/true 1.04 | corr 0.11, pred/true 1.17 | corr 0.16, pred/true 1.09 |
| Klickitat | mid | 2 | corr 0.08, pred/true 1.06 | corr 0.10, pred/true 1.18 | corr 0.14, pred/true 1.10 |
| Klickitat | mid | 87358 | corr 0.08, pred/true 1.07 | corr 0.09, pred/true 1.19 | corr 0.14, pred/true 1.11 |

## During training (planning seed 0) and midpoint refreshes

- Allensville ntrl seed 1: validate 1000: 84.6, 2000: 82.9, 3000: 79.6, 4000: 78.8; training 6.6 min, total 7.0 min, rollbacks 0, NVIDIA GeForce RTX 5070
- Allensville ntrl seed 2: validate 1000: 85.8, 2000: 74.2, 3000: 76.2, 4000: 78.3; training 6.7 min, total 7.1 min, rollbacks 0, NVIDIA GeForce RTX 5070
- Allensville ntrl seed 87358: validate 1000: 75.0, 2000: 82.1, 3000: 82.5, 4000: 85.0; training 6.8 min, total 7.2 min, rollbacks 2, NVIDIA GeForce RTX 5070
- Allensville ctl seed 1: validate 3000: 79.6, 4000: 79.2; training 3.9 min, total 4.3 min, rollbacks 0, NVIDIA GeForce RTX 5070
- Allensville ctl seed 2: validate 3000: 70.8, 4000: 82.1; training 4.0 min, total 4.4 min, rollbacks 0, NVIDIA GeForce RTX 5070
- Allensville ctl seed 87358: validate 3000: 82.9, 4000: 85.4; training 4.1 min, total 4.4 min, rollbacks 0, NVIDIA GeForce RTX 3090
- Allensville mid seed 1: validate 3000: 87.9, 4000: 87.5; training 97.2 min, total 97.5 min, rollbacks 0, NVIDIA GeForce RTX 5070
  - refreshes: 30, mean 181 s; drives reaching their goal (epoch: %) 2001: 84, 2501: 98, 3001: 99, 3501: 99, 4001: 99, 4501: 99; median log(T/target) 2001: -0.03, 2501: 0.00, 3001: -0.01, 3501: -0.00, 4001: -0.00, 4501: -0.00
- Allensville mid seed 2: validate 3000: 90.4, 4000: 88.8; training 96.8 min, total 97.1 min, rollbacks 0, NVIDIA GeForce RTX 5070
  - refreshes: 30, mean 180 s; drives reaching their goal (epoch: %) 2001: 81, 2501: 98, 3001: 99, 3501: 99, 4001: 99, 4501: 99; median log(T/target) 2001: -0.02, 2501: -0.01, 3001: -0.00, 3501: -0.00, 4001: -0.01, 4501: -0.00
- Allensville mid seed 87358: validate 3000: 88.8, 4000: 86.7; training 97.6 min, total 97.9 min, rollbacks 0, NVIDIA GeForce RTX 5070
  - refreshes: 30, mean 181 s; drives reaching their goal (epoch: %) 2001: 82, 2501: 91, 3001: 95, 3501: 98, 4001: 99, 4501: 99; median log(T/target) 2001: -0.02, 2501: -0.01, 3001: 0.00, 3501: 0.00, 4001: -0.02, 4501: -0.00
- Lakeville ntrl seed 1: validate 1000: 50.4, 2000: 45.4, 3000: 50.4, 4000: 48.2; training 7.1 min, total 7.8 min, rollbacks 0, NVIDIA GeForce RTX 5070
- Lakeville ntrl seed 2: validate 1000: 52.1, 2000: 51.8, 3000: 50.4, 4000: 50.7; training 7.1 min, total 7.9 min, rollbacks 0, NVIDIA GeForce RTX 5070
- Lakeville ntrl seed 87358: validate 1000: 51.1, 2000: 49.3, 3000: 51.8, 4000: 48.9; training 7.5 min, total 8.3 min, rollbacks 0, NVIDIA GeForce RTX 3090
- Lakeville ctl seed 1: validate 3000: 47.9, 4000: 50.7; training 4.2 min, total 5.0 min, rollbacks 0, NVIDIA GeForce RTX 5070
- Lakeville ctl seed 2: validate 3000: 52.1, 4000: 51.4; training 4.2 min, total 5.0 min, rollbacks 0, NVIDIA GeForce RTX 5070
- Lakeville ctl seed 87358: validate 3000: 42.9, 4000: 49.3; training 4.2 min, total 4.9 min, rollbacks 0, NVIDIA GeForce RTX 5070
- Lakeville mid seed 1: validate 3000: 62.9, 4000: 65.7; training 112.9 min, total 114.2 min, rollbacks 0, NVIDIA GeForce RTX 3090
  - refreshes: 30, mean 208 s; drives reaching their goal (epoch: %) 2001: 40, 2501: 58, 3001: 64, 3501: 68, 4001: 72, 4501: 65; median log(T/target) 2001: -0.20, 2501: -0.04, 3001: -0.02, 3501: -0.02, 4001: -0.01, 4501: -0.04
- Lakeville mid seed 2: validate 3000: 60.0, 4000: 62.5; training 145.6 min, total 146.2 min, rollbacks 0, NVIDIA GeForce RTX 5070
  - refreshes: 30, mean 277 s; drives reaching their goal (epoch: %) 2001: 50, 2501: 61, 3001: 60, 3501: 67, 4001: 70, 4501: 72; median log(T/target) 2001: -0.22, 2501: -0.03, 3001: -0.00, 3501: -0.01, 4001: -0.01, 4501: -0.01
- Lakeville mid seed 87358: validate 3000: 57.5, 4000: 62.9; training 96.2 min, total 96.8 min, rollbacks 0, NVIDIA GeForce RTX 3090
  - refreshes: 30, mean 178 s; drives reaching their goal (epoch: %) 2001: 43, 2501: 56, 3001: 63, 3501: 68, 4001: 66, 4501: 70; median log(T/target) 2001: -0.38, 2501: -0.10, 3001: -0.05, 3501: -0.04, 4001: -0.04, 4501: -0.01
- Klickitat ntrl seed 1: validate 1000: 57.5, 2000: 61.1, 3000: 53.6, 4000: 59.3; training 7.0 min, total 7.7 min, rollbacks 0, NVIDIA GeForce RTX 5070
- Klickitat ntrl seed 2: validate 1000: 57.5, 2000: 56.4, 3000: 55.4, 4000: 55.0; training 7.1 min, total 7.8 min, rollbacks 0, NVIDIA GeForce RTX 5070
- Klickitat ntrl seed 87358: validate 1000: 57.5, 2000: 59.6, 3000: 57.1, 4000: 56.8; training 7.4 min, total 8.1 min, rollbacks 0, NVIDIA GeForce RTX 3090
- Klickitat ctl seed 1: validate 3000: 56.8, 4000: 57.9; training 4.2 min, total 4.9 min, rollbacks 0, NVIDIA GeForce RTX 5070
- Klickitat ctl seed 2: validate 3000: 54.3, 4000: 58.9; training 4.3 min, total 5.0 min, rollbacks 0, NVIDIA GeForce RTX 5070
- Klickitat ctl seed 87358: validate 3000: 57.9, 4000: 55.7; training 4.2 min, total 4.9 min, rollbacks 0, NVIDIA GeForce RTX 5070
- Klickitat mid seed 1: validate 3000: 65.4, 4000: 68.2; training 145.0 min, total 146.2 min, rollbacks 0, NVIDIA GeForce RTX 3090
  - refreshes: 30, mean 269 s; drives reaching their goal (epoch: %) 2001: 59, 2501: 74, 3001: 74, 3501: 75, 4001: 77, 4501: 78; median log(T/target) 2001: -0.20, 2501: -0.04, 3001: -0.02, 3501: -0.01, 4001: -0.01, 4501: -0.00
- Klickitat mid seed 2: validate 3000: 71.1, 4000: 70.0; training 144.4 min, total 145.7 min, rollbacks 0, NVIDIA GeForce RTX 3090
  - refreshes: 30, mean 268 s; drives reaching their goal (epoch: %) 2001: 54, 2501: 68, 3001: 74, 3501: 73, 4001: 75, 4501: 73; median log(T/target) 2001: -0.32, 2501: 0.02, 3001: -0.01, 3501: 0.00, 4001: -0.01, 4501: -0.01
- Klickitat mid seed 87358: validate 3000: 72.9, 4000: 65.0; training 103.1 min, total 103.8 min, rollbacks 0, NVIDIA GeForce RTX 3090
  - refreshes: 30, mean 192 s; drives reaching their goal (epoch: %) 2001: 53, 2501: 67, 3001: 72, 3501: 73, 4001: 72, 4501: 72; median log(T/target) 2001: -0.43, 2501: -0.03, 3001: -0.00, 3501: -0.01, 4001: -0.02, 4501: -0.04
