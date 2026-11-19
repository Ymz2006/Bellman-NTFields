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
