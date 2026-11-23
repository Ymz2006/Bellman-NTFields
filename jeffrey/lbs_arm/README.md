# LBS on the cabinet arm (6 joints)

## Request (2026-11-20 ~23:00)

"ok now I want you to revisit the cabinet scenes. See if this LBS method can work. Here are some knobs you can change: fourier embedding
frequency, scaling the lift (you will lift to 7 dimensions, so the 6 joints + h. the joints are normalized to have range 1, and you can
change the range of h ... You can try to make the range of h a function of the degrees of freedom for example. Try to get the arm to work
on the cabinets, see if the current method can do that. Also compare with previous results for metric ntfield and p ntfields, I think you
can try digging them up in the logs but they basically had 0 percent success rate."

## Setup

- Scene: `experiments/cabinet/scenes/cabinet.yml` (table + cupboard with three compartments), the release UR5e + Robotiq 2F-140, the six
  frozen suites of `experiments/cabinet/tests` (600 queries). Final numbers come from the unchanged cabinet evaluator
  (`experiments/ntrl_cabinet/evaluate.py`: native RK4 /600, ntrl-demo MPPI planning seeds 0/1/2 /1800, release scorer), through
  `evaluate6.py`, which replaces only the two field hooks (as `jeffrey/joint_range/evaluate.py`). The release MPPI is not run.
- `arm.py`: the scene in plain torch. Sphere centres fitted from cuRobo's spheres (`export_robot.py`, container layout), error 5e-7 m;
  distances equal to the label code's and the scorer's at 200,000 configurations (7e-7 m); the straight-line verdict of this scene's
  scorer equals the suites' flag on 600 of 600 queries (202 solvable by a straight line).
- Coordinates: x = q / q_range (release normalization, every joint has range 1).
- Clearance d = min(d_obstacle, 0.005 + 5 (d_self - 0.002)) m, speed = clip(d, 0.005, 0.05) / 0.05. NOT the release label
  min(d_obstacle, 0.5 d_self): with that label the self term sets the speed of 86 % of the valid configurations in the sampling box and
  8 % reach full speed. With this d: valid by the scorer iff d >= 0.005 m (checked, 100 %), 70 % of valid configurations at full speed.
- Cells: 1 cell = 0.05 / 0.175 m of clearance, so margin / offset / speed profile / lift profile are those of the mazes.
  E = 18.2 cells per unit of joint space = median |grad_x d| in the band (quantiles 10/25/50/75/90 %: 8.6 / 12.7 / 18.2 / 28.2 / 37.1;
  per joint root mean square 11.4, 11.3, 11.1, 9.5, 11.1, 3.4). Frozen queries are 0.06 to 0.57 apart = 1.1 to 10.4 cells.
- Lift onto 7 dimensions: input (x_1..x_6, h), h = gain G(d) / E. Gain 1: h range 0.118 (G_max 2.14 cells, depth 0). The range of h
  and the frequency of its Fourier column are the same knob (h enters only as 2 pi h b_h).
- Bellman: radius 0.663 cells = 0.036, K 16, cone 6, lengths up to 8 radii, exact clearance along each step (no table in 6-D), on 512
  of the 2,000 pairs of each minibatch (`--bell-n`), weight x (0.03 / radius)^2.
- Segment: as jeffrey/lbs (safe traversals, min leaves, tree + sub-paths, from epoch 2,000), 4,096 traversals per refresh, pairs of
  free configurations with the grasp point in the sampling box at most 0.6 apart.
- Planner: ntrl-demo's MPPI with the mazes' scaling (step 0.0615 cells = 0.00338, tolerance 0.00225, temperature 222), 50 samples,
  horizon 5, 600 steps, joints clamped to their limits; the same for every arm of this study.
- Training data (`prepare.py`): 500,000 pairs, ntrl-demo's arm recipe (x0 in the band with the grasp point in the sampling box,
  x1 = x0 + step up to 0.5).

## Runs

- export `91c5e157`, prepare `9d9afe1e` (E 18.2). Smoke: field `e261dc3a`, eval `51663d87`.
- Queue: `queue/jobs.txt` (`label|cab|seed|pipeline arguments[|gpu]`), `queue/runner.py`, `queue/job.py` (field, then eval).

## Previous results on the same suites (registered runs; ntrl-demo MPPI /1800 unless noted)

- ntrl-demo best setting `664b5d50`: 596 (33.1 %) [free_to_free 234, free_to_compartment 43, compartment_to_free 67,
  compartment_to_compartment 1, same_compartment 26, anywhere 225 of 300 each]; RK4 174/600. Baseline recipe `3c53b360`: 520; RK4 181.
- P-NTFields `53c1dd59`: 635 (35.3 %) [265 / 43 / 64 / 5 / 15 / 243]; RK4 199; its own planner 227/600.
- Straight line: 202/600 (82 / 17 / 11 / 1 / 17 / 74).
- Queries that need a detour (398 of 600, 1,194 MPPI runs): ntrl-demo 113 (9.5 %), baseline recipe 72, P-NTFields 150 (12.6 %).
  compartment_to_compartment: 1, 2 and 5 of 300. This is the "basically 0 %" of the request.

## Results (official cabinet evaluator, ntrl-demo MPPI /1800, training seed 87358)

| run | field | eval | solved | detour /1194 | per suite (f2f, f2c, c2f, c2c, same, anywhere; of 300) | RK4 /600 |
|---|---|---|---|---|---|---|
| ntrl-demo control (this study's data, planner) | `eab861c8` | `910564a4` | 611 (33.9 %) | 131 (11.0 %) | 253, 33, 57, 0, 18, 250 | 203 |
| LBS, gain 1 (h range 0.118), frequency 1 | `793479a8` | `69a9fa28` | 1138 (63.2 %) | 560 (46.9 %) | 282, 106, 270, 48, 152, 280 | 217 |

Own scorer agrees with the evaluator to within a few runs (610 / 1133).

## Why the first LBS run stops short (2026-11-21, `stalls.py`, `report/stalls_793479a8.json`)

Validation is flat from epoch 500 to 5,000 (354 to 381 of 600): the segment loss, on from epoch 2,000, changes nothing. About 205 of
600 plans do not arrive; few collide on the way.

- Planner settings do not matter: steps x3, samples 400, horizon 10, temperature /4 and x4, contact rollouts rejected all give 381 to
  409 of 600 (step x2: 331).
- Arrival depends on the clearance of the goal: 21 % for goals 5 to 9 mm from an obstacle, 47 to 66 % in between, 93 % for goals more
  than 50 mm away. The clearance of the start makes no difference (54 to 72 %).
- Plans stop next to the goal: median 10 % of the start-goal distance left (0.44 cells), 12 mm from an obstacle, all 600 steps used;
  the straight move to the goal collides for 200 of 208.
- Restarted on the suite's certified witness path 0.5 cells (8 planner steps) before the goal, the planner solves 172 of those 208
  (1 cell 166, 2 cells 149, 4 cells 99). So the field guides the last stretch but brings the arm to the goal from a side with no way in.
- The field still sees through: T(start, goal) / straight distance is 1.27 on line-feasible queries and 1.39 to 1.44 on detour
  queries, whose witness paths are 1.9 to 3.8 times the straight distance.
- The segment loss never saw such goals: traversal endpoints are uniform free configurations at least 0.1 cells above contact
  (34 mm clearance); 0.1 % have the hand in the cupboard (1 % at `--min-clear 0`), against 50 % of test starts and goals, whose median
  clearance in the cupboard is 14 mm. The training pairs have 3.8 % of x0 in the cupboard.

Fix under test: `--pair-cup 0.5 --min-clear 0 --safe-margin 0` (each traversal endpoint is, with probability 0.5, a random free
configuration with the grasp point in the cupboard box, from a pool of 200,000; pool clearance q10/50/90 = 0.023 / 0.054 / 0.117
cells, test goals in the cupboard 0.022 / 0.049 / 0.117). Smoke `ba1783d7`. Run label `lbscup`.

Update 03:25: `lbscup` (field `818e7646`, eval `46c1e829`) = 1138/1800, identical to the default. Its traversals reach about 20 % and
stall about 63 % for the whole run, so the segment loss has nothing to learn from where it is needed. Lift gain 2.45 (`61828666`, eval
`7ad54eaf`) = 1107/1800, level with gain 1.

What blocks the plans (`stalls.py --only block`): at the stop the hand is still 20 cm (median) from its goal position, 0.29 rad of
joint motion away, and the straight move passes through the cupboard for 179 of 208 (penetration 15 mm or more for three quarters).
The field's time to the goal at the stop (0.032) is below its time on the witness path at the same distance (0.036): near goals in the
cupboard the field is close to plain distance. One "cell" of this study (1 / E = 0.055 of a joint range, about 0.7 rad on the shoulder
joints) moves the hand by tens of centimetres, so a compartment is about one cell across and the panels are thin walls.

Next: training set with half of the pairs starting in the cupboard (`prepare --extra "--cup-share 0.5"`, run `25f4b800`: x0 in the
cupboard 52.9 %, both ends 3.9 %; scene key `cabc`), runs `ntrlc`, `lbsd`, `lbsdc`; joint Fourier frequency x2, x4.

Update 05:10: lift gain 6 (`6750161e`, eval `cb2f9882`) = 1056/1800 (58.7 %): range of h 0.12 / 0.29 / 0.71 -> 63.2 / 61.5 / 58.7 %, so a wider h loses.
Training set with 53 % of the pairs starting in the cupboard (prepare `25f4b800`, scene key `cabc`): ntrl-demo `f2943872` (eval `ad102ef0`) = 579 (first set 611),
LBS `78627133` (eval `77e5f7a5`) = 1149 (first set 1138). More samples in the cupboard, for the pairs or the segment endpoints, do not move the result; `lbsdc`
(both together) was dropped from the queue. Queue now: `lbsf2` (5070), `lbsf4`, `lbsf05`, `lbsg04` (gain 0.41, h range 0.05), `noL` (Bellman + segment, no lift), `lbsfs`.

Update 06:45: joint Fourier frequency x4 (`6e662858`, eval `e872cb73`) = 390/1800 (21.7 %), below the straight line. `stalls.py --only ends`: 543/600 plans do not
arrive (208 at x1); 440 of them sit at a local minimum (none lower within 3 planner steps; 63/208 at x1); median stop 1.12 x the start-goal distance from the goal
(0.10 at x1) with T fallen to 64 %. Training losses are higher too (ntrl loss 6.1e-3 vs 2.4e-3, mid loss 0.51 vs 0.27). x2 is at ~235/600 during training
(first run 354-381). So raising the frequency breaks the field; the cell scale (x18) is out of reach in 6-D.

Update 08:30: frequency x2 (`74894796`, eval `73c7cca2`) = 722/1800 (40.1 %); x0.5 (`584af8c7`, eval `3282c2db`) = 1141 (63.4 %), level with x1.
Frequency so far: x0.5 63.4, x1 63.2, x2 40.1, x4 21.7 %. x0.25 added (`lbsf025`), at 393-394/600 during training. Running: `lbsf025` (5070), `lbsg04` (3090).

Update 11:25: lift gain 0.41 (h range 0.05; `3591894b`, eval `d4d7babb`) = 1082/1800 (60.1 %). Range of h: 0.05 / 0.12 / 0.29 / 0.71 -> 60.1 / 63.2 / 61.5 / 58.7 %.
No lift (Bellman + segment; `44b684a2`, eval `2292f9fb`) = 521 (28.9 %), below the ntrl-demo control (611). Queued: `noS` (lift + Bellman), `noB` (lift + segment).

## Real-world cupboard (`real_world_6dof`), 2026-11-21

Request: "queue up this one real_world_6dof. It is the real world scenario". The scene comes from `jeffrey/real_world` (measured cupboard, one UR5e).

- **Planning copy of the scene:** `scenes/real_world_6dof.yml`, written by `make_real_scene.py`. The measured scene's pedestal top touches the arm's base; the release
  scorer checks the two fixed shoulder spheres too and rejected every configuration (lower shoulder sphere 30.5 mm inside the pedestal). The copy lowers the pedestal top
  by 0.04 m (9.5 mm clear). Nothing else differs.
- **Test suites:** `run.py tests --scene real_world_6dof` (run `553cff16`), generator `make_tests_rw.py` = the cabinet generator with its pool pre-filter taken from the
  scene's sampling box (the cabinet's "near the table" filter would drop the lower zones). 600 certified queries, straight line solves 226
  (58 / 30 / 25 / 0 / 58 / 55 per suite). Frozen by `run.py freeze --run ...` into `tests_real_world_6dof/`.
- **Export** `d4035b1b`, **prepare** `f452e2ae` (E 15.8 cells per unit; 24.8 % of the pairs start in the cupboard). Queue scene key `rw6`.
- **Runs queued:** `rw_ntrl` (ntrl-demo control) and `rw_lbs` (`--lift time --bell 1 --traversals 4096`, the first cabinet LBS setting). No earlier baselines exist for this scene.
- `report.py` reads each run's scene from its prepared run; `page.py` has a section for this scene. `viewer.py` builds the 3-D path viewer (`viewer_scenes.json` lists the
  scenes and runs; artifact https://claude.ai/artifact/JzeY8pHydwecHDoowyJXnz).

Update 12:30: frequency x0.25 (`5f629752`, eval `d3ecaf0a`) = 1166/1800 (64.8 %). Frequency: x0.25 64.8, x0.5 63.4, x1 63.2, x2 40.1, x4 21.7 %.
Update 13:05: per-joint frequency by sensitivity (`b11be4ae`, eval `8b1c27cc`) = 1109/1800 (61.6 %), level with x1 within a run's noise.

### Update 14:45 (2026-11-21): real-world LBS, no-segment ablation

| run | scene | solved of 1,800 | detour runs | field / evaluation |
|---|---|---|---|---|
| straight line (control) | real_world_6dof | 226 of 600 (37.7 %) | 0 | – |
| rw_ntrl (ntrl-demo control) | real_world_6dof | 630 (35.0 %) | 146 of 1,122 (13.0 %) | `9fa94e84` / `799684da` |
| rw_lbs (LBS, gain 1, frequency × 1) | real_world_6dof | **1,149 (63.8 %)** | 512 of 1,122 (45.6 %) | `cd9646ee` / `813a077f` |
| noS (LBS without the segment loss) | cabinet | 1,147 (63.7 %) | 561 of 1,194 | `6a50478c` / `d4d926ac` |

- Real-world, per suite (f2f, f2c, c2f, c2c, same, any; of 300): LBS 254, 158, 187, 58, 252, 240; ntrl-demo 199, 80, 94, 6, 81, 170. RK4 250 / 210 of 600.
- The segment loss adds nothing on the cabinet (1,147 without it, 1,138 with it). noB is training.
- Viewer (Version 2) has the real-world scene; `viewer_scenes.json` lists the runs it shows.

## Two arms: `real_world_12dof` (requested 2026-11-21 ~14:30: "queue up the 12dof as well, keep the gpus busy")

The study's 6-joint code could not run this scene: the scene model, the suite generator, and the release scorer / evaluator (cuRobo's
robot) are all one arm. What was added, all in the host venv (`run12.py` registers; its own snapshots, `run.py` and the 6-joint sources
are unchanged):

| file | what |
|---|---|
| `scenes/real_world_12dof.yml` | planning copy (`make_real_scene.py`): both pedestals' tops lowered 4 cm, as for 6 DOF |
| `arm12.py` | `ArmSet(CabinetArm)`: every arm is arm.py's arm on its own base; x = 12 normalized joints (left, right). valid = each arm by the release rule (all spheres ≥ 5 mm from the cuboids, self ≥ 2 mm) and the arms ≥ 5 mm apart (all 53 × 53 sphere pairs). clearance d = min(d_obs, d_inter, 0.005 + 5 (d_self − 0.002)). `check()` compares with `jeffrey/real_world/robot.py`: 100,000 of 100,000 valid flags equal, distances to 0.3 µm |
| `make_tests12.py` | certified suites, the cabinet recipe carried to two arms (see its docstring). Witness = straight line, RRT-Connect in 12 joints, or one arm at a time (6-joint RRT-Connect with the other arm held, either order) |
| `prepare12.py` | prepare.py for `ArmSet`; asserts that every stored witness passes `core.score` at 0.001 rad and that the straight-line flags are `core.score`'s |
| `pipeline12.py` | pipeline6.py unchanged, with `arm12` as the scene module and `core.score` at 0.001 rad per joint |
| `run12.py` | registry modes `tests`, `freeze`, `prepare`, `field` |
| `queue/job.py` | a prepared scene with more than 6 joints goes through `run12.py field`; no evaluation run (`"eval": null`) |

Choices and their reasons:
- **No official evaluator.** The count of the field run is the result: ntrl-demo's MPPI (`core.plan`), planning seeds 0/1/2, scored
  by `core.score` with `ArmSet`'s valid rule at 0.001 rad per joint. The same function certifies the suites.
- **Witness planner.** RRT-Connect in all 12 joints with the cabinet budget (10,000 iterations) found a witness for about 1 in 8
  non-straight candidates in a probe (29 of 183). Adding one-arm-at-a-time brought the certified share to 69 of 192 (36 %); by
  suite 14, 9, 9, 4, 17, 16 of 32. The suites therefore lean towards queries these planners can solve; the 6-DOF suites certify 60–85 %.
- **Pairs 0.02 to 4 √2 rad** (4 rad per arm). Uniform among in-range pairs of random 4,096 × 4,096 blocks of the endpoint pools
  (a random pair is in range with probability 4e-5 to 2e-4).
- **Sampling.** Uniform 12-joint samples with both grasp points in the box are valid 0.7 % of the time. `ArmSet.candidates()` draws
  each arm from its own valid-alone, in-box configurations; this has the same distribution as rejection from uniform 12-joint
  samples (checked: joint means, standard deviations, clearance quartiles and cupboard share agree) and is about 60 times faster.
  `free()`, `measure()` and the training pairs use it.
- **Training pairs** `r_max = 0.5 √2`. Only about 7 % of the x1 survive (6 DOF: 29 %), so the kept gaps are short (median 0.06).
- **Traversal pairs** `--pair-max 0.85` (0.6 √2): at 0.6 no random pair of free configurations is in range (median distance 1.36).
- **Cost.** One clearance is 3.0 M/s compiled on the 3090 (6 DOF: 5.1 M/s); the arm-to-arm term costs little.

Registered: tests smoke `016adb17` (failed: 300-iteration smoke budget), tests `829bdba1` (killed by me, round limit 8 too low),
tests `0defb4fb` (the suite).

### Fillers queued on the 5070 (the user: "keep the gpus busy")

Training seeds 87359 and 87360 of the first LBS run and its control, cabinet and real_world_6dof (`lbs2`, `ntrl2`, `rw_lbs2`,
`rw_ntrl2`, `lbs3`, ...). Every result so far is one training seed.

### Update 16:10 (2026-11-21)

- **noB** (lift + segment, no Bellman update), cabinet, seed 87358: field `fb3d627a`, official `6f854c9f`: **1,134 of 1,800 (63.0 %)**,
  detour 555 of 1,194. With LBS 1,138, noS 1,147 and noL 521, each of the Bellman update and the segment loss can be dropped on its
  own; without the lift the run is below the control. `liftonly` (`--lift time --weight 0`, neither term) is queued on the 3090
  between the two 12-DOF runs to say whether one of the two is needed at all.
- **12-DOF suites** (tests `0defb4fb`, frozen in `tests_real_world_12dof/`, manifest sha `8dd0cc62…`): 600 certified in 7 rounds,
  59 min. Certified of candidates by suite: free → free 158 / 320, free → zone 139 / 480, zone → free 106 / 320,
  zone → another 103 / 1,120, same zone 178 / 320, anywhere 157 / 320 (841 / 2,880, 29 %). Witness sources of the 600: straight
  line 55, RRT-Connect in 12 joints 255, one arm at a time 290. Median start-goal distance about 5.1 rad (cap 5.66).
- **12-DOF prepare** `5fdd3960` (queue key `rw12`): E 13.8 cells, 500,000 pairs, 600 / 600 witnesses pass the scorer, straight line
  55 / 600, model check 200,000 / 200,000 flags. Field smoke `dd047d1f` (40 epochs, 3 / 60; integration test).
- Queued on the 3090: `rw12_ntrl` (started 16:03), `liftonly`, `rw12_lbs`.

### Update 16:35 (2026-11-21): the first 12-DOF control does not count; second pair set

- `rw12_ntrl` (field `3ca3e6a4`, prepare `5fdd3960`): 11 of 1,800, 340 of 600 queries never arrive (they end as far from the goal
  as they started). **Not a fair result; my data design.**
- **Planner control** (`euclid.py`, `run12.py euclid`, run `0611c599`): the same MPPI on T = |x − g| arrives on every query and
  solves 39, 42, 37 of 600 (118 of 1,800, 6.6 %; 113 of the 165 straight-line runs, 5 detour runs). 200 samples: 8.1 %; 2,400
  steps: identical. So the planner settings are adequate in 12 joints and the control above was worse than no field.
- **Cause.** `prepare12.py` drew a pair as `prepare.py` does: x0 in the slow band, x1 = x0 + step of uniform length up to
  r_max = 0.707, kept if x1 is not in collision. With two arms a long step almost always ends in collision, so the kept pairs
  were short (length quantiles 10/50/90 %: 0.007 / 0.063 / 0.316; test queries 0.35 / 0.46 / 0.61). 3.6 % of the pairs are as
  long as the median query (one-arm real-world scene: 22 %), and the long ones end outside the task box (an in-box x1 at
  length 0.7 is accepted 0.007 % of the time, an x1 anywhere 0.6 %).
- **Fix** (`sample_pairs(bins=20, box=0.5, reps=256)`, flags `--gap-bins`, `--x1-box`): the kept pairs fill 20 equal-width length
  bins equally; in every bin half of the x1 have both grasp points in the task box; each x0 is tried with 256 steps. Result
  (prepare `2850cb28`, key `rw12b`, 8 min): 25,000 pairs per bin, x1 in the box 56.1 % (one-arm scene 55.8 %), x1 at full speed
  25 %. r_max is unchanged (the same rule as the one-arm scenes; 10 % of the queries are longer than it, as in 6 DOF).
- Queue: `rw12_lbs` dropped before it started; `rw12b_ntrl`, `rw12b_lbs` (`--pair-max 0.85`) on the 3090.
- **liftonly** (`--lift time --weight 0`, cabinet): field `b59cd66d`, own scorer 1,118 of 1,800 (LBS 1,138 official). The lift
  alone gives the cabinet result; see the official count in the report.

### Update 20:50 (2026-11-21): 12-DOF LBS, lift only, and why the segment loss could not act

| run (seed 87358) | scene | solved of 1,800 | detour |
|---|---|---|---|
| `liftonly` `b59cd66d` / `261af963` | cabinet | 1,115 (61.9 %) official | 534 / 1,194 |
| `rw_lift` `c6d89aec` / `57dfe0f8` | real_world_6dof | 1,162 (64.6 %) official | 521 / 1,122 |
| straight line | real_world_12dof | 55 / 600 (9.2 %) | 0 |
| Euclidean field + MPPI `0611c599` | real_world_12dof | 118 (6.6 %) own scorer | 5 / 1,635 |
| `rw12b_ntrl` `78ce002c` | real_world_12dof | 30 (1.7 %) | 12 |
| `rw12b_lift` `9b85e241` | real_world_12dof | 194 (10.8 %) | 118 |
| `rw12b_lbs` `ac867258` (236 min) | real_world_12dof | 183 (10.2 %) | 126 |

- One arm: the lift alone equals LBS on both scenes (cabinet seed 87359: LBS `f4e8487c` 1,123, ntrl `2e95bb3b` 586).
- Two arms: lift only = LBS = about 10 %. About 490 of 600 runs stall; arrival of the lift-only run by query distance quintile
  (seed 0): 50, 27, 21, 4, 1 of 120 (0.17–0.40 … 0.55–0.80).
- **Why S could not act in `rw12b_lbs`:** `draw_pairs` takes random free configurations at most `--pair-max` apart. In 12 joints
  those pairs have length quantiles 1/10/50/90 % = 0.50 / 0.65 / 0.79 / 0.84; 95 % of the test queries are shorter than 0.65.
  Per refresh 3–23 of 4,096 traversals reached (about 3,960 stalled), so the midpoint loss saw almost only stalled sub-paths.
- **Fix:** `--pair-data` (trainer6 `data_pairs`, `draw_data_pairs`): traversal pairs from the training pairs with both ends free
  (in box, clearance ≥ contact + min_clear) in 10 equal length bins (15,390 usable; 945–4,580 per bin). Smoke `c1954a87`: 60–71
  of 512 reach at epochs 21–31. Queued on the 3090: `rw12b_ls_pd` (lift + segment), then the lift-only knob sweep
  (`rw12b_lift_f05`, `_g2`, `_f025`, `_g04`, `_g35`), then `rw12b_lbs_pd`.

### Close pairs: predicted against true travel time (user request, 2026-11-21 21:00)

`closepairs.py` (`run12.py closepairs --prepared … --fields name=<field run> …`): a valid in-box, b = a + random direction × 0.25 to
4 cells, kept if the straight step is free; truth = `core.step_time` (the Bellman update's step cost); 5,000 pairs per length.
Runs: real_world_6dof `ef2a7321`, real_world_12dof `0ffd036a`, cabinet with the ablations `97cf4df3`.

| median predicted / true (share within 20 %) | 0.25 cells | 1 cell | 4 cells |
|---|---|---|---|
| 6 DOF ntrl-demo | 1.14 (55 %) | 1.13 (61 %) | 1.01 (79 %) |
| 6 DOF lift only | 1.03 (79 %) | 1.02 (81 %) | 0.96 (84 %) |
| 6 DOF LBS | 1.40 (19 %) | 1.43 (19 %) | 1.45 (15 %) |
| 12 DOF ntrl-demo | 0.99 (18 %) | 1.29 (18 %) | 1.56 (14 %) |
| 12 DOF lift only | 0.89 (25 %) | 1.04 (25 %) | 1.16 (33 %) |
| 12 DOF LBS | 2.45 (10 %) | 3.29 (7 %) | 4.57 (3 %) |

- 12 DOF: open-space pairs are overestimated (ntrl 1.9×, lift 1.5× at 1 cell), slow-band pairs underestimated (0.47×, 0.41×);
  37 % of the pairs are in the slow band (6 DOF: 7 %).
- Cabinet ablations (0.5 / 2 cells): lift 0.99 / 0.98, lift + segment 0.99 / 1.00, lift + Bellman 1.21 / 1.27, LBS 1.24 / 1.28:
  the overestimate comes with the Bellman update. Not isolated further.

### Update 21:15 (2026-11-21): leaf experiment (user request) and queue change

- User: "experiment with 6dof lbs + [the midpoint-loss artifact] the versions of this … I think the min leaves should help … maybe
  increase its loss weight." The 6-DOF LBS runs already use the artifact's main version (safe drives, min leaves, tree + sub-paths,
  weight 1, 1-cell leaves). Runs on real_world_6dof (LBS overestimates close pairs 1.40–1.45 there), full LBS with one change:
  `rw_lbs_w3` (`--weight 3`, 3090), `rw_lbs_w10` (`--weight 10`, 5070), `rw_lbs_lpath` (`--leaf path`, 3090), `rw_lbs_lnet`
  (`--leaf net`, 5070). Each scored by the official evaluator and `closepairs.py`.
- Dropped with the user's agreement: seed 87360 repeats (`lbs3`, `ntrl3`, `rw_lbs3`, `rw_ntrl3`) and `rw12b_lbs_pd`.
- `rw12b_lift_f05` (12-DOF lift only, Fourier frequency × 0.5): field `540c0bf9`, **673 of 1,800 (37.4 %)**, detour 521 / 1,635;
  frequency × 1: 194. Arrived in contact 5–9 of 600, did not arrive about 410.
- `rw_lbs2` (real_world_6dof LBS, seed 87359): `e092e8a8` / `f4e9e21a` 1,177 (65.4 %); seed 87358: 1,149.
- `rw12b_ls_pd` (12-DOF lift + segment, `--pair-data`): field `62cd88bd`, 260 of 1,800 (14.4 %), detour 175 / 1,635; 1,488 of 4,096 traversals reach at the last refresh. Queued `rw12b_ls_pd_f05` (the same at frequency × 0.5) behind the knob runs. `rw_ntrl2` `ccb90c49` / `5698b216`.

### Update 00:05 (2026-11-22): leaf experiment, first results

| run on real_world_6dof (seed 87358) | official | close pairs, median predicted / true (0.25 · 1 · 4 cells) |
|---|---|---|
| LBS, min leaves, weight 1 (`cd9646ee`) | 1,149 | 1.40 · 1.43 · 1.45 |
| `rw_lbs_w3` `7dd2f93a` / `ad87cf83` | 1,143 | 1.39 · 1.42 · 1.42 |
| `rw_lbs_w10` `26841e12` / `bc9d8e12` | 1,156 | 1.29 · 1.30 · 1.30 |
| lift only (`c6d89aec`) | 1,162 | 1.03 · 1.02 · 0.96 |

Close-pair run `d01b15fe`. A heavier segment loss lowers the overestimate slowly and leaves planning unchanged.
- Smoke `c558d7ef`: along the training drives 99.9 % of the leaf chords are free and the true straight-step time is 0.94 of the
  drive's own time over the leaf, so path leaves are nearly truth-anchored and min leaves pick the drive's time wherever the
  network overestimates. The loss sees only pieces of the drives (about 130,000 nodes per refresh); the Bellman update acts on all
  500,000 training pairs.
- New options: `--leaf step` (leaf = true straight-step time between its ends where free; trainer6), `--bell-goal` (the Bellman
  backup's last candidate is the straight step towards the pair's other end; core.bellman_term). My hypothesis for the Bellman
  inflation, not yet tested: the backup is a minimum over sampled steps, never below the true backup; when no sampled step is near
  the best direction the fixed point is T inflated by 1 / (progress per unit cost of the best sampled step). It would explain
  1.2–1.45 with 6 joints and 2.4–4.6 with 12.
- Queued: `rw_lbs_lpath` (3090, running), `rw_lbs_bgoal` (3090 next), `rw_lbs_lnet` (5070, running), `rw_lbs_w30` (5070 next).
- Report sections added: `#close` (close pairs, three line charts) and `#leaves` (segment-loss versions).

### Update 02:40 (2026-11-22): leaf experiment results and where the bias comes from

| run on real_world_6dof | official | close pairs 0.25 · 1 · 4 cells |
|---|---|---|
| `rw_lbs_lpath` `2e8bb124` / `f1aef4c5` (path leaves) | 1,155 | 1.42 · 1.45 · 1.45 |
| `rw_lbs_lnet` `5bea368d` / `0e02db55` (net leaves) | 1,166 | 1.44 · 1.48 · 1.49 |
| `rw_lbs_bgoal` `2a52f997` / `50e5a4af` (`--bell-goal`) | 1,133 | 1.30 · 1.32 · 1.35 |

Close-pair run `e45e2e53`. No leaf rule or weight removes the overestimate; planning is the same for all (63–65 %).
- `belldiag.py` (`run12.py belldiag`): the Bellman backup's signed residual on trained fields. real_world_6dof `ab571488`: mean
  T − y = −0.55 step radii on the lift-only field (−0.52 with the goal step), −0.02 on LBS; mean T 0.185 vs 0.299.
  real_world_12dof `433e9a5a`: −1.94 on lift only, +0.41 on LBS; mean T 0.392 vs 1.552. So my sampled-minimum hypothesis is at most
  a small part. Reading: training pairs start in the slow band, where every field underestimates the time (close pairs by speed:
  0.34–0.57 of the truth); the backup with the true step cost asks for more time; the network raises T everywhere instead (LBS
  open-space steps 1.48, slow steps 0.45; lift only 1.05, 0.34).
- Cabinet knob runs, close pairs `a8170736` (LBS, 1 cell): frequency × 0.25 1.10, × 0.5 1.08, × 1 1.25, × 2 2.02, × 4 3.83; h range
  0.05 / 0.12 / 0.29 / 0.71: 1.24 / 1.25 / 1.30 / 1.35. The overestimate follows the frequency; the two high-frequency runs are
  the ones whose planning collapsed (40.1 %, 21.7 %).

### Update 05:10 (2026-11-22)

- **Weight 30** (`rw_lbs_w30`, field `2f17ddbe`, eval `04de7517`, close pairs `2a6828de`): official 1,157 / 1,800; close-pair ratio at 1 cell 1.16.
  Weight trend 1 / 3 / 10 / 30 = 1.43 / 1.42 / 1.30 / 1.16 (lift only 1.02): about 65 % of the gap closed at weight 30, planning unchanged (63.5–64.3 %).
  Steps next to obstacles go down with the rest (0.39 of the truth; LBS 0.45, lift only 0.34), so the segment loss pulls the whole field toward the lift-only field.
- **12-DOF lift-only knobs** (own scorer, of 1,800): frequency ×0.25 666 (`8f4713f3`), ×0.5 673, ×1 194; h range 0.38 / 0.06 / 0.54 = 182 / 141 / 139
  (`dcdd9a11`, `615635f4`, `35f0ab78`). Only the frequency matters; ×0.25 is level with ×0.5.
- **12-DOF lift + segment at ×0.5** (`d0cf3f53`, `--pair-data`): 679 vs lift only 673. The segment loss was active (1,565–1,739 of 4,096 traversals reached) and adds nothing.
  At ×0.5 almost no path ends in contact (11–16 of 1,800); about 62 % of the runs stop before the goal.
- Queued (my additions): `rw12b_ntrl_f05` (control at half frequency), `rw_lbs_w100`, `rw_lbs_f05`, `rw_lift_f05` on the 5070; `rw12b_lbs_pd_f05` running on the 3090.
- 05:05: **12-DOF ntrl-demo control at frequency ×0.5** (`133b76ca`): 48 / 1,800 (×1: 30); 1,740 of its 1,800 paths end in contact. The half-frequency gain (673) needs the lift.
- 07:40: **Weight 100** (`rw_lbs_w100`, field `1c7c406c`, close pairs `afd74bb7`): official 1,139 / 1,800; close-pair ratio 1.05 at 1 cell (0.25 / 4 cells: 1.05 / 1.10), 64 % of steps within 20 %
  (lift only 81 %, LBS weight 1 19 %). Weight trend 1 / 3 / 10 / 30 / 100 = 1.43 / 1.42 / 1.30 / 1.16 / 1.05. Near-obstacle steps 0.36 of the truth (lift only 0.34).
- 08:30: **12-DOF full LBS at frequency ×0.5** (`71c6d343`, `--pair-data`): 596 / 1,800 (lift only 673, lift + segment 679); 62 paths in contact (lift only 16).
  Close pairs at ×0.5 (`78f55697`), median at 1 cell (open space / next to obstacles): ntrl 1.06 (1.58 / 0.38), lift 0.77 (1.10 / 0.32), lift + segment 0.80 (1.15 / 0.33),
  LBS 1.65 (2.42 / 0.72). Half frequency halves the 12-DOF LBS overestimate (3.29 at ×1) and does not remove it; "half frequency ≈ unbiased" holds on the cabinet only so far.
- Queued on the 3090 (my addition): seed 87359 repeats of lift only, lift + segment and LBS at ×0.5 (`rw12b_*_f05_s2`), to judge 596 against 673.
- 09:30: seed 87359 at 12-DOF ×0.5: lift only 639 (`5ff97e4f`), lift + segment 628 (`f2c07bb8`); first seed 673 / 679. Seeds differ by 30–50 runs; lift only and lift + segment are level.
- 10:35: **One-arm real-world at frequency ×0.5**: LBS (`121e678e`, eval `a35eb399`) official **1,242** / 1,800 (×1: 1,149; seed 87359 1,177), close-pair ratio 1.11 (×1: 1.43), 69 % within 20 %;
  lift only (`0e849084`, eval `ebedfeb4`) 1,181 (×1: 1,162), ratio 0.99, 87 % within 20 % (close pairs `bfb27dd9`). One seed. Queued on the 5070 (my additions): `rw_lbs_f05_s2` (seed 87359), `rw_lbs_f025`.
- 13:10: 12-DOF full LBS ×0.5, seed 87359 (`b127b8d5`): 565 (seed 87358: 596). Below lift only by 74 and 77 runs on the two seeds: the Bellman update costs planning with two arms at this frequency. Queued (my addition): `rw12b_lbs_pd_f025` on the 3090.
- 13:20: one-arm real-world LBS ×0.5, seed 87359 (`58f9b67f`, eval `25cda11f`): **1,290** (seed 87358: 1,242; ×1 seeds 1,149 / 1,177); close-pair ratio 1.14 (`82baab58`). Queued: `rw_lift_f05_s2` (lift-only seed 87359 at ×0.5), `rw_ntrl_f05` (control at ×0.5).
- 16:10: one-arm real-world LBS ×0.25 (`1b5aa297`, eval `3cb827fc`): 1,223; close-pair ratio 1.04, 72 % within 20 % (`a0c12aa6`). Level with ×0.5 (1,242 / 1,290) in planning, closer in values.
- 16:30: one-arm real-world at ×0.5: lift only seed 87359 (`85295aaa`) 1,146 (seed 87358: 1,181), so LBS (1,242 / 1,290) is ahead on both seeds by 61 and 144; ntrl-demo control at ×0.5 (`eb0c2c00`) 666 (×1: 630 / 653).
- 17:05: 12-DOF full LBS at ×0.25 (`e63151c0`): 612 (×0.5: 596 / 565; lift only ×0.25: 666). Close pairs (`ec68700f`): LBS 1.42 at 1 cell, 1.95 in open space
  (×1 / ×0.5: 3.29 / 1.65); lift only ×0.25 0.67 (0.96 open space, 0.30 next to obstacles). Lower frequency does not close the gap to lift only with two arms.
- 21:20: cabinet LBS ×0.25, seed 87359 (`79858f70`, eval `bababfcf`): 1,153 (seed 87358: 1,166; ×1 seeds 1,138 / 1,123). +28 and +30 runs on the two seeds. Queue empty.
