# Arm placement: where to put the two arms (12 joints) in front of the measured cupboard

## Request (2026-11-23 06:05 UTC)

> Check out this repo, especially the arm tasks. I want you to also git clone https://github.com/ruiqini/ntrl-demo as reference. You
> will be training the arm tasks and the goal is to find a good position for the 12 degree of freedom arms such that they achieve the
> highest success rate. You are allowed to move them further away from the cabinet in order to look for the best placement of these
> two arms. They have to both be symmetric about the cabinet center though, but these 2 arms further back might make this task easier.
> Once you believe you have found the optimal configuration, you can find some configurations where both arms are in the cabinet, and
> they both reach different points of the cabinet, or some other difficult and complex configurations. Make sure they all work, and
> give me say 10 working start/end configurations. Use all gpus. Train these 2 models across all of the placements of the arms.
> [...] move the arms further and further away while they can still reach parts of the insides of the cabinet [...] Once it has a
> high success rate in these complex scenarios such as involving planning from inside the cabinet to inside the cabinet, you can
> stop and sample a few paths in 12d space. You will create an artifact that tells me all of your findings, and save the trained
> models for both methods as well. Actually, you have 10 hours [...] optimize the training so you can give multiple trainings on one
> gpu. also use the large vram to make data sampling much faster with bigger batches.

My reading of "these 2 models" (not confirmed by the user): the two methods of `jeffrey/lbs_arm`'s two-arm study,
**ntrl-demo** (metric NTFields, the reference clone, `--lift none --weight 0`) and **the repository's LBS field**, in the setting that
was best with two arms there (`--lift time --weight 0 --freq 0.5`: the travel-time lift at half Fourier frequency; "lift" below).

## This machine (2026-11-23)

A rented container with two 48 GB RTX 4090, no Docker, no cuRobo, no assets, no earlier `results/`, an empty `baselines/`. What I set up:

- `~/.venvs/bellman-ntfield` (Python 3.12, torch 2.7.1+cu128, numpy 2.2.6), as `environment/local/README.md` describes the host venv.
- `baselines/ntrl-demo`: `git clone --depth 1 https://github.com/ruiqini/ntrl-demo` (commit `5ed065f`, the commit this repository records
  for that path), sources unmodified. `git status` shows the path as modified only because upstream tracks Python bytecode caches that an import rewrites.
- `arm_export.npz` (`armexport.py`): `jeffrey/lbs_arm/export_robot.py` needs cuRobo. The arm's collision model it exports was already
  in `jeffrey/real_world/robot_geometry.npz`; `armexport.py` turns that file back into the keys `arm12.build_model` reads.
  Check: `prepare12.py` on the earlier study's scene and frozen suite certifies 600 of 600 stored witnesses and reproduces its 55
  straight-line flags (`arm12.check`: 200,000 of 200,000 valid flags equal).
- The registry database is new on this machine (`~/.local/share/ntfields-experiments`); earlier run ids quoted from other READMEs are not in it.
- **Nothing on this instance survives a recycle** (`/workspace` is not a volume). The trained models and the report are in this
  folder (`models/`, `report/`), which is in the Git working tree but not committed.

## Setup

- **Placement** `(dx, s, h)`: `dx` = distance from the bases' axis to the cupboard's front face, `s` = half spacing of the bases (left at
  y = +s, right at y = −s, symmetric about the cupboard's centre plane), `h` = base height above the floor. `place.py` writes the scene
  with `jeffrey/real_world/make_scenes.py`'s own functions (same measured cupboard, pedestals, zone boxes, sampling-box rule) and the
  pedestal tops lowered 4 cm as in `jeffrey/lbs_arm`. `(0.25, 0.30, 0.60)` reproduces `jeffrey/lbs_arm/scenes/real_world_12dof.yml` exactly.
  Moving the arms back = a larger `dx`. The grasp-point sampling box runs from 0.08 m in front of the bases to the cupboard's back
  (the existing rule), so it grows with `dx`.
- **Reach** (`place.py reach`): `make_scenes.Reach` per arm (4 cm voxels of the three zone interiors, flood fill, other arm at home).
- **Test suites** (`suites.py`): `jeffrey/lbs_arm/make_tests12.py`'s recipe (its functions are imported) with 50 queries per stratum
  instead of 100, round sizes set per stratum, and **3,000 RRT-Connect iterations per stage instead of 10,000**: at 10,000 one
  placement took about 70 minutes of a 4090 shared three ways. Fewer iterations certify fewer of the hard candidates, so these suites
  lean a little more towards queries the witness planner solves quickly than the earlier study's suite does; every placement of the
  sweep is generated the same way, including (0.25, 0.30, 0.60) itself (`p_d025_s30_h60`).
  `base` = the same placement on the earlier study's frozen 600-query suite, as the bridge to its numbers.
- **Training data** (`prep.py`): `prepare12.py` unchanged (500,000 pairs, 20 length bins, half of the x1 in the task box) with 2^20
  candidates per draw instead of 2^18.
- **Fields** (`pipeline12.py`, unchanged): 5,000 epochs, training seed 87358, validation every 500 epochs, then the frozen queries
  planned by ntrl-demo's MPPI (50 samples, horizon 5, 600 steps; planning seeds 0/1/2) and scored at 0.001 rad per joint by the rule
  that certified the suites (each arm by the release rule, arms 5 mm apart).
- **Controls** next to every field: the same planner on the Euclidean field `T = |x − g|` (`euclid.py`), and the straight line.
- **Scheduling** (`queue/sched.py`): several registered jobs per GPU; see "Throughput" below.

## Runs and findings

Findings page: `report/arm_placement.html` (`page.py`), published as the artifact "2026-11-23 Two-Arm Placement".
Every number below is from a registered run; `report/tables.md` (`report.py`) lists all of them, `readme_table.py` writes the tables here.

### What was run

1. **Reach** over a grid (`reach_grid`, run `b7019c90`): dx 0.25 to 1.05 m in 5 cm steps × s ∈ {0.20, 0.30, 0.46} × h ∈ {0.60, 0.80, 1.00}.
   At 1.05 m an arm reaches fewer than two zones, so no zone → another zone query exists; `suites.py` refuses such placements
   (`tests_p_d105_s30_h60`, `tests_p_d105_s20_h80` failed by that assertion). **1.00 m is the last distance with a full suite.**
2. **Sweep in distance** at s = 0.30, h = 0.60: 0.55, 0.70, 0.78, 0.85, 0.92, 1.00 m, plus `base` (0.25 m on the earlier study's frozen 600
   queries). Per placement: suite → prepare → Euclid control → ntrl-demo field and lift field (training seeds 87358 and 87359).
3. **Spacing and height**: s = 0.46 at 0.55, 0.70, 0.85, 1.00 m; s = 0.20 at 0.85 and 1.00 m; h = 0.80 at 0.85 m (s = 0.20, 0.30) and at
   1.00 m (s = 0.20, 0.30); h = 1.00 at 1.00 m (s = 0.20); the series s = 0.20, h = 0.80 at 0.85, 0.90, 0.95, 1.00 m.
   `p_d100_s46_h80` has no suite (an arm reaches fewer than two zones). The best placement has a third training seed (87360).
4. **Planners on the trained fields** (`evalfield.py`, `bidir.py`; no training): ntrl-demo's planner start → goal (`forward`), the same
   planner goal → start (`reverse`), two heads walking towards each other (`bidir`, "two-ended"), both ends to a retracted pose (`hub`),
   at 300 to 4,800 steps and 50 or 200 samples.
5. **Zone → another zone sets** (`queue/addhard.py`): 200 certified queries of that stratum only, at `p_d100_s20_h80`,
   `p_d100_s30_h60` and `p_d085_s30_h60`.
6. **Demo** (`demo.py`): ten of those queries with their planned 12-joint paths, picked among the queries that both trained lift
   fields solve with all three planning seeds (`--require`): run `0011827a` at `p_d100_s20_h80` (137 of 200 queries qualify;
   `report/demo.{json,npz}`, shown on the page) and run `384b0f7c` at `p_d100_s30_h60` (135 of 200; `report/demo_p_d100_s30_h60.{json,npz}`).
   An earlier pick without `--require` is run `02918f3e`.
7. **Controls at `p_d100_s30_h60`**: the lift at Fourier frequency 0.25 and 1.0, the forward planner with 2,400 and 4,800 steps, 200
   samples per step, the two-ended planner with 4,800 steps per end. The full LBS loss (Bellman + segment terms, run `1ffec9fb`) ran at 8 s
   per epoch next to the other jobs and was killed at epoch 1,500 of 5,000.

Not finished, on purpose: the suites of `p_d025_s30_h60` (0.25 m with this study's recipe), `p_d040_s30_h60`, `p_d055_s30_h80` and
`p_d070_s30_h80` ran for more than two hours each and were killed (close placements certify few candidates per round). Two lift runs
died from CUDA out-of-memory in their final evaluation and were not repeated (`lift_base_s87359`, `lift_p_d055_s46_h60_s87359`).

### Results

<!-- table:start -->
Solved runs of 900 (300 queries × 3 planning seeds; `base`: of 1,800), in brackets the zone → another zone runs solved of 150 (`base`: of 300), then the run id (last eight characters).
Several entries in a cell = training seeds 87358; 87359. "two-ended" = `bidir.plan_bidir`, steps per end. Reach = share of the zones’ volume at least one arm / both arms reach. Depth = deepest grasp point
behind the front face among the zone → zone test queries.

| placement | dx, s, h (m) | reach % | depth max (cm) | straight line | Euclid (control) | lift, ntrl-demo planner | lift, two-ended 600 | lift, two-ended 2,400 | ntrl-demo, own planner | ntrl-demo, two-ended 600 |
|---|---|---|---|---|---|---|---|---|---|---|
| `base` | 0.25, 0.30, 0.60 | 82 / 67 | 47 | 55/600 | 129 `c695d483` | 645 (11) `6f97b124` | 980 (81) `7865ee82` | 1149 (156) `3eadfa34` | 15 (0) `9d3d0970`; 32 (0) `0e194eb8` | 10 (0) `fd8d7f0b` |
| `p_d055_s30_h60` | 0.55, 0.30, 0.60 | 72 / 56 | 50 | 63/300 | 163 `fd5a8538` | 460 (0) `fec861d0`; 461 (2) `a7b63341` | 608 (3) `49942e44`; 586 (8) `25fe75e8` | 684 (22) `9a665a90`; 669 (26) `488eea98` | 56 (0) `1e26c000`; 27 (0) `4831fbe6` | 50 (0) `ce02e885` |
| `p_d055_s46_h60` | 0.55, 0.46, 0.60 | 69 / 41 | 50 | 59/300 | 153 `68c1178b` | 474 (5) `1a7d6e92` | 638 (20) `cf129558` | 733 (59) `51d21287` | 42 (0) `b8d04579`; 37 (0) `6020e2ee` | 45 (0) `e687f234` |
| `p_d070_s30_h60` | 0.70, 0.30, 0.60 | 53 / 35 | 38 | 71/300 | 169 `bafc89bb` | 494 (6) `044d2892`; 464 (3) `0639bd01` | 636 (19) `3f64f748`; 616 (9) `1b008647` | 718 (54) `950291f2`; 709 (63) `e576b227` | 43 (0) `92105da3`; 37 (0) `2fe25406` | 43 (0) `f1c96b83` |
| `p_d070_s46_h60` | 0.70, 0.46, 0.60 | 49 / 22 | 36 | 67/300 | 171 `33f63293` | 515 (0) `23f4a5d8`; 538 (5) `d676d213` | 698 (49) `581fb73b`; 692 (41) `f847b080` | 769 (86) `d4bae9df`; 759 (76) `af6448d4` | 65 (0) `67b6b502`; 68 (0) `135b490b` | 48 (0) `66e87ec4` |
| `p_d078_s30_h60` | 0.78, 0.30, 0.60 | 42 / 25 | 32 | 74/300 | 184 `b00e6470` | 496 (9) `5e77d15b`; 478 (16) `b87773ed` | 669 (52) `39a57d8e`; 663 (39) `68f66bc4` | 786 (107) `48cab08a`; 788 (99) `b81b28a0` | 43 (0) `d3ae0fdf` | 43 (0) `55709d04` |
| `p_d085_s20_h60` | 0.85, 0.20, 0.60 | 32 / 22 | 28 | 80/300 | 203 `3e91805d` | 488 (4) `c4dacd08`; 472 (3) `fe3eb0de` | 683 (32) `ad128471`; 688 (29) `8712d24b` | 779 (78) `cd17f178`; 785 (81) `181a4769` | 58 (0) `5ffe4bc1` | 49 (0) `1856f2d1` |
| `p_d085_s20_h80` | 0.85, 0.20, 0.80 | 33 / 22 | 27 | 69/300 | 170 `a9d2d72f` | 532 (10) `51b5338a`; 495 (7) `2b809171` | 679 (39) `42f79111`; 704 (42) `93e00fea` | 773 (101) `42f79111`; 791 (83) `93e00fea` | 68 (0) `bdb6e8ea` | 67 (0) `93037d51` |
| `p_d085_s30_h60` | 0.85, 0.30, 0.60 | 32 / 16 | 24 | 82/300 | 211 `e319d1f2` | 534 (7) `d2256e3b`; 518 (13) `101875d5` | 709 (45) `fb542980`; 714 (56) `3637d367` | 816 (104) `f450406f`; 802 (91) `2c2b3422` | 76 (0) `2cc33bd9`; 64 (0) `ecc3883c` | 62 (0) `0af986ea` |
| `p_d085_s30_h80` | 0.85, 0.30, 0.80 | 32 / 16 | 24 | 82/300 | 193 `14f24c3d` | 512 (7) `2066c8db`; 497 (2) `baace42b` | 721 (58) `a95114ea`; 715 (57) `235ecf80` | 808 (110) `441463f4`; 807 (108) `63bbe3c6` | 65 (0) `1ef89bd0` | 59 (0) `cb0fda26` |
| `p_d085_s46_h60` | 0.85, 0.46, 0.60 | 28 / 7 | 24 | 77/300 | 208 `c1aaacd0` | 485 (0) `460db164`; 476 (11) `6466c47d` | 736 (74) `5360270c`; 723 (69) `29ccf76c` | 803 (111) `ca0e72e8`; 796 (116) `7db8e45c` | 86 (0) `fea309c5` | 69 (0) `26cfa5bc` |
| `p_d090_s20_h80` | 0.90, 0.20, 0.80 | 25 / 16 | 23 | 85/300 | 204 `73693299` | 519 (3) `7f54c3a8`; 502 (3) `c023a8bc` | 722 (46) `678ca666`; 707 (52) `2dbb3012` | 813 (99) `678ca666`; 809 (115) `2dbb3012` | 58 (0) `03636365` | 56 (0) `0713a592` |
| `p_d092_s30_h60` | 0.92, 0.30, 0.60 | 22 / 9 | 21 | 69/300 | 177 `eafbb4c1` | 453 (6) `2d5cfa54`; 475 (0) `579baca4` | 747 (63) `17f6c365`; 720 (64) `2cc2c4e0` | 831 (122) `17f6c365`; 804 (114) `93401c97` | 67 (0) `f75760e2` | 70 (0) `35541141` |
| `p_d095_s20_h80` | 0.95, 0.20, 0.80 | 18 / 10 | 16 | 84/300 | 208 `c7a99056` | 511 (0) `030e36b5`; 502 (0) `3a87c255` | 776 (77) `6a572f82`; 781 (82) `b610e56d` | 867 (139) `6a572f82`; 862 (144) `b610e56d` | 69 (0) `513d3bab` | 65 (0) `326d826f` |
| `p_d100_s20_h60` | 1.00, 0.20, 0.60 | 13 / 5 | 14 | 77/300 | 193 `c528abe4` | 490 (3) `069308d6`; 517 (3) `0a8fbb3e` | 745 (57) `c1c5fe62`; 758 (69) `7f1aa02a` | 850 (134) `85f51684`; 853 (127) `4f29abd6` | 83 (0) `26116494` | 57 (0) `8a38c0ea` |
| `p_d100_s20_h80` | 1.00, 0.20, 0.80 | 13 / 5 | 13 | 84/300 | 214 `61d65a49` | 496 (1) `ae193339`; 504 (7) `18bff97d`; 514 (5) `85e1adbe` | 786 (75) `c58921c0`; 792 (75) `7de65266`; 767 (70) `2a89fe68` | 874 (143) `2623ad3f`; 876 (139) `5bde5ca5`; 868 (140) `2a89fe68` | 100 (0) `c93aa172` | 74 (0) `5df336a5` |
| `p_d100_s20_h100` | 1.00, 0.20, 1.00 | 13 / 5 | 12 | 91/300 | 231 `006b299c` | 565 (3) `d82d4024`; 511 (5) `a11c0ccc` | 803 (72) `1401a0a5`; 792 (62) `15b5d43a` | 873 (137) `1401a0a5`; 875 (135) `15b5d43a` | 67 (0) `ca80b34c` | 67 (0) `18fe94e7` |
| `p_d100_s30_h60` | 1.00, 0.30, 0.60 | 12 / 3 | 13 | 83/300 | 219 `e26e61f1` | 488 (0) `3dc70b43`; 481 (0) `c3143613` | 756 (89) `70285d61`; 767 (88) `68182942` | 850 (139) `94d32e92`; 852 (140) `4880fb13` | 102 (0) `7c2b6fac`; 99 (0) `d7c72ed2` | 69 (0) `05fb3b44` |
| `p_d100_s30_h80` | 1.00, 0.30, 0.80 | 12 / 3 | 14 | 93/300 | 220 `cacb9c20` | 514 (6) `18faa6bd`; 527 (16) `87c6b8d3` | 773 (71) `578043a4`; 786 (97) `9fce73a6` | 854 (133) `66d01d97`; 837 (125) `ecf8531a` | 81 (0) `218fe33f` | 84 (0) `61f87f18` |
| `p_d100_s46_h60` | 1.00, 0.46, 0.60 | 9 / 0 | 12 | 97/300 | 248 `f759d8a9` | 492 (2) `9e59cec6` | 839 (138) `e5f62e60` | 865 (148) `e5f62e60` | 115 (0) `955bfebf` | 112 (0) `21cccd9c` |

Other fields and planner settings (same notation):

| placement | field | training seed | planner | steps | solved | run |
|---|---|---|---|---|---|---|
| `base` | lift | 87358 | bidir | 1200 | 1075/1800 (126) | `3eadfa34` |
| `base` | lift | 87358 | forward | 600 | 645/1800 (11) | `7865ee82` |
| `base` | lift | 87358 | reverse | 600 | 681/1800 (13) | `7865ee82` |
| `base` | lift025 (`--lift time --weight 0 --freq 0.25`) | 87358 | ntrl-demo planner | 600 | 643/1800 (17) | `901ef5ee` |
| `base` | lift1 (`--lift time --weight 0`) | 87358 | ntrl-demo planner | 600 | 193/1800 (7) | `7e1ec2a0` |
| `base` | ntrl | 87358 | bidir | 1200 | 10/1800 (0) | `116148f5` |
| `base` | ntrl | 87358 | bidir | 2400 | 10/1800 (0) | `116148f5` |
| `base` | ntrl | 87358 | reverse | 600 | 28/1800 (0) | `fd8d7f0b` |
| `base` | ntrl05 (`--lift none --weight 0 --freq 0.5`) | 87358 | ntrl-demo planner | 600 | 50/1800 (0) | `e84850b7` |
| `p_d055_s30_h60` | lift | 87358 | bidir | 1200 | 642/900 (9) | `9a665a90` |
| `p_d055_s30_h60` | lift | 87358 | reverse | 600 | 463/900 (0) | `49942e44` |
| `p_d055_s30_h60` | lift | 87359 | bidir | 1200 | 630/900 (14) | `488eea98` |
| `p_d055_s30_h60` | lift | 87359 | reverse | 600 | 439/900 (1) | `25fe75e8` |
| `p_d055_s30_h60` | ntrl | 87358 | bidir | 1200 | 47/900 (0) | `805f8bf9` |
| `p_d055_s30_h60` | ntrl | 87358 | bidir | 2400 | 47/900 (0) | `805f8bf9` |
| `p_d055_s30_h60` | ntrl | 87358 | reverse | 600 | 48/900 (0) | `ce02e885` |
| `p_d055_s46_h60` | lift | 87358 | bidir | 1200 | 698/900 (42) | `51d21287` |
| `p_d055_s46_h60` | lift | 87358 | reverse | 600 | 483/900 (4) | `cf129558` |
| `p_d055_s46_h60` | ntrl | 87358 | bidir | 1200 | 45/900 (0) | `dd185a72` |
| `p_d055_s46_h60` | ntrl | 87358 | bidir | 2400 | 45/900 (0) | `dd185a72` |
| `p_d055_s46_h60` | ntrl | 87358 | reverse | 600 | 44/900 (0) | `e687f234` |
| `p_d070_s30_h60` | lift | 87358 | bidir | 1200 | 684/900 (36) | `950291f2` |
| `p_d070_s30_h60` | lift | 87358 | reverse | 600 | 490/900 (0) | `3f64f748` |
| `p_d070_s30_h60` | lift | 87359 | bidir | 1200 | 656/900 (28) | `e576b227` |
| `p_d070_s30_h60` | lift | 87359 | reverse | 600 | 471/900 (3) | `1b008647` |
| `p_d070_s30_h60` | ntrl | 87358 | bidir | 1200 | 43/900 (0) | `9b965e10` |
| `p_d070_s30_h60` | ntrl | 87358 | bidir | 2400 | 43/900 (0) | `9b965e10` |
| `p_d070_s30_h60` | ntrl | 87358 | reverse | 600 | 64/900 (0) | `f1c96b83` |
| `p_d070_s46_h60` | lift | 87358 | bidir | 1200 | 738/900 (72) | `d4bae9df` |
| `p_d070_s46_h60` | lift | 87358 | reverse | 600 | 510/900 (4) | `581fb73b` |
| `p_d070_s46_h60` | lift | 87359 | bidir | 1200 | 734/900 (58) | `af6448d4` |
| `p_d070_s46_h60` | lift | 87359 | reverse | 600 | 491/900 (4) | `f847b080` |
| `p_d070_s46_h60` | ntrl | 87358 | bidir | 1200 | 46/900 (0) | `5955c565` |
| `p_d070_s46_h60` | ntrl | 87358 | bidir | 2400 | 46/900 (0) | `5955c565` |
| `p_d070_s46_h60` | ntrl | 87358 | reverse | 600 | 74/900 (0) | `66e87ec4` |
| `p_d078_s30_h60` | lift | 87358 | bidir | 1200 | 740/900 (88) | `48cab08a` |
| `p_d078_s30_h60` | lift | 87358 | reverse | 600 | 511/900 (0) | `39a57d8e` |
| `p_d078_s30_h60` | lift | 87359 | bidir | 1200 | 725/900 (69) | `b81b28a0` |
| `p_d078_s30_h60` | lift | 87359 | reverse | 600 | 491/900 (6) | `68f66bc4` |
| `p_d078_s30_h60` | ntrl | 87358 | bidir | 1200 | 43/900 (0) | `d3f32fbb` |
| `p_d078_s30_h60` | ntrl | 87358 | bidir | 2400 | 43/900 (0) | `d3f32fbb` |
| `p_d078_s30_h60` | ntrl | 87358 | reverse | 600 | 60/900 (0) | `55709d04` |
| `p_d085_s20_h60` | lift | 87358 | bidir | 1200 | 733/900 (48) | `cd17f178` |
| `p_d085_s20_h60` | lift | 87358 | reverse | 600 | 526/900 (3) | `ad128471` |
| `p_d085_s20_h60` | lift | 87359 | bidir | 1200 | 726/900 (47) | `181a4769` |
| `p_d085_s20_h60` | lift | 87359 | reverse | 600 | 550/900 (9) | `8712d24b` |
| `p_d085_s20_h60` | ntrl | 87358 | bidir | 1200 | 47/900 (0) | `122a4bcb` |
| `p_d085_s20_h60` | ntrl | 87358 | bidir | 2400 | 46/900 (0) | `122a4bcb` |
| `p_d085_s20_h60` | ntrl | 87358 | reverse | 600 | 56/900 (0) | `1856f2d1` |
| `p_d085_s20_h80` | ntrl | 87358 | bidir | 2400 | 67/900 (0) | `93037d51` |
| `p_d085_s30_h60` | lift | 87358 | bidir | 1200 | 758/900 (74) | `f450406f` |
| `p_d085_s30_h60` | lift | 87358 | hub | 600 | 439/900 (51) | `3f4c6c7f` |
| `p_d085_s30_h60` | lift | 87358 | reverse | 600 | 494/900 (2) | `fb542980` |
| `p_d085_s30_h60` | lift | 87359 | bidir | 1200 | 762/900 (73) | `2c2b3422` |
| `p_d085_s30_h60` | lift | 87359 | reverse | 600 | 489/900 (2) | `3637d367` |
| `p_d085_s30_h60` | ntrl | 87358 | bidir | 1200 | 64/900 (0) | `6ae6d47e` |
| `p_d085_s30_h60` | ntrl | 87358 | bidir | 2400 | 62/900 (0) | `6ae6d47e` |
| `p_d085_s30_h60` | ntrl | 87358 | reverse | 600 | 63/900 (0) | `0af986ea` |
| `p_d085_s30_h80` | lift | 87358 | bidir | 1200 | 771/900 (88) | `441463f4` |
| `p_d085_s30_h80` | lift | 87358 | reverse | 600 | 463/900 (6) | `a95114ea` |
| `p_d085_s30_h80` | lift | 87359 | bidir | 1200 | 766/900 (86) | `63bbe3c6` |
| `p_d085_s30_h80` | lift | 87359 | reverse | 600 | 454/900 (2) | `235ecf80` |
| `p_d085_s30_h80` | ntrl | 87358 | bidir | 1200 | 59/900 (0) | `e7faf854` |
| `p_d085_s30_h80` | ntrl | 87358 | bidir | 2400 | 59/900 (0) | `e7faf854` |
| `p_d085_s30_h80` | ntrl | 87358 | reverse | 600 | 54/900 (0) | `cb0fda26` |
| `p_d085_s46_h60` | lift | 87358 | bidir | 1200 | 771/900 (92) | `ca0e72e8` |
| `p_d085_s46_h60` | lift | 87358 | reverse | 600 | 503/900 (1) | `5360270c` |
| `p_d085_s46_h60` | lift | 87359 | bidir | 1200 | 764/900 (93) | `7db8e45c` |
| `p_d085_s46_h60` | lift | 87359 | reverse | 600 | 505/900 (5) | `29ccf76c` |
| `p_d085_s46_h60` | ntrl | 87358 | bidir | 1200 | 67/900 (0) | `46fc190d` |
| `p_d085_s46_h60` | ntrl | 87358 | bidir | 2400 | 64/900 (0) | `46fc190d` |
| `p_d085_s46_h60` | ntrl | 87358 | reverse | 600 | 61/900 (0) | `26cfa5bc` |
| `p_d090_s20_h80` | ntrl | 87358 | bidir | 2400 | 56/900 (0) | `0713a592` |
| `p_d092_s30_h60` | lift | 87359 | bidir | 1200 | 767/900 (93) | `93401c97` |
| `p_d092_s30_h60` | lift | 87359 | reverse | 600 | 476/900 (7) | `2cc2c4e0` |
| `p_d092_s30_h60` | ntrl | 87358 | bidir | 1200 | 66/900 (0) | `dba59fc7` |
| `p_d092_s30_h60` | ntrl | 87358 | bidir | 2400 | 66/900 (0) | `dba59fc7` |
| `p_d092_s30_h60` | ntrl | 87358 | reverse | 600 | 86/900 (0) | `35541141` |
| `p_d095_s20_h80` | lift | 87358 | bidir | 4800 | 876/900 (141) | `fa481b35` |
| `p_d095_s20_h80` | lift | 87359 | bidir | 4800 | 873/900 (148) | `03b7aa6e` |
| `p_d095_s20_h80` | ntrl | 87358 | bidir | 2400 | 65/900 (0) | `326d826f` |
| `p_d100_s20_h60` | lift | 87358 | bidir | 1200 | 810/900 (104) | `85f51684` |
| `p_d100_s20_h60` | lift | 87359 | bidir | 1200 | 822/900 (102) | `4f29abd6` |
| `p_d100_s20_h60` | ntrl | 87358 | bidir | 1200 | 58/900 (0) | `f125864d` |
| `p_d100_s20_h60` | ntrl | 87358 | bidir | 2400 | 58/900 (0) | `f125864d` |
| `p_d100_s20_h80` | lift | 87358 | bidir | 1200 | 845/900 (120) | `2623ad3f` |
| `p_d100_s20_h80` | lift | 87358 | bidir | 4800 | 884/900 (148) | `1ce0918a` |
| `p_d100_s20_h80` | lift | 87359 | bidir | 1200 | 851/900 (123) | `5bde5ca5` |
| `p_d100_s20_h80` | lift | 87359 | bidir | 4800 | 881/900 (144) | `51460503` |
| `p_d100_s20_h80` | lift | 87360 | bidir | 4800 | 878/900 (148) | `665cbdc8` |
| `p_d100_s20_h80` | ntrl | 87358 | bidir | 1200 | 74/900 (0) | `e45b8ac7` |
| `p_d100_s20_h80` | ntrl | 87358 | bidir | 2400 | 74/900 (0) | `e45b8ac7` |
| `p_d100_s20_h100` | lift | 87358 | bidir | 4800 | 883/900 (143) | `95e0864b` |
| `p_d100_s20_h100` | lift | 87359 | bidir | 4800 | 889/900 (146) | `5b272a0d` |
| `p_d100_s20_h100` | ntrl | 87358 | bidir | 2400 | 67/900 (0) | `18fe94e7` |
| `p_d100_s30_h60` | lift | 87358 | bidir, 200 samples | 600 | 760/900 (86) | `e30553f9` |
| `p_d100_s30_h60` | lift | 87358 | bidir | 1200 | 804/900 (116) | `94d32e92` |
| `p_d100_s30_h60` | lift | 87358 | bidir | 4800 | 868/900 (145) | `514ae6b4` |
| `p_d100_s30_h60` | lift | 87358 | forward | 2400 | 494/900 (0) | `52e9a20c` |
| `p_d100_s30_h60` | lift | 87358 | forward | 4800 | 494/900 (0) | `52e9a20c` |
| `p_d100_s30_h60` | lift | 87358 | reverse | 600 | 481/900 (0) | `70285d61` |
| `p_d100_s30_h60` | lift | 87359 | bidir | 1200 | 827/900 (127) | `4880fb13` |
| `p_d100_s30_h60` | lift | 87359 | reverse | 600 | 470/900 (2) | `68182942` |
| `p_d100_s30_h60` | lift025 (`--lift time --weight 0 --freq 0.25`) | 87358 | ntrl-demo planner | 600 | 488/900 (1) | `257ed2bc` |
| `p_d100_s30_h60` | lift025 | 87358 | bidir | 600 | 769/900 (84) | `1fc0527f` |
| `p_d100_s30_h60` | lift025 | 87358 | bidir | 2400 | 868/900 (139) | `1fc0527f` |
| `p_d100_s30_h60` | lift1 (`--lift time --weight 0`) | 87358 | ntrl-demo planner | 600 | 194/900 (0) | `22989754` |
| `p_d100_s30_h60` | lift1 | 87358 | bidir | 600 | 238/900 (16) | `50aa6b81` |
| `p_d100_s30_h60` | lift1 | 87358 | bidir | 2400 | 267/900 (34) | `50aa6b81` |
| `p_d100_s30_h60` | ntrl | 87358 | bidir | 1200 | 69/900 (0) | `0dfbeaf7` |
| `p_d100_s30_h60` | ntrl | 87358 | bidir | 2400 | 68/900 (0) | `0dfbeaf7` |
| `p_d100_s30_h60` | ntrl | 87358 | reverse | 600 | 108/900 (0) | `05fb3b44` |
| `p_d100_s30_h80` | lift | 87358 | bidir | 1200 | 824/900 (109) | `66d01d97` |
| `p_d100_s30_h80` | lift | 87359 | bidir | 1200 | 817/900 (115) | `ecf8531a` |
| `p_d100_s30_h80` | ntrl | 87358 | bidir | 1200 | 81/900 (0) | `f12dd5bc` |
| `p_d100_s30_h80` | ntrl | 87358 | bidir | 2400 | 81/900 (0) | `f12dd5bc` |
| `p_d100_s46_h60` | ntrl | 87358 | bidir | 2400 | 107/900 (0) | `21cccd9c` |

Zone → another zone sets (200 queries × 3 planning seeds = 600 runs):

| placement | field | training seed | planner | steps | solved of 600 | queries solved by at least one / by all three planning seeds | did not arrive | arrived in contact | run |
|---|---|---|---|---|---|---|---|---|---|
| `p_d085_s30_h60` | lift | 87358 | bidir | 2400 | 406 | 155 / 111 | 182 | 12 | `53eb30a8` |
| `p_d085_s30_h60` | ntrl | 87358 | bidir | 2400 | 0 | 0 / 0 | 62 | 538 | `3b7f6a55` |
| `p_d100_s20_h80` | lift | 87358 | bidir | 2400 | 547 | 194 / 166 | 41 | 12 | `2637cdcf` |
| `p_d100_s20_h80` | lift | 87358 | bidir | 4800 | 567 | 195 / 180 | 21 | 12 | `f56c1386` |
| `p_d100_s20_h80` | lift | 87359 | bidir | 2400 | 542 | 193 / 163 | 41 | 18 | `6e991378` |
| `p_d100_s20_h80` | lift | 87359 | bidir | 4800 | 572 | 195 / 183 | 10 | 18 | `51814880` |
| `p_d100_s20_h80` | ntrl | 87358 | bidir | 2400 | 0 | 0 / 0 | 10 | 590 | `84a48bf0` |
| `p_d100_s30_h60` | lift | 87358 | bidir | 600 | 307 | 133 / 70 | 276 | 18 | `cecebb27` |
| `p_d100_s30_h60` | lift | 87358 | bidir | 2400 | 537 | 191 / 161 | 44 | 19 | `cecebb27` |
| `p_d100_s30_h60` | lift | 87358 | forward | 600 | 7 | 3 / 2 | 597 | 0 | `cecebb27` |
| `p_d100_s30_h60` | lift | 87358 | forward | 2400 | 7 | 3 / 2 | 596 | 0 | `cecebb27` |
| `p_d100_s30_h60` | lift | 87359 | bidir | 600 | 273 | 120 / 61 | 312 | 16 | `9732193c` |
| `p_d100_s30_h60` | lift | 87359 | bidir | 2400 | 534 | 189 / 166 | 50 | 17 | `9732193c` |
| `p_d100_s30_h60` | lift | 87359 | forward | 600 | 7 | 3 / 2 | 598 | 0 | `9732193c` |
| `p_d100_s30_h60` | lift | 87359 | forward | 2400 | 7 | 3 / 2 | 596 | 0 | `9732193c` |
| `p_d100_s30_h60` | ntrl | 87358 | bidir | 600 | 0 | 0 / 0 | 17 | 583 | `7b340e55` |
| `p_d100_s30_h60` | ntrl | 87358 | bidir | 2400 | 0 | 0 / 0 | 15 | 585 | `7b340e55` |
| `p_d100_s30_h60` | ntrl | 87358 | forward | 600 | 0 | 0 / 0 | 36 | 564 | `7b340e55` |
| `p_d100_s30_h60` | ntrl | 87358 | forward | 2400 | 0 | 0 / 0 | 31 | 569 | `7b340e55` |

Lift field, two-ended planner, 2,400 steps per end, by the depth of the query’s deepest hand (solved / runs; all training seeds evaluated at that budget):

| placement | no arm inside | up to 7 cm inside | 7 to 13 cm | 13 to 25 cm | deeper than 25 cm |
|---|---|---|---|---|---|
| `base` | 96/150 = 64 % | 84/117 = 72 % | 115/195 = 59 % | 532/864 = 62 % | 322/474 = 68 % |
| `p_d055_s30_h60` | 397/432 = 92 % | 38/42 = 90 % | 264/312 = 85 % | 394/540 = 73 % | 260/474 = 55 % |
| `p_d055_s46_h60` | 202/210 = 96 % | 49/57 = 86 % | 128/156 = 82 % | 228/297 = 77 % | 126/180 = 70 % |
| `p_d070_s30_h60` | 489/504 = 97 % | 75/78 = 96 % | 269/324 = 83 % | 466/636 = 73 % | 128/258 = 50 % |
| `p_d070_s46_h60` | 510/528 = 97 % | 102/102 = 100 % | 230/294 = 78 % | 525/654 = 80 % | 161/222 = 73 % |
| `p_d078_s30_h60` | 505/534 = 95 % | 70/90 = 78 % | 325/378 = 86 % | 600/708 = 85 % | 74/90 = 82 % |
| `p_d085_s20_h60` | 539/564 = 96 % | 101/114 = 89 % | 468/534 = 88 % | 442/558 = 79 % | 14/30 = 47 % |
| `p_d085_s20_h80` | 530/558 = 95 % | 115/126 = 91 % | 441/522 = 84 % | 447/552 = 81 % | 31/42 = 74 % |
| `p_d085_s30_h60` | 545/564 = 97 % | 159/168 = 95 % | 494/564 = 88 % | 402/486 = 83 % | 18/18 = 100 % |
| `p_d085_s30_h80` | 510/528 = 97 % | 118/138 = 86 % | 474/540 = 88 % | 492/570 = 86 % | 21/24 = 88 % |
| `p_d085_s46_h60` | 540/570 = 95 % | 199/210 = 95 % | 451/540 = 84 % | 409/480 = 85 % | 0/0 |
| `p_d090_s20_h80` | 552/582 = 95 % | 199/210 = 95 % | 548/630 = 87 % | 323/378 = 85 % | 0/0 |
| `p_d092_s30_h60` | 525/546 = 96 % | 343/378 = 91 % | 593/672 = 88 % | 174/204 = 85 % | 0/0 |
| `p_d095_s20_h80` | 563/582 = 97 % | 335/348 = 96 % | 695/720 = 97 % | 136/150 = 91 % | 0/0 |
| `p_d100_s20_h100` | 587/594 = 99 % | 377/390 = 97 % | 778/810 = 96 % | 6/6 = 100 % | 0/0 |
| `p_d100_s20_h60` | 570/588 = 97 % | 312/330 = 95 % | 813/870 = 93 % | 8/12 = 67 % | 0/0 |
| `p_d100_s20_h80` | 872/882 = 99 % | 745/765 = 97 % | 984/1035 = 95 % | 17/18 = 94 % | 0/0 |
| `p_d100_s30_h60` | 569/594 = 96 % | 372/384 = 97 % | 737/798 = 92 % | 24/24 = 100 % | 0/0 |
| `p_d100_s30_h80` | 562/582 = 97 % | 482/510 = 95 % | 626/684 = 92 % | 21/24 = 88 % | 0/0 |
| `p_d100_s46_h60` | 291/297 = 98 % | 254/258 = 98 % | 317/342 = 93 % | 3/3 = 100 % | 0/0 |
<!-- table:end -->

### Findings

Written by `readme_table.py` from `report/notes.json` (the findings page's text, which `notes.py` computes from the registered runs; run ids are in the tables above).

<!-- findings:start -->
1. **Best placement.** **1.00 m** from the cupboard front, bases ±0.20 m, 0.80 m high: 97 % solved (874, 876, 868 of 900 runs on 3 training seeds). With twice the planner budget (4,800 steps per end): 884, 881, 878 of 900. The other placements at that distance: ±0.20 m, 0.60 m high 95 %; ±0.20 m, 1.00 m high 97 %; ±0.30 m, 0.60 m high 95 %; ±0.30 m, 0.80 m high 94 %; ±0.46 m, 0.60 m high 96 %. The 1.00 m high base at ±0.20 m ties it (873, 875 of 900); 0.80 m is named here because it has more training seeds, the 200-query set and the demo. It is the reach limit: either arm still reaches 13 % of the zone volume (bottom 5 %, middle 26 %, top 8 %); at 1.05 m, and at 1.00 m with the bases ±0.46 m apart and 0.80 m high, an arm reaches fewer than two zones.
2. **The price.** At 1.00 m the grasp point is 6 cm inside the cupboard in the median zone → zone query and never more than 13 cm (shelves: 53 cm deep). At 0.95 m, bases ±0.20 m, 0.80 m high it goes up to 16 cm in, an arm reaches 18 % of the zone volume, and the lift field solves **96 %** (94 % of zone → zone). At 0.85 m, bases ±0.30 m, 0.60 m high it goes up to 24 cm in, an arm reaches 32 % of the zone volume, and the lift field solves **90 %** (65 % of zone → zone). If the shelves’ depth matters, move the bases closer: five centimetres cost about one point, ten to fifteen about seven.
3. **Does moving back help?** Yes. Solved share with the lift field and ntrl-demo’s planner: 51 % at 0.55 m, 53 % at 0.70 m, 54 % at 0.78 m, 58 % at 0.85 m, 52 % at 0.92 m, 54 % at 1.00 m (0.25 m, earlier suite: 36 %). With the two-ended planner at 2,400 steps per end: 75 % at 0.55 m, 79 % at 0.70 m, 87 % at 0.78 m, 90 % at 0.85 m, 91 % at 0.92 m, 95 % at 1.00 m.
4. **Inside the cupboard to inside the cupboard.** Zone → another zone at 1.00 m: 3 % with ntrl-demo’s planner, **94 %** with the two-ended planner (2,400 steps per end). On the larger set of 200 such queries: 1,089 of 1,200 runs. With 4,800 steps per end: 1,139 of 1,200. At 0.85 m, bases ±0.30 m, 0.60 m high: 406 of 600. At 1.00 m, bases ±0.30 m, 0.60 m high: 1,071 of 1,200.
5. **The two models.** Lift field (LBS, lift only, half Fourier frequency): 56 % with ntrl-demo’s planner, 97 % two-ended. ntrl-demo field: 11 % and 8 %. The ntrl-demo field arrives in contact on most runs at every placement; distance helps it too, from 1 % to 11 %.
6. **What made the difference.** The planner first: the same lift field goes from 36 % to 64 % at 0.25 m and from 54 % to 95 % at 1.00 m when it is used from both ends. The field is symmetric, yet its planner leaves a zone far more reliably than it enters one. Then the distance, and part of that gain is that far placements only have shallow queries: 85 % at 0.55 m, 83 % at 0.70 m, 86 % at 0.78 m, 88 % at 0.85 m, 88 % at 0.92 m, 92 % at 1.00 m of the queries 7 to 13 cm inside the cupboard are solved.

**Free space and reach.** Why it gets easier. Close to the cupboard, most arm poses with the hands in the task box collide with a panel or with the other arm: 7.8 % are collision-free at 0.55 m, 10.8 % are collision-free at 0.70 m, 12.3 % are collision-free at 0.78 m, 13.3 % are collision-free at 0.85 m, 14.0 % are collision-free at 0.92 m, 17.5 % are collision-free at 1.00 m (0.8 % at 0.25 m). Far away the arms enter almost straight, with only the forearm and hand inside, and they cross each other less. The price is reach: the blue line in the chart is the share of the three zones’ volume that at least one arm can still put its grasp point in.

**Depth, and the control at equal depth.** Reach as a volume hides how shallow the far placements work. Depth of the grasp point behind the cupboard’s front face in the zone → another zone test queries: median 13 cm at 0.55 m, 10 cm at 0.70 m, 9 cm at 0.78 m, 8 cm at 0.85 m, 7 cm at 0.92 m, 6 cm at 1.00 m; deepest 50 cm at 0.55 m, 38 cm at 0.70 m, 32 cm at 0.78 m, 24 cm at 0.85 m, 21 cm at 0.92 m, 13 cm at 1.00 m (0.25 m, earlier suite: median 15 cm, deepest 47 cm).

Control: is the task easier at a distance, or are the queries only shallower? The second chart compares placements at equal depth (each query filed by its deepest hand at either end). Queries up to 7 cm inside: 90 % at 0.55 m, 96 % at 0.70 m, 78 % at 0.78 m, 95 % at 0.85 m, 91 % at 0.92 m, 97 % at 1.00 m. Queries 7 to 13 cm inside: 85 % at 0.55 m, 83 % at 0.70 m, 86 % at 0.78 m, 88 % at 0.85 m, 88 % at 0.92 m, 92 % at 1.00 m. Queries 13 to 25 cm inside: 73 % at 0.55 m, 73 % at 0.70 m, 85 % at 0.78 m, 83 % at 0.85 m, 85 % at 0.92 m.

**Which queries.** With ntrl-demo’s planner (left) distance helps the queries that end in free space and does nothing for those that end inside the cupboard. At 1.00 m, bases ±0.30 m, 0.60 m high the lift field solves 87 % of zone → free and 7 % of free → zone, although T(a, b) = T(b, a). Run from the goal to the start, the same planner gives the mirror image (80 % and 10 %). So the field knows the way out of a zone and the planner loses the last stretch on the way in. The two-ended planner (right) only ever asks for the way out.

**The two-ended planner.** `bidir.plan_bidir`: one head starts at the start, one at the goal. In every iteration each head takes one ordinary MPPI step of ntrl-demo’s planner towards the other head’s current position. They stop when they meet; the path is the first head’s path followed by the second one’s reversed. Same field, same step, same 50 samples, no collision checker. At 1.00 m, bases ±0.30 m, 0.60 m high: ntrl-demo’s planner, 600 steps 54 %; two-ended, 300 steps per end (the same 600 moves) 77 %; two-ended, 600 per end 85 %; 1,200 per end 91 %; 2,400 per end 95 %; 4,800 per end 96 %. More steps do not rescue the one-ended planner: 55 % at 2,400 steps, 55 % at 4,800 steps. More samples per step do not help either (84 % with 200 samples at 600 steps per end). A third variant, both ends walking to a retracted pose chosen by the field, was worse (49 % at 0.85 m against 79 % two-ended).

**Spacing and height.** Both bases stay mirror images about the cupboard’s centre plane. Two-ended planner, 2,400 steps per end, mean of the training seeds: 0.55 m, ±0.46 m, 0.60 m high: 81 %; 0.70 m, ±0.46 m, 0.60 m high: 85 %; 0.85 m, ±0.20 m, 0.60 m high: 87 %; 0.85 m, ±0.20 m, 0.80 m high: 87 %; 0.85 m, ±0.30 m, 0.80 m high: 90 %; 0.85 m, ±0.46 m, 0.60 m high: 89 %; 0.90 m, ±0.20 m, 0.80 m high: 90 %; 0.95 m, ±0.20 m, 0.80 m high: 96 %; 1.00 m, ±0.20 m, 0.60 m high: 95 %; 1.00 m, ±0.20 m, 0.80 m high: 97 %; 1.00 m, ±0.20 m, 1.00 m high: 97 %; 1.00 m, ±0.30 m, 0.80 m high: 94 %; 1.00 m, ±0.46 m, 0.60 m high: 96 %. At 1.00 m the placements differ by a few points, on different sets of 300 queries each; a difference of two to three points between two of them is about what the query sets alone can produce.

**The ten configurations.** Ten zone → another-zone queries at 1.00 m, bases ±0.20 m, 0.80 m high, both arms inside the cupboard at the start and at the goal, each arm ending in a different zone from where it started. All ten are solved with all three planning seeds by both lift fields that were trained there (training seeds 87358 and 87359; two-ended planner, 2,400 steps per end), and each plan was re-checked at 0.001 rad. The ntrl-demo field solves none of them. The planner draws random samples, so a repeat is not identical: the two saved lift models, loaded on their own and run on just these ten with fresh draws (`use_model.py`), solved 10 and 9 of them. The plans are valid, not short: the heads wander, and a plan is two to seven times as long as the straight joint distance. They are the ten hardest by this order: straight line blocked, both arms change zone, longest joint distance, at most two per zone pattern.

**Setup notes and controls.** - **Two models:** my reading of the request, not confirmed: ntrl-demo (cloned from github.com/ruiqini/ntrl-demo, commit `5ed065f`, unmodified) and the repository’s LBS field in its best two-arm setting from `jeffrey/lbs_arm` (travel-time lift, half Fourier frequency, no Bellman or segment term). 5,000 epochs, training seeds 87358 and 87359.
- **Scene:** the measured cupboard and both UR5e with grippers, as in `jeffrey/real_world`. A run is solved when every configuration of the path, checked every 0.001 rad per joint, keeps 5 mm from the cupboard, floor and pedestals, 5 mm between the arms and 2 mm self-clearance.
- **Test queries:** per placement, six kinds × 50, each certified by a witness path (straight line, RRT-Connect in 12 joints, or one arm at a time). The witness planner ran 3,000 iterations per stage instead of the earlier study’s 10,000, so these sets lean somewhat more towards queries a sampling planner solves quickly. The 0.25 m point uses the earlier study’s frozen 600 queries. Its suite with this study’s recipe, 0.40 m, and the 0.80 m height at 0.55 and 0.70 m were stopped: their suites took over two hours each.
- **Different queries at every placement.** A placement changes which queries exist, so the comparison across distances is between tasks defined the same way, not between identical queries.
- **Reproduction check:** on the earlier study’s placement and frozen suite the lift field solves 645 of 1,800 (earlier: 673 and 639 on two seeds), ntrl-demo 15.
- **Controls on the lift at 1.00 m, bases ±0.30 m, 0.60 m high:** Fourier frequency 0.25 instead of 0.5: 54 % with ntrl-demo’s planner, 96 % two-ended (0.5: 54 % and 95 %). Frequency 1.0: 22 % with ntrl-demo’s planner. The full LBS loss (Bellman and segment terms) took 8 seconds per epoch next to the other jobs and was stopped at epoch 1,500 of 5,000; its validation counts there were 141, 151, 150 of 300 at epochs 500, 1,000, 1,500 against 163, 163, 158 for the lift alone.
- **Training length:** the lift field’s validation count is flat from epoch 500 on (163 of 300 at epoch 500, 165 at epoch 5,000, at 1.00 m). The 5,000 epochs were kept so that every run matches the earlier study.
- **Planner steps:** a step is 0.0615 cells, about 0.03 rad of joint motion. 2,400 steps per end is eight times the moves of the default budget; planning 300 queries that way took 80 seconds on a shared card.
<!-- findings:end -->

### Saved models and files

- `models/<placement>/{lift,ntrl}_s<seed>.pt`: weights of every finished field (`model_state_dict`, `B_state_dict`, `embed_config`, `epoch`),
  with the placement's `scene.json`, `robot_model.npz` and `queries.npz`. `models/best/`: the full checkpoints (with optimizer state),
  scene and queries of the placement named best. `models/index.json`: run ids and solved counts. Written by `finalize.py`.
- `use_model.py`: loads one of those files and plans with it (two-ended or forward), scoring every plan at 0.001 rad.
- `queue/picks.py` + `report/demo_p_d055_s46_h60.json/.npz`: a third set of ten pairs, asked for on 2026-11-23 18:00 UTC, at 0.55 m,
  bases ±0.46 m, 0.60 m high (81 % of test runs solved there). Picked from that placement's own 100 test queries with both arms inside
  the cupboard at both ends (50 zone → another zone, 50 inside one zone) by the same order as before (straight line blocked, arms
  changing zone, joint distance, at most two per zone pattern). Only one lift model exists there (training seed 87358; the second seed
  died from out-of-memory during the main session), so "works" means all three planning seeds of that one model, two-ended planner,
  2,400 steps per end, re-checked at 0.001 rad: 60 of the 100 queries qualify (per seed 64, 69, 65), the ntrl-demo field (control)
  solves 1 of 100. Runs: lift `f4d88dc9`, ntrl-demo `fbc7641b`, picking `6af9ff33`. All ten are zone → another zone with the straight
  line blocked and both arms changing zone; the deepest hand is 7 to 30 cm inside. Reload check with fresh draws (`use_model.py`,
  planning seeds 3 and 4, not registered): 9 of 10 and 9 of 10; ntrl-demo field 0 of 10.

  | # | query | left arm | right arm | straight distance (rad) | plan length, seed 0 (rad) | lift solved | ntrl-demo solved |
  |---|---|---|---|---|---|---|---|
  | 1 | 182 | top → middle | bottom → top | 5.60 | 11.0 | 3/3 | 0/3 |
  | 2 | 181 | middle → bottom | top → middle | 5.52 | 36.6 | 3/3 | 0/3 |
  | 3 | 195 | top → middle | bottom → middle | 5.44 | 25.9 | 3/3 | 0/3 |
  | 4 | 152 | bottom → top | top → middle | 5.28 | 22.7 | 3/3 | 0/3 |
  | 5 | 154 | top → middle | bottom → top | 5.26 | 78.8 | 3/3 | 0/3 |
  | 6 | 179 | top → middle | top → bottom | 5.24 | 15.4 | 3/3 | 0/3 |
  | 7 | 178 | bottom → top | middle → bottom | 5.19 | 9.2 | 3/3 | 0/3 |
  | 8 | 153 | top → bottom | top → middle | 5.18 | 35.3 | 3/3 | 0/3 |
  | 9 | 199 | bottom → middle | top → middle | 4.91 | 6.5 | 3/3 | 0/3 |
  | 10 | 191 | top → middle | top → bottom | 4.80 | 34.5 | 3/3 | 0/3 |

- `viewer.py` + `viewer_template.html`: build the 3-D viewer in `report/viewer/` (published as "2026-11-23 Arm Placement Viewer",
  https://claude.ai/artifact/3Ms5y25wahMnPrqHAw3pUF). It shows every placement (26 scene files: 20 with trained fields, 6 without test
  queries) with the cupboard fixed and the bases moving, the test queries of each placement with a saved plan, and the two sets of ten
  picked configurations with their nine saved plans each. The arms are drawn as the collision spheres of the checker; the page's
  kinematics agree with `arm12` to 0.008 mm (four random configurations, checked in node). The plans per placement come from one more
  registered evaluation each (`queue/addviewer.py`: lift training seed 87358 and the ntrl-demo field as control, two-ended planner,
  2,400 steps per end, planning seed 0, paths saved): over the 20 placements the lift field solves 5,488 of 6,300 queries and the
  ntrl-demo field 398. Run ids: base `8a0e8960`; p_d055_s30_h60 `5ec97b96`; p_d055_s46_h60 `123e24ee`; p_d070_s30_h60 `f215352f`;
  p_d070_s46_h60 `d0e48b61`; p_d078_s30_h60 `45634a46`; p_d085_s20_h60 `c0a966b7`; p_d085_s20_h80 `59fada94`; p_d085_s30_h60 `e692d261`;
  p_d085_s30_h80 `92ff0ba8`; p_d085_s46_h60 `2f851b9d`; p_d090_s20_h80 `71acb723`; p_d092_s30_h60 `49f614af`; p_d095_s20_h80 `eeac0c90`;
  p_d100_s20_h100 `b502f3be`; p_d100_s20_h60 `cf90b176`; p_d100_s20_h80 `2df652b0`; p_d100_s30_h60 `6fff3582`; p_d100_s30_h80 `bead6d5f`;
  p_d100_s46_h60 `746f483a`. `viewer.py` recomputes every plan's verdict from the checker's samples and asserts that it equals the
  registered run's flag (it did for all 12,000 placement plans and all 180 plans of the picked configurations). The clearance strip
  leaves out the two fixed shoulder spheres, which always sit about 10 mm above their pedestal.
  Check (not registered; the planner draws random samples, so a repeat on ten queries alone is not the registered run's plan):
  `models/p_d100_s20_h80/lift_s87358.pt` solves 10 of the ten demo queries, `lift_s87359.pt` 9, `ntrl_s87358.pt` 0 (planning seed 0);
  `models/p_d100_s30_h60/lift_s87358.pt` solved 10 of 10 of that placement's demo set.
- `report/demo.npz` (`start_rad`, `goal_rad`: (10, 12) radians, [left arm 6, right arm 6]; the planned paths of every planning seed and
  field) and `report/demo.json` (zones, clearances, skeleton key frames); the same for the second set in
  `report/demo_p_d100_s30_h60.*`. `report/depth.json` (`depth.py`): depth of the grasp points and the solved share by depth.
- To rebuild the report: `report.py` → `finalize.py --best p_d100_s20_h80 --demo results/registered_20261123T132742Z-0011827a` →
  `depth.py` → `notes.py --best p_d100_s20_h80` → `page.py` → `readme_table.py`.
- `scenes/<placement>.yml`: every scene that was written.

### Throughput on the two 48 GB cards

- One training process leaves a 4090 mostly idle. A small double-backward loop (4,000 points per step) as 1 / 6 / 12 / 24 processes on
  one card that was also generating suites: 103 / 67 / 37 / 23 steps per second each, so 103 / 404 / 446 / 550 in total. The scheduler
  (`queue/sched.py`) therefore runs 20 to 30 registered jobs at once, each with a load share and a VRAM reservation (three jobs died
  from out-of-memory before the reservation existed).
- CUDA MPS made the mix slower (6 processes: 36 steps per second each with MPS, 67 without) and was switched off.
- Batches: the pair sampler draws 2^20 candidates per round (prepare: 0.6 to 5 minutes, 8 minutes in the earlier study); the suite
  generator batches all strata's RRT queries (about 1,400 per round). Certifying test queries was the slow part and is bound by memory
  bandwidth (collision checks at 0.001 rad), not by batch size.

### Limits

- **"These 2 models" is my reading** (ntrl-demo and the lift field); the user did not confirm it.
- Every placement has its own test queries, generated by one recipe. The comparison across placements is between tasks defined the
  same way, not between identical queries, and far placements have shallower queries (depth table above).
- The suites used 3,000 RRT-Connect iterations per stage, not 10,000. `base` (the frozen suite, 10,000 iterations, 100 per stratum) is
  therefore a harder set than the others; `p_d025_s30_h60` with this study's recipe did not finish.
- The two-ended planner's plans are valid but long (two to seven times the straight joint distance in the demo); nothing shortens them.
- One to three training seeds per placement; with ntrl-demo's planner two seeds differ by up to 54 runs of 900, with the two-ended
  planner at 2,400 steps by up to 27 (0.92 m) and usually by fewer than 15.
- The placements at 1.00 m are within three points of each other on different query sets; `p_d100_s20_h80` (872.7 of 900 over three
  seeds) and `p_d100_s20_h100` (874 over two) are tied.

## Re-creation of `p_d055_s46_h60` on the workstation (2026-11-23, local)

Request: "go into import/arm_placement_export_2026-11-23 and re-create the results for 0.55 m · ±0.46 m · 0.60 m high".
This folder was copied from `import/arm_placement_export_2026-11-23/code/jeffrey/arm_placement` and runs here on the RTX 3090 (GPU 1),
host venv, local registry. Every declared input hashes the same as in the original runs (58 files; ntrl-demo `5ed065f`, robot geometry,
scene `7017283e…`). The working `suites.py` and `prep.py` changed after the original runs, so `run.py --snap <hash>` (new) runs an
existing snapshot, checked against its own content hash; every run below used the original run's snapshot and arguments (the suite's
config tag `dc08ee7788` equals the original's). Scripts: `repro/chain.py` (suite → prepare → fields → planners → picks; `--frozen` = on
the exported prepared data), `repro/control_exported.sh`, `repro/compare.py` (table → `repro/compare.md`).

Four columns: the original runs; the **exported checkpoints re-planned here** on the original queries (planner and hardware only);
the fields **retrained here on the original training data and queries** (training seed 87358); and the **whole chain from scratch here**
(new suite, new training data). Solved of 900 runs, zone → another zone runs solved of 150 in brackets.

| p_d055_s46_h60 | original (4090) | original checkpoints re-planned here | retrained here, original data + queries | from scratch here (new suite) |
|---|---|---|---|---|
| Euclid (control) | 153 | | 151 | 210 |
| lift, ntrl-demo planner 600 | 474 (5) | 475 (6) | 464 (2) | 527 (0) |
| lift, reverse 600 | 483 (4) | 490 (5) | 504 (2) | 450 (2) |
| lift, two-ended 600 | 638 (20) | 650 (23) | 671 (27) | 627 (27) |
| lift, two-ended 1,200 | 698 (42) | 704 (43) | 704 (35) | 674 (38) |
| **lift, two-ended 2,400** | **733 (59)** | **742 (60)** | **741 (53)** | **712 (62)** |
| ntrl-demo, own planner 600 | 42 (0) | 42 (0) | 40 (0) | 54 (0) |
| ntrl-demo, two-ended 2,400 | 45 (0) | 47 (0) | 39 (0) | 61 (0) |
| lift seed 87359, ntrl-demo planner / two-ended 2,400 | (died, out of memory) | | | 518 (5) / 691 (43) |
| ntrl-demo seed 87359, own planner 600 | 37 (0) | | | 27 (0) |
| hard set (100 in-cupboard queries): solved 3/3 by lift / by ntrl-demo | 60 / 1 | | 58 / 1 | 65 / 0 |

Findings:
1. **The result reproduces.** Lift with the two-ended planner at 2,400 steps: 733 originally, 742 / 741 / 712 here; ntrl-demo stays
   at 39–61 in every column. The gap between the two methods (≈ 690 runs of 900) is the same in all four.
2. **Not bit-identical, on three levels.** (a) Same checkpoint, same planning seed: the planner gives the same count twice on one card
   (733 and 733 in the original), but 1–12 runs more on the 3090 (GPU arithmetic). (b) Same seed and inputs, retrained: the
   initialization and Fourier matrix are equal (to 6e-8), the trained weights are not (max difference 0.27, weights up to 1.9), because
   training draws fresh samples on the GPU and its arithmetic differs; the retrained fields land within 10–33 runs, inside the study's
   own seed spread (up to 54 with ntrl-demo's planner). (c) The suite: same seeds and the same candidate pools (identical counts per
   stratum), but the GPU RRT-Connect witnesses certify a slightly different subset (zone → zone 176 here vs 169), so the 50 drawn per
   stratum differ: **88 of 300 queries are shared**. The new set has more straight-line-feasible queries (69 vs 59) and a stronger
   Euclid control (210 vs 153), i.e. it is somewhat easier for the forward planner, while the lift two-ended count is 21 lower.
3. **The ten picks belong to one model.** The original ten were chosen as queries the original lift model solves with all three
   seeds. That model, re-planned here, still solves all ten 3/3 (30 of 30). The field retrained on the same inputs solves only 3 of
   them 3/3 (17 of 30 runs); its own pick (`fc914429`) shares 3 queries with the original ten (195, 199, 191). The share of the
   hard set that qualifies is stable (60, 58, 65 of 100); which queries qualify is not. ntrl-demo solves none of any of the three sets.

Runs (local registry). Suite `b687b76d`, prepare `11356432`, Euclid `e5462df2`, fields lift `089d42cd` / `357d53ee` (87359),
ntrl `e343a228` / `910f5cd9` (87359), planners lift `33106626` `78ffc5d5` `2ada4e29` (87359), ntrl `d67cd04c` `e62c527f`,
picks evaluations `1991fced` `31202de5`, picks `e274f0e8`. Original data: Euclid `389ddb9c`, lift `e96c3257`, ntrl `2401133c`,
planners `daebfa31` `c33852e1` `36f8b67e` `fda053f2`, picks evaluations `a4632d02` `c1dc078c`, picks `fc914429` (a first try,
`c53cc01a`, failed after picking: the exported `scene.json` names `/workspace/...`; rerun on `repro/prep_frozen_local`, the same files
with that one path pointed at the local scene file of the same hash). Exported checkpoints re-planned: `d5ddd746` `a2a5b797`
`b72137ec` (lift), `d21e7b39` `1c5a71c0` `b58e5c53` (ntrl-demo).
Each training took 15–39 min alone on the 3090 (47–79 min on the shared 4090); the suite took 21 min (62).

Viewer of the re-creation: `repro/viewer_repro.py` builds `repro/viewer/` (the study's viewer with this placement three times: original,
retrained here on the original data, re-created from scratch; per-query plans from the `picks_*` runs, planning seed 0, every verdict
asserted equal to the registered flag). Published as "2026-11-24 Arm Placement Re-creation", https://claude.ai/artifact/Lv34E6FyXPhTJoE6TP2jBB.
