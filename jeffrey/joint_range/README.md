# Joint-range test: ntrl-demo on in-area cabinet pairs, full-range vs data-range input scaling

*2026-10-31, RTX 3090 and RTX 5070, Docker image `bellman600-env:local`.*

**Question (user).** Regenerate the cabinet training pairs so that the end effector is in the test area, and make two
datasets from them:

- **Test 1:** the full range of motion of every joint maps onto [−0.5, 0.5] (the release normalization: ±2π for five
  joints, ±π for the elbow).
- **Test 2:** each joint's actual range of motion in the sample data, [min, max], maps onto [−0.5, 0.5].

Train the baseline ntrl-demo on both, on the cabinet scene, with the MPPI planner scaled properly, and compare the
success rates.

## Data (`data.py`, run `772f8677`)

This is the ntrl-demo baseline recipe (`experiments/ntrl_cabinet/data.py`, imported unchanged; its defaults reproduce
the registered cabinet-metric data) with one addition:
- **x0:** uniform over the joint limits, with the grasp point in the scene's sampling box (the test area; every
  frozen test endpoint lies in it). It must be collision-checked and near an obstacle: 0 < d − 0.01 < 0.05, with
  d = min(obstacle distance, 0.5 × self distance).
- **x1:** x0 plus a random step of up to 0.5 release units, inside the joint limits. **New:** its grasp point must
  also be in the sampling box. As in the recipe, x1 is labelled, not rejected, when it touches something (24 % do).
- **Size:** 400,000 pairs from 114.8M uniform draws, in 42 s. The gripper is inside the cupboard for 0.7 % of x0 and
  2.7 % of x1.

**The data's range of motion is the full range.** Over all 800k configurations, every joint's min/max reaches its
limits (zoom = full width / data width):

| Joint | min (rad) | max (rad) | Zoom, all points (Test 2) | Zoom, collision-checked x0 only (Test 2b) |
|---|---:|---:|---:|---:|
| shoulder pan | −6.283 | 6.283 | 1.00001 | 1.0000 |
| shoulder lift | −6.283 | 6.283 | 1.00001 | 1.0000 |
| elbow | −3.142 | 3.142 | 1.00002 | **1.137** (±2.763 rad) |
| wrist 1–3 | −6.283 | 6.283 | 1.00000–1.00002 | 1.0000 |

The gripper box does not narrow the joint ranges. Each joint has two windings in ±2π, the wrists barely move the
grasp point, and the shoulder can reach the box from in front or over the top. The pan joint does leave about 14 % of
its range unused, but in scattered gaps, so its min/max is still ±2π. The only real restriction is the elbow among
collision-free configurations: past ±2.76 rad the forearm hits the upper arm. Only the random-step x1 points, which
may be in contact, reach ±π.

So Test 2 as specified is the identity map. I added **Test 2b** (`derive.py`, run `f99d9cab`), which fits the range to
the collision-checked x0 only. That zooms the elbow 1.137× and leaves the rest unchanged; 6.6 % of the x1 elbow values
(all self-contact steps) map slightly outside [−0.5, 0.5], up to ±0.569. Normals are mapped by the chain rule and
renormalised; speeds are the same physical labels in every set.

## Planner scaling (`field.py`, `evaluate.py`)

All three planners work in release coordinates (x = q / q_range) in every test, with their unchanged settings:
- **Native RK4:** unchanged.
- **Release MPPI:** 1536 samples, horizon 7, ≤ 400 steps, step 0.01, β 12.5, goal weight 4.
- **ntrl-demo MPPI:** 50 samples, horizon 5, 200 steps, step 0.015, temperature 50, tolerance 0.01.

Their steps, noise, joint bounds and goal tolerance are therefore identical in radians across tests. Only the network
input is remapped: y = x·k − b. `evaluate.py` imports `experiments/ntrl_cabinet/evaluate.py` unchanged and replaces its
two field hooks. Checks (`evaluation/scaling_check.json`):
- **Test 1 (identity map):** the replaced ntrl MPPI reproduces the original paths bit for bit, and the field
  difference is 0.
- **Tests 2 and 2b:** the field equals the network on hand-mapped coordinates to within 2×10⁻⁶. For 2b, 0.8 % of the
  frozen test endpoints have their elbow outside the data range.
- **Cost scale:** the median field value on the test queries is 0.305–0.315 in every run. The fixed MPPI temperatures
  therefore act on the same cost scale.

## Results (frozen cabinet suites: RK4 /600, MPPI /1800 = 600 queries × planning seeds 0/1/2)

| Run | Scaling | Training seed | GPU | RK4 | Release MPPI | ntrl-demo MPPI |
|---|---|---:|---|---:|---:|---:|
| `c0cb1a6e` | Test 1, full range | 87358 | 3090 | 167 | 577 | 488 |
| `49652673` | Test 2, data range (identity) | 87358 | 5070 | 178 | 593 | 513 |
| `66249640` | Test 1, full range | 1 | 5070 | 171 | 604 | 483 |
| `09711eb2` | Test 2b, collision-free range (elbow 1.137×) | 87358 | 3090 | 177 | **609** | **540** |
| `3c53b360` (earlier) | old baseline: 500k pairs, x1 not required in the box | 87358 | | 181 | 613 | 520 |

By suite, ntrl-demo MPPI /300:

| Run | free→free | free→comp. | comp.→free | comp.→comp. | same comp. | anywhere |
|---|---:|---:|---:|---:|---:|---:|
| Test 1 s87358 | 209 | 31 | 36 | 2 | 9 | 201 |
| Test 2 s87358 | 216 | 34 | 33 | 1 | 12 | 217 |
| Test 1 s1 | 212 | 32 | 32 | 3 | 7 | 197 |
| Test 2b s87358 | 215 | **48** | **46** | **6** | 11 | 214 |
| old baseline | 238 | 20 | 24 | 2 | 11 | 225 |

## Findings

1. **Test 1 vs Test 2: no difference, by construction.** The data range is the full range (zoom ≤ 1.00002), so Test 2
   is Test 1 with a 2×10⁻⁵ change of input scale. The three effectively identical runs (Test 1 with two seeds,
   Test 2) spread over 167–178 RK4, 577–604 release MPPI and 483–513 ntrl-demo MPPI. That spread is the noise floor
   for one training run.
2. **Test 2b (elbow fitted to its collision-free range) is the best of the four, but it is one seed.** It reaches
   540 ntrl-demo MPPI against 483–513, and 609 release MPPI against 577–604. The gain sits in the compartment suites
   (free→compartment 48 vs 31–34, compartment→free 46 vs 32–36). RK4 is within noise. Confirming it needs more
   training seeds of 2b and of Test 1.
3. **The in-area data does not beat the old baseline.** Requiring x1 in the box (with 400k instead of 500k pairs)
   gives 483–513 ntrl-demo MPPI against 520. It trades free-space successes (free→free 209–216 vs 238) for
   compartment→free (32–36 vs 24). The two changes (box constraint, pair count) are confounded.

## Test 3: in-cabinet training pairs mixed in at 0–100 % (`cabinet_data.py`, `mix.py`, `fitcheck.py`)

**Question (user).** Generate about 200k training pairs that each have at least one point in the cabinet, so at least
50 % of their points are in it. Mix them with in-area pairs (both points in the test area, cabinet not required) in
10 % steps, and report the success rates. Use both GPUs for sampling.

I read "test cases" as the training pairs. The frozen suites already have 67 % of their 600 queries with at least one
endpoint in the cupboard (197 none, 202 one, 201 both), so they stay unchanged.

### Data

- **In-cabinet set A** (`cabinet_data.py`, run `210851Z-ac3c7c7c`). The `data.py` in-area recipe, plus a requirement
  that the x0 or the x1 grasp point is in the cupboard's outer box (x 0.56–0.88, y 0.13–0.53, z −0.035–0.665).
  - Yield: 200,000 pairs from 1.872 × 10⁹ uniform draws in 97 s on both GPUs (3090 65 s, 5070 95 s). That is about 7×
    faster than one GPU without the pre-screen.
  - Pre-screen: URDF grasp-frame forward kinematics (`experiments/face_speed/robot.py`, box + 1 cm margin), so only
    draws near the box reach cuRobo. The exact accept/reject still uses cuRobo's grasp positions. A start-up check
    over 500k draws found a maximum deviation of 1.1 µm and 0 missed pairs.
  - Batching: 1M-draw batches, one worker per GPU with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The first
    attempt (`210717Z-216fe4c2`, 2M batches) ran out of memory on the 5070.
  - Composition: x1 is in the cupboard in 85.8 % of pairs, x0 in 22.5 %, both in 8.3 %. Of all endpoints inside the
    cupboard, 65 % are at the speed floor (clearance ≤ 5 mm or in collision), 10 % have ≥ 2.5 cm clearance and 0.3 %
    are free (≥ 5 cm). The set mostly says "stepping into the cupboard is slow".
- **In-area set B.** Test 1's 400k pairs (run `772f8677`, `full/`). Three per cent of its pairs have a cupboard point.
- **Mixtures** (`mix.py`). 200,000 pairs each: the first round(f · 200k) pairs of A plus the first 200k − that of B,
  shuffled, for f = 0, 0.1, …, 1. The mixtures are nested, and f = 0 is the first half of Test 1's data.
  - Release coordinates (Test 1's full-range scaling, identity map).
  - Shares of pairs with a cupboard point and of points in the cupboard are recomputed by forward kinematics.

Training and evaluation are as in Test 1: `train.py --variant metric --epochs 5000 --seed 87358`, one training run per
fraction. Even fractions ran on GPU 1 and odd ones on GPU 0, about 11 min per run.

### Results

Columns: RK4 /600, MPPI /1800. The last two columns are success rates (%) split by the number of query endpoints in
the cupboard, out of 197 / 202 / 201 queries.

| in-cabinet share | pairs with a cupboard point | points in the cupboard | RK4 | release MPPI | ntrl-demo MPPI | ntrl MPPI by endpoints in cupboard 0 / 1 / 2 | release MPPI 0 / 1 / 2 | run |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 0 % | 3.1 % | 1.7 % | 173 | 612 | 505 | 71.7 / 11.4 / 2.0 | 79.5 / 16.8 / 6.6 | `211045Z-3afa500d` |
| 10 % | 12.8 % | 6.9 % | 172 | 609 | 499 | 70.6 / 12.4 / 1.2 | 77.8 / 18.0 / 6.6 | `211100Z-892183cc` |
| 20 % | 22.5 % | 12.2 % | 169 | 580 | 494 | 71.1 / 10.6 / 1.7 | 78.7 / 13.5 / 5.5 | `212120Z-7fb82c6c` |
| 30 % | 32.2 % | 17.4 % | 191 | 632 | 537 | 72.4 / 16.3 / 1.7 | 83.6 / 18.5 / 4.3 | `212046Z-cd780569` |
| 40 % | 41.8 % | 22.7 % | 186 | 637 | 523 | 73.3 / 12.7 / 2.2 | 81.0 / 20.3 / 5.8 | `213209Z-878950bf` |
| 50 % | 51.5 % | 27.9 % | 189 | 618 | 525 | 76.1 / 11.1 / 1.3 | 82.2 / 17.8 / 4.0 | `213030Z-ce160baa` |
| 60 % | 61.2 % | 33.2 % | 184 | 626 | 550 | 76.5 / 14.4 / 1.8 | 80.9 / 20.1 / 4.3 | `214255Z-8382a03f` |
| 70 % | 70.9 % | 38.4 % | 180 | 636 | 530 | 75.0 / 12.9 / 1.5 | 82.1 / 20.8 / 4.1 | `214028Z-00ac905b` |
| 80 % | 80.6 % | 43.7 % | 178 | 603 | 503 | 72.4 / 11.2 / 1.2 | 81.4 / 16.0 / 4.1 | `215340Z-43d58912` |
| 90 % | 90.3 % | 48.9 % | 180 | 634 | 523 | 73.8 / 13.5 / 0.8 | 81.2 / 21.5 / 4.0 | `215012Z-41858b8d` |
| 100 % | 100 % | 54.2 % | 175 | 611 | 515 | 71.4 / 13.5 / 1.8 | 78.8 / 18.2 / 5.8 | `220420Z-9105ec65` |

For reference: Test 1 (400k in-area pairs) scored 167–171 / 577–604 / 483–488, and the old baseline 181 / 613 / 520.

**Field check** (`fitcheck.py`, read-only). Each field's speed 1/|∇τ| at both pair endpoints is compared with its
label, on three fixed point sets:

- the last 20k pairs of A, which no fraction ≤ 0.9 trained on;
- the first 20k pairs of A, which every fraction ≥ 0.1 trained on;
- the last 20k pairs of B, which no mixture trained on.

Contact means a label at the floor (target speed 0.012); clear means clearance ≥ 2.5 cm.

| in-cabinet share | cupboard points (held-out A): rank corr. | field median speed, contact / clear | contact read faster than 0.5 | wall contrast clear/contact (labels 63×) | outside the cupboard (held-out B): rank corr., contrast (labels 79×), contact > 0.5 |
|---:|---:|---|---:|---:|---|
| 0 % | 0.26 | 0.65 / 0.81 | 70 % | 1.25× | 0.63, 3.0×, 32 % |
| 30 % | 0.35 | 0.35 / 0.65 | 33 % | 1.85× | 0.59, 2.7×, 32 % |
| 50 % | 0.36 | 0.29 / 0.60 | 27 % | 2.1× | 0.57, 2.2×, 32 % |
| 70 % | 0.40 | 0.20 / 0.56 | 20 % | 2.8× | 0.55, 2.2×, 34 % |
| 100 % | 0.47 | 0.15 / 0.55 | 14 % | 3.7× | 0.45, 1.7×, 40 % |

The trained set (first 20k of A) gives nearly the same numbers: medians within 0.03, rank correlation 0.00–0.04
higher, 3.6× contrast at 100 %.

### Findings

1. **The cabinet share has no effect beyond seed noise.**
   - Spread: across the 11 fractions the totals have standard deviations of 7 (RK4), 17 (release MPPI) and 17
     (ntrl MPPI). Repeat runs of one setup give about the same (ranges 11 / 27 / 30 over three runs).
   - Trend: none is monotone. The 30–70 % runs sit a little higher (ntrl 523–550, about 27 above the other six on
     average). That is a post-hoc split of 11 noisy runs, not a tested effect; it would need more seeds.
   - Both-in-cabinet queries stay at 0.8–2.2 % (ntrl MPPI) and 4–7 % (release MPPI) at every fraction. This includes
     100 %, where 54 % of the training points are in the cupboard.
   - Per query, 100 % against 0 % gains 6 and loses 7 of the 603 both-in-cabinet query-seed units for ntrl MPPI, and
     +14 / −19 for release MPPI. That is churn, not a gain.
2. **The planners succeed in the cabinet only where the straight line already does.**
   - 183 of the 201 both-in-cabinet queries need a detour (their straight joint-space line collides). On those:
     - ntrl MPPI solves 1 / 3 / 5 of 549 query-seed units at 0 / 50 / 100 %;
     - release MPPI solves 4 / 2 / 9;
     - RK4 solves 1 / 2 / 3 of 183.
   - The in-cabinet successes come from the other 18 queries, and the planners lose many even of those (ntrl 11 → 6
     of 54).
   - At least 97 % of in-cabinet failures are collisions. About 90 % of the failed ntrl MPPI paths reach the goal
     tolerance, so they drive through panels rather than stall.
3. **The data does reach the network: it sees the walls better, but far too softly.**
   - From 0 % to 100 %, on held-out in-cabinet pairs:
     - the share of contact points the field reads as fast drops from 70 % to 14 %;
     - the wall contrast rises from 1.25× to 3.7×;
     - the rank correlation rises from 0.26 to 0.47.
   - The labels ask for 63×. Pairs the field trained on fit barely better than held-out ones (3.6× at 100 %), so this
     is a limit of the fit, not a generalization gap.
   - A 3.7× slowdown makes a panel a mild cost rather than a barrier: a path that crosses a few centimetres of panel is
     still cheaper than most detours. The planners therefore keep cutting through.
   - This matches the earlier ntrl-cabinet finding that the field is nearly flat in the 0–5 cm clearance band
     (`experiments/ntrl_cabinet/README.md`).
4. **Outside the cupboard the fit degrades as in-area pairs are replaced** (contrast 3.0× → 1.7×, contact read as fast
   32 % → 40 %). The no-cupboard queries do not move (ntrl 71–77 %), because 79 % of them are straight-line feasible.
5. **Tendencies, one seed each, not established:**
   - Single-endpoint detour queries drift up slightly (ntrl 36 → 52 and release MPPI 50 → 68 of 522 units, 0 → 100 %).
   - Release MPPI solves fewer both-in-cabinet queries from 20 % on (24–35 of 603 units against 40 at 0–10 %), mostly
     straight-line ones (22 and 26 of 54 at 50 % and 100 %, against 36 at 0 %).

**Bottom line.** Where the training data sits is not the bottleneck for this baseline. More in-cabinet data moves the
field the right way, but ntrl-demo's network represents the panels about 17× too softly (3.7× against 63×), so its
planners still cut through them.

## Test 4: collision pairs at one third, and a strong cabinet (`collide_data.py`, `make_strong.py`, `run4.py`, `pipeline4.py`, `fitcheck4.py`)

**Question (user).** Train on "the most mixed pairs", configurations with the arm inside the cabinet and outside, plus
configurations that collide, labelled with the lowest speed, at one third of the whole sample set. Then the same
training recipe in a second environment, the **strong cabinet**: walls 5× thicker, the cabinets the same size.

**How I read it.**
- **Mixed pairs:** exactly one of the two grasp points inside the cupboard's outer box (Test 3's set A required at
  least one; 92 % of its pairs had exactly one).
- **Collision pairs, one third of the 200k:** x0 uniform with its grasp point in the test area and touching or inside
  the cupboard or the table (the label's obstacle distance ≤ 0); x1 is the recipe's random step from it.
  - The label already gives every end within 5 mm of contact the lowest speed (0.1, trained as s′ = 0.012). So the new
    part is pairs anchored inside obstacles: the recipe's x0 was always collision-free, while its x1 already lands in
    contact often (73 % of mixed-pair x1).
  - Self contact alone is left out. It is 46 % of the colliding in-area poses (preflight, 10M draws), does not depend
    on the cabinet, and the random steps already reach it.
- **Strong cabinet** (`make_strong.py` → `experiments/cabinet/scenes/cabinet_strong.yml`):
  - Every panel is 0.10 m instead of 0.02 m: back, two sides, top and both shelf boards.
  - Every compartment keeps its inside size (0.30 deep, 0.36 wide, 0.225 / 0.21 / 0.205 m tall), so the panels grow
    outward. The opening, the floor and the centre line stay, and the bottom compartment is exactly where it was.
  - The thicker boards lift the middle compartment by 0.08 m and the top one by 0.16 m. The cupboard measures
    0.40 × 0.56 × 0.94 m instead of 0.32 × 0.40 × 0.70.
  - Test area by the scenes' task rule (height = cupboard; width 1.5× and depth 2.5× the cupboard), except that it
    starts at x 0.08 as before rather than over the robot base: x 0.08–0.96, y −0.09–0.75, z −0.035–0.905.
  - Own frozen suites from the same generator and seeds (`run.py cabinet-tests`, `freeze_tests.py` → `tests_strong/`).
- **Same recipe in both scenes**, generated in each scene. Each scene also gets a **control**: 200k mixed pairs and no
  collision pairs, so the two changes can be told apart. Training and evaluation are as in Test 3 (metric variant,
  5,000 epochs, training seed 87358, planning seeds 0/1/2). The strong cabinet's two sets were trained again with seed 1
  after the first runs collapsed.

### Data (`collide_data.py`; runs `021848Z-afa3dd8b` cabinet, `022101Z-c171d565` strong cabinet)

Both GPUs, 131 s and 87 s. Each run writes `test4/` (66,667 collision + 133,333 mixed pairs), `control/` (200k mixed
pairs, test4's mixed pairs first) and `heldout/` (20k of each kind that neither set contains).

| | cabinet test4 | cabinet control | strong test4 | strong control |
|---|---:|---:|---:|---:|
| pairs with exactly one grasp point in the cupboard | 71 % | 100 % | 71 % | 100 % |
| pair ends in collision (clearance ≤ 0) | 54 % | 37 % | 57 % | 43 % |
| pair ends at the lowest speed (≤ 5 mm) | 56 % | 39 % | 59 % | 44 % |
| x1 in collision | 74 % | 73 % | 81 % | 86 % |

- **Collision starts touch** the table in 69 % of pairs, the cupboard in 42 % (both 11 %), in both scenes. 16 % have the
  grasp point inside the cupboard.
- **Mixed pairs:** x0 is the cupboard end in 16 % (cabinet) and 7 % (strong) of pairs; otherwise x1 steps in.
- **Draws:** 2.24B (cabinet) and 1.14B (strong) uniform draws for the 220k mixed pairs; 38M and 24M for the 90k
  collision pairs.
- **First attempt** (`021157Z-c51f49b5`, the 5070 alone) ran out of GPU memory at the start of the collision stream,
  after the 2,240-batch mixed stream. The collision stream now runs first, with half the label chunk, on both GPUs.

**Strong-cabinet suites** (`021328Z-8986144a` → `tests_strong/`; the first attempt `021034Z-ac7c98e5` used `python3`
instead of `./docker-python`, which looked for the absent pinned image). Same six suites of 100. The thick panels
block more straight lines into and out of the compartments:

| straight-line feasible queries | free→free | free→comp. | comp.→free | comp.→comp. | same comp. | anywhere | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| cabinet | 82 | 17 | 11 | 1 | 17 | 74 | 202 |
| strong cabinet | 77 | 6 | 6 | 0 | 20 | 88 | 197 |

### Results (each scene on its own frozen suites: RK4 /600, MPPI /1800)

| scene | training set | seed | RK4 | release MPPI | ntrl-demo MPPI | ntrl MPPI % by endpoints in cupboard 0 / 1 / 2 | ntrl MPPI % straight-line / detour queries | field | run |
|---|---|---:|---:|---:|---:|---|---|---|---|
| cabinet | Test 3 100 % (≥ 1 in cupboard) | 87358 | 175 | 611 | 515 | 71.4 / 13.5 / 1.8 | 71.5 / 6.9 | | `220420Z-9105ec65` |
| cabinet | control | 87358 | 191 | 636 | 518 | 72.4 / 12.9 / 2.0 | 69.0 / 8.4 | | `022230Z-b9f62b88` |
| cabinet | test4 | 87358 | 170 | 615 | 498 | 72.6 / 10.1 / 1.3 | 69.3 / 6.5 | | `022230Z-061c5d00` |
| strong | control | 87358 | 25 | 257 | 127 | 20.9 / 0.2 / 0.2 | 19.6 / 0.9 | collapsed | `023311Z-4f029c0b` |
| strong | control | 1 | 174 | 614 | 503 | 74.7 / 8.3 / 1.2 | 72.9 / 6.0 | | `030834Z-bc235281` |
| strong | test4 | 87358 | 20 | 240 | 137 | 20.8 / 1.8 / 0.3 | 21.2 / 1.0 | collapsed | `023339Z-856a30d0` |
| strong | test4 | 1 | 24 | 236 | 139 | 22.4 / 0.3 / 0.5 | 21.2 / 1.2 | collapsed | `030834Z-a99127ec` |

Queries by endpoints in the cupboard: 197 / 202 / 201 (cabinet) and 199 / 201 / 200 (strong). For both-in-cupboard
queries that need a detour (183 and 180), ntrl MPPI solves 0–1.1 % in every run (0 for the collapsed ones).

**Field check** (`fitcheck4.py`, read-only): each field on the held-out pairs of its own scene, both pair ends.
Contrast = median field speed at clear ends (clearance ≥ 2.5 cm) / at contact ends (≤ 5 mm); the labels ask for 63×.

| field | all ends: contact / clear, contrast | contact read > 0.5 | in the cupboard | outside | collision pairs |
|---|---|---:|---:|---:|---:|
| cabinet, Test 3 100 % | 0.25 / 0.66, 2.6× | 22 % | 4.2× | 2.0× | 2.2× |
| cabinet, control | 0.22 / 0.70, 3.2× | 21 % | 6.0× | 2.4× | 2.6× |
| cabinet, test4 | 0.11 / 0.64, 5.8× | 11 % | 3.3× | 7.7× | 6.7× |
| strong, control seed 1 | 0.19 / 0.74, 3.9× | 22 % | 9.9× | 2.3× | 2.6× |
| strong, collapsed (all three) | 0.010 / 0.013, 1.2× | 0 % | 0.8× | 1.2× | 1.2× |

**The collapse.**
- **What the collapsed fields look like:** median learned T on held-out pairs is 10.0–10.1, against 0.22–0.31 for the
  healthy fields. That is 72× the pair length, the slowest speed (1/0.012 = 83) almost everywhere.
- **Speed at collision-free poses:** 0.012 at the x0 of mixed pairs, whose label is 0.81 (99 % below 0.05). The healthy
  strong field reads 0.69 there.
- **Why it stays there:** ntrl-demo's loss weights every pair by exp(−0.5 T)
  (`baselines/ntrl-demo/models/metric/model_function_metric.py`, `loss_n`). That weight falls from 0.86–0.90 to 0.006–0.007,
  so a field that is slow everywhere makes every pair's loss about 130× lighter.
- **When it happens:** the strong control (seed 87358) is collapsed at its first saved checkpoint (epoch 1,000, T 10.6)
  and never recovers. Its validation loss ends at 0.022, against 0.006 for the original cabinet's and for the healthy
  strong control.

### Findings

1. **Collision pairs did not help planning in the cabinet.**
   - Totals: test4 against control gives 170 / 615 / 498 against 191 / 636 / 518. The difference is inside the
     repeat-run noise of one training run (ranges 11 / 27 / 30).
   - Both-in-cupboard queries: 1.3 % against 2.0 % (ntrl MPPI).
   - The field is harder overall (5.8× against 3.2×; contact read as fast 11 % against 21 %), but mostly at the table.
     Outside the cupboard the contrast rises to 7.7× from 2.4×, and 69 % of the collision starts touch the table.
   - Inside the cupboard the field is softer: 3.3× against 6.0×, and in-cupboard collision poses read 0.12 against
     0.048. Test4 has a third fewer mixed pairs (133k), the pairs that teach the cupboard.
2. **The strong cabinet collapses the training in 3 of 4 runs** (both test4 seeds, one of two control seeds); none of
   the original-cabinet runs did.
   - A collapsed field reads the slowest speed everywhere, so its planners fail even in free space. RK4 solves 7 of 100
     free→free queries; 87 of the 93 failed paths never reach the goal.
   - The trigger is not established. The strong data has more contact at x1 (86 % against 73 % in the control sets),
     but cabinet test4 had more lowest-speed ends overall (56 %) and trained normally.
3. **When it trains, the strong cabinet fits its panels harder but plans no better.**
   - In-cupboard contrast is 9.9× against 6.0× for the cabinet control, and 13× against 5.6× on the mixed pairs. That
     is still 5–6× short of the labels.
   - Totals are 174 / 614 / 503 against 191 / 636 / 518.
   - One-end-in-cupboard queries do worse (8.3 % against 12.9 %), in line with the strong suites' fewer straight-line
     entries. Both-in-cupboard queries stay at 1.2 %.
4. **The detour problem of Test 3 is unchanged.** Every healthy run solves 6–8 % of the queries that need a detour (the
   collapsed ones about 1 %), and at most 1.1 % of the both-in-cupboard ones.

**Bottom line.** Neither one third of collision pairs nor 5× thicker panels gets ntrl-demo into the compartments.
Collision pairs harden the field where they land (mostly the table) at the expense of in-cupboard data. Thick panels
are fitted about 1.6× harder when training works, but in this environment ntrl-demo's exp(−0.5 T) loss weighting
usually finds the cheap "slow everywhere" solution instead. One healthy run per condition is too few to rank the
conditions within the noise.

## Axis sensitivity of the true field in joint space (`axis_sensitivity.py`, output `axis_sensitivity/`)

Question (2026-11-01): the mixed-pairs model sees the back panel (T rises along +x inside a compartment) but not the side
panels or shelf boards (flat along y, z). Is a Cartesian y or z move intrinsically harder to represent in joint space?
Read-only analysis on the Test 4 viewer grid (`test4_viewer/cabinet`: 20,841 IK poses at 2.5 cm, 5,975 in the cupboard,
edge costs, the `mixed` b9f62b88 and `collide` 061c5d00 checkpoints). At every pose p with joints q:

- joint velocity for a 1 m/s gripper move along d ∈ {x, y, z}: `dq_d = J_p⁺ e_d` (least-norm position IK; also the
  orientation-held `J⁻¹[e_d; 0]` and the grid's own poses `(Q[p+d] − Q[p−d]) / 2h`, which is the line the viewer draws);
- true joint gradient `dT/dq` from the eikonal identity of the graph truth: `|dT/dx| = 1/s'(x)` along the arrival
  direction of the Dijkstra path (x = q / q_range, the network input); directional rate `g_d = dT/dq · dq_d`
  (time per metre); per unit joint motion `g_d / |dx_d|`; "stretch" `|dx_d|` = network-input distance per metre;
- the same products with the learned gradient (autograd), and central differences of true and learned T along the grid
  lines as the like-for-like check (`check_vs_central_difference`; Spearman 0.6–0.8 against the eikonal rate, ratio ≈ 1
  along y, 1.3–1.7 along x and z where the eikonal rate is the local slope and the 5 cm secant is smaller).

Start = the middle-compartment preset (the `in front of the cupboard` start gives the same numbers). Medians:

| region (poses) | axis | rad per m | stretch (x-units per m) | true slope per m, eikonal / secant | true per unit joint motion | learned secant, mixed / collide | learned ÷ true (secant) mixed |
|---|---|---|---|---|---|---|---|
| cupboard, contact (4,549) | x | 3.36 | 0.36 | 10.5 / 4.8 | 13 (secant) | 1.80 / 0.40 | 0.68 |
| | y | 1.50 | 0.15 | 4.8 / 5.6 | 33 | 0.33 / 0.15 | 0.11 |
| | z | 1.38 | 0.14 | 4.7 / 3.3 | 19 | 0.36 / 0.20 | 0.19 |
| cupboard, free > 5 mm (1,142) | x | 3.19 | 0.33 | 0.55 / 0.37 | 1.1 | 0.43 / 0.34 | 1.05 |
| | y | 1.50 | 0.16 | 0.18 / 0.40 | 2.0 | 0.17 / 0.15 | 0.35 |
| | z | 1.56 | 0.16 | 0.17 / 0.28 | 1.6 | 0.14 / 0.16 | 0.54 |
| outside, free (10,932) | x | 2.81 | 0.24 | 0.17 / 0.18 | 0.8 | 0.24 / 0.22 | 1.23 |
| | y | 2.25 | 0.24 | 0.11 / 0.12 | 0.5 | 0.15 / 0.17 | 1.34 |
| | z | 2.35 | 0.24 | 0.12 / 0.12 | 0.5 | 0.08 / 0.12 | 0.68 |

Alignment of the gradient with the joint direction of each axis, |cos|, cupboard contact: truth 0.33 / 0.37 / 0.41 (x / y / z),
mixed model 0.64 / 0.25 / 0.31. True `|dT/dx|` there is 83 (speed floor 0.012); the mixed model's is 10.6 (median), collide 1.8.

On the three lines through the middle preset (`middle_lines.json`; secant slopes per metre): entering the back panel
(x = 0.80–0.86) truth 8 / 20 / 30, mixed 7.7 / 13.7 / 20.1; the side panels (y = 0.23 and 0.40–0.43) truth 9–10 both
sides, mixed 0.28 / 0.41–0.45; the shelf boards (z = 0.165–0.24 and 0.39–0.49) truth 10–15, mixed 0.27 / 0.28–0.46.

Findings:

1. Kinematics compresses y and z inside the cupboard by ≈ 2.2×: one metre of gripper travel needs 3.3 rad (0.35 input
   units) along x but 1.5 rad (0.15) along y or z, because the arm is near full extension there (outside, all three are
   2.2–2.8 rad). A compartment is 0.094 input units deep (x) but 0.061 wide (y) and 0.034 high (z). The same Cartesian
   slope therefore needs 1.5–2.5× more T change per unit joint motion along y/z (33 and 19 vs 13 in the contact zone);
   the eikonal bound (1/s' = 83) is the same for all axes, so y and z sit closer to it.
2. That factor of ~2 does not explain the model. The mixed model reproduces the x slope within a factor 1.5 (0.68 of the
   truth at contact, 1.05 free) but only 0.11 / 0.19 along y / z at contact and 0.35 / 0.54 free, i.e. 5–10× too flat,
   and its gradient points along the x joint direction (|cos| 0.64 vs 0.25 / 0.31) while the truth is isotropic.
3. The training pairs explain the direction: in the control (mixed) set every pair has exactly one end in the cupboard
   (both-in-cupboard 0 %), the outside end is in front of it (x < 0.56) in 93 % of pairs, and the pair displacement is
   x-dominant (median |u_x| 0.72 vs |u_y| 0.38, |u_z| 0.45 for compartment ends; x is the largest component in 59 %).
   The eikonal loss fixes the gradient magnitude at the cupboard end; its direction is set by where the partners are,
   so T is supervised along "towards the front" and nothing pulls it up at the side panels or shelf boards.

Command:

    CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/joint_range/axis_sensitivity.py --data jeffrey/joint_range/test4_viewer/cabinet --output jeffrey/joint_range/axis_sensitivity

## Both ends in the cupboard (`cupboard_data.py`, `run4.py cupdata`, output `axis_sensitivity_v2/`)

Test of the axis-sensitivity conclusion (2026-11-01, user: "generate pairs with both ends in the cupboard and retrain").
`cupboard_data.py` is the Test 4 mixed recipe with the membership test changed: both grasp points in the cupboard's
outer box (x0 window 0 < d − 0.01 < 5 cm, x1 the recipe's random step). Data run `051803Z-cf499c8b` (24.7 billion
draws, 15 min on both GPUs): `both/` 200k pairs, `combo/` 100k mixed + 100k both, `heldout/` 20k both. The both set is
what was missing: the largest gripper-displacement component is y in 40 % and z in 47 % of pairs (mixed set: x in 59 %),
median displacement 8 cm, x1 in collision 47 %. Trainings (baseline recipe, seed 87358): both-only `053322Z-8f7496df`
(3090), combo `053324Z-3e02c3ec` (5070).

| model | RK4 /600 | release MPPI /1800 | ntrl MPPI /1800 | in-cupboard contact, learned secant slope ÷ truth: x / y / z |
|---|---|---|---|---|
| control, mixed only (b9f62b88) | 191 | 636 | 518 | 0.68 / 0.11 / 0.19 |
| both-only (8f7496df) | 102 | 475 | 322 | 0.23 / 0.07 / 0.10 |
| combo (3e02c3ec) | 173 | 619 | 498 | 0.57 / 0.07 / 0.16 |

(Repeat-run noise RK4 11, rMPPI 27, nMPPI 30.) Along the y line through the middle preset the truth rises to 2.4 at the
side panels; mixed, both-only and combo reach 0.20, 0.18 and 0.13. Along z (shelf boards) all three stay below 0.17
against a truth of 0.9–1.2. The both-only model also loses the back panel (slope 1–8 vs 8–30) and collapses outside
the cupboard, where it has no pairs. **Pairs displaced along y and z do not teach the side panels or shelf boards; the
pair-geometry explanation of the axis-sensitivity section is wrong.**

Mechanism (scripts in the session scratchpad, numbers from the three checkpoints and the training sets):

1. The network does not fit its own labels at colliding endpoints in any set: predicted speed 1/|dT/dx1| at colliding
   x1 is 0.10 (mixed model on the mixed set) and 0.29 (both-only model on the both set) against the target 0.012, while
   free endpoints are fit to within 20 %. Those endpoints carry 58–66 % of the remaining eikonal loss.
2. The eikonal term is bounded: 0.01 · (√(s'·|dT/dx|) − 1)² is at most 0.01 per endpoint however badly a wall is
   under-fit (speed 0.1 vs 0.3 at a contact endpoint differ by 0.0043 vs 0.0064), so a smooth field that averages
   a thin wall away costs little. The Fourier input mapping has |2πB| at most 21, a finest half-period of 0.15 input
   units; a side panel plus its 5 cm band is 0.008 input units thick along the y joint direction, the back panel band
   0.018.
3. What decides the fit is the 6-D thickness of the slow region, not the Cartesian axis: for colliding x1 training
   endpoints, the distance along random joint directions to clearance > 5 mm (min over 32 directions, input units)
   predicts the fitted speed with Spearman −0.6: escape ≥ 0.06 → speed 0.017–0.024 (fit), 0.015–0.03 → 0.13–0.21,
   < 0.007 → 0.35–0.51 (ignored). The both set's colliding endpoints are mostly thin (median escape 0.0075 vs 0.03 in
   the mixed set), so adding 200k of them changes nothing. On the grid lines through the middle preset the contact poses
   escape in 0.005–0.03 units along x (median over directions 0.12) but 0.002–0.03 along y (0.08) and 0.002–0.01 along z
   (0.035): the back panel sits at the arm's reach limit, a thick region of extended configurations, while the side
   panels and shelf boards are slabs with free space on both sides, thinner than the field's resolution.

Command: `BELLMAN600_DERIVED_IMAGE=bellman600-env:local python3 jeffrey/joint_range/run4.py cupdata --scene cabinet`,
then `run4.py train --set both|combo`; probe `axis_sensitivity.py --models both=registered_... combo=registered_...`.

## Files and commands

New files only (`experiments/ntrl_cabinet` and ntrl-demo unchanged): `data.py`, `derive.py`, `field.py`,
`evaluate.py`, `pipeline.py`, `run.py`; for Test 3 `cabinet_data.py`, `mix.py` and the read-only `fitcheck.py`; for Test 4
`collide_data.py`, `pipeline4.py`, `run4.py` (a separate runner, so Tests 1–3 keep their declared inputs), `make_strong.py` and the
read-only `fitcheck4.py`, plus the scene `experiments/cabinet/scenes/cabinet_strong.yml` and its frozen suites `experiments/cabinet/tests_strong/`.
The Test 4 time-field page: `test4_field.py` (host precompute per scene into `test4_viewer/<scene>/`: IK grid, ground truth, line
labels, the two networks, the arm's collision spheres), `test4_template.html` → `build_test4_viewer.py` → `test4_viewer.html`,
checked with `timefield_check.js` per scene.

    BELLMAN600_DERIVED_IMAGE=bellman600-env:local python3 jeffrey/joint_range/run.py data --gpu 1
    BELLMAN600_DERIVED_IMAGE=bellman600-env:local python3 jeffrey/joint_range/run.py derive --data-run registered_20261031T192723Z-772f8677 --gpu 1
    BELLMAN600_DERIVED_IMAGE=bellman600-env:local python3 jeffrey/joint_range/run.py test --data-run registered_20261031T192723Z-772f8677 --scaling full --seed 87358 --gpu 1
    BELLMAN600_DERIVED_IMAGE=bellman600-env:local python3 jeffrey/joint_range/run.py test --data-run registered_20261031T192723Z-772f8677 --scaling range --seed 87358 --gpu 0
    BELLMAN600_DERIVED_IMAGE=bellman600-env:local python3 jeffrey/joint_range/run.py test --data-run registered_20261031T192723Z-772f8677 --scaling full --seed 1 --gpu 0
    BELLMAN600_DERIVED_IMAGE=bellman600-env:local python3 jeffrey/joint_range/run.py test --data-run registered_20261031T194120Z-f99d9cab --scaling anchor --seed 87358 --gpu 1

Test 3 (one mixture per call; the sweep ran fractions 0.0, 0.2, …, 1.0 on GPU 1 and 0.1, 0.3, …, 0.9 on GPU 0):

    BELLMAN600_DERIVED_IMAGE=bellman600-env:local python3 jeffrey/joint_range/run.py cabdata
    BELLMAN600_DERIVED_IMAGE=bellman600-env:local python3 jeffrey/joint_range/run.py mix --cab-run registered_20261031T210851Z-ac3c7c7c --data-run registered_20261031T192723Z-772f8677 --fraction 0.5 --gpu 0
    CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/joint_range/fitcheck.py --runs results/registered_<mix run>... --output <dir>

Test 4 (data on both GPUs; trainings one per GPU):

    python3 jeffrey/joint_range/make_strong.py --check
    BELLMAN600_BUILD_NETWORK=host BELLMAN600_DERIVED_IMAGE=bellman600-env:local ./docker-python run.py cabinet-tests --scene cabinet_strong --gpu 1
    python3 experiments/cabinet/freeze_tests.py results/registered_20261101T021328Z-8986144a
    BELLMAN600_DERIVED_IMAGE=bellman600-env:local python3 jeffrey/joint_range/run4.py data --scene cabinet
    BELLMAN600_DERIVED_IMAGE=bellman600-env:local python3 jeffrey/joint_range/run4.py data --scene cabinet_strong
    BELLMAN600_DERIVED_IMAGE=bellman600-env:local python3 jeffrey/joint_range/run4.py train --scene cabinet --data-run registered_20261101T021848Z-afa3dd8b --set test4 --gpu 1
    BELLMAN600_DERIVED_IMAGE=bellman600-env:local python3 jeffrey/joint_range/run4.py train --scene cabinet_strong --data-run registered_20261101T022101Z-c171d565 --set control --seed 1 --gpu 0
    CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/joint_range/fitcheck4.py --runs results/registered_<train run>... --output <dir>
