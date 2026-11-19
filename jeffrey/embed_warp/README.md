# Embedding-space warp: gated Fourier octaves as a diffeomorphism of ntrl-demo's embedding

Study of 2026-11-12 (request: revisit the plus-maze tests; instead of warping the maze, warp the *embedding*: a diffeomorphism of
the embedding space that stretches where travel time / curvature is high; try (1) a stretch from the geometry via the band sampling,
(1.5) a diffusion / elastic physics-style gate at the free/obstacle boundary, (2) a gate from the network's curvature trained
separately, and a gate learned through the travel-time network; make it work in general scenes: disks, plus maze, Gibson).
Findings page: https://claude.ai/artifact/FqBhMFaPPqFAYP9fCJJRXS (report/embed_warp.html, `page.py`). Numbers in this file are from all 107 registered runs (queue finished 2026-11-12 ~07:05 EST); `report/tables.md` (`report.py`) is
regenerated from the registry and is authoritative.

## The map

ntrl-demo: T(x₀, x₁) = metric(enc(γ(x₀)), enc(γ(x₁))), γ(x) = [sin 2πBx, cos 2πBx] ∈ R²⁵⁶, B 128 × 3 ~ N(0, 1) truncated. The
encoder is Lipschitz in γ, so the rate at which T can change along a direction is set by the arc length of the embedding there,
the Gram JᵀJ of J = dγ/dx = (2π)² BᵀB ≈ (2π)²·128·σ²·I: one resolution, everywhere, in every direction. A warp of the 3-D
C-space (every earlier study) can only add resolution along some direction more than others; a map of the embedding space can
add it isotropically because R²⁵⁶ has room.

The network's first residual block adds the input to a 256-wide layer, so the input must stay 2 × 128: the octaves are a
reassignment of ntrl-demo's own rows (architecture and parameter count 656,900 unchanged; `--octaves ''` is ntrl-demo bit for
bit, checked against the registered identity run 20261103T163539Z-48369bc1 on the 3090):

    base rows (128 − K m) at ntrl-demo's frequency;  K octaves of m rows, whitened (B_kᵀB_k ∝ I exactly), scaled by 2, 4, 8 (16)
    Γ(x) = (γ₀, γ₁, …, γ_K)        Ψ(u₀, u₁, …) = (u₀, μ₁(u₀)u₁, …, μ_K(u₀)u_K),  μ ∈ [μ_min, 1]      (coupling layer = diffeomorphism)
    JᵀJ = g₀ + Σ μ_k² g_k + Σ m |∇μ_k|² n nᵀ      (cross terms vanish: J_kᵀγ_k = Σ(cos·sin − sin·cos) = 0)

Measured (`features.isotropy`, central differences of the full input mapping): band anisotropy 1.002 whitened (1.44 unwhitened,
from 16 random directions per octave), stretch ×7.4 at the walls for ×2..×16 and ×3.8 for ×2..×8, ×0.69 in the open with 64 base
rows. The constructed C-space warp of the gap study had anisotropy 6 in the band arms.

Gates (`gates.py`, `features.py`): `geo` clearance smoothstep (open ≤ 0.5 cells and inside obstacles, closed ≥ 1.5), `kde`
density of the training band x₀ alone, `diff` masked Jacobi heat from the walls through free space, `curv` Hessian norm (or
eikonal misfit) of a trained field at every grid node, `learn` sigmoid(MLP(u₀)) trained by the T loss (own AdamW group), `const`
μ ≡ 1 (always on, the uniform multi-scale control), none = ntrl-demo.

## Files

| file | what |
|---|---|
| `scene.py` | `Disks` (22-cell box, 40 disks r 0.5–1.4 cells, gaps ≥ 0.7) and `Gibson` (P-NTFields apartment 1, 1024² signed-distance table, E = 64 or 32 cells, 9 × 5 robot lattice) with jeffrey/maze_scale's Maze interface, so its sampler / truth / MPPI / scorer run unchanged; `load_scene` for maze, plus, gap and these |
| `features.py` | the embedding map (`EmbedNN`), `GridGate` / `LearnedGate` / `ConstGate`, `load_field`, `isotropy` |
| `gates.py` | geo / kde / diff / curv tables |
| `trainer.py` | ntrl-demo's loop step for step (+ gate parameter group, optional budget, `--td-cells` = the package loss with the TD step as a parameter, equality checked) |
| `pipeline.py` | field stage: gate, train, plan (3 planner seeds), diagnostics (speed fit, near-wall slope / direction / attraction, bumps, failures by arrival, isotropy, gate statistics, slices) |
| `prepare.py` | prepare stage for disks / Gibson; `--from <prepare> --inside-share` = data variant with x₁ inside obstacles |
| `evalq.py` | plan a held-out query set with registered fields |
| `run.py` | registry (stages prepare / field / evalq; sources frozen in `snap/<hash>/`) |
| `report.py`, `figs.py`, `page.py` | `report/runs.json`, `report/tables.md`, figures, the findings page |

Prepares: plus maze 20261102T042454Z-62864e61 (240 queries; held-out 1,200-set 20261103T174058Z-405b526e), plus + 20 % inside
pairs 20261112T065945Z-91ae7981, disks 20261112T063116Z-23faefc0 (200 queries), Gibson 64 20261112T063206Z-6e271a95 (250),
Gibson 32 20261112T081337Z-6b5d9544 (211). Success = MPPI plan (ntrl-demo's planner, seeds 0/1/2) that arrives and keeps the
exact clearance ≥ 0 along the path. One training of the same recipe moves by 3–6 points between seeds; read seeds first.

## Results (pooled over training seeds where several ran; `report/tables.md` has every run)

| scene | ntrl-demo | ×2..×16 always on | clearance gate ×2..×16 | learned gate ×2..×16 | ×2,×4,×8 always on (`oct8`) | clearance gate ×2,×4,×8 (`geo8`) |
|---|---:|---:|---:|---:|---:|---:|
| plus 240-set, 3 seeds | 70.9 (70.1 / 74.6 / 68.1) | 66.0 (1 seed) | 75.7 (68.1 / 89.3 / 69.9) | 73.8 (69.4 / 81.8 / 70.0) | **78.6** (68.9 / 82.8 / 84.2) | 76.5 (76.5 / 75.8 / 77.1) |
| plus held-out 1,200-set, 3 seeds | 68.4 (65.8 / 73.1 / 66.2) | 68.8 (1 seed) | 74.3 (67.8 / 77.4 / 77.6) | 72.8 (68.9 / 78.6 / 70.9) | **79.3** (74.9 / 79.7 / 83.2) | 73.6 (73.3 / 72.2 / 75.4) |
| disks, 3 seeds | 83.9 | 95.4 | 94.9 | 95.7 | 93.4 (2) | 94.9 |
| Gibson 64 cells (1 seed) | 66.5 | 67.2 | 68.4 | 66.3 | 67.6 | 67.9 |
| Gibson 32 cells, 3 seeds | 81.9 | 76.0 (1) | 75.8 (1) | 80.3 | 74.9 (1) | 78.9 |

Held-out 1,200-set, one seed (87358) unless noted: μ_min 0 73.9 (2 seeds), μ_min 0.3 (`geom3`) 77.8, base rows ×1.41 (`geob`)
77.2, ×2..×8 μ_min 0 75.5, learned gate + inside pairs 73.5 (3 seeds), ×2,×4,×8,×12 always on 73.3, ×2,×4,×8 with 24 / 32 rows per
octave 72.2 / 73.1, unwhitened 72.4, ×3,×6 67.7, ×2,×4 67.9, clearance gate ×2,×4 68.5, band-density gate 58.4, diffusion gate 69.2,
learned gate with budget 66.8.

For reference on the plus 1,200-set (cross-warp study, same planner and scorer): uniform B × 4 75.1 %, the hand-constructed pm6
warp 80.1 %. `oct8` reaches that with no warp and no gate.

What the gates do on the plus maze (seed 87358, 240-set): `kde` 59.6, `diff` 67.2, `geot` (band 0.25–0.75) 68.3, `geow` (1–3)
70.6, `geob` (base ×1.41) 71.7, `geo24` (24 rows / octave) 67.6, `learnb` (budget) 68.3, `learnx` (MLP on x: never closes) 72.5,
`learn_in` (learned gate + inside pairs) 72.9, `learn8` 66.7. The learned gate on u₀, started at 0.5 everywhere and trained by
nothing but the travel-time loss, opens to 0.87 at the walls and closes to 0.07 in the open cross (`report/gates_plus.png`).

## Mechanism

* Every octave field improves every local measure: speed fit 0.58 → 0.8, near-wall eikonal slope 0.34 → 0.6, direction into
  the wall 0.87 → 0.96, attraction into walls 0.077 → 0.02, local minima per true path 0.77 → 0.25; runs that arrived through a
  wall 125 → 4–69 of 720. The leaks become stalls: pressed into a thin wall 1–3 cells from the goal, descent pointing at the goal
  through the wall, T at 0.6 of the truth, no barrier along the true path. Per query: the 44 queries with detour factor 1.3–1.6 are
  solved by no field (ntrl-demo 66 wall / 45 stall of 132 runs; gated 6 wall / 105–111 stall). The far-side value of T is a
  loss property: no pair endpoint lies inside an obstacle, so the eikonal never integrates the floor speed across a wall.
* Loss-side tests (seed 87358): 20 % inside-obstacle pairs collapse ntrl-demo (60.1, wall contacts 125 → 269, attraction 0.26),
  leave the clearance gate unchanged (66.8), help the learned gate a little (72.9). A TD step of 0.05 cells (ntrl-demo's: 0.66,
  longer than the 0.1-cell walls) halves ntrl-demo's wall contacts without changing success (68.1) and makes the octave fields
  ripple (3.6–4 bumps per path, 57 %): the long TD step is both a source of walk-throughs and the regularizer of the octaves.
* Octave sweep, clearance gate: stretch 2.0 → 71.7, 3.8 → 76.5, 7.4 → 68.1, 9.0 → 67.6. Always on: ×2,×4 69.6, ×3,×6 74.6,
  ×2,×4,×8 68.9 / 82.8 / 84.2, ×2,×4,×8,×12 76.7, ×2..×16 66.0; rows 24 / 32 per octave 72.4 / 71.5, unwhitened 76.2 (within the
  seed spread). The wall barrier is complete at ×8; ×16 adds ripple, and that is the one setting where the gate pays.
* Gibson: the stall points are bowl-shaped local minima in the middle of rooms with T ≈ one cell of travel to a goal 4–25 cells
  away (T/T* 0.56, |∇T| 0.1): the metric network's far-field sinks, untouched by local resolution. At 32 cells the fixed gates
  add flat spots in rooms (their transition zone covers most of a 3–6-cell room) and lose 2–3 points; the learned gate is level.

## Verdict

The isotropy claim holds and is measurable (anisotropy 1.00). The multi-scale embedding is what helps: whitened octaves at
×2, ×4, ×8 replacing 48 of ntrl-demo's 128 rows, with no gate, give +11 points on the held-out plus maze and +10 on the disks,
zero to few wall contacts, and nothing on the Gibson apartment. Every gated variant also beats the control on the held-out set
(+4 to +6 over three seeds) but none beats the ungated ×2..×8 set; gating by geometry (band, diffusion, curvature) or by a learned
map is needed only to make too-high octaves safe in open space (at ×16 the always-on control drops to the ntrl-demo level and the
gates hold +5), and at the right top frequency it costs. The remaining plus-maze
failures (thin wall between start and goal) and the Gibson failures (room sinks) are far-field value errors that need a
non-local term on top of this embedding.

Open: a per-scene rule for the top octave (×8 at 22 cells; the Gibson scales were not swept), the octaves with the midpoint /
segment-DP losses, gap mazes.

## Gap mazes on the 10 × 10 base (2026-11-12/13)

Request (2026-11-12 evening): "run the learned gate from T, constructed gate from environment, diffusion based gate from around the
boundary of the free space and obstacle space on the plus scenarios 10x10 where the gap increases". Findings page:
https://claude.ai/artifact/L2WFDYXHvvnwiqmpuieA4N (`report/embed_gap.html`, built by `gappage.py` + `gaptext.py`, figures `gapfigs.py`,
stall locations `gapdiag.py`; tables `report/gap_tables.md`). Queue: `queue/runner.py` + `queue/jobs.txt` (121 runs, all succeeded, registry
notes on each).

Mazes: the cross-warp study's maze10 gap prepares (10 × 10 maze cut into four 5 × 5 quadrants, open band 6k cells; ×1 5877176f, ×2
01c9f536, ×4 25e73d68, ×8 cf3b0b02, ×12 98f3cab7; 300 queries, ntrl-demo's own pairs). Gates as in the main study (`geo`, `diff`, `learn`)
plus `const` (always on) and ntrl-demo, training seeds 87358 / 1 / 2. Two octave families: fixed ×2,×4,×8 (`*8` labels) and per cell
(×2,×4,×8)·E/22.1 (`*c` labels; = fixed at ×1, ×2).

Success %, three seeds pooled:

| | ×1 | ×2 | ×4 | ×8 | ×12 |
|---|---:|---:|---:|---:|---:|
| ntrl-demo | 70.2 | 45.1 | 31.1 | 25.3 | 26.4 |
| always on, per cell | 77.9 | 64.9 | 40.6 | 30.0 | 30.2 |
| clearance gate, per cell | 78.3 | 63.4 | 41.9 | 29.8 | **33.3** |
| diffusion gate, per cell | 78.7 | 60.4 | 42.0 | 30.6 | 31.6 |
| learned gate, per cell | 79.2 | 58.0 | 43.6 | 31.1 | 31.2 |
| fixed ×2..×8 (four variants) | = | = | 38.3–41.8 | 27.4–28.7 | 26.2–26.6 |

Reference (cross-warp, seed 87358, same pairs, trained on the real maze): constructed C-space warp 86.3 / 68.6 / 28.6 / 16.7 / 17.9, with
open–open 46.7 % at ×12; every embedding keeps open–open at 97–100 %.

* Per-cell octaves beat ntrl-demo at every gap (+8–9, +13–20, +10–12, +4–6, +4–7); fixed octaves fade to ntrl-demo by ×12, so the octaves
  must scale with the cell (the main study's open "top octave rule").
* Gate choice is within seed noise from ×1 to ×8; at ×12 (top octave ×29.7) the clearance gate beats always-on on each seed (33.2/31.1/35.7
  vs 30.8/29.2/30.6; local minima per path 2.05 vs 2.86). Clearance and diffusion gates keep band anisotropy 1.00; the learned gate
  rediscovers the geometry but loses isotropy on wide mazes (median band anisotropy 1.7–4.1 at ×12 per cell; with fixed octaves it half
  closes, μ 0.3–0.4, and leaks: 1,510 wall arrivals at ×12).
* Unsolved: between-quadrant queries 0–5 % from ×8 on for every field. `gapdiag.py` (seed 87358, ×12): 44–48 % of stalls never leave the
  start quadrant (moved ~0.5 cells); `report/gap_far.png`: walls carry barriers but T inside the quadrant falls straight toward the far goal
  (no detour). Data test: the cross-warp warped-maze pairs (9× more pairs with both ends in one quadrant), trained on the real maze, seed
  87358 (`*_zp` labels, prepares 4ef430ba ×12 / 0edca351 ×4): ×4 +2..+10, ×12 no change (30.7–33.8 vs 30.4–35.7; ntrl 26.9). Far-field value,
  not resolution or local pair count.
* ntrl-demo ×2 on this machine 45.1 % (3 seeds) vs cross-warp's 52.0 % (seed 87358, other GPU); ×4/×8/×12 identical to the decimal.
