# Free-space-averaged embeddings as input to Eikonal-only (P-NTFields) planning

Second phase of the Gibson embedding test (first phase: `../README.md`). Question: if the P-NTFields network is trained
with its own Eikonal loss only (no geodesic supervision) and the free-space-averaged embedding is fed as an extra
per-point input, does planning on the resulting travel-time field get better, in particular across thin walls?

## How the embeddings are generated

1. **Occupancy grid.** The Gibson mesh (`mesh_z_up_scaled.off`, already scaled per axis into the unit cube
   [-0.5, 0.5]³ by P-NTFields) is voxelised at 128³ with a conservative rasterisation: every triangle is sampled on a
   barycentric lattice at quarter-cell spacing and every cell touched is marked occupied. Two face-adjacent free cells
   therefore never straddle a surface (checked by casting 1000 random face-adjacent pairs against the exact mesh:
   0 leaks). The scan walls are single surfaces, so they come out 2 cells thick. Free cells form 53 components in
   scene 0 (largest 87 %); the sealed ones are kept, the embedding is simply computed inside each.
2. **Initial field** φ₀ on the free cells, one of: 16 channels of i.i.d. Gaussian noise (`random`), the coordinates
   themselves (`coords`, 3 channels), or the P-NTFields Fourier features sin/cos(2π x·B) with B ~ 0.5·N(0, 1)
   (`fourier`, 32 channels).
3. **Masked ball averaging.** Jacobi iteration on the GPU with r = 1 cell (the 3×3×3 box, smaller than the 2-cell walls):

       φ_{k+1}(x) = Σ_{o ∈ box} m(x+o) φ_k(x+o) / Σ_{o ∈ box} m(x+o),    m = free-cell mask, x free.

   Occupied cells never contribute and are never written, so a wall is an insulating boundary: the average on one
   side is taken only over that side. A plain blur with the same stencil but no mask is the control (`blur`).
4. **Snapshots and standardisation.** The field is stored at K = 8, 32, 128, 512, 2048, 8192 iterations (diffusion
   length ≈ √K cells), each channel standardised to zero mean and unit variance over the free cells, and the snapshots
   are concatenated. For the random init only the coarse snapshots K = 512, 2048, 8192 are kept (the fine ones are
   noise and dominate the feature distance; phase 1). For the coordinate init the stored quantity is φ_k − x, the
   drift of the free-space centroid of the ball of influence, which is what carries the wall information. The averaged
   Fourier features are stored as they are (at σ = 0.5 they are near-linear in x, so their average behaves like
   centroid-drifted coordinates with 32 different projections).
5. **Lookup.** At a query point the embedding is read by trilinear interpolation over the 8 surrounding cell centres
   with occupied corners masked out and the weights renormalised; a point inside a wall gets 0. The lookup is
   differentiable in x (piecewise linear), so the Eikonal loss and the planner gradient flow through it. The grids are
   frozen; nothing in the embedding is trained.

Channels fed to the network: `avgcoords` 18 (3 × 6 snapshots), `avgfourier` 192 (32 × 6), `random` 48 (16 × 3 coarse),
`blur` 48. Cost on a 3090 at 128³: 33 s (coords), 5 min (random / blur), 5.5 min (Fourier).

Phase-1 probe values of these grids (AUC of ‖φ(x) − φ(y)‖ at separating wall-straddling from corridor pairs; 0.5 is
chance): coordinates 0.56, Fourier 0.67, `random` 1.00, `blur` 0.48, `avgfourier` 1.00, `avgcoords` 0.99.

## Eikonal setup (P-NTFields recipe, re-implemented with autograd)

- **Network** `model.py`: the P-NTFields `NN` (256-wide Fourier input → 3 residual Softplus(β=10) encoder blocks of
  width 128 → symmetric combination by elementwise log-sum-exp max and min of the two encodings → 3 residual generator
  blocks of width 256 → sigmoid(0.1 y) = τ; travel time T(x₀, x₁) = |x₁ − x₀| / τ). Same initialisation. The only change
  is that the input is a list of feature modules, so a frozen embedding grid can be concatenated to the Fourier
  mapping (or replace it). Verified against the baseline's hand-derived passes on the pretrained checkpoint: τ
  identical, ∇τ and Δτ equal to 1e-6 relative, loss 0.18154 vs 0.18154.
- **Loss** (their `Loss`): with s_pred = 1 / (|∇_{x} T| + γ Δ_x τ) at both end points (γ = 0.001, the viscosity
  term), mean of s_pred/s + s/s_pred − 2, where s = clip(d_mesh, margin/10, margin)/margin is their speed
  (margin = 0.5/12) and is annealed as α s + 1 − α with α: 0.5 → 1.07 over the schedule.
- **Derivatives**: ∇τ and the two Laplacians Δ_{x₀}τ, Δ_{x₁}τ are computed in one forward-mode pass (`tau_jet`): every
  layer propagates (value, first derivative, diagonal second derivative) along the six input coordinates, with closed
  forms for the Fourier features (±(2πB)² f) and for the masked trilinear lookup (multilinear in each cell, so the only
  second derivative comes from the mask renormalisation, f″ = −2 f′ den′/den). This is what P-NTFields does by hand for
  the Fourier input; here it is generic over the feature list. Checked against reverse-mode autograd (double backward)
  on random nets with Fourier, Fourier + 18-channel grid, 192-channel grid and on the pretrained checkpoint: τ, ∇τ, Δτ,
  loss and all parameter gradients agree to 1e-6–1e-4 relative. Cost per 10⁴-pair step with viscosity: 70 ms for every
  input type, against 520–1800 ms for autograd through the grid lookup (which is why a first round was run with γ = 0,
  see the findings).
- **Data**: their sampler (`dataprocessing/speed_sampling_gpu.py`, run in their docker image into
  `datasets_retrain/gibson/<scene>/`): 10⁶ pairs per scene, x₀ within the margin shell of the mesh, x₁ anywhere in the
  box, exact mesh distances from BVH queries; their fixed B matrix from `datasets/gibson/<scene>/B.npy`.
- **Training** `training.py`: AdamW (lr 1e-3 → 5e-4, weight decay 0.1), 6 batches of 10⁴ pairs per epoch, 10 000
  epochs, loss scaled by 1 / previous residual, rollback to one of the five previous epoch states when the residual
  rises by more than 20 % (the rollback copies hold the trainable weights only, not the frozen grids).
- **Planner** `plan.py` (their `test/gib_plan.py`): both end points move by 0.03 · (−∇T / |∇T|²) (a step of length
  0.03 × local speed) until they are within 0.06, at most 500 steps. The polyline is checked against a 256³
  conservative occupancy grid at half-cell spacing; success = fronts met and no collision. Path length is compared with
  the grid geodesic (128³, 26-neighbour, line-of-sight diagonals).
- **Pairs**: 200 start cells with clearance ≥ margin in the largest free component. For each start one `uniform` goal
  (Euclidean distance ≥ 0.1) and up to three `wall` goals (geodesic ≥ 1.5 × Euclidean, Euclidean 0.05–0.3: locally
  separated by a wall).
- **Reference**: the released `Gib_multi` checkpoint (trained jointly on both scenes with viscosity) evaluated with
  the same pairs, once through their own `Gradient` pass and once through the autograd planner (identical results).

Variants trained per scene: `fourier` (baseline), `avgfourier` (replaces the Fourier mapping), `fourier+avgcoords`,
`fourier+random`, `fourier+blur` (control). Registered runs: see the findings section.

## Findings

Success = fronts met and polyline collision-free; "collision" = fraction of met pairs whose path crosses a wall (the
fronts met in 100 % of pairs for every model, so 1 − success = collision); len/geo = path length / grid geodesic over
the successful pairs. Scene 0: 200 uniform + 310 wall pairs; scene 1: 200 + 556. Tables from `report.py`.

### Round 1: viscosity dropped (γ = 0), registered 2026-10-28

Runs `20261028T173543Z-4fc07745` (scene 0), `20261028T173743Z-004a96ae` (scene 1, crashed after two variants: the
rollback queue deep-copied the 1.6 GB averaged-Fourier grid five times on a shared GPU; fixed afterwards) and
`20261028T174143Z-879fb4d4` (scene 0, Fourier with γ = 0.001 through autograd, 5.9 h).

| run | model | γ | uniform success | uniform collision | uniform len/geo med | uniform len/geo p90 | wall success | wall collision | wall len/geo med | wall len/geo p90 | final residual | min residual | rollbacks | train h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | pretrained Gib_multi | – | 0.980 | 0.020 | 0.972 | 1.038 | 0.842 | 0.158 | 1.087 | 1.168 | – | – | – | – |
| s0 | fourier | 0 | 0.710 | 0.290 | 0.967 | 1.110 | 0.065 | 0.935 | 1.158 | 1.407 | 0.231 | 0.0049 | 624 | 0.9 |
| s0 | fourier+avgcoords | 0 | 0.590 | 0.410 | 0.949 | 0.997 | 0.000 | 1.000 | – | – | 0.151 | 0.0052 | 1086 | 2.2 |
| s0 | avgfourier | 0 | 0.775 | 0.225 | 1.016 | 1.140 | 0.248 | 0.752 | 1.097 | 1.181 | 1.010 | 0.0322 | 734 | 1.9 |
| s0 | fourier+random | 0 | 0.535 | 0.465 | 0.943 | 0.985 | 0.000 | 1.000 | – | – | 0.200 | 0.0061 | 1228 | 1.2 |
| s0 | fourier+blur | 0 | 0.625 | 0.375 | 0.978 | 1.209 | 0.048 | 0.952 | 1.148 | 1.806 | 0.309 | 0.0062 | 958 | 0.6 |
| s0 | **fourier, γ = 0.001** | 0.001 | **0.995** | 0.005 | 0.967 | 1.030 | **0.865** | 0.135 | 1.068 | 1.133 | 0.043 | 0.0021 | 299 | 5.9 |
| s1 | pretrained Gib_multi | – | 0.510 | 0.490 | 0.977 | 1.065 | 0.149 | 0.851 | 1.117 | 1.340 | – | – | – | – |
| s1 | fourier | 0 | 0.385 | 0.615 | 0.969 | 1.103 | 0.022 | 0.978 | 1.241 | 2.315 | 0.438 | 0.0092 | 374 | 0.9 |
| s1 | fourier+avgcoords | 0 | 0.370 | 0.630 | 0.952 | 1.049 | 0.002 | 0.998 | 1.081 | 1.081 | 0.365 | 0.0097 | 1764 | 2.4 |

- **The viscosity term is what makes the field respect walls.** The same recipe with γ = 0.001 reaches 0.995 / 0.865
  on scene 0, matching the released checkpoint (0.98 / 0.84); with γ = 0 the identical network and data give 0.71 /
  0.07. Nothing else differs. Without the Laplacian the final anneal (α → 1.07, speed at the wall surface 0.037)
  destabilises training: the residual rises from 0.005 to 0.15–1.0 in the last 2 000 epochs with 600–1 800 rollbacks,
  versus 0.002 → 0.04 with 300 rollbacks for the viscous run.
- **Every failure is a wall crossing, never a lost front.** All models bring both fronts together on 100 % of pairs;
  the γ = 0 fields simply have no barrier at the thin walls, so the planner walks through them (wall success ≤ 0.25).
- **The embeddings do not substitute for viscosity.** Averaged Fourier as the only input is the best γ = 0 variant on
  wall pairs (0.25 vs 0.07 for Fourier) even though its residual is the worst; averaged coordinates and the random
  average make it worse than the Fourier baseline on both pair kinds (0.00 wall), and the unmasked blur control is
  indistinguishable from the Fourier baseline. These numbers are not a test of the embedding idea; they are a test of
  training without viscosity, and that fails for every input.

### What the objective asks for: the optimum crosses most of the "wall" pairs

The P-NTFields speed s = clip(d, m/10, m)/m is 0.1 at the mesh surface, never 0, so a thin wall is a slab that costs
about 6.6 m ≈ 0.28 time units to cross (2·[m at speed 0.1 + ∫_{m/10}^{m} m/d dd]) whereas a detour costs its extra
length at speed ≈ 1. `walltime.py` computes, per evaluation pair, the minimal travel time on the 128³ grid through free
cells only (T_free) and through all cells with occupied cells at speed 0.1 (T_all); when T_all < T_free the viscosity
solution of the Eikonal equation crosses a wall and a planner that follows it collides by construction
(`results/gibson_embed_eikonal_analysis/walltime_round1_*.json`).

| scene | pairs | optimum crosses a wall | T_free / T_all (median, crossing pairs) | pretrained success: optimum detours / crosses | viscous Fourier: detours / crosses | γ = 0 Fourier: detours / crosses |
|---|---|---|---|---|---|---|
| 0 | uniform (200) | 9.5 % | 1.12 | 1.000 / 0.789 | 1.000 / 0.947 | 0.740 / 0.421 |
| 0 | wall (310) | 72.3 % | 1.34 | 1.000 / 0.781 | 1.000 / 0.812 | 0.163 / 0.027 |
| 1 | uniform (200) | 49.5 % | 1.53 | 0.941 / 0.071 | – | 0.743 / 0.020 |
| 1 | wall (556) | 91.0 % | 2.29 | 0.940 / 0.071 | – | 0.200 / 0.004 |

- The pretrained checkpoint and the viscous Fourier net are collision-free on **100 %** of scene 0 pairs whose optimum
  detours, and on 94 % in scene 1. Their "failures" are almost all pairs where the objective itself prefers the
  crossing; in scene 1 (rooms a few margins apart, detour 2.3× the crossing) the checkpoint crosses on 93 % of them,
  which is the Eikonal answer, not a model error. The collision-free success ceiling of a faithful Eikonal planner is
  therefore ≈ 0.90 / 0.28 (scene 0 uniform / wall) and ≈ 0.50 / 0.09 (scene 1); the viscous nets beat it only because
  they smooth the crossing valley away on part of the crossing-optimal pairs (scene 0: 81 % of them).
- The γ = 0 nets fail on both kinds of pairs (16 % success on the detour-optimal wall pairs), which is the model error
  described above, separate from the objective's preference.
- Consequence for this study: "success" mixes model fidelity with the speed model's willingness to cross. The
  embeddings can only be judged on (i) the detour-optimal pairs and (ii) how close the planned path's travel time is to
  the optimum (`analysis.py`, which re-plans with the saved models and integrates 1/s along the path).

**Path travel time against the optimum** (`analysis.py`, `results/gibson_embed_eikonal_analysis/round1_*/`): time/opt
is the travel time of the planned polyline under the speed model (1/s integrated on the 256³ grid) divided by T_all
from the 128³ grid; medians over the pairs whose fronts met. The 128³ optimum is a few per cent too high (26-neighbour
metric, thicker walls), which is why faithful models sit at 0.94–0.97 rather than 1; only differences between models
are meaningful.

| scene | model | γ | wall success | success, detour-optimal (n) | success, crossing-optimal (n) | time/opt median | time/opt p90 | time/opt detour-opt | time/opt crossing-opt | within 5 % |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | pretrained | – | 0.839 | 1.000 (86) | 0.777 (224) | 1.064 | 1.461 | 0.968 | 1.140 | 0.455 |
| 0 | fourier | 0.001 | 0.865 | 1.000 (86) | 0.812 (224) | 1.054 | 1.389 | 0.960 | 1.142 | 0.487 |
| 0 | fourier | 0 | 0.065 | 0.163 (86) | 0.027 (224) | 1.506 | 3.483 | 1.904 | 1.361 | 0.061 |
| 0 | fourier+avgcoords | 0 | 0.000 | 0.000 (86) | 0.000 (224) | 1.069 | 1.472 | 1.323 | 1.047 | 0.397 |
| 0 | avgfourier | 0 | 0.248 | 0.523 (86) | 0.143 (224) | 1.392 | 2.114 | 1.370 | 1.393 | 0.087 |
| 0 | fourier+random | 0 | 0.000 | 0.000 (86) | 0.000 (224) | 1.081 | 1.596 | 1.256 | 1.052 | 0.361 |
| 0 | fourier+blur | 0 | 0.048 | 0.151 (86) | 0.009 (224) | 1.573 | 3.939 | 1.913 | 1.463 | 0.094 |
| 1 | pretrained | – | 0.146 | 0.960 (50) | 0.065 (506) | 1.050 | 1.225 | 0.972 | 1.057 | 0.500 |
| 1 | fourier | 0 | 0.022 | 0.200 (50) | 0.004 (506) | 1.128 | 1.958 | 1.694 | 1.115 | 0.243 |
| 1 | fourier+avgcoords | 0 | 0.002 | 0.020 (50) | 0.000 (506) | 1.140 | 2.143 | 1.551 | 1.126 | 0.272 |

(uniform pairs: pretrained 0.939 / 0.936 / 1.015, viscous Fourier 0.935 / 0.934 / 1.064, γ = 0 Fourier 1.024 / 1.003 /
1.428, γ = 0 Fourier+avgcoords 0.981 / 0.965 / 1.032, blur 1.185 / 1.127 / 1.444 for median / detour-opt / crossing-opt.)

- The viscous nets follow the optimum where it detours (0.96–0.97, the grid bias) and pay 14 % extra time where it
  crosses, because they detour anyway: the sharp crossing valley is smoothed out, which is what makes them
  collision-free there. The pretrained checkpoint in scene 1 (crossing 93 % of the time) is within 6 % of the optimum.
- At γ = 0 the Fourier net is not a solution at all: 1.9× the optimal time on the detour-optimal wall pairs (it crosses
  a 0.28-time-unit slab instead of taking a shorter detour) and 1.36× on the crossing ones.
- **The masked embeddings do carry the wall into the γ = 0 field, "success" just could not see it.** With averaged
  coordinates or the random average concatenated, the γ = 0 net is within 5 % of the optimum on the crossing-optimal
  wall pairs (1.047 / 1.052 vs 1.36 for Fourier) and 1.26–1.32 instead of 1.90 on the detour-optimal ones, while the
  unmasked blur control stays at the Fourier level (1.46 / 1.91). Their success is 0.000 because they do what the
  objective says and cross; the γ = 0 Fourier net also crosses, but far from optimally. Averaged Fourier alone is in
  between (1.39 / 1.37). This is the sharpness the embedding was built for, showing up as fidelity to a sharp optimum,
  and it is the reason round 2 (viscosity on) is needed to tell whether that fidelity survives, and helps, when the
  training itself is stable.

### Round 2: viscosity kept (γ = 0.001, forward-mode derivatives), registered 2026-10-29

Runs `20261029T005220Z-bbfd6e1c` (scene 0) and `20261029T005350Z-29e084ea` (scene 1), both on the 3090 in parallel
(≈ 3 h per variant). Both were stopped after 17.5 h: their last variants (`fourier+blur` in scene 0, `fourier+random`
in scene 1) had diverged to NaN early in the α = 0.5 phase and were grinding through the rollback loop. The completed
variants are in `results.json`; the registry marks the runs failed (exit −15).

| scene | model | γ | uniform success | uniform collision | wall success | wall collision | residual at α=0.5 (epoch 2000) | final residual | min residual | rollbacks | train h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | pretrained Gib_multi | – | 0.980 | 0.020 | 0.842 | 0.158 | – | – | – | – | – |
| 0 | **fourier** | 0.001 | **0.995** | 0.005 | **0.858** | 0.142 | 0.002 | 0.045 | 0.0020 | 310 | 2.9 |
| 0 | avgfourier | 0.001 | 0.280 | 0.720 | 0.000 | 1.000 | 0.082 | 2.072 | 0.067 | 447 | 3.2 |
| 0 | fourier+avgcoords | 0.001 | 0.290 | 0.710 | 0.000 | 1.000 | 0.076 | 2.581 | 0.071 | 461 | 3.1 |
| 0 | fourier+random | 0.001 | 0.360 | 0.640 | 0.003 | 0.997 | 0.011 | 1.387 | 0.011 | 2868 | 4.1 |
| 0 | fourier+blur | 0.001 | – | – | – | – | NaN from epoch ≈ 1200 | – | 0.059 | 11082 at epoch 2300 | stopped |
| 1 | pretrained Gib_multi | – | 0.510 | 0.490 | 0.149 | 0.851 | – | – | – | – | – |
| 1 | **fourier** | 0.001 | **0.520** | 0.480 | **0.149** | 0.851 | 0.004 | 0.100 | 0.0038 | 88 | 2.8 |
| 1 | avgfourier | 0.001 | 0.220 | 0.780 | 0.000 | 1.000 | 0.082 | 2.247 | 0.063 | 1030 | 3.4 |
| 1 | fourier+avgcoords | 0.001 | 0.235 | 0.765 | 0.000 | 1.000 | 0.084 | 2.456 | 0.063 | 1391 | 3.5 |
| 1 | fourier+random | 0.001 | – | – | – | – | NaN from epoch ≈ 1500 | – | 0.103 | 20882 at epoch 3200 | stopped |

- **The retrained Fourier baseline reproduces the released checkpoint in both scenes** (0.995 / 0.858 and 0.520 /
  0.149 vs 0.98 / 0.84 and 0.51 / 0.15), so the forward-mode implementation of the recipe is faithful and the
  scene 1 numbers are the ceiling set by the speed model (see above), not a training problem.
- **With viscosity on, every grid-feature variant fails to fit the Eikonal equation from the start.** During the
  α = 0.5 phase the Fourier net reaches a residual of 0.002–0.004; the averaged-coordinate and averaged-Fourier nets
  plateau at 0.08 (30× higher) and the random average at 0.01, and from there the residual climbs monotonically with α
  to 1.4–2.6 (the Fourier net ends at 0.05–0.10). Two variants diverged to NaN while α was still 0.5. Planning on these
  fields is worse than at γ = 0 (uniform 0.22–0.36, wall 0).
- **Round 1 vs round 2 isolates the conflict.** Without viscosity the same grid inputs fit the α = 0.5 phase as well as
  Fourier (min residual 0.005–0.006 vs 0.005) and only lost it in the anneal; with viscosity they never fit. So the
  grid inputs are not incompatible with the Eikonal residual, they are incompatible with the Laplacian term.
- **Mechanism, measured** (`scratchpad/visc_diag.py`, scene 0, 10⁴ training pairs): the Fourier features have
  |∂f/∂x| rms 4.3 and |∂²f/∂x²| rms 47 per unit length; the averaged-coordinate grid has 39 and 27 600 (99th
  percentile of |∂f/∂x| 167: the fine snapshots K = 8–128 change across single cells), the coarse random grid 3.5 and
  1 600. The large second derivatives are the mask renormalisation of the trilinear lookup, f″ = −2 f′ den′/den, in
  every cell that touches a wall; the lookup is also only C⁰ across cell faces, so the true Laplacian of τ contains
  deltas the forward-mode pass does not even see. Through J^T H J and these terms the Laplacian of τ at wall-adjacent
  points is dominated by the input, not by what the network learns. The trained outcome is a collapse of curvature:
  the viscous Fourier net ends with |Δτ| median 188 (the viscosity term is 7 % of the first-order term at the median,
  20 % at p90, 32 % at p99, so it is doing real work near walls), the viscous averaged-coordinate net with |Δτ|
  median 0.2 and a viscosity term of 0.0 % — it has flattened τ to make the penalty go away and can no longer form
  the sharp front the anneal demands. The random-grid net kept curvature (p99 of |Δτ| 706) but the central-difference
  Laplacian at 2 cells is 3× its exact Laplacian, i.e. τ is kinked at cell faces, and training needed 2 868 rollbacks.
- **Fields** (`fields.py`, `results/gibson_embed_eikonal_analysis/fields_scene0.png`: T from the start over the start's
  z-slice, walls in black, planned path white if collision-free, red if not; top row a wall pair whose optimum
  detours, bottom row one whose optimum crosses). The pretrained and viscous-Fourier fields jump across the wall and
  wrap around its end, and the planner detours in both rows. The γ = 0 Fourier field has no wall structure at all (the
  path zig-zags through it). The γ = 0 averaged-coordinate field has the wall imprinted as a sharp step, which is why
  its paths are near time-optimal, but the step is a crossing cost rather than a barrier and the planner goes
  straight through in both rows. The viscous averaged-coordinate field is a set of near-concentric circles: the wall
  is gone, which is the curvature collapse measured above.
**Path travel time, round 2** (`results/gibson_embed_eikonal_analysis/round2_scene{0,1}/`, wall pairs, medians):

| scene | model | success, detour-optimal | time/opt detour-opt | time/opt crossing-opt | within 5 % |
|---|---|---|---|---|---|
| 0 | pretrained | 1.000 | 0.968 | 1.140 | 0.455 |
| 0 | fourier | 1.000 | 0.970 | 1.159 | 0.429 |
| 0 | fourier+avgcoords | 0.000 | 1.294 | 1.062 | 0.303 |
| 0 | fourier+random | 0.012 | 1.340 | 1.104 | 0.113 |
| 1 | pretrained | 0.960 | 0.972 | 1.057 | 0.500 |
| 1 | fourier | 0.960 | 0.977 | 1.042 | 0.572 |
| 1 | fourier+avgcoords | 0.000 | 1.449 | 1.138 | 0.245 |

The retrained Fourier net matches the checkpoint on path time as well as on success (0.97 of the grid optimum on
detour-optimal pairs in both scenes); the viscous grid variants are 1.3–1.45× the optimum where it detours.

### Controls (scene 0, 2 000-epoch compressed schedule), registered 2026-10-29

Runs `20261029T184654Z-8b8dfbc8` (`--mode fd`: central-difference Laplacian at 1 cell, which sees the kinks of the
lookup but not the renormalisation deltas), `20261029T184754Z-5d51ffa3` (forward-mode Laplacian, same schedule) and
`20261029T184854Z-736d11a5` (γ = 10⁻⁴). The anneal starts at epoch 400 instead of 2 000. The forward-mode
`fourier+random` run reproduces the round-2 failure on this schedule, so the schedule is a fair proxy.

| control | variant | residual at epoch 400 (end of α = 0.5) | final residual | rollbacks | uniform success | wall success |
|---|---|---|---|---|---|---|
| jet, γ = 1e-3 | fourier | 0.0041 | 0.084 | 20 | 0.965 | 0.790 |
| jet, γ = 1e-3 | fourier+random | 0.072 | 2.415 | 379 | 0.290 | 0.000 |
| fd 1 cell, γ = 1e-3 | fourier | 0.0044 | 0.207 | 22 | 0.950 | 0.548 |
| fd 1 cell, γ = 1e-3 | fourier+random | 0.207 | 5.670 | 31 | 0.290 | 0.000 |
| fd 1 cell, γ = 1e-3 | fourier+avgcoords | 0.054 | 0.252 | 361 | 0.820 | 0.171 |
| jet, γ = 1e-4 | fourier+random | 0.016 | 0.277 | 748 | 0.565 | 0.019 |
| jet, γ = 1e-4 | fourier+avgcoords | 0.044 | 0.334 | 318 | 0.580 | 0.003 |

- **Finite-difference viscosity hurts Fourier and the random grid.** The central difference at 1 cell sees the kinks
  of the piecewise-linear lookup directly: `fourier+random` starts its flat phase at 0.21, 3× worse than with the exact
  Laplacian (0.07), and diverges to 5.7. It also degrades the Fourier baseline (wall 0.55 vs 0.79): the difference is
  a biased Laplacian for the high-frequency Fourier features.
- **fd is the one partial exception for averaged coordinates.** `fourier+avgcoords` ends at 0.25 (2.58 with the exact
  Laplacian on the long schedule) and plans at 0.82 / 0.17: still 3× the baseline residual and far below its 0.79 wall
  success, but the only viscous grid variant whose field keeps any wall structure. Plausibly the 1-cell difference
  averages the renormalisation singularity over a cell where the exact pass evaluates it pointwise, while for the
  random grid the fine-scale kinks dominate and the difference sees them in full. Path-time tables for the control
  models: `results/gibson_embed_eikonal_analysis/ctl_fd_scene0/`, `ctl_g4_scene0/`.
- **γ = 10⁻⁴ moves the plateau, not the outcome.** The grid variants reach 0.016 / 0.044 in the flat phase (0.07 /
  0.08 at γ = 1e-3, 0.004 for Fourier) but climb through the anneal to 0.28 / 0.33 with 748 / 318 rollbacks; planning
  0.565 / 0.019 and 0.580 / 0.003.
- The compressed schedule costs the baseline something (0.084 / 0.79 vs 0.045 / 0.858 at 10 000 epochs); comparisons
  are within the schedule.

**Path travel time, control models** (`results/gibson_embed_eikonal_analysis/ctl_{fd,g4}_scene0/`, wall pairs,
medians):

| control | model | success, detour-opt (86) | success, crossing-opt (224) | time/opt detour-opt | time/opt crossing-opt | within 5 % |
|---|---|---|---|---|---|---|
| – | pretrained | 1.000 | 0.777 | 0.968 | 1.140 | 0.455 |
| fd, γ = 1e-3 | fourier | 0.988 | 0.379 | 1.006 | 1.108 | 0.335 |
| fd, γ = 1e-3 | fourier+avgcoords | 0.605 | 0.004 | 1.137 | 1.056 | 0.419 |
| fd, γ = 1e-3 | fourier+random | 0.000 | 0.000 | 1.305 | 1.064 | 0.316 |
| jet, γ = 1e-4 | fourier+random | 0.070 | 0.000 | 1.285 | 1.062 | 0.300 |
| jet, γ = 1e-4 | fourier+avgcoords | 0.012 | 0.000 | 1.366 | 1.073 | 0.255 |

- The fd averaged-coordinate net is half faithful: it detours on 60 % of the detour-optimal wall pairs at 14 % extra
  time and crosses on 99.6 % of the crossing-optimal pairs within 6 % of the optimum, so its within-5 % fraction
  (0.42) is above the same-schedule Fourier net (0.34) and near the checkpoint (0.46). Its wall is a cost the
  optimiser sometimes pays, where the Fourier net's wall is a barrier it detours around (0.99 on detour-optimal pairs).
- Every other control model behaves like the round-1 grid nets: near-optimal where crossing is optimal (1.06–1.07),
  crossing anyway where a detour would be faster (1.29–1.37× the optimum, 0–7 % success). Lowering γ did not change that.

## Running

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/gibson_embed/eikonal/run.py --scene 0
# or directly
python jeffrey/gibson_embed/eikonal/pipeline.py --scene 0 --out <dir> [--variants fourier,fourier+avgcoords,...] [--epochs N]
```
