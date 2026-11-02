# Free-space ball-average embedding of Gibson scenes

Test of the idea: embed the sampled obstacle space of a Gibson scene, then iterate the free-space ball average
`phi_{k+1}(x) = mean of phi_k over B_r(x) ∩ C_free`, with `r` below the thinnest wall and many iterations, so that
information travels along corridors but never through walls, and the embedding acquires sharp features at thin walls.
The second half of the test asks whether such an embedding helps a P-NTFields-style network predict geodesic distance
with sharp jumps across walls.

Everything is in this folder and runs on the host venv on GPU 1 (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`).
Dates: 2026-10-27.

## Setup

| piece | file | what it does |
|---|---|---|
| occupancy grid | `grid.py` | Gibson mesh (per-axis scaled to the unit cube) sliced at z = 0 (2-D, 512²) or voxelised (3-D, 128³) with conservative rasterisation (quarter-cell sampling), so two face-adjacent free cells never straddle a surface. A leak check against the exact mesh confirms this (0 of 3000 / 1500 pairs cut). Scan walls are single surfaces → 2 cells thick. |
| averaging | `diffuse.py` | Jacobi iteration of the masked ball average on the GPU: `phi ← Σ_o m_o·shift(phi, o) / Σ_o m_o` over the free cells in the ball, `r = 1` cell (unit box) by default. Snapshots at K = 8, 32, 128, 512, 2048, 8192 are standardised per channel over free cells and concatenated (diffusion length ≈ √K cells). Options: `masked=False` (plain blur, the control), `r` ≥ wall thickness (leaky), line-of-sight masks for longer offsets, inits random / Fourier (P-NTFields σ = 0.5) / coordinates, and a QR-orthogonalised spectral variant (≈ Laplacian eigenmaps of the free-space graph). |
| ground truth | `truth.py` | geodesic distance from each source by GPU min-plus relaxation, 8/26 neighbours, diagonals only with line of sight. 96 sources in the largest free component, 3 cells ≤ clearance ≤ 0.05 units (near walls, where the sharp pairs are); 24 held out. |
| probe | `probe.py` | no training: local pairs (≤ 30 cells), *wall* pairs (geodesic ≥ 2.5 × Euclid) vs *corridor* pairs (≤ 1.15 ×); AUC of ‖φ(x) − φ(y)‖ as a wall detector, stratified by Euclidean distance so that pair length cannot leak. |
| network | `train.py` | P-NTFields shape: per-point encoder on [x, features(x)] → symmetric [h_s + h_x, |h_s − h_x|] → generator → ρ = log(d_geo / ‖s − x‖), L1 loss, 4000 Adam steps of 8192 pairs (half local). Discontinuity is built into the *input*: frozen grids are read with a masked bilinear/trilinear lookup that uses free corner cells only, so a wall is a jump of the features. Variants: coords, Fourier σ = 0.5 / 4, each averaged embedding, a learnable masked grid (random or embedding init), Fourier + embedding. |
| evaluation | `train.py` | held-out sources: mean |log(pred / true)| over all pairs, wall pairs, corridor pairs; recall / false-alarm of wall pairs at predicted ratio ≥ 2. Training-source sweep 72 / 24 / 8. |

Scenes: `baselines/P-NTFields/datasets/gibson/{0,1}/mesh_z_up_scaled.off`. Scene 0 at z = 0 is mostly open floor with a few
wall segments and sealed boxes (17 free components, largest 88 %); scene 1 is a connected apartment of rooms and corridors
(6 components, largest 99.6 %).

## Result 1: the averaged embedding separates the two sides of a wall without any training

AUC of ‖φ(x) − φ(y)‖ for wall vs corridor pairs (0.5 = chance):

| embedding | scene 0, 2-D | scene 1, 2-D | scene 0, 3-D |
|---|---|---|---|
| coordinates | 0.540 | 0.523 | 0.561 |
| Fourier σ = 0.5 (P-NTFields input) | 0.523 | 0.528 | 0.673 |
| masked average, random init, r = 1, all scales (**the idea**) | 0.864 | 0.942 | 0.976 |
| same, coarse scales only (K = 512, 2048, 8192) | **0.998** | **0.993** | – |
| control: unmasked blur (ball reaches through walls) | 0.409 | 0.508 | 0.478 |
| control: masked but r = 3 cells ≥ wall thickness (leaky) | 0.543 | 0.611 | – |
| r = 3 with line-of-sight masks | 0.980 | 0.980 | 0.990 |
| masked average of the Fourier features | 0.999 | 0.973 | 1.000 |
| masked average of the coordinates (centroid drift) | 0.970 | 0.862 | 0.986 |
| spectral (orthogonalised subspace iteration) | 1.000 | 1.000 | – |

Both conditions of the note are confirmed by the controls: the unmasked blur and the leaky ball (r ≥ wall thickness) are at
chance, and the leak is fixed either by r < thickness or by line-of-sight masking of the long offsets. Many iterations are
needed, and the fine scales are pure noise (per-snapshot probe of the random-init embedding):

| K | 8 | 32 | 128 | 512 | 2048 | 8192 | all concatenated |
|---|---|---|---|---|---|---|---|
| AUC, scene 0 | 0.49 | 0.56 | 0.66 | 0.86 | 1.00 | 1.00 | 0.86 |
| AUC, scene 1 | 0.58 | 0.63 | 0.75 | 0.93 | 0.99 | 1.00 | 0.94 |
| wall / corridor feature distance, scene 1 | 1.05 | 1.12 | 1.34 | 2.11 | 4.50 | 10.6 | 1.67 |

One step only knows about distance r; K ≥ 2048 (diffusion length ≈ 45 cells ≈ a room width) is where the two sides of a
wall decorrelate while a corridor stays continuous. Concatenating all scales *lowers* the AUC because the noise of the fine
scales dominates the feature distance; the coarse-only embedding is what should be used. `figures.py` renders the
channels (`embedding.png`): noise at K ≤ 128, smooth fields with a visible jump across every thin wall at K ≥ 2048.

A side finding, contrary to my first expectation: averaging the smooth P-NTFields Fourier features gives the sharpest
embedding of all. At σ = 0.5 those features are near-linear in x, and the masked average of a linear field accumulates the
free-space centroid drift (the same mechanism as the averaged coordinates, 0.97 / 0.86): points next to a wall are pushed
away from it, in opposite directions on the two sides.

## Result 2: downstream, the embedding helps when sources are scarce or the scene is complex; averaging smooth features beats averaging noise

Held-out sources, mean |log(pred / true)|: all pairs / wall pairs, with the number of training sources (dev runs; the
registered runs below reproduce the configuration).

| input | scene 0 (2-D), 72 | 24 | 8 | scene 1 (2-D), 72 | 24 | 8 |
|---|---|---|---|---|---|---|
| coordinates | **0.005 / 0.012** | – | – | 0.101 / 0.283 | – | – |
| Fourier σ = 0.5 | 0.007 / 0.024 | **0.036 / 0.126** | 0.166 / 0.326 ¹ | 0.054 / 0.158 | 0.147 / 0.601 | 0.338 / 1.218 |
| masked average, random init, all scales | 0.015 / 0.044 | 0.036 / 0.258 | 0.121 / 0.625 | 0.045 / 0.177 | 0.129 / 0.537 | 0.279 / 1.213 |
| masked average, random init, coarse scales | 0.016 / 0.052 | 0.032 / 0.144 | 0.093 / 0.408 | 0.040 / 0.146 | 0.111 / 0.448 | 0.270 / 1.192 |
| control: unmasked blur | 0.016 / 0.068 | 0.032 / 0.205 | 0.126 / 0.955 | 0.040 / 0.115 | 0.139 / 0.576 | 0.306 / 1.400 |
| masked average of Fourier | 0.007 / 0.029 | 0.034 / 0.167 | 0.147 / 0.361 ¹ | **0.020 / 0.044** | **0.057 / 0.140** | **0.212 / 0.834** |
| masked average of coordinates | 0.007 / 0.037 | 0.038 / 0.184 | 0.084 / 0.624 | 0.033 / 0.087 | 0.094 / 0.289 | 0.266 / 1.009 |
| spectral | 0.015 / 0.061 | 0.034 / 0.218 | 0.083 / 0.330 | 0.042 / 0.158 | 0.104 / 0.383 | 0.257 / 1.134 |
| learnable masked grid, random init | 0.007 / 0.047 | 0.043 / 0.628 | 0.254 / 1.366 | 0.027 / 0.071 | 0.094 / 0.332 | 0.274 / 1.041 |
| Fourier + random-init average | 0.010 / 0.055 | 0.036 / 0.186 | **0.077 / 0.314** | 0.024 / 0.078 | 0.070 / 0.255 | 0.202 / 0.893 |

¹ corridor error 0.27–0.33 and false-alarm rate 0.2: with 8 sources the Fourier nets predict detours everywhere; every
averaged embedding keeps the corridor error at 0.03–0.06.

What this says:

- **Simple scene, dense sources: nothing beats coordinates.** In scene 0 (and in 3-D scene 0: coords 0.006 / 0.023,
  Fourier 0.010 / 0.053, random-init average 0.017 / 0.131) the 72 near-wall sources already show the network every wall
  from both sides, and a coordinate MLP with Softplus(β = 10) fits the jumps on held-out sources. A frozen random
  embedding only adds inputs to memorise: its training error is as low as anyone's but held-out wall error is 2–5× worse.
- **Complex scene or few sources: the averaged Fourier features win at every source count.** Scene 1: 2.7× lower error
  than plain Fourier with 72 sources (3.6× on wall pairs), 2.6× with 24 (4.3× on walls), 1.6× with 8. Mechanism: the
  averaged Fourier/coordinate features are a *geometrically consistent* coordinate system (position plus a drift away from
  nearby walls, discontinuous across them), so the network only has to learn a metric on them. Averaged random channels
  are arbitrary smooth functions that the network must first decode into geometry.
- **The literal idea (average random features) beats its blur control only in 2-D, only when sources are scarce, and
  only with the fine scales dropped**: scene 0 wall error 0.144 vs 0.205 (24 sources) and 0.408 vs 0.955 (8 sources);
  scene 1 0.448 vs 0.576 and 1.19 vs 1.40. With all scales the random-init embedding is no better than the blur
  downstream even though its probe AUC is far higher (0.86–0.94 vs 0.41–0.51): the fine-scale channels let the network
  memorise sources. In 3-D it never beats the blur (table below).
- **8 sources is below what any input needs to learn walls** (wall log-error ≈ 1 = factor e), but the embeddings fail
  gracefully (corridor error stays low) while Fourier predicts detours everywhere.
- The learnable masked grid (9 M parameters) matches the best frozen inputs at 72 sources and collapses below that.
  Initialising it from the fine-scale channels of the embedding was a trap (held-out error 0.145); from the coarse
  channels it behaves like the random init.

3-D (scene 0, 128³), all / wall error with 72 → 24 → 8 sources:

| input | 72 | 24 | 8 |
|---|---|---|---|
| coordinates | **0.006 / 0.023** | – | – |
| Fourier σ = 0.5 | 0.010 / 0.053 | 0.033 / 0.245 | 0.064 / 0.636 |
| masked average, random init, all scales | 0.017 / 0.131 | 0.034 / 0.426 | 0.074 / 0.678 |
| masked average, random init, coarse scales | 0.019 / 0.143 | 0.035 / 0.482 | 0.077 / 0.783 |
| control: unmasked blur | 0.016 / 0.114 | 0.034 / 0.306 | 0.064 / 0.811 |
| masked average of Fourier | 0.012 / 0.059 | 0.032 / 0.332 | 0.066 / 0.633 |
| masked average of coordinates | 0.008 / 0.068 | **0.020 / 0.173** | 0.069 / **0.405** |
| Fourier + random-init average | 0.010 / 0.081 | 0.034 / 0.300 | 0.062 / 0.707 |

In 3-D the random-init average never beats the blur control downstream, coarse scales or not, although its probe AUC is
0.98–1.00 against 0.48: wall pairs are rare in 3-D (0.4 % of local pairs) and the network does not get enough of them
to decode arbitrary smooth channels into geometry. The averaged coordinates (18 channels of free-space centroid drift)
transfer instead: 1.7× lower error than Fourier at 24 sources and 1.6× lower wall error at 8.

## Pitfalls found on the way (all fixed in the code)

- A Softplus output head dies under the L1 loss (median target ρ ≈ 0.07): the pre-activation drifts to −16 within ten
  steps and the gradient is exactly zero, so every variant trains to the same flat loss. The head is now linear;
  predictions are clamped at 0 only in evaluation.
- Uniformly random sources sit in open space (6 of 8 had no wall-straddling local pair in 3-D); sources are now drawn
  with clearance between 3 cells and 0.05 units.
- `pkill -f` / `pgrep -f` with a pattern that appears in your own command line kills your shell; use `pattern[.]py`.
- Concatenating multi-scale snapshots looks harmless but the fine scales dominate both the feature distance and what the
  network can overfit; keep K ≥ 512.

## Running

```
# full pipeline into a directory (2-D scene 0, ~80 min on a shared 3090 with the 72/24/8 sweep)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 systemd-run --user --scope -p MemoryMax=8G --quiet -- \
  ~/.venvs/bellman-ntfield/bin/python jeffrey/gibson_embed/pipeline.py \
  --mesh baselines/P-NTFields/datasets/gibson/0/mesh_z_up_scaled.off --dim 2 --res 512 --out <dir> --train-counts 24,8
# registered run (writes specs/gibson-embed_*.json, output in results/registered_<id>/)
~/.venvs/bellman-ntfield/bin/python jeffrey/gibson_embed/run.py --scene 0 --dim 2 --res 512 --extra "--train-counts 24,8"
# reports and figures
python jeffrey/gibson_embed/report.py <dir>            # markdown tables (probe, per-scale, downstream)
python jeffrey/gibson_embed/scales.py <dir>            # per-snapshot probe of the saved embedding
python jeffrey/gibson_embed/figures.py <dir> --variants coords,fourier,diff-random,diff-fourier,blur-random,learned-grid
```

Registered runs (host venv, torch 2.7.1+cu128, RTX 3090): scene 0 2-D `20261027T224652Z-58ce777e`, scene 1 2-D
`20261027T224700Z-fe61fdf7`, scene 0 3-D `20261027T233404Z-109e29b6`, plus the coarse-scale-only downstream rows for
2-D (left out of the 2-D default variant list by mistake) as `20261028T002336Z-ff7cf94a` (scene 0) and
`20261028T002344Z-dbadad05` (scene 1); their tables are in `registered_runs.md` in this folder (generated with
`report.py` when the runs finished). The registry
verifies the hashes of the source files during a run, so do not edit the pipeline sources while one is in progress.

## What I did not do, and would need direction on

- An Eikonal-loss (P-NTFields-style) training of the same inputs: the speed field would have to be near zero inside the
  2-cell walls and the PDE becomes stiff; the pair-regression on exact geodesics isolates the input question instead.
- Using the embedding inside the frozen Bellman600 pipeline; nothing here touches it.
