# Stretching SE(3) before the encoder: a learned warp for neural travel-time fields

**Summary.**
* **The idea:** put a learned diffeomorphism φ: SE(3) → SE(3) between the pose and the Fourier encoder of a travel-time
  network. φ stretches space next to obstacles, so the same smooth encoder can make the field sharp there, and compresses
  open free space, where there is little to represent.
* **Status:** implemented and tested on 2026-11-02 / 03 on the lifted homotopy head (hlift) together with the relative normal
  loss (RNL).
* **Results so far:**
  * It trains stably when fine-tuned from a trained field and learns the free-space compression.
  * In a house scene it makes the field steeper at the walls (closer to the target slope).
  * It does not improve planning over the RNL alone: the sharper field makes the forward MPPI stall more often.
  * One test is still running: warp + central-difference RNL against the central RNL alone.

Everything below is in this repository (G-NTFields). Code:
`models/regular_planning/graph_method_triangle/warp_decoder.py` and `models/regular_planning/hlift_rnl/warp_losses.py`.
Chronological log: `EXPERIMENTS.md`, entries W0-W10.

---

## 1. Background needed to read this

* **Problem.** Plan the motion of a rigid body among obstacles.
  * A pose is x = (t, R) ∈ SE(3).
  * Coordinates are normalised: the scene is scaled into [-0.5, 0.5]³, so the scene is about 1 unit wide.
  * Rotations are handled in the Lie algebra. The metric on twists (v, ω) is |v|² + ρ²|ω|² with ρ = 1/(2π), so one full turn
    counts as one unit of length.
* **Travel-time field.** A network learns T(x, y), the travel time from a start pose x to a goal pose y, from the Eikonal
  equation |c(x)| = 1 / S*(x).
  * c(x) is the Lie gradient of T with respect to the start pose: a 6-vector, translation part in the body frame.
  * S*(x) is a clearance speed label: 1 when the body is at least `margin` away from every obstacle, falling to a small
    floor at contact.
  * The speed the network implies is S(x) = 1 / |c(x)|.
* **Planner and success.** A forward MPPI descends T(·, y) from x to y.
  * A plan succeeds only if the MPPI converged and an exact collision check of the placed body passes at every waypoint.
  * Failures are **collision** (the path goes through an obstacle) or **not converged** (a stall: the planner is stuck in a
    local minimum of T).
  * The two can overlap: a stalled plan can also collide.
* **hlift head** (`graph_homotopy_lift`). The head reads a pose through both unit-quaternion lifts (t, q) and (t, -q) of its
  rotation, using the double cover S³ → SO(3).
  * Encoder f: Fourier features followed by Lipschitz-capped layers. It is smooth: its wavelengths are on the order of the
    whole scene.
  * Embeddings: e_even = ½(f(t, q) + f(t, -q)) and e_odd = ½(f(t, q) - f(t, -q)).
  * The two lifts give two sheets. These are two homotopy classes of rotation paths, and the dual planner descends both.
  * A closed-form straight-line ("cone") term is computed from the true poses and added to the learned part.
* **RNL** (relative normal loss; Shape-NTFields, arXiv 2610.22762, Sec. IV-C; this package's `README.md`).
  * At a band pose x0 (clearance below the margin), take a short step along a twist ξ that closes clearance, on the exp
    curve x0·exp(εξ) lifted to both quaternion branches.
  * Require the network speed to fall at least as fast as the target does:

    ```
    G = (S(x0·exp(εξ)) - S(x0)) / ε
    L = 1[S*(x0) < 1] · (ρ_th - clip(G, ρ_th, G_max))²      ρ_th = the target speed's own slope along ξ (negative)
    ```

  * In the production version S(x0), the "anchor", is detached. The "central" version differentiates both ends.
  * Fine-tuned into trained hlift fields with the SE(3)-normal probe (`nrm`), the RNL raises planning success with plain
    MPPI in every group: loop 78.0 vs 73.5 %, Gibson 76.3 vs 71.2 %. See `README.md`.

## 2. The idea (as proposed)

1. Before the Fourier embedding, pass every pose through a function φ: SE(3) → SE(3) that is a diffeomorphism.
2. Near obstacles φ stretches space, so the chain SE(3) →φ SE(3) → embedding → network can learn sharper features there.
   In open free space φ compresses space, since there is little information there.
3. Train φ with the RNL. At a boundary the RNL should stretch space toward the obstacle and compress it on the side away
   from it.
4. Sample poses that are "pretty free", with a big free radius, and suck in the space around them.
5. Limit the compression / expansion through the Jacobian of φ, e.g. to 2x or 3x.
6. Train it as a standalone model.

## 3. Why stretching should sharpen the field

The warp is applied to both endpoints, so the field is T(x, y) = H(φ(x), φ(y)), with H the head. By the chain rule in Lie
coordinates:

```
c(x) = J_φ(x)ᵀ · c_H(φ(x))            S(x) = 1 / |J_φ(x)ᵀ c_H(φ(x))|
```

* J_φ(x) is the 6 × 6 differential of φ (section 5), and c_H is the head's gradient in warped coordinates.
* **Resolution.** A smooth encoder can only change its output over some length ℓ in its own input coordinates. Through φ,
  that becomes a length ℓ / σ in true coordinates, where σ is the local stretch (a singular value of J_φ).
  * A stretch σ = 3 next to a wall makes features 3x sharper there with the same encoder.
  * Compression elsewhere hands resolution away from regions that do not need it.
* **The ideal warp ("slowness-matched").** The target is |c(x)| = 1 / S*(x).
  * If φ stretched by 1 / S*(x) toward the obstacle, the head's field in warped coordinates would have unit speed
    everywhere. That is the easy, free-space-like field a smooth encoder represents well.
  * S* falls to a few percent at contact, so this needs stretches of tens. Every variant caps it.
* **Premise at the time.**
  * The detached RNL lowered the speed next to the walls without making its slope steeper, and we attributed that to the
    encoder's smoothness.
  * Later diagnostics (section 8) show the central-difference RNL steepens the slope with the same encoder and no warp. So
    smoothness is not the binding limit; the planner's tolerance of sharp fields is.

## 4. Architecture (`graph_method_triangle/warp_decoder.py`, arch `graph_homotopy_lift_warp`)

φ acts on the lifted pose (t, q) ∈ R³ × S³ through K residual exponential steps:

```
t_{k+1} = t_k + a_t · tanh(u_k)                      world-frame translation increment
q_{k+1} = q_k ⊗ Exp(a_r · tanh(w_k))                 body-frame rotation increment; Exp(w) = unit quaternion of the
                                                     rotation by |w| rad about w/|w| (series-safe at w = 0)
(u_k, w_k) = MLP_k( γ(t_k, β·vec R_k) )
γ(z) = [sin 2π zB, cos 2π zB, z],  z ∈ R¹²
```

**Feature matrix B.**
* Translation rows use one block of 32 random frequencies per scale σ_t:
  * {2, 6, 18} cycles per unit by default;
  * {1, 3, 6} in the "smooth" variants.
* Rotation rows use scale σ_r = 1 in the coarsest block, reduced in proportion in the finer blocks.
* B is shared by all steps.

**MLP and defaults.**
* Each MLP: 204 → 128 → 128 → 6 with SiLU.
* K = 3, a_t = 0.04 units per step, a_r = 0.25 rad per step, β = 0.15.
* The largest possible displacement is therefore 0.12 units and 0.75 rad.

Properties:

1. **Identity at start.** The last layer of every step is zero-initialised, so φ = id.
   * A trained hlift checkpoint loads unchanged: only the `warp.*` parameters are new.
   * A fine-tune therefore starts exactly at the trained field.
2. **Commutes with the deck transformation.** The features read R, which is even in q, so φ(t, -q) = (t', -q'). The head
   keeps all its invariants:
   * both sheets and deck invariance;
   * T(x, x) = 0;
   * the symmetry T(x, y) = T(y, x).
3. **Same φ on start and goal.** Both endpoints go through `lift_embeddings`, which the warp head overrides.
4. **Only the encoder input is warped.** The cone term and the sheet geometry use the true poses.
5. **Diffeomorphism only where enforced.**
   * Each step is the identity plus a bounded smooth increment, so φ is a local diffeomorphism wherever det J_φ > 0.
   * This is not guaranteed by construction: φ is not a flow and not a coupling layer.
   * It is enforced on sampled poses by the Jacobian bound of section 5: softly by a penalty, optionally hard by rejecting
     updates.

## 5. Jacobian and bounds (`hlift_rnl/warp_losses.py`)

```
J_φ(g)[:, i] = d/dε  η( φ(g·exp(ε E_i)) )  at ε = 0
η(h) = ( R'ᵀ (t_h - t'),  ρ · rotvec(q'* ⊗ q_h) )       (t', q') = φ(g),  ρ = 1/(2π)
```

* E_1..3 are unit body translations; E_4..6 are body rotations of 1/ρ rad. With this scaling the stretch is measured in
  the Eikonal term's metric.
* J is computed by central differences with h = 2e-3: 12 extra warp evaluations per pose. It is differentiable with respect
  to the warp parameters and matches autograd to 1e-6.
* What J tells you:
  * singular values σ_i: stretch factors;
  * log det J: log of the volume change;
  * the 3 × 3 translation block J_tt: volume change of positions only.

Bounds (the "2x or 3x" limit):

```
bound(g) = relu(log σ_max - log k)² + relu(-log k - log σ_min)²       keeps every σ in [1/k, k]
```

* **Soft bound.** Applied on 256 random start / goal poses of every batch, with k = 3 (2.5 in the hard-bound variants) and
  weight 1 or 30.
* **Hard bound** (`--hard-k K`).
  * After each try, J on 2048 fixed training poses must satisfy σ ∈ [1/K, K].
  * Otherwise the epoch is rolled back, exactly like a rejected Eikonal step.
  * After 10 rejections in a row the warp's learning rate halves.
  * A fold can therefore never be accepted.

## 6. What trains the warp

All terms are added to the head's loss (lifted Eikonal term + ramped RNL). Epochs are accepted or rejected on the Eikonal
loss alone.

1. **Eikonal + RNL.** The warp is in the forward pass, so both gradients reach φ. Stretching at a pose raises |c| there and
   lowers the speed, which is what the RNL asks for at the probe end.
2. **Free-space compression** (the "suck in" term, `--free-c 2 --free-weight 0.3`, ramped with the RNL).
   * **Which poses count as free:**
     * goal poses of training pairs at full clearance speed (S* = 1);
     * whose free radius is in the top quarter of such poses. The free radius is the distance from the body centre to the
       nearest scene point, minus the body's circumscribed radius.
     * An absolute radius threshold left no free pose for the long loop body in its room, hence the quantile.
   * Penalty: relu(log det J + log c)², which pushes the volume to at most 1/c (half for c = 2).
3. **Band expansion** (variant `wnfLb`, `--band-c 2 --band-weight 0.3`).
   * Applies to band starts, i.e. clearance below half the margin.
   * Penalty: relu(log 2 - log det J_tt)², which pushes the translation volume to at least twice its size.
4. **Slowness-matched stretch** (variants `wsTb`, `wsTh`).
   * Target: log det J_tt = 3α·log(1/S*) - log c, clamped to ±3·log(slow_max). This is a stretch of (1/S*)^α per axis, with
     free space halved.
   * Loss: (log det J_tt - target)², on random batch poses with their speed labels.

**Schedules.**
* **Fine-tune** (queue phase `warp`):
  * starts from the trained hlift checkpoint of seed 0 and runs 1500 epochs;
  * the RNL weight and cap ramp in over the first 750 epochs;
  * the warp has its own AdamW group (weight decay 0, lr 2e-3, or 1e-3 for the hard-bound variants); the rest of the head
    trains at lr 5e-4.
* **Standalone from scratch** (phase `warpscratch`, variant `wnfs`):
  * 8000 epochs, with the RNL ramping in from epoch 2000 to 4000;
  * the warp is frozen at the identity until epoch 2000 (`--warp-start 2000`), with bound weight 30 and a_r = 0.1.
  * Without the freeze the warp distorts at once: after 50 epochs the singular values span 0.26-2.7 on loop and
    0.002-17.7 on the Eastville rectangle. The untrained field's gradients are large and a soft bound does not hold against
    them.

## 7. Variants

All variants use the SE(3)-normal probe. ε is 0.02 on the shapes cells and 0.01 on the Gibson cells.

| name | RNL | warp features σ_t | warp terms |
|---|---|---|---|
| `wn` | detached anchor | 2 / 6 / 18 | bound k 3 |
| `wnf` | detached | 2 / 6 / 18 | bound k 3 + free compression ×½ |
| `wnfL` | detached | 1 / 3 / 6 | bound k 3 + free compression ×½, translation-block volumes |
| `wnfLb` | detached | 1 / 3 / 6 | `wnfL` + band expansion ×2 |
| `wsTb` | detached | 1 / 3 / 6 | soft bound k 3 (weight 30) + slowness target, α ½ |
| `wsTh` | detached | 1 / 3 / 6 | soft bound k 2.5 (weight 30) + hard bound 3 + slowness target, α ½, capped at ×2 per axis |
| `w0f` | off (Eikonal only) | 2 / 6 / 18 | bound k 3 + free compression ×½ |
| `wnfs` | detached, from scratch | 2 / 6 / 18 | bound k 3 (weight 30) + free compression, a_r 0.1, frozen until epoch 2000 |
| `wcen` | central difference | 1 / 3 / 6 | soft bound k 2.5 (weight 30) + hard bound 3 + free compression (**running**) |

* Reference rows:
  * `nrm`: the RNL alone, no warp.
  * `nrmcen`: the central RNL alone, the baseline of `wcen`.
  * `off`: the control, i.e. the same fine-tune without the RNL or a warp.
* `wnf2` (k 2) and `w0` (warp, Eikonal only) are defined in the queue but were not run.

## 8. Results

### 8.1 Planning

**Setup.**
* Seed 0, the first 500 cases of each cell's test set, dual planner, exact checks.
* Every cell is planned with plain MPPI; the two Gibson cells are also planned with the learned face guard (gEF).
* Cells:
  * loop env9 and crank env5: shape bodies in 3D scenes;
  * Allensville and Eastville: a rectangle body in Gibson house slices.

Table: exact success % (collision % / not converged %).

| field | loop env9 | crank env5 | Allensville | Allensville + gEF | Eastville | Eastville + gEF |
|---|---|---|---|---|---|---|
| hlift as trained | 82.2 (17.6 / 0.2) | 57.2 (42.8 / 0.0) | 66.0 (28.8 / 8.6) | 90.6 (0.0 / 9.4) | 82.8 (12.4 / 6.2) | 93.2 (0.0 / 6.8) |
| control `off` | 81.2 (18.8 / 0.0) | 59.2 (40.8 / 0.0) | 63.8 (26.4 / 11.0) | 87.2 (0.8 / 12.0) | 83.8 (10.0 / 7.0) | 93.0 (0.0 / 7.0) |
| **RNL alone `nrm`** | **90.4** (9.4 / 0.2) | **66.8** (33.2 / 0.0) | **78.0** (13.6 / 11.2) | **88.0** (0.0 / 12.0) | **86.2** (4.8 / 10.0) | **90.2** (0.0 / 9.8) |
| `wn` | 91.0 (8.6 / 0.4) | 60.8 (39.0 / 0.2) | 76.8 (10.2 / 14.8) | 83.8 (0.0 / 16.2) | 86.6 (5.8 / 8.8) | 90.2 (0.2 / 9.8) |
| `wnf` | 92.6 (6.8 / 0.6) | 61.8 (37.4 / 1.0) | 78.0 (10.6 / 13.8) | 84.8 (0.2 / 15.2) | 86.0 (6.4 / 9.0) | 90.6 (0.2 / 9.4) |
| `wnfL` | 90.0 (9.0 / 1.0) | 61.6 (38.4 / 0.0) | 72.0 (9.0 / 20.8) | 78.2 (0.2 / 21.8) | 86.2 (6.0 / 10.2) | 89.2 (0.0 / 10.8) |
| `wnfLb` | 91.8 (7.2 / 1.2) | 62.2 (37.8 / 0.0) | 73.4 (9.8 / 20.4) | 78.8 (0.2 / 21.2) | 85.0 (6.0 / 10.8) | 88.6 (0.0 / 11.4) |
| `wsTb` | 24.0 (76.0 / 0.0) | 64.0 (36.0 / 0.2) | 74.4 (10.4 / 18.0) | 81.0 (0.0 / 19.0) | 86.8 (5.6 / 9.6) | 89.6 (0.2 / 10.4) |
| `wsTh` | 89.8 (9.6 / 0.6) | 59.6 (40.4 / 0.2) | 72.6 (11.0 / 19.2) | 80.4 (0.0 / 19.6) | 86.2 (5.4 / 10.4) | 89.0 (0.0 / 11.0) |
| `w0f` | 83.8 (15.2 / 1.0) | 56.2 (43.8 / 0.0) | 68.6 (20.8 / 13.4) | 85.2 (0.6 / 14.2) | 80.6 (13.2 / 8.2) | 90.8 (0.4 / 8.8) |

Each warp against the RNL alone on the same cases. The figures are the cases only the warp solves / the cases only the RNL
alone solves, pooled over the six columns, with the exact McNemar p:

| warp | vs `nrm` |
|---|---|
| `wnf` | +118 / -147 (p 0.085) |
| `wn` | +92 / -144 (p 9e-4) |
| `wnfL` | +98 / -210 (p 2e-10) |
| `wnfLb` | +96 / -195 (p 7e-9) |
| `wsTh` | +81 / -191 (p 2e-11) |
| `wsTb` | +80 / -479 (p 3e-70) |
| `w0f` | +76 / -248 (p 2e-22) |

**Standalone from scratch** (loop env9, 1000 cases):

| field | success (collision / not converged) |
|---|---|
| warp `wnfs` | 88.8 (10.6 / 0.6) |
| RNL alone from scratch | 90.4 |
| hlift reference, same seed and recipe | 80.3 (19.6 / 0.1) |

### 8.2 What the warp does to the field (`scripts/diag_rnl.py`, 20000 fixed band samples per cell)

**Columns.**
* **slope:** the mean speed slope dS/dε along the SE(3)-normal probe on the +q sheet. More negative means the speed falls
  faster toward the obstacle.
* **target:** the analytic target slope.
* **above:** the share of samples whose slope is shallower than the target.
* **N0:** the median band residual S*·|c|; above 1 means slower than the target.

The Gibson cell's margin is smaller in normalised units, hence its larger slopes.

| field | loop env9: slope / above / N0 (target -17.8) | crank env5 (target -18.6) | Allensville (target -54.7) |
|---|---|---|---|
| control `off` | -12.4 / 69 % / 0.93 | -14.1 / 70 % / 0.99 | -41.3 / 63 % / 0.98 |
| RNL alone `nrm` | -11.7 / 77 % / 1.11 | -13.3 / 77 % / 1.14 | -43.7 / 63 % / 1.04 |
| central RNL `nrmcen` | -14.9 / 61 % / 0.90 | -16.5 / 59 % / 0.92 | pending (rerun when the warp phase ends) |
| `wnf` | -11.7 / 77 % / 1.13 | -13.9 / 75 % / 1.09 | -47.0 / 60 % / 1.02 |
| `wnfLb` | -11.6 / 78 % / 1.14 | -14.0 / 74 % / 1.08 | **-48.1** / 59 % / 1.02 |
| `wsTh` | -11.7 / 77 % / 1.12 | -14.0 / 74 % / 1.08 | -46.3 / 62 % / 1.05 |
| `w0f` | -12.3 / 71 % / 0.97 | -14.3 / 68 % / 0.97 | -41.0 / 64 % / 1.00 |

* **Attraction on Allensville.** Attraction is the share of normal probes that end inside the obstacle yet still lower T.
  It drops from 2.4 % (control) to 0.67 % with the RNL alone, and to 0.44-0.64 % with the warps trained with the RNL
  (1.9 % for `w0f`, the warp without the RNL).
* **Central RNL planning** (pilot4, same 500 cases): loop env9 74.4 % (14.6 / 13.2), crank env5 58.8 % (36.6 / 4.6).
* **Shape of the learned warp** (`scripts/warp_slice.py`). Measured as the translation area stretch on a horizontal slice at
  a fixed rotation:
  * Locally it stretches and compresses by 2-4x (log2 area between about -2 and +1.3).
  * The median log2 stretch is ≈ 0 at every distance band from the scene: below 0.03, 0.03 to 0.08, and above 0.08 units.
    The bands are -0.01 / -0.01 / -0.04 for loop `wnf` and -0.10 / -0.05 / -0.20 for Allensville `wnfLb`.
  * With σ_t 2 / 6 / 18 the stretch is speckle.
  * With σ_t 1 / 3 / 6 in the house it has structure (compressed room interiors, stretched corridors) but no rim that
    follows the walls.
  * The free-space term works: on loop `wnf` the free poses have log det ≈ -0.9, i.e. a volume of ×0.41.
* **Stability.**
  * Fine-tuned under the soft bound k 3, the warp stays moderate: loop `wnf` ended with singular values within
    [0.38, 2.09].
  * The slowness-matched target folds the warp under a soft bound: `wsT` reached singular values 0.0015-97 by epoch 300;
    `wsTb` on loop reached ~0.001-100 by epoch 1000, leaving 76 % collisions.
  * Under the hard bound (`wsTh`) the warp is stable, with 0 rejections.

## 9. Interpretation

1. **The warp does what the losses ask:**
   * the free-space compression is learned;
   * the hard bound keeps it a local diffeomorphism on the checked poses;
   * deck equivariance is exact.
2. **On loop and crank there is nothing for it to steepen.**
   * The detached RNL lowers the speed of the whole wall band (N0 0.93 → 1.1). Its slope gets slightly shallower, probably
     because the probe ends that it pushes down are themselves the anchors of other band samples.
   * With and without a warp the slopes are equal (loop) or near the control (crank).
3. **In the house it works as intended, but planning gets worse.**
   * On the Allensville rectangle (long flat walls) warp + RNL makes the field about 10 % steeper at the walls than the RNL
     alone (-47 / -48 vs -43.7; target -54.7) and lowers attraction.
   * But stalls nearly double (not converged 11.2 → 20.4 % for `wnfLb`). The fewer collisions (13.6 → 9.8 %) do not pay
     for them.
4. **A sharper field is harder for the forward MPPI.** It meets more local minima.
   * The 2D Gibson study saw the same: hash grids, multi-scale Fourier encoders and distillation to grid solutions all
     raised non-convergence.
   * The guard (gEF) does not rescue the warp: with the guard the warps lose as well (Allensville + gEF 78-85 % vs 88 %).
5. **A warp is not needed to steepen the slope.**
   * The central-difference RNL steepens it without a warp (loop -12.4 → -14.9, crank -14.1 → -16.5), and it stalls as well
     (loop 74.4 %, 13 % not converged).
   * The binding constraint is the planner's tolerance of sharp fields, not the encoder's resolution.
6. **The ideal warp is unstable.** The slowness-matched warp folds under a soft bound and, under a hard bound, loses like the
   others.

## 10. Open questions and ideas not yet tried

* **`wcen` vs `nrmcen` (running).** Can a warp give the central RNL's steeper slope with fewer stalls, i.e. sharpness at a
  lower planning cost? This is the last direct test of the idea in this study.
* **Diffeomorphic by construction.**
  * Integrate a learned time-dependent vector field on SE(3) (a flow with an exact inverse), or use coupling-type steps.
  * The bound would then only limit the stretch, not prevent folds.
* **Anisotropic, geometry-tied stretch.**
  * Stretch only along the clearance normal (`normal.npy`), by (1/S*)^α.
  * This replaces learning an isotropic volume change from the losses.
* **Separate fields for the planner.**
  * Let the planner descend a smooth field and use the sharp, warped field only where collisions are decided.
  * Or make the planner robust to local minima.

## 11. How to run

All commands run in the `pytorchserver:latest` container from the repository root (see the root `CLAUDE.md`); the queues run
with the host `python3`.

```
# unit tests (6): identity at init + checkpoint load, deck equivariance and head invariants, finite-difference Jacobian =
# autograd, finite gradients at the identity, penalties, Lie gradient through the warp
python -m unittest tests.test_hlift_warp -v

# one fine-tune (what the queue runs for loop3d_env9 / wnf)
python train/regular_planning/train_hlift_rnl.py --dataPath datasets/3dshape/data/loop3d_env9 --name loop3d_env9_wnf_s0 \
    --seed 0 --epochs 1500 --out_root ./Experiments/hlift_rnl/warp \
    --init-from Experiments/graph_homotopy/loop3d_env9_hlift_s0/latest.pt --rnl-ramp-len 750 \
    --rnl-probes normal --rnl-eps 0.02 --warp --jac-k 3 --free-c 2 --free-weight 0.3 --warp-lr 2e-3

# the study's queue (trainings, dual-planner plans, exact checks); variant flags in VARIANTS of run_rnl_queue.py
python3 models/regular_planning/hlift_rnl/scripts/run_rnl_queue.py --phase warp \
    --cells loop3d_env9,crank3d_env5,rect_small_eastville_dc,rect_small_allensville_dc \
    --variants wnf,wnfLb,wsTb,wnfL,wsTh,wn,w0f,nrmcen,wcen --variant-major --seeds 0 --epochs 1500 --ramp 750 --cases 500 --guards
python3 models/regular_planning/hlift_rnl/scripts/run_rnl_queue.py --phase warpscratch --scratch \
    --cells loop3d_env9,rect_small_eastville_dc --variants wnfs --seeds 0 --epochs 8000 --ramp 2000 --ramp-start 2000 --guards

# tables, paired comparison, field diagnostics, picture of a warp
python3 models/regular_planning/hlift_rnl/scripts/table_rnl.py --phase warp
python3 models/regular_planning/hlift_rnl/scripts/paired_rnl.py --phase warp --a wnf --b nrm
python models/regular_planning/hlift_rnl/scripts/diag_rnl.py --cell loop3d_env9 --eps 0.02 \
    --ckpt nrm=Experiments/hlift_rnl/pilot/loop3d_env9_nrm_s0/latest.pt \
    --ckpt wnf=Experiments/hlift_rnl/warp/loop3d_env9_wnf_s0/latest.pt --out results/hlift_rnl/diag/warp_loop3d_env9.json
python models/regular_planning/hlift_rnl/scripts/warp_slice.py --cell loop3d_env9 --case 3 \
    --ckpt Experiments/hlift_rnl/warp/loop3d_env9_wnf_s0/latest.pt --out results/hlift_rnl/warpfig/warp_loop3d_env9_wnf.png
```

**Notes.**
* On Gibson cells the queue also passes the cell's own interior-pricing flags, as their hlift fields were trained with them.
* Warp flags of `train_hlift_rnl.py`:
  * `--warp`, `--warp-K`, `--warp-hidden`, `--warp-n-freq`, `--warp-sigmas-t`, `--warp-sigma-r`, `--warp-beta`,
    `--warp-amp-t`, `--warp-amp-r` (architecture);
  * `--jac-k`, `--jac-weight`, `--jac-rows`, `--jac-block`, `--hard-k` (bounds);
  * `--free-c`, `--free-weight`, `--free-radius` (compression);
  * `--band-c`, `--band-weight`, `--band-s` (band expansion);
  * `--slow-weight`, `--slow-c`, `--slow-alpha`, `--slow-max` (slowness target);
  * `--warp-wd`, `--warp-lr`, `--warp-start` (optimiser).
* Outputs:
  * checkpoints: `Experiments/hlift_rnl/<phase>/<cell>_<variant>_s<seed>/latest.pt`;
  * plans: `results/hlift_rnl/<phase>/<run>/<set>__dual[__gEF].json`;
  * tables: `results/hlift_rnl/<phase>/summary.md`;
  * diagnostics: `results/hlift_rnl/diag/warp_<cell>.json`.

## 12. Files

* `models/regular_planning/graph_method_triangle/warp_decoder.py`: `SE3Warp`, `HomotopyLiftWarpNN`, the arch registration
  (imported lazily by `model_network_graph.build_network`).
* `models/regular_planning/hlift_rnl/warp_losses.py`: the Jacobian in Lie coordinates, the bound and volume penalties, the
  pool of free poses.
* `models/regular_planning/hlift_rnl/model_function_rnl.py`: `WARP_DEFAULTS` and the warp terms in `Function.Loss`.
* `models/regular_planning/hlift_rnl/model_train_rnl.py`:
  * partial load of hlift checkpoints;
  * the warp's own parameter group, learning rate and freeze;
  * the hard bound in the accept / reject loop.
* `train/regular_planning/train_hlift_rnl.py`: entry point and flags.
* `models/regular_planning/hlift_rnl/scripts/`:
  * `run_rnl_queue.py`: variants, phases `warp` / `warpscratch`;
  * `table_rnl.py`, `paired_rnl.py`;
  * `diag_rnl.py`, `warp_slice.py`.
* `tests/test_hlift_warp.py`: 6 tests.
* `models/regular_planning/hlift_rnl/EXPERIMENTS.md`: log entries W0-W10.
