# jeffrey/segment_dp: loss ablations, distance specialists, the segment-DP transition, position + offset; DP-chain losses, TD and ball-Bellman terms, larger cabinets; warm start, clutter and platforms, cabinet only; segment DP on a correct local field

*2026-10-27, 13:05–20:32 on the RTX 3090 and RTX 5070 (both shared with another session's guard
evaluations). Every number is from a registered run (index at the end). MPPI = release MPPI successes /1800 (600
queries × planning seeds 0/1/2), RK4 = native RK4 /600, both scored by the release scorer (no-wall tilt 20°, cabinet
tilt gate off). Training seed 87358 unless a cell lists several (mean, then each seed).*

## Summary

**1. Loss ablations of the original method** (bank, 5 rounds, fifth-round evaluation; 3 training seeds for the
two requested ablations).
- **Dropping the Bellman term changes nothing**: cabinet MPPI 690 vs 694, no-wall 1588 vs 1590 (seed means; the
  seeds overlap). With this bank the backup takes the 0.01 step in 97–99.7 % of rows, so the Bellman target is a
  discretized eikonal condition: a field trained without it is still Bellman-consistent (final Bellman residual
  0.00036 vs 0.00027 when it is trained).
- **Dropping the Eikonal term costs on every seed**: cabinet 670 vs 694 (−24), no-wall 1458 vs 1590 (−131). Without
  it the eikonal residual of the field is 3.6× larger (0.233 vs 0.065): the 8 random action directions of the
  backup do not pin the gradient norm.
- **The third term decides most**: the winner-direction term (the direction to the best *safe* bank action). Without
  it no-wall collapses to 613 and eikonal-only to 529, almost all tilt violations; on the cabinet the eikonal needs
  either Bellman term next to it (Bellman+Eikonal 701, Eikonal+direction 690, Eikonal alone 569).

**2a. ~20 eikonal specialists, no transition bank** (20 segments of 0.08 over |q−g| ∈ [0, 1.6], one model each,
trained only with the eikonal loss on pool pose pairs of their segment).
- **Do they fit their own distances better? Only the shortest ones.** Against path-optimized ground truth the two
  shortest specialists are 2–3× more accurate than the original method (cabinet relative error 0.08–0.11 vs
  0.23–0.36) and unbiased. From gap 0.16 on the specialists are worse: they overestimate by 24–56 % (cabinet) and
  the model trained on gaps 0.06–0.18 predicts segments 3–5 better than their own specialists (0.09/0.15/0.23 vs
  0.23/0.41/0.48). The eikonal residual only fixes |∇T|; the *level* of T comes from T(g, g) = 0, which only the
  shortest segments see, so a mid-range specialist satisfies the slope on its shell at the wrong level.
- **For planning, the anchored short-range models are what counts.** On no-wall 20 specialists plan far better
  than one eikonal model on the same pairs (1522 vs 368: the generalist's short-range directions are poor, tilt
  failures), and the **two shortest specialists alone** — model 2 answering every longer gap, no bank, no Bellman —
  beat the original method on both scenes: no-wall 1689 / 1686 / 1679 over three seeds (mean 1685) vs 1590, cabinet 716 / 746 / 724 (mean 729) vs 694. A single eikonal
  model trained only on gaps ≤ 0.18 does as well (cabinet 737, no-wall 1675): what pays is fitting short distances
  well, not splitting them over models.
- With 20 specialists, neighbouring models disagree by up to 20–30 % at the segment edges; where the inner model is
  higher the blend is a barrier for gradient descent, and on long cabinet queries the 20-specialist field converges
  0/400 times with RK4.

**2b. The segment-DP transition** T_(m+1)(i, j) = min_k T_≤m(i, k) + T_≤m(k, j) with |i−k|, |k−j| inside the
distance the frozen models 1..m were trained on (segments 1–2 are the eikonal specialists; 3–20 learned from it).
- **It fixes long distances.** The DP chain is the only field whose long-range travel times are right: cabinet
  relative error 0.14 at gaps 0.72–1.6 (falling from 0.25 at 0.4 to 0.10 at 1.6) vs 0.25–0.42 for every other
  field, Spearman 0.75 there (DP-c 0.65, the offset DP chain 0.88 at a 37 % upward bias, every non-DP field ≤ 0.21,
  the straight-line travel time 0.51); no-wall
  0.36 vs 0.47–0.95, Spearman 0.67–0.72 vs ≤ 0.23. Neighbouring DP segments agree within ~1 %, so the field is consistent across
  edges; on 400 long cabinet queries it converges with RK4 138 times (original method 89) and succeeds most often (29 vs 23); with MPPI every field that converges at all ends at 8–9 of the first 100 long queries (all failures collisions), so the better long-range field shows in gradient descent, not yet in MPPI success.
- **The split point decides the bias.** With split points restricted to pool poses the sparse lens adds a detour
  that compounds over the first levels (targets 22–32 % above truth at gaps 0.16–0.48; 3–7 % above beyond 0.64).
  With continuous, gradient-refined split points the first levels are exact (within 3 %), but the min then finds
  points where the frozen model reads low — regions no pose covers — and the chain ends 30 % low at long range.
- **On the frozen (short) benchmarks it is roughly level**: cabinet 706 (pool) / 716 (continuous) vs 694; no-wall
  1442 vs 1590 (its mid-range overestimate) — the test queries (≤ 0.68) barely use the long segments.

**2c. Position + offset** (T(q, d) with d = g − q an input of its own, same encoder and parameter count).
- It does not help: original method cabinet 652 vs 694, no-wall 1496 vs 1590; specialists cabinet 688 = 688,
  no-wall 1399 vs 1522; the generalist cabinet 633 vs 713, no-wall 608 vs 368 (the only gain, from a very low
  level); DP chain cabinet 694 vs 706 (its targets 25–37 % too high, though it ranks long pairs best of all fields). The offset network fits the eikonal residual 2–3× worse (0.09–0.19 vs
  0.04–0.07 at the end of training) and overestimates short gaps (bias 1.4–1.5 on no-wall).

**3. Loss, decoder, split count, TD and ball-Bellman terms on the DP chain** (cabinet, chains to segment 9 on the
eikonal specialists 1–2, one training seed unless stated; section 3).
- **Loss**: L1 plans best on one seed (736 vs SmoothL1 716, relative 729, log-L2 725, log-L1 718, L2 710), but the
  training-seed spread of a DP chain is ±11 (DP3 716 / 709 / 694; L1 736 / 718 / 714): L1 is +17 on seed means with its
  worst seed level with DP3's best, a tendency rather than a separation; every loss fits the
  same targets equally (truth error 0.22–0.23), so the differences are what the loss does with the targets' upward
  skew, not accuracy. **Decoder**: the original T = ‖q−g‖(1 + softplus r) already never reads below the lower bound;
  an additive decoder changes nothing (711), nor does an unbounded one (704) — the bound never binds.
- **Split candidates do not matter beyond 32**: 32 / 128 / 384 / 1,536 candidates give identical targets (0.1 %)
  and chains within seed noise (721 / 716 / 706); 8 cost 0.1–1.7 %. The overestimate is the pool's, not the search's.
- **TD from the short-range networks** reads the local speed within 7 % and is unbiased at 0.16–0.4, but a chain
  trained on it alone drifts low with range (0.82 at segment 9) — a small-step condition pins the slope, not the
  level. **Ball Bellman** alone is the best-calibrated field of the study (bias 1.05 / 1.01, error 0.12 / 0.18) but
  plans no better than DP (714). **Combined**, DP + TD + bell plans best of all (742 / 736 / 725 over three training
  seeds vs DP3's 716 / 709 / 694, +28 on seed means) and DP + bell 728: the terms lower the DP chain's bias (1.16 / 1.10 vs 1.21 / 1.16) at unchanged direction.
- The planner rewards the DP chains with the lowest bias, not the fields with the best directions (cosine 0.82–0.85
  at 0.16–0.4 for all sixteen chains).

**4. Larger cabinets** (compartments 10 % and 20 % roomier, their own pool, speed and original-method checkpoint).
- **The convergence gain holds and grows**: RK4 descent on the DP chain reaches the goal on 149 / 150 of 400 long
  queries vs 88 / 82 for the original method (cabinet: 138 vs 89) and 0 for the specialists; collision-free successes
  21 / 22 vs 13 / 22 (the chain converges more but collides more — it fixes the level, not the near-obstacle direction).
  Frozen MPPI: chain 732 / 725, original method 682 / 646, specialists 721 / 726; RK4: the chain is the best field on
  every scene (254 / 256).

**5. Warm-starting each DP level from the previous level** (cabinet, DP3 recipe, three training seeds).
- Same field, reached sooner: the inherited weights already read the next level's targets within 2–7 % from segment 4
  on, and after training the fit is unchanged. Planning: MPPI 713 / 725 / 710 (mean 716) vs 716 / 709 / 694 (706),
  inside a chain's seed spread; RK4 +4 on every seed (245 / 245 / 243 vs 241 / 240 / 240). A free speed-up, not a
  different field.

**6. Clutter and platforms** (two easier arm scenes, unconstrained planning, own frozen suites, pool, speed and
original-method checkpoint; DP and DP + TD + bell as full 20-segment chains).
- The frozen suites are short (no query beyond 0.55) and the specialists win them: clutter 1294 vs DP 1251 / 1248 and
  DP + TD + bell 1281 / 1285; platforms 1576 vs 1493 / 1503 and 1546 (original method 1165 / 1532). The deficit is
  entirely on detour queries, all collisions.
- On 400 long queries the chains win as on the cabinet: RK4 converges on 151 / 143 (clutter) and 142 / 139
  (platforms) vs 104 / 90 for the original method and 0 / 3 for the specialists; DP + TD + bell succeeds most (75, 89).
- Mechanism (splice and probe controls): the segment-3 model decides the detours; near obstacles, where MPPI's
  rollouts sit at the speed floor, the specialists keep the eikonal slope (S·|∇T| 0.87–0.94) and the DP models have
  half of it (0.43–0.51; DP + TD + bell 0.56–0.59). The specialists' fields fall at their 0.48 seam on 93–96 % of rays,
  which is why they never converge from far away and why splicing them into the chain is not a fix.

**7. Cabinet only** (pool restricted to configurations with the grasp point inside a compartment: 221,975; test =
the frozen compartment→compartment and same-compartment strata, 600 MPPI runs).
- In-cabinet MPPI: DP + TD + bell cabinet-only 40 / 41 / 36 on three training seeds, vs 27 for the original method
  and 19–31 for the full-pool fields; DP cabinet-only 37 / 25 (two seeds, the second below the original method). The
  gain is on same-compartment queries (36 / 38 / 33 of 300 vs 24 for the original method, 5–12 for the full-pool
  chains; worst seed +9, seed range 5) and is an MPPI gain (RK4 7–9 of 200 vs 8). Compartment→compartment gets worse
  (3–4 vs 12–19) and everything outside collapses (overall 483–527 of 1800 vs 688–742; long queries ≤ 1 of 400).
- Route control: successful same-compartment paths stay inside the compartment; the full-pool chains (2.7 % of their pool
  inside) lead the arm out and collide. Compartment→compartment paths must cross the space in front of the
  compartments, which a cabinet-only pool does not contain.

**8. Segment DP on a correct local field** (path-optimized local travel times; specialists 1–2 regressed on them, SL,
truth error 0.073 vs 0.089; the full 20-segment DP chain on top, three training seeds; section 8).
- The DP's long-range *values* do not become correct: even exact pieces leave level 3 at 1.146 × the truth (the
  pool's split points; the base adds 3 points), and on the accurate base that overhead compounds to 1.30–1.34 at
  gap 1.5. The eikonal chain's 1.04–1.08 there is a downward drift inherited from its base that cancels the same
  overhead (decomposition: no selection by the min, identical via-split overhead, 5–22 % lower teacher on the same pieces).
- Its *shape* becomes right: Spearman 0.89–0.90 vs 0.73–0.75 at 0.72–1.6, 35–36 % monotone rays vs 17–19 %, and RK4
  reaches the goal on 322 / 299 / 330 of the 400 long queries vs 138 / 135 (original method 89). Splices: carried by the
  DP levels trained on the correct base (305 with the eikonal specialists swapped in), not by the local field at
  planning time (141). Collision-free long-query success stays at 28–31 / 400 (MPPI 7–10 / 100): collisions.
- SL12 alone is the best field on the frozen cabinet suites: RK4 273, MPPI 760 (vs ES12 716, original method 688–698);
  the DP levels on top cost 45–55 MPPI there (706–716).

## What was asked

1. Ablate the Eikonal loss of the original method, then the Bellman loss.
2. Test segment DP without the offline transition bank: (a) about 20 models, one per distance range, trained with
   the regular eikonal method — do they fit their distances better? (b) the DP transition: model m+1's prediction
   from i to j equals the min over k of model m's predictions i → k and k → j, with both inside model m's trained
   distance. (c) A position + offset network instead of two positions, so the distance is an input and the
   segments are ranges of the offset.
3. (Sections 3–4) DP-chain losses, decoders, split counts, TD and ball-Bellman terms; larger cabinets.
4. (Section 5) Warm-start each DP level from the previous level's weights.
5. (Section 6) Run the segment DP and the best loss combination (DP + TD + bell) on other, easier arm scenes —
   clutter, platforms — still unconstrained planning.
6. (Section 7) "Cabinet only": train the methods on configurations where the end effector is inside the cabinet;
   test on the queries inside it.
7. (Section 8) Test whether the segment DP is good with a correct local time field in place of the two eikonal
   specialists at the smallest ranges, the DP on the later models as before; compare the long-range tasks with the
   chain on the learned eikonal specialists.

## Protocol

**Unchanged:** network blocks and size (338,177 parameters; the offset network 338,241), AdamW 2e-4, weight decay
1e-6, gradient-norm clip 2, batch 256, 15,620 updates per model; native RK4 and the release MPPI with its original
settings on the frozen 600 no-wall queries and the frozen certified cabinet suites.

**Part 1** is `experiments/from_source/training.py` with loss weights B/E/D (`ablation/train.py`; 6/2/2 reproduces
the baseline checkpoint bit for bit). A zero weight removes a term from the objective; the backup and the gradient
are still computed because the direction term is defined through them.

**Part 2 uses no transitions.** Poses are the pool of collision-checked configurations the bank was built on
(`bank/source_q.npy`, positions only): on the no-wall task 99 % of the joint box sits at the 0.1 speed floor, so poses
are not drawn from the box. Speeds come from the frozen speed network.
- Segments: 20 of 0.08 over |q−g| ∈ [0, 1.6] (frozen queries 0.02–0.68; random pool pairs reach ~1.7). Model m
  trains on its segment widened by 0.016 per side; the field routes a pair by |q−g| and ramps linearly between
  neighbours over ±0.016 around each edge (continuous value and gradient).
- Pairs (`sampler.py`): q uniform over the poses that reach the segment, a target distance uniform in it, g the pose
  closest to that distance (64 nearest neighbours + a 16k random subset; q itself and duplicates excluded).
- (a) **ES**: one eikonal model per segment (the Eikonal term of `method/loss.py` alone), trained as one batched
  ensemble with separate batches, losses, clipping and optimizer state (equal to single models to 2e-6). **EG**: one
  model, each row drawing its segment uniformly (the union of the specialists' pairs). **Specialists 1–2 only**: the
  first two ES models, the second answering every longer gap (`derived.py`; also trained directly, seeds 1 and 2).
- (b) **DP** (`dp.py`): segments 1–2 are the ES models (the pool has no split points for segment 2: with both halves
  ≤ 0.096 and |i−j| up to 0.18, 70–100 % of lenses are empty). For segment m+1 = 3..20: 400k pairs, the target
  min_k T_≤m(i,k) + T_≤m(k,j) over pool poses k with |i−k|, |k−j| ≤ e_m + 0.016 (384 candidates: half nearest the
  straight line, half uniform over the lens), a fresh model regressed on it (SmoothL1, the Bellman term's loss).
  **DP-c** (`dp2.py`): the same with continuous split points (96 pool + 32 points along the segment with noise, the
  best 4 refined by 6 normalized gradient steps, all projected onto the lens; 150k pairs per segment).
- (c) **-off**: `fields.py` OffsetCost — the shared encoder sees q and d = g − q; the pair head sees
  [E(q), E(d), E(q)·E(d), cos 2πhd, |d|] instead of the symmetric two-position features — in ES, EG, DP and the
  original method (`bellman.py --arch offset`).

**Ground truth** (`truth.py`, diagnostic only): for ~20k probe pairs of the pool (up to 8 per segment for each of
128 goals), T* = the cheapest of three concrete paths under the frozen speed (∫ ds/S): the 24-NN graph Dijkstra
route, that route optimized as 20 waypoints (Adam, re-resampled by arc length), and the straight line (optimized as
well where competitive). It is an upper bound: a 4× longer search from three other starts finds a ≥ 2 % cheaper
path for 4–21 % of cabinet pairs (median change 0). `fit.py` reports per segment the mean |log T/T*|, median T/T*,
the Spearman rank of T/|q−g| against T*/|q−g| (does the field know which pairs are slow) and the cosine between
−∇_q T and the first step of the best path.

**Evaluation wrapper** (`evaluate.py`): the unchanged evaluators with only the field loader swapped; the planner's
optional CUDA-Graph replay of the cost (bitwise-checked against eager) is off for multi-segment fields because the
distance routing cannot be captured. Control: the wrapper re-evaluates the B5 cabinet checkpoint query for query
identically (RK4 229, MPPI 688 [230, 229, 229]).

## 1. Loss ablations

|Field|Cabinet RK4 /600|Cabinet MPPI /1800|No-wall RK4 /600|No-wall MPPI /1800|
|---|---:|---:|---:|---:|
|6/2/2 (baseline)|231 (229/229/236)|694 (688/697/698)|540 (550/537/532)|1590 (1598/1586/1585)|
|6/0/2 (no Eikonal)|225 (228/226/220)|670 (675/671/665)|508 (508/511/505)|1458 (1469/1450/1456)|
|0/2/2 (no Bellman)|229 (232/229/225)|690 (687/686/696)|532 (537/526/533)|1588 (1583/1600/1582)|
|0/2/0 (Eikonal only)|184|569|162|529|
|6/0/0 (Bellman only)|228|670|415|1315|
|6/2/0 (no direction)|228|701|187|613|

- Seeds: the two requested ablations and the baseline have three training seeds on both scenes; the Bellman-only,
  no-direction and eikonal-only controls one.
- **Bellman term.** Removing it leaves the seeds overlapping the baseline's (cabinet 686–696 vs 688–698, no-wall
  1582–1600 vs 1585–1598; means −5 and −1). The Bellman residual of a field trained
  without the Bellman term stays at the trained level (cabinet 0.00036 vs 0.00027): with 97–99.7 % of backups on the
  0.01 step the target is T(q) = c(q, q+0.01u) + T(q+0.01u, g), the eikonal condition along the best of 8 random
  directions, which the Eikonal + direction terms already enforce.
- **Eikonal term.** Removing it costs every seed (cabinet −13 to −33, no-wall −129 to −136 MPPI, same seed paired). The Bellman min
  over 8 random directions constrains −∇T only along the best of them; the gradient norm is left free (eikonal
  residual 0.233 vs 0.065 cabinet).
- **Direction term.** Not asked, but it explains the rest: on no-wall, dropping it (6/2/0) or keeping only the
  eikonal (0/2/0) collapses success to 613/529, failures are tilt violations (53–90 per suite), and the field's
  short-range direction cosine falls from 0.93 to 0.83/0.77. Adding the eikonal to the Bellman term without the
  direction term is worse than the Bellman term alone (613 vs 1315): natural (long) pairs dominate the eikonal term and
  bend short-range gradients off the orientation manifold, and only the direction term — toward the best *safe* bank
  action — puts them back. On the cabinet the eikonal needs either Bellman-derived term (701 / 690 vs 569 alone).

## 2a. Eikonal specialists per distance segment

|Field|Cabinet RK4 /600|Cabinet MPPI /1800|No-wall RK4 /600|No-wall MPPI /1800|
|---|---:|---:|---:|---:|
|Original method (B5, bank)|231 (229/229/236)|694 (688/697/698)|540 (550/537/532)|1590 (1598/1586/1585)|
|EG: 1 eikonal model, all segments|238|713|117|368|
|ES: 20 eikonal specialists|226|688|463|1522|
|Specialists 1–2 only (derived from ES)|239|716|555|1689|
|2 specialists trained alone, seeds 1/2|245 (251/239)|735 (746/724)|546 (543/548)|1682 (1686/1679)|
|Specialist 1 only (derived from ES)|244|721|—|—|
|1 eikonal model on gaps ≤ 0.176 only (EG16)|255|737|553|1675|
|DP: segments 3–20 by the transition, pool split points|241|706|467|1442|
|DP-c: continuous split points|232|716|—|—|

Fit to ground truth. Labels: B5 the original method; wB-E-D the loss ablations; EG / ES the eikonal generalist /
20 specialists; ES12 specialists 1–2 only; ES1 specialist 1 only; ES2s1, ES2s2 two specialists trained alone; EG16
one model on gaps ≤ 0.176; DP / DPc the DP chain with pool / continuous split points; -off the position + offset
network; straight the straight-line travel time (a field-free reference). Cabinet (registered fit stages):

**Relative error mean |log T/T*|**, mean over the segments of each gap range

|Field|0–0.16|0.16–0.4|0.4–0.72|0.72–1.6|
|---|---:|---:|---:|---:|
|straight|0.184|0.424|0.748|1.038|
|B5|0.294|0.215|0.360|0.378|
|w6-0-2|0.337|0.264|0.402|0.417|
|w0-2-2|0.300|0.214|0.350|0.360|
|w0-2-0|0.250|0.226|0.321|0.321|
|w6-0-0|0.267|0.202|0.360|0.414|
|w6-2-0|0.240|0.184|0.351|0.392|
|EG|0.112|0.156|0.332|0.362|
|EG16|0.086|0.158|0.333|0.339|
|ES|0.100|0.378|0.373|0.293|
|ES12|0.089|0.155|0.321|0.320|
|ES1|0.083|0.161|0.317|0.308|
|ES2s1|0.093|0.155|0.322|0.324|
|ES2s2|0.097|0.157|0.315|0.305|
|DP|0.095|0.236|0.227|0.140|
|DPc|0.088|0.139|0.291|0.330|
|EG-off|0.202|0.227|0.318|0.266|
|ES-off|0.118|0.215|0.352|0.247|
|DP-off|0.120|0.232|0.275|0.308|
|B5-off|0.304|0.216|0.348|0.337|

**Bias median T/T***, mean over the segments of each gap range

|Field|0–0.16|0.16–0.4|0.4–0.72|0.72–1.6|
|---|---:|---:|---:|---:|
|straight|1.054|1.305|2.159|2.909|
|B5|0.819|0.909|0.806|0.671|
|w6-0-2|0.840|0.881|0.762|0.642|
|w0-2-2|0.825|0.916|0.821|0.685|
|w0-2-0|1.115|1.125|0.908|0.723|
|w6-0-0|0.928|0.958|0.806|0.645|
|w6-2-0|0.855|0.942|0.820|0.660|
|EG|1.018|1.011|0.850|0.685|
|EG16|1.022|1.001|0.882|0.727|
|ES|1.038|1.426|0.965|0.752|
|ES12|1.026|1.013|0.900|0.737|
|ES1|1.021|1.030|0.933|0.785|
|ES2s1|1.022|1.010|0.893|0.738|
|ES2s2|1.038|1.030|0.910|0.755|
|DP|1.035|1.215|1.173|1.077|
|DPc|1.024|0.983|0.848|0.712|
|EG-off|1.149|1.140|0.968|0.796|
|ES-off|1.047|1.231|1.240|0.850|
|DP-off|1.053|1.246|1.308|1.365|
|B5-off|0.836|0.948|0.845|0.708|

**Spearman (which pairs are slow)**, mean over the segments of each gap range

|Field|0–0.16|0.16–0.4|0.4–0.72|0.72–1.6|
|---|---:|---:|---:|---:|
|straight|0.946|0.835|0.749|0.507|
|B5|0.695|0.654|0.263|0.130|
|w6-0-2|0.843|0.730|0.346|0.174|
|w0-2-2|0.692|0.632|0.254|0.178|
|w0-2-0|0.678|0.573|0.233|0.189|
|w6-0-0|0.821|0.747|0.378|0.205|
|w6-2-0|0.771|0.724|0.300|0.172|
|EG|0.889|0.772|0.302|0.169|
|EG16|0.941|0.792|0.313|0.151|
|ES|0.929|0.577|0.149|0.120|
|ES12|0.942|0.796|0.332|0.155|
|ES1|0.949|0.803|0.362|0.176|
|ES2s1|0.937|0.805|0.344|0.171|
|ES2s2|0.938|0.816|0.356|0.178|
|DP|0.936|0.874|0.841|0.753|
|DPc|0.942|0.889|0.782|0.649|
|EG-off|0.791|0.652|0.254|0.175|
|ES-off|0.892|0.791|0.113|-0.056|
|DP-off|0.891|0.826|0.910|0.878|
|B5-off|0.486|0.479|0.132|0.061|

**Direction cosine**, mean over the segments of each gap range

|Field|0–0.16|0.16–0.4|0.4–0.72|0.72–1.6|
|---|---:|---:|---:|---:|
|straight|0.879|0.747|0.623|0.641|
|B5|0.934|0.836|0.710|0.689|
|w6-0-2|0.917|0.834|0.717|0.683|
|w0-2-2|0.933|0.837|0.709|0.688|
|w0-2-0|0.902|0.769|0.673|0.667|
|w6-0-0|0.917|0.839|0.716|0.693|
|w6-2-0|0.937|0.837|0.722|0.691|
|EG|0.932|0.844|0.727|0.697|
|EG16|0.937|0.836|0.664|0.520|
|ES|0.935|0.794|0.474|0.523|
|ES12|0.932|0.841|0.681|0.546|
|ES1|0.932|0.810|0.628|0.483|
|ES2s1|0.935|0.844|0.681|0.539|
|ES2s2|0.937|0.848|0.685|0.545|
|DP|0.935|0.834|0.719|0.686|
|DPc|0.924|0.836|0.740|0.731|
|EG-off|0.916|0.770|0.596|0.563|
|ES-off|0.925|0.791|0.331|0.367|
|DP-off|0.926|0.807|0.679|0.625|
|B5-off|0.923|0.824|0.698|0.678|

T*/|q−g| (ground truth), segment means: 0–0.16: 1.31, 0.16–0.4: 1.27, 0.4–0.72: 1.54, 0.72–1.6: 1.74

**ES: error of each specialist on its own segment vs the best other specialist on that segment**

|Segment|0.00|0.08|0.16|0.24|0.32|0.40|0.48|0.56|0.64|0.72|0.80|0.88|0.96|1.04|1.12|1.20|1.28|1.36|1.44|1.52|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|own|0.09|0.08|0.23|0.41|0.48|0.44|0.35|0.34|0.36|0.36|0.34|0.33|0.27|0.26|0.32|0.27|0.26|0.26|0.27|0.28|
|best other|0.13|0.07|0.09|0.15|0.23|0.30|0.33|0.32|0.31|0.29|0.28|0.27|0.27|0.26|0.26|0.26|0.26|0.25|0.26|0.26|
|which|2|1|2|2|2|1|1|1|14|14|14|14|14|13|14|14|14|14|14|14|

**ES-off: error of each specialist on its own segment vs the best other specialist on that segment**

|Segment|0.00|0.08|0.16|0.24|0.32|0.40|0.48|0.56|0.64|0.72|0.80|0.88|0.96|1.04|1.12|1.20|1.28|1.36|1.44|1.52|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|own|0.12|0.11|0.14|0.22|0.27|0.33|0.41|0.37|0.33|0.29|0.27|0.25|0.24|0.23|0.24|0.23|0.24|0.25|0.24|0.24|
|best other|0.17|0.10|0.10|0.13|0.18|0.21|0.21|0.21|0.24|0.26|0.26|0.24|0.22|0.21|0.19|0.18|0.17|0.16|0.15|0.13|
|which|2|1|2|3|4|4|5|5|5|5|14|9|9|9|9|9|9|9|9|9|

No-wall:

**Relative error mean |log T/T*|**, mean over the segments of each gap range

|Field|0–0.16|0.16–0.4|0.4–0.72|0.72–1.6|
|---|---:|---:|---:|---:|
|straight|0.386|1.060|1.150|1.147|
|B5|0.204|0.226|0.708|0.906|
|w6-0-2|0.249|0.277|0.773|0.954|
|w0-2-2|0.215|0.240|0.732|0.921|
|w0-2-0|0.279|0.278|0.657|0.829|
|w6-0-0|0.207|0.232|0.716|0.935|
|w6-2-0|0.247|0.251|0.670|0.874|
|EG|0.346|0.273|0.654|0.834|
|EG16|0.088|0.212|0.712|0.909|
|ES|0.142|0.359|0.651|0.869|
|ES12|0.135|0.210|0.671|0.862|
|ES2s1|0.145|0.212|0.685|0.884|
|ES2s2|0.138|0.209|0.668|0.856|
|DP|0.137|0.227|0.367|0.357|
|EG-off|0.397|0.413|0.494|0.471|
|ES-off|0.346|0.446|0.483|0.677|
|B5-off|0.251|0.260|0.657|0.841|

**Bias median T/T***, mean over the segments of each gap range

|Field|0–0.16|0.16–0.4|0.4–0.72|0.72–1.6|
|---|---:|---:|---:|---:|
|straight|1.312|3.416|3.085|3.127|
|B5|0.914|0.954|0.514|0.399|
|w6-0-2|0.872|0.882|0.472|0.379|
|w0-2-2|0.876|0.918|0.496|0.392|
|w0-2-0|1.104|1.099|0.560|0.432|
|w6-0-0|0.987|0.986|0.509|0.387|
|w6-2-0|1.061|1.054|0.548|0.413|
|EG|1.366|1.122|0.569|0.430|
|EG16|1.018|0.945|0.512|0.396|
|ES|1.110|1.317|0.619|0.417|
|ES12|1.102|1.018|0.542|0.416|
|ES2s1|1.121|1.000|0.535|0.407|
|ES2s2|1.114|1.019|0.542|0.419|
|DP|1.104|1.162|0.817|0.718|
|EG-off|1.475|1.420|0.777|0.643|
|ES-off|1.386|1.446|0.812|0.513|
|B5-off|1.002|1.065|0.566|0.426|

**Spearman (which pairs are slow)**, mean over the segments of each gap range

|Field|0–0.16|0.16–0.4|0.4–0.72|0.72–1.6|
|---|---:|---:|---:|---:|
|straight|0.865|0.711|0.453|0.270|
|B5|0.604|0.525|0.172|0.018|
|w6-0-2|0.686|0.451|0.160|-0.059|
|w0-2-2|0.685|0.534|0.184|0.048|
|w0-2-0|0.112|0.182|0.028|0.045|
|w6-0-0|0.673|0.490|0.204|0.044|
|w6-2-0|0.241|0.308|0.090|0.020|
|EG|0.437|0.352|0.049|-0.006|
|EG16|0.916|0.675|0.183|0.025|
|ES|0.826|0.441|-0.042|0.024|
|ES12|0.835|0.624|0.164|0.010|
|ES2s1|0.825|0.603|0.140|0.015|
|ES2s2|0.853|0.626|0.162|0.026|
|DP|0.838|0.768|0.721|0.671|
|EG-off|0.516|0.381|0.232|0.216|
|ES-off|0.728|0.407|0.227|0.071|
|B5-off|0.206|0.249|0.078|0.008|

**Direction cosine**, mean over the segments of each gap range

|Field|0–0.16|0.16–0.4|0.4–0.72|0.72–1.6|
|---|---:|---:|---:|---:|
|straight|0.865|0.684|0.385|0.417|
|B5|0.925|0.792|0.551|0.465|
|w6-0-2|0.923|0.799|0.528|0.436|
|w0-2-2|0.923|0.785|0.545|0.463|
|w0-2-0|0.772|0.500|0.289|0.319|
|w6-0-0|0.900|0.770|0.513|0.502|
|w6-2-0|0.825|0.612|0.380|0.359|
|EG|0.699|0.518|0.338|0.351|
|EG16|0.878|0.753|0.494|0.374|
|ES|0.898|0.695|0.219|0.303|
|ES12|0.887|0.751|0.485|0.367|
|ES2s1|0.880|0.743|0.486|0.369|
|ES2s2|0.885|0.751|0.484|0.359|
|DP|0.896|0.789|0.552|0.506|
|EG-off|0.795|0.628|0.408|0.420|
|ES-off|0.886|0.657|0.191|0.235|
|B5-off|0.920|0.780|0.536|0.464|

T*/|q−g| (ground truth), segment means: 0–0.16: 1.31, 0.16–0.4: 1.30, 0.4–0.72: 2.52, 0.72–1.6: 2.96

**ES: error of each specialist on its own segment vs the best other specialist on that segment**

|Segment|0.00|0.08|0.16|0.24|0.32|0.40|0.48|0.56|0.64|0.72|0.80|0.88|0.96|1.04|1.12|1.20|1.28|1.36|1.44|1.52|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|own|0.09|0.17|0.22|0.40|0.45|0.52|0.66|0.70|0.73|0.77|0.80|0.78|0.80|0.84|0.85|0.88|0.92|0.95|0.98|0.99|
|best other|0.28|0.08|0.11|0.17|0.33|0.51|0.61|0.67|0.70|0.72|0.74|0.72|0.72|0.75|0.75|0.77|0.80|0.85|0.89|0.90|
|which|9|1|1|2|3|5|6|1|1|1|1|1|1|1|1|1|1|1|1|1|

**ES-off: error of each specialist on its own segment vs the best other specialist on that segment**

|Segment|0.00|0.08|0.16|0.24|0.32|0.40|0.48|0.56|0.64|0.72|0.80|0.88|0.96|1.04|1.12|1.20|1.28|1.36|1.44|1.52|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|own|0.18|0.43|0.53|0.42|0.38|0.48|0.46|0.49|0.50|0.58|0.60|0.55|0.58|0.61|0.66|0.68|0.74|0.77|0.82|0.88|
|best other|0.29|0.21|0.19|0.24|0.37|0.41|0.43|0.43|0.41|0.40|0.40|0.36|0.35|0.35|0.35|0.36|0.36|0.38|0.39|0.39|
|which|20|1|20|20|20|5|6|6|6|6|6|6|6|6|6|6|6|6|6|6|

- **Specialists fit only the shortest distances better.** Segments 1–2 are 2–3× more accurate than the original
  method and nearly unbiased; from 0.16 the specialists overestimate (cabinet bias 1.24–1.56 at 0.16–0.48) and lose
  direction (cosine 0.36 at 0.40–0.56). The by-model table shows the cause: model 1 or 2, trained on gaps ≤ 0.18 only,
  predicts most longer segments better than the specialists trained on them. The eikonal residual constrains |∇T|
  only; T's level is pinned by T(g, g) = 0 through the factor |q−g|, which matters only at short gaps. A shell
  specialist can satisfy the slope everywhere on its shell at any level.
- **Planning follows short-range accuracy.** The frozen queries are short (no-wall ≤ 0.68, median ≈ 0.23), and the
  planners act on the field near the robot. The generalist must fit every distance with one network and its
  short-range directions are poor (no-wall cosine 0.70 at gaps < 0.16, vs 0.90 for the specialists): 368 vs 1522.
  The two shortest specialists alone plan best of all fields (cabinet 716 / 746 / 724, no-wall 1689 / 1686 / 1679 over three seeds) — the eikonal method with only short
  pairs, no bank — and so do the shortest specialist alone (cabinet 721) and one model trained on gaps ≤ 0.176
  (cabinet 737, no-wall 1675): with short queries, the specialization that matters is "short pairs only".
- **Segment edges.** Mean log(T_inner/T_outer) at the 19 inner edges: ES −0.30 … +0.19 (inner model higher at
  0.48, 0.56, 1.12: a barrier for descent toward the goal), DP within ±0.016 except −0.14 at 0.16 (the first DP model
  starts 15 % above the eikonal base: a step down toward the goal, not a barrier), DP-c ±0.04. On 400 long cabinet queries ES converges 0 times with RK4.

## 2b. The segment-DP transition

Per-segment target / truth (median DP target over median T*, cabinet):

|Segment (gap)|0.16|0.24|0.32|0.40|0.48|0.56|0.64|0.72|0.80|0.88|0.96|1.04|1.12|1.20|1.28|1.36|1.44|1.52|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|DP cabinet (pool)|1.22|1.27|1.29|1.32|1.21|1.16|1.06|1.03|1.06|1.04|1.04|1.06|1.07|1.07|1.07|1.06|1.06|1.05|
|DP-c cabinet (continuous)|1.02|1.00|0.97|0.93|0.83|0.79|0.72|0.70|0.71|0.70|0.70|0.71|0.71|0.71|0.70|0.70|0.70|0.69|
|DP no-wall (pool)|1.21|1.22|1.19|0.97|0.70|0.72|0.76|0.74|0.73|0.75|0.75|0.73|0.75|0.73|0.72|0.69|0.69|0.69|

- **Pool split points** (the poses the planner can use): the lens of segment 3 holds 17 poses (median) and the best
  split adds a 16 % detour; the excess compounds over segments 3–6 (targets +22 … +32 %) and then fades as lenses
  grow (detour 1.06 → 1.02; targets 3–7 % above the truth beyond 0.64). Each DP model fits its targets closely
  (held-out |log| error 0.09 at segment 3, 0.006 at segment 20).
- **Continuous split points** (DP-c: min over pool poses, points along the segment and their gradient-refined
  versions; refined points win > 99.9 %): segments 3–5 within 3 % of the truth, then the chain drifts low (0.93 at
  segment 6, ~0.70 from segment 9). Off-pool split points let the min use regions the frozen models never saw (inside
  obstacles, off the free space): the optimizer finds where they read low.
- **Long distances.** The DP field is the most accurate at long range on both scenes (fit tables above) and the most
  consistent across edges. Long cabinet queries (400, gap 0.8–1.5, endpoints of the frozen suites, uncertified —
  comparable only between fields):

|Field|RK4 success /400|RK4 converged /400|RK4 success, first 100|MPPI seed 0, first 100|
|---|---:|---:|---:|---:|
|B5|23|89|7|9|
|ES|0|0|0|4|
|ES12|20|67|7|8|
|DP|29|138|10|9|
|DPc|26|124|8|8|
|EG|29|78|8|9|
|ES2s1|22|59|8|9|
|ES-off|0|0|0|2|
|B5-off|25|114|9|9|
|w0-2-2|24|99|7|8|

- **No-wall.** The true travel time jumps from 1.35× to 2.8× the gap beyond 0.48 (paths follow the orientation
  manifold around); DP gets closest (relative error 0.37/0.36 at 0.4–0.72/0.72–1.6 vs ≥ 0.47; bias 0.72 at the longest
  range vs ≤ 0.64 for every other field, 0.82 at 0.4–0.72 where the offset networks reach 0.78–0.81) but still
  underestimates: the min picks split points whose pieces the frozen models under-read. Planning on the short frozen queries is below the
  specialists (1442 vs 1522; mid-range overestimate).

## 2c. Position + offset

|Field|Cabinet RK4 /600|Cabinet MPPI /1800|No-wall RK4 /600|No-wall MPPI /1800|
|---|---:|---:|---:|---:|
|B5 (two positions)|231 (229/229/236)|694 (688/697/698)|540 (550/537/532)|1590 (1598/1586/1585)|
|B5-off (position + offset)|218|652|502|1496|
|EG|238|713|117|368|
|EG-off|211|633|211|608|
|ES|226|688|463|1522|
|ES-off|225|688|412|1399|
|DP|241|706|467|1442|
|DP-off|220|694|—|—|

- The offset network never beats the two-position network on the frozen benchmarks except for the no-wall generalist
  (608 vs 368, both far below every other field). With the same encoder for q and d = g − q, it fits the eikonal
  residual 2–3× worse (end-of-training loss 0.09–0.19 vs 0.04–0.07 for the cabinet specialists) and overestimates
  short gaps (no-wall bias 1.39–1.48 below 0.16); at long range it is less biased than the two-position eikonal
  fields (no-wall relative error 0.47–0.68 vs 0.83–0.87), not than DP.
- **DP chain on the offset network** (cabinet; base = the offset specialists 1–2): RK4 220, MPPI 694 vs 241 / 706 for the two-position chain. Its targets start from the less accurate offset base and stay 24–41 % above the truth at every level (two-position chain: 22–32 % early, 3–7 % late), yet it ranks long pairs best of all fields (Spearman 0.91 / 0.88 at 0.4–0.72 / 0.72–1.6 vs 0.84 / 0.75): the offset network gets the shape of the long-range field, and the chain inherits the base's level.

## 3. Loss, decoder, split count, TD and ball-Bellman terms on the DP chain

Asked on 2026-10-29 (after the part-2b write-up): (1) other regression losses than SmoothL1; (2) other distance
functions, or a decoder that adds over the minimum possible travel time — the speed is ≤ 1, so |q−g| is a lower
bound that the field should never go under; (3) does the number of split points sampled change the accuracy, and by
how much; (4) a TD loss built on the first two networks: they are accurate at short range, so step a small Δt from q
toward the goal, read the speed from their gradient and ask T(q, g) = T(q + Δt u*, g) + Δt / S(q); (5) Bellman, DP
and TD each alone, then combined.

**Setup** (`dp3.py`; its defaults reproduce `dp.py` bitwise). Cabinet only; base = the eikonal specialists 1–2 of
the ES run; the chain is trained for segments 3–9 (gaps 0.16–0.72; every frozen query is ≤ 0.68) and the saved field
keeps the ES specialists for segments 10–20. Same seed, 400k pairs, 15,620 updates and a fresh model per level, AdamW
2e-4, clip 2. Each level is scored against the ground truth of part 2 on 1,024 truth pairs of its own segment, and in
every level the targets of those pairs are recomputed with 8 / 32 / 128 / 384 / 1,536 split candidates (task 3).
- **dp**: loss(T(i,j), min_k T_≤m(i,k) + T_≤m(k,j)) with the chosen regression loss; the targets are the pool split
  points of part 2b.
- **td**: SmoothL1(T(q,g), T(q + Δt u*, g) + |Δq| / S_base(q)), Δt = 0.01, u* = −∇_q T of the live chain, and
  S_base(q) = 1 / |∇_q T_base(q, q + 0.04 v)| for a random unit v — the eikonal relation of the base specialists read
  as a speed probe. The chain value at the stepped point is the live chain (the model in training plus the levels
  below it), detached.
- **bell**: SmoothL1(T(q,g), min_u [T_base(q, q + ρu) + T(q + ρu, g)]), ρ = 0.05, u over the descent direction and 8
  random unit directions: a Bellman backup whose local cost is the base field (a ball in which the specialists are
  accurate) and whose continuation is the live chain, detached.
- Weights 1 / 1 / 1 when combined; td and bell always use SmoothL1, `--loss` applies to the dp term.
- **Decoders** on the same trunk: mult T = |d|(1 + softplus r) (the original network), add T = |d| + softplus r,
  exp T = |d| exp r (unbounded control); d = g − q.

<!-- T3A variants -->
|Field|Variant|MPPI /1800 (seeds)|RK4 /600|Truth error, segments 3–9|Median T/T*, segments 3–9|Train min|
|---|---|---:|---:|---:|---:|---:|
|DP3|DP, SmoothL1, mult decoder, 384 candidates (the dp.py chain)|716 (238/238/240); seed 1: 709; seed 2: 694|241|0.228 (0.20–0.26)|1.12–1.22|32|
|DP3-l1|DP, L1|736 (245/246/245); seed 1: 718; seed 2: 714|244|0.218 (0.19–0.24)|1.09–1.20|32|
|DP3-l2|DP, L2|710 (238/237/235)|241|0.228 (0.20–0.25)|1.12–1.22|99*|
|DP3-logl1|DP, L1 of log T|718 (239/239/240)|244|0.222 (0.19–0.25)|1.11–1.21|32|
|DP3-logl2|DP, L2 of log T|725 (243/242/240)|242|0.228 (0.20–0.26)|1.12–1.21|97*|
|DP3-rel|DP, relative error (T − y)/y|729 (240/246/243)|243|0.224 (0.20–0.25)|1.08–1.20|32|
|DP3-add|DP, decoder T = ‖q−g‖ + softplus r|711 (236/237/238)|238|0.229 (0.20–0.26)|1.10–1.21|32|
|DP3-exp|DP, decoder T = ‖q−g‖ exp r|704 (234/234/236)|245|0.226 (0.19–0.25)|1.12–1.22|32|
|DP3-c32|DP, 32 split candidates|721 (239/243/239)|242|0.227 (0.20–0.26)|1.15–1.22|16|
|DP3-c1536|DP, 1536 split candidates|706 (237/237/232)|241|0.231 (0.20–0.25)|1.13–1.23|182*|
|TD|TD only|714 (236/238/240)|234|0.209 (0.11–0.27)|0.82–1.05|15|
|BELL|ball Bellman only|714 (237/238/239)|239|0.154 (0.10–0.19)|0.98–1.05|18|
|TD+BELL|TD + ball Bellman|716 (240/238/238)|242|0.155 (0.10–0.19)|0.94–1.04|20|
|DP+TD|DP + TD|719 (239/242/238)|240|0.221 (0.19–0.25)|1.10–1.20|40|
|DP+BELL|DP + ball Bellman|728 (243/242/243)|241|0.197 (0.17–0.22)|1.07–1.16|43|
|DP+TD+BELL|DP + TD + ball Bellman|742 (247/246/249); seed 1: 736; seed 2: 725|244|0.197 (0.17–0.22)|1.07–1.16|45|

Train minutes with * ran on the GPU shared with another session's job (about 4× slower); the others are comparable.
MPPI is the frozen certified cabinet benchmark (600 queries × 3 planning seeds); "seed 1 / seed 2" are training-seed
repeats of the same chain. Truth error = mean |log T/T*| over the 1,024 truth pairs of each level; median T/T* likewise.

**Per level**, truth error (mean |log T/T*|) and median T/T*:

<!-- T3B truth error per level -->
|Field|0.16–0.24|0.24–0.32|0.32–0.40|0.40–0.48|0.48–0.56|0.56–0.64|0.64–0.72|
|---|---:|---:|---:|---:|---:|---:|---:|
|DP3|0.204|0.234|0.255|0.255|0.233|0.221|0.197|
|DP3-l1|0.191|0.220|0.242|0.244|0.228|0.212|0.188|
|DP3-l2|0.205|0.239|0.247|0.254|0.233|0.220|0.195|
|DP3-logl1|0.197|0.227|0.247|0.248|0.230|0.213|0.191|
|DP3-logl2|0.206|0.236|0.249|0.255|0.234|0.220|0.196|
|DP3-rel|0.196|0.219|0.247|0.251|0.236|0.223|0.198|
|DP3-add|0.202|0.233|0.252|0.256|0.237|0.223|0.200|
|DP3-exp|0.205|0.229|0.253|0.249|0.231|0.219|0.194|
|DP3-c32|0.204|0.232|0.252|0.256|0.230|0.218|0.196|
|DP3-c1536|0.216|0.244|0.254|0.255|0.234|0.221|0.197|
|TD|0.106|0.134|0.190|0.240|0.269|0.263|0.263|
|BELL|0.104|0.124|0.146|0.176|0.186|0.178|0.165|
|TD+BELL|0.100|0.120|0.149|0.179|0.187|0.179|0.169|
|DP+TD|0.186|0.222|0.245|0.253|0.231|0.218|0.195|
|DP+BELL|0.166|0.193|0.216|0.223|0.209|0.198|0.175|
|DP+TD+BELL|0.168|0.195|0.216|0.220|0.208|0.195|0.175|

<!-- T3C median T/T* per level -->
|Field|0.16–0.24|0.24–0.32|0.32–0.40|0.40–0.48|0.48–0.56|0.56–0.64|0.64–0.72|
|---|---:|---:|---:|---:|---:|---:|---:|
|DP3|1.20|1.21|1.22|1.20|1.16|1.16|1.12|
|DP3-l1|1.18|1.19|1.20|1.17|1.15|1.14|1.09|
|DP3-l2|1.20|1.22|1.20|1.20|1.17|1.17|1.12|
|DP3-logl1|1.19|1.20|1.21|1.19|1.17|1.15|1.11|
|DP3-logl2|1.20|1.21|1.21|1.20|1.17|1.16|1.12|
|DP3-rel|1.19|1.18|1.20|1.17|1.15|1.14|1.08|
|DP3-add|1.19|1.21|1.20|1.18|1.15|1.15|1.10|
|DP3-exp|1.20|1.21|1.22|1.19|1.17|1.16|1.12|
|DP3-c32|1.19|1.21|1.22|1.21|1.16|1.17|1.15|
|DP3-c1536|1.20|1.23|1.22|1.20|1.18|1.17|1.13|
|TD|1.05|1.03|1.01|0.99|0.94|0.88|0.82|
|BELL|1.05|1.05|1.04|1.02|1.02|1.01|0.98|
|TD+BELL|1.04|1.04|1.03|1.01|1.00|0.98|0.94|
|DP+TD|1.17|1.19|1.20|1.19|1.15|1.14|1.10|
|DP+BELL|1.15|1.16|1.16|1.15|1.12|1.11|1.07|
|DP+TD+BELL|1.15|1.16|1.16|1.14|1.12|1.10|1.07|

**Split candidates** (task 3): in every level of the baseline chain the DP targets of the 1,024 truth pairs recomputed
with k candidates — median target/T* / mean |log target/T*| — and the truth error of the chains trained with 32 /
384 / 1,536 candidates:

<!-- T3D candidate sweep -->
|Segment|Lens poses (median)|8 candidates|32 candidates|128 candidates|384 candidates|1536 candidates|Trained chain: 32|384|1536|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|3 (0.16–0.24)|17|1.180 / 0.198|1.179 / 0.197|1.179 / 0.197|1.179 / 0.197|1.179 / 0.197|0.204|0.204|0.216|
|4 (0.24–0.32)|130|1.217 / 0.242|1.210 / 0.235|1.210 / 0.235|1.210 / 0.235|1.210 / 0.235|0.232|0.234|0.244|
|5 (0.32–0.40)|465|1.214 / 0.251|1.203 / 0.247|1.202 / 0.247|1.202 / 0.247|1.202 / 0.247|0.252|0.255|0.254|
|6 (0.40–0.48)|1256|1.210 / 0.258|1.201 / 0.255|1.199 / 0.254|1.199 / 0.254|1.199 / 0.254|0.256|0.255|0.255|
|7 (0.48–0.56)|3063|1.188 / 0.239|1.171 / 0.234|1.167 / 0.234|1.167 / 0.234|1.167 / 0.234|0.230|0.233|0.234|
|8 (0.56–0.64)|6832|1.179 / 0.225|1.165 / 0.221|1.164 / 0.221|1.164 / 0.221|1.164 / 0.221|0.218|0.221|0.221|
|9 (0.64–0.72)|13890|1.142 / 0.201|1.128 / 0.196|1.123 / 0.196|1.118 / 0.196|1.117 / 0.196|0.196|0.197|0.197|

**Extra terms**: the speed read from the base gradient against the pool's true speed at q, and how often the descent
direction wins the ball min:

<!-- T3E extra-term diagnostics -->
|Field|Loss parts at the end (dp / td / bell)|Speed from the base gradient: median \|log S/S*\||Descent direction wins the ball min|
|---|---|---:|---:|
|TD|— / 0.00008 / —|0.071|—|
|BELL|— / — / 0.00031|0.068|0.96|
|TD+BELL|— / 0.00010 / 0.00029|0.068|0.95|
|DP+TD|0.00053 / 0.00013 / —|0.072|—|
|DP+BELL|0.00066 / — / 0.00048|0.070|0.97|
|DP+TD+BELL|0.00073 / 0.00011 / 0.00043|0.070|0.96|

**Fit by gap range** (`fit.py` on the ~20k truth pairs; 0–0.16 is the shared base; 0.72–1.6 is the ES padding and
omitted):

<!-- T3F rel_error -->
|Relative error mean \|log T/T*\||0–0.16|0.16–0.4|0.4–0.72|
|---|---:|---:|---:|
|ES|0.100|0.378|0.373|
|DP3|0.095|0.230|0.227|
|DP3-l1|0.093|0.216|0.219|
|DP3-l2|0.095|0.229|0.226|
|DP3-logl1|0.093|0.222|0.221|
|DP3-logl2|0.094|0.229|0.227|
|DP3-rel|0.094|0.219|0.228|
|DP3-add|0.095|0.228|0.230|
|DP3-exp|0.095|0.228|0.224|
|DP3-c32|0.095|0.228|0.225|
|DP3-c1536|0.095|0.236|0.227|
|TD|0.089|0.142|0.259|
|BELL|0.089|0.124|0.177|
|TD+BELL|0.089|0.123|0.180|
|DP+TD|0.092|0.217|0.225|
|DP+BELL|0.092|0.191|0.202|
|DP+TD+BELL|0.092|0.192|0.201|

<!-- T3F bias -->
|Bias median T/T*|0–0.16|0.16–0.4|0.4–0.72|
|---|---:|---:|---:|
|ES|1.038|1.426|0.965|
|DP3|1.035|1.208|1.155|
|DP3-l1|1.034|1.190|1.135|
|DP3-l2|1.035|1.207|1.161|
|DP3-logl1|1.034|1.197|1.146|
|DP3-logl2|1.035|1.204|1.153|
|DP3-rel|1.034|1.189|1.133|
|DP3-add|1.035|1.201|1.140|
|DP3-exp|1.035|1.206|1.157|
|DP3-c32|1.035|1.208|1.163|
|DP3-c1536|1.035|1.215|1.167|
|TD|1.027|1.031|0.909|
|BELL|1.027|1.049|1.007|
|TD+BELL|1.026|1.038|0.984|
|DP+TD|1.031|1.186|1.139|
|DP+BELL|1.032|1.158|1.109|
|DP+TD+BELL|1.031|1.158|1.104|

<!-- T3F spearman -->
|Spearman (which pairs are slow)|0–0.16|0.16–0.4|0.4–0.72|
|---|---:|---:|---:|
|ES|0.929|0.577|0.149|
|DP3|0.936|0.872|0.829|
|DP3-l1|0.938|0.888|0.846|
|DP3-l2|0.936|0.877|0.833|
|DP3-logl1|0.938|0.888|0.844|
|DP3-logl2|0.937|0.890|0.832|
|DP3-rel|0.938|0.888|0.825|
|DP3-add|0.937|0.873|0.816|
|DP3-exp|0.936|0.873|0.834|
|DP3-c32|0.936|0.873|0.834|
|DP3-c1536|0.936|0.874|0.833|
|TD|0.942|0.869|0.716|
|BELL|0.942|0.904|0.869|
|TD+BELL|0.942|0.903|0.880|
|DP+TD|0.939|0.883|0.830|
|DP+BELL|0.940|0.900|0.866|
|DP+TD+BELL|0.940|0.904|0.867|

<!-- T3F cosine -->
|Direction cosine|0–0.16|0.16–0.4|0.4–0.72|
|---|---:|---:|---:|
|ES|0.935|0.794|0.474|
|DP3|0.935|0.833|0.664|
|DP3-l1|0.934|0.842|0.668|
|DP3-l2|0.935|0.836|0.661|
|DP3-logl1|0.935|0.845|0.666|
|DP3-logl2|0.936|0.835|0.665|
|DP3-rel|0.936|0.842|0.668|
|DP3-add|0.935|0.840|0.668|
|DP3-exp|0.935|0.833|0.661|
|DP3-c32|0.935|0.835|0.660|
|DP3-c1536|0.935|0.834|0.656|
|TD|0.927|0.817|0.649|
|BELL|0.930|0.821|0.643|
|TD+BELL|0.928|0.824|0.649|
|DP+TD|0.934|0.839|0.672|
|DP+BELL|0.935|0.843|0.674|
|DP+TD+BELL|0.935|0.839|0.670|

### Findings

1. **Regression loss (task 1).** L1 736, relative 729, log-L2 725, log-L1 718, SmoothL1 716, L2 710 — a 26-point
   spread, with the training-seed repeats below. The fit to the truth is the same for every loss (error 0.22–0.23,
   median T/T* 1.19–1.21 at 0.16–0.4, 1.13–1.16 at 0.4–0.72): each chain fits its targets to within 2–9 % held-out,
   and the 12–22 % overestimate is in the targets, not in the regression. The robust losses (L1, relative) sit
   2 % lower than the mean-seeking ones (bias 1.19 / 1.13 vs 1.21 / 1.16): the target errors are skewed upward
   (the pairs whose lens forces a long detour), and L1 follows the median of its targets where SmoothL1 and L2 follow
   the mean. Which is also why the L1 chain plans best in this sweep.
2. **Lower bound and decoder (task 2).** The original decoder already never predicts below the bound:
   T = |q−g|(1 + softplus r) ≥ |q−g| for every r. The 'add' decoder T = |q−g| + softplus r is the same bound in
   another parametrization and changes nothing (711 vs 716, identical fit); the unbounded 'exp' decoder is also the
   same (704, identical fit). The bound never binds because the chain's error has the opposite sign and size: the
   truth is 1.3–1.5× the gap and the chain reads 1.1–1.2× the truth. A distance-function decoder (the straight-line
   integral ∫ ds/S as the base term, the network a residual over it) was not run: every DP target evaluates 2 × 384
   pieces, and the integral would multiply that by its samples.
3. **Number of split candidates (task 3). It does not matter beyond 32.** Targets from 32, 128, 384 and 1,536
   candidates agree to 0.1 % in level and 0.001 in error at every level; 8 candidates cost 0.1–1.7 % in level
   (most at segments 7–9, lenses of 3–14k poses) and ≤ 0.007 in error. The winner always comes from the half of
   the candidates nearest the straight line (100 % of pairs at every level). The chains agree: 32 candidates 721,
   384 716, 1,536 706 (seed noise), and the 32-candidate chain trains in half the time. The targets' 12–22 %
   overestimate is therefore not a search error; it is the pool's: the best *available* split point costs a detour
   (16 % at segment 3, where a lens holds 17 poses) and the pieces of later levels inherit it.
4. **TD from the short-range networks (task 4).** The speed probe works: the base gradient gives the local speed
   within 7 % (median |log S/S*| 0.068–0.078 at every level). The TD chain alone is unbiased at 0.16–0.4 (median
   1.03) and the most accurate chain of all at segments 3–4 (error 0.11 / 0.13), but it drifts low with range (0.91
   at 0.4–0.72, 0.82 at segment 9) and ranks long pairs worst (Spearman 0.72 at 0.4–0.72 vs 0.83 for DP): with a
   0.01 step the target is 99 % the chain's own value one step closer to the goal, so each level bootstraps from
   itself and is anchored only through the base. A small-step consistency pins the slope, not the level — the same
   failure as the mid-range specialists of 2a, now with the sign of the drift reversed by the backup. Planner 714,
   level with DP.
5. **Ball Bellman (task 5).** Alone it is the best-calibrated field of the study: median T/T* 1.05 / 1.01, error
   0.12 / 0.18 at 0.16–0.4 / 0.4–0.72 (DP 0.23 / 0.23), Spearman 0.90 / 0.87 (DP 0.87 / 0.83). The ρ = 0.05 ball
   backs up through the base's *measured* local cost, and the descent direction wins the min in 95–97 % of rows, so
   the term is a DP transition with a short first piece and a bootstrapped second piece — it keeps the DP's level
   anchoring (the first piece is read from the base at every level) without the pool's detour. Planner 714.
   **Combinations**: DP + bell 728, DP + TD + bell 742 (+8–11 on every planning seed over DP), DP + TD 719,
   TD + bell 716. Adding the ball term to DP lowers its bias (1.16 / 1.10 vs 1.21 / 1.16) and improves its rank
   (0.90 / 0.87 vs 0.87 / 0.83) with the direction cosine unchanged (0.84 / 0.67 for every DP chain).
6. **What the planner responds to.** All sixteen chains plan within 704–742 on the frozen benchmark, and their
   direction cosine at 0.16–0.4 — the range MPPI works in — is 0.82–0.85 for all of them. The three best planners
   (DP + TD + bell 742, L1 736, DP + bell 728) are the chains with the lowest bias among the DP chains, not those
   with better directions; the two terms that fix the bias outright (TD, bell alone) do not plan better than DP.
   The 20–26-point gains are at the edge of the training-seed spread of comparable fields (±5–10 per field, section
   Caveats). Training-seed repeats (seeds 0 / 1 / 2): DP3 716 / 709 / 694 (mean 706), DP3-l1 736 / 718 / 714 (mean 723), DP + TD + bell 742 / 736 / 725 (mean 734). A DP chain's training-seed spread is therefore about ±11 MPPI, wider than the ±5–10 of the earlier fields. DP + TD + bell survives it: +28 on seed means, and its worst seed (725) is above DP3's best (716). L1 is +17 on seed means but its worst seed (714) is level with DP3's best, so the loss result is a tendency, not a separation; the single-seed differences between the other losses, decoders and candidate counts (704–729) are within the spread.

## 4. Larger cabinets: does the DP chain's convergence gain hold?

**Setup.** The project's two larger cabinets, `cabinet_plus10` and `cabinet_plus20` (`experiments/cabinet/make_scenes.py`:
every compartment opening 1.1× / 1.2× as wide as the cabinet's, same boards and panels), each with its own pipeline run
(`c5ebfc5b` / `c41ae437`: the original method's fifth-round checkpoint B5, its prepared pool and speed network, and its
frozen MPPI / RK4 suites). On each scene, the section-2 recipe unchanged: 20 eikonal specialists (ES, one per
0.08 segment, 7 min) and the full 20-segment DP chain on top of specialists 1–2 (DP: 18 levels, 400k pairs and
384 split candidates per level, 15,620 updates per model, 78–80 min). Evaluation as in 2b: the scene's frozen
suites (600 queries × 3 planning seeds for MPPI, /600 for RK4), and 400 long queries (gap 0.8–1.5, endpoints of the
frozen suites, uncertified, comparable only between fields) by RK4 descent — "converged" = the descent reaches the
goal, collisions ignored; "success" = reaches it collision-free — with the first 100 also planned by MPPI (seed 0).
One training seed per field; the cabinet rows repeat section 2b for comparison.

<!-- T4Z cabinet long rows -->
|Scene|Field|RK4 success /400|RK4 converged /400|RK4 success, first 100|MPPI seed 0, first 100|
|---|---|---:|---:|---:|---:|
|cabinet|B5|23|89|7|9|
|cabinet|ES|0|0|0|4|
|cabinet|DP|29|138|10|9|

<!-- T4 long queries larger cabinets -->
|Scene|Field|RK4 success /400|RK4 converged /400|RK4 success, first 100|MPPI seed 0, first 100|
|---|---|---:|---:|---:|---:|
|cabinet_plus10|B5|13|88|2|5|
|cabinet_plus10|ES|0|0|0|2|
|cabinet_plus10|DP|21|149|3|3|
|cabinet_plus20|B5|22|82|9|10|
|cabinet_plus20|ES|0|0|0|8|
|cabinet_plus20|DP|22|150|6|8|

Failure modes of the 400 descents (a query can fail both ways; success = neither):

<!-- T4C failure modes -->
|Scene|Field|Never reaches the goal|Collides|
|---|---|---:|---:|
|cabinet|B5 / ES / DP|311 / 400 / 262|249 / 275 / 288|
|cabinet_plus10|B5 / ES / DP|312 / 400 / 251|279 / 253 / 298|
|cabinet_plus20|B5 / ES / DP|318 / 400 / 250|230 / 259 / 285|

<!-- T4B frozen benchmarks larger cabinets -->
|Scene|Field|RK4 /600|MPPI /1800 (planning seeds)|
|---|---|---:|---:|
|cabinet_plus10|B5|224|682 (230/227/225)|
|cabinet_plus10|ES|238|721 (242/240/239)|
|cabinet_plus10|DP|254|732 (244/247/241)|
|cabinet_plus20|B5|223|646 (216/215/215)|
|cabinet_plus20|ES|248|726 (242/242/242)|
|cabinet_plus20|DP|256|725 (243/241/241)|

(Cabinet frozen scores for comparison, section 2c: B5 231 / 694, ES 226 / 688, DP 241 / 706.)

- **The convergence gain replicates and grows with the cabinet.** Gradient descent on the DP chain reaches the goal
  on 138 / 149 / 150 of the 400 long queries (cabinet / +10 % / +20 %) against 89 / 88 / 82 for the original method
  and 0 for the specialists on every scene — 1.55×, 1.69× and 1.83× the original method. The specialists' zero is the
  same mechanism as in 2a: beyond their segment the specialists have no level, so the stitched field has no gradient
  towards the goal at gap 0.8–1.5; the chain gives the long range a level (section 2b: the most accurate field at
  0.72–1.6 on the cabinet) and the descent follows it.
- **Collision-free success does not grow with it.** Successes are 29 / 21 / 22 for the chain and 23 / 13 / 22 for the
  original method: ahead on the cabinet and the +10 % cabinet, tied on the +20 % one. The failure table shows why:
  the chain cuts the "never reaches the goal" failures from 311–318 to 250–262, but its descents collide on 285–298
  queries, 10–55 more than the original method's (230–279). Converging and colliding are decided by different parts
  of the field: the level of T at long range decides whether the descent keeps moving towards the goal, and that is
  what the chain fixes; where the path goes near an obstacle is decided by the direction of ∇T at short range, which
  the chain inherits from its pool-detour targets and the specialist pieces (direction cosine 0.84 at 0.16–0.4 for
  every DP chain in section 3, the same as the original method's 0.83). The original method collides least on the +20 % cabinet
  (230 vs 249 / 279) while the chain's collisions stay at 285–298, so the two meet there. A certified success rate on long queries would need a safer direction field, not
  a better level.
- **On the frozen benchmarks the chain is the best or level with the best on all three scenes.** MPPI: DP 706 / 732 /
  725 against the original method's 694 / 682 / 646 and the specialists' 688 / 721 / 726; RK4: DP 241 / 254 / 256,
  the best field on every scene. The original method loses 48 points from the cabinet to the +20 % cabinet while the
  specialists and the chain stay at 720–730, so the chain's margin over it grows (+12 → +50 → +79). On the +20 % cabinet
  the chain and the specialists tie (725 vs 726: the frozen queries are short, where the chain reuses specialists
  1–2 and the DP models' 12–22 % overestimate buys nothing, section 3 finding 6). All margins over the specialists
  are within the ±5–10 training-seed spread; the margins over the original method on the larger cabinets are not.
- **MPPI does not profit on the long queries.** First-100 MPPI successes are 9 / 5 / 10 for the original method,
  4 / 2 / 8 for the specialists and 9 / 3 / 8 for the chain: MPPI's rollouts work at 0.16–0.4 (section 3), where the
  three fields agree in direction, and every field improves with wider openings (the specialists 2 → 8 from +10 %
  to +20 %). The RK4 descent is what reads the long-range level, and that is where the chain's gain shows.
- **The chain is the same chain on all three scenes.** Per level, the pool detour (1.17 at segment 3 → 1.015 at
  segment 20), the target/gap ratio (1.53–1.57 at segment 3, 1.85–1.87 at segment 20), the lens sizes (16 poses at
  segment 3, 378k at segment 20) and the held-out fit (0.09 → 0.006 |log| error) agree to two digits between the
  cabinet and its +10 % / +20 % versions: 10–20 % more clearance changes the pool's statistics very little, so what
  differs between the scenes is the specialist pieces and the original method's checkpoint, not the DP targets.

## 5. Warm-starting each DP level from the previous level

**Setup.** `dp3.py --init warm`: model m+1 starts from model m's trained weights instead of a fresh default
initialization (same architecture, so the state dict loads as is; the RNG is seeded once per run either way).
Everything else is the DP3 recipe of section 3 (cabinet, eikonal specialists 1–2 of `b7ab5a49`, SmoothL1, 384 split
candidates, chain to segment 9), three training seeds, compared with DP3's three seeds.

|Init|MPPI /1800 (training seeds 0 / 1 / 2)|mean|RK4 /600|
|---|---|---:|---|
|fresh (DP3)|716 / 709 / 694|706|241 / 240 / 240|
|warm|713 / 725 / 710|716|245 / 245 / 243|

- **Same field, reached sooner.** Before any update, the inherited weights already read the next level's targets
  within 15 % at segment 3 (held-out |log| error 0.150) and within 2–7 % from segment 4 on (0.065 → 0.018); after
  training the held-out error is the same or slightly lower (segment 9: 0.016 vs 0.018), and the truth fit is
  unchanged. Neighbouring levels answer nearly the same function — T at gap g and at gap g + 0.08 differ mostly by
  scale — so the previous level is a close initial guess, and the optimizer spends its 15,620 updates refining it.
  Time per level drops slightly (254 vs 271 s, of which 190–230 s is the target computation, not training).
- **Planning: +10 MPPI on seed means, +4 RK4 on every seed.** The MPPI gain sits inside the ±11 seed spread of a
  DP chain (warm's worst seed, 710, is level with fresh seeds 1 and 0); the RK4 gain is small but consistent on all
  three seeds. Warm starting is a free speed-up with no downside; it is not a different field.

## 6. Other scenes: unconstrained planning in clutter and between platforms

**Setup.** Two easier arm scenes, still the cabinet task's unconstrained planning (no zones or task axes,
clearance-only speed, release scorer with its tilt gate off): `cabinet_clutter` (the bottle, jar and kettle of
`obstacle_scenes/clutter.yml` on the table) and `cabinet_platforms` (the two raised platforms of
`obstacle_scenes/platforms.yml`), in `experiments/cabinet/scenes/`. The obstacles are copied verbatim; the cupboard's
compartments are replaced by regions around the objects (clutter: near_bottle / near_jar / near_kettle, each
footprint + 0.08 m up to 0.05 m above the top) or above and between the platforms (platforms: cup_top / pour_top /
trench), and the cupboard's outer box by a `focus:` box, their bounding box (`geometry.scene_boxes` reads it when
present). Each scene has its own frozen suites (`tests_clutter/`, `tests_platforms/`: 6 strata × 100 certified
queries, frozen by `freeze_tests.py`), its own pipeline run (prepared pool, speed network, and the original method's
fifth-round checkpoint B5), the 20 eikonal specialists (ES), and the two DP chains asked for — DP alone (DP3
recipe) and DP + TD + bell (section 3's best) — as full 20-segment chains on specialists 1–2.

<!-- T6F frozen -->
|Scene|Field|MPPI /1800 (planning seeds)|RK4 /600|straight-line-feasible queries, MPPI|detour queries, MPPI|
|---|---|---:|---:|---:|---:|
|clutter|B5|1165 (388/388/389)|401|865 /882|300 /918|
|clutter|ES|1294 (430/433/431)|448|835 /882|459 /918|
|clutter|DP|1251 (417/419/415)|447|853 /882|398 /918|
|clutter|DP (seed 1)|1248 (415/419/414)|438|857 /882|391 /918|
|clutter|DP+TD+BELL|1281 (432/426/423)|450|855 /882|426 /918|
|clutter|DP+TD+BELL (seed 1)|1285 (425/429/431)|455|855 /882|430 /918|
|clutter|DP with ES models 3–6|1306 (433/439/434)|448|844 /882|462 /918|
|platforms|B5|1532 (512/508/512)|516|1063 /1077|469 /723|
|platforms|ES|1576 (528/523/525)|524|1047 /1077|529 /723|
|platforms|DP|1493 (496/498/499)|511|1054 /1077|439 /723|
|platforms|DP (seed 1)|1503 (501/501/501)|516|1054 /1077|449 /723|
|platforms|DP+TD+BELL|1546 (516/516/514)|526|1055 /1077|491 /723|
|platforms|DP with ES models 3–6|1587 (530/528/529)|524|1056 /1077|531 /723|

<!-- T6D distance -->
|Scene|Field|gap 0–0.16|0.16–0.32|0.32–0.48|0.48–0.64|
|---|---|---:|---:|---:|---:|
|clutter|queries (MPPI out of 3×)|44|383|156|17|
|clutter|B5|90|719|308|48|
|clutter|ES|102|830|333|29|
|clutter|DP|100|781|322|48|
|clutter|DP (seed 1)|100|789|311|48|
|clutter|DP+TD+BELL|102|804|327|48|
|clutter|DP+TD+BELL (seed 1)|101|806|327|51|
|clutter|DP with ES models 3–6|102|831|335|38|
|platforms|queries (MPPI out of 3×)|47|392|151|10|
|platforms|B5|120|974|411|27|
|platforms|ES|126|1020|409|21|
|platforms|DP|126|959|381|27|
|platforms|DP (seed 1)|126|960|389|28|
|platforms|DP+TD+BELL|126|990|400|30|
|platforms|DP with ES models 3–6|126|1020|414|27|

<!-- T6L long -->
|Scene|Field|RK4 success /400|RK4 converged /400|never reaches the goal|collides|RK4 success, first 100|MPPI seed 0, first 100|
|---|---|---:|---:|---:|---:|---:|---:|
|clutter|B5|57|104|296|130|9|10|
|clutter|ES|0|0|400|151|0|6|
|clutter|DP|66|151|249|198|13|9|
|clutter|DP+TD+BELL|75|143|257|148|13|10|
|clutter|DP with ES models 3–6|3|13|387|182|0|8|
|platforms|B5|57|90|310|73|14|16|
|platforms|ES|3|3|397|86|1|18|
|platforms|DP|63|142|258|191|17|14|
|platforms|DP+TD+BELL|89|139|261|112|19|18|
|platforms|DP with ES models 3–6|8|19|381|170|2|18|



- **The frozen suites are short and the specialists win them.** No frozen query on either scene is longer than
  0.55 (median 0.29), so the DP levels never answer beyond segment 7. ES leads both chains on MPPI (clutter 1294 vs
  DP 1251 / 1248 over two training seeds and DP + TD + bell 1281 / 1285; platforms 1576 vs 1493 / 1503 and 1546). RK4 is level
  (447–450 and 511–526). DP + TD + bell again beats DP (+30 and +53, the cabinet's +28 in section 3), but neither chain
  reaches the specialists (DP + TD + bell's two clutter seeds sit 9–13 below them, inside a chain's seed spread; DP's sit 43–46 below). The deficit is entirely on queries the straight line cannot solve: on straight-line-feasible
  queries the chains are slightly ahead (clutter 853–857 vs 835; platforms 1054–1055 vs 1047), on detour queries
  they are behind (398 / 391 / 426 vs 459; 439 / 449 / 491 vs 529). Every MPPI failure on a detour query is a
  collision, for every field.
- **On long queries the chains win as on the cabinet.** RK4 descent converges on 151 / 143 (clutter) and 142 / 139
  (platforms) of 400 for DP / DP + TD + bell, against 104 / 90 for the original method and 0 / 3 for the specialists;
  collision-free successes 66 / 75 and 63 / 89 vs 57 / 57. DP + TD + bell collides far less than DP (148 vs 198,
  112 vs 191), so it converts the most convergences into successes on both scenes. MPPI on the first 100 is level
  for every field (6–10 and 14–18).

**Why the specialists win the frozen suites (controls).** The truth fit says the opposite should happen: on the
scene's ground-truth pairs the DP models are the more accurate ones at mid range (clutter, segments 4–6: bias
1.20–1.23 vs 1.37–1.63, direction cosine 0.76–0.83 vs 0.34–0.77), and they tie at segment 3 (bias 1.22 / 1.22,
cosine 0.88 / 0.87) — also when the pairs are split by how blocked the straight line is. Three controls locate the
difference:

1. *Splicing* (`derived.py --splice`, the models of chosen segments taken from the other field, everything else
   unchanged): DP with the specialists' models 3–6 plans like the specialists or better (clutter 1306, platforms
   1587), the specialists with DP's models 3–6 like DP (1248). The segment-3 model alone carries two thirds of
   it (1288 / 1548); models 4–6 the rest (1262).

<!-- T6S splice -->
|Scene|Field|MPPI /1800|RK4 /600|
|---|---|---:|---:|
|clutter|DP with ES models 3–6|1306|448|
|clutter|ES with DP models 3–6|1248|447|
|clutter|DP with ES model 3|1288|437|
|clutter|DP with ES models 4–6|1262|436|
|platforms|DP with ES models 3–6|1587|524|
|platforms|DP with ES model 3|1548|521|
|cabinet|ES|688|226|
|cabinet|DP|706|241|
|cabinet|DP with ES models 3–6|695|226|
|cabinet|DP with ES model 3|716|238|

2. *The obstacle barrier.* MPPI's rollouts perturb the arm off the pool: of the configurations within 0.04 of the
   truth pairs' starts, 21–24 % sit at the speed network's floor (0.1, inside the 5 mm clearance band), against
   4–6 % of the pool. There the eikonal relation says |∇T| = 1/S — T must rise steeply towards the obstacle. The
   specialists keep it (S·|∇T| = 0.87–0.94 at the floor, 1.04–1.07 in free space); the DP models do not (0.43–0.51
   at the floor, 1.28–1.42 in free space), and DP + TD + bell is in between (0.56–0.59): its TD term uses the local
   speed. Relative to its own free-space slope the DP chain's barrier is 2.1–2.5× weaker (floor / free-space slope 0.33–0.40 vs 0.81–0.86). The DP models learn values at
   pool pairs; their slope near an obstacle is whatever the regression interpolates, whereas the eikonal loss sets the
   slope itself at every sampled state. The order of the barrier (ES > DP + TD + bell > DP) is the order of detour
   successes on both scenes.

<!-- T6B barrier -->
|Scene|Field|S·∣∇T∣ at the speed floor, segment 3 / 4|S·∣∇T∣ at S > 0.9, segment 3 / 4|floor / free ratio, segment 3|
|---|---|---:|---:|---:|
|clutter|ES|0.91 / 0.90|1.05 / 1.07|0.86|
|clutter|DP|0.48 / 0.44|1.34 / 1.42|0.36|
|clutter|DP+TD+BELL|0.59 / 0.56|1.37 / 1.31|0.43|
|platforms|ES|0.88 / 0.89|1.04 / 1.04|0.85|
|platforms|DP|0.51 / 0.47|1.28 / 1.36|0.40|
|platforms|DP+TD+BELL|0.58 / 0.58|1.32 / 1.26|0.44|
|cabinet|ES|0.87 / 0.94|1.07 / 1.05|0.81|
|cabinet|DP|0.45 / 0.43|1.36 / 1.42|0.33|

Perturbed states at the speed floor (segment 3 / 4): clutter 21 / 22 %; platforms 21 / 24 %; cabinet 18 / 20 %. Eikonal: S·∣∇T∣ = 1 everywhere.

3. *The same probe on the cabinet* gives the same barrier deficit (DP 0.43–0.45 vs ES 0.87–0.94), yet there DP
   beats the specialists on detour queries (197 vs 180 MPPI successes on the 398 queries × 3 seeds) and splicing the specialists' models 3–6
   into DP *costs* 11 (695 vs 706; segment 3 alone +10, within seed spread). The cabinet's detours are large — in
   and out of compartments, around boards — and are decided by the mid-range level, where the specialists'
   models 4–6 overestimate by 40–70 % (cabinet truth pairs, segments 4–5: bias 1.41–1.70 vs DP's 1.21–1.24); in clutter and between platforms
   the detours are local and the barrier decides.

**The splice is not a fix.** DP with the specialists' models 3–6 loses the long range: RK4 converges on 13 (clutter)
and 19 (platforms) of 400 long queries instead of 151 / 142. Along a ray from the goal, T must increase outwards for
a descent from far away to come in (`probes.py ridge`). Where a specialist meets the next model at gap 0.48, T falls
going outwards on 93–96 % of rays for the specialists alone and on 49–59 % for the splice, against 10–11 % for DP:
one ridge at the same gap on most rays, which the descent cannot cross — the specialists' T/gap peaks at 2.4 near
gap 0.36 and drops to 1.6 by 0.52, so the stitched field is lower beyond the seam than before it. That is also why the
specialists alone never converge on long queries. (The original method's T falls on 17–19 % of steps everywhere along
the ray, not at one gap, and its descents converge on 90–104.) What the chain lacks is the specialists' near-obstacle
slope, not their level: an eikonal term on off-pool, near-obstacle states in the DP levels (the TD term already moves
it a quarter of the way) would be the direct test.

<!-- T6R ridge -->
|Scene|Field|T falls outwards: steps at gap 0.32–0.64|at the 0.48 seam|RK4 converged /400 (long queries)|
|---|---|---:|---:|---:|
|clutter|B5|18 %|19 %|104|
|clutter|ES|38 %|96 %|0|
|clutter|DP|8 %|11 %|151|
|clutter|DP+TD+BELL|11 %|16 %|143|
|clutter|DP with ES models 3–6|20 %|49 %|13|
|platforms|B5|17 %|17 %|90|
|platforms|ES|34 %|93 %|3|
|platforms|DP|8 %|10 %|142|
|platforms|DP+TD+BELL|12 %|17 %|139|
|platforms|DP with ES models 3–6|18 %|59 %|19|

## 7. Cabinet only: training where the grasp point is inside a compartment

**Setup.** The same cabinet, speed network and frozen suites, but every training pair, DP split point and
eikonal sample is drawn from a pool restricted to configurations whose grasp point (end effector, forward
kinematics) lies inside one of the three compartments (`run_stage.py pool` → `pool_region.py`). The frozen test
queries are not used for training: the pool comes from the cabinet pipeline run's own pool (10,600 of its 399,581
poses, 2.7 %, are inside a compartment) plus the pipeline's coverage configurations whose grasp point is inside and
which pass the bank's state gate (clearance ≥ 5 mm, self ≥ 2 mm, joints within the normalized box). On this pool:
the 20 eikonal specialists (ES-cabonly), the DP chain on its specialists 1–2 (DP-cabonly, two training seeds) and
DP + TD + bell (DP+TD+BELL-cabonly, three training seeds), as full 20-segment chains with the recipe of section 6. The "cabinet only" test
is the two frozen strata whose start *and* goal have the grasp point inside a compartment —
compartment→compartment and same compartment, 200 queries × 3 planning seeds; the other strata show what training
only inside costs elsewhere. For reference the full-pool fields include a 20-segment DP + TD + bell chain on the
cabinet (section 3's was 9 segments).

<!-- T7P pool -->
|Source|kept|of|bottom / middle / top|
|---|---:|---:|---|
|full pool (pipeline run's prepared/bank)|10,600|399,581|2,068 / 3,755 / 4,777|
|coverage configurations, grasp in a compartment and state gate passed|211,375|2,000,010|50,635 / 105,942 / 132,251 (before the gate)|
|**cabinet-only pool**|**221,975**|||

<!-- T7F frozen -->
|Field|training pool|MPPI /1800|RK4 /600|free→free|free→comp.|comp.→free|comp.→comp.|same comp.|anywhere|**in-cabinet MPPI /600**|in-cabinet RK4 /200|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|B5|full|688|229|268|66|63|3|24|264|**27**|8|
|ES|full|688|226|275|77|54|11|8|263|**19**|6|
|DP|full|706|241|281|74|72|16|5|258|**21**|9|
|DP+TD+BELL (9 segments, §3)|full|742|244|282|77|85|19|12|267|**31**|10|
|DP+TD+BELL|full|717|241|282|76|76|12|7|264|**19**|7|
|ES-cabonly|cabinet only|423|24|174|49|17|8|21|154|**29**|8|
|DP-cabonly|cabinet only|496|61|206|33|22|3|34|198|**37**|9|
|DP-cabonly (seed 1)|cabinet only|483|64|201|42|27|3|22|188|**25**|7|
|DP+TD+BELL-cabonly|cabinet only|513|73|201|36|28|4|36|208|**40**|7|
|DP+TD+BELL-cabonly (seed 1)|cabinet only|527|79|199|44|34|3|38|209|**41**|9|
|DP+TD+BELL-cabonly (seed 2)|cabinet only|513|80|205|41|29|3|33|202|**36**|9|
Strata MPPI out of 300 (100 queries × planning seeds 0 / 1 / 2). In-cabinet = compartment→compartment + same compartment, the two strata whose start and goal both have the grasp point inside a compartment.

- **Inside the compartments: better than every full-pool field, including the original method.** On the 200
  in-cabinet queries DP + TD + bell cabinet-only succeeds 40 / 41 / 36 of 600 times on three training seeds (mean
  39), against 27 for the original method and 19–31 for the full-pool chains and specialists. The gain is on
  **same-compartment** queries: 36 / 38 / 33 of 300 against 24 for the original method and 5–12 for the full-pool
  chains — the worst seed is 9 above the original method, and the seed range (5) is smaller than that margin. DP
  cabinet-only is less stable: 37 and 25 in-cabinet (same compartment 34 and 22), so its second seed falls below the
  original method; the TD and bell terms are what make the cabinet-only chain reliable here, as they did on the full
  cabinet (section 3). The gain is an MPPI gain: plain RK4 descent succeeds on 7 / 9 / 9 of 200 in-cabinet queries,
  level with the original method's 8. Caveat: the original method is one training run, so its own seed spread on
  these 600 runs is not measured. The specialists trained on the cabinet-only pool reach 21 same-compartment
  successes (8 on the full pool).
- **Between compartments it does not help** — compartment→compartment drops to 3–4 of 300 from the full-pool chains'
  12–19 (the original method: 3) — **and outside it collapses**: free→free 199–206 vs 268–282, 483–527 of
  1800 overall vs 688–742, RK4 61–80 of 600 vs 226–244. A field trained only on in-compartment configurations has no
  values for the rest of the arm's workspace.
- **Long queries** (training seed 0 of each cabinet-only field): the cabinet-only fields succeed on 1 of the 400 at most (RK4 converges on 30 for DP + TD + bell
  cabinet-only, 11 for DP, none for the specialists; full-pool DP 138 / 29 successes). Of the 105 long queries with both ends
  in a compartment no field of either pool succeeds (converged: full-pool DP 34, DP + TD + bell 29, the original method 19,
  the cabinet-only chains 14 / 5). Long in-cabinet queries are compartment→compartment queries.

<!-- T7L long -->
|Field|RK4 success /400|RK4 converged /400|both ends in a compartment, success / converged /105|one end in /202|neither /93|RK4 success, first 100|MPPI seed 0, first 100|
|---|---:|---:|---:|---:|---:|---:|---:|
|B5|23|89|0 / 19|5 / 43|18 / 27|7|9|
|ES|0|0|0 / 0|0 / 0|0 / 0|0|4|
|DP|29|138|0 / 34|6 / 71|23 / 33|10|9|
|DP+TD+BELL|29|129|0 / 29|5 / 66|24 / 34|8|8|
|ES-cabonly|0|0|0 / 0|0 / 0|0 / 0|0|1|
|DP-cabonly|0|11|0 / 5|0 / 5|0 / 1|0|0|
|DP+TD+BELL-cabonly|1|30|0 / 14|0 / 14|1 / 2|0|2|
Long queries: the 400 uncertified pairs (gap 0.8–1.6) of sections 2–4; ends located by forward kinematics of the grasp point against the compartment boxes of cabinet.yml.

**Why same-compartment improves and compartment→compartment does not (control).** Where do the planned paths go?
For every MPPI path on the in-cabinet strata (3 planning seeds), the share of waypoints whose grasp point is inside a
compartment (`cabroute.py`, forward kinematics against the compartment boxes):

|stratum / field|successes /300|waypoints inside, successful paths|waypoints inside, failed paths|
|---|---:|---:|---:|
|same compartment, B5|24|60 %|44 %|
|same compartment, DP (full pool)|5|62 %|18 %|
|same compartment, DP + TD + bell (full pool)|7|48 %|18 %|
|same compartment, DP cabinet-only|34|85 %|48 %|
|same compartment, DP + TD + bell cabinet-only|36|85 %|51 %|
|compartment→compartment, DP (full pool)|16|20 %|14 %|
|compartment→compartment, DP + TD + bell (full pool)|12|28 %|13 %|
|compartment→compartment, DP cabinet-only|3|23 %|46 %|
|compartment→compartment, DP + TD + bell cabinet-only|4|21 %|46 %|

- *Same compartment:* the successful paths stay inside (48–85 % of their waypoints), the failed ones leave. The
  full-pool chains send the arm out of the compartment — 82 % of the waypoints of their failed paths are outside —
  and collide with the boards on the way: in the full pool only 2.7 % of the poses are inside a compartment, so the
  in-compartment values are fitted on few pairs and the DP minimum over split points prefers detours through the
  well-sampled space outside. The cabinet-only chains have only inside split points and inside pairs; their paths
  keep 52–55 % of all their waypoints inside (full-pool chains 19 %) and succeed five to seven times as often.
- *Compartment→compartment:* a successful path has to leave one compartment and enter another — 72–80 % of the
  chains' successful waypoints are outside. The cabinet-only pool has no configurations there, so the DP chain cannot place a split
  point on that route and its values between compartments are extrapolation; its failed paths stay inside (46 %,
  against 13–14 % for the full-pool chains) instead of leaving.

**Reading.** Restricting the pool trades the whole workspace for density where the queries are: same-compartment
queries improve because the in-compartment values are now trained on 222k inside configurations instead of ~11k,
not because the DP changed. With DP + TD + bell the improvement also clears the original method, which plans
same-compartment queries well from the full pool, on all three training seeds (in-cabinet 36–41 vs 27); with DP
alone it does on one seed of two. The route control says what a pool for cabinet planning needs: the
compartments densely (cabinet-only) *and* the space in front of them that every compartment→compartment path crosses
— a cabinet-weighted pool (compartments plus the region outside their openings, on top of the full pool) is the
direct next test.

## 8. Segment DP on a correct local field

*2026-10-31, 08:00–12:30 UTC, RTX 3090 and RTX 5070 (the 3090 shared with another session for the first hour).*

**Question.** The DP chain of sections 2b–7 starts from two eikonal specialists (ES, segments 1–2, gap ≤ 0.16) and
builds segments 3–20 by T_(m+1)(i, j) = min_k T_≤m(i, k) + T_≤m(k, j). The eikonal specialists are 9 % off the
ground truth at short range. If the local field were correct, would the DP be good at long range? Compared on the
long-range tasks (400 long cabinet queries, fit to the ground truth at gap 0.72–1.6) against the chain on the learned
eikonal specialists.

**A correct local travel time.** `local_truth.py` computes T*(q, g) = min over paths of ∫ ds / S for short pool pairs
the way `truth.py` builds the ground truth: a polyline from the straight line (8 waypoints, 100 Adam steps, the
speed network under bf16 while optimizing, cost re-measured densely in float32), the cheaper of that path and the
straight line; and its gradients by the envelope theorem. Against `truth.py`'s T* on the probe pairs: mean |log| 0.005
(segment 1) and 0.017 (segment 2), S·|∇T*| = 1.000; the eikonal specialists are at 0.08–0.09. Labels: 400k pairs
per segment, drawn by `PairSampler` exactly as the specialists' pairs (runs `c6afd62b`, `063ecaf6`).

It costs 3.5 ms per pair on the 3090 (the speed network is a 6 × 512 residual MLP), so it cannot replace the base
inside the chain: level 3 alone queries ~28M short pieces (~27 GPU-hours), later levels more. Two uses instead:
- **Exact pieces** (`dp_exact.py`, no network): the DP targets of levels 3–4 on the truth probe pairs with every
  piece path-optimized — what a perfect base gives.
- **SL, supervised specialists 1–2** (`supervised.py`): the same network, segments, optimizer and 15,620 updates as
  the specialists, regressed on the labels ((log T/T*)² + a gradient term |S(∇T − ∇T*)|², which pins norm and
  direction). This is the most correct local field the chain can be trained on: truth error 0.073 vs 0.089 at
  gap 0–0.16 (bias 1.016 vs 1.026), direction cosine 0.961 vs 0.932. The network caps it, not the labels: even on
  pairs whose optimal path is within 2 % of straight it is 4.7 % off (ES 5.8 %). Held-out label error on segment 2
  is 0.077 at the protocol's size and updates, 0.065 with 4× the updates, 0.063 with an 8× wider network and 0.046
  with both (dev runs, not used: an architecture change); a hybrid T = straight-line integral × learned bend factor
  did not fit (0.24).

The chain on SL (**DP-SL**) is `dp3.py` unchanged (its defaults are `dp.py`): 18 levels, 400k pairs and 384 pool
split candidates per level, 15,620 updates per model, three training seeds (87358 / 1 / 2). Reference: the chain on
the eikonal specialists (**DP-ES**), seed 87358 (`f57e1332`, section 2b) and a new seed-1 run of the full chain.

**Local fields** (fit to the ground truth, cabinet probe pairs; ES12 / SL12 = specialists 1–2, model 2 answering every
longer gap):

<!-- T8A local fields -->
|Field|Rel. error 0–0.08|0.08–0.16|Bias 0–0.08|0.08–0.16|Spearman 0–0.16|Cosine 0–0.16|
|---|---:|---:|---:|---:|---:|---:|
|ES12|0.095|0.083|1.018|1.034|0.942|0.932|
|SL12|0.082|0.064|1.008|1.023|0.953|0.961|

**DP targets from exact pieces vs the two bases** (`dp_exact.py`; median and spread of target / T*, the winning split's
detour (|i−k| + |k−j|) / |i−j|, and how often a base picks the same split as the exact pieces):

<!-- T8B exact pieces -->
|Level|Pieces|Pairs|Median target / T*|Mean abs log|p90|Winning detour|Same split as exact|
|---|---:|---:|---:|---:|---:|---:|---:|
|3|exact (path-optimized)|1024|1.146|0.160|1.322|1.185|—|
|3|SL base|1024|1.164|0.171|1.356|1.175|0.837|
|3|ES base|1024|1.179|0.197|1.382|1.177|0.789|
|4|exact (path-optimized)|256|1.074|0.088|1.163|1.105|—|

**Per-level DP target / T*** (median over that level's probe pairs, 384 candidates; column = lower edge of the segment)
and the trained model:

<!-- T8C per-level targets -->
|Chain|0.16|0.24|0.32|0.40|0.48|0.56|0.64|0.72|0.80|0.88|0.96|1.04|1.12|1.20|1.28|1.36|1.44|1.52|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|DP-ES s0 (levels 3-9: DP3)|1.18|1.21|1.20|1.20|1.17|1.16|1.12|—|—|—|—|—|—|—|—|—|—|—|
|DP-ES s1|1.18|1.21|1.20|1.19|1.16|1.17|1.12|1.09|1.08|1.06|1.07|1.07|1.07|1.06|1.06|1.06|1.05|1.04|
|DP-SL s0|1.16|1.21|1.23|1.23|1.25|1.26|1.27|1.28|1.28|1.30|1.29|1.30|1.30|1.30|1.30|1.30|1.31|1.30|
|DP-SL s1|1.16|1.19|1.21|1.22|1.24|1.25|1.27|1.28|1.28|1.30|1.29|1.29|1.30|1.31|1.31|1.31|1.32|1.31|
|DP-SL s2|1.16|1.20|1.22|1.23|1.25|1.26|1.27|1.29|1.29|1.31|1.31|1.31|1.32|1.34|1.33|1.34|1.34|1.34|

Trained model, median T / T* on the same probe pairs:

|Chain|0.16|0.24|0.32|0.40|0.48|0.56|0.64|0.72|0.80|0.88|0.96|1.04|1.12|1.20|1.28|1.36|1.44|1.52|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|DP-ES s0 (levels 3-9: DP3)|1.20|1.21|1.22|1.20|1.16|1.16|1.12|—|—|—|—|—|—|—|—|—|—|—|
|DP-ES s1|1.19|1.21|1.20|1.19|1.17|1.17|1.13|1.10|1.08|1.06|1.07|1.07|1.07|1.06|1.05|1.05|1.05|1.04|
|DP-SL s0|1.20|1.23|1.24|1.25|1.25|1.26|1.27|1.29|1.29|1.30|1.29|1.30|1.30|1.30|1.30|1.30|1.31|1.30|
|DP-SL s1|1.18|1.21|1.21|1.23|1.25|1.25|1.27|1.28|1.28|1.30|1.29|1.30|1.30|1.30|1.31|1.31|1.32|1.32|
|DP-SL s2|1.18|1.22|1.22|1.24|1.25|1.26|1.28|1.29|1.30|1.32|1.31|1.32|1.32|1.33|1.33|1.34|1.35|1.34|

**Why the levels drift apart** (`dp_select.py`, 256 probe pairs per level; medians; teacher bias = the chain's
composite of models 1..m−1 over the path-optimized cost of the same two pieces; selection = chosen / line-nearest):

<!-- T8G selection decomposition -->
|Level|Chain|Target / T*|Teacher bias at chosen split|at line-nearest split|Selection|True overhead, chosen split|line-nearest split|
|---|---:|---:|---:|---:|---:|---:|---:|
|5|DP-ESs1|1.214|1.125|1.150|0.978|1.068|1.079|
|5|DP-SL|1.228|1.129|1.156|0.977|1.065|1.079|
|7|DP-ESs1|1.159|1.112|1.084|1.026|1.056|1.108|
|7|DP-SL|1.250|1.157|1.144|1.012|1.055|1.108|
|9|DP-ESs1|1.162|1.008|1.008|1.000|1.051|1.130|
|9|DP-SL|1.258|1.165|1.139|1.023|1.049|1.130|
|11|DP-ESs1|1.091|0.852|0.854|0.998|1.110|1.242|
|11|DP-SL|1.277|1.086|1.096|0.991|1.107|1.242|

**Fit of the final fields by gap range** (DP-SL+ES12 / DP-ES+SL12 = the splices: one chain's models 3–20 with the
other base's specialists 1–2):

<!-- T8D fit by range -->
**Relative error**

|Field|0–0.16|0.16–0.4|0.4–0.72|0.72–1.6|
|---|---:|---:|---:|---:|
|B5|0.294|0.215|0.360|0.378|
|ES12|0.089|0.155|0.321|0.320|
|SL12|0.073|0.117|0.307|0.306|
|DP-ES|0.095|0.236|0.227|0.140|
|DP-ESs1|0.095|0.228|0.230|0.139|
|DP-SL|0.079|0.203|0.224|0.257|
|DP-SLs1|0.079|0.188|0.222|0.260|
|DP-SLs2|0.079|0.193|0.233|0.279|
|DP-SL+ES12|0.093|0.203|0.224|0.257|
|DP-ES+SL12|0.081|0.236|0.227|0.140|

**Bias**

|Field|0–0.16|0.16–0.4|0.4–0.72|0.72–1.6|
|---|---:|---:|---:|---:|
|B5|0.819|0.909|0.806|0.671|
|ES12|1.026|1.013|0.900|0.737|
|SL12|1.016|1.019|0.928|0.774|
|DP-ES|1.035|1.215|1.173|1.077|
|DP-ESs1|1.035|1.201|1.165|1.063|
|DP-SL|1.022|1.220|1.257|1.299|
|DP-SLs1|1.022|1.201|1.250|1.301|
|DP-SLs2|1.022|1.206|1.257|1.322|
|DP-SL+ES12|1.035|1.220|1.257|1.299|
|DP-ES+SL12|1.022|1.215|1.173|1.077|

**Spearman**

|Field|0–0.16|0.16–0.4|0.4–0.72|0.72–1.6|
|---|---:|---:|---:|---:|
|B5|0.695|0.654|0.263|0.130|
|ES12|0.942|0.796|0.332|0.155|
|SL12|0.953|0.882|0.389|0.203|
|DP-ES|0.936|0.874|0.841|0.753|
|DP-ESs1|0.936|0.865|0.818|0.733|
|DP-SL|0.944|0.893|0.928|0.896|
|DP-SLs1|0.946|0.895|0.929|0.890|
|DP-SLs2|0.946|0.890|0.928|0.894|
|DP-SL+ES12|0.938|0.894|0.928|0.896|
|DP-ES+SL12|0.943|0.873|0.841|0.753|

**Direction cosine**

|Field|0–0.16|0.16–0.4|0.4–0.72|0.72–1.6|
|---|---:|---:|---:|---:|
|B5|0.934|0.836|0.710|0.689|
|ES12|0.932|0.841|0.681|0.546|
|SL12|0.961|0.855|0.679|0.535|
|DP-ES|0.935|0.834|0.719|0.686|
|DP-ESs1|0.934|0.837|0.726|0.701|
|DP-SL|0.961|0.851|0.747|0.711|
|DP-SLs1|0.961|0.848|0.750|0.708|
|DP-SLs2|0.960|0.852|0.752|0.715|
|DP-SL+ES12|0.935|0.849|0.747|0.711|
|DP-ES+SL12|0.955|0.835|0.719|0.686|

**Monotonicity along rays from the goal** (`probes.py ridge`, 2,000 rays from the goals of the long probe pairs;
`probes_section8_ridge.json`):

<!-- T8H ridge probe -->
|Field|T falls outwards: gap 0.04–0.32|0.32–0.64|0.64–0.96|Monotone rays|
|---|---:|---:|---:|---:|
|B5|4.9 %|20.8 %|20.3 %|6.6 %|
|ES12|9.8 %|22.6 %|26.0 %|3.9 %|
|SL12|5.5 %|21.0 %|22.9 %|6.6 %|
|DP-ES|4.9 %|7.9 %|7.2 %|19.4 %|
|DP-ESs1|5.2 %|8.7 %|7.9 %|17.1 %|
|DP-SL|1.5 %|7.0 %|4.9 %|34.7 %|
|DP-SLs1|2.0 %|5.2 %|4.8 %|35.4 %|
|DP-SLs2|1.6 %|6.2 %|5.0 %|36.0 %|
|DP-SL+ES12|2.8 %|7.0 %|4.9 %|33.1 %|
|DP-ES+SL12|3.8 %|7.9 %|7.2 %|20.2 %|

**Long queries** (the 400 uncertified cabinet queries of sections 2b–4, gap 0.8–1.5; RK4 "converged" = reaches the
goal, collisions ignored; MPPI planning seed 0 on the first 100; DP-SLs2 has no MPPI run):

<!-- T8E long queries -->
|Field|RK4 success /400|RK4 converged /400|Never reaches goal|Collides|RK4 success, first 100|MPPI seed 0, first 100|
|---|---:|---:|---:|---:|---:|---:|
|B5|23|89|311|249|7|9|
|ES12|20|67|333|224|7|8|
|SL12|26|82|318|239|10|9|
|DP-ES|29|138|262|288|10|9|
|DP-ESs1|28|135|265|282|10|10|
|DP-SL|31|322|78|345|12|9|
|DP-SLs1|28|299|101|334|10|8|
|DP-SLs2|30|330|70|348|10|—|
|DP-SL+ES12|28|305|95|347|9|7|
|DP-ES+SL12|31|141|259|287|12|10|

**Frozen cabinet benchmarks** (600 queries; MPPI × planning seeds 0/1/2):

<!-- T8F frozen benchmarks -->
|Field|RK4 /600|MPPI /1800 (planning seeds)|
|---|---:|---:|
|B5|229|688 (230/229/229)|
|ES12|239|716 (239/238/239)|
|SL12|273|760 (254/254/252)|
|DP-ES|241|706 (237/237/232)|
|DP-ESs1|240|709 (237/238/234)|
|DP-SL|248|716 (238/239/239)|
|DP-SLs1|258|708 (232/237/239)|
|DP-SLs2|248|706 (234/237/235)|
|DP-SL+ES12|225|681 (225/228/228)|
|DP-ES+SL12|268|740 (246/247/247)|

### Findings

1. **A correct base barely changes the DP targets.** At level 3 (gap 0.16–0.24) the target from exact,
   path-optimized pieces is still 1.146 × the truth; SL pieces give 1.164, the eikonal pieces 1.179 (same 1,024
   probe pairs, same split candidates). The base costs 3 points; the pool's split points cost 15 (the best pool split
   detours by 18 %: the lens of a 0.2 gap holds a median of 30 poses). At level 4, exact pieces up to 0.256 give 1.074.
2. **With an accurate base the overestimate compounds; the eikonal chain was right at long range by cancellation.**
   Per-level DP target / T*: DP-SL climbs 1.16 → 1.21 → 1.23 → … → 1.30–1.34 at gap 1.5, identically on three
   seeds; DP-ES rises to 1.21 and then falls to 1.04 (both seeds within 0.01). The decomposition at levels 5–11
   (`dp_select.py`, 256 probe pairs each): the min does not pick under-read pieces in either chain (teacher bias at
   the chosen split / at the split nearest the straight line = 0.98–1.03), and the true via-split overhead is the
   pool's and identical in both (1.05–1.11). What differs is the teacher's level on the *same* pieces: from level 7 on
   the eikonal chain's mid-range models read 5–22 % lower (level 11: 0.85 vs 1.09 of the path-optimized pieces). Each
   chain's models reproduce their own DP targets on those pieces within 1 % (dev check), so the drift sits in the
   targets and is inherited level by level from the eikonal base. DP-SL is the textbook case — every level
   multiplies the pool overhead — and the eikonal chain's long-range accuracy (relative error 0.14, bias 1.06–1.08 at
   0.72–1.6) is a downward drift that happens to cancel it.
3. **The long-range shape gets much better.** At 0.72–1.6 DP-SL ranks which pairs are slow far better (Spearman
   0.89–0.90 vs 0.73–0.75) and points slightly better (cosine 0.71 vs 0.69–0.70), at a 30 % higher level. Along rays
   from the goal, T falls going outwards on 1.5–2.0 % of steps near the goal (0.04–0.32) vs 4.9–5.2 %, and 35–36 % of
   rays are monotone out to 0.96 vs 17–19 %: fewer dips for a descent to stop in.
4. **Long queries: gradient descent reaches the goal 2.3× as often.** RK4 converges on 322 / 299 / 330 of the 400
   long queries (three seeds) vs 138 / 135 for DP-ES and 89 for the original method; "never reaches the goal" falls
   from 262–265 to 70–101. A uniform overestimate does not bend the descent; the dips do.
5. **Collision-free success stays at the collision ceiling.** RK4 successes 31 / 28 / 30 vs 29 / 28 (original method
   23): the extra arrivals collide (collisions 334–348 vs 282–288). MPPI on the first 100 long queries is 9 / 8 vs
   9 / 10 — every field 7–10, as in sections 2b and 4: the long-query failures are collisions, which neither the level
   nor the shape of T fixes.
6. **Splices: the gain lives in the DP levels trained on the correct base, not in the local field used for
   planning.** DP-SL's models 3–20 with the eikonal specialists as segments 1–2 still converge 305 / 400 (monotone rays
   33 %); DP-ES's models with the SL specialists 141 (20 %). The local field matters through what the DP builds from it.
7. **Short (frozen) benchmarks: the correct local field helps wherever it is used, the DP levels above it cost.**
   SL12 alone (SL model 2 answering every longer gap) is the best field of this study: RK4 273 / 600, MPPI 760 / 1800
   (254 / 254 / 252) vs ES12 239 / 716, DP + TD + bell 742 and the original method 688–698. Putting the DP levels on top
   gives DP-SL 248–258 / 706–716 (DP-ES 240–241 / 706–709); DP-ES with the SL specialists 268 / 740, DP-SL with the
   eikonal specialists 225 / 681 (the two halves no longer meet at 0.16). The DP levels' ~20 % overestimate at
   0.16–0.4, where MPPI's rollouts work, costs 45–55 MPPI successes against SL12.

**Answer.** With a correct local field the segment DP's *values* do not become correct: the pool's split points add
15 % at the first DP level and the recursion multiplies that, so the long range ends 30 % high; the eikonal chain only
looked accurate because its error drifted the other way. Its *shape* does become right — rank correlation at long
range 0.90, a monotone field on twice as many rays, and gradient descent that reaches the goal on ~80 % of long
queries instead of ~34 % — robustly over three seeds and carried by the DP levels themselves. The long-query
success rate does not follow, because those failures are collisions. Next steps this points to: remove the
per-level overhead (denser split points at short range, or dividing each level's targets by its measured exact-piece
overhead) to get the level right as well, and a collision-aware direction for the long-query successes.

## Caveats

- One training seed for most part-2 fields; training-seed spread of comparable fields is ±5–10 MPPI on the
  cabinet and ±10 on no-wall (baseline 688/697/698 and 1598/1586/1585). The two-specialist result has 3 seeds per scene.
- The ground truth is an upper bound (see Protocol); it is common to all fields and used only to compare them.
- Section 8's path-optimized pieces start from the straight line: accurate in the median at every length used (median
  ratio to truth.py's T* 1.000–1.003), with a tail where a longer piece misses a homotopy, so `dp_select.py` reports medians.
- The long queries are uncertified; DP-c and DP use 150k / 400k pairs per segment (DP-c's split points cost 4×).
- The specialists run was restarted once after a sampler bug (a pose paired with itself, 3.5 % of segment-1 pairs,
  zero gradient); every reported run uses the fixed sampler. Runs stopped for that or for the routing speed-up are
  marked failed in the registry and listed below.

## Run index

|Scene/label|Run|
|---|---|
|cabinet/B5-off|`20261027T224340Z-8468d15a`|
|cabinet/DP|`20261027T184605Z-f57e1332`|
|cabinet/DP-off|`20261027T205902Z-79094a5f`|
|cabinet/DPc|`20261027T190444Z-a00736c2`|
|cabinet/EG|`20261027T201137Z-f2fef22a`|
|cabinet/EG-off|`20261027T223726Z-7ce17ad8`|
|cabinet/EG16|`20261027T230747Z-7999899d`|
|cabinet/ES|`20261027T175328Z-b7ab5a49`|
|cabinet/ES-off|`20261027T202012Z-9378bc06`|
|cabinet/ES2s1|`20261027T224955Z-597239ba`|
|cabinet/ES2s2|`20261027T225708Z-c250004b`|
|cabinet/derived-ES1|`20261027T230425Z-a227cbc2`|
|cabinet/derived-ES2|`20261027T184946Z-75cbbb90`|
|cabinet/fit-a|`20261027T231356Z-2ca76b78`|
|cabinet/fit-b|`20261028T003021Z-b0ca8707`|
|cabinet/long-mppi-100-a|`20261027T232223Z-5e5441a4`|
|cabinet/long-mppi-100-b|`20261027T232818Z-8c808076`|
|cabinet/long-mppi-100-c|`20261027T232444Z-b23ecf71`|
|cabinet/long-rk4-400-a|`20261027T231630Z-80084e96`|
|cabinet/long-rk4-400-b|`20261027T231402Z-bf88dd59`|
|cabinet/truth|`20261027T175501Z-5150664b`|
|cabinet/w0-2-0|`20261027T174231Z-633eb44d`|
|cabinet/w0-2-2|`20261027T172500Z-a3e0297d`|
|cabinet/w0-2-2s1|`20261027T221201Z-b687c4ce`|
|cabinet/w0-2-2s2|`20261027T221827Z-9a0b13df`|
|cabinet/w6-0-0|`20261027T222448Z-433e1689`|
|cabinet/w6-0-2|`20261027T171214Z-189e5ae4`|
|cabinet/w6-0-2s1|`20261027T215932Z-19a7bab5`|
|cabinet/w6-0-2s2|`20261027T220548Z-96b4f8a1`|
|cabinet/w6-2-0|`20261027T223057Z-0f1ddc73`|
|cabinet/w6-2-2|`20261027T044847Z-c42c6e8f`|
|cabinet/w6-2-2s1|`20261027T182552Z-1e04efd5`|
|cabinet/w6-2-2s2|`20261027T183810Z-3b2e6860`|
|nowall/B5-off|`20261027T210115Z-43823be1`|
|nowall/DP|`20261027T190922Z-4d3efe16`|
|nowall/EG|`20261027T185505Z-56d89309`|
|nowall/EG-off|`20261027T205024Z-3037b3f8`|
|nowall/EG16|`20261027T231531Z-bda27d55`|
|nowall/ES|`20261027T183924Z-aa1a5943`|
|nowall/ES-off|`20261027T203416Z-f1a59da4`|
|nowall/ES2s1|`20261027T214421Z-8b54a196`|
|nowall/ES2s2|`20261027T215150Z-63e41d43`|
|nowall/derived-ES2|`20261027T210721Z-a4e00145`|
|nowall/fit-a|`20261027T220543Z-36b16b72`|
|nowall/fit-b|`20261027T235758Z-a8dfdcb9`|
|nowall/truth|`20261027T181431Z-bf1c963f`|
|nowall/w0-2-0|`20261027T175926Z-878173c1`|
|nowall/w0-2-2|`20261027T173508Z-024b8dbe`|
|nowall/w0-2-2s1|`20261027T213102Z-d63f12f9`|
|nowall/w0-2-2s2|`20261027T213744Z-f3d9f907`|
|nowall/w6-0-0|`20261027T182552Z-a3255db9`|
|nowall/w6-0-2|`20261027T171214Z-10b82c30`|
|nowall/w6-0-2s1|`20261027T211849Z-1e996798`|
|nowall/w6-0-2s2|`20261027T212451Z-4e5f5883`|
|nowall/w6-2-0|`20261027T183210Z-0be7e8eb`|
|nowall/w6-2-2|`20261027T045906Z-45797805`|
|nowall/w6-2-2s1|`20261027T183907Z-1e328ab5`|
|nowall/w6-2-2s2|`20261027T211213Z-13c28f64`|
|cabinet/DP3|`20261029T040917Z-466ee170`|
|cabinet/DP3s1|`20261029T151513Z-76ae7fe8`|
|cabinet/DP3s2|`20261029T141519Z-68a45bd7`|
|cabinet/DP3-l1|`20261029T121552Z-79d7b1db`|
|cabinet/DP3-l1s1|`20261029T155352Z-2cb4180c`|
|cabinet/DP3-l1s2|`20261029T170548Z-dc768c66`|
|cabinet/DP3-l2|`20261029T040917Z-63632a68`|
|cabinet/DP3-logl1|`20261029T081844Z-e1999f92`|
|cabinet/DP3-logl2|`20261029T070358Z-cdff6a7c`|
|cabinet/DP3-rel|`20261029T125258Z-fef80907`|
|cabinet/DP3-add|`20261029T085540Z-4eee9240`|
|cabinet/DP3-exp|`20261029T093223Z-c795ff72`|
|cabinet/DP3-c32|`20261029T100910Z-26fa3fda`|
|cabinet/DP3-c1536|`20261029T095517Z-0d231d9d`|
|cabinet/TD|`20261029T053711Z-dcd15e08`|
|cabinet/BELL|`20261029T055713Z-8eee09d0`|
|cabinet/TD+BELL|`20261029T075313Z-96be0690`|
|cabinet/DP+TD|`20261029T062014Z-bdcef5b1`|
|cabinet/DP+BELL|`20261029T070509Z-c63f2bdd`|
|cabinet/DP+TD+BELL|`20261029T044637Z-672a0731`|
|cabinet/DP+TD+BELLs1|`20261029T163228Z-4c0c1ec5`|
|cabinet/DP+TD+BELLs2|`20261029T172604Z-31dc56c0`|
|cabinet/fit (ES, DP3, DP3-l1, DP3-rel; 1,024 truth pairs per range)|`20261029T133000Z-f4f8f613`|
|cabinet/fit (ES + 14 section-3 chains)|`20261029T141304Z-89dc7552`|
|cabinet_plus10/ES|`20261029T103103Z-ff71f729`|
|cabinet_plus10/DP|`20261029T104520Z-05aa3831`|
|cabinet_plus10/long-rk4-400|`20261029T120857Z-14223f51`|
|cabinet_plus10/long-mppi-100|`20261029T121020Z-fa8b9c92`|
|cabinet_plus20/ES|`20261029T133004Z-efaec808`|
|cabinet_plus20/DP|`20261029T134245Z-d57b7484`|
|cabinet_plus20/long-rk4-400|`20261029T150812Z-07fee55d`|
|cabinet_plus20/long-mppi-100|`20261029T150941Z-b14ca3f1`|
|cabinet/DP3-warm|`20261029T214026Z-4d9e41cd`|
|cabinet/DP3-warms1|`20261029T221438Z-4a76c7e4`|
|cabinet/DP3-warms2|`20261029T224748Z-61730b25`|
|cabinet_clutter/tests (frozen by freeze_tests.py into tests_clutter/)|`20261029T223856Z-1fb06127`|
|cabinet_platforms/tests (tests_platforms/)|`20261029T224139Z-6888b674`|
|cabinet_clutter/pipeline (prepared pool, speed, B5)|`20261029T224420Z-46d34528`|
|cabinet_clutter/ES|`20261029T225548Z-62fd51b9`|
|cabinet_clutter/DP|`20261029T233126Z-a3e83ec0`|
|cabinet_clutter/DPs1|`20261030T014126Z-7b6c0ad0`|
|cabinet_clutter/DP+TD+BELL|`20261029T232202Z-4cac6e35`|
|cabinet_clutter/DP+TD+BELLs1|`20261030T034348Z-95263b93`|
|cabinet_clutter/truth|`20261030T031830Z-175e0555`|
|cabinet_clutter/fit|`20261030T034324Z-32da5bfe`|
|cabinet_clutter/DP with ES 3–6 (splice)|`20261030T041013Z-51b3d4a2`|
|cabinet_clutter/ES with DP 3–6 (splice)|`20261030T041656Z-a722747d`|
|cabinet_clutter/DP with ES 3 (splice)|`20261030T042144Z-bd26b64d`|
|cabinet_clutter/DP with ES 4–6 (splice)|`20261030T042710Z-cb5880ee`|
|cabinet_clutter/long-rk4-400|`20261030T013213Z-c446cb9c`|
|cabinet_clutter/long-mppi-100|`20261030T013408Z-14a41319`|
|cabinet_clutter/long-rk4-400 (splice)|`20261030T045932Z-3ba29134`|
|cabinet_clutter/long-mppi-100 (splice)|`20261030T050013Z-0a1fe18a`|
|cabinet_platforms/pipeline (prepared pool, speed, B5)|`20261029T230806Z-9847a6d1`|
|cabinet_platforms/ES|`20261029T231936Z-de4d4ef9`|
|cabinet_platforms/DP|`20261030T003155Z-5f1386e2`|
|cabinet_platforms/DPs1|`20261030T024214Z-ea398794`|
|cabinet_platforms/DP+TD+BELL|`20261030T011331Z-43f0aca7`|
|cabinet_platforms/truth|`20261030T034328Z-233a9d43`|
|cabinet_platforms/fit|`20261030T040813Z-e93d311b`|
|cabinet_platforms/DP with ES 3–6 (splice)|`20261030T044818Z-4c60a5d9`|
|cabinet_platforms/DP with ES 3 (splice)|`20261030T045417Z-6f0d3ec1`|
|cabinet_platforms/long-rk4-400|`20261030T030825Z-c6b7a86d`|
|cabinet_platforms/long-mppi-100|`20261030T031039Z-d55cd075`|
|cabinet_platforms/long-rk4-400 (splice)|`20261030T050231Z-547f79c1`|
|cabinet_platforms/long-mppi-100 (splice)|`20261030T050313Z-bd7739f3`|
|cabinet/DP with ES 3–6 (splice)|`20261030T043635Z-2bd5d23c`|
|cabinet/DP with ES 3 (splice)|`20261030T044251Z-287c0a05`|
|cabinet/DP+TD+BELL, 20 segments|`20261030T051716Z-502d7e3a`|
|cabinet/cabinet-only pool (stage pool)|`20261030T050941Z-f1796b59`|
|cabinet/ES-cabonly|`20261030T050944Z-4dc9e58c`|
|cabinet/DP-cabonly|`20261030T053951Z-b3a145d0`|
|cabinet/long-rk4-400 (B5, ES, DP, DP+TD+BELL, ES-cabonly, DP-cabonly)|`20261030T065351Z-87bf2327`|
|cabinet/long-mppi-100 (same fields)|`20261030T065714Z-61d3599d`|
|cabinet/DP-cabonly (seed 1)|`20261030T070858Z-d5f54853`|
|cabinet/DP+TD+BELL-cabonly|`20261030T064829Z-b08f963c`|
|cabinet/DP+TD+BELL-cabonly (seed 1)|`20261030T082135Z-99622d98`|
|cabinet/DP+TD+BELL-cabonly (seed 2)|`20261030T082135Z-e9519644`|
|cabinet/long-rk4-400 (DP+TD+BELL-cabonly)|`20261030T081658Z-bc9ecce4`|
|cabinet/long-mppi-100 (DP+TD+BELL-cabonly)|`20261030T081742Z-efeca026`|

|cabinet/labels part 0 (local_truth.py, 400k pairs)|`20261031T080528Z-c6afd62b`|
|cabinet/labels part 1|`20261031T082853Z-063ecaf6`|
|cabinet/SL (supervised specialists 1–2; frozen eval = SL12)|`20261031T084706Z-a023cefc`|
|cabinet/exact pieces, levels 3–4 (ES)|`20261031T082932Z-130521b1`|
|cabinet/exact pieces, level 3 (ES, SL)|`20261031T093118Z-b154200f`|
|cabinet/DP-SL|`20261031T090315Z-73517929`|
|cabinet/DP-SLs1|`20261031T095306Z-1efd501c`|
|cabinet/DP-SLs2|`20261031T102706Z-1e4249af`|
|cabinet/DP-ESs1 (full 20-segment chain)|`20261031T083504Z-8f620c9d`|
|cabinet/select (DP-ESs1, DP-SL)|`20261031T095711Z-8457340c`|
|cabinet/DP-SL+ES12 (splice)|`20261031T102726Z-ac8ea9f8`|
|cabinet/DP-ES+SL12 (splice)|`20261031T104412Z-9c7ac814`|
|cabinet/fit (B5, ES12, SL12, DP-ES, DP-ESs1, DP-SL)|`20261031T104435Z-3e868963`|
|cabinet/fit (DP-SLs1, splices)|`20261031T111435Z-80557e2c`|
|cabinet/fit (DP-SLs2)|`20261031T113605Z-f470cf02`|
|cabinet/long-rk4-400 (B5, ES12, SL12, DP-ES, DP-ESs1, DP-SL)|`20261031T102736Z-dd851457`|
|cabinet/long-rk4-400 (splices)|`20261031T110523Z-4fe70ae3`|
|cabinet/long-rk4-400 (DP-SLs1)|`20261031T111333Z-efd271de`|
|cabinet/long-rk4-400 (DP-SLs2)|`20261031T113523Z-115307ec`|
|cabinet/long-mppi-100 (DP-ESs1, DP-SL, SL12)|`20261031T103339Z-4ce892d7`|
|cabinet/long-mppi-100 (DP-SLs1, splices)|`20261031T111538Z-9e090901`|

Stopped or failed (not used): `20261027T174832Z-d52e0c6e` (cabinet ES before the sampler fix), `20261027T182219Z-c5833f56`
(no-wall ES, stopped to switch the routing implementation; rerun as `20261027T183924Z-aa1a5943`), `20261027T223102Z-730689a3`
(long-query stage with MPPI on all 400 queries, stopped as too slow: ~9 s per query; replaced by RK4 on 400 + MPPI on the
first 100), `20261031T080533Z-ff142556` (labels part 1 on the 5070, out of GPU memory at 208k of 400k; rerun on the 3090), `20261031T110024Z-becc4778` (long stage with '/' in its labels), `20261031T093015Z-bc045ba7` / `-9accd199` (DP-SL seeds 1 / 2 started by a queue-script bug next to two other chains, stopped at once and rerun), and the smoke runs of each entry.

## Commands

```bash
export BELLMAN600_DERIVED_IMAGE=bellman600-env:local BELLMAN600_DOCKER_MEMORY=8G     # Docker backend (bwrap unavailable)
python3 jeffrey/segment_dp/ablation/run.py --scene cabinet --weights 6,0,2 [--seed 1] --gpu 1          # part 1
python3 jeffrey/segment_dp/run.py --scene nowall --trainer eikonal --mode specialists --gpu 0          # ES
python3 jeffrey/segment_dp/run.py --scene nowall --trainer eikonal --mode generalist --gpu 0           # EG
python3 jeffrey/segment_dp/run.py --scene nowall --trainer eikonal --mode specialists --edges 0,0.08,0.16 --seed 1 --label ES2s1 --gpu 0
python3 jeffrey/segment_dp/run.py --scene cabinet --trainer dp --base-run registered_<ES run> --gpu 1  # DP
python3 jeffrey/segment_dp/run2.py --scene cabinet --base-run registered_<ES run> --pairs 150000 --candidates 96 --continuous 32 --refine 6 --gpu 1   # DP-c
python3 jeffrey/segment_dp/run.py --scene cabinet --trainer eikonal --mode specialists --arch offset --gpu 1   # ES-off (EG-off, DP-off likewise)
python3 jeffrey/segment_dp/run.py --scene cabinet --trainer bellman --arch offset --gpu 1              # B5-off
python3 jeffrey/segment_dp/run_stage.py truth --scene cabinet --gpu 1                                 # ground truth
python3 jeffrey/segment_dp/run_stage.py derived --runs registered_<ES run>=ES --first 2 --gpu 1       # specialists 1-2 only
python3 jeffrey/segment_dp/run_stage.py fit --truth-run registered_<T> --runs registered_<A>=ES ... --by-model ES --gpu 1
python3 jeffrey/segment_dp/run_stage.py long --runs registered_<A>=DP ... --mppi-seeds 0 --gpu 1
python3 jeffrey/segment_dp/run3.py --scene cabinet --base-run registered_<ES> --truth-run registered_<T> --last-segment 9 [--loss l1|l2|logl1|logl2|rel] [--decoder add|exp] [--candidates 32|1536] [--losses dp,td,bell] [--seed 1] --gpu 0   # section 3 chains
python3 jeffrey/segment_dp/run_stage.py fit --runs registered_<ES>=ES registered_<DP3>=DP3 ... --gpu 1   # section 3 truth fit
python3 jeffrey/segment_dp/run.py --scene cabinet_plus10 --trainer eikonal --mode specialists --gpu 0   # section 4 (cabinet_plus20 likewise)
python3 jeffrey/segment_dp/run.py --scene cabinet_plus10 --trainer dp --base-run registered_<ES plus10> --gpu 0
python3 jeffrey/segment_dp/run_stage.py long --runs registered_<B5 plus10 pipeline>=B5 registered_<ES plus10>=ES registered_<DP plus10>=DP --count 400 --gpu 0
python3 jeffrey/segment_dp/run_stage.py long --runs ... --count 100 --mppi-seeds 0 --gpu 0
python3 jeffrey/segment_dp/run3.py --scene cabinet --base-run registered_<ES> --truth-run registered_<T> --last-segment 9 --init warm [--seed 1] --gpu 0   # section 5
python3 run.py cabinet-tests --scene cabinet_clutter --gpu 0; python3 experiments/cabinet/freeze_tests.py results/registered_<tests run>   # section 6
python3 run.py cabinet --scene cabinet_clutter --gpu 0                                                # pipeline: pool, speed, B5
python3 jeffrey/segment_dp/run.py --scene cabinet_clutter --trainer eikonal --mode specialists --prepared-run registered_<clutter pipeline> --gpu 0   # (cabinet_platforms likewise)
python3 jeffrey/segment_dp/run3.py --scene cabinet_clutter --prepared-run registered_<clutter pipeline> --base-run registered_<ES clutter> [--losses dp,td,bell] [--seed 1] --gpu 0
python3 jeffrey/segment_dp/run_stage.py truth --scene cabinet_clutter --prepared-run registered_<clutter pipeline> --gpu 0
python3 jeffrey/segment_dp/run_stage.py derived --scene cabinet_clutter --runs registered_<DP>=DP --splice registered_<ES> --segments 3-6 --gpu 0
python3 jeffrey/segment_dp/probes.py barrier --truth results/<truth>/truth/truth.npz --prepared results/<pipeline>/prepared --fields ES=results/<ES>/training/field.pt DP=...   # host
python3 jeffrey/segment_dp/probes.py ridge --truth results/<truth>/truth/truth.npz --fields ES=... DP=...
python3 jeffrey/segment_dp/run_stage.py pool --scene cabinet --gpu 0                                   # section 7: cabinet-only pool
python3 jeffrey/segment_dp/run.py --scene cabinet --trainer eikonal --mode specialists --prepared-run registered_<pool run> --label ES-cabonly --gpu 0
python3 jeffrey/segment_dp/run3.py --scene cabinet --prepared-run registered_<pool run> --base-run registered_<ES-cabonly> [--losses dp,td,bell] --label cabonly --gpu 0
python3 jeffrey/segment_dp/run4.py labels --part 0/2 --gpu 1 [--chunk 2048]                        # section 8: path-optimized local labels
python3 jeffrey/segment_dp/run4.py supervised --label-runs registered_<part 0> registered_<part 1> --gpu 1   # SL (+ SL12 frozen eval)
python3 jeffrey/segment_dp/run4.py dp3 --base-run registered_<SL> --truth-run registered_<T> [--seed 1] --gpu 1   # DP-SL
python3 jeffrey/segment_dp/run3.py --scene cabinet --base-run registered_<ES> --last-segment 20 --truth-run registered_<T> --seed 1 --label DP20-ESs1 --gpu 0
python3 jeffrey/segment_dp/run4.py exact --truth-run registered_<T> --runs registered_<ES>=ES registered_<SL>=SL --exact-segments 3 --exact-pairs 1024 --gpu 1 --chunk 2048
python3 jeffrey/segment_dp/run4.py select --truth-run registered_<T> --runs registered_<DP-ES>=DP-ESs1 registered_<DP-SL>=DP-SL --gpu 1 --chunk 2048
python3 jeffrey/segment_dp/run_stage.py derived --runs registered_<DP-SL> --splice registered_<ES> --segments 1-2 --gpu 1
```

## Files

`fields.py` (TravelCost, OffsetCost, batched Ensemble, SegmentField routing, checkpoint I/O), `sampler.py` (pool,
pair sampling), `eikonal.py` (ES / EG trainer), `dp.py` / `dp2.py` (DP chain, pool / continuous split points),
`bellman.py` (original method with either network), `ablation/train.py` (loss weights), `truth.py`, `fit.py`,
`derived.py`, `long.py` (wraps `../gap_curriculum/long.py`), `evaluate.py` (wraps the frozen evaluators), `pipeline*.py`,
`run*.py`, `run_stage.py`, `launch.py` (Docker or host backend with launch.py's mounts), `report.py`. Section 3–4: `dp3.py` (DP / TD / ball-Bellman terms, losses, decoders, candidate sweep, per-level truth fit), `pipeline3.py`, `run3.py`; `run.py` and `run_stage.py` take `--scene cabinet_plus10 | cabinet_plus20`. Sections 5–7: `dp3.py --init warm`; `derived.py --splice DONOR --segments A-B` (models of chosen segments taken from another field); `pool_region.py` (`run_stage.py pool`: the cabinet-only pool); `run_stage.py truth|pool --prepared-run` (a scene's own pipeline run); `probes.py` (host diagnostics: barrier, ridge); scenes `experiments/cabinet/scenes/cabinet_clutter.yml`, `cabinet_platforms.yml` with frozen suites `tests_clutter/`, `tests_platforms/`. Section 8: `local_truth.py` (path-optimized local travel times and their envelope gradients), `supervised.py` (SL specialists), `dp_exact.py` (exact-piece DP targets), `dp_select.py` (target bias decomposition), `pipeline4.py`, `run4.py` (stages labels / supervised / dp3 / exact / select), `probes_section8_ridge.json`.
