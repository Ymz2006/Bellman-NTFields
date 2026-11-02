# jeffrey/eikonal_bellman: an eikonal foundation for Bellman updates to longer travel times

*2026-10-28, session started 01:15 EDT on the RTX 3090 (cabinet) and RTX 5070 (no-wall). Every number is from a
registered run (index at the end). MPPI = release MPPI successes /1800 (600 queries × planning seeds 0/1/2), RK4 = native
RK4 /600, release scorer (cabinet tilt gate off, no-wall tilt 20°). Training seed 87358 unless stated. Tables are generated from the registered runs by `report.py`;
the findings sections were written as the runs finished.*

## Research question (user, 2026-10-28)

The eikonal loss on close pairs gives a travel-time field that is accurate locally but falls off at longer distances. Can
that field be the **foundation** for **Bellman updates** that cover the longer distances? Subtask 1: a curriculum that
learns the small distances with the eikonal equation and then trains increasing distances (segment-DP style), with the
original network kept frozen for the close features and further network(s) — several, one per range, or one — trained
for the longer travel times. Subtask 2: analyze the pitfalls, then try something different on top.

## What was known before this session (jeffrey/segment_dp, jeffrey/gap_curriculum)

- One eikonal model trained on pool pairs with gap ≤ 0.176 (**EG16**) is 2–3× more accurate than the original method at
  short range (cabinet relative error 0.086 vs 0.29 below 0.16) and plans better on the frozen (short, ≤ 0.68) queries:
  cabinet MPPI 737 vs 694, no-wall 1675 vs 1590. Extrapolated to long gaps it underestimates (cabinet bias 0.73 at
  0.72–1.6, no-wall 0.40).
- The original Bellman term over the offline bank is not a DP recursion: the min takes the 0.01 step in 97–99.7 % of
  backups because the field underestimates, so it is a discretized eikonal condition between pairs of the same gap.
- A segment-DP chain T(i,j) = min_k T(i,k) + T(k,j) from the two shortest eikonal specialists is the only field whose
  long-range travel times are right (cabinet relative error 0.14 at 0.72–1.6 vs ≥ 0.25) but had two pitfalls: with
  **pool split points** the sparse lens adds a detour that compounds (+22–32 % at gaps 0.16–0.48); with **continuous,
  gradient-refined split points** the min finds configurations where the frozen models read low — regions no pose covers
  — and the chain ends 30 % low. On the frozen short benchmarks it was level with the original method and below the
  foundation alone.

## Method of this session

Foundation F = the registered EG16 field of each scene (one 338,177-parameter TravelCost trained with the eikonal loss on
pool pairs of gap 0.004–0.176), **frozen**. Field edges 0, 0.16, 0.24, 0.32, …, 1.6: segment 0 (|q−g| ≤ 0.16) is F;
every further segment of width 0.08 is learned from Bellman targets built on frozen fields, level by level, so each
entry is built from entries of shorter gap that are already final (interval DP). Blend 0.016 at every edge (linear ramp,
continuous value and gradient), as in segment_dp.

**Bellman target of a pair (i, j) in segment [e_m, e_m+1]** (`chain.py`):

|Scheme|Target|First leg|Tail|
|---|---|---|---|
|**EFL** (eikonal first leg)|min_k F(i,k) + T_<m(k,j)|the foundation, \|i−k\| ≤ 0.176|the frozen composite of everything learned so far, \|k−j\| ≤ e_m + 0.016|
|**symmetric** (segment_dp/dp.py)|min_k T_<m(i,k) + T_<m(k,j)|the composite, \|i−k\| ≤ e_m + 0.016|the composite, \|k−j\| ≤ e_m + 0.016|
|**vi** (control)|min_k F(i,k) + T_r−1(k,j)|the foundation, \|i−k\| ≤ 0.176|the previous round's field at **any** gap; pairs of all gaps every round|

EFL is the literal form of the research question: a Bellman backup whose one-step cost is the eikonal field (jumps of up
to 0.176 instead of the bank's 0.01 steps) and whose tail is only read where it has been trained. The vi control removes
the trusted-range restriction (value iteration on natural pairs, started from the foundation's extrapolation).

**Networks** (`--models`): *multi* — a fresh TravelCost per segment (initialized from F's weights), the composite is F +
17 models; *single* — one long-range network L (initialized from F) for every gap > 0.16, trained level by level on the
new segment's targets plus an equal share of the earlier segments' targets (rehearsal), the field is F + L; *finetune* —
F itself is fine-tuned level by level (one network for every gap; the frozen original F still provides the first leg),
with or without 256 short pairs per batch under the eikonal loss (`--rehearsal`).

**Split points** (`--split`): *pool* — 128 pool poses in the lens (64 nearest the straight line, 64 uniform; empty lens →
pair dropped); *gated* — the pool candidates plus 32 continuous points along the straight segment with noise, the best 2
refined by 4 normalized gradient steps on the frozen composite, every off-pool point valid only if it is within ρ of a
pool pose (ρ = the 95th percentile of the pool's nearest-neighbour distance: cabinet 0.097, no-wall 0.050) and, on the
cabinet, collision-free by the exact release clearance (`jeffrey/geometry.py`, 5 mm obstacle / 2 mm self margins).

**Loss of a learned segment**: SmoothL1(T, target) (the Bellman term's loss), optionally + 2 × the eikonal residual and
+ 2 × (1 − cos) between −∇_q T and the foundation's descent direction toward the winning split point (`--weights 6,2,2`,
the original method's three terms with eikonal-field actions). **Estimator** (`--estimator double`): two models per level
on disjoint pairs; k is selected with composite A, the target read with composite B (winner's-curse diagnostic).

Per level: 400k pairs (200k / 150k for gated), targets computed once, 15,620 updates of batch 256, AdamW 2e-4, weight
decay 1e-6, clip 2 (as the original method). **Diagnostics**: for the probe pairs of segment_dp's path-optimized ground
truth T* (~1,000 per segment), the median target / T* and model / T* at every level (`--truth`); the fit stage
(`segment_dp/fit.py`: relative error, bias, Spearman, direction cosine per gap range) on the final field; the frozen
benchmarks; the 400 uncertified long cabinet queries (`long.py`, RK4; MPPI seed 0 on the first 100). Docker backend
`bellman600-env:local` (modified environment, recorded per run).

## Summary

*Session 2026-10-28, 01:15–13:15 EDT, both GPUs; 37 registered training runs and 29 registered diagnostic stages (run index at
the end). RK4 /600 and release MPPI /1800 on the frozen suites; the pool-only foundation EG16 is RK4 255 / MPPI 737 (seeds
734–760) on the cabinet and 553 / 1675 (1675–1681) on no-wall; the original method 229 / 688 and 550 / 1598.*

1. **Where the short-range eikonal field falls off.** Extrapolated beyond its 0.176 training range the foundation is right
   on every pair whose optimal path is nearly straight (median T/T* 0.99–1.03 up to gap 0.8 on the cabinet, 0.93–0.97 on
   no-wall) and reads the *slow* pairs — detours around the cupboard, wrap-arounds of the orientation manifold — at half
   their value (0.48–0.60 cabinet, 0.36–0.45 no-wall), exactly as the original method does. Inside its range it reads
   slow pairs right (0.94–1.00). "Falls off" means: it does not know which long pairs are slow.
2. **The Bellman extension on the frozen foundation fixes the long-range field.** Every chain ranks long pairs far better
   than the foundation (Spearman at gaps 0.72–1.6: 0.72–0.82 vs 0.15 cabinet, 0.70–0.73 vs 0.03 no-wall) and the
   three-term chains (value + eikonal + direction) are the best long-range fields of all sessions (K8/K11: Spearman
   0.81–0.82, direction cosine 0.71, eikonal residual 0.1). Each construction carries a bias of its own: pool split
   points inflate direct pairs 12–20 % (the sparse-lens detour, inherited by every level of the eikonal-first-leg chain,
   decaying to 2 % in the symmetric chain); continuous split points — gated by exact clearance and pool support or not,
   on the pool-only or the augmented foundation — under-read the slow pairs (0.57–0.68 of T*), one third winner's curse
   of the min, two thirds the foundation's blind spots off the pool; a tight support gate brings back the pool chain. And
   on no-wall any eikonal term on the shared long-range network collapses the values beyond gap 0.48 to half (the
   endpoint residual re-imposes the straight-line rate).
3. **For the planners the loss on the learned segments decides, not the chain.** On the frozen suites the cabinet is
   insensitive to all of it (every chain 712–746 vs the foundation's 731–760; 59 % of queries fail by collision inside the
   compartments with every field). On no-wall value-regressed segments *hurt* (1335–1636 vs 1675) because they are nearly
   flat off the orientation manifold (a 0.03 step off the manifold lowers T for 24–32 % of samples vs 7–8 % under the
   foundation) and the planners tilt; the eikonal and direction terms restore the wall and the three-term chain gives
   the best RK4 of any field (**569**/600, N8) with MPPI 1692–1706 (N8, N11).
4. **Keep the original network for the close features — frozen, or rehearsed.** Fine-tuning the foundation on the
   Bellman targets with the eikonal loss on short pairs in every batch improves the close-range fit (relative error 0.06
   vs 0.09) and gives no-wall MPPI 1710 and 1714 on two training seeds (N5; cabinet 746, K5); without rehearsal seventeen
   levels of value regression overwrite it (no-wall 1335, cabinet 693). But the control — the pool-only foundation trained
   for the same 281,160 eikonal updates with no Bellman targets at all — gives the same close-range field and MPPI
   **1720**, so the fine-tuning's gain is its extra eikonal training, not the Bellman targets. Several networks or one: with values only, one
   network for the whole long range is the worst configuration (no-wall 1447 vs 1591 for one per segment); with the
   three-term loss one network is the best. Value iteration without the trusted-range order (natural pairs, unrestricted
   tail) moves the frontier of correct values by about one hop every two rounds and never crosses the wrap-around gap on
   no-wall: the order is what spends every level's compute at the frontier.
5. **The practical results need no chain.** (a) The short-range eikonal foundation trained 18× longer (281,160 updates,
   37 min) is the best no-wall field of all sessions: RK4 559, MPPI **1720** (95.6 %; original method 1598, 15,620-update
   foundation 1675–1681). (b) At equal updates, an eikonal foundation trained with half of its pairs' endpoints moved off
   the pool (F-aug, σ = 0.03–0.05) is +18 MPPI / +7 RK4 over the pool-only foundation on every no-wall training seed
   (σ = 0.05: 1703 / 1692 / 1696, σ = 0.03: 1708 / 1688 / 1703, vs 1675 / 1680 / 1681; identical fit statistics — the gain
   is the field off the manifold; σ = 0.1 gives 1676), and within seed noise on the cabinet (752–761 vs 734–760).

## Results tables

### Planning on the frozen suites

**Cabinet** (RK4 /600, release MPPI /1800 with planning seeds 0/1/2; the pool-only foundation EG16 = 255 / 737 and seeds 1, 2 = 253 / 734, 253 / 760; original method 229 / 688)

|Label|Scene|Scheme|Models|Split|Weights|RK4 /600|MPPI /1800 (seeds)|Train min|Eval min|
|---|---|---|---|---|---|---:|---|---:|---:|
|K1|cabinet|efl|multi|pool|1/0/0|249|731 (243/244/244)|35|4|
|K4|cabinet|symmetric|multi|pool|1/0/0|252|733 (245/244/244)|38|4|
|F-aug-cab|cabinet|foundation (generalist-augmented σ=0.05, 128 aug rows)|1|—|eikonal|255|752 (253/251/248)|3|2|
|K3|cabinet|efl|multi|gated +clear rho=0.097|1/0/0|252|731 (246/240/245)|71|4|
|K7|cabinet|efl|multi x2|gated +clear rho=0.097|1/0/0|—|—|39|0|
|F-aug-cab-s1|cabinet|foundation (generalist-augmented σ=0.05, 128 aug rows)|1|—|eikonal|255|759 (253/253/253)|2|2|
|F-aug-cab-s2|cabinet|foundation (generalist-augmented σ=0.05, 128 aug rows)|1|—|eikonal|261|761 (254/254/253)|2|2|
|EG16-cab-s1|cabinet|foundation (generalist)|1|—|eikonal|253|734 (243/244/247)|2|2|
|EG16-cab-s2|cabinet|foundation (generalist)|1|—|eikonal|253|760 (255/250/255)|2|2|
|K8|cabinet|efl|single|pool|6/2/2|252|734 (246/245/243)|53|3|
|K3b|cabinet|efl|multi|gated +clear rho=0.030|1/0/0|—|—|32|0|
|K2|cabinet|efl|single|pool|1/0/0|252|712 (238/237/237)|33|3|
|K5|cabinet|efl|finetune|pool|1/0/0 +reh256|240|746 (250/250/246)|58|3|
|K11|cabinet|efl|single|pool|6/2/2|254|737 (249/243/245)|52|3|
|EG16|cabinet|foundation (generalist)|1|—|eikonal|255|737 (246/246/245)|3|3|
|K9|cabinet|vi|single|pool|1/0/0|251|715 (242/236/237)|35|3|
|K6|cabinet|efl|finetune|pool|1/0/0|235|693 (231/230/232)|33|2|
|K10|cabinet|efl|multi|cont|1/0/0|—|—|14|0|
|N5s1|nowall|efl|finetune|pool|1/0/0 +reh256|563|1714 (572/571/571)|58|3|
|F-aug-nw-sig03-s1|nowall|foundation (generalist-augmented σ=0.03, 128 aug rows)|1|—|eikonal|550|1688 (563/563/562)|2|3|
|F-aug-nw-sig03-s2|nowall|foundation (generalist-augmented σ=0.03, 128 aug rows)|1|—|eikonal|563|1703 (568/568/567)|2|3|

**No-wall** (the pool-only foundation EG16 = 553 / 1675; original method 550 / 1598)

|Label|Scene|Scheme|Models|Split|Weights|RK4 /600|MPPI /1800 (seeds)|Train min|Eval min|
|---|---|---|---|---|---|---:|---|---:|---:|
|N1|nowall|efl|multi|pool|1/0/0|535|1591 (530/529/532)|49|7|
|F-aug-nw|nowall|foundation (generalist-augmented σ=0.05, 128 aug rows)|1|—|eikonal|556|1703 (568/568/567)|3|3|
|N10|nowall|efl|multi|pool|1/0/0|543|1636 (547/543/546)|49|5|
|F-aug-nw-s1|nowall|foundation (generalist-augmented σ=0.05, 128 aug rows)|1|—|eikonal|560|1692 (563/566/563)|3|3|
|N8|nowall|efl|single|pool|6/2/2|569|1692 (565/563/564)|66|4|
|N2|nowall|efl|single|pool|1/0/0|463|1447 (483/481/483)|46|4|
|F-aug-nw-s2|nowall|foundation (generalist-augmented σ=0.05, 128 aug rows)|1|—|eikonal|558|1696 (566/565/565)|3|3|
|F-aug-nw-sig03|nowall|foundation (generalist-augmented σ=0.03, 128 aug rows)|1|—|eikonal|557|1708 (569/569/570)|3|3|
|F-aug-nw-sig10|nowall|foundation (generalist-augmented σ=0.1, 128 aug rows)|1|—|eikonal|559|1676 (559/558/559)|3|3|
|EG16-nw-s1|nowall|foundation (generalist)|1|—|eikonal|551|1680 (560/560/560)|2|3|
|EG16-nw-s2|nowall|foundation (generalist)|1|—|eikonal|548|1681 (561/560/560)|2|3|
|N3|nowall|efl|multi|gated rho=0.030|1/0/0|522|1612 (538/537/537)|85|5|
|N5|nowall|efl|finetune|pool|1/0/0 +reh256|564|1710 (569/572/569)|72|4|
|N6|nowall|efl|finetune|pool|1/0/0|439|1335 (444/445/446)|45|2|
|N9|nowall|vi|single|pool|1/0/0|538|1610 (537/537/536)|49|4|
|N11|nowall|efl|single|pool|6/2/2|567|1706 (569/569/568)|66|4|
|EG16-nw-long|nowall|foundation (generalist)|1|—|eikonal|559|1720 (573/573/574)|37|4|

### Planning per suite

**Cabinet** (MPPI per suite /300: free→free, free→compartment, compartment→free, compartment→compartment, same compartment, anywhere)

|Label|f→f|f→c|c→f|c→c|same_c|anywhere|MPPI|RK4 per suite|RK4|
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
|K1|288|71|69|12|27|264|731|95, 28, 23, 5, 10, 88|249|
|K4|285|71|75|10|27|265|733|95, 28, 26, 5, 9, 89|252|
|F-aug-cab|288|75|77|13|23|276|752|94, 25, 28, 4, 10, 94|255|
|K3|285|67|73|16|20|270|731|95, 26, 26, 7, 11, 87|252|
|F-aug-cab-s1|294|86|79|8|17|275|759|99, 32, 23, 3, 6, 92|255|
|F-aug-cab-s2|291|84|80|13|14|279|761|97, 29, 26, 7, 7, 95|261|
|EG16-cab-s1|288|74|71|9|22|270|734|96, 28, 26, 6, 7, 90|253|
|EG16-cab-s2|287|80|74|23|17|279|760|96, 28, 21, 10, 6, 92|253|
|K8|285|69|75|12|24|269|734|95, 27, 24, 4, 10, 92|252|
|K2|280|68|66|17|23|258|712|95, 27, 26, 5, 11, 88|252|
|K5|288|76|71|6|32|273|746|92, 26, 24, 3, 6, 89|240|
|K11|284|70|78|14|19|272|737|95, 25, 29, 6, 6, 93|254|
|EG16|285|70|80|10|26|266|737|95, 27, 25, 5, 13, 90|255|
|K9|282|69|64|11|25|264|715|94, 27, 26, 4, 9, 91|251|
|K6|266|92|58|9|22|246|693|89, 30, 23, 2, 5, 86|235|

**No-wall** (mixed, screwdriver, cutting, hammering, stirring, wiping)

|Label|mixed|screwdriver|cutting|hammering|stirring|wiping|MPPI|RK4 per suite|RK4|
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
|N1|215|268|282|274|279|273|1591|75, 88, 96, 94, 93, 89|535|
|F-aug-nw|258|284|294|294|288|285|1703|80, 96, 96, 97, 94, 93|556|
|N10|226|278|289|285|282|276|1636|77, 91, 97, 91, 94, 93|543|
|F-aug-nw-s1|259|282|291|292|283|285|1692|84, 96, 97, 97, 93, 93|560|
|N8|250|282|299|291|285|285|1692|84, 97, 99, 97, 96, 96|569|
|N2|166|224|273|255|263|266|1447|55, 73, 86, 84, 83, 82|463|
|F-aug-nw-s2|256|288|294|294|282|282|1696|82, 97, 98, 94, 92, 95|558|
|F-aug-nw-sig03|265|285|291|291|288|288|1708|86, 91, 97, 93, 95, 95|557|
|F-aug-nw-sig10|245|285|291|285|285|285|1676|88, 92, 98, 95, 91, 95|559|
|EG16-nw-s1|250|281|295|291|285|278|1680|83, 93, 97, 93, 94, 91|551|
|EG16-nw-s2|252|282|294|291|286|276|1681|82, 93, 98, 95, 90, 90|548|
|N3|216|270|292|282|276|276|1612|72, 87, 95, 88, 92, 88|522|
|N5|260|285|294|297|288|286|1710|85, 94, 98, 98, 95, 94|564|
|N6|156|191|254|234|252|248|1335|50, 62, 85, 80, 83, 79|439|
|N9|214|274|291|273|282|276|1610|72, 88, 98, 94, 92, 94|538|
|N11|261|282|297|294|287|285|1706|85, 98, 97, 97, 95, 95|567|
|EG16-nw-long|262|285|297|297|288|291|1720|83, 93, 98, 96, 95, 94|559|

### Per level: median Bellman target / T* on the ground-truth probe pairs of each segment

**Cabinet**

|Level (gap)|K1|K4|K3|K7|K8|K3b|K2|K5|K11|K9|K6|K10|N5s1|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1 (0.16–0.24)|1.165|1.165|1.011|1.011|1.165|1.131|1.165|1.165|1.160|0.760|1.165|1.001|1.053|
|2 (0.24–0.32)|1.183|1.185|1.000|1.003|1.180|1.154|1.183|1.097|1.176|0.752|1.207|0.994|1.020|
|3 (0.32–0.40)|1.198|1.180|0.980|0.989|1.183|1.170|1.189|1.079|1.179|0.762|1.199|0.973|0.987|
|4 (0.40–0.48)|1.191|1.160|0.961|0.972|1.159|1.165|1.189|1.074|1.155|0.774|1.201|0.954|0.845|
|5 (0.48–0.56)|1.182|1.134|0.937|0.957|1.072|1.143|1.169|1.034|1.064|0.790|1.179|0.910|0.529|
|6 (0.56–0.64)|1.200|1.129|0.909|0.933|0.977|1.149|1.178|1.003|0.974|0.805|1.185|0.872|0.520|
|7 (0.64–0.72)|1.181|1.089|0.839|0.890|0.920|1.132|1.166|0.937|0.922|0.820|1.179|0.810|0.528|
|8 (0.72–0.80)|1.177|1.059|0.796|0.880|0.889|1.128|1.170|0.911|0.892|0.837|1.173|0.771|0.516|
|9 (0.80–0.88)|1.163|1.057|0.792|—|0.890|—|1.160|0.902|0.893|0.853|1.169|—|0.523|
|10 (0.88–0.96)|1.159|1.042|0.775|—|0.888|—|1.153|0.885|0.892|0.869|1.159|—|0.538|
|11 (0.96–1.04)|1.160|1.044|0.775|—|0.892|—|1.152|0.889|0.895|0.888|1.155|—|0.541|
|12 (1.04–1.12)|1.163|1.047|0.778|—|0.900|—|1.161|0.897|0.903|0.904|1.164|—|0.528|
|13 (1.12–1.20)|1.154|1.046|0.781|—|0.904|—|1.146|0.896|0.905|0.919|1.149|—|0.523|
|14 (1.20–1.28)|1.152|1.039|0.774|—|0.905|—|1.144|0.882|0.906|0.936|1.147|—|0.505|
|15 (1.28–1.36)|1.144|1.042|0.773|—|0.906|—|1.138|0.887|0.910|0.953|1.140|—|0.500|
|16 (1.36–1.44)|1.144|1.039|0.768|—|0.907|—|1.135|0.893|0.909|0.969|1.138|—|0.481|
|17 (1.44–1.52)|1.137|1.037|0.766|—|0.909|—|1.123|0.888|0.905|0.982|1.129|—|0.472|
|18 (1.52–1.60)|1.123|1.024|0.758|—|0.902|—|1.114|0.875|0.902|0.997|1.117|—|0.468|

**No-wall**

|Level (gap)|N1|N10|N8|N2|N3|N5|N6|N9|N11|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1 (0.16–0.24)|1.053|1.101|1.053|1.053|1.015|1.053|1.053|0.425|1.101|
|2 (0.24–0.32)|1.043|1.094|1.040|1.043|1.016|1.018|1.059|0.432|1.094|
|3 (0.32–0.40)|1.040|1.078|1.021|1.039|1.000|0.996|1.042|0.438|1.066|
|4 (0.40–0.48)|0.983|1.015|0.870|0.967|0.932|0.850|0.970|0.445|0.909|
|5 (0.48–0.56)|0.745|0.779|0.516|0.737|0.658|0.524|0.734|0.454|0.536|
|6 (0.56–0.64)|0.718|0.749|0.495|0.701|0.632|0.519|0.704|0.462|0.519|
|7 (0.64–0.72)|0.731|0.774|0.502|0.706|0.638|0.533|0.703|0.469|0.524|
|8 (0.72–0.80)|0.728|0.764|0.495|0.700|0.623|0.524|0.698|0.477|0.514|
|9 (0.80–0.88)|0.753|0.798|0.499|0.723|0.637|0.525|0.723|0.485|0.519|
|10 (0.88–0.96)|0.777|0.821|0.521|0.743|0.653|0.540|0.748|0.493|0.545|
|11 (0.96–1.04)|0.801|0.834|0.524|0.776|0.661|0.547|0.772|0.500|0.553|
|12 (1.04–1.12)|0.801|0.835|0.525|0.783|0.665|0.532|0.784|0.508|0.551|
|13 (1.12–1.20)|0.807|0.842|0.535|0.786|0.678|0.523|0.790|0.515|0.567|
|14 (1.20–1.28)|0.815|0.845|0.530|0.788|0.675|0.509|0.786|0.524|0.565|
|15 (1.28–1.36)|0.813|0.843|0.531|0.791|0.677|0.500|0.787|0.532|0.564|
|16 (1.36–1.44)|0.815|0.845|0.524|0.786|0.671|0.479|0.786|0.542|0.559|
|17 (1.44–1.52)|0.814|0.843|0.523|0.777|0.668|0.466|0.780|0.549|0.563|
|18 (1.52–1.60)|0.823|0.853|0.525|0.781|0.666|0.467|0.782|0.556|0.566|

### Fit of the final fields to the ground truth (fit stages; means over the 0.08 segments of each gap range)

**Cabinet**

|Field|rel. error 0–0.24|rel. error 0.24–0.72|rel. error 0.72–1.6|bias 0–0.24|bias 0.24–0.72|bias 0.72–1.6|Spearman 0–0.24|Spearman 0.24–0.72|Spearman 0.72–1.6|cosine 0–0.24|cosine 0.24–0.72|cosine 0.72–1.6|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|K1|0.122|0.232|0.167|1.077|1.197|1.155|0.919|0.848|0.740|0.924|0.705|0.600|
|K4|0.122|0.221|0.132|1.076|1.146|1.043|0.919|0.860|0.750|0.925|0.760|0.687|
|F-aug-cab|0.095|0.279|0.319|1.018|0.926|0.737|0.921|0.456|0.161|0.922|0.730|0.544|
|K3|0.084|0.211|0.257|1.017|0.934|0.773|0.936|0.837|0.715|0.922|0.721|0.625|
|F-aug-cab-s1|0.095|0.278|0.321|1.017|0.936|0.734|0.927|0.453|0.161|0.925|0.734|0.550|
|F-aug-cab-s2|0.095|0.279|0.322|1.022|0.932|0.734|0.927|0.456|0.178|0.923|0.730|0.538|
|EG16-cab-s1|0.088|0.284|0.339|1.006|0.914|0.723|0.933|0.465|0.148|0.923|0.720|0.527|
|EG16-cab-s2|0.089|0.284|0.338|1.023|0.911|0.720|0.935|0.466|0.165|0.923|0.717|0.524|
|K8|0.177|0.252|0.153|1.144|1.107|0.898|0.898|0.803|0.809|0.931|0.783|0.712|
|K3b|0.112|0.223|0.239|1.060|1.159|1.257|0.922|0.840|0.668|0.916|0.673|0.515|
|K2|0.122|0.221|0.160|1.078|1.180|1.143|0.909|0.858|0.746|0.928|0.772|0.688|
|K5|0.062|0.236|0.186|1.010|1.008|0.887|0.965|0.731|0.551|0.928|0.750|0.606|
|K11|0.182|0.245|0.148|1.149|1.102|0.902|0.892|0.827|0.823|0.926|0.781|0.710|
|K9|0.124|0.226|0.185|1.080|1.178|0.924|0.914|0.851|0.595|0.925|0.735|0.594|
|K6|0.200|0.225|0.162|1.153|1.191|1.145|0.853|0.859|0.743|0.930|0.772|0.688|
|N5s1|0.059|0.471|0.674|1.007|0.716|0.508|0.926|0.609|0.464|0.877|0.620|0.437|
|F-aug-nw-sig03-s1|0.099|0.577|0.936|1.033|0.636|0.386|0.864|0.291|0.020|0.856|0.575|0.391|
|F-aug-nw-sig03-s2|0.091|0.565|0.910|1.021|0.648|0.396|0.890|0.314|0.029|0.865|0.577|0.372|

**No-wall**

|Field|rel. error 0–0.24|rel. error 0.24–0.72|rel. error 0.72–1.6|bias 0–0.24|bias 0.24–0.72|bias 0.72–1.6|Spearman 0–0.24|Spearman 0.24–0.72|Spearman 0.72–1.6|cosine 0–0.24|cosine 0.24–0.72|cosine 0.72–1.6|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|N1|0.091|0.308|0.259|1.030|0.870|0.797|0.896|0.771|0.723|0.878|0.597|0.434|
|F-aug-nw|0.108|0.550|0.878|1.054|0.668|0.408|0.868|0.323|0.027|0.849|0.566|0.360|
|N10|0.120|0.300|0.233|1.086|0.910|0.829|0.878|0.779|0.726|0.870|0.599|0.437|
|N8|0.119|0.479|0.643|1.072|0.742|0.522|0.856|0.570|0.486|0.873|0.646|0.521|
|N2|0.099|0.321|0.286|1.034|0.850|0.768|0.863|0.763|0.730|0.881|0.647|0.524|
|F-aug-nw-s1|0.102|0.566|0.912|1.032|0.651|0.395|0.869|0.299|0.011|0.850|0.571|0.370|
|F-aug-nw-s2|0.099|0.558|0.900|1.026|0.656|0.400|0.860|0.324|0.044|0.858|0.580|0.378|
|F-aug-nw-sig03|0.102|0.567|0.926|1.049|0.647|0.388|0.887|0.312|0.019|0.850|0.567|0.385|
|F-aug-nw-sig10|0.156|0.568|0.905|1.114|0.655|0.397|0.809|0.288|-0.017|0.840|0.573|0.381|
|EG16-nw-s1|0.101|0.554|0.895|1.027|0.654|0.402|0.862|0.345|0.036|0.852|0.568|0.369|
|EG16-nw-s2|0.097|0.560|0.901|1.021|0.652|0.399|0.871|0.332|0.024|0.852|0.564|0.364|
|N3|0.087|0.363|0.418|1.018|0.799|0.660|0.897|0.746|0.699|0.870|0.548|0.358|
|N5|0.059|0.469|0.673|1.006|0.720|0.509|0.932|0.607|0.449|0.881|0.622|0.439|
|N6|0.158|0.323|0.287|1.048|0.854|0.766|0.732|0.763|0.733|0.904|0.643|0.520|
|N9|0.091|0.348|0.725|1.032|0.822|0.485|0.900|0.768|0.321|0.883|0.632|0.433|
|N11|0.152|0.466|0.589|1.130|0.774|0.553|0.857|0.614|0.530|0.869|0.646|0.517|
|EG16-nw-long|0.060|0.542|0.864|1.010|0.670|0.416|0.931|0.339|0.048|0.877|0.594|0.380|

## Findings

### 1. Where the foundation falls off

**Where the foundation falls off.** Split the ground-truth probe pairs of each gap range into *fast* pairs (T* ≤ 1.8 ×
gap: the optimal path is about straight) and *slow* pairs (T* > 1.8 × gap: the path goes around the cupboard, or wraps
around the orientation manifold on no-wall; they are 18 % of cabinet pairs at 0.16–0.4, 38–46 % beyond 0.4; no-wall 17 %
then 60–93 %). The frozen foundation EG16, extrapolated beyond its 0.176 training range, predicts the **fast pairs right
at every gap** (cabinet median T/T* 1.01–1.03 up to 0.8, 0.84 at 0.8–1.6; no-wall 0.93–0.97, 0.77) and the **slow pairs
at half their value** (cabinet 0.48–0.60, no-wall 0.36–0.45) — exactly as the original method does (B5 no-wall 0.36–0.41).
The short-range eikonal field extrapolates as if every path were straight; "falling off at long distances" is the
slow pairs only. (`wrap.py` on the registered truth and fields; unregistered diagnostic.)

Inside its training range the foundation is not blind: on no-wall the slow pairs below gap 0.176 are predicted at
0.99 (0–0.08) and 0.94 (0.08–0.176) of T* (the original method: 0.64 / 0.73), and the deficit appears only beyond the
range (0.85 at 0.176–0.24, 0.69 at 0.24–0.32, 0.36–0.45 further out). On the cabinet the same holds: slow pairs 0.98 (0–0.08), 1.00 (0.08–0.176), still 1.00 at 0.176–0.24 and 0.86 at
0.24–0.32 (the original method: 0.51 / 0.67 / 0.72 / 0.68). The local information the Bellman extension needs
is there — for pool poses; what the chains do with it is the question. (T* is an upper bound from a graph route plus path optimization,
so part of any slow-pair deficit may be the reference's own slack; all fields are measured against the same T*.)

### 2. The Bellman extension with pool split points

**EFL chain with pool split points (K1 cabinet, N1 no-wall).** The Bellman extension learns that the slow pairs are
slow (cabinet slow-pair median 0.84 → 1.03 from 0.16–0.4 to 0.8–1.6; no-wall 0.58–0.79) and ranks long pairs far better
than the foundation (Spearman at 0.72–1.6: cabinet 0.74 vs 0.15, no-wall 0.72 vs 0.03), but every level inherits the
sparse-pool detour of the first leg: fast pairs are inflated 20–29 % (cabinet) / 7–11 % (no-wall), target/T* stays
1.12–1.20 at every cabinet level (per-level diagnostic), and the per-level detour never shrinks the way it does in the
symmetric chain because the first leg is always a ≤ 0.176 hop through ~100 pool poses. Planning on the frozen (short)
suites: cabinet RK4 249, MPPI 731 (foundation alone 255 / 737; original method 229 / 688) — level with the foundation;
no-wall RK4 535, MPPI 1591 (foundation 553 / 1675) — **worse than the foundation, back at the original method's
1590**, and the loss is tilt: MPPI seed-0 tilt failures 65 vs 38, RK4 45 vs 9. The learned segments are regressions on
values; their gradients carry no information about the orientation manifold (no eikonal or direction term), so beyond
gap 0.16 they steer off the 20° tilt limit — the chain re-creates the no-direction-term failure of the original
method. Segment edges: a 14 % (cabinet) / 6 % (no-wall) step at the foundation edge (the first level's detour bias),
≤ 3 % at later edges; the eikonal residual |log S|∇T|| of the learned segments is 0.45 (cabinet) / 0.5–0.7 (no-wall)
vs 0.14–0.19 for the foundation at short range.

**Symmetric chain, same foundation (K4, cabinet; segment_dp's transition with 128 candidates).** Planning RK4 252,
MPPI 733 — the same as K1 and the foundation (every cabinet failure is a collision: 354–357 of the 600 seed-0 MPPI
runs for all three fields). Its targets start with the same 16.5 % detour at level 1 but the lens grows from 17 to
377k poses, so the detour shrinks (target/T* 1.165 → 1.13 at level 5 → 1.06 at level 8 → 1.02 at level 18) where
EFL's stays at 1.12–1.20; fast pairs 1.14–1.23, slow pairs 0.81–0.92; long-range bias 1.04 (K1 1.16), Spearman 0.75
(K1 0.74), direction cosine 0.69 at 0.72–1.6 (K1 0.60, the foundation 0.52): a learned long leg gives a smoother
gradient than a hop to one of ~100 pool poses. On the frozen (short) suites the long-range field is irrelevant.

### 3. Split points: the detour bias, the blind spots and the winner's curse

**Gated continuous split points (K3, cabinet; 200k pairs per level, exact clearance + support gates).** The detour
bias is gone: level-1 target/T* 1.011 (K1 1.165), direct pairs exact at every gap (median 0.99–1.00, mean |log| error
0.03–0.05 even at 0.8–1.6), short- and mid-range relative error the best of any field (0.084 / 0.211 vs the
foundation's 0.088 / 0.285). But the chain drifts low with level (target/T* 1.01, 1.00, 0.98, 0.96, 0.94, 0.91, 0.84,
0.80 … 0.77 at level 16), and the drift is all in the slow pairs: 0.65 at 0.16–0.4, 0.57 at 0.4–0.56, 0.64–0.68 beyond
(pool chains 0.81–1.03, the foundation 0.48–0.60). Refined continuous points win 81–99 % of the pairs although the
gates reject 17–22 % of them (support) and 6–18 % (clearance): the refinement minimizes F(i,k) + T(k,j) over a
continuum and finds the k whose legs the models under-read — valid, in free space, near pool poses, but through gaps the
foundation never saw as slow: the gates certify validity (collision-free, within 0.097 of a pool pose — a large region in
6-D, the pool's own nearest-neighbour distance is 0.045), not that the foundation is accurate there; the pool chain's split
points are pool poses, where the foundation reads slow legs right, and it recovers the slow pairs. **A tight support gate (within 0.03 of a pool pose; K3b cabinet, N3 no-wall) confirms it the hard way**: the gate
rejects 86–100 % of the continuous candidates, pool poses win 91–98 % of the pairs from level 2 on, and the chain is
the pool chain again — cabinet target/T* 1.13–1.17 (detour 1.09–1.13), fast pairs 1.18–1.26 and slow pairs 0.78–0.90
against the truth (K1: 1.21–1.29 / 0.84–0.95), no-wall collapsing to 0.63–0.66 beyond gap 0.48. There is no gate setting between the two biases, because the region where the foundation reads slow legs
correctly is the pool itself. Long-range bias 0.77, Spearman 0.72, cosine 0.63. Planning RK4 252, MPPI 731 (246/240/245)
— again the foundation's number (all failures collisions); training 71 min.

**How much of the drift is the winner's curse? About a third (K7, the gated chain to gap 0.8 with two models per
level on disjoint pairs; k selected with composite A, the target read with composite B).** Double- vs single-estimator
target/T* on the same truth pairs, levels 1–8: 1.011/1.010, 1.003/1.000, 0.989/0.982, 0.972/0.964, 0.957/0.934,
0.933/0.898, 0.890/0.847, 0.880/0.820. The selection bias of the min over ~160 noisy readings is < 1 % per level at
first and 4–6 % per level by levels 7–8 (the models' held-out error grows from 3 % to 6 %), and it compounds: at gap
0.72–0.8 the double chain reads 0.88 of T* where the single chain reads 0.80–0.82. The remaining two thirds of the drift
is systematic — the slow legs the foundation under-reads off the pool (next paragraphs).

**Ungated continuous split points on the augmented foundation (K10, cabinet, diagnostic to gap 0.8).** Same trajectory
as the gated chain on the pool-only foundation: target/T* 1.00, 0.99, 0.97, 0.95, 0.91, 0.87, 0.81, 0.77 (K3: 1.01 … 0.80;
the previous session's DP-c: 1.02 … 0.70), direct pairs exact (0.99–1.00), slow pairs 0.56–0.63. Training the foundation
with endpoints moved off the pool does not change what the continuous minimization finds; neither did the validity
gates. The under-read legs are slow legs off the pool, and only split points **on** the pool avoid them (at the price of
the detour).

### 4. The loss on the learned segments decides planning; several networks or one

**The three-term loss on the learned segment repairs planning (N8, no-wall: one long-range network L for every gap >
0.16, EFL pool targets, loss 6 × SmoothL1(T, target) + 2 × eikonal residual + 2 × (1 − cos) to the foundation's descent
direction toward the winning split point).** RK4 **569**/600 — the best no-wall RK4 of any field in any session
(foundation 553, F-aug 556/560, original method 550, best gap-curriculum recipe 563) — and MPPI 1692 (565/563/564), level
with F-aug (1703 / 1692) and above the foundation (1675); tilt failures RK4 3 (N1 45, foundation 9), MPPI 31 (65 / 38);
mixed suite 250 (foundation 241, N1 215). Its long-range **values** are the worst of the chains (bias 0.74 at 0.24–0.72,
0.52 at 0.72–1.6; the targets collapse to 0.50–0.53 of T* from gap 0.48 on): the eikonal term at the endpoints enforces
|∇T| = 1/S, which on the thin orientation manifold is the straight-line rate, so it pulls the segment back toward the
blind extrapolation and fights the Bellman targets. The planners do not use those values; they follow the gradient, and
the gradient is now eikonal-consistent and points along safe directions (direction cosine 0.65 / 0.52 at mid / long
range vs 0.60 / 0.43 for the value-only chain; eikonal residual |log S|∇T|| 0.13–0.14 on every segment beyond 0.24 vs
0.5–0.7 for the value-only chain and 0.4–1.2 for the extrapolated foundation). The 0.16 edge is a 26 % step down toward
the goal (no barrier). Training 66 min (double backward), evaluation 4 min (two models).

**One long-range network with values only (N2, no-wall: the same network and targets as N8 without the eikonal and
direction terms).** The worst field of the session: RK4 463, MPPI 1447 (483/481/483), below the original method (550 /
1598); MPPI seed-0 tilt failures 110, RK4 125 (foundation 38 / 9). Against the ground truth it is as good as the
multi-model chain (bias 0.85 / 0.77, Spearman 0.76 / 0.73, direction cosine 0.65 / 0.52 at mid / long range — better
than N1's 0.60 / 0.43), so the truth-fit statistics, all measured at pool poses, do not see the failure. What the planners
see is the field **off** the pool: for 16k pool pairs (gap 0.2–0.6) and a random step of 0.03 from q, 39 % of the steps
leave the fast region (speed drop > 0.2: off the orientation manifold). On those steps T falls — the planner is rewarded
for tilting — for 7.9 % of the steps under the foundation (mean rise of T per unit step 3.2), 7.9 % / 2.6 for F-aug,
6.7 % / 2.3 for N8, 11 % / 1.4 for the original method, **24 % / 0.74 for the value-only multi chain N1 and 32 % / 0.39
for N2**: the value-regressed segments are nearly flat off the manifold (`offmanifold.py`, unregistered diagnostic). The
eikonal term restores the wall (|∇T| = 1/S with the speed network's floor off the manifold), the direction term keeps the
descent along the foundation's safe directions. On the cabinet the free space is fat (only 13 % of random 0.03 steps lower the
speed by > 0.2, i.e. approach an obstacle); there the value-only chains also weaken the wall (leaving rewarded 23–25 %,
mean rise 0.76–0.82, vs 18 % / 1.3–1.4 for the foundations and 22 % / 0.80 for the original method), without a planning
effect: the cabinet's failures are collisions inside the compartments at the foundation's own rate. Its targets
track N1's (0.70–0.74 of T* from gap 0.48), so the single-network mode is not what collapses N8's long-range values;
the eikonal term is. **Several networks or one?** With values only, one network for every gap > 0.16 is far worse
than one per segment (1447 vs 1591; each segment model is initialized from the foundation and trained on a narrow gap
range, so it keeps more of the foundation's structure off the pool); with the three-term loss, one network is the best
field of the session (N8). The loss decides, not the number of networks.

**The three-term chain on the cabinet (K8: one long-range network, EFL pool targets, 6/2/2).** Planning RK4 252, MPPI
734 (246/245/243) — the foundation's number once more (all failures collisions; per suite 285/69/75/12/24/269 vs the
foundation's 285/70/80/10/26/266). As a field it is the best long-range one of the session: Spearman 0.81 at 0.72–1.6
(K1 0.74, K3 0.72, foundation 0.15), direction cosine 0.78 / 0.71 at mid / long range (K1 0.71 / 0.60, K3 0.72 / 0.63,
foundation 0.71 / 0.52), eikonal residual 0.08–0.13 on every segment beyond 0.24 (K1 0.40–0.47), long-range bias 0.90
(the targets settle at 0.90 of T* from level 8 on: the eikonal pull is mild where the free space is fat), slow pairs
0.66–0.82 and the obstacle wall off the pool restored (leaving rewarded 17 %, mean rise 1.07, vs 23–25 % / 0.8 for the
value-only chains). Its weakness is the mid range: the single network smooths the level-1 detour inflation over
0.16–0.56 (fast pairs 1.29 / 1.18 at 0.16–0.4 / 0.4–0.56; the pool split points again). On the frozen cabinet suites
none of this moves the planners: 11 fields in this session and the foundation all sit at RK4 249–255 / MPPI 731–737
(seed spread of the foundation 734–760), and 59 % of the queries fail by collision inside the compartments with every
field.

**One long-range network with values only on the cabinet (K2).** RK4 252, MPPI 712 (238/237/237): the lowest MPPI of
the cabinet chains (K1/K4/K3/K8 731–734, foundation 737, seeds 734–760), collisions 362 of 600 seed-0 runs. Its field
is as good as the multi chain's against the truth (bias 1.08 / 1.18 / 1.14, Spearman 0.91 / 0.86 / 0.75, direction cosine
0.93 / 0.77 / 0.69 — better than K1's 0.71 / 0.60) and it weakens the obstacle wall off the pool like the other
value-only chains (leaving rewarded 26 %, mean rise 0.75). Same pattern as no-wall, attenuated: without the eikonal and
direction terms, one network for the whole long range is the worst configuration; with them (K8) it is harmless on the
cabinet and the best on no-wall.

**Tight-gate chain on no-wall (N3; evaluated).** RK4 522, MPPI 1612 (538/537/537), tilt failures 59 / 44 (MPPI / RK4):
a value-only pool-like chain, as N1 (535 / 1591). Long-range bias 0.66 (N1 0.80), Spearman 0.70, cosine 0.36.

### 5. Fine-tuning the foundation: rehearsal keeps it, forgetting destroys it

**Fine-tuning the foundation itself, with eikonal rehearsal (K5, cabinet: one network for every gap, F's weights
fine-tuned level by level on EFL pool targets with 256 short pairs under the eikonal loss in every batch; the frozen
original F still supplies the first leg).** The close-range field does not degrade — it improves: relative error 0.062 at
0–0.24 (foundation 0.088; per segment 0.062 / 0.053 / 0.070 vs 0.097 / 0.076 / 0.092), eikonal residual 0.06–0.10 below
0.24 (foundation 0.14): 17 more levels of eikonal training on short pairs is simply more training. The rehearsal also
resists the inflated pool targets (level-1 target 1.165, model 1.01; the chain settles at 0.89 of T* from level 8 on, as
K8), so the mid range is unbiased (bias 1.01 at 0.24–0.72) while the slow pairs stay low (0.58–0.69). Planning: MPPI 746
(250/250/246), the highest of the cabinet chains (731–737; foundation 737, seeds 734–760), but RK4 240 with 32 endpoint
failures (foundation 255, 3): the fine-tuned single field is slower to converge under pure gradient descent. Training 58
min; evaluation 2.6 min (one network).

**Fine-tuning the foundation with eikonal rehearsal on no-wall (N5).** RK4 564, MPPI **1710** (569/572/569) — the
highest MPPI of the session (F-aug 1703 / 1692 / 1696, σ = 0.03 1708; the three-term chain N8 1692; foundation 1675;
original method 1598), tilt failures MPPI 30 / RK4 3, mixed suite 260 (foundation 241). As on the cabinet the close-range
field improves (relative error 0.059 at 0–0.24 vs 0.091; per segment 0.044 / 0.059 / 0.072 vs 0.090 / 0.086 / 0.096), the
off-manifold wall is kept (leaving rewarded 12 %, mean rise 2.3) and the long-range values collapse with any eikonal
term on the shared network (bias 0.51 at 0.72–1.6). Caveat: N5 also receives 17 × 15,620 more eikonal updates on short
pairs than the foundation, so part of its gain may be longer eikonal training rather than the Bellman targets; a
control with the same number of eikonal-only updates follows. **Replicate with training seed 1: RK4 563, MPPI 1714
(572/571/571), tilt 27 / 4, close-range relative error 0.046 / 0.059 / 0.072** — the recipe holds across seeds (1710 / 1714
vs the pool-only foundation's 1675 / 1680 / 1681).

**The control settles it (EG16-nw-long: the pool-only eikonal foundation trained for 281,160 updates instead of 15,620,
the number of updates N5 received in total; no Bellman targets, no off-pool endpoints).** RK4 559, MPPI **1720**
(573/573/574), tilt failures MPPI 26 / RK4 4, close-range relative error 0.046 / 0.061 / 0.073, eikonal residual
0.07–0.10 — the same close-range field as N5 and a slightly higher MPPI. N5's gain over the foundation is its extra
eikonal training, not its Bellman targets; and the longer-trained foundation also beats the augmented one trained for
15,620 updates (1692–1708). **The best no-wall field of all sessions is the short-range eikonal foundation trained 18×
longer** — 37 minutes of training, one network, 1720 / 1800 (95.6 %) against the original method's 1598. Whether the
off-pool augmentation adds to the longer training was not tested (it would be the next 6-minute run).

**Fine-tuning the foundation without rehearsal (N6, no-wall).** RK4 439, MPPI 1335 (444/445/446) — far below the
original method (550 / 1598); tilt failures MPPI 143, RK4 152. The close-range field is destroyed: relative error at gaps
0–0.08 / 0.08–0.16 / 0.16–0.24 of 0.21 / 0.14 / 0.12 (foundation 0.09 / 0.09 / 0.10), eikonal residual 0.5 on every
segment including the shortest (foundation 0.18). With rehearsal (N5) the same network is the best field of the session
(1710). **Keeping the original network for the close features is essential**: either literally frozen (the chains) or
through the eikonal loss on short pairs in every batch (N5/K5); seventeen levels of value regression on longer pairs
otherwise overwrite it.

**Fine-tuning without rehearsal on the cabinet (K6).** RK4 235, MPPI 693 (231/230/232) — the worst cabinet field (original
method 688–698, foundation 737); close-range relative error 0.225 / 0.178 / 0.197 at 0–0.08 / 0.08–0.16 / 0.16–0.24
(foundation 0.097 / 0.076 / 0.092), eikonal residual 0.31 (0.14). The same forgetting as N6, expressed as collisions
(365 of 600 RK4 runs) instead of tilt.

### 6. An eikonal foundation trained with off-pool endpoints (F-aug)

**Augmented foundation (F-aug, no-wall; `foundation.py`).** The same eikonal training with half of every batch moving
one endpoint off the pool (N(0, 0.05²) per coordinate, clamped to the box; the frozen speed network evaluated at the
moved point, mean speed 0.16 — the floor) so the field is defined where off-pool split points and off-manifold motions
are read. Alone it is the best no-wall field of all sessions so far: RK4 556, MPPI 1703 (568/568/567) and, with training
seeds 1 and 2, 560 / 1692 and 558 / 1696 (mean MPPI 1697; σ = 0.03 instead of 0.05: 557 / 1708, 550 / 1688, 563 / 1703 over seeds 87358, 1, 2; σ = 0.1: 559 / 1676) vs
the pool-only foundation 553 / 1675, 551 / 1680 and 548 / 1681 (seeds 87358, 1, 2; mean 1679): **+18 MPPI and +7 RK4 on
every seed, the three augmented seeds all above the three pool-only seeds**; the fit statistics of the two foundations
are identical (the gain is off the pool),
the two-specialist eikonal field 1689/1686/1679 and the original method 550 / 1598. The gain is local: MPPI
seed-0 tilt failures 29 vs 38 and collisions 7 vs 12 — the field now rises when the arm leaves the orientation manifold,
so the terminal cost penalizes tilting samples. It does **not** learn the wrap-around cost of slow long pairs (slow-pair
median 0.37–0.43 of T*, unchanged; short-range relative error 0.108 vs 0.091): the eikonal residual at endpoints in a
thin layer around the manifold cannot encode the cost of crossing it for pairs far apart.

On the cabinet (F-aug-cab) the moved endpoints mostly stay in free space (mean speed 0.50 vs 0.16 on no-wall), the
field is unchanged against the truth (short-range relative error 0.095 vs 0.088, slow pairs 0.48–0.62) and planning
is RK4 255, MPPI 752 (253/251/248), and with training seeds 1 and 2 255 / 759 and 261 / 761 (mean MPPI 757), vs
255 / 737, 253 / 734 and 253 / 760 (seeds 87358, 1, 2; mean 744) for the pool-only foundation — +13 on average, within the
training-seed spread (the pool-only seeds range 734–760), so on the cabinet the augmentation is not a demonstrated gain;
the fit statistics of the two foundations are identical to the third decimal. The seed-87358 run gained on
free→compartment (+5), compartment→compartment (+3) and anywhere (+10), lost on compartment→free (−3) and same (−3).

**EFL pool chain on the augmented foundation (N10, no-wall).** RK4 543, MPPI 1636 (547/543/546), tilt failures 50:
between the chain on EG16 (N1: 535 / 1591, tilt 65) and its own foundation (556 / 1703, tilt 29). The better
foundation helps the chain, but value-only learned segments still lose the manifold; long-range bias 0.83 (N1 0.80).

**Three-term chain on the augmented cabinet foundation (K11 = K8 with F-aug-cab as the foundation).** RK4 254, MPPI 737
(249/243/245) — the cabinet's number again (its foundation alone: 752); long-range Spearman 0.82, cosine 0.71, bias 0.90:
indistinguishable from K8. On the cabinet the long-range field is the best of the session and the planners do not care.

**Three-term chain on the augmented no-wall foundation (N11 = N8 with F-aug-nw as the foundation, pool split points).**
RK4 567, MPPI 1706 (569/569/568), tilt failures MPPI 29 / RK4 2, mixed suite 261 (foundation 241): the two best components
of the session combined — within noise of each alone (N8 569 / 1692, F-aug 556–560 / 1692–1708, N5 564 / 1710) and
above every pool-only field (foundation 1675–1681, original method 1598). Long-range values as N8's (bias 0.55, Spearman
0.53): the eikonal pull again.

### 7. Control: value iteration without the trusted-range order

**Control: value iteration without the trusted-range order (K9, cabinet; `chain2.py --scheme vi`).** Every round draws
400k pairs of all gaps 0.16–1.6, the first leg is the foundation (≤ 0.176, pool split points) and the tail is the previous
round's field at any gap, starting from the foundation's extrapolation; 18 rounds, the chain's compute. The correct values
propagate outward one hop at a time: median model/T* over all gaps 0.76 → 0.80 (round 6) → 0.87 (10) → 0.94 (14) → 1.00
(18); per segment after 18 rounds 1.18–1.20 at 0.16–0.56 (the pool detour, as the chains), 1.03 at 0.72–0.8, 0.96 at
0.88–0.96, 0.84–0.81 at 1.36–1.6 — the frontier of correctness moved by about 0.9 in 18 rounds while the segment chains
cover 1.6 in 17 levels, and the long-range ranking is the worst of the cabinet fields (Spearman 0.60 at 0.72–1.6 vs
0.74–0.82 for the chains). Planning RK4 251, MPPI 715 (value-only single network, as K2). With an underestimating tail the
min prefers the smallest first leg; the segment order is what spends every level's compute at the frontier and reads the
tail only where it is already right. (The 2-round smoke test moved by < 1 % per round at 2k pairs.) On no-wall (N9, 18
rounds) the wave does not cross the wrap-around gap at all: per segment after 18 rounds 1.06 / 1.05 / 1.04 / 0.94 at
0.16–0.48, 0.67 / 0.62 at 0.48–0.64, falling to 0.40 at 1.6; long-range Spearman 0.32 (the chains 0.70–0.73); planning
RK4 538, MPPI 1610 (value-only single network: tilt 57 / 38).

### 8. Long cabinet queries

**Long cabinet queries (400 uncertified pairs of the suites' endpoints, gap 0.8–1.5; native RK4).** Successes /
converged: K1 23 / 104, K4 30 / 130, K3 23 / 82, K2 32 / 143, K8 29 / 94, K5 21 / 66, K11 30 / 85 (previous session,
two-specialist foundation: original method 23 / 89, symmetric DP 29 / 138). The fields with the best long-range values
(K8, K11) are not the ones that converge most (the single pool chains K2, K4), and every field succeeds on 5–8 % of the
queries: gradient descent on a 1.0-gap travel-time field through the cupboard is not how these queries get solved,
whatever the field's long-range accuracy. The foundations alone: EG16 23 / 62, F-aug-cab 25 / 67 — the chains converge
1.3–2.3× more often and succeed the same. Release MPPI (seed 0) on the first 100: K3 8, K8 10, K11 11, every failure
a collision (previous session: 8–9 for every field that converged at all). The long-range field is right (K8/K11
Spearman 0.81–0.82) and the planners still collide: for these queries the limit is the planner and the compartments,
not the travel-time values.

## Answers

**RQ1 — can the short-range eikonal field be the foundation for Bellman updates that cover longer distances?** As a
*field*: yes. Built on the frozen foundation with the trusted-range order, the Bellman extension gives the only fields
that know which long pairs are slow (Spearman 0.72–0.82 at gaps 0.72–1.6 vs 0.15 / 0.03 for the foundation), and with
the eikonal and direction terms on the learned segments it also has eikonal-consistent, well-directed gradients there.
Its absolute values carry the bias of the split-point construction (pool: +12–20 % on direct pairs; continuous: slow pairs
at 0.6–0.7 of T*), and on the thin no-wall manifold the eikonal term costs the long-range values. As a *planner's field*
on the frozen (short) suites the extension is neutral on the cabinet and, on no-wall, harmful with values only and
helpful with the three-term loss (RK4 569, MPPI 1692–1706 vs the foundation's 553 / 1675) — about as helpful as the
augmented foundation alone (556–560 / 1692–1708) or the rehearsed fine-tuning (563–564 / 1710–1714 on two seeds), and
no more helpful than simply training the short-range eikonal foundation 18× longer (559 / 1720): on these suites the
Bellman extension has not yet bought anything that more eikonal training does not. Long cabinet queries (gap 0.8–1.5)
stay at 5–8 % success with every field, so the correct long-range values do not yet reach the planner.

**Subtask 1 — curriculum, frozen foundation, several networks or one.** The curriculum order (trusted-range tail) is
necessary: value iteration without it crawls (one hop per two rounds; stuck at the wrap-around gap on no-wall). The frozen
foundation is necessary unless the eikonal loss on short pairs is rehearsed in every batch, which then improves it. Several
networks vs one: the loss decides — with values only, several (one per segment) is far safer; with value + eikonal +
direction, one long-range network is the best and the cheapest to evaluate (2.6–4 min vs 4–7 for 19 models).

**Subtask 2 — pitfalls and what was tried on top.** Pitfalls, each with its diagnostic: (1) the detour bias of pool split
points, inherited by every level of the eikonal-first-leg chain; (2) the foundation's blind spots off the pool, found by any
continuous minimization of the two-leg sum, not removed by validity gates or by training the foundation with off-pool
endpoints, and only avoided by split points on the pool; (3) the winner's curse of the min, ~1/3 of the gated drift (double
estimator); (4) value-regressed segments flatten the field off the free manifold — the mechanism behind every no-wall loss
— invisible to every truth-fit statistic measured at pool poses; (5) the endpoint eikonal term re-imposes the straight-line
extrapolation on long pairs; (6) catastrophic forgetting when the foundation is fine-tuned without rehearsal; (7) the
cabinet's frozen suites do not respond to any field change (collisions inside the compartments at the foundation's rate).
On top: the three-term loss with eikonal-field actions (N8/K8/N11/K11), the double estimator (K7), the augmented
foundation (F-aug; +18 MPPI at equal training), rehearsed fine-tuning (N5/K5; 1710–1714) — and the control that puts them
in perspective: the foundation alone, trained for the same number of eikonal updates, reaches 1720.

## Caveats

- One training seed per chain (87358). The foundations have three seeds each (cabinet pool-only 734–760, augmented
  752–761; no-wall pool-only 1675–1681, augmented 1692–1703), so the chain differences below ±15 MPPI on the cabinet are
  within seed noise; the no-wall differences discussed (1335 / 1447 / 1591 / 1612 / 1636 / 1692 / 1710 vs 1675) are not.
- The ground truth T* is an upper bound (graph route + path optimization, segment_dp/truth.py); every "slow pair" statement
  compares fields against the same reference. The direct/detour and fast/slow splits, the off-manifold statistics and the
  segment-edge statistics are unregistered diagnostics (`wrap.py`, `offmanifold.py`, `edges.py`) run on registered fields.
- The gated chains used 200k (cabinet) / 150k (no-wall) pairs per level, the pool chains 400k; the double-estimator and
  tight-gate runs stop at gap 0.8 and are not evaluated on the planners.
- The long cabinet queries are uncertified; MPPI on them was run for three fields and 100 queries with one planning seed.
- Docker backend `bellman600-env:local` on the user's workstation (modified environment relative to the pinned image,
  recorded in every run's environment_override.json). Nothing committed to Git; no frozen asset or baseline was replaced.

## Files and commands

`chain.py` (the trainer: schemes, model modes, split points, gates, estimator, losses, per-level truth diagnostics),
`chain2.py` (chain.py + `--scheme vi`), `foundation.py` (augmented eikonal foundation), `pipeline.py` / `pipeline2.py`,
`run.py` / `run2.py` (registry entries; stages through `../segment_dp/run_stage.py`), `report.py` (tables and run index
from the queue state files), `wrap.py`, `offmanifold.py`, `edges.py` (unregistered diagnostics).

```bash
export BELLMAN600_DERIVED_IMAGE=bellman600-env:local BELLMAN600_DOCKER_MEMORY=7G
# EFL chain, one model per segment, pool split points (K1 / N1)
python3 jeffrey/eikonal_bellman/run.py --scene cabinet --base-run registered_20261027T230747Z-7999899d \
    --truth-run registered_20261027T175501Z-5150664b --scheme efl --models multi --split pool --pairs 400000 --label K1 --gpu 1
# gated continuous split points with the exact clearance gate (K3); double estimator to gap 0.8 (K7)
python3 jeffrey/eikonal_bellman/run.py --scene cabinet ... --split gated --clearance --pairs 200000 --label K3 --gpu 1
python3 jeffrey/eikonal_bellman/run.py --scene cabinet ... --split gated --clearance --estimator double --last-edge 0.8 --no-evaluate --label K7
# one long-range network with the three-term loss (N8); fine-tuned foundation with rehearsal (N5)
python3 jeffrey/eikonal_bellman/run.py --scene nowall --base-run registered_20261027T231531Z-bda27d55 \
    --truth-run registered_20261027T181431Z-bf1c963f --scheme efl --models single --split pool --weights 6,2,2 --label N8 --gpu 0
python3 jeffrey/eikonal_bellman/run.py --scene nowall ... --models finetune --rehearsal 256 --label N5 --gpu 0
# augmented foundation (F-aug) and the vi control (chain2.py)
python3 jeffrey/eikonal_bellman/run2.py --scene nowall --trainer foundation --augment 0.5 --sigma 0.05 --label F-aug-nw --gpu 0
python3 jeffrey/eikonal_bellman/run2.py --scene cabinet --base-run registered_<EG16> --truth-run registered_<truth> --scheme vi --models single --label K9 --gpu 1
# diagnostic stages (segment_dp/run_stage.py): fit against the truth, long cabinet queries
python3 jeffrey/segment_dp/run_stage.py fit --truth-run registered_<truth> --runs registered_<id>=K1 --gpu 1
python3 jeffrey/segment_dp/run_stage.py long --runs registered_<id>=K8 ... --count 100 --mppi-seeds 0 --gpu 1
```

## Run index

|Scene/label|Run|Kind|
|---|---|---|
|cabinet/K1|`20261028T060044Z-1b2cd617`|training + evaluation|
|cabinet/fit-K1|`20261028T063941Z-81240f9b`|fit stage|
|cabinet/K4|`20261028T064011Z-0badcab1`|training + evaluation|
|cabinet/fit-K4|`20261028T072202Z-f48f02ad`|fit stage|
|cabinet/F-aug-cab|`20261028T072205Z-215d6df6`|training + evaluation|
|cabinet/fit-F-aug-cab|`20261028T072656Z-d47a0f74`|fit stage|
|cabinet/K3|`20261028T072658Z-2c5bc107`|training + evaluation|
|cabinet/fit-K3|`20261028T084224Z-64107440`|fit stage|
|cabinet/K7|`20261028T084227Z-3f98999c`|training only (diagnostic, no evaluation)|
|cabinet/F-aug-cab-s1|`20261028T092108Z-0758c67b`|training + evaluation|
|cabinet/F-aug-cab-s2|`20261028T092556Z-e4be5cd6`|training + evaluation|
|cabinet/EG16-cab-s1|`20261028T093047Z-af3b9cfd`|training + evaluation|
|cabinet/EG16-cab-s2|`20261028T093534Z-f1a82adc`|training + evaluation|
|cabinet/fit-F-aug-cab-variants|`20261028T094016Z-66ccbeb0`|fit stage|
|cabinet/K8|`20261028T094019Z-6e3b27e1`|training + evaluation|
|cabinet/fit-K8|`20261028T103620Z-c1757f1f`|fit stage|
|cabinet/K3b|`20261028T103623Z-e8a223d8`|training only (diagnostic, no evaluation)|
|cabinet/fit-K3b|`20261028T110755Z-1087b50d`|fit stage|
|cabinet/K2|`20261028T110758Z-8c9b7a7d`|training + evaluation|
|cabinet/fit-K2|`20261028T114429Z-5badee4c`|fit stage|
|cabinet/K5|`20261028T114431Z-e2c9d748`|training + evaluation|
|cabinet/fit-K5|`20261028T124451Z-f4a3e995`|fit stage|
|cabinet/K11|`20261028T124454Z-da2b8c04`|training + evaluation|
|cabinet/fit-K11|`20261028T134018Z-21faf2af`|fit stage|
|cabinet/long-rk4|`20261028T134021Z-f5c5fc3b`|long-query stage|
|cabinet/EG16|`20261027T230747Z-7999899d`|training + evaluation|
|cabinet/long-mppi|`20261028T134305Z-73c9d0d6`|long-query stage|
|cabinet/long-rk4-found|`20261028T134713Z-6f1efded`|long-query stage|
|cabinet/K9|`20261028T134735Z-3e2db9dd`|training + evaluation|
|cabinet/fit-K9|`20261028T142526Z-16d85eb0`|fit stage|
|cabinet/K6|`20261028T142528Z-ff3e330d`|training + evaluation|
|cabinet/fit-K6|`20261028T150024Z-8d12e8f3`|fit stage|
|cabinet/K10|`20261028T150027Z-95cee985`|training only (diagnostic, no evaluation)|
|cabinet/N5s1|`20261028T151624Z-dd46a19a`|training + evaluation|
|cabinet/fit-N5s1|`20261028T161742Z-cf22b816`|fit stage|
|cabinet/F-aug-nw-sig03-s1|`20261028T161927Z-96e59fd1`|training + evaluation|
|cabinet/F-aug-nw-sig03-s2|`20261028T162455Z-71a65861`|training + evaluation|
|cabinet/fit-F-aug-sig03-seeds|`20261028T163014Z-cc9a8c3d`|fit stage|
|nowall/N1|`20261028T060046Z-fe366555`|training + evaluation|
|nowall/fit-N1|`20261028T065700Z-b0c8fb57`|fit stage|
|nowall/F-aug-nw|`20261028T065713Z-47af7f84`|training + evaluation|
|nowall/fit-F-aug-nw|`20261028T070320Z-8329958d`|fit stage|
|nowall/N10|`20261028T070322Z-a13dac12`|training + evaluation|
|nowall/fit-N10|`20261028T075729Z-76fb27a0`|fit stage|
|nowall/F-aug-nw-s1|`20261028T075732Z-054bf0f4`|training + evaluation|
|nowall/N8|`20261028T080339Z-8b21befd`|training + evaluation|
|nowall/fit-N8|`20261028T091332Z-6d03fe13`|fit stage|
|nowall/N2|`20261028T091335Z-90d51560`|training + evaluation|
|nowall/fit-N2|`20261028T100324Z-9fd9a12b`|fit stage|
|nowall/F-aug-nw-s2|`20261028T100327Z-09201c22`|training + evaluation|
|nowall/F-aug-nw-sig03|`20261028T100920Z-0a52952f`|training + evaluation|
|nowall/F-aug-nw-sig10|`20261028T101525Z-d0c11ca1`|training + evaluation|
|nowall/EG16-nw-s1|`20261028T102121Z-88552164`|training + evaluation|
|nowall/EG16-nw-s2|`20261028T102711Z-f1b02e94`|training + evaluation|
|nowall/fit-F-aug-variants|`20261028T103302Z-6fc4f754`|fit stage|
|nowall/N3|`20261028T103305Z-4e0f669c`|training + evaluation|
|nowall/fit-N3|`20261028T120323Z-e1c9a5ed`|fit stage|
|nowall/N5|`20261028T120326Z-b1a70fb7`|training + evaluation|
|nowall/fit-N5|`20261028T131837Z-d33bda49`|fit stage|
|nowall/N6|`20261028T131840Z-fe05d927`|training + evaluation|
|nowall/fit-N6|`20261028T140630Z-390aac65`|fit stage|
|nowall/N9|`20261028T140633Z-f2a44ffa`|training + evaluation|
|nowall/fit-N9|`20261028T145928Z-ee53da8c`|fit stage|
|nowall/N11|`20261028T145931Z-777ea691`|training + evaluation|
|nowall/fit-N11|`20261028T160923Z-faec7a0d`|fit stage|
|nowall/EG16-nw-long|`20261028T160926Z-508b4d62`|training + evaluation|
|nowall/fit-EG16-nw-long|`20261028T164932Z-6949635e`|fit stage|
