# jeffrey/gap_curriculum: interval-DP (gap) curriculum for the Bellman TravelCost

*2026-10-27, one session of ~10 h on the RTX 3090 (cabinet) and RTX 5070 (no-wall). Tables are generated from the
registered runs (run index at the end); one training seed unless a row lists several.*

## Summary

- **10-round baseline (B10)** — cabinet RK4 243/600, release MPPI 718/1800 (5 rounds: 229/688, reproduced flag
  for flag on this machine); no-wall 543/600, 1611/1800 (5 rounds: 550/1598). Over training seeds 87358/1/2/3, B10
  gives cabinet MPPI 718/725/707/723 and no-wall 1611/1617/1613/1624. More rounds alone stop helping: B20 gives 721
  and 1629.
- **The gap curriculum as specified (C10, round k trains |q−g| ≤ 0.2k)** — a small gain that holds on every seed
  on the cabinet (MPPI +12/+7/+21, RK4 +2/+9/+26) and is marginal on no-wall (MPPI +11/+1/+8, RK4 ≈ 0). Its rounds
  7–10 already contain ≈ 97 % of all pairs, so they are baseline rounds and the short gaps trained first fade.
- **Keeping the introduced radii (G10: round k draws each pair's radius from r_1..r_k)** — a large gain on every
  seed: cabinet MPPI 757/777/784/749 (mean 766.8, +49 over B10), RK4 268.5 (+31); no-wall MPPI 1664/1660/1655/1662
  (1660.2, +44), RK4 unchanged. With 20 rounds (G20, r_k = 0.1k) it keeps improving where the baseline does not:
  cabinet 820/760/784 (mean 788, RK4 271), no-wall 1685/1676/1680 (1680).
- **The order is not what helps the planners.** W10 trains every round on G10's time-averaged gap mixture with no
  order: cabinet MPPI 776/768/774/756 (mean 768.5 vs G10's 766.8), no-wall 1665/1669/1664/1655 (1663.2 vs 1660.2).
  At 20 rounds the matched unordered W20g gives cabinet 786/785/773 (781.3 vs G20's 788.0) and no-wall
  1676/1666/1680 (1674.0 vs 1680.3). What raises success is how much of the run is spent on short gaps, not when.
- **Why.** The frozen queries are short (start–goal 0.02–0.68, median ≈ 0.25 normalized), natural training pairs
  are long (median 0.99), and all three losses are enforced only on the training pairs: short-gap-heavy training
  fits the eikonal equation 2.5–5× better below gap 0.2. Moving *only* the Eikonal term onto the curriculum's pairs
  reproduces the whole gain; moving only Bellman + direction gives a third of it. And the interval-DP premise does not hold
  for this bank: the Bellman min takes the 0.01 step in 97–99.7 % of backups, so each target comes from a pair of
  the *same* gap, and safe actions that make real progress toward a goal are rare.
- **Where the order does matter: the long-range field (cabinet, 10 rounds).** Against a graph ground truth
  (speed-weighted kNN Dijkstra over the bank), the ordered curricula rank slow long pairs (gap ≥ 1.0) far better
  (Spearman C10 0.22–0.28, G10 0.19–0.25) than any unordered run (W10 0.08–0.10, M10 0.06–0.07, B10 0.02–0.03): long
  pairs are never fitted to crude early targets. It does not translate into planning: on 400 uncertified long cabinet
  queries the ordered runs gain ~3 RK4 successes and nothing with MPPI; on no-wall no field ranks long pairs, and at
  20 rounds the difference shrinks (G20 0.16/0.20 vs W20g 0.14/0.14).
- **Practical outcome**: a short-gap-heavy goal mixture — ordered (G) or not (W) — is a pure sampling change worth
  +26 to +77 release MPPI per training seed on the cabinet (mean ≈ +49) and +31 to +54 on no-wall (≈ +45) at 10
  rounds, and ≈ +60 / +45–50 over the 20-round baseline; cabinet RK4 ≈ +30, no-wall RK4 unchanged. Denser
  short-gap mixtures do better still at 10 rounds: cabinet W10a (half the pairs from the 0.2-ball) 790/770/778
  (mean 779, +45 to +72 per seed); no-wall W10b (with 0.1-balls) 1688/1677/1673 (mean 1679, +60 to +77) with RK4
  563/554/550 (+19). Best single runs: cabinet G20 / G30 820, no-wall W10b 1688 (93.8 %).

## Idea (user, 2026-10-27)

Interval DP fills `dp[i][j]` (travel time from configuration i to j) by increasing gap, so every entry is built from
entries of smaller gap that are already final. The Bellman TravelCost is trained against a **frozen** copy of the
previous round, which suits the same order: round k trains only pairs `(q, g)` with `|q - g| <= r_k` (normalized
joint units), so the Bellman tail `T_{k-1}(successor, g)` is queried where the previous round has already fitted,
and `r_k` grows until the last round covers every pair. First a 10-round baseline (the clean recipe with 10 instead
of 5 rounds), then the curriculum with 10 rounds.

## What changes and what does not

`training.py` is `experiments/from_source/training.py` with one change: how the goal of each pair is drawn.
Unchanged: the 338,177-parameter TravelCost, the three losses 6/2/2 (`method/loss.py`, imported), AdamW 2e-4,
weight decay 1e-6, clip 2, batch 256, 2 epochs per round, the target network frozen per round and refreshed at the
round's end, and the source permutation and natural goal draws from the same numpy generator (a curriculum draws
and discards the natural goals, so every rule visits the same sources in the same order). `--goals natural
--rounds 5` reproduces the baseline trainer bit-for-bit (smoke checkpoint hash equal). Training seed 87358 is
primary; seeds 1, 2 (3) are replicates. The evaluated model is always the **last** round's checkpoint.

|Name|Goal rule (round k of 10, `r_k = 0.2 k`)|
|---|---|
|B10 (baseline)|goal uniform over the bank sources (median gap 0.99)|
|C10 (the curriculum)|goal uniform over the bank sources within `r_k` of the source (never the source or a duplicate)|
|M10 (control)|every pair draws its radius uniformly from all of `r_1..r_10`, in every round: C10's mixture over the run, no order|
|G10 (curriculum with rehearsal)|every pair draws its radius uniformly from `r_1..r_k`: C10's order, the radii already introduced keep equal shares, round 10 is M10's distribution|
|W10 (control for G10)|every round the radius is drawn from `{0.2, 0.2, 0.2, 0.4, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0}`: G10's time-averaged mixture (mean training gap 0.50 vs 0.48), no order|

Further variants (§ Variants): 5 and 20 rounds, `--min-step 0.05` (Bellman backup and winner direction only over
safe actions of step ≥ 0.05 — a mechanism test that changes the target), `--split` (which loss terms see the
curriculum pairs), a uniform-gap mixture (`shell`) and a travel-time order (`value`, Dijkstra-like).
Ball goals are drawn exactly (whole pool, or a random pool subset first with a whole-pool fallback, which keeps
every in-ball source equally likely).

## Protocol

- **cabinet**: speed network and offline bank of the registered cabinet pipeline `registered_20261019T034213Z-f9c501f9`
  (399,581 sources × 32 actions); the frozen certified cabinet suites (6 × 100); native RK4 and the release MPPI
  (original settings, planning seeds 0/1/2), release scorer in the cabinet scene, tilt gate off. RTX 3090.
- **nowall**: the frozen assets (`assets/bank`, `assets/speed.pt`) and the frozen 600 queries (`method/evaluate.py`,
  scorer tilt 20°). RTX 5070.
- Host backend (bwrap + `~/.venvs/bellman-ntfield`, torch 2.7.1+cu128), a modified environment relative to the
  pinned image; every run records `environment_override.json`. The 5-round cabinet rerun here reproduces every
  per-query success flag of the original pinned-image run (229/600, 688/1800); the no-wall 5-round rerun gives
  550/600, 1598/1800 (frozen reference 548–549, 1597).
- MPPI planning seeds 0/1/2 are planner seeds and give nearly identical results. Training seeds are the noise that
  matters: B10 spans cabinet MPPI 707–725 and no-wall 1611–1617; the cabinet mixtures vary more (M10 720–789).

## Results

### Curriculum, controls and training seeds

Every round is 2 epochs (no-wall 3,124, cabinet 3,122 updates), so 10 rounds are 31,240 / 31,220 updates and 20 rounds
twice that. Release MPPI sums planning seeds 0/1/2. The Δ column pairs runs with the same training seed (same
initialization and source order) against B10 (10 rounds) or B20 (20 rounds).

**cabinet, 10 rounds** (RK4 /600, release MPPI /1800; training seeds 87358, 1, 2, 3)

|Variant|RK4 per seed|RK4 mean|MPPI per seed|MPPI mean|Δ MPPI vs B10 / B20 (same seed)|
|---|---|---:|---|---:|---|
|B5|229|229.0|688|688.0|—|
|B10|243, 240, 227, 240|237.5|718, 725, 707, 723|718.2|—|
|C10|245, 249, 253|249.0|730, 732, 728|730.0|+12, +7, +21|
|M10|253, 249, 262|254.7|740, 720, 789|749.7|+22, -5, +82|
|G10|263, 275, 264, 272|268.5|757, 777, 784, 749|766.8|+39, +52, +77, +26|
|W10|271, 265, 270, 262|267.0|776, 768, 774, 756|768.5|+58, +43, +67, +33|
|W10a|262, 269, 261|264.0|790, 770, 778|779.3|+72, +45, +71|
|W10b|277, 271, 262, …|270.0|773, 764, 777, …|771.3|+55, +39, +70, —|

**nowall, 10 rounds** (RK4 /600, release MPPI /1800; training seeds 87358, 1, 2, 3)

|Variant|RK4 per seed|RK4 mean|MPPI per seed|MPPI mean|Δ MPPI vs B10 / B20 (same seed)|
|---|---|---:|---|---:|---|
|B5|550|550.0|1598|1598.0|—|
|B10|543, 533, 538, 534|537.0|1611, 1617, 1613, 1624|1616.2|—|
|C10|540, 531, 540|537.0|1622, 1618, 1621|1620.3|+11, +1, +8|
|M10|532, 535, 528|531.7|1649, 1650, 1659|1652.7|+38, +33, +46|
|G10|543, 541, 530, 547|540.2|1664, 1660, 1655, 1662|1660.2|+53, +43, +42, +38|
|W10|535, 523, 524, 540|530.5|1665, 1669, 1664, 1655|1663.2|+54, +52, +51, +31|
|W10a|528, …, …|528.0|1660, …, …|1660.0|+49, —, —|
|W10b|563, 554, 550, 545|553.0|1688, 1677, 1673, 1678|1679.0|+77, +60, +60, +54|

**cabinet, 20 rounds** (RK4 /600, release MPPI /1800; training seeds 87358, 1, 2, 3)

|Variant|RK4 per seed|RK4 mean|MPPI per seed|MPPI mean|Δ MPPI vs B10 / B20 (same seed)|
|---|---|---:|---|---:|---|
|B20|240|240.0|721|721.0|—|
|W20|258|258.0|785|785.0|+64|
|G20|280, 270, 264|271.3|820, 760, 784|788.0|+99, —, —|
|W20g|268, 269, 267|268.0|786, 785, 773|781.3|+65, —, —|

**nowall, 20 rounds** (RK4 /600, release MPPI /1800; training seeds 87358, 1, 2, 3)

|Variant|RK4 per seed|RK4 mean|MPPI per seed|MPPI mean|Δ MPPI vs B10 / B20 (same seed)|
|---|---|---:|---|---:|---|
|B20|535|535.0|1629|1629.0|—|
|W20|521|521.0|1657|1657.0|+28|
|G20|549, 531, 525|535.0|1685, 1676, 1680|1680.3|+56, —, —|
|W20g|547, 535, 543|541.7|1676, 1666, 1680|1674.0|+47, —, —|

Release MPPI per suite, mean over the training seeds:

**cabinet: release MPPI per suite /300, mean over training seeds**

|Variant (seeds)|f→f|f→c|c→f|c→c|same c|any|
|---|---:|---:|---:|---:|---:|---:|
|B5 (1)|268|66|63|3|24|264|
|B10 (4)|274|78|67|3|31|266|
|C10 (3)|277|75|73|4|31|270|
|M10 (3)|281|79|84|3|25|278|
|G10 (4)|288|84|80|11|30|274|
|W10 (4)|287|85|81|6|33|277|
|W10a (3)|288|92|85|7|27|281|
|W10b (3)|284|89|79|13|30|276|

**nowall: release MPPI per suite /300, mean over training seeds**

|Variant (seeds)|mixed|scr|cut|ham|stir|wipe|
|---|---:|---:|---:|---:|---:|---:|
|B5 (1)|207|267|291|276|278|279|
|B10 (4)|217|272|291|282|279|275|
|C10 (3)|225|269|291|281|279|275|
|M10 (3)|237|280|292|286|280|277|
|G10 (4)|244|278|292|288|282|276|
|W10 (4)|240|280|290|289|282|281|
|W10a (1)|239|279|292|294|282|274|
|W10b (4)|248|283|293|291|283|280|

### Per round (diagnostic)

Native RK4 of every round's checkpoint (`rounds.py`; never used to pick a model). A curriculum's first rounds train
only short pairs, so it plans better early; the unordered W10 catches up with G10 by round 2 (cabinet) / round 4
(no-wall). At 20 rounds G20 stays above W20g for most rounds on this seed, but the final MPPI order flips on seed 1.

|Run|r1|r2|r3|r4|r5|r6|r7|r8|r9|r10|r11|r12|r13|r14|r15|r16|r17|r18|r19|r20|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|cabinet/B10|208|213|217|228|229|230|235|244|246|243|||||||||||
|cabinet/C10|233|231|242|242|239|239|249|252|248|245|||||||||||
|cabinet/G10|235|235|239|250|252|254|253|259|262|263|||||||||||
|cabinet/G20|245|255|257|268|260|270|265|268|266|276|280|280|272|278|277|275|280|284|268|280|
|cabinet/M10|212|234|237|237|242|246|253|251|251|253|||||||||||
|cabinet/W10|219|235|241|251|252|246|256|264|260|271|||||||||||
|cabinet/W20g|227|242|247|249|252|260|260|264|260|258|274|270|272|269|271|278|269|272|259|268|
|nowall/B10|446|491|518|540|550|542|541|539|544|543|||||||||||
|nowall/C10|476|525|539|545|537|541|541|541|539|540|||||||||||
|nowall/G10|478|532|543|555|554|547|543|550|541|543|||||||||||
|nowall/M10|434|500|529|548|546|547|545|541|538|532|||||||||||
|nowall/W10|433|502|527|544|541|546|540|530|530|535|||||||||||


## Mechanism

### 1. The test queries live at short gaps; natural training pairs do not

Start–goal distance of the frozen queries (normalized joint units): cabinet 0.06–0.57 (median ≈ 0.27), no-wall
0.02–0.68 (median ≈ 0.22). Natural training pairs (two random bank sources): median 0.99, 1 % below 0.38, 5 %
below 0.56. All three losses are evaluated on the training pairs, so the baseline fits the Bellman, Eikonal and
direction terms almost only at gaps longer than any test query, and the planners follow a field whose short-range
part is extrapolated.

Round-10 field statistics on 8,192 fixed probe pairs (`diagnose.py`; Eikonal residual = mean of
log(S(q)·|∇_q T|)², Bellman |T − BT|/BT with the field as its own tail), training seed 87358:

|Cabinet|Eikonal, gap < 0.1|0.1–0.2|0.2–0.3|Bellman, gap < 0.1|0.1–0.2|T/|q−g| < 0.1|0.2–0.3|1.4–2|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|B10|0.252|0.142|0.087|0.186|0.055|1.38|1.39|1.20|
|C10|0.206|0.120|0.099|0.162|0.046|1.61|1.57|1.29|
|M10|0.093|0.048|0.054|0.131|0.035|2.22|1.56|1.21|
|G10|0.095|0.052|0.053|0.136|0.034|2.36|1.57|1.25|
|W10|0.089|0.044|0.053|0.136|0.035|2.44|1.60|1.22|

|No-wall|Eikonal, gap < 0.1|0.1–0.2|0.2–0.3|Bellman, gap < 0.1|0.1–0.2|T/|q−g| < 0.1|0.2–0.3|1.4–2|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|B10|0.411|0.178|0.112|0.222|0.062|1.44|1.36|1.21|
|C10|0.355|0.163|0.105|0.204|0.057|1.58|1.45|1.27|
|M10|0.138|0.067|0.073|0.140|0.045|2.45|1.47|1.20|
|G10|0.102|0.061|0.069|0.138|0.047|2.32|1.42|1.22|
|W10|0.126|0.066|0.077|0.145|0.041|2.78|1.51|1.22|

The rules that keep short pairs in every round (M10, G10, W10) fit the eikonal equation 2.5–4× better below 0.2
and are more Bellman-consistent there; their short-range travel time rises to the slowness the speed field implies
(most bank sources are within 5 cm of a surface, speed < 1). C10 trains short pairs only early: rounds 7–10 of C10
already contain ≈ 97 % of all pairs, so they are baseline rounds and the short gaps fall back to their natural
share. B10's training-seed replicates give the same statistics to ±0.01, so these differences are not seed noise.

### 2. Why the gap order itself does not help the planners

Interval DP works because `dp[i][j]` depends only on strictly smaller gaps. The Bellman backup over this offline
bank does not have that structure:

- **The min almost always takes the 0.01 step.** With the round-10 fields as the tail, the winning action is a
  0.01-step in 97–99 % of cabinet backups and 99.3–99.7 % of no-wall backups, at every gap; the successor is on
  average 0.005 (cabinet) / 0.003 (no-wall) closer to the goal. Because the fields underestimate the travel time
  (§ 3), every action costs more than it saves and the smallest step costs least. The backup is a local,
  eikonal-like condition between `(q, g)` and `(q + 0.01 u, g)` — pairs of the *same* gap — not a recurrence over
  smaller subproblems, so training short gaps first does not make the long-gap targets "final".
- **Progress actions are rare.** A safe action that brings the source ≥ 0.05 closer to a random goal exists in 2–5 %
  of no-wall rows (9 of 32 actions are safe, almost all of them 0.01/0.05 steps) and 15–60 % of cabinet rows;
  ≥ 0.1 closer: 0.3–1.5 % and 2–33 %.

Per-round RK4 shows what the order does do: the first round of a curriculum trains only short pairs and its
round-1 field plans better (cabinet 233–235 vs 208–219, no-wall 476–478 vs 433–446), but the matched unordered
mixture W10 is level with G10 from round 2 (cabinet) / round 4 (no-wall), and by round 10 the rehearsal mixtures
(G10, W10) lead.

### 3. Ground truth: the order does improve the long-range field

`truth.py` (diagnostic): a 24-nearest-neighbour graph over the bank sources, edge cost = trapezoid ∫ ds / speed
(9 points); Dijkstra from 64 seeded goals gives the travel time D, the same graph with Euclidean edge costs gives E,
so D/E is the travel time per unit length with the graph zig-zag divided out (no exact collision veto: a common
reference, not the true optimum). ~16k probe pairs, half within 0.7 of the goal.

|Cabinet|D/E (reference)|T/|q−g| B10|C10|G10|W10|Spearman(T/|q−g|, D/E) B10|C10|G10|G10 s1|W10|M10|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|gap 0.1–0.2|1.35|1.20|1.42|1.41|1.48|0.77|0.79|0.82|0.85|0.73|0.74|
|0.2–0.3|1.47|1.27|1.42|1.38|1.40|0.73|0.77|0.77|0.78|0.79|0.76|
|0.5–0.7|1.94|1.23|1.35|1.27|1.26|0.26|0.27|0.26|0.26|0.24|0.24|
|1.0–1.4|1.96|1.19|1.30|1.25|1.22|0.03|0.22|0.21|0.19|0.08|0.06|
|1.4–2.0|2.08|1.20|1.29|1.24|1.22|0.02|0.28|0.25|0.21|0.10|0.07|

All fields underestimate the travel time, more so at long range (value iteration from the 1.05 |q − g|
initialization rises only by a tiny-step increment per round; T/|q−g| grows every round and 10 rounds beat 5). At
short range the short-gap-heavy fields sit closest. At **long** range the *ordered* curricula (C10, G10) rank which
pairs are slow 2.5–3× better than any unordered run: long pairs are never fitted to crude early targets, and what
the network learned about slow regions from short pairs carries over (C10 after round 1, trained on gaps ≤ 0.2
only, already ranks long pairs better than B10 after 10 rounds: 0.09/0.04 vs 0.03/0.02). This is the DP-like effect
of the order — on the no-wall bank no field ranks long pairs at all (Spearman ≈ 0 at gaps ≥ 0.7 for every run).

The frozen suites contain no long queries. On 400 **uncertified** long cabinet queries (pairs of the suites'
validated endpoints with gap 0.8–1.5; `long.py`), native RK4 succeeds rarely for every field (mostly no convergence
in 400 steps). The ordered runs are ~3 queries ahead with RK4 and level with MPPI (planning seed 0):

|Field|RK4 /400|MPPI seed 0 /400|Stage run|
|---|---:|---:|---|
|B5|23|27|`20261027T063025Z-43c2957d`|
|B10|26|29|`20261027T063025Z-43c2957d`|
|B10s1|26|27|`20261027T063025Z-43c2957d`|
|B10s2|27|29|`20261027T063025Z-43c2957d`|
|C10|29|29|`20261027T063025Z-43c2957d`|
|C10s1|31|30|`20261027T063025Z-43c2957d`|
|C10s2|29|27|`20261027T063025Z-43c2957d`|
|M10|25|29|`20261027T063025Z-43c2957d`|
|M10s1|21|28|`20261027T063025Z-43c2957d`|
|G10|29|30|`20261027T063025Z-43c2957d`|
|G10s1|30|31|`20261027T063025Z-43c2957d`|
|W10|28|31|`20261027T063025Z-43c2957d`|


## Variants

|Variant|Rule|Cabinet RK4 /600|Cabinet MPPI /1800|No-wall RK4 /600|No-wall MPPI /1800|
|---|---|---:|---:|---:|---:|
|B10ms|natural goals, backup only over steps ≥ 0.05|248|750|498|1555|
|G10ms|G10, backup only over steps ≥ 0.05|258|765|493|1630|
|W10ms|W10, backup only over steps ≥ 0.05|260|769|229|1614|
|B20|natural goals, 20 rounds × 2 epochs|240|721|535|1629|
|G20|grow-mix, 20 rounds (r_k = 0.1k)|280|820|549|1685|
|W20|W10 mixture (0.2 ×6, 0.4 ×4, 0.6 ×3, 0.8 ×2, 1.0 ×2, 1.2, 1.4, 2.0), 20 rounds, no order|258|785|521|1657|
|W20g|G20 time-averaged mixture (0.1 ×4, 0.2 ×2, 0.3 ×2, 0.4 ×2, 0.5 ×2, 0.6 … 1.0, 1.2, 1.3, 1.5), 20 rounds, no order|268|786|547|1676|
|G20s1|G20, training seed 1|270|760|531|1676|
|G20s2|G20, training seed 2|264|784|525|1680|
|W20gs1|W20g, training seed 1|269|785|535|1666|
|W20gs2|W20g, training seed 2|267|773|543|1680|
|G30|grow-mix, 30 rounds (r_k = 0.0667k)|271|820|…|…|
|G5|grow-mix, 5 rounds (r_k = 0.4k)|245|722|545|1625|
|E10eik|split: Bellman + direction on natural goals, Eikonal on G10 goals|255|764|534|1665|
|E10bel|split: Bellman + direction on G10 goals, Eikonal on natural goals|247|734|535|1633|
|U10|gap uniform in [0, 2] every round (shell), no order|256|751|553|1672|
|V10|travel-time order: goals with T_{k-1}(q, g) ≤ 0.28k (Dijkstra-like)|242|718|539|1618|
|W10a|mixture 0.2 ×5, 0.4 ×2, 0.6, 1.0, 2.0|262|790|528|1660|
|W10b|mixture 0.1 ×3, 0.2 ×2, 0.4 ×2, 0.6, 1.0, 2.0|277|773|563|1688|
|W10bs1|W10b, training seed 1|271|764|554|1677|
|W10bs2|W10b, training seed 2|262|777|550|1673|
|W10bs3|W10b, training seed 3|…|…|545|1678|
|W10as1|W10a, training seed 1|269|770|…|…|
|W10as2|W10a, training seed 2|261|778|…|…|
|W20b|W10b-type mixture (0.1 ×6, 0.2 ×4, 0.4 ×4, 0.6 ×2, 1.0 ×2, 2.0 ×2), 20 rounds|271|795|544|1689|

- **More rounds.** The baseline saturates at 10 rounds (B20: cabinet 721 vs B10 718, no-wall 1629 vs 1616), but the
  short-gap-heavy rules keep improving with 20 rounds (cabinet G20 / W20g means 788 / 781, no-wall 1680 / 1674), so
  at 20 rounds they lead the baseline by ≈ 60–67 MPPI on the cabinet and ≈ 45–51 on no-wall. Ordered (G20, r_k = 0.1k) and its
  matched unordered mixture (W20g) are again equal within seed noise; the per-round RK4 of G20 sits above W20g's
  for most rounds on training seed 87358, but seed 1 reverses the final MPPI order (760 vs 785).
- **Progress-only backups (`--min-step 0.05`).** Excluding the 0.01 and keeping ≥ 0.05 steps helps the natural
  baseline on the cabinet (B10ms 750 vs 718) and hurts it on no-wall (1555 vs 1611), where only ~2 safe 0.05-steps
  exist per source. It also inflates the target for pairs closer than one step (T/|q−g| 5–9.5 below gap 0.1),
  which breaks RK4's final approach on no-wall (W10ms: 368 endpoint failures). Under it ordered and unordered are
  equal on the cabinet (G10ms 765, W10ms 769); on no-wall G10ms keeps RK4 better (493 vs 229) on one seed.
- **Which loss term needs the short pairs (`--split`, one seed).** Bellman + direction on the natural pairs and only
  the Eikonal term on G10's pairs (E10eik) recovers essentially the whole G10 gain (cabinet 764, no-wall 1665 vs
  G10's 767 / 1660 means). The reverse (E10bel: Bellman + direction on G10's pairs, Eikonal on natural pairs) gets
  only +16 / +22 over B10 (734 / 1633). The gain is the eikonal equation being enforced where the planners operate.
- **Fewer rounds (G5, r_k = 0.4k).** Five rounds with rehearsal reach the 10-round baseline: cabinet 722 (B10 718),
  no-wall 1625 (1614), at half the updates.
- **Other gap mixtures, no order.** Gap uniform in [0, 2] every round (U10, one seed): cabinet 751, no-wall 1672
  with RK4 553. Half of the pairs from the 0.2-ball (W10a): cabinet 790/770/778 (three seeds, mean 779, the best
  10-round cabinet mixture), no-wall 1660 (one seed). Adding 0.1-balls (W10b): cabinet RK4 277/271/262, MPPI 773/764/777 (three seeds,
  mean 771); no-wall RK4 563/554 and MPPI 1688/1677 (two seeds; the second ran on the RTX 3090), the best no-wall
  result of the session at 10 rounds. Within the range tried, more short-gap mass keeps helping or saturates.
- **Travel-time order (V10, Dijkstra-like: round k takes goals with T_{k−1}(q, g) ≤ 0.28k, no rehearsal).** Cabinet
  718, no-wall 1618 — the baseline's level, like C10: ordering by travel time instead of distance does not help either.
- **The densest mixture with 20 rounds (W20b: 0.1 ×6, 0.2 ×4, 0.4 ×4, 0.6 ×2, 1.0 ×2, 2.0 ×2, no order, one seed).**
  Cabinet RK4 271, MPPI 795; no-wall RK4 544, MPPI 1689 (93.8 %) — on par with the best runs; at this point extra
  rounds and denser mixtures add little.
- **30 rounds (G30, cabinet).** RK4 271, MPPI 820 — the same as G20 (seed 87358): the rehearsal curriculum
  saturates by about 20 rounds.

## What would let the gap order act as DP

- **Transitions that make progress.** The backup can only build long pairs from shorter ones if some safe action
  brings the source substantially closer to the goal. The offline bank has 8 random directions per source, and the
  tiny step wins the min almost always (§ Mechanism). A bank with goal-directed or much denser actions would give
  the order something to work with; that is a new bank (or online transitions) and needs your decision.
- **Queries where long-range structure matters.** The order improves the long-range field on the cabinet; the
  frozen suites are all short. A certified long-query suite would measure whether that matters for planning.
- **Independently of the order**, putting a large share of the training pairs at short gaps (W10/G10) is a pure
  sampling change that raised release MPPI by +26 to +77 per seed on the cabinet and +31 to +54 on no-wall, and
  moving only the Eikonal term onto those pairs is enough (E10eik).

## Run index

|Run id (`results/registered_<id>`)|State|Label|Registry name|
|---|---|---|---|
|`20261027T041104Z-626c3e47`|succeeded||gap-curriculum nowall smoke-route b2c39206f4 smoke|
|`20261027T041132Z-d06c4990`|succeeded|cabinet/B10|gap-curriculum cabinet B10 fb6fce47ea full|
|`20261027T041133Z-ea75338e`|succeeded|nowall/B10|gap-curriculum nowall B10 6963d54d89 full|
|`20261027T042053Z-1e8f344d`|succeeded|cabinet/C10|gap-curriculum cabinet C10 516aa1ff1f full|
|`20261027T042133Z-5374cf46`|succeeded|nowall/C10|gap-curriculum nowall C10 8bc77ffa59 full|
|`20261027T043413Z-bfdaa720`|succeeded|cabinet/M10|gap-curriculum cabinet M10 b7a2462ec4 full|
|`20261027T043953Z-24560561`|succeeded|nowall/M10|gap-curriculum nowall M10 7712b0d303 full|
|`20261027T044847Z-c42c6e8f`|succeeded|cabinet/B5|gap-curriculum cabinet B5 055bbe4f9b full|
|`20261027T045431Z-a37c13f2`|succeeded||gap-curriculum-rounds cabinet 20261027T042053Z-1e8f344d f0fb9b14e2 full|
|`20261027T045632Z-af2d7b95`|succeeded||gap-curriculum-rounds cabinet 20261027T041132Z-d06c4990 3996b7d111 full|
|`20261027T045837Z-64341aeb`|succeeded||gap-curriculum-rounds nowall 20261027T042133Z-5374cf46 b5c6fd9b5f full|
|`20261027T045906Z-45797805`|succeeded|nowall/B5|gap-curriculum nowall B5 183a9dd0c7 full|
|`20261027T050125Z-3124c383`|succeeded||gap-curriculum-rounds nowall 20261027T041133Z-ea75338e 17292ebb93 full|
|`20261027T050405Z-5a67946b`|succeeded||gap-curriculum-rounds cabinet 20261027T043413Z-bfdaa720 b91911fe44 full|
|`20261027T050615Z-82bb2ef3`|succeeded||gap-curriculum-rounds nowall 20261027T043953Z-24560561 b9f41ac0b3 full|
|`20261027T051032Z-d7863eed`|succeeded|nowall/G10|gap-curriculum nowall G10 759a28db52 full|
|`20261027T051040Z-ff7c5a64`|succeeded|cabinet/G10|gap-curriculum cabinet G10 7ab68bd0c6 full|
|`20261027T051113Z-7d896d5b`|succeeded||gap-curriculum-truth cabinet B5 B10 C10 M10 8290dfd7a1 full|
|`20261027T051345Z-9f7fb6b4`|succeeded||gap-curriculum-truth nowall B5 B10 C10 M10 9acb8b4800 full|
|`20261027T052036Z-f790822e`|succeeded|cabinet/B10s1|gap-curriculum cabinet B10s1 95f7682349 full|
|`20261027T052327Z-c64c695a`|succeeded|nowall/B10s1|gap-curriculum nowall B10s1 094144482d full|
|`20261027T052422Z-09b77595`|succeeded||gap-curriculum-rounds cabinet 20261027T051040Z-ff7c5a64 656dac9d19 full|
|`20261027T052710Z-51f5f3b0`|succeeded||gap-curriculum-rounds nowall 20261027T051032Z-d7863eed abd5ff20ca full|
|`20261027T053012Z-918f4980`|succeeded|cabinet/C10s1|gap-curriculum cabinet C10s1 2e78d9522b full|
|`20261027T053356Z-91029673`|succeeded|nowall/C10s1|gap-curriculum nowall C10s1 e0ffe0828b full|
|`20261027T053913Z-ad52c7e4`|succeeded|cabinet/M10s1|gap-curriculum cabinet M10s1 0635bbd3c3 full|
|`20261027T054424Z-c9dbbd3e`|succeeded|nowall/M10s1|gap-curriculum nowall M10s1 9e0df866e8 full|
|`20261027T054804Z-c121892f`|succeeded|cabinet/G10s1|gap-curriculum cabinet G10s1 b0b3408a9a full|
|`20261027T055510Z-41c553da`|succeeded|nowall/G10s1|gap-curriculum nowall G10s1 0351e1b6e6 full|
|`20261027T055701Z-f1c4a3e8`|succeeded|cabinet/W10|gap-curriculum cabinet W10 2648676115 full|
|`20261027T060549Z-15dc472b`|succeeded|cabinet/B10s2|gap-curriculum cabinet B10s2 eaa0e2160e full|
|`20261027T060553Z-1f1fc233`|succeeded|nowall/W10|gap-curriculum nowall W10 2473cf06ee full|
|`20261027T060702Z-4cdce123`|succeeded||gap-curriculum-rounds cabinet 20261027T055701Z-f1c4a3e8 f6ce507f22 full|
|`20261027T061427Z-e0b399d6`|succeeded|cabinet/C10s2|gap-curriculum cabinet C10s2 5535f6101d full|
|`20261027T061643Z-a88e9d4c`|succeeded|nowall/B10s2|gap-curriculum nowall B10s2 670e3ca28e full|
|`20261027T061752Z-8115525d`|succeeded||gap-curriculum-truth cabinet G10 G10s1 W10 34e627f91e full|
|`20261027T062413Z-33029867`|succeeded|cabinet/M10s2|gap-curriculum cabinet M10s2 b191e65eb4 full|
|`20261027T062620Z-aac39732`|succeeded||gap-curriculum-long cabinet B5 B10 B10s1 B10s2 C10 C10s1 C10s2 M10 M10s1 G10 G10s1 W10 ec894e9ec3 full|
|`20261027T062715Z-78ce9865`|succeeded|nowall/C10s2|gap-curriculum nowall C10s2 11bfd89330 full|
|`20261027T063025Z-43c2957d`|succeeded||gap-curriculum-long cabinet B5 B10 B10s1 B10s2 C10 C10s1 C10s2 M10 M10s1 G10 G10s1 W10 b70058ff8f full|
|`20261027T063503Z-6a46b1a5`|succeeded|cabinet/G10s2|gap-curriculum cabinet G10s2 fd74c26048 full|
|`20261027T063807Z-9fcdfd4b`|succeeded|nowall/M10s2|gap-curriculum nowall M10s2 8abeabd0ac full|
|`20261027T064757Z-42448729`|succeeded|cabinet/W10s1|gap-curriculum cabinet W10s1 d05745f130 full|
|`20261027T064929Z-f530a2cd`|succeeded|nowall/G10s2|gap-curriculum nowall G10s2 744028b229 full|
|`20261027T070045Z-d57ab030`|succeeded|cabinet/W10s2|gap-curriculum cabinet W10s2 c5791ffe52 full|
|`20261027T070055Z-940f6c1e`|succeeded|nowall/W10s1|gap-curriculum nowall W10s1 be217a6dac full|
|`20261027T071217Z-f614df0f`|succeeded||gap-curriculum-truth nowall G10 G10s1 W10 4c835c916b full|
|`20261027T071334Z-4b9fc917`|succeeded|cabinet/G10ms|gap-curriculum cabinet G10ms c75d16cb6c full|
|`20261027T071456Z-e94e22b7`|succeeded||gap-curriculum-rounds nowall 20261027T060553Z-1f1fc233 1da89711f6 full|
|`20261027T071756Z-0f0fd107`|succeeded|nowall/W10s2|gap-curriculum nowall W10s2 7a66f610bd full|
|`20261027T072610Z-805fc208`|succeeded|cabinet/W10ms|gap-curriculum cabinet W10ms 8502b50862 full|
|`20261027T072912Z-f307d096`|succeeded|nowall/G10ms|gap-curriculum nowall G10ms 4fa3fdf83e full|
|`20261027T073856Z-d76fd149`|succeeded|cabinet/B10ms|gap-curriculum cabinet B10ms 74c26f0fc8 full|
|`20261027T074013Z-aab3cbd2`|succeeded|nowall/W10ms|gap-curriculum nowall W10ms 6a7325f3a0 full|
|`20261027T075043Z-30b45521`|succeeded|cabinet/G20|gap-curriculum cabinet G20 77f9952321 full|
|`20261027T075141Z-b0c5187f`|succeeded|nowall/B10ms|gap-curriculum nowall B10ms 943e385286 full|
|`20261027T080124Z-5446cb8a`|succeeded|nowall/G20|gap-curriculum nowall G20 593890996f full|
|`20261027T080748Z-3fe3fa09`|succeeded|cabinet/B20|gap-curriculum cabinet B20 a3661ea659 full|
|`20261027T080919Z-1b62808a`|succeeded||gap-curriculum-rounds cabinet 20261027T075043Z-30b45521 3337366fe5 full|
|`20261027T082011Z-5c38d608`|succeeded|nowall/B20|gap-curriculum nowall B20 c669bf85d5 full|
|`20261027T082238Z-71513bf9`|succeeded|cabinet/W20|gap-curriculum cabinet W20 9f0f05320f full|
|`20261027T083708Z-5346aa47`|succeeded|nowall/W20|gap-curriculum nowall W20 4b5021d84a full|
|`20261027T083806Z-a0efdda4`|succeeded|cabinet/W20g|gap-curriculum cabinet W20g 9f1f1b526e full|
|`20261027T085457Z-8f250463`|succeeded|cabinet/G20s1|gap-curriculum cabinet G20s1 c8f4e7d470 full|
|`20261027T085509Z-91bd7d51`|succeeded|nowall/W20g|gap-curriculum nowall W20g 1645aeef50 full|
|`20261027T085616Z-1ce059b9`|succeeded||gap-curriculum-rounds cabinet 20261027T083806Z-a0efdda4 75e2c06024 full|
|`20261027T090427Z-c80ae3ce`|succeeded||gap-curriculum-truth cabinet B20 G20 W20 W20g adb10cbad4 full|
|`20261027T091402Z-778cc6f5`|succeeded|cabinet/W20gs1|gap-curriculum cabinet W20gs1 cb54d2e583 full|
|`20261027T091421Z-456ebb48`|succeeded|nowall/G20s1|gap-curriculum nowall G20s1 29654e491a full|
|`20261027T093058Z-f8da69a5`|succeeded|cabinet/G20s2|gap-curriculum cabinet G20s2 18b3153579 full|
|`20261027T093310Z-2397a726`|succeeded|nowall/W20gs1|gap-curriculum nowall W20gs1 b887512731 full|
|`20261027T094747Z-de4c5c7f`|succeeded|cabinet/W20gs2|gap-curriculum cabinet W20gs2 ffc12674bf full|
|`20261027T095158Z-69cd418d`|succeeded|nowall/G20s2|gap-curriculum nowall G20s2 9d3a9dd03e full|
|`20261027T100435Z-b21e47fe`|succeeded|cabinet/B10s3|gap-curriculum cabinet B10s3 5b9932c731 full|
|`20261027T101041Z-df0fddb0`|succeeded|nowall/W20gs2|gap-curriculum nowall W20gs2 e67673a6f7 full|
|`20261027T101258Z-f092f929`|succeeded|cabinet/G10s3|gap-curriculum cabinet G10s3 65d82d1ef6 full|
|`20261027T102153Z-029a83e6`|succeeded|cabinet/W10s3|gap-curriculum cabinet W10s3 29377aa03b full|
|`20261027T102930Z-4c3f1048`|succeeded|nowall/E10eik|gap-curriculum nowall E10eik e756dd0d37 full|
|`20261027T103049Z-cc156de8`|succeeded|cabinet/E10eik|gap-curriculum cabinet E10eik 91b92a86c3 full|
|`20261027T104205Z-dfd4572d`|succeeded|cabinet/E10bel|gap-curriculum cabinet E10bel a357a661d9 full|
|`20261027T104320Z-291a823c`|succeeded|nowall/E10bel|gap-curriculum nowall E10bel b31a12b1af full|
|`20261027T105317Z-b6fb9043`|succeeded|cabinet/G5|gap-curriculum cabinet G5 e5319f9b49 full|
|`20261027T105649Z-e1e10887`|succeeded|nowall/G5|gap-curriculum nowall G5 3474f2d119 full|
|`20261027T105852Z-00256d77`|succeeded|cabinet/U10|gap-curriculum cabinet U10 cd209b3e6f full|
|`20261027T110358Z-93433d96`|succeeded|nowall/U10|gap-curriculum nowall U10 9e874ae00d full|
|`20261027T111148Z-86bc5d73`|succeeded|cabinet/V10|gap-curriculum cabinet V10 d9c5ec6c09 full|
|`20261027T112143Z-a159ccd7`|succeeded|nowall/V10|gap-curriculum nowall V10 da0c34e3fc full|
|`20261027T112642Z-60874e60`|succeeded|cabinet/W10a|gap-curriculum cabinet W10a 3c4185d47a full|
|`20261027T113541Z-80b59b61`|succeeded|cabinet/W10b|gap-curriculum cabinet W10b 47e5da1777 full|
|`20261027T114233Z-1aa1ab44`|succeeded|nowall/W10a|gap-curriculum nowall W10a ffe7cd578a full|
|`20261027T114543Z-aa31c573`|succeeded|cabinet/G30|gap-curriculum cabinet G30 5ae92df8a5 full|
|`20261027T115347Z-f26afe6c`|succeeded|nowall/W10b|gap-curriculum nowall W10b 10128eb8d2 full|
|`20261027T120517Z-a0948d29`|succeeded|nowall/B10s3|gap-curriculum nowall B10s3 e8149badc4 full|
|`20261027T121059Z-8f297d77`|succeeded|cabinet/W10bs1|gap-curriculum cabinet W10bs1 14c7895881 full|
|`20261027T121540Z-5b70ffca`|succeeded|nowall/G10s3|gap-curriculum nowall G10s3 b013419359 full|
|`20261027T122059Z-01830f9c`|succeeded|cabinet/W10bs2|gap-curriculum cabinet W10bs2 3e4b29defd full|
|`20261027T122639Z-72b9fc12`|succeeded|nowall/W10s3|gap-curriculum nowall W10s3 9e40150ab6 full|
|`20261027T123052Z-02c012ee`|succeeded|nowall/W10bs1|gap-curriculum nowall W10bs1 ae68f215d4 full|
|`20261027T124337Z-97f64f1d`|succeeded|cabinet/W10as1|gap-curriculum cabinet W10as1 0ae4ff2aaf full|
|`20261027T124338Z-325c2874`|succeeded|nowall/W10bs2|gap-curriculum nowall W10bs2 6123d88df3 full|
|`20261027T125236Z-4dd184ba`|succeeded|cabinet/W10as2|gap-curriculum cabinet W10as2 f799cd9442 full|
|`20261027T125504Z-6a9ae143`|succeeded|nowall/W10bs3|gap-curriculum nowall W10bs3 f47f153b78 full|
|`20261027T130133Z-e97e8aaf`|succeeded|cabinet/W20b|gap-curriculum cabinet W20b 565b167f84 full|
|`20261027T130639Z-60d49f65`|succeeded|nowall/W20b|gap-curriculum nowall W20b 10f02ce9a3 full|

## Commands

```bash
export BELLMAN600_BACKEND=host BELLMAN600_HOST_MEMORY=12G
PY=~/.venvs/bellman-ntfield/bin/python
$PY jeffrey/gap_curriculum/run.py --scene cabinet --rounds 10 --gpu 1                                    # B10
$PY jeffrey/gap_curriculum/run.py --scene cabinet --rounds 10 --goals ball --radii linear --gpu 1        # C10
$PY jeffrey/gap_curriculum/run.py --scene cabinet --rounds 10 --goals ball --radii linear --mix --gpu 1  # M10
$PY jeffrey/gap_curriculum/run.py --scene cabinet --rounds 10 --goals ball --radii linear --grow-mix --gpu 1   # G10
$PY jeffrey/gap_curriculum/run.py --scene cabinet --rounds 10 --goals ball --radii 0.2,0.2,0.2,0.4,0.4,0.6,0.8,1.0,1.4,2.0 --mix --gpu 1  # W10
$PY jeffrey/gap_curriculum/run2.py --scene cabinet --rounds 10 --goals ball --radii linear --grow-mix --split eikonal --gpu 1          # split
$PY jeffrey/gap_curriculum/run_stage.py rounds --runs registered_<id> --gpu 1                            # RK4 of every round (diagnostic)
$PY jeffrey/gap_curriculum/run_stage.py truth --runs registered_<A>=B10 registered_<B>=C10 --gpu 1      # graph ground truth
$PY jeffrey/gap_curriculum/run_stage.py long --runs registered_<A>=B10 registered_<B>=C10 --mppi-seeds 0 --gpu 1   # long-gap diagnostic
$PY jeffrey/gap_curriculum/report.py --runs registered_<A>=B10 registered_<B>=C10 --reference B10 --diagnostics
```

Files: `training.py` (trainer), `training2.py` + `pipeline2.py` + `launch2.py` + `run2.py` (the `--split`
decomposition, separate files so in-flight runs kept their declared inputs), `diagnose.py` (per-round field
statistics by gap), `rounds.py` (RK4 of every round), `truth.py` (kNN-Dijkstra ground truth), `long.py` (long-gap
diagnostic), `run_stage.py` + `stage_launch.py` (registry entry for the diagnostics), `report.py`, `summary.py`.
