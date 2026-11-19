# Octave detuning: exact octave frequencies vs frequencies near them

Request (2026-11-17): "I want to see if the octave method is actually good because of its specific frequency or because
it is higher frequency. Have some tests where the frequencies are around the same as the octave frequencies, but
slightly perturbed so they are not exactly octave. See if it is still the same as the octave. I want to see all
possible changes, including convergence rates."

"The octave method" is taken to be `jeffrey/grid_sync`'s `octave` matrix (the field the later studies call "octaves",
e.g. `jeffrey/speed_fourier`): ntrl-demo's 128 Fourier rows keep their directions and are put on the squares
max(|f_x|, |f_y|) = f_grid / L, L = 2^k cells (f_grid = E / 2: half period one cell), K equal groups, so every row is
exactly in step with every L-th grid line. (The `jeffrey/embed_warp` octaves, ×2 / ×4 / ×8 multiples of ntrl-demo's
Gaussian rows, have no specific frequency to detune and are not tested here.)

Status: done (2026-11-17 15:47; all 124 trainings registered, none failed). Findings page (v6):
https://claude.ai/artifact/QRt86R9pVk2CBgEzX3ayXw (rebuild: `spatial.py`, `report.py`, `stalls.py`, then `page.py`;
republish `report/octave_detune.html`).

## Findings (final, 2026-11-17 15:55; all 124 trainings)

Answer: the octave method works because its band reaches the cell scale (top octave: half period one cell), not because of
its exact frequencies. Moving the band down loses walls and success, and a locked band one octave down (×0.5) is the worst
variant. Moving it up gains no success, and far up (×2) trades wall crossings for local minima. Sitting exactly on the octave
positions (locked to the grid) adds nothing measurable on 13×13 and about 2 points on 22×22 (pooled over the six detunings
that keep the band: −2.2 ± 1.0, about two standard errors), together with slightly faster convergence and a slightly better fit.

Paired change in success (points, ± SE from training noise) against the same seed's exact-frequency trainings (the exact
octave, the nulls ×1.0001 / ×0.9999, and the grid-sync run where it ran on the other GPU). 3 training seeds per cell unless
noted. SD of one run 1.83 points on 13×13 (6 degrees of freedom) and 3.03 on 22×22 (14). Exact octave: 94.8 % / 48.1 %.

| field | 13×13 | 22×22 |
|---|---|---|
| every row ×0.5 (locked, one octave down) | −15.8 ± 1.2 | −16.2 ± 2.0 |
| ×0.707 | −6.8 ± 1.2 | −12.8 ± 2.0 |
| ×0.841 | −2.1 ± 1.2 | −11.4 ± 2.0 |
| ×0.9 | −5.0 ± 1.2 | −9.4 ± 2.0 |
| ×0.95 | +0.6 ± 1.2 | −5.2 ± 1.5 (5 seeds) |
| ×1.05 | −1.0 ± 1.2 | −2.2 ± 1.5 (5 seeds) |
| ×1.1 | −0.7 ± 1.2 | −1.7 ± 2.0 |
| ×1.189 | −1.5 ± 1.2 | −1.1 ± 2.0 |
| ×1.414 | −2.8 ± 1.2 | +1.9 ± 1.5 (5 seeds) |
| ×2 (locked, one octave up) | −5.2 ± 1.2 | −0.2 ± 1.5 (5 seeds) |
| jitter ±5 / ±10 / ±20 % | −2.0 / −0.2 / −0.2 ± 1.2 | −2.7 ± 2.0 / −1.6 ± 1.5 (5 seeds) / −3.6 ± 2.0 |
| log-uniform fill (no discrete frequency) | −1.1 ± 1.2 | −1.8 ± 2.0 |
| pooled: ×1.05, ×1.1, the three jitters, the fill | −0.9 ± 0.7 | −2.2 ± 1.0 |
| top octave ×0.9 / every other octave ×0.9 | −1.7 / −0.2 ± 1.2 | – |
| nulls ×1.0001 / ×0.9999 | −0.0 / −1.4 ± 1.3 | −0.6 / −1.2 ± 1.6 (5 seeds) |
| ntrl-demo | −35.0 ± 1.2 | −23.0 ± 2.0 |

- Slight detuning: no single variant that keeps the band can be told apart from training noise; pooled they cost nothing on
  13×13 and about 2 points on 22×22. The log-uniform fill, with no two rows on the same frequency and none locked, is as good as
  the jitters, so the method needs rows across the band, not the octave values.
- Lower: ×0.9 and ×1.1 break the lock by the same amount (top octave 2.0 / 3.5 rad out of phase at the 13×13 / 22×22
  border), and only the lower one loses (every seed). The top octave carries the ×0.9 fit loss: lowering only it gives final
  loss +15 % (whole shift +15 %), speed correlation −0.021 (−0.018) and +3.6 points of wall crossings (+6.7); lowering every
  other octave changes nothing measurable.
- The two failure kinds move in opposite directions (failures split by arrival). Lower bands have soft walls: ×0.5 changes
  wall crossings by +31.4 points on 22×22 (z +14) and never-arrived by −15.2; +16.1 points of wall crossings on 13×13. ×2 has
  walls that hold (−19.0 on 22×22) but the learned T gains local minima along the true paths (+2.59 per path on 22×22, z +17,
  exact 0.57; +0.15 on 13×13, z +10, exact 0.04) and never-arrived rises by +19.2 / +7.2 points. ×2 fits better (final loss
  −16 % / −6 %, speed correlation +0.041 on 22×22).
- Where the stalled runs stop (`stalls.py`, planning seed 0; last MPPI position, moved < 0.1 cell over 20 steps; clearance
  < 0.15 cell = against a wall, else open corridor; on route = within 0.4 cell of the oracle path): exact 22×22 18.7 % of runs
  against a wall, 8.8 % in the open (2.8 % on route); ×2 29.9 % in the open (16.6 % on route): the field points the right way
  and the planner stalls in a ripple. ×0.5 stops against a wall less often (8.2 %): the planner goes through instead. 13×13:
  ×2 7.2 % in the open against 1.2 %.
- Convergence and fit, the six detunings that keep the band: success at epoch 2,000 on 22×22 −3.8 ± 1.0 points, narrowing to
  −2.2 ± 1.0 at 5,000; on 13×13 −2.5 ± 1.3 at 1,000 and −2.0 ± 1.2 at 2,000. At 5,000: final loss +5 % / +5 %, held-out speed
  correlation −0.009 ± 0.002 / −0.020 ± 0.003, near-wall slope −0.04 / −0.05 ± 0.01. Every unlocked shift from ×1.05 to ×1.41
  ends with a 2–4 % higher final loss than the octave although its band is higher, while ×2 (locked) ends lower: the lock helps
  the fit (the scan cannot split the ×2 gain between the lock and its added half-cell octave). On 22×22 the loss of the
  unlocked higher shifts settles later the higher they sit (+282 / +849 / +1,011 / +1,864 epochs at ×1.05 / ×1.1 / ×1.19 /
  ×1.41, z +1.5 … +9.9; ×2 +486, z +2.6); on 13×13 every shift settles within noise of the octave or earlier (lower bands
  about 500 epochs earlier: they stop improving sooner). ×2 leads early on 22×22 (+5.5 points at epoch 1,000, +4.4 at 2,000)
  and ends level; on 13×13 it trails (−9.8 at 1,000, −5.2 at 5,000).
- Where in the maze (`spatial.py`, paired by seed since v6): a lock effect would lose near-wall slope in the outer third of the
  maze and keep it in the inner third. No group does: outer minus inner third ×0.9 / ×0.95 +0.05 ± 0.03 (13×13) and
  −0.01 ± 0.03 (22×22), ×1.05 / ×1.1 −0.01 / 0.00 ± 0.03, jitter −0.01 / 0.00 ± 0.03. The one difference beyond a standard
  error (lower shifts, 13×13) goes the other way. (Before v6 the ring chart compared unpaired group means, which suggested a
  border loss for the higher shifts on 13×13; paired by seed it is not there.)
- ×1.41 on 22×22 looked like a gain after two seeds (+7.3 ± 2.5); seeds 3 and 4 (queued for that reason) bring it to +1.9 ± 1.5.

## Round 2 and the fill (2026-11-17 09:00; 67 of 120 runs)

Paired change in success (points, ± SE from training noise) against the same seed's exact-frequency trainings; 3 training
seeds per cell; SD of one run 1.8 points on 13×13 and 3.2 on 22×22.

| variant | 13×13 | 22×22 |
|---|---|---|
| every row ×0.9 | −5.0 ± 1.2 (3/3 seeds lost) | −9.4 ± 2.1 (3/3 lost) |
| every row ×0.95 | +0.6 ± 1.2 | −4.1 ± 2.1 |
| every row ×1.05 / ×1.1 | −1.0 / −0.7 ± 1.2 | −2.2 / −1.7 ± 2.1 |
| jitter ±5 / ±10 / ±20 % | −2.0 / −0.2 / −0.2 ± 1.2 | −2.7 / +0.5 / −3.6 ± 2.1 |
| fill (log-uniform ±½ octave, no discrete frequency) | −1.1 ± 1.2 | −1.8 ± 2.1 |
| pooled: the six that do not lower the band | −0.9 ± 0.7 | −1.9 ± 1.2 |
| nulls ×1.0001 / ×0.9999 | −0.0 / −1.4 ± 1.3 | −1.3 / −1.0 ± 2.2 |

- The exact octave values are not what makes the method work: every detuning that keeps the band where it is or moves it up
  is within noise, including the log-uniform fill, where no two rows share a frequency and none is locked to the grid.
  Pooled over those six (`report.py: GROUPS['not_lower']`; the shared per-seed reference is in the SE) the exact values are
  worth −0.9 ± 0.7 / −1.9 ± 1.2 points: at most a couple of points.
- The level does matter: every row ×0.9 loses on every seed of both mazes, ×1.1 does not, and both break the lock by the same
  amount (top-octave edge phase error 2.0 rad on 13×13, 3.5 rad on 22×22). ×0.95 is in between.
- What ×0.9 loses is the walls: runs that reach the goal through a wall 9.3 % vs 2.3 % on 13×13 and 43.5 % vs 24.2 % on
  22×22; never-arrived goes down; final loss +15 % / +25 %; near-wall cos and held-out speed fit lower (z −6 … −11). The
  near-wall slope drops as much in the inner third of the maze as in the outer third, so it is not the phase drift (which grows
  toward the border).
- The top octave carries it (part-of-band test, 13×13, 3 seeds, 10:20). Only the top octave ×0.9 (its 32 rows; every other
  row exactly locked): final loss +15 % (z +9.5; ×0.9: +15 %), speed correlation −0.021 (z −7.1; ×0.9: −0.018), near-wall
  slope −0.063, wall crossings +3.6 points (z +3.2; ×0.9: +6.7), success −1.7 ± 1.2. Every row but the top ×0.9 (top exact):
  loss −1 %, speed correlation +0.000, wall crossings +0.4, success −0.2 ± 1.2, i.e. nothing. Since ×1.1 (top half period
  0.9 cell) costs nothing, the condition is that the band reaches the cell scale, not that it sits on it.
- What the exact values do buy: faster convergence on 22×22 (success at epoch 2,000 −4.1 ± 0.7 points pooled over the six,
  the fill −6.2, z −5.5; mostly gone by 5,000) and a slightly better fit at epoch 5,000 (final loss +5 %, speed correlation
  −0.009 / −0.023, near-wall slope −0.04 / −0.05, z −3 … −6, both mazes). On 13×13 the convergence lag is small
  (−2.5 ± 1.3 at epoch 1,000).
- The higher shifts and jitters on 13×13 lose a little near-wall slope toward the border (−0.02 inner, −0.05 outer): the
  drift is visible in the field, but it does not cost success.

Run ids: `queue/state.json` (done map `gpu|n|seed|spec -> run id`); registry search `octdetune`.

## Round 1 and the nulls (2026-11-17 04:45)

- Training noise is large and must be the yardstick. The exact octave rerun on the 5070 is bit-identical to the grid-sync
  runs, but a 1e-4 change of the frequencies (×1.0001, ×0.9999) or the other GPU moves success by up to 5 points on 13×13
  (seed 87358: 94.8 / 94.5 / 89.7 %) and 9 points on 22×22 (seed 87358: 56.9 / 48.3 %). SD of one run: 2.0 points on 13×13,
  3.9 on 22×22. Every comparison below is paired by seed against the mean of that seed's exact-frequency trainings.
- Slight detuning keeps the success: shift ×0.95 / ×1.05 / jitter ±10 % are +0.7 / −0.9 / −0.1 ± 1.4 points on 13×13 and
  −4.4 / −2.5 / +0.2 ± 2.7 on 22×22, all within noise; the gain over ntrl-demo (−34.9 / −23.3 points) is untouched.
- The exact frequencies give a slightly better fit: every lock-breaking variant (and the grid-sync controls) has a lower held-out
  speed correlation (−0.005 … −0.013 on 13×13, z −2 … −5), a 3–7 % higher final loss and a slightly lower near-wall slope,
  and on 22×22 trails at epoch 2,000 (−3.5 … −5 points) before catching up by 5,000. The loss is spread evenly over the maze
  (inner and outer rings alike), so it is not a local phase drift.
- The earlier grid-sync lock result (octave vs octaveperm / octavecircle) shrinks against this noise: −1.5 / −1.0 ± 1.4 points on
  13×13, −3.4 / −5.8 ± 2.2 on 22×22 (z −1.1 … −2.6).

## Setup

Mazes, data, network, losses, optimizer, 5,000 epochs, planner and scorer are those of `jeffrey/maze_scale` /
`jeffrey/grid_sync`, unchanged: 13×13 (prepare `20261101T192504Z-13cf1d04`, 200 queries × planning seeds 0/1/2 = 600
runs) and 22×22 (`20261101T193403Z-cea05796`, 240 queries, 720 runs). Only the Fourier matrix changes. Every variant
starts from the exact octave matrix of the same training seed (same draw, so the same row directions, the same row
grouping, the same heading column and the same network initialization) and multiplies each row's xy frequency by a
factor (`features.py`):

| spec | factor | what it tests |
|---|---|---|
| `octave` | 1 | the method (rerun here; on the 5070 bit-identical to the grid-sync runs) |
| `shift1.0001` | 1.0001 for every row | null perturbation: the same frequencies, a different floating-point trajectory (training noise floor) |
| `shift0.95`, `shift1.05`, `shift0.9`, `shift1.1` | 0.95 … 1.1 for every row | coherent detuning: levels still a factor 2 apart, none on f_grid / L; lower vs higher |
| `jitter0.05`, `jitter0.1`, `jitter0.2` | exp(u_i), u_i ~ U(−1, 1) ln(1 + d), centred per octave | incoherent detuning: each row a little off its octave, each octave's geometric-mean frequency exact |
| `jitter0.4142` | exp(u_i), u_i ~ U(−1, 1) ln √2, centred per octave | log-uniform fill: each octave spread over ±half an octave, so the rows cover the whole band evenly with no discrete or locked frequency (same band as the octaves) |
| `shift0.707`, `shift0.841`, `shift1.189`, `shift1.414`, `shift2` | 2^−0.5 … 2^1 | the wide scan: is higher better? (×2 is locked again, one octave up) |
| `shift0.5` | 0.5 for every row | locked again, one octave down (the mirror of ×2): lock kept, top level at f_grid / 2 |
| `top0.9`, `rest0.9` | 0.9 for the top octave only / for every row but the top | which part of the band carries the ×0.9 loss (13×13 only; together they make ×0.9) |

The lock is measured by the edge phase error: a row's phase mismatch with an exactly locked row at the outermost grid line
(all rows agree at the maze centre), π |d − d*| n / 2 with d its dominant frequency in grid units and d* the nearest power
of two. Top-octave edge error: ±5 % → 1.0 rad on 13×13 and 1.7 rad on 22×22; ±10 % → 2.0 / 3.5 rad; jitter 10 % → 0.8 /
1.2 rad (median); so even ±5 % breaks the top octaves' lock across the maze while barely moving the frequency band.

Seeds and GPUs: every (maze, training seed) group runs on one GPU, so its variants pair exactly. GPU 0 (RTX 5070):
13×13 seeds 87358 / 1 / 2 and 22×22 seed 87358 (the GPU of the grid-sync runs, so the earlier `octaveperm` /
`octavecircle` controls pair exactly too); GPU 1 (RTX 3090): 22×22 seeds 1 / 2 with every variant, then seeds 3 / 4 with
the round-1 variants plus ×1.41 and ×2 (added at 11:40, after two seeds showed ×1.41 at +7.3 points). The 3090 was shared
with another session's queue until 11:23 (about 16 min per job, 9 min after) and again from about 13:00 (one job at a time). 124 trainings in all. Rounds: (1) octave, null, ±5 %, jitter 10 %; (2) ±10 %, jitter 5 % / 20 %, then the log-uniform fill; (3) the wide scan,
then ×0.5 and (13×13) top0.9 / rest0.9, added after round 2 showed the ×0.9 loss.

Measures (all variants, epoch 5,000): success pooled and by distance bin; per-query gained / lost runs against the exact
octave of the same seed; failures split by arrival (never arrived vs arrived after touching a wall); median
|log T/T*|, T/T*, Spearman, descent cosine; held-out speed fit; near-wall slope and direction; local minima along true
paths; doorway profiles; path length and travel time of successful paths; final loss and epoch rollbacks.
Convergence: the per-epoch loss and rollbacks from the registry console log; success (planning seed 0) and the field
measures at the kept checkpoints (`convergence.py`: every 500 epochs for this study's runs, every 1,000 for the earlier
fields; ntrl-demo saves a checkpoint every 100 epochs), all evaluated on the 3090 with planning seed 0; the page's
convergence charts average training seeds 87358 / 1 / 2 for every line, the paired changes use every seed. Loss settling
epoch: the first epoch whose trailing 100-epoch mean is within 10 % of the run's final 100-epoch mean.

Context fields (earlier registered runs, same mazes and queries): ntrl-demo, B × 3 / B × 6 (Gaussian rows at a higher
frequency, no octave structure), `octaveperm` (octave magnitudes shuffled within each octave) and `octavecircle`
(constant magnitude per octave).

## Reproduce

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/octave_detune/run.py \
        --prepared results/registered_20261101T193403Z-cea05796 --seed 87358 --spec shift1.05
    setsid nohup ~/.venvs/bellman-ntfield/bin/python jeffrey/octave_detune/queue/runner.py > jeffrey/octave_detune/queue/runner.out 2>&1 &
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/octave_detune/convergence.py --loop --old
    CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/octave_detune/spatial.py   # near-wall slope by ring
    ~/.venvs/bellman-ntfield/bin/python jeffrey/octave_detune/report.py                          # reads spatial.json too
    ~/.venvs/bellman-ntfield/bin/python jeffrey/octave_detune/stalls.py                          # where never-arrived runs stop (CPU)
    ~/.venvs/bellman-ntfield/bin/python jeffrey/octave_detune/page.py                            # report/octave_detune.html

Each run executes a frozen copy of `features.py`, `diagnostics.py` and `octdetune.py` (`snap/<hash>/`, made by `run.py`).
