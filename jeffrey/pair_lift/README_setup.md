# Pair lift: a coordinate of the pair on the travel-time lift

Study of 2026-11-18/19. Request (attached pages: "2026-11-11 Travel-Time Lift", https://claude.ai/artifact/HBHUi6dRwWctNp1eDQMs9d, and
"2026-11-18 Lift Variants", https://claude.ai/artifact/Bu7xZpbydnqWwN9URoKEHp):

> Check the previous work on lifts, and why they fail. The main reason is that at the end of the day, the neural network sees 2 points on
> opposite sides of a thin wall in 2d but the third dimension is 0. so it gets as input (x,y,0) and (x+0.01,y,0). They get mapped to very
> similar embeddings, the network is unable to represent a sharp change in travel time between them [...] My idea is that 2 embeddings
> should also depend on their geodesic distance on the lifted space [...] drawing a straight line between the 2 points which will result
> in a curve and find the length of this curve. [...] Task 1) Try several ways of appending the distance that make logical sense, for
> example normalization + append d copies of it (2, 8, 32, 128...), a fourier embedding, [...] the training data should come in pairs so
> you can directly append this distance to the input, making it a 4d input instead of the lifted 3d. [...] sampling in free space only and
> also sampling in obstacle space. Task 2) [...] Just give a 4d input, x,y,h and one more called c. for pairs where you can not go from
> one to the other linearly, just make c for one of them -1 and the other 1. If one of them is in the obstacle space, you can just set
> both to 0. Use all gpus, keep going until you have results of this idea working

The Lift Variants page was made on another machine (its `jeffrey/lift_warp` folders are not here). Everything below is built on
`jeffrey/lift_scale`, the local rebuild of the lift: the SE(2) rectangle robot of every local maze study (0.5 x 0.2 cells,
q = (x, y, phi)), ntrl-demo's speed label (margin 0.175 cells, floor 0.012), ntrl-demo's MPPI and the exact scorer. The lifted input is
therefore (x, y, phi, h) and the pair coordinate is a fifth input where the request says "4d".

## Method as implemented

    T(q0, q1) = metric( enc(fourier([q0, h(q0), c0])), enc(fourier([q1, h(q1), c1])) )        ntrl-demo's network and metric, one more Fourier column

`pairnet.py`. The pair coordinate comes from the straight C-space segment between the two configurations (a sample every 0.1 cells,
clearance from a table of the scene's exact clearance: 16 nodes per cell, 49 headings, trilinear; the same in training and planning):

* `--pair bit` (Task 2): psi = 1 if the segment crosses an obstacle (clearance below -0.0175 cells at an interior sample), else 0;
  c0 = -psi, c1 = +psi. `--pair-inside zero` adds the request's rule (both 0 when an end is inside an obstacle).
* `--pair len` (Task 1): the lifted curve of the segment, e = sum of sqrt(|dq|^2 + dh^2) - |dq| in cells (`--pair-what excess`; `length`
  keeps the whole length; `updown` keeps sum |dh| - |h(end) - h(start)|, the part that goes up and comes back down, which is 0 for every
  clear segment along which h is monotone). A clear segment outside the speed bands has e = 0; every wall crossed adds 2 G_max = 7.2 cells.
  psi = tanh(e / s) (`--pair-norm tanh|frac|log|raw`, `--pair-scale s`).
* Sign: `anti` (-psi, +psi: the two ends are 2 psi apart in the new coordinate) or `same` (psi at both ends, the literal "append the
  distance to both inputs").
* Place: `input` (a Fourier column b_c, N(0, 1) truncated times `--pair-freq`), `rows` (r of the 128 Fourier rows see only c, at
  frequencies freq x 2^[-2, 2]), `pre` / `post` (k copies of amp c / 2 appended to the 256-wide embedding before / after the instance norm;
  after the norm, 16 copies add exactly 0.2 amp psi to T).

`pairtrain.py`: jeffrey/lift_scale/trainer.py's loop (ntrl-demo's loss, beta, epoch-repeat rule; the push of the lift page on pairs whose
segment is blocked, the blocked flags of jeffrey/lift_scale's stage `blocked`) plus the triangle cap:

    for each end q of a pair (q, g): q' = q + delta, delta uniform in a ball of radius 0.03 box lengths, kept if the step stays out of obstacles
    v = relu(|T(q, g) - T(q', g)| - tau),  tau = |delta| x mean of 1 / S along the step            (tau >= T*(q, q'))
    loss += beta (tri mean(v^2) + tri1 mean(v))

`pairpipe.py` = jeffrey/lift_scale/pipeline.py with these options (train, MPPI planning seeds 0 / 1 / 2, exact scoring, lift_scale's
diagnostics). `run.py` registers a run (sources frozen in `snap/<hash>/`), `queue/runner.py` runs `queue/jobs.txt` (GPU 0 one run, GPU 1
two), `status.py` prints the queue with headline numbers, `pairjumps.py` measures the step a field holds across a wall.
With `--pair none` and no cap the pipeline is lift_scale's (100-epoch loss 5.7605e-04 on the RTX 3090 against 5.7672e-04 of the registered
lift run made on the RTX 5070).

Added during the study (the pair inputs did not work; see the findings):

* `repfit.py` (stage `repfit`, labels `fit_*`): the representation test. Each network is regressed on the true travel time (the prepare
  run's grid solver, 48 goals, 600,000 pairs) and then measured: travel times, the step held across walls, planning.
* `pairtrain.bellman_term` (`--bell w --bell-k K --bell-cone c --bell-r r --bell-long L --bell-norm --bell-table`) and `--td-free`:
  the free-step Bellman backup. For each end q of a training pair (q, g), K straight steps q -> q' that stay out of obstacles (2 down the
  field's gradient, c around it, the rest random; lengths in [r / 2, r], or log-uniform up to L r), y = min [tau(q, q') + T(q', g)] with
  no gradient, loss += w mean (T - y)^2. tau is the step's travel time, integrated with the antiderivative of 1 / S between samples
  (`pairtrain.Slowness`; the 6-point trapezoid rule of the first runs, `--bell-trapz`, overestimates steps that start deep in a band).
  `--bell-table` reads the clearance along the steps from `pairnet.ClearTable` (32 nodes per cell). `--bell-r cell` keeps the radius at
  0.66 cells on any scene and `--bell-norm` multiplies the weight by (0.03 / r)^2. `--td-free` drops ntrl-demo's own TD term for pairs
  whose step crosses an obstacle.
* `forensics.py` (sensitivity to the pair column, cliffs in open space, detours by wall count, step-time quadrature, the backup's
  residual on a trained field -> `report/forensics.json`), `pairjumps.py` (`report/jumps.json`), `report.py` (`report/runs.json`,
  `report/tables.md`), `figs.py`, `page.py` + `pagetext.py` (the findings page, on jeffrey/lift_scale's page assets), `pagecheck.py`,
  `readme.py` (this file = README_setup.md + the page's text + report/tables.md).

More options of the backup: `--bell-long L` draws the lengths of the candidate steps log-uniformly up to L radii (`box:f` = up to f box
lengths) and `--bell-keep` leaves the candidates around the descent direction at their short lengths. `--octaves cell` adds lift_scale's
three octave bands scaled with the scene. Obstacle-space sampling uses the scenes `plus22in` / `plus40in` / `apt64in` of `queue/scenes.json`
(embed_warp's data variant: one end of 20 % of the pairs inside an obstacle) with `--depth 0.3` (the lift continued 0.3 cells into the
walls); a configuration inside an obstacle has no free step, so the backup leaves it to ntrl-demo's loss.

Recipes:

* 22-cell maze (label `bxt1`): `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table`
* larger scenes (label `bxt1nL`): the same plus `--bell-r cell --bell-norm --bell-long 8`
* with octaves (label `bxt1nL_oct`): the same plus `--octaves cell`; label `rec` has `--bell-long box:0.24` in place of `--bell-long 8`
