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

## Models trained with pairs inside obstacles (2026-11-19, second request)

Request, 14:50: "see the lift flat top whole interior method [of the Lift Variants page]. This one is working well because it samples inside
the obstacles. This is why it is able to re-create the accurate speed. 1) train this travel time, and only use its speed, sampling in the
boundaries. 2) base the bellman update on this method [...] skip the bellman loss when you are in the boundary. The bottom line is that
there is already a model that satisfies the eikonal equation, but does not get the viscosity solution. Try to add the bellman updates to
this model so that it works." Later: "I think it was only around 20% of pairs that have an end in the obstacle."

The Lift Variants study (`jeffrey/lift_warp`, point robot in 2-D) is not on this machine: its folders, result folders and registry entries
are absent. The models here are rebuilds on the SE(2) rectangle robot:

* `wholedata.py` (`run.py --stage data --prepared <prepare run> --label <name> --extra '--inside-pairs 0.2'`): whole-interior pairs.
  ntrl-demo's sampler with its two lower bounds dropped (d(x0) < margin, any d(x1)); `--inside-pairs f` thins the pairs with an end inside
  an obstacle to the share f, `--x0-inside f` the first ends. Scenes `plus22w` (unthinned: 68 % of the pairs with an end inside),
  `plus22w3`, `plus22w20` (20 %) in `queue/scenes.json`.
* Flat top = the lift's default depth (0.0175 cells); `--depth 0.3` continues the lift 0.3 cells into the walls (the page's ridge) and is
  used with lift_scale's `plus22in` / `plus40in` / `apt64in` pairs (the second end of 20 % of the pairs inside an obstacle).
* `speedcheck.py <run ids>` -> `report/speed.json`: S |grad T| of a trained field by clearance, the step cost from its speed table against
  the true one, and a collision check from the speed alone. `crossings.py <run ids>` -> `report/crossings.json`: the cost of wall
  crossings by a model's speed table against the true cost.
* Idea 1, the speed of a frozen model as the backup's step cost: `run.py --field <field run>` + `--bell-tau field`
  (`pairnet.SpeedTable`: log |grad_q T| on the clearance table's grid, 4 random partners per node; step cost = length x mean slowness,
  33 samples per radius). `--speed-check s`: a candidate step must keep the tabulated speed above s (replaces the geometric check);
  `--bell-nofree`: every step is a candidate. (`--bell-tau learned` uses the training network's own speed; not pursued.)
* Idea 2, the backup added to the model's own training: `--bell-free` applies the backup only to pairs with both ends out of obstacles;
  `--bell-nofree` (without `--bell-free`) applies it to every pair with no collision check, a step through an obstacle costing its
  floor-speed travel time. `--in-weight w` weights ntrl-demo's loss of the pairs with an end inside; `run.py --init <field run>` is a
  warm start from a finished field of the same network and seed.
* `queue/direct.py <gpu> '<label>|<scene>|<seed>|<arguments>' [--field id] [--init id] [--repeat-reason '...']` launches one run outside
  the runner (the queue was paused for these tests: `queue/PAUSE` and the `#PAUSED` lines of `queue/jobs.txt`); `queue/direct.txt` lists
  them for `status.py` and `report.py`.

Recipes (22-cell maze): idea 1 = `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-tau field --speed-check 0.04`
with `--field` = lift_scale's `lift_in` run 20261117T075605Z-a19f8f34 (label `bxt1_sR_scheck`); idea 2 = the same backup with
`--depth 0.3 --bell-free` on `plus22in` (label `bxt1_inF`).

## Findings

Findings page: https://claude.ai/artifact/1Q3exN3woGsjQDvNC4NSow (`report/pair_lift.html`, built by `page.py`; the text below is the page's, generated by `readme.py`).

pair_lift study · follow-up to the Travel-Time Lift (https://claude.ai/artifact/HBHUi6dRwWctNp1eDQMs9d) and Lift Variants
(https://claude.ai/artifact/Bu7xZpbydnqWwN9URoKEHp) pages · rectangle robot in SE(2), ntrl-demo’s MPPI, exact scoring · every number from
registered runs

### A coordinate of the pair does not fix the lift. The lifted length of a short step, used as its cost, does.

Two tasks were asked for: give the lifted network the length of the lifted curve of the straight segment between the two points (Task 1),
or simply a −1 / +1 flag when that segment is blocked (Task 2). As a network input, neither helps on the 22 × 22 plus maze: 59.2 % and
58.2 % of the queries solved against 68.2 % for lift + push and 65.0 % for ntrl-demo, with the detours still seen through. The lifted
network turns out to be able to hold the step across a wall once it is shown the true travel times, so what is missing is not an input but
a loss that selects the right field. The same quantity, the lifted length of a straight segment, supplies that loss when it is used as the
cost of a short step in a Bellman backup: **98.4 % of the queries solved** over 3 training seeds (97.6 / 100.0 / 97.5), 83.3 % of the
travel times within 10 % of the truth, 0.1 % of the detours seen through, with no push and no pair input. With candidate steps up to 5.3
cells long: 97.4 % (93.1 / 99.2 / 100.0), and detours 57 % too long in excess. On larger scenes the backup needs its weight scaled with
the step and the longer steps; with both, one training seed per scene: plus 30 81.5 % (ntrl-demo 56.8, lift + push 64.2, with octaves
76.2); plus 40 71.5 % (ntrl-demo 59.2, lift + push 63.2, with octaves 68.8); Allensville 43 100.0 % (ntrl-demo 90.3, lift + push 99.6,
with octaves 99.9); apartment 64 77.6 % with steps up to 0.24 box lengths (ntrl-demo 66.3, lift + push 73.0, with octaves 72.8); Lakeville
83 97.9 % (ntrl-demo 69.1, lift + push 76.5, with octaves 97.9); Klickitat 96 100.0 % with steps up to 0.24 box lengths (ntrl-demo 71.2,
lift + push 85.5, with octaves 98.7). A second request (Part 6): a lift trained with pairs inside obstacles has the right speed everywhere
without the right travel time. With only that model’s speed in the backup, as the cost of a step and as the collision check, **97.8 % of
the queries are solved** (98.8 / 95.3 / 99.4); with the backup added to that model’s own training, 88.8 % (80.0 / 98.3 / 88.2).

### Answers

**1. Is the diagnosis right: two points across a thin wall get similar embeddings, so no sharp change?** [true of the trained fields · not
a limit of the network] In the lift field the travel time between mirror configurations across a wall (0.8 cells apart) is 2.1 cells where
the true detour is 8 cells or more, and of a true step of 8.2 cells in the travel time to a goal it holds 0.8 (lift + push: 3.5). But the
same network regressed on the true travel times holds 0.95 of the step and plans 83.9 % (0.94 and 86.8 % with uniform samples). After
6,000 steps of the same regression it holds 0.29: the step is representable and slow to learn. What the lift lacks on this maze is a loss
term that asks for it.

**2. Task 2, the flag c = −1 / +1 for blocked pairs** [no] Flag alone: 40.7 % solved, travel times 30 times too long (a random Fourier
column puts the two ends of every blocked pair far apart from the start, at about 17 box lengths, and nothing brings them back). With the
push: 53.8 %. With the push and a cap that ties the two branches together: 59.2 %, 91 % of the detours seen through. With pairs sampled
inside obstacles and the rule “0 when an end is inside”: 58.6 %.

**3. Task 1, the lifted length of the straight segment appended** [no] Every way tried plans below lift + push and keeps the detours seen
through: saturated length 58.2 %; wall count 56.8 % (lift + push 68.2 %).

**4. Why do the pair inputs fail?** Three measured reasons. **The loss learns to ignore them.** The input jumps wherever the straight
segment starts to cross an obstacle, in open space, where the true travel time is continuous; the term that keeps the field continuous
there removes the dependence on the input within 100 epochs (zeroing the pair column of the trained flag network changes its travel times
by a median 0.33 cells), and the push then lifts blocked and clear pairs together. **A flag does not separate the two faces of a wall when
both are blocked from the goal**, which is the usual case in a maze. **An input that does separate them (a count, an unsaturated length)
puts cliffs in open space**: fitted to the truth, such fields change by more than twice the step on 63 % of the 0.1-cell steps across a
line where the count changes (0.4 % for the lift alone) and plan 63.5 % with 99.4 % of their travel times within 10 %.

**5. What makes the idea work?** [yes] Using the lifted length where it is exact: over a short straight step it is the step’s travel time,
and it is very large when the step crosses a wall. A Bellman backup over short free steps, T(q, g) = min over steps of [step cost + T(step
end, g)], raises exactly the configurations a see-through field gets wrong. Lift + backup on the 22-cell maze: **98.4 %** solved, 83.3 %
of travel times within 10 %, 0.1 % seen through; a true step of 8.2 cells across a wall is held as 9.6. With steps up to 5.3 cells: 97.4 %
solved, 63.1 % of travel times within 10 %.

**6. Does it hold on other seeds and scenes?** [seeds yes · houses yes · larger mazes and the apartment partly] 22-cell maze, 3 training
seeds: 97.6 / 100.0 / 97.5 % solved; with long steps 93.1 / 99.2 / 100.0 %. Larger scenes with the settings unchanged (steps of 0.66
cells): plus 40 64.3 %; apartment 64 64.4 %; Klickitat 96 67.7 %, no better than the references: the term fades as the scene grows (Part
4). With the weight scaled with the step and steps up to 5.3 cells: plus 30 **81.5 %** (ntrl-demo 56.8, lift + push 64.2, with octaves
76.2); plus 40 **71.5 %** (ntrl-demo 59.2, lift + push 63.2, with octaves 68.8); Allensville 43 **100.0 %** (ntrl-demo 90.3, lift + push
99.6, with octaves 99.9); apartment 64 **77.6 %** with steps up to 0.24 box lengths (ntrl-demo 66.3, lift + push 73.0, with octaves 72.8);
Lakeville 83 **97.9 %** (ntrl-demo 69.1, lift + push 76.5, with octaves 97.9); Klickitat 96 **100.0 %** with steps up to 0.24 box lengths
(ntrl-demo 71.2, lift + push 85.5, with octaves 98.7).

**7. Sampling inside obstacles** [no gain so far] With the backup as it is, pairs with one end inside an obstacle (20 % of the pairs)
break the recipe: 59.4 % solved on the 22-cell maze against 98.4 % on free-space pairs, 60.8 % against 73.7 % on the apartment. The backup
has nothing to hold on to for a goal inside a wall, and the travel times to such goals run away (median 24.6 box lengths). With the backup
kept to pairs with both ends free the see-through failure is gone (0.8 % seen through) and 88.8 % are solved, still below the run on
free-space pairs. Apartment, same change: 65.3 % (73.7 % on free-space pairs). Part 5.

**8. A model with the right speed: use only its speed (idea 1), or add the backup to it (idea 2)** [both remove the see-through · runs
still training] Idea 1: keep only the speed of a model trained with pairs inside obstacles and integrate it along a step as the backup’s
step cost. Idea 2: add the backup to that model’s own training, skipped inside obstacles. The flat-top lift with 20 % of the pairs inside
does not reproduce the Lift Variants result here (98.5 % of the detours seen through); the local model with the right speed everywhere is
the lift continued into the walls (63.3 % solved alone). Idea 1 with that model’s speed and a collision check made from the same speed:
**97.8 % solved**, 0.6 % seen through. Idea 2: **88.8 % solved** (80.0 / 98.3 / 88.2), 0.8 % seen through. Part 6.

### What was run

Everything is built on `jeffrey/lift_scale`, the local rebuild of the lift (the Lift Variants page was made on another machine; its code
is not on this workstation). The robot is the 0.5 × 0.2-cell rectangle of every local maze study, so a configuration is q = (x, y,
heading) and the lifted input is (q, h): the pair coordinate is a fifth input where the request says “4d”. Training pairs are ntrl-demo’s
(first end in the speed band next to an obstacle, second end anywhere free), 5,000 epochs, ntrl-demo’s loss unless a term is named. Main
scene: the 22 × 22 plus maze (walls 0.1 cells thick, 0.3 cells in configuration space), 240 queries × 3 planning seeds. Reference arms are
the lift-scale study’s registered runs of the same pipeline.

```
pair feature of (q0, q1), from the straight segment q0 → q1 sampled every 0.1 cells:
  flag     ψ = 1 if the segment enters an obstacle, else 0                                        (Task 2)
  length   e = Σ √(|dq|² + dh²) − |dq|   the lifted curve’s excess over the segment, in cells:   (Task 1)
           0 on a clear segment outside the speed bands, about 7.2 cells for every wall crossed; ψ = tanh(e / s), or e itself
  count    ψ = number of obstacles the segment enters
fed as      c0 = −ψ, c1 = +ψ   (or ψ at both ends)   through one more Fourier column, its own Fourier rows,
            or k copies appended to the 256-wide embedding before / after the instance norm
```

### Part 1 · Can the lifted network hold a step across a wall at all?

This is the premise of both tasks, so it was tested directly. Each network was regressed on the true travel time (the grid solver’s, 48
goals, 600,000 pairs) and then asked two things: how much of the true step in travel time it holds between mirror configurations on the
two faces of a wall, and how it plans.

After 6,000 steps every network holds about a third of the step (0.30 ntrl-demo, 0.29 lift, 0.33 lift + wall count) while 96.6 % of the
lift’s travel times are already within 10 %: the bulk of the field is learned first and the two faces of the walls last. Trained longer,
with most samples next to walls, the plain lifted network holds **0.95** of the step and plans 83.9 % (0.94 and 86.8 % with uniform
samples and 30,000 steps). ntrl-demo without the lift also holds the step (0.94) but plans 58.8 %, with 33.9 % of its plans through a
wall: the lift is what makes the band next to the wall learnable.

So on this maze, with walls 0.3 cells thick in configuration space, the two faces of a wall are not too close for the network. They are
too close for the losses used so far to tell apart. The pair inputs do not make the step easier to fit either: at 6,000 steps they hold
0.29–0.33 of it, and at 20,000 the flag is level with the lift (0.97, 85.3 % solved) while the count and the unsaturated length plan far
worse (63.5 %, 63.2 %, 61.8 %) although 99.4 % of their travel times are within 10 %.

The reason is in the chart above. A pair input that tells the two faces of a wall apart also changes in open space, wherever the straight
segment to the goal starts to cross one more obstacle. On 5.8 % of 0.1-cell steps in open space the wall count changes. The truth is
continuous there. The count-fed field is not: its travel time changes by a median 2.8 times the step across such a line (90th percentile
8.3), and a planner that follows the field stops at those cliffs (34.4 % of its plans stall). The flag has few such cliffs (2.1 %) because
the fit learned to ignore it.

#### The lifted length appended to the embedding: 2, 8, 32, 128 copies

The first form in the request. ψ = tanh of the up-and-down part of the lifted curve of the straight segment (cells; 0 on a clear segment
along which the lift is monotone, saturating at one wall), −ψ / 2 at one end and +ψ / 2 at the other, appended k times to the 256-wide
embedding, either before the instance norm or after it. After the norm the metric reads the copies directly: every 16 of them add 0.2 ×
amplitude × ψ to the travel time (amplitude 0.3; 0.075 for 128 copies, so that 32 and 128 add the same 2.7 cells). Same test as above:
fitted to the truth for 20,000 steps, then planned.

A few copies do no harm and no measurable good: 2 and 8 copies plan 86.2 % and 86.2 % before the norm, 83.8 % and 87.2 % after it, against
83.9 % for the lift alone (the fits without a pair input range from 83.9 to 86.8 %), and all hold 0.95–0.99 of the step. Many copies hurt:
32 copies 77.2 % before the norm and 54.3 % after it, 128 copies 40.8 % and 53.2 %, with 58.5 % and 46.8 % of the plans stalled and travel
times less accurate than the lift’s own (90.3 % and 92.8 % within 10 % against 99.8 %). The more of the embedding the feature occupies,
the more the travel time follows it, and it is a function of the straight segment: it jumps in open space where the segment starts to
cross a wall, and it says the same thing for a wall with a door next to it and a wall with a 20-cell detour.

### Part 2 · The pair feature as a network input, trained without the truth

The pair input can only make the right field representable. ntrl-demo’s loss accepts fields that pass straight through obstacles (lift
page, Part 3), so every arm here also gets the lift page’s push on pairs whose segment is blocked. One more term is needed because the
input is discontinuous: the eikonal term is a derivative at a point and cannot tie the field on the two sides of a line where the input
jumps. The cap does: over random free steps of up to 0.66 cells, the travel time may not change by more than the step’s own travel time.

**Flag alone.** 40.7 % solved. From the first epoch the blocked pairs sit at a travel time of about 11 box lengths: the −1 / +1 column
shifts the phases of all 128 Fourier rows, so the two ends of a blocked pair start out as unrelated points. ntrl-demo’s loss weights every
pair by exp(−T / 2), so at that travel time the pair no longer counts, and nothing corrects it. **Flag + push**: the same, 53.8 %.

**Flag + push + cap.** 59.2 % solved, travel times fine (81.0 % within 10 %), detours seen through (91 %). The cap brings the blocked
pairs down in the first 100 epochs, and it does so by making the network insensitive to the flag: with the pair column zeroed the trained
field’s travel times change by a median 0.51 cells at epoch 100, 0.42 at 2,000 and 0.33 at the end. The push then has no handle that
separates blocked from clear pairs: at epoch 2,000 clear pairs are 7 % long and the detours are still seen through. Even pairs blocked by
a single obstacle, where a flag is all the information needed, stay seen through (89 %, detour index 0.20).

**The lifted length, in the ways the request lists.** Saturated (tanh of the up-and-down part of the lifted curve, which is 0 on clear
segments): 58.2 %. Wall count: 56.8 %. One more thing was tried and dropped: with the derivative of the length feature inside the eikonal
term (so that the loss sees the total derivative of the field the planner reads) the loss diverges within 100 epochs, because the
feature’s gradient reaches 1,300 per box length where a segment grazes a wall. The length arms above leave that derivative out.

The step across a wall, which the pair input was meant to supply: with the flag, a true step of 8.2 cells is held as 0.7, and the travel
time between the two faces themselves is 1.5 cells (lift alone 0.8 and 2.1).

### Part 3 · The lifted length as the cost of a short step

The lifted curve of a straight segment has an exact meaning only when the segment is short: its length is then the travel time of the step
(along a wall normal the two are equal by construction of the lift), and a step that crosses a wall is longer than any detour worth
taking. That is a local cost, and a local cost with a minimum over steps is the Bellman equation of the travel time. The term added to
ntrl-demo’s loss:

```
for each end q of a training pair (q, g): 16 straight steps q → q′ of 0.33–0.66 cells that stay out of obstacles
    (2 down the field’s own gradient, 6 around it, 8 in random directions)
y = min over the steps of [ τ(q, q′) + T(q′, g) ]        τ = travel time of the step = ∫ ds / S along it;  no gradient through y
loss += mean (T(q, g) − y)²
and ntrl-demo’s own temporal-difference step is dropped where it crosses an obstacle
```

A field that descends into a wall has no free step that lowers it there, so y is above T and the term raises it, at that configuration
only; the correction then travels back along the detour one step per backup. No pair is pushed and nothing is asked of pairs that are
already consistent. ntrl-demo has a term of this kind, a 0.66-cell step down the gradient, but takes the step without a collision check:
at the start of training 38 % of those steps cross an obstacle and 25 % at the end, each one equating the travel time on one face of a
wall with the travel time on the other.

**Lift + backup: 98.4 % solved** (97.6 / 100.0 / 97.5 over 3 training seeds), 0.9 % of the plans through a wall and 0.7 % stalled; 83.3 %
of travel times within 10 % (median learned / true 1.032), line-of-sight pairs 91.0 %, 0.1 % of the detours seen through. The detours are
18 % long in excess (detour index 1.18). Queries with a true path of 32–64 cells, which no earlier lift arm solved without an inflated
field: 95 %.

Dropping the wall-crossing TD steps without adding the backup gives 49.3 % and a field 38 % short: ntrl-demo’s step was also what kept the
far field’s scale. The backup without the lift (first version): 72.6 % with 24.6 % of the plans through a wall, so the lift still does its
own job, the band next to the wall. Keeping ntrl-demo’s wall-crossing TD steps next to the backup: 98.5 %. Weight 3: 85.7 %. Half the step
radius: 79.3 %. **Longer steps** (lengths log-uniform between 0.33 and 5.3 cells): 97.4 % solved (93.1 / 99.2 / 100.0), no detour seen
through from epoch 600 on, but detours 57 % too long in excess and 63.1 % of travel times within 10 %: a long step carries the correction
farther per update, and the minimum over 16 sampled steps overestimates more when the steps are long. A pair input on top of the working
recipe does not help: flag 78.5 %, wall count on the first version 61.2 % against 96.8 %.

Across a wall (chart in Part 2): a true step of 4.0 cells is held as 4.4, 8.2 as 9.6, 13.5 as 16.0, and the travel time between the two
faces themselves, 0.8 cells apart, is 19.2 cells where the lift alone says 2.1. The network and its input are the lift’s, unchanged. The
embeddings of the two faces are far apart because the loss asked for it.

[figure: Travel time to a goal (star) deep inside a maze quadrant, at the goal’s heading; contours every cell, obstacles for the robot’s
centre in grey. In the truth the cells next to the goal but across a wall are far (lighter). Lift + push and the flag network see through
those walls; the backup field has them.]

#### The first version ran 28 % long, and why

The first two backup runs planned 96.8 % and 95.6 % but with every travel time about 28 % too long (6.5 % within 10 %), at weight 1 and at
weight 0.1 alike. The cause was the step’s travel time: a 6-point trapezoid rule on 1 / S. In the band 1 / S falls from 83 to 9 within
0.0175 cells of the wall, so a step that starts there is overestimated: 1.09 times the reference on average and 2.5 times at the 99th
percentile. On the trained field the band configurations were 0.04 cells below their targets and the open-space ones 0.14 cells above: an
equilibrium between an inflated target at the walls and the correct one elsewhere. The step time now uses the antiderivative of 1 / S
between samples (0.997 of the reference on average, 1.004 at the 99th percentile). The inflated fields plan as well as the correct one, as
over-priced walls did on the earlier pages; only the correct one has travel times that can be used.

### Part 4 · Larger scenes

First with nothing retuned and the step kept at 0.66 cells: plus 40 64.3 % (ntrl-demo 59.2, lift + push 63.2, with octaves 68.8);
apartment 64 64.4 % (ntrl-demo 66.3, lift + push 73.0, with octaves 72.8); Klickitat 96 67.7 % (ntrl-demo 71.2, lift + push 85.5, with
octaves 98.7). The backup has almost no effect there. Its residual T − y is a travel time in box lengths, and a step of 0.66 cells is 0.03
box lengths on the 22-cell maze, 0.0165 on the 40-cell maze, 0.010 on the apartment and 0.007 in the largest house: the squared residual,
and with it the term’s weight against ntrl-demo’s own loss, falls to 30 %, 12 % and 5 %. On the 40-cell maze the share of detours seen
through barely moves during training (83 % at epoch 600, 69 % at the end).

**Weight scaled with the step** (× (0.03 / radius)², so the term has the strength it has on the 22-cell maze): 40-cell maze 68.2 %,
detours seen through 65 % at epoch 100 and 33 % at the end, still falling. The correction moves 0.66 cells per backup and the longest
paths are over 100 cells. Steps of 0.03 box lengths instead (1.2 cells on the 40-cell maze): 68.7 %. **Scaled weight and steps up to 5.3
cells**: plus 30 81.5 % solved, 0 % of detours seen through, 57.9 % of travel times within 10 % (ntrl-demo 56.8, lift + push 64.2, with
octaves 76.2); plus 40 71.5 % solved, 0 % of detours seen through, 55.2 % of travel times within 10 % (ntrl-demo 59.2, lift + push 63.2,
with octaves 68.8); Allensville 43 100.0 % solved, 0 % of detours seen through, 94.5 % of travel times within 10 % (ntrl-demo 90.3, lift +
push 99.6, with octaves 99.9); apartment 64 76.5 % solved, 84 % of detours seen through, 34.7 % of travel times within 10 % (ntrl-demo
66.3, lift + push 73.0, with octaves 72.8); Lakeville 83 97.9 % solved, 1 % of detours seen through, 82.2 % of travel times within 10 %
(ntrl-demo 69.1, lift + push 76.5, with octaves 97.9); Klickitat 96 94.0 % solved, 52 % of detours seen through, 21.0 % of travel times
within 10 % (ntrl-demo 71.2, lift + push 85.5, with octaves 98.7).

On the 40-cell maze the long-step field has every detour (none seen through) and still stalls on 28.2 % of the plans. All of the queries
with a true path of 64–128 cells fail (0 % solved; 32–64 cells: 55 %), and short detours are over-priced: a true detour of 2.3 cells is
learned as 7.2, one of 4.6 as 12.9. The 30-cell maze shows the same two things at a smaller size: 0 % of detours seen through, 17.9 % of
the plans stalled, queries of 32–64 cells 36 % solved. **Houses.** Allensville 43 100.0 % solved against 99.6 % for lift + push and 99.9 %
with the octaves; Lakeville 83 97.9 % solved against 76.5 % for lift + push and 97.9 % with the octaves; Klickitat 96 94.0 % solved
against 85.5 % for lift + push and 98.7 % with the octaves. The backup reaches the level of the octave arm with the lift’s plain Fourier
features and no push. The largest house is the exception in its travel times: the field is 18 % short and 52 % of its detours are still
seen through, mostly the long ones (12–24 cells); a step of at most 5.3 cells is 0.055 box lengths there, against 0.24 on the 22-cell
maze. With the longest step set to 0.24 box lengths there too (23 cells): **100.0 % solved**, 82.0 % of travel times within 10 %, 5 % of
detours seen through. The longest step has to follow the size of the scene, not of the cell. On the apartment the backup plans a few
points above the earlier arms (one training seed) but the detours stay seen through (84 %), as under every arm of the lift-scale study;
its training also rolls back 1095 times. That scene is still open.

#### How far is the backup from what the network can do?

The representation test of Part 1, on the two scenes where the backup falls short. Regressed on the true travel times, the lifted network
plans 70.5 % on the 40-cell maze (74.9 % with the octaves), with 99.5 % of its travel times within 10 % and no query of 64–128 cells
solved. The backup field plans 71.5 % there. So on the 40-cell maze the backup plans as well as a field that was shown the right travel
times, and a better loss alone would not go further. The plans that fail stop in front of a wall they have to go around (28.2 % stalled,
0.2 % through a wall), and the fitted field holds only 0.55 of the true step across walls (0.94 with the octaves) after the 20,000 steps
that gave 0.95 on the 22-cell maze: here the step across a wall is the part the plain lifted network is slow to fit, and the octaves help.
On the apartment the fitted lifted network plans 83.1 % and holds 0.86 of the step (71.3 % and 0.99 with the octaves), against 77.6 % for
the backup, whose field holds almost none of the large steps there (a true step of 17 cells is held as 1.6): on the apartment the loss is
still the limit.

#### Backup and octaves together

The octaves of the lift-scale study (three more bands of Fourier rows at 2, 4 and 8 times ntrl-demo’s frequencies on the 22-cell maze,
scaled with the scene) are a representation fix; the backup is a selection fix. On the 40-cell maze each alone stops near 70 % (lift +
push + octaves 68.8 %, lift + backup 71.5 %). Together, with steps up to 5.3 cells: **92.1 % solved**, no detour seen through, and the
queries with a true path of 64–128 cells, which no earlier field solved (not even the ones fitted to the truth), 72 % solved. The share
solved was still rising at the end of training (76 % at epoch 3,000 on planning seed 0). On the apartment the same pair gives 74.4 %, with
76 % of detours still seen through. The same with the longest step scaled to the scene (0.24 box lengths instead of 5.3 cells; one
training seed): plus 22 **100.0 %** (ntrl-demo 65.0, lift + push 68.2, with octaves 69.0); plus 30 **92.4 %** (ntrl-demo 56.8, lift + push
64.2, with octaves 76.2); plus 40 **81.4 %** (ntrl-demo 59.2, lift + push 63.2, with octaves 68.8); apartment 64 **73.7 %** (ntrl-demo
66.3, lift + push 73.0, with octaves 72.8). On the 40-cell maze the scaled steps (up to 9.6 cells) are 10.7 points below the 5.3-cell
steps, and almost all of the difference is in the queries of 64–128 cells (0 % against 72 %); with one training seed each this is not yet
separated from seed-to-seed variation.

### Part 5 · Training pairs inside obstacles

Every run above trains on pairs with both ends in free space. The data variant of the earlier studies puts one end of 20 % of the pairs
inside an obstacle; there the speed label is at its floor (0.012) and the lift is continued 0.3 cells into the walls, so a wall is a ridge
up to 25 cells high in the lifted input. In the lift-scale study this variant did not make the walls felt (22-cell maze: lift 63.3 %
solved, lift + push 65.3 %; apartment: lift 63.6 %). Here it is combined with the backup, unchanged recipes, one training seed.

**With the backup as it is, obstacle samples break the recipe.** 22-cell maze: 59.4 % solved against 98.4 % on free-space pairs (three
training seeds, 97.6 / 100.0 / 97.5), 71 % of the detours seen through, and even the travel times of pairs in line of sight are within 10
% for only 70 % (91 %). Apartment: 60.8 % against 73.7 %, 86 % seen through (41 %).

Measured on the trained 22-cell field, on 20,000 of its own training pairs: the travel time to a goal inside a wall has a median of 24.6
box lengths, where the true value is at most about 1.1 (0.3 cells at the speed floor); ntrl-demo’s loss sits at 2 × 10⁻² from epoch 100 to
the end (4.7 × 10⁻⁴ on free-space pairs); the backup’s own residual on the pairs with both ends free is 13 times larger than in the run on
free-space pairs (1.0 × 10⁻³ against 7.7 × 10⁻⁵), and the pairs with a goal inside a wall, 19 % of the pairs, carry 36 % of the backup
term. A goal inside a wall has no free step that reaches it, so the backup at the other end of such a pair copies travel times to that
goal from neighbour to neighbour with nothing that fixes their level; ntrl-demo’s loss weight exp(−T / 2) lowers the loss of a pair whose
time drifts upward. That is the reading the first control tests.

**Control: the backup only for pairs with both ends free** (a pair with an end inside an obstacle is left to ntrl-demo’s loss): 88.8 %
solved, 0.8 % seen through, ntrl-demo’s loss back at 4.9 × 10⁻⁴. The see-through failure is gone, so it came from the backup on goals
inside walls. The run is still below the free-space pairs (97.6 / 100.0 / 97.5 % over three seeds): 11.2 % of the plans stall and the
travel times are within 10 % for 81 % of the pairs (83 %). With one seed, obstacle samples show no gain on this maze even with the backup
kept off them. **Control: ntrl-demo’s loss of the obstacle pairs times 0.02**, backup unchanged: 59.9 % solved, 87 % seen through. The
Task 2 rule on this data (flag −1 / +1 for blocked pairs, both 0 when an end is inside an obstacle; push and cap, no backup): 58.6 %
solved, 100 % seen through. Backup on free pairs only, apartment: 65.3 % solved against 73.7 % on free-space pairs (40 % against 41 % seen
through).

### Part 6 · A model with the right speed and the wrong travel time

The request: the “lift, flat top, whole interior” model of the Lift Variants page (https://claude.ai/artifact/Bu7xZpbydnqWwN9URoKEHp)
samples pairs inside obstacles, so it recovers the speed there; it satisfies the eikonal equation without being the viscosity solution.
Two uses were asked for: (1) keep only its speed and integrate it along a candidate step as the backup’s step cost, in place of the lifted
length; (2) add the backup to that model’s own training, skipped where a pair has an end inside an obstacle. Everything here is the
22-cell maze, training seed 87358, one training per arm unless a range is given.

#### Rebuilding the model on the local setup

The page’s code is not on this machine and its robot was a point in 2-D; here it is the rectangle in SE(2) of every local study. Three
lifts trained alone, with no push and no backup: **flat top, 20 %**: the lift that stops 0.0175 cells into an obstacle, on pairs whose two
ends may lie anywhere inside one, thinned so that 20 % of the pairs have an end inside (the share of the original); **flat top, 68 %**:
the same sampler unthinned (the rectangle’s obstacles are 0.3–0.6 cells thick, so most first ends land inside); **lift into the walls, 20
%**: the lift continued 0.3 cells into the walls (the page’s “ridge”), on the lift-scale study’s pairs with the second end of 20 % inside
an obstacle.

**The page’s result is not reproduced by the flat top at 20 %.** It solves 58.1 % and sees through 98.5 % of the detours, the level of the
lift on free-space pairs (55.0 %, 98.3 %); the page reported 65.5 % and 18 % on its plus maze. The reason is in the speed: inside
obstacles its S |∇T| is 0.23 (1 = exact; 0.01 with no pair inside, 0.80 on the page), so a wall costs 0.35 of its true crossing time. With
68 % of the pairs inside it reaches 0.58 and sees through 60 %, and pays as the page describes: travel times 12 % long, the predicted
speed 11 % too slow in open space. **The model with the right speed everywhere is the lift continued into the walls:** S |∇T| 1.01 inside
obstacles, 0.99 in the band, 1.00 in open space, a step through a wall at 0.97 of its true cost. It is the model of the request: the
eikonal equation holds inside and outside the walls, and it is not the viscosity solution (63.3 % solved, 54 % of the detours seen
through, travel times within 10 % for 81 % of the pairs).

The same holds on the 40-cell maze: the lift continued into the walls on that maze’s pairs (20 % with the second end inside) has S |∇T|
0.98 inside obstacles, 0.99 in the band and 0.97 in open space with the octaves of Part 4 (0.97 / 1.00 / 0.97 without), a wall crossing at
0.96 of its true cost, while it solves 61.2 % of the queries and sees through 99.8 % of the detours.

S |∇T|: median over 120,000 configurations uniform in the C-space box, the partner a random free configuration. Step cost: straight steps
of 0.015–0.03 box lengths (0.33–0.66 cells on the 22-cell maze, 0.6–1.2 on the 40-cell one), the cost integrated from the model’s speed
table (the one the backup reads) against the true step time; inside an obstacle the true speed is ntrl-demo’s floor, 0.012. The last two
columns are a collision check made from the speed alone: a step is called blocked when the tabulated speed falls below 0.04 at a sample;
every flagged free step lies within 0.05 cells of an obstacle.

#### Idea 1 · Only its speed, as the cost of a step in the backup

The frozen model’s predicted speed 1 / |∇T| is tabulated on the grid of the clearance table (32 nodes per cell, 49 headings, the log of
the slowness averaged over 4 random partners). The backup of Part 3 is unchanged except for the cost of a candidate step: its length times
the mean of the tabulated slowness along it, in place of the integral of the true speed (the lifted length). The network is trained on
free-space pairs, as in Part 3. Steps through obstacles are handled in one of two ways, neither of which reads the scene: **no check** (a
step through a wall is a candidate and costs what the speed says) or a **check from the speed** (a step is dropped when the tabulated
speed falls below a threshold at a sample).

**With the speed of the lift-into-walls model and the check from that speed: 97.8 % solved**, 0.6 % of the detours seen through, travel
times within 10 % for 74 % of the pairs, against 98.4 % (97.6 / 100.0 / 97.5), 0.1 % and 83 % for the true speed with the geometric
collision check. Per training seed: 98.8 / 95.3 / 99.4 %. The weakest seed loses 4.7 % of its plans, every one a plan that crosses a wall
on a 32–64-cell query (on planning seed 0: 14 plans, 0.15–0.27 cells deep; 0.15 cells is half a wall when the robot travels along it). The
true-speed runs lose plans the same way on the same queries (0.8 / 1.9 / 0.0 % of their plans; on planning seed 0, 3 and 5 plans). The
likely reading: the field is right on both sides of a thin wall and ntrl-demo’s planner, which does not test collisions, steps across. The
check itself has no gap: of 591,800 wall crossings sampled against the table that seed read, none passes at any threshold from 0.04 to
0.2. Without any check: 80.4 % solved (0 % of the 32–64-cell queries). The control with the true speed and no check: 99.2 % solved, 100 %
of the 32–64-cell queries, 0.0 % seen through. With the flat-top 68 % model’s speed and no check: 83.3 % solved and 0.8 % seen through;
every query with a true path up to 32 cells is solved (600 of 600) and none of the 120 longer ones; travel times are 27 % long, which is
that model’s speed error in open space carried into the field.

**What the step cost has to get right.** With the true speed the backup needs no collision check on this maze: 99.2 % solved, every
32–64-cell query, 0.0 % of the detours seen through. At ntrl-demo’s floor speed (0.012) an obstacle is a very slow strip: crossing the
thinnest face-on section of a wall (0.3 cells for the rectangle) costs 25 cells of travel time, more than a crossing saves anywhere in
this maze. The margin is thin. The lift-into-walls model prices a wall crossing at 0.84–0.97 of its true cost (5 % and 50 % quantiles over
54,687 crossing steps), the flat-top 68 % model at 0.40–0.71. With the lift-into-walls model’s speed and no check the field solved 96.2 %
at epoch 3,000 on planning seed 0 and ended at 80.4 %: at the end the detours of 12–24 cells are 24 % seen through and 0 % of the
32–64-cell queries are solved. The table is the only difference to the true-speed run, and taking the crossing steps out of the candidates
(next) restores the result, so the crossings are the cause: priced 3–16 % low, they open the longest detours. With the check made from the
same speed the walls are hard again and the under-pricing no longer matters: 97.8 % solved, 99 % of the 32–64-cell queries. What the
model’s speed costs is accuracy of the travel times: 6 % long at the median, within 10 % for 74 % of the pairs (83 % with the true speed).

In these runs ntrl-demo’s own TD term still skips steps that cross an obstacle by the geometric test, as in Part 3. With that term left as
ntrl-demo has it, so that no loss term tests a step against the scene: **99.6 % solved**, 0.7 % seen through, travel times within 10 % for
89 % of the pairs (one seed). The lift and the speed labels come from the scene as in every run of this page.

#### Idea 1 on the 40-cell maze

The large-scene recipe of Part 4 (lift + octaves + backup with candidate steps up to 5.3 cells), with the step cost and the check taken
from the speed of that maze’s own lift-into-walls model (with octaves): **80.1 % solved**, 0.6 % of the detours seen through, travel times
within 10 % for 70 % of the pairs, against 92.1 %, 0.0 % and 63 % with the true speed and the geometric check (one training seed each). On
seed 87358 the two agree up to 32 cells and differ on the longest queries: 73 % against 90 % of the 32–64-cell queries and 0 % against 72
% of the 64–128-cell ones. The 64–128-cell class also separated the two true-speed variants of Part 4 (0 % against 72 % of it solved, 81.4
% against 92.1 % overall), so one seed does not say whether the model’s speed loses it.

#### Idea 2 · The backup added to the model, skipped inside obstacles

The model with the right speed (lift into the walls, 20 % of the pairs with an end inside) trained with the backup of Part 3, the backup
applied only to pairs with both ends out of obstacles: **88.8 % solved over 3 training seeds (80.0 / 98.3 / 88.2), 0.8 % of the detours
seen through**, against 63.3 % and 54 % for the model alone. The backup removes nearly all of the see-through in every seed (2.2 / 0.2 /
0.1 %). What varies is the longest class of queries: 0 / 100 / 31 % of the 32–64-cell queries solved, against 96–100 % of all the shorter
ones. On seed 87358 the field solved 98.3 % at epoch 3,000 on planning seed 0 and lost the longest queries in the last 2,000 epochs; where
those plans stall, the learned travel time is 0.78–0.85 of the truth, so the far field is short again there. The same backup on free-space
pairs holds them in all three seeds (97.6 / 100.0 / 97.5 %), and idea 1 keeps the pairs inside obstacles out of the second training
altogether. With the backup applied to every pair and the geometric check it fails (59.4 %, Part 5).

**The backup on every pair, with no collision check**: a step through an obstacle is a candidate and costs its floor-speed travel time, so
the backup is the Bellman equation of the same passable-wall problem whose eikonal equation the pairs inside obstacles train, and a goal
inside a wall is reached by steps into it. 67.4 % solved, 15.0 % seen through, travel times within 10 % for 64 % of the pairs, 0 % of the
32–64-cell queries. That is worse than skipping the pairs inside obstacles, and it did not improve during training (60–61 % on planning
seed 0 at epochs 1,000, 2,000 and 3,000). No control was run on why.

The flat-top lift on the 20 % whole-interior pairs with the same backup: 68.2 % solved, 18.9 % seen through (58.1 % and 98.5 % without the
backup). Started from the trained model instead of from scratch: 87.8 % solved, 6.5 % seen through.

### What this does not show

The pair input was tested on the rectangle robot in SE(2), not on the point robot of the lift pages; the three failure mechanisms do not
depend on the robot, but the numbers are this setup’s. The pair inputs were trained with the push and the cap; a pair input under some
other selection term was tested only once, on top of the backup. The backup needs a collision check of short straight steps and the
clearance along them, the same information the lift and the push already use; its weight, number of candidate steps and radius were chosen
on the 22-cell maze, where a three times larger weight or half the radius lose 12–18 points, all on the longest queries. The backup’s step
time reads the clearance from a table (32 nodes per cell, within 1 % of the exact step time on average); the run with the exact rectangle
clearance was stopped at epoch 2,400 to free the GPU and is not a result.

### Every run

Study folder `jeffrey/pair_lift` (README.md has the setup and the run ids). Scores: 240 queries × MPPI planning seeds 0 / 1 / 2 on the 22
× 22 plus maze, exact full-robot clearance. Travel-time measures: 24,000 random (free configuration, goal) pairs against the grid solver.
One training of a recipe moves by 3–6 points between seeds.

## Tables (report.py; every number from registered runs)


### References (jeffrey/lift_scale, same pipeline)

| scene | arm | seed | id | solved % | wall % | stall % | T within 10 % | T/T* | LOS within 10 % | detour index | seen through % | S x slope innermost | rollbacks | minutes | arguments |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| plus22 | ntrl-demo | 87358 | 20261117T054519Z-9a3bc556 | 65.0 | 15.1 | 19.9 | 46.3 | 1.104 | 50.6 | 1.37 | 2.8 | 0.08 | 8 | 6.4 | `` |
| plus22 | lift | 87358 | 20261117T055249Z-d953394e | 55.0 | 0.0 | 45.0 | 25.5 | 0.821 | 43.5 | -0.20 | 98.3 | 1.00 | 0 | 13.1 | `--lift time` |
| plus22 | lift + push | 87358 | 20261117T061710Z-c95ca37c | 68.2 | 1.0 | 30.8 | 89.4 | 1.002 | 99.4 | 0.47 | 52.9 | 0.86 | 149 | 14.3 | `--lift time --push page` |
| plus22 | lift + push + octaves | 87358 | 20261117T093110Z-3477083e | 68.1 | 0.0 | 31.9 | 68.4 | 0.963 | 80.2 | -0.02 | 77.0 | 0.83 | 276 | 15.0 | `--lift time --push page --octaves 2,4,8` |
| plus22 | lift + strong push + octaves | 87358 | 20261117T101750Z-b9e07f67 | 83.8 | 13.6 | 2.6 | 16.3 | 1.157 | 21.5 | 1.85 | 0.8 | 0.86 | 353 | 26.3 | `--lift time --push const:1e-2 --octaves 2,4,8` |

### Fields trained without the truth

| scene | arm | seed | id | solved % | wall % | stall % | T within 10 % | T/T* | LOS within 10 % | detour index | seen through % | S x slope innermost | rollbacks | minutes | arguments |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| plus22 | bit | 87358 | 20261119T045331Z-3a5d2cfd | 40.7 | 1.8 | 57.5 | 0.0 | 29.530 | 0.0 | 81.75 | 0.0 | 1.08 | 0 | 10.4 | `--lift time --pair bit` |
| plus22 | bit_push | 87358 | 20261119T045311Z-00d33089 | 53.8 | 0.3 | 46.0 | 23.3 | 25.147 | 43.9 | 87.31 | 0.0 | 0.32 | 761 | 22.2 | `--lift time --pair bit --push page` |
| plus22 | ud_push_tri | 87358 | 20261119T050421Z-bfb97974 | 58.2 | 0.3 | 41.5 | 78.1 | 0.974 | 99.5 | 0.09 | 99.5 | 0.85 | 0 | 18.4 | `--lift time --pair len --pair-what updown --pair-scale 1 --pair-nograd --push page --tri1 0.5 --tri 10` |
| plus22 | bit_push_tri | 87358 | 20261119T045251Z-ffba52df | 59.2 | 0.0 | 40.8 | 81.0 | 0.978 | 99.7 | 0.16 | 90.9 | 0.85 | 0 | 33.5 | `--lift time --pair bit --push page --tri1 0.5 --tri 10` |
| plus22 | count_f025 | 87358 | 20261119T054033Z-7869b4de | 56.8 | 0.0 | 43.2 | 74.7 | 0.971 | 99.2 | 0.06 | 99.7 | 0.87 | 16 | 17.0 | `--lift time --pair count --pair-norm raw --pair-scale 1 --pair-freq 0.25 --push page --tri1 0.5 --tri 10` |
| plus22 | tdfree | 87358 | 20261119T055803Z-84929489 | 49.3 | 0.0 | 50.7 | 8.7 | 0.622 | 15.7 | -0.56 | 99.8 | 0.99 | 0 | 9.5 | `--lift time --td-free` |
| plus22 | bell1 | 87358 | 20261119T054423Z-f493358c | 96.8 | 0.4 | 2.8 | 6.5 | 1.276 | 12.2 | 2.12 | 0.0 | 1.23 | 404 | 47.9 | `--lift time --bell 1 --td-free` |
| plus22 | bell01 | 87358 | 20261119T054443Z-d462e732 | 95.6 | 0.6 | 3.9 | 1.1 | 1.282 | 1.1 | 2.18 | 0.7 | 1.06 | 569 | 48.8 | `--lift time --bell 0.1 --td-free` |
| plus22 | bell1_count | 87358 | 20261119T062503Z-b895cf51 | 61.2 | 0.6 | 38.2 | 50.5 | 1.096 | 73.6 | 1.46 | 1.9 | 1.03 | 67 | 31.9 | `--lift time --pair count --pair-norm raw --pair-scale 1 --pair-freq 0.25 --bell 1 --td-free` |
| plus22 | ntrl_bell1 | 87358 | 20261119T063223Z-72f35cfb | 72.6 | 24.6 | 2.8 | 36.1 | 1.170 | 60.8 | 2.22 | 0.0 | 0.08 | 129 | 41.0 | `--lift none --bell 1 --td-free` |
| plus22 | bxt1 | 87358 | 20261119T065723Z-b148dcbc | 97.5 | 0.8 | 1.7 | 84.5 | 1.026 | 95.7 | 1.31 | 0.1 | 0.99 | 483 | 32.1 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table` |
| plus22 | bxt1_r15 | 87358 | 20261119T073003Z-d342af01 | 79.3 | 2.8 | 17.9 | 94.8 | 1.013 | 96.9 | 1.06 | 5.7 | 1.00 | 221 | 31.9 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-r 0.015 --bell-table` |
| plus22 | bxt3 | 87358 | 20261119T071343Z-95f04a0c | 85.7 | 0.0 | 14.3 | 82.4 | 0.974 | 84.3 | 1.12 | 0.2 | 1.01 | 136 | 54.9 | `--lift time --bell 3 --td-free --bell-k 16 --bell-cone 6 --bell-table` |
| apt64 | bxt1 | 87358 | 20261119T080213Z-45290212 | 64.4 | 0.0 | 35.6 | 19.9 | 0.699 | 69.6 | -0.17 | 99.9 | 1.00 | 5 | 27.0 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell` |
| plus40 | bxt1 | 87358 | 20261119T073553Z-b75cfe2c | 64.3 | 0.0 | 35.7 | 71.0 | 1.051 | 68.8 | 0.38 | 69.0 | 0.99 | 0 | 58.1 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell` |
| Klickitat | bxt1 | 87358 | 20261119T080904Z-37a261b0 | 67.7 | 0.0 | 32.3 | 33.4 | 0.859 | 43.5 | 0.49 | 51.1 | 0.95 | 0 | 44.2 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell` |
| plus22 | bxt1 | 1 | 20261119T082923Z-d2107e1a | 97.6 | 1.9 | 0.4 | 77.4 | 1.050 | 83.9 | 1.20 | 0.0 | 1.02 | 518 | 33.3 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table` |
| plus22 | bxt1 | 2 | 20261119T083413Z-cad0f4e4 | 100.0 | 0.0 | 0.0 | 88.0 | 1.019 | 93.4 | 1.04 | 0.1 | 0.98 | 469 | 56.2 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table` |
| plus22 | bxt1L | 87358 | 20261119T090253Z-f1b7016b | 100.0 | 0.0 | 0.0 | 66.7 | 1.029 | 91.3 | 1.65 | 0.0 | 1.02 | 522 | 46.0 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-long 8` |
| plus40 | bxt1n | 87358 | 20261119T085333Z-aba5379e | 68.2 | 0.1 | 31.7 | 83.3 | 1.037 | 94.0 | 0.70 | 32.7 | 1.00 | 0 | 57.2 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm` |
| apt64 | bxt1nL | 87358 | 20261119T094903Z-7dc8cc67 | 76.5 | 0.0 | 23.5 | 34.7 | 0.844 | 76.8 | 0.20 | 84.2 | 1.06 | 1095 | 47.8 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long 8` |
| plus40 | bxt1nL | 87358 | 20261119T093043Z-7840e590 | 71.5 | 0.2 | 28.2 | 55.2 | 1.079 | 81.7 | 1.92 | 0.0 | 1.06 | 140 | 72.3 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long 8` |
| plus40 | bxt1b | 87358 | 20261119T095053Z-b953c20d | 68.7 | 0.0 | 31.3 | 15.2 | 1.167 | 20.6 | 1.76 | 3.5 | 1.01 | 157 | 60.9 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table` |
| Klickitat | bxt1nL | 87358 | 20261119T103714Z-b462ad74 | 94.0 | 0.0 | 6.0 | 21.0 | 0.820 | 25.7 | 0.48 | 52.4 | 1.15 | 90 | 40.6 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long 8` |
| Allensville | bxt1nL | 87358 | 20261119T111813Z-dfeaeac0 | 100.0 | 0.0 | 0.0 | 94.5 | 0.976 | 94.7 | 0.92 | 0.2 | 0.98 | 21 | 39.0 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long 8` |
| Lakeville | bxt1nL | 87358 | 20261119T105154Z-d5ee7cbe | 97.9 | 0.0 | 2.1 | 82.2 | 0.962 | 87.1 | 0.87 | 1.2 | 1.07 | 56 | 67.2 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long 8` |
| plus30 | bxt1nL | 87358 | 20261119T104333Z-1723575f | 81.5 | 0.6 | 17.9 | 57.9 | 1.061 | 88.2 | 1.81 | 0.0 | 1.06 | 372 | 83.6 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long 8` |
| plus22 | bxt1L | 1 | 20261119T115733Z-800e84b9 | 93.1 | 5.0 | 1.9 | 59.8 | 1.063 | 86.8 | 1.54 | 0.0 | 1.06 | 601 | 46.8 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-long 8` |
| plus22 | bxt1_tdall | 87358 | 20261119T120743Z-1f8414c4 | 98.5 | 0.7 | 0.8 | 87.8 | 1.005 | 94.6 | 1.24 | 0.1 | 0.96 | 368 | 57.8 | `--lift time --bell 1 --bell-k 16 --bell-cone 6 --bell-table` |
| plus22 | bxt1L | 2 | 20261119T115923Z-a989e6a9 | 99.2 | 0.0 | 0.8 | 62.8 | 1.057 | 88.7 | 1.52 | 0.0 | 1.03 | 492 | 75.1 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-long 8` |
| plus40 | bxt1nK | 87358 | 20261119T124434Z-58b1f4e1 | 66.7 | 0.0 | 33.3 | 67.4 | 1.062 | 79.2 | 0.83 | 20.0 | 1.01 | 0 | 42.6 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long 8 --bell-keep` |
| Klickitat | bxt1nLB | 87358 | 20261119T130554Z-6a9ff52a | 100.0 | 0.0 | 0.0 | 82.0 | 0.937 | 86.7 | 0.82 | 4.9 | 1.02 | 145 | 68.5 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long box:0.24` |
| apt64 | bxt1nL_oct | 87358 | 20261119T133953Z-64e2c6ec | 74.4 | 0.5 | 25.1 | 23.1 | 0.819 | 44.8 | 0.14 | 75.9 | 1.00 | 590 | 44.1 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long 8 --octaves cell` |
| plus22 | bxt1K | 87358 | 20261119T131443Z-8d5d2253 | 95.6 | 0.8 | 3.6 | 79.1 | 1.025 | 94.1 | 1.31 | 0.1 | 1.00 | 501 | 86.4 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-long 8 --bell-keep` |
| apt64 | bxt1nLB | 87358 | 20261119T142403Z-68ac3e37 | 77.6 | 0.0 | 22.4 | 43.2 | 0.896 | 84.5 | 0.30 | 74.0 | 1.06 | 1385 | 51.2 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long box:0.24` |
| plus40 | bxt1nL_oct | 87358 | 20261119T141444Z-818d4b0f | 92.1 | 0.0 | 7.9 | 62.7 | 1.003 | 76.9 | 1.71 | 0.0 | 0.96 | 38 | 67.6 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long 8 --octaves cell` |
| plus30 | rec | 87358 | 20261119T153624Z-59bc35da | 92.4 | 0.0 | 7.6 | 58.4 | 1.041 | 79.8 | 1.75 | 0.0 | 1.03 | 77 | 43.5 | `--lift time --octaves cell --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long box:0.24` |
| plus40 | rec | 87358 | 20261119T153604Z-aa584d33 | 81.4 | 0.7 | 17.9 | 52.7 | 1.003 | 65.4 | 1.89 | 0.0 | 0.98 | 50 | 70.2 | `--lift time --octaves cell --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long box:0.24` |
| plus22 | bxt1_bit | 87358 | 20261119T153143Z-f5cf29c0 | 78.5 | 7.4 | 14.2 | 86.7 | 1.010 | 96.1 | 1.09 | 4.5 | 0.96 | 132 | 76.1 | `--lift time --pair bit --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table` |
| plus22 | rec | 87358 | 20261119T162013Z-854fbc8f | 100.0 | 0.0 | 0.0 | 64.9 | 1.053 | 88.0 | 1.54 | 0.0 | 1.07 | 163 | 43.6 | `--lift time --octaves cell --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long box:0.24` |
| plus22in | bxt1_in | 87358 | 20261119T164633Z-5691e027 | 59.4 | 0.0 | 40.6 | 68.6 | 1.031 | 70.1 | 0.34 | 70.7 | 1.11 | 148 | 65.1 | `--lift time --depth 0.3 --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table` |
| apt64 | rec | 87358 | 20261119T164823Z-ce32f880 | 73.7 | 1.2 | 25.1 | 35.4 | 0.893 | 54.9 | 0.59 | 40.6 | 0.97 | 2820 | 87.7 | `--lift time --octaves cell --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long box:0.24` |
| apt64in | rec_in | 87358 | 20261119T170414Z-9c98b304 | 60.8 | 0.4 | 38.8 | 20.2 | 0.761 | 52.0 | -0.02 | 85.7 | 1.07 | 5675 | 86.3 | `--lift time --depth 0.3 --octaves cell --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long box:0.24` |
| plus22in | bxt1_inF | 87358 | 20261119T181623Z-b50acc81 | 88.2 | 0.0 | 11.8 | 68.0 | 1.073 | 83.7 | 1.37 | 0.1 | 1.00 | 347 | 39.8 | `--lift time --depth 0.3 --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-free` |
| plus22in | bit_in_zero | 87358 | 20261119T175203Z-ed651028 | 58.6 | 0.0 | 41.4 | 80.4 | 0.990 | 98.9 | 0.13 | 99.5 | 0.85 | 0 | 65.5 | `--lift time --depth 0.3 --pair bit --pair-inside zero --push page --tri1 0.5 --tri 10` |
| plus22in | bxt1_inW | 87358 | 20261119T183044Z-2eac66cd | 59.9 | 0.0 | 40.1 | 78.3 | 0.992 | 91.6 | 0.24 | 86.5 | 1.05 | 9103 | 84.9 | `--lift time --depth 0.3 --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --in-weight 0.02` |
| apt64in | rec_inF | 87358 | 20261119T185754Z-5320e16a | 65.3 | 0.4 | 34.3 | 25.0 | 0.865 | 33.0 | 0.63 | 40.2 | 1.02 | 1029 | 103.2 | `--lift time --depth 0.3 --octaves cell --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long box:0.24 --bell-free` |
| plus22w | liftW | 87358 | 20261119T195829Z-e518c164 | 58.1 | 2.1 | 39.9 | 32.7 | 1.123 | 10.0 | 0.45 | 60.0 | 1.15 | 0 | 8.4 | `--lift time` |
| plus22 | bxt1_sA_soft | 87358 | 20261119T200727Z-1a875cd5 | 83.3 | 0.0 | 16.7 | 2.8 | 1.267 | 4.6 | 1.89 | 0.8 | 1.02 | 242 | 39.0 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-tau field --bell-nofree` |
| plus22w20 | liftW | 87358 | 20261119T203113Z-543f4eb9 | 58.1 | 0.0 | 41.9 | 71.5 | 0.969 | 92.9 | 0.08 | 98.5 | 1.01 | 0 | 27.2 | `--lift time` |
| plus22w20 | liftW_bellF | 87358 | 20261119T203131Z-05a03b99 | 68.2 | 0.0 | 31.8 | 83.4 | 1.015 | 90.2 | 0.95 | 18.9 | 1.05 | 4 | 65.9 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-free` |
| plus22 | bxt1_nofree | 87358 | 20261119T204647Z-f0bcdbe0 | 99.2 | 0.0 | 0.8 | 80.1 | 1.039 | 92.1 | 1.29 | 0.0 | 1.00 | 382 | 35.1 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-nofree` |
| plus22 | bxt1_sR_scheck | 87358 | 20261119T210918Z-1975cf26 | 99.4 | 0.1 | 0.4 | 58.8 | 1.088 | 65.9 | 1.32 | 0.3 | 0.99 | 369 | 66.8 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-tau field --speed-check 0.04` |
| plus22 | bxt1_sR_soft | 87358 | 20261119T212216Z-ceae1199 | 80.4 | 0.0 | 19.6 | 70.4 | 1.071 | 78.0 | 1.21 | 3.3 | 0.96 | 378 | 36.0 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-tau field --bell-nofree` |
| plus22in | bxt1_inF | 1 | 20261119T213740Z-3ce3b0af | 80.0 | 0.0 | 20.0 | 85.5 | 1.041 | 94.9 | 1.17 | 2.2 | 1.01 | 459 | 69.9 | `--lift time --depth 0.3 --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-free` |
| plus22in | bxt1_inF | 2 | 20261119T221226Z-2c694064 | 98.3 | 0.0 | 1.7 | 88.1 | 0.983 | 92.3 | 0.92 | 0.2 | 1.00 | 379 | 58.3 | `--lift time --depth 0.3 --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-free` |
| plus40in | liftR | 87358 | 20261119T221757Z-d6b6c73f | 60.8 | 0.0 | 39.2 | 38.4 | 0.871 | 60.5 | -0.35 | 99.8 | 0.96 | 0 | 31.5 | `--lift time --depth 0.3` |
| plus22 | bxt1_sR_scheck | 1 | 20261119T221757Z-4c30b931 | 98.8 | 0.0 | 1.2 | 81.5 | 1.045 | 92.7 | 1.28 | 0.7 | 0.96 | 550 | 60.1 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-tau field --speed-check 0.04` |
| plus22in | bxt1_in_soft | 87358 | 20261119T224830Z-bfc24999 | 67.4 | 0.0 | 32.6 | 64.2 | 1.053 | 76.5 | 1.28 | 15.0 | 1.02 | 12 | 70.6 | `--lift time --depth 0.3 --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-nofree` |
| plus40in | liftR_oct | 87358 | 20261119T224932Z-fa50bf49 | 61.2 | 0.0 | 38.8 | 64.2 | 0.937 | 90.6 | -0.02 | 99.8 | 0.96 | 0 | 22.2 | `--lift time --depth 0.3 --octaves cell` |
| plus40 | bxt1nL_oct_sR_scheck | 87358 | 20261119T231158Z-2c310b6b | 80.1 | 0.6 | 19.3 | 70.3 | 1.029 | 86.3 | 1.49 | 0.6 | 0.98 | 53 | 85.6 | `--lift time --octaves cell --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long 8 --bell-tau field --speed-check 0.04` |
| plus22 | bxt1_sR_scheck | 2 | 20261119T231806Z-52f30f00 | 95.3 | 4.7 | 0.0 | 80.7 | 1.049 | 87.9 | 1.05 | 0.9 | 0.99 | 471 | 36.2 | `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-tau field --speed-check 0.04` |
| plus22in | bxt1_inF_init | 87358 | 20261119T235919Z-d467855e | 87.8 | 2.4 | 9.9 | 84.2 | 1.002 | 94.7 | 0.90 | 6.5 | 0.95 | 559 | 32.6 | `--lift time --depth 0.3 --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-free` |
| plus22 | bxt1_sR_scheck_tdall | 87358 | 20261120T003158Z-7480e308 | 99.6 | 0.0 | 0.4 | 89.1 | 1.054 | 94.4 | 1.19 | 0.7 | 0.98 | 392 | 34.6 | `--lift time --bell 1 --bell-k 16 --bell-cone 6 --bell-table --bell-tau field --speed-check 0.04` |

### Supervised fits to the true travel time (representation test)

| scene | arm | seed | id | steps | band share | solved % | wall % | stall % | T within 10 % | T/T* | detour index | seen through % | step across walls / true (true >= 3 cells) | arguments |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| plus22 | fit_lift | 87358 | 20261119T051532Z-bb58a134 | 6000 | 0.0 | 66.7 | 0.0 | 33.3 | 96.6 | 0.980 | 0.89 | 8.9 | 0.29 | `--lift time` |
| plus22 | fit_count_f025 | 87358 | 20261119T051752Z-0744cf09 | 6000 | 0.0 | 60.6 | 0.0 | 39.4 | 95.9 | 1.001 | 0.96 | 7.7 | 0.33 | `--lift time --pair count --pair-norm raw --pair-scale 1 --pair-freq 0.25` |
| plus22 | fit_ntrl | 87358 | 20261119T052308Z-3456d8e3 | 6000 | 0.0 | 51.2 | 48.8 | 0.0 | 95.8 | 1.014 | 1.02 | 4.6 | 0.30 | `--lift none` |
| plus22 | fit_bit | 87358 | 20261119T052113Z-100d3f8e | 6000 | 0.0 | 64.4 | 0.1 | 35.4 | 96.5 | 1.002 | 0.95 | 6.4 | 0.29 | `--lift time --pair bit` |
| plus22 | fit_count_f1 | 87358 | 20261119T052358Z-5c15bcc3 | 6000 | 0.0 | 62.5 | 0.0 | 37.5 | 95.3 | 1.002 | 0.97 | 5.9 | 0.33 | `--lift time --pair count --pair-norm raw --pair-scale 1 --pair-freq 1` |
| plus22 | fit_lenraw_box | 87358 | 20261119T052538Z-ce29b9ea | 6000 | 0.0 | 60.3 | 1.0 | 38.8 | 95.1 | 1.001 | 0.95 | 6.5 | 0.33 | `--lift time --pair len --pair-what length --pair-norm raw --pair-scale box` |
| plus22 | fit_udraw_f025 | 87358 | 20261119T052448Z-e443510f | 6000 | 0.0 | 59.3 | 0.0 | 40.7 | 95.8 | 1.000 | 0.96 | 7.1 | 0.32 | `--lift time --pair len --pair-what updown --pair-norm raw --pair-scale 7.2 --pair-freq 0.25` |
| plus22 | fit_lift_oct8 | 87358 | 20261119T052723Z-c0689232 | 6000 | 0.0 | 68.5 | 0.0 | 31.5 | 97.7 | 1.000 | 0.98 | 1.6 | 0.50 | `--lift time --octaves 2,4,8` |
| plus22 | fit_count_same | 87358 | 20261119T052633Z-b8f27f92 | 6000 | 0.0 | 59.3 | 0.0 | 40.7 | 95.6 | 1.004 | 0.97 | 7.4 | 0.33 | `--lift time --pair count --pair-norm raw --pair-scale 1 --pair-freq 0.25 --pair-sign same` |
| plus22 | fit_lift_30k | 87358 | 20261119T052903Z-f76e30f7 | 30000 | 0.0 | 86.8 | 10.8 | 2.4 | 99.8 | 1.000 | 1.00 | 0.0 | 0.94 | `--lift time --steps 30000` |
| plus22 | fit_liftB | 87358 | 20261119T053023Z-fbac8fa0 | 20000 | 0.6 | 83.9 | 7.1 | 9.0 | 99.8 | 1.007 | 1.02 | 0.1 | 0.95 | `--lift time --band 0.6 --steps 20000` |
| plus22 | fit_countB_f1 | 87358 | 20261119T053313Z-cd5cac82 | 20000 | 0.6 | 63.5 | 2.1 | 34.4 | 99.4 | 1.004 | 1.01 | 0.2 | 0.81 | `--lift time --pair count --pair-norm raw --pair-scale 1 --pair-freq 1 --band 0.6 --steps 20000` |
| plus22 | fit_count_f1_30k | 87358 | 20261119T052843Z-bec17a64 | 30000 | 0.0 | 66.1 | 1.1 | 32.8 | 99.5 | 1.007 | 1.01 | 0.2 | 0.88 | `--lift time --pair count --pair-norm raw --pair-scale 1 --pair-freq 1 --steps 30000` |
| plus22 | fit_ud_tanh | 87358 | 20261119T053723Z-30de5ac0 | 6000 | 0.0 | 63.8 | 4.4 | 31.8 | 96.1 | 1.000 | 0.97 | 5.8 | 0.28 | `--lift time --pair len --pair-what updown --pair-scale 1` |
| plus22 | fit_countB_f025 | 87358 | 20261119T053703Z-5ecaf932 | 20000 | 0.6 | 64.4 | 5.8 | 29.7 | 99.6 | 1.003 | 1.00 | 0.2 | 0.88 | `--lift time --pair count --pair-norm raw --pair-scale 1 --pair-freq 0.25 --band 0.6 --steps 20000` |
| plus22 | fit_vec | 87358 | 20261119T053943Z-57d9ef4c | 6000 | 0.0 | 66.0 | 0.0 | 34.0 | 96.6 | 1.002 | 1.00 | 7.5 | 0.31 | `--lift vec` |
| plus22 | fit_udrawB_f025 | 87358 | 20261119T060753Z-0bb19ddd | 20000 | 0.6 | 63.2 | 4.9 | 31.9 | 99.6 | 1.012 | 1.03 | 0.0 | 0.90 | `--lift time --pair len --pair-what updown --pair-norm raw --pair-scale 7.2 --pair-freq 0.25 --band 0.6 --steps 20000` |
| plus22 | fit_lenrawB_box | 87358 | 20261119T061143Z-97f421cd | 20000 | 0.6 | 61.8 | 1.4 | 36.8 | 99.5 | 0.997 | 0.97 | 0.1 | 0.80 | `--lift time --pair len --pair-what length --pair-norm raw --pair-scale box --band 0.6 --steps 20000` |
| plus22 | fit_bitB | 87358 | 20261119T061533Z-40b85496 | 20000 | 0.6 | 85.3 | 1.1 | 13.6 | 99.8 | 1.004 | 1.02 | 0.0 | 0.97 | `--lift time --pair bit --band 0.6 --steps 20000` |
| plus22 | fit_lift_oct8B | 87358 | 20261119T061923Z-2413d0b1 | 20000 | 0.6 | 84.6 | 13.1 | 2.4 | 99.9 | 0.997 | 1.00 | 0.0 | 0.99 | `--lift time --octaves 2,4,8 --band 0.6 --steps 20000` |
| plus22 | fit_ntrlB | 87358 | 20261119T062243Z-472600a2 | 20000 | 0.6 | 58.8 | 33.9 | 7.4 | 99.7 | 0.995 | 0.98 | 0.0 | 0.94 | `--lift none --band 0.6 --steps 20000` |
| plus40 | fit_liftB | 87358 | 20261119T132734Z-79b0398f | 20000 | 0.6 | 70.5 | 0.0 | 29.5 | 99.5 | 0.993 | 0.96 | 0.7 | 0.55 | `--lift time --band 0.6 --steps 20000` |
| apt64 | fit_liftB | 87358 | 20261119T133053Z-9120bb20 | 20000 | 0.6 | 83.1 | 0.0 | 16.9 | 97.5 | 1.001 | 1.00 | 0.3 | 0.86 | `--lift time --band 0.6 --steps 20000` |
| plus40 | fit_lift_octB | 87358 | 20261119T133344Z-4f5a9775 | 20000 | 0.6 | 74.9 | 0.0 | 25.1 | 99.7 | 1.007 | 1.02 | 0.1 | 0.94 | `--lift time --octaves 3.62,7.24,14.48 --band 0.6 --steps 20000` |
| apt64 | fit_lift_octB | 87358 | 20261119T133703Z-0236554c | 20000 | 0.6 | 71.3 | 0.1 | 28.5 | 99.8 | 1.000 | 1.00 | 0.0 | 0.99 | `--lift time --octaves 5.79,11.58,23.17 --band 0.6 --steps 20000` |
| plus22 | fit_pre2B | 87358 | 20261119T144123Z-ac36c9f0 | 20000 | 0.6 | 86.2 | 7.4 | 6.4 | 99.8 | 1.000 | 1.00 | 0.0 | 0.98 | `--lift time --pair len --pair-what updown --pair-scale 1 --pair-place pre --pair-copies 2 --band 0.6 --steps 20000` |
| plus22 | fit_pre8B | 87358 | 20261119T145213Z-4427705b | 20000 | 0.6 | 86.2 | 0.1 | 13.6 | 99.7 | 1.000 | 1.00 | 0.0 | 0.99 | `--lift time --pair len --pair-what updown --pair-scale 1 --pair-place pre --pair-copies 8 --band 0.6 --steps 20000` |
| plus22 | fit_pre32B | 87358 | 20261119T150303Z-eb8f94ed | 20000 | 0.6 | 77.2 | 2.4 | 20.4 | 99.3 | 1.000 | 1.00 | 0.0 | 0.98 | `--lift time --pair len --pair-what updown --pair-scale 1 --pair-place pre --pair-copies 32 --band 0.6 --steps 20000` |
| plus22 | fit_post2B | 87358 | 20261119T151543Z-54fa631c | 20000 | 0.6 | 83.8 | 10.6 | 5.7 | 99.6 | 0.993 | 0.99 | 0.0 | 0.95 | `--lift time --pair len --pair-what updown --pair-scale 1 --pair-place post --pair-copies 2 --pair-amp 0.3 --band 0.6 --steps 20000` |
| plus22 | fit_pre128B | 87358 | 20261119T151353Z-15181721 | 20000 | 0.6 | 40.8 | 0.7 | 58.5 | 90.3 | 1.001 | 0.98 | 1.0 | 0.91 | `--lift time --pair len --pair-what updown --pair-scale 1 --pair-place pre --pair-copies 128 --band 0.6 --steps 20000` |
| plus22 | fit_post8B | 87358 | 20261119T151933Z-6ed591c5 | 20000 | 0.6 | 87.2 | 6.0 | 6.8 | 98.8 | 1.009 | 1.03 | 0.0 | 0.96 | `--lift time --pair len --pair-what updown --pair-scale 1 --pair-place post --pair-copies 8 --pair-amp 0.3 --band 0.6 --steps 20000` |
| plus22 | fit_post32B | 87358 | 20261119T152223Z-e2b820a9 | 20000 | 0.6 | 54.3 | 0.3 | 45.4 | 92.8 | 0.993 | 0.98 | 0.2 | 0.90 | `--lift time --pair len --pair-what updown --pair-scale 1 --pair-place post --pair-copies 32 --pair-amp 0.3 --band 0.6 --steps 20000` |
| plus22 | fit_post128B | 87358 | 20261119T152243Z-9d706657 | 20000 | 0.6 | 53.2 | 0.0 | 46.8 | 92.6 | 0.991 | 0.96 | 0.1 | 0.88 | `--lift time --pair len --pair-what updown --pair-scale 1 --pair-place post --pair-copies 128 --pair-amp 0.075 --band 0.6 --steps 20000` |

### Jobs that did not finish

* `bell03_k16|plus22|87358|--lift time --bell 0.3 --bell-k 16 --td-free` -> 20261119T063343Z-fbef26cb
* `bx1|plus22|87358|--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6` -> 20261119T064233Z-a3d6270c
* `bxt1nLB|plus40|87358|--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long box:0.24` -> 20261119T152334Z-44d3ecfa
* `bxt1nLB|plus30|87358|--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long box:0.24` -> 20261119T153123Z-d85dea7e
* `rec|Klickitat|87358|--lift time --octaves cell --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table --bell-r cell --bell-norm --bell-long box:0.24` -> 20261119T185634Z-ff87a1d9
