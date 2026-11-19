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

The working recipe: `--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6 --bell-table` (22-cell maze), plus
`--bell-r cell --bell-norm --bell-long 8` on larger scenes.

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
cells long: 100.0 %, and detours 65 % too long in excess. On larger scenes the backup has to be rescaled and the gain is smaller: plus 40
71.5 % (ntrl-demo 59.2, lift + push 63.2, with octaves 68.8); apartment 64 76.5 % (ntrl-demo 66.3, lift + push 73.0, with octaves 72.8).

### Answers

**1. Is the diagnosis right: two points across a thin wall get similar embeddings, so no sharp change?** [true of the trained fields · not
a limit of the network] In the lift field the travel time between mirror configurations across a wall (0.8 cells apart) is 2.1 cells where
the true detour is 8 cells or more, and of a true step of 8.2 cells in the travel time to a goal it holds 0.8 (lift + push: 3.5). But the
same network regressed on the true travel times holds 0.95 of the step and plans 83.9 % (0.94 and 86.8 % with uniform samples). After
6,000 steps of the same regression it holds 0.29: the step is representable and slow to learn. What the lift lacks on this maze is a loss
term that asks for it.

**2. Task 2, the flag c = −1 / +1 for blocked pairs** [no] Flag alone: 40.7 % solved, travel times 30 times too long (a random Fourier
column puts the two ends of every blocked pair far apart from the start, at about 17 box lengths, and nothing brings them back). With the
push: 53.8 %. With the push and a cap that ties the two branches together: 59.2 %, 91 % of the detours seen through.

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
of travel times within 10 %, 0.1 % seen through; a true step of 8.2 cells across a wall is held as 9.6. With steps up to 5.3 cells: 100.0
% solved, 66.7 % of travel times within 10 %.

**6. Does it hold on other seeds and scenes?** [seeds yes · larger scenes partly] 22-cell maze, 3 training seeds: 97.6 / 100.0 / 97.5 %
solved; with long steps 100.0 %. Larger scenes with the settings unchanged (steps of 0.66 cells): plus 40 64.3 %; apartment 64 64.4 %;
Klickitat 96 67.7 %, no better than the references: the term fades as the scene grows (Part 4). With the weight scaled with the step and
steps up to 5.3 cells: plus 40 **71.5 %** (ntrl-demo 59.2, lift + push 63.2, with octaves 68.8); apartment 64 **76.5 %** (ntrl-demo 66.3,
lift + push 73.0, with octaves 72.8).

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
own job, the band next to the wall. Weight 3: 85.7 %. Half the step radius: 79.3 %. **Longer steps** (lengths log-uniform between 0.33 and
5.3 cells): 100.0 % solved, no detour seen through from epoch 600 on, but detours 65 % too long in excess and 66.7 % of travel times
within 10 %: a long step carries the correction farther per update, and the minimum over 16 sampled steps overestimates more when the
steps are long. A pair input on top of the working recipe does not help: wall count on the first version 61.2 % against 96.8 %.

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
paths are over 100 cells. **Scaled weight and steps up to 5.3 cells**: plus 40 71.5 % solved, 0 % of detours seen through, 55.2 % of
travel times within 10 % (ntrl-demo 59.2, lift + push 63.2, with octaves 68.8); apartment 64 76.5 % solved, 84 % of detours seen through,
34.7 % of travel times within 10 % (ntrl-demo 66.3, lift + push 73.0, with octaves 72.8).

On the 40-cell maze the long-step field has every detour (none seen through) and still stalls on 28.2 % of the plans. All of the queries
with a true path of 64–128 cells fail (0 % solved; 32–64 cells: 55 %), and short detours are over-priced: a true detour of 2.3 cells is
learned as 7.2, one of 4.6 as 12.9. On the apartment the backup plans a few points above the earlier arms (one training seed) but the
detours stay seen through (84 %), as under every arm of the lift-scale study; its training also rolls back 1095 times. That scene is still
open.

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

### Jobs that did not finish

* `bell03_k16|plus22|87358|--lift time --bell 0.3 --bell-k 16 --td-free` -> 20261119T063343Z-fbef26cb
* `bx1|plus22|87358|--lift time --bell 1 --td-free --bell-k 16 --bell-cone 6` -> 20261119T064233Z-a3d6270c
