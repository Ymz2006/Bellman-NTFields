# Sharp features on top of the octave Fourier embedding

Request (2026-11-18 01:55 EST): "I want to revisit the idea of inputting sharp features on top of the sin/cos fourier
embeddings. Idea 1) (sin(x))^2 and (cos(x))^2 and even to higher powers. You will test this idea by mixing in higher
exponents of sin and cos. I want you to try power of 2,4,8,16. 2 ways to try: 1) you remove the previous ones to keep the
same number of sin/cos that gets multiplied. 2) you can add more, but make sure to have a baseline to see if there is
benefits it comes directly from the new powers instead of more sin/cos. Idea 2) f(x) = (8/π) arcsin(cos(x)) You will try
this sharp feature and see if it can be better. Idea 3) max(0, tanh(x − h)) · f(x) You will add this function, which
allows for more specific features at specific moments. You can sample for different values of h in the tanh. Try many
things with each idea to ensure that it really doesn't work, be comprehensive."

Status: sharp-feature sweep stopped at the user's request at 12:00 (81 trainings, 80 succeeded); follow-up with the midpoint
loss in `jeffrey/sharp_midpoint`. Findings page: https://claude.ai/artifact/2rFDQ85CEsAJgrbXsWECZq (rebuild: `report.py`,
`page.py`; republish `report/sharp_features.html`).

## Findings

**No sharp feature beats the octave.** The best tie it, as plain inputs of the same count do. Even powers lose at every p and in
every form; the triangle as written loses when it replaces the rows; the gate as written loses a few points. Paired change in
success against the exact-octave trainings of the same seed (points, ± SE from the training noise; SD of one run 3.03 on 22×22, 1.83
on 13×13):

| spec | 22×22 (3 seeds) | 13×13 (2 seeds) |
|---|---|---|
| `Adup@all` copy of the 128 rows (control, 512 inputs) | +3.0 ± 2.0 | −0.4 ± 1.5 |
| `Asc@new128` 128 new octave rows ("more sin/cos", 512 inputs) | +2.2 ± 2.0 | +2.0 ± 1.5 |
| `Atri@all` triangle pair (8/π) arcsin(sin, cos) added | +1.6 ± 2.0 | −3.3 ± 1.5 |
| `Agtri@allx2` gate × triangle, 2 random h per row, added | −4.5 ± 2.0 | −0.4 ± 1.5 |
| `Rtri@all` triangle replacing every row | −7.6 ± 2.0 | −11.2 ± 1.5 |
| `Rpowmix@half` sin^p, cos^p, p = 2…16 mixed, half the rows | −14.0 ± 2.0 | −22.2 ± 1.5 |
| `Apowmix@all` sin^p, cos^p, p mixed, added to every row | −17.2 ± 2.0 | −18.8 ± 1.5 |

Sweep, 22×22, seed 87358 (one training each; the octave of that seed averages 51.9 %; SD of one run 3.0 points):
- Powers replacing half the rows: p = 2 / 4 / 8 / 16 29.2 / 33.5 / 31.5 / 22.2 %; every row 12.2 / 12.2 / 12.1 / 5.1 % (mix 10.3 %);
  top octave only p = 4 / 16 23.9 / 31.0 %; a quarter of the rows (mix) 38.8 %; half the directions each with its own
  (sin, cos, sin⁴, cos⁴) 28.5 %. Sign-keeping |s|^p sgn(s) on half the rows: p = 4 41.9 %, p = 16 34.4 %.
- Powers added (512 inputs): p = 2 / 4 / 8 / 16 29.3 / 27.6 / 27.8 / 26.2 %, mix 26.7 %, all four (1,280 inputs) 24.4 %, lower octaves
  only 27.6 %, standardized p = 16 40.4 %, mix at 0.3× / 0.1× size 41.4 / 43.3 %. Plain controls: copy 51.7 %, 2× frequency pair
  42.2 %, 128 new rows 43.3 %, 512 new rows (1,280 inputs) 54.0 %.
- Triangle: every row (amplitude 4) 38.6 %, amplitude 1 41.2 %, sin / cos × 3.27 (same RMS, gain control) 53.1 %, half the rows 48.5 %,
  added 43.6 %, unit added 52.1 %.
- Gates (added): as written 44.4 %, unit triangle 43.9 %, triangle pair 43.1 %, sin / cos 49.7 %, gate alone 42.6 %, h at grid vertices
  50.7 %, slope 3 / 0.3 38.2 / 41.5 %, top octave × 8 offsets 32.4 %, on the triangle rows 39.6 %.
- Grid-axis rows (crests / edges on the wall lines, 16–20 extra inputs): plain 52.8 %, triangle 52.9 %, sin⁴ 41.4 %, sin¹⁶ 39.2 %, gate
  alone with an edge on every wall line 34.9 %, gate × triangle 32.6 %.

Mechanism (`ripple.py`: T along the true paths; `stalls.py`: where never-arrived runs stop; `usage.py`: weight on the sharp columns):
- The sharp features fit the local loss better and plan worse. Powers added: held-out speed correlation 0.85 vs 0.80–0.82, near-wall
  slope 0.76 vs 0.62–0.71, runs through a wall 27 vs about 130 of 720, lower final loss; but 11 local minima per long true path
  (octave 0.3–0.5) and 70 % of runs never arrive. Along the true paths the eikonal residual |S |∇T| − 1| drops to 0.16 (octave
  0.44–0.51) while 21 % of the path length runs uphill toward the goal (octave under 1.1 %): a sawtooth T satisfies the eikonal loss,
  which constrains only |∇T|. Planner level: runs stopped in an open corridor 98 (powers added) and 182 (sin⁸ on every row) of 240
  against 6–22 for the octave.
- Even powers are mirror-symmetric about every crest of their row. sin² z and cos² z are both ½ ∓ ½ cos 2z: added, they fold T
  (17.6 % of the path uphill, −22.6 points); the full pair sin 2z, cos 2z, a superset of their information, does not (0.6 %, −9.7).
  Replacing half the rows, the sign-keeping power removes the local minima (0.6 vs 4.5 per path).
- The network uses the sharp columns (weight share near or above their share of the input; grid-axis powers three times above it), so
  they are not ignored; turned down to 0.1 of their size the powers stop folding and gain nothing over plain inputs.
- Before training (`screen.py`): no feature gives the embedding more resolution near walls than in the open (arc-length ratio
  0.97–1.02 for every spec, the grid-axis rows included), because the slow band lies 0.15–0.36 cell from the wall line (half the wall
  plus the robot's half extent, which depends on its heading) while the crests sit on the line.

Not run (queue stopped): seeds 1 / 2 for the variants that tied on seed 87358 (`Rtri@half`, `Rsc3.27@all`, `Asc@new512`, `Aharm2@all`,
`Apowmix.g0.1@all`, `Rspow4@half`, `Atri1@all`, `Agtri.grid@allx2`, `Agsc@all`, `Atri@axis`, `Asc@axis`), seed 2 of the 13×13 core,
and the 13×13 sweep. Incidents: the octave validation run on the shared 5070 ran out of memory (`20261118T072230Z-0ddf5937`, noted;
replaced by the 40-epoch loss-for-loss check); a runner bug launched `Rpowmix@quarter` on both GPUs at 07:54, the 5070 copy was
killed (`20261118T125415Z-f8093d9b`, noted) and the runner fixed (per-GPU job list, global cap, adoption of running jobs).

## Reading of the request

- "The sin/cos Fourier embeddings" = the grid-synced octaves (`jeffrey/grid_sync` mode `octave`, the best embedding so
  far: 94.8 % on 13×13, 48.1 % on 22×22 against ntrl-demo's 59 / 23 %). x in the three formulas is the phase of a row,
  z_i = 2π B_i · (x, y, φ).
- Idea 1, way 1 ("remove the previous ones to keep the same number"): the pair (sin z_i, cos z_i) of some rows becomes
  (sin^p z_i, cos^p z_i); the input stays 256 wide and the network is ntrl-demo's. Way 2 ("add more"): the powered pairs
  are appended (512 inputs, or 1,280 with every power), and the baselines are the same number of extra plain sin / cos:
  128 new octave rows (`Asc@new128`, "more sin/cos"), a copy of the 128 rows (`Adup@all`, the same information in the
  same width) and the rows at twice their frequency (`Aharm2@all`; sin² z = (1 − cos 2z) / 2).
- Idea 2: f(z) = (8/π) arcsin(cos z) is a triangle wave of amplitude 4; its sine partner (8/π) arcsin(sin z) keeps the
  pair's phase. Tested as written and at amplitude 1, plus sin / cos scaled to the triangle's RMS (gain control).
- Idea 3: max(0, tanh(a (z − h))) · f(z) with f of idea 2, a = 1, the offsets h sampled (the phase of a random point of
  the maze, of a random grid vertex, or of every grid line), appended to the octaves.

## Setup

Mazes, data, network, losses, optimizer, 5,000 epochs, planner and scorer are those of `jeffrey/maze_scale` /
`jeffrey/grid_sync` / `jeffrey/octave_detune`, unchanged: 13×13 (prepare `20261101T192504Z-13cf1d04`, 200 queries ×
planning seeds 0/1/2 = 600 runs) and 22×22 (`20261101T193403Z-cea05796`, 240 queries, 720 runs). Every spec starts from
the exact octave matrix of the same training seed (`features.octave`, bit-identical to grid_sync / octave_detune).

Reference: every earlier training of the same maze and seed with the exact octave embedding (octave_detune's octave and
its nulls ×1.0001 / ×0.9999, plus grid_sync's run where it ran on the other GPU): 3 runs per seed on 13×13, 3–4 on
22×22. Training noise: SD of one run 1.83 points (13×13) and 3.03 points (22×22). A spec's paired change is its success
minus the mean of its seed's reference runs, averaged over seeds.

- Replace specs keep ntrl-demo's network bit for bit (`sharpnet.py`): the same parameters, initialization and data
  order as the octave run of the seed; only the replaced columns of the input differ.
- Add specs widen the three layers that read the input (`pe_gate[0]`, `pe_gate[1]`, `encoder[1]`) to 256 + m inputs
  (+196,608 parameters at m = 256); the first block's residual carries the 256 base features. All add specs of one width
  share initialization and data order, so a sharp feature and its plain control differ only in the features.
- Pipeline check: 40 training epochs through this code reproduce the registered 3090 octave run
  (`20261117T055515Z-f128d826`) loss for loss, both with spec `octave` and with `Rsc1@all` (the replace path with plain
  values).

Spec grammar (`features.py`): `R<fn>@<sel>` replaces, `A<fn>@<sel>[x<k>]` appends; fn = `pow<p>`, `pow<p>z`
(standardized), `spow<p>` (signed: sgn(s) |s|^p, no fold), `powmix` (p = 2, 4, 8, 16 in turn), `powcat` (all four),
`tri<A>` (default A = 4, the formula as written), `sc<g>`, `harm<k>`, `gtri<A>` (gate × triangle), `gtrip`, `gsc`
(gate × sin/cos), `gate` (gate alone); options `.partner`, `.grid`, `.a<slope>`; sel = `all`, `half`, `quarter`,
`top` (the level whose half period is one cell), `low`, `new<M>`, `axis` (rows on the coordinate axes at every octave
level: the only rows whose crests, kinks and gate edges lie exactly on the wall lines).

## Reproduce

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/sharp_features/run.py \
        --prepared results/registered_20261101T193403Z-cea05796 --seed 87358 --spec Rpowmix@half
    setsid nohup ~/.venvs/bellman-ntfield/bin/python jeffrey/sharp_features/queue/runner.py > jeffrey/sharp_features/queue/runner.out 2>&1 &
    OMP_NUM_THREADS=4 ~/.venvs/bellman-ntfield/bin/python jeffrey/sharp_features/screen.py     # input mapping only, CPU
    ~/.venvs/bellman-ntfield/bin/python jeffrey/sharp_features/report.py                      # report/summary.json, tables.md

Each run executes a frozen copy of `features.py`, `sharpnet.py`, `diagnostics.py` and `sharp.py` (`snap/<hash>/`).
Run ids: `queue/state.json` (done map `gpu|n|seed|spec -> run id`); registry search `Sharp features`.
