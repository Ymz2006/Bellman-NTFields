# Lift scale: the travel-time lift and the push on larger plus mazes and Gibson floor plans

Study of 2026-11-17. Request (attached page: "2026-11-11 Travel-Time Lift", https://claude.ai/artifact/HBHUi6dRwWctNp1eDQMs9d):

> explore this lift idea. I want you to do more testing with the plus maze and gibson environments, especially the larger ones. Try the
> lift push and also lift approaches. I also want you to see why it doesn't work well in certain scenarios. Use all gpus, 12 hours

The study folders of that page (`jeffrey/lift_warp`, `warp_ntrl`, `chord_warp`, `maxmin_warp`) are not on this workstation (the registry
has no runs between 2026-11-08 and 11-11), so the lift and the push were rebuilt here from the page's description, on the scenes and the
planner every local maze study uses. Two things differ from the page and are part of what was tested:

* **Configuration space instead of the plane.** The local scenes are jeffrey/maze_scale's: a 0.5 x 0.2-cell rectangle robot,
  q = (x, y, phi) with phi = 0.25 theta / E, the exact signed clearance of the rectangle, ntrl-demo's speed label (margin 0.175 cells,
  floor 0.012). The page's scenes were point robots in the plane. In C-space the clearance gradient is not a unit vector
  (nu = |grad_q d| / E, median 1.07-1.16, 90th percentile 1.35-2.1), the case the page lists as not tested.
* **Scenes.** Plus mazes of 22, 30 and 40 cells a side (maze_scale's generator with a centred + of 6, 8 and 12 cells; the 22-cell one
  is the registered prepare of the earlier studies), the P-NTFields apartment at 32 and 64 cells a side (embed_warp's prepares) and three
  Gibson houses as floor plans at 0.25 m per cell: Allensville (43 cells), Lakeville (83), Klickitat (96).

## Method as implemented

    phi(q) = (q, h(q))        h = gain * G(d(q)) / E        G(d) = int_d^margin sqrt(1 / S(d')^2 - 1) dd'    (cells; 0 for d >= margin;
                                                                continued at the floor slope 83.3 down to d = -depth, default -offset)
    T = net(fourier([q0, h0]), fourier([q1, h1]))                one more column in ntrl-demo's Fourier matrix (private generator; B itself is
                                                                drawn from the global RNG as ntrl-demo draws it)
    loss = beta [ ntrl-demo's loss - lambda(epoch) mean(T over blocked pairs with T < bound) + mu ramp mean(relu(sqrt(S |grad T|) - 1)^2 at both ends) ]

`lift.py`: G is a table (G(offset) = 0.685 cells, G(-offset) = 3.60 cells, band length int dd / S = 0.748 cells). h enters the graph as
h(q0) + G'(d) grad_q d . (q - q0) (value and first derivative exact; no second derivatives are needed by the loss). Kinds: `none`
(ntrl-demo), `time` (the lift), `clear` / `dead` (controls), `vec` (the vector lift h n, three extra columns, full Jacobian by autograd).

`trainer.py`: ntrl-demo's loop step for step (as jeffrey/embed_warp/trainer.py). With `--lift none` and no push the loss of the first 60
epochs equals the registered ntrl-demo run of the same seed (20261112T063256Z-fd5d28b2) to every printed digit. Push schedule `page`:
lambda 0 at epoch 200, 1e-2 from 600 to 2,000, 3e-3 from 3,000 on; mu = 0.1; bound 3 box lengths (the largest true travel time between
configurations with clearance >= 0.1 cells is 0.9-1.9 box lengths on these scenes, so the bound never acts on a connected pair). The
one-sided term is faded in with the push's first ramp: at full weight from the first push epoch (`--mu-abrupt`, the first two runs) the
epoch-repeat rule rolled back 1,323 times in 100 epochs (8 minutes). `--mu1` adds a linear (exact) one-sided penalty.

`blocked.py` (stage `blocked`): for every training pair, whether its straight C-space segment crosses an obstacle (clearance below
-offset at a sample every 0.1 cells). 62-85 % of the training pairs are blocked. The page's 48-point check misses 0.2-1.6 % of them.

`scenes.py`: `House` = a Gibson house (~/Downloads/gibson_tiny/<name>/mesh_z_up.obj) cut by five horizontal planes 0.4-1.6 m above the
main floor, rasterized conservatively, gaps under 0.3 m sealed to find the inside, everything else an obstacle; signed distance table,
the rectangle's clearance as the minimum over a 9 x 5 lattice (as embed_warp's apartment). Lindenwood leaks through a wider gap and
was not used.

`house.py` (stage `house`) = jeffrey/embed_warp/prepare.py on a House. `pipeline.py` (stage `field`): train, plan (maze_scale's MPPI,
planning seeds 0 / 1 / 2, exact scoring), then `diag.py`: wall-layer fit S |grad T| by speed bin on held-out band points; travel times
against the prepare run's ground truth on 24,000 random (free configuration, goal) pairs; line-of-sight pairs (T* <= 1.05 D); the
detour index rho = (T - D) / (T* - D) on pairs with T* >= 1.3 D, by detour length; where the runs that did not arrive stopped; the same
field measures at epochs 100 ... 5,000 with planning seed 0 at 1,000 / 2,000 / 3,000.

Read-only forensics on registered checkpoints: `jumps.py` (the step in travel time a field holds across a wall), `heading.py` (radial,
planar and heading slope on line-of-sight pairs), `losscomp.py` (ntrl-demo's loss term by term). `run.py` registers the stages (sources
frozen in `snap/<hash>/`), `queue/runner.py` runs `queue/jobs.txt` (GPU 0 one run, GPU 1 two; shared with the octave_detune session),
`report.py` -> `report/runs.json`, `report/tables.md`; `figs.py`, `page.py` + `pagetext.py` + `pagelib.py` -> `report/lift_scale.html`.
