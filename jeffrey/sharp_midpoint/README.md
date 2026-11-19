# Sharp features with the travel-time midpoint loss

Request (2026-11-18 11:58 EST), after `jeffrey/sharp_features` found that no sharp feature beats the octave embedding and
that the sharp ones fold the learned travel time into a sawtooth the eikonal loss accepts: "drop everything, try the sharp
features with the midpoint loss".

Status: done (2026-11-18 16:14 EST; 20 trainings, all succeeded). Findings page: the "With the midpoint loss" section of
https://claude.ai/artifact/2rFDQ85CEsAJgrbXsWECZq (built by `jeffrey/sharp_features/page.py`, which reads `report/summary.json`).

## Findings

**The midpoint loss does not rescue the sharp features.** It helps plain fields and barely moves the sharp ones, whose travel time
stays folded. 22×22, training seeds 87358 / 1 / 2 (one seed where noted); each run is forked at epoch 2,000 of its base run; change
against the exact octave with the same loss on the same seed (± SE, sd of one run 3.03 points × sqrt(2 / k)):

| base field | success without → with the loss | with the loss vs octave + loss | share of the true paths uphill, without → with |
|---|---|---|---|
| `octave` exact octave | 46.3 → 54.9 % | – | 2.2 → 0.5 % |
| `Asc@new128` 128 extra plain octave rows (control) | 48.6 → 58.1 % | +3.2 ± 2.5 | 1.4 → 0.5 % |
| `Agtri@allx2` gate × triangle, as written | 41.9 → 44.2 % | −10.7 ± 2.5 | 1.3 → 1.1 % |
| `Rtri@all` triangle replacing every row, as written | 38.8 → 41.5 % | −13.4 ± 2.5 | 2.6 → 1.8 % |
| `Rpowmix@half` sin^p, cos^p mixed, half the rows | 32.4 → 35.0 % | −20.0 ± 2.5 | 13.7 → 12.7 % |
| `Apowmix@all` sin^p, cos^p mixed, added | 29.2 → 32.3 % | −22.6 ± 2.5 | 18.9 → 17.9 % |
| `Apow16@axis` sin¹⁶ on the grid-axis rows (seed 87358) | 39.2 → 44.6 % | −17.1 | 3.1 → 2.4 % |
| `Agtri@topx8` dense top-octave gates (seed 87358) | 32.4 → 35.7 % | −26.0 | 14.6 → 14.7 % |

- The loss adds +8.6 (octave) and +9.5 (plain rows) points to the plain fields and removes their small fold; it adds only +2 to +3 to
  the sharp fields (the gate as written +2.3, the triangle +2.7, the powers +2.6 / +3.1).
- Mechanism: the loss learns from the field's own MPPI traversals. On the folded fields they stall: with the powers added 93–95 % of
  the 8,192 traversals of a refresh stall (octave about 64 %), so the midpoint trees hold 20,000–34,000 nodes of short paths against
  about 280,000 for the octave; the long-range targets that would unfold the sawtooth never reach the loss.
- The 3-seed means of the base fields here are those of the single runs each fork came from (the octave fork runs are 56.9 / 44.4 /
  37.6 %, against the reference means 51.9 / 45.0 / 42.2 % in `jeffrey/sharp_features`).

Run ids: `queue/state.json` (done map `gpu|n|seed|spec|label -> run id`); smoke `20261118T170026Z-26a5f729`; registry search
`Sharp midpoint`.

## Setup

- Base fields: the registered `jeffrey/sharp_features` runs (22×22 maze, prepare `20261101T193403Z-cea05796`), each with
  its kept epoch-2000 checkpoint, and for the spec `octave` the exact-octave run of `jeffrey/octave_detune` of the same seed.
- Each run resumes that checkpoint (network, AdamW state) and trains epochs 2001–5000 with `jeffrey/midpoint_dp`'s loop and
  its best setting (MAIN: `--weight 1 --leaf min --traverse safe --nodes both`, ramp 2000→2500, leaves of one cell, 8,192
  traversals every 100 epochs). The comparison that answers the question is spec + midpoint against octave + midpoint of the
  same seed. The spec's own 5,000-epoch run (no midpoint) stands for the resumed control: in `jeffrey/midpoint_dp` the
  control (`--weight 0`) matched the plain run (60 vs 59 % on 13×13, 24 vs 23 % on 22×22).
- Code: copies of `jeffrey/midpoint_dp` (`trainer.py`, `traverse.py`, `tree.py`, `fastnet.py`, `closenet.py`, `pipeline.py`,
  `run.py`) and of `jeffrey/sharp_features` (`features.py`, `sharpnet.py`, `diagnostics.py`, `ripple.py`). Changes: the
  trainer builds `sharpnet.SharpNN` with the base run's spec; `fastnet._forward` takes either 2πB (the plain path, unchanged)
  or the network's own `input_mapping`, and with more than 256 inputs the first block's residual takes the 256 base features
  (as `SharpNN.out`); the pipeline loads the field with `sharpnet.load_field` and adds the sharp-features diagnostics
  (wall slope, local minima, uphill share, eikonal residual, failures by arrival). Check: the fast path matches `NN.out`
  within 4e-6 for every base spec; smoke run `20261118T170026Z-26a5f729`.
- Queue: `queue/runner.py` (jobs `gpu|n|seed|spec|label`, label `main` or `ctl`), two trainings at once.

Round 1, seed 87358: `octave`, `Apowmix@all`, `Rtri@all`, `Agtri@allx2`, `Asc@new128`, `Rpowmix@half`, `Apow16@axis`,
`Agtri@topx8`. Round 2, seeds 1 and 2: `octave`, `Apowmix@all`, `Rtri@all`, `Agtri@allx2`, `Asc@new128`, `Rpowmix@half`.

## Reproduce

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/sharp_midpoint/run.py \
        --prepared results/registered_20261101T193403Z-cea05796 --base-run results/registered_20261118T074520Z-30f86686 \
        --seed 87358 --label main --extra="--weight 1 --leaf min --traverse safe --nodes both"
    setsid nohup ~/.venvs/bellman-ntfield/bin/python jeffrey/sharp_midpoint/queue/runner.py > jeffrey/sharp_midpoint/queue/runner.out 2>&1 &
    ~/.venvs/bellman-ntfield/bin/python jeffrey/sharp_midpoint/report.py
