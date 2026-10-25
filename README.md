# Bellman600 Clean — reproducible offline method

**For a new machine or colleague, start with [COLLEAGUE_QUICKSTART.md](COLLEAGUE_QUICKSTART.md).**

```bash
python3 run.py doctor-source
python3 run.py from-source --smoke
python3 run.py from-source
```

`from-source` starts with FK pose sampling and physical labels. It requires only
the frozen runtime/robot/evaluation bundle and pinned Docker image; **precomputed
speed, bank, checkpoint, reference trajectories and historical pose pools are
not required**. Private GitHub Release `reproducibility-v2` provides downloads.
Use `from-source-prepare` to generate only training data, speed and bank.
All 600 evaluation queries are frozen and checked into Git; never resample them.

The entries below document the original frozen-input baseline and optional
historical exact reproduction. They have distinct input requirements.


A clean Bellman travel-cost baseline for a fixed UR5e scene. It uses an
**offline transition bank + a frozen pretrained speed model** to train a
symmetric scalar network `T(q,g)` with 338,177 parameters. Only three loss terms
are computed:

```text
L = 6 × Bellman SmoothL1 + 2 × Eikonal + 2 × winner-direction
```

The network, losses and training loop keep the validated clean implementation.
Deployment needs only the TravelCost weights; speed and bank are used only for
training. There are currently no online-transition experiments or automatic
ablation queues.

This standalone directory passed a full migration reproduction: the checkpoint
hash and every per-query success flag match the original clean retraining. See
[VALIDATION.md](VALIDATION.md).

## Quick start

Requires Linux, an NVIDIA GPU, Docker with GPU container support, and
Python 3.10+. The pinned image already exists on the original development
machine; see the [environment notes](environment/README.md).

```bash
cd /home/exx/Documents/bellman600-clean
python3 run.py doctor
python3 run.py reproduce --smoke   # 4 training steps and a 2-query check; not an official success rate
python3 run.py reproduce           # 5 rounds of training from scratch + full evaluation
python3 run.py evaluate            # evaluate the frozen deployed checkpoint only
python3 run.py train               # train only
```

All research runs are registered through this repository's own registry, and
outputs are written to a new `results/registered_<id>/`. The default shared
database is `~/.local/share/ntfields-experiments`; set the `NTFIELDS_REGISTRY`
environment variable to use another location. Exact repeats are blocked; when
you have a specific reason to reproduce, use:

```bash
python3 run.py reproduce --repeat-reason 'Verify full reproduction after migrating to a new machine'
```

Do not modify registered inputs while a run is in progress. Nothing is
downloaded automatically, and neither the old ntfields-g1 repository nor an
external release directory is needed. `/clean`, `/repo` and `/workspace` are
mount paths inside Docker, not host directory dependencies.

## Layout

|Directory/file|Purpose|
|---|---|
|`method/`|Network, the three losses, offline training and evaluation adapter|
|`assets/`|Frozen bank, speed, deployed checkpoint, reference trajectories; not in Git|
|`runtime/`|Frozen planner, strict scorer, robot and query-suite compatibility runtime|
|`gui/`|Replay of frozen MPPI and RK4 trajectories|
|`tools/`|Experiment registry, asset verification/packing, reproduction and result validation|
|`evidence/`|Pre-migration deployed evaluation and the most recent retraining evidence|
|`baseline.json`|The single deployed-baseline pointer, using relative paths|
|`asset_manifest.json`|SHA256 and size of every frozen asset|
|`PROVENANCE.json`|Source files and record of migration changes|
|`VALIDATION.md`|Validation status and numerical limits|

## Results and training conventions

|Version|MPPI seed0/1/2|MPPI total success rate|RK4|
|---|---|---|---|
|Frozen deployed checkpoint|532/533/532|1597/1800 = 88.72%|549/600|
|Verified retraining from scratch|532/533/532|1597/1800 = 88.72%|548/600|

There are 600 distinct queries, 100 for each of 6 tasks. MPPI repeats every
query with 3 planning seeds; these are not 3 training seeds. There is only one
training seed (87358), so these results cannot support claims of independent
generalization or statistical significance.

Training is fixed at 5 rounds × 2 epochs, batch 256, 15,620 updates; AdamW with
lr 2e-4, weight_decay 1e-6 and gradient clipping at 2. Round 5 is always the one
evaluated; a checkpoint must not be picked by intermediate scores. The offline
bank has 399,725 source poses × 32 candidate actions. On the original machine
training takes about 4–5 minutes and training plus evaluation about 8–10
minutes; shared GPU load affects timing.

**In the original `reproduce`, "from scratch" means TravelCost starts from
random initialization.** `rebuild-reproduce` additionally retrains speed from the
source training data and regenerates the bank before training and evaluating
TravelCost. The no-wall baseline keeps the frozen speed and bank as regression
references; the wall experiment has its own preparation pipeline. The provenance
contract for the frozen speed and bank is kept in `assets/bank/manifest.json`.

MPPI uses 1536 samples, horizon 7, at most 400 steps, step_size 0.01, beta 12.5
and goal_cost_weight 4. There is no wall; table and self-collision are kept, a
20-degree OR constraint applies, and scoring interpolates at most 0.001 rad per
joint. The architecture guarantees symmetry, a zero diagonal and a Euclidean
lower bound; it does not guarantee the triangle inequality or continuous-path
safety.

## Assets and Git migration

The local directory includes the assets; large weights and the bank are not
committed to Git. After cloning, you also need an asset archive that matches
`asset_manifest.json`:

```bash
python3 tools/assets.py unpack /path/to/bellman600-clean-assets.tar.gz
python3 tools/assets.py verify
```

Create an asset archive (written to the Git-ignored `dist/` by default):

```bash
python3 tools/assets.py pack
```

A new machine also needs the pinned Docker image; the asset archive does not
include the ~8.7 GB image. See the environment notes for the image migration
commands. The archive and checksum files from the first delivery are in `dist/`.

## Trajectory replay

Use the Python environment on the original machine that has the GUI
dependencies installed:

```bash
/home/exx/miniforge3/bin/python gui/viewer.py --port 8901
# To access it from another machine, explicitly add --host 0.0.0.0
```

By default it listens only on localhost and shows this repository's deployed
baseline: MPPI in blue, RK4 in purple. The GUI only replays saved trajectories;
it cannot plan online or patch safety. Dependency versions are listed in
`environment/gui-packages.json`.

## Provenance and release

This is a new, independent Git history; the original repository is left
unchanged. No remote is configured and nothing has been published to GitHub.
`runtime/` is compatibility code frozen to preserve the protocol, not a new
optional training method; no new license is granted for these third-party
sources. See `runtime/README.md` for provenance and licensing notes.

## Wall direct-transfer test

The same 600-query scene was evaluated with only a 25 cm wall added, using the
original frozen no-wall checkpoint. MPPI scored 1494/1800 (83.00%); on the
572-query subset with valid endpoints it scored 1494/1716 (87.06%).
Every trajectory is point-for-point identical to the no-wall run. This is a
direct-transfer test, not a result after training with the wall.
See the [comparison report](evidence/wall25/COMPARISON.md) and the
[protocol](experiments/wall25/README.md).

```bash
python3 run.py wall-audit
python3 run.py wall-evaluate
```

## Retraining with the wall

Wall25-specific speed, offline bank and TravelCost were trained from scratch and
fully evaluated. MPPI scored 492/491/492, 1475/1800 in total (81.94%), lower
than the 83.00% from direct transfer in the same scene; 85.96% on the 572-query
valid-endpoint subset. RK4 scored 494/600 (82.33%). The run took 12 min 58 s in
total, used a single training seed, and did not replace the no-wall baseline.
See the [results](evidence/wall25_train/COMPARISON.md) and the
[training protocol](experiments/wall25_train/README.md).

```bash
python3 run.py wall-reproduce
```

Moving this experiment to another machine also needs the supplementary archive
`dist/wall25-training-inputs.tar.gz` (not part of the original baseline asset
archive). Extract it at the repository root:
`tar -xzf /path/to/wall25-training-inputs.tar.gz`.
The training entry point automatically checks the four input hashes in
`experiments/wall25_train/source_manifest.json`.
The full run outputs (weights, bank, trajectories) are stored in
`results/registered_20260930T013432Z-7905f45e/` and are not in Git.

## No-wall speed and transition bank reconstruction

The original no-wall input-generation pipeline is now available:

```bash
python3 run.py rebuild-reproduce --smoke
python3 run.py rebuild-reproduce
```

This rebuilds speed from the original training dataset, regenerates the offline
transition bank, checks equality against the frozen baseline, then trains and
evaluates TravelCost. Source data are packaged in
`dist/no-wall-training-inputs.tar.gz`; extract at the repository root on another
machine. Speed and bank no longer need to be accepted as precomputed inputs for
training, although frozen copies remain available as regression references.
See [protocol](experiments/no_wall_rebuild/README.md) and
[validation](evidence/no_wall_rebuild/REPORT.md). Original pose pools and labels
are the source dataset; all physical labels are also regenerated and audited.

## Independent source-to-evaluation validation

The complete original pose sampler, physical labels, speed, bank, TravelCost and
fixed 600-query evaluation have been validated without precomputed training
inputs. See [full evidence](evidence/from_source/REPORT.md).

## Obstacle scenes

Three further scenes append obstacles to the no-wall scene: raised platforms under both task
regions, a standing cupboard with shelves, and one household object per task region. A scene is
retrained from source with the unchanged recipe, then scored on the fixed 600 queries and on the scene's
own frozen, certified 600-query test suites (so far: platforms and clutter):

```bash
BELLMAN600_DERIVED_IMAGE=bellman600-env:sm86 ./docker-python run.py obstacle-scene --scene platforms --gpu 2
```

See the [protocol](experiments/obstacle_scenes/README.md) and the
[comparison with the no-wall run](evidence/obstacle_scenes/COMPARISON.md).

## Cabinet task and the metric NTFields baseline

A motion-planning task with only the shelf scene's cupboard on the table and no orientation rule: the
grasp-point sampling box is as tall as the cupboard, 1.5× as wide and 2.5× as deep, and the speed
depends on clearance only. Bellman was trained from source on it with the unchanged TravelCost recipe,
and the metric NTFields baseline (`ntrl-demo`, vendored unmodified) was trained on the same task. Both
are scored on 600 frozen, certified cabinet queries: Bellman RK4 229/600 (38.2%) and release MPPI
688/1800 (38.2%); metric NTFields RK4 182/600 (30.3%), release MPPI 595/1800 (33.1%) and its own MPPI
509/1800 (28.3%). See the [protocol and results](experiments/cabinet/README.md) and the
[comparison](evidence/cabinet/COMPARISON.md).

```bash
BELLMAN600_DERIVED_IMAGE=bellman600-env:sm86 ./docker-python run.py cabinet --gpu 2
BELLMAN600_DERIVED_IMAGE=bellman600-env:sm86 ./docker-python run.py cabinet-metric --gpu 2
```

Two easier variants give every compartment 10% and 20% more room (openings 1.1x / 1.2x as wide, compartments
1.1x / 1.2x as tall; `--scene cabinet_plus10` / `cabinet_plus20`). Both were prepared, trained and scored the
same way on their own frozen suites. Neither method gained: Bellman release MPPI 37.9% and 35.9%, metric
NTFields 33.3% and 34.2% (straight line 31.0% and 34.7%). See the
[clearance sweep](evidence/cabinet/CLEARANCE.md).
