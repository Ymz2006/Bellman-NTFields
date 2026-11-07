# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Read `AGENTS.md` (hard rules), `README.md` and `VALIDATION.md` first. `COLLEAGUE_QUICKSTART.md` covers setting up a fresh machine.
The README's absolute paths (`/home/exx/...`, `/home/exx/miniforge3`) belong to the original lab machine, not this workstation.

IMPORTANT: WHEN MAKING AN ARTIFACT, ITS NAME WILL ALWAYS FIRST INCLUDE THE DATE 
ARTIFACTS SHOULD HIDE LARGE TABLES UNDER DROP DOWN, AND USE GRAPHS INSTEAD

## What this repository is

A frozen, reproducible baseline: a symmetric Bellman travel-cost network `T(q, g)` for a fixed UR5e scene, trained
offline from a frozen speed model and a transition bank, then scored on 600 frozen queries (6 tasks × 100) by a strict
full-arm collision scorer. On top of it are research experiments (`experiments/`) and the user's own research folder (`jeffrey/`).

Invariants (see `AGENTS.md`; change them only with explicit user direction):
- `method/` is the fixed method: `network.py` (338,177-parameter `TravelCost`), `loss.py` (Bellman 6, Eikonal 2, direction 2),
  `training.py` (5 rounds, 15,620 updates, training seed 87358; round 5 is always the one evaluated), `evaluate.py`.
- `runtime/` (planner, scorer, robot, query suites) and `assets/` are frozen. `assets/`, `results/`, `specs/`, `dist/`
  and `training_inputs/` are git-ignored; check asset hashes with `python3 tools/assets.py verify`.
- Evaluation queries are never resampled, repaired or filtered. MPPI planning seeds 0/1/2 are not training seeds, so "1800"
  means 600 queries × 3 planner seeds.
- New results never overwrite `assets/checkpoint.pt` or `baseline.json`.

## Running things: always through the registry

Every run is registered with `tools/experiment_registry/registry.py` (SQLite at `~/.local/share/ntfields-experiments`,
override with `NTFIELDS_REGISTRY`). A run gets a fresh `results/registered_<UTC>-<hash>/`. Its fingerprint hashes every
declared input file, so **do not edit a declared input (source, scene, checkpoint, bank) while a run that declares it is
in flight**; the run is invalidated. Exact repeats are blocked; pass `--repeat-reason '<concrete reason>'` to repeat one.
Never call `launch.py` directly; it is the command the registry runs.

```bash
python3 tools/experiment_registry/registry.py list --search <text>   # look for prior results first
python3 tools/experiment_registry/registry.py show <RUN_ID>
python3 tools/experiment_registry/registry.py doctor
python3 tools/experiment_registry/registry.py note <RUN_ID> '<conclusion>'
```

Top-level entry point (`run.py` writes a spec into `specs/`, checks it, then runs it; `--prepare-only` only writes the spec):

```bash
python3 run.py reproduce --smoke          # 4 training steps + 2-query check; integration test, not a score
python3 run.py from-source --smoke        # same for the full pose-sampling -> speed -> bank -> train -> eval pipeline
python3 run.py cabinet --scene cabinet --gpu 1
python3 run.py --help                     # actions: train/evaluate/reproduce, wall-*, rebuild-reproduce, from-source*,
                                          # obstacle-*, cabinet*, face-speed*, point-field*, ntrl-cabinet, pntfields-cabinet
```

There is no unit-test suite or linter. "Testing" means a `--smoke` run. `python3 tools/validate_result.py results/registered_<id>`
checks a full no-wall `reproduce` against the frozen retraining evidence (checkpoint hash and every success flag).
The `tests*/` folders under `experiments/` are frozen query suites, not code tests.

## This workstation (not the pinned environment)

The pinned image `sha256:e7749bdb…` is absent here and cannot drive the RTX 5070. Every result on this machine is a
modified-environment result and records `environment_override.json`. GPUs: `0` = RTX 5070 (12 GB, drives the display),
`1` = RTX 3090 (24 GB). Use `CUDA_DEVICE_ORDER=PCI_BUS_ID` so indices match `nvidia-smi`.

Two backends run the frozen code unchanged with the container path layout (`/clean` = repo, `/repo` = `runtime/`,
`/workspace` = vendored planner, `/outputs` writable). See `environment/local/README.md`:
- Host (bubblewrap): `BELLMAN600_BACKEND=host ~/.venvs/bellman-ntfield/bin/python run.py <action> ...`
- Docker: `BELLMAN600_DERIVED_IMAGE=bellman600-env:local BELLMAN600_BUILD_NETWORK=host ./docker-python run.py <action> ...`
  (plain `python3 run.py` asks Docker for the absent pinned image).

RAM is capped by `BELLMAN600_HOST_MEMORY` / `BELLMAN600_DOCKER_MEMORY` (default 20G). The machine is the user's daily
computer (30 GB RAM): check `free -g` before heavy jobs, memory-map `.npy` files and process them in chunks, and run at
most two heavy trainings at once.

## Code layout that spans several files

- **`launch.py`** holds the container contract: `MOUNTS`, `VARIABLES` (including `PYTHONPATH` into `/clean/method`,
  `/repo/method`, the vendored ntrl code), `host_command` (bwrap) and the per-action command builders. `run.py` builds
  the registry spec; `launch.py` turns an action into the in-container command.
- **`experiments/<name>/`** each have `run.py` (`execute(args)`, called from top-level `run.py`), a `pipeline.py`
  run inside the container, a `README.md` protocol, and usually scenes (`scenes/*.yml`) and frozen test suites.
  `experiments/cabinet/run.py:scene_paths` maps a scene name to its scene file and `tests_<suffix>/` suite.
  Matching evidence and comparison reports are in `evidence/<name>/`.
- **`baselines/ntrl-demo`** (metric NTFields) and **`baselines/P-NTFields`** are vendored third-party baselines, kept unmodified.
- **`jeffrey/`** is the user's research area. One subfolder per study (`segment_dp`, `maze_scale`, `midpoint_dp`, ...),
  each with a `README.md` holding the request, setup, registered run ids and findings. Studies register their own runs
  through their own `run.py` (pattern: `jeffrey/run.py`, `jeffrey/segment_dp/launch.py`, `jeffrey/gibson_embed/run.py:environment`)
  so that the top-level `run.py` / `launch.py`, which in-flight runs declare as inputs, stay unedited. Newer studies
  (`midpoint_dp`, `maze_scale`) run in the host venv directly and freeze their sources into `snap/<hash>/` at registration,
  so working copies can change while runs are in flight. Reports are produced by `report.py` (tables from registered
  runs) and `page.py` + `page_template.html` (HTML findings page published as an artifact).

When adding a study, follow that pattern: a new `jeffrey/<study>/` folder, its own registered `run.py`, results only
in `results/registered_<id>/`, and findings with run ids in its `README.md`. Report a control next to every result
(for example the Euclidean field next to a learned field, or the same loop without the new term), and when a result
contradicts theory, explain the mechanism with a control rather than only stating the number.

Do not push, create a remote, or publish releases without the user's authorization.
