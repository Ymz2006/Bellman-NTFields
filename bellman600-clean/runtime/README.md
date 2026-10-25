# Frozen compatibility runtime

Copied from `pure-l2-half-goalcost4-beta12-repro-20260919` to remove external
host-directory dependencies. Includes the unchanged planner/scorer, normalization,
scene,600-query suites, FK and checkpoint-interface compatibility modules,
and robot description/meshes. Historical release training entry points, training
arrays, and release model weights are omitted.

This directory supports the clean Bellman method; its legacy compatibility
classes are not alternate trainable baselines. Preserve executable runtime
files and evaluation assets when comparing methods.

File-level source hashes are recorded in ../PROVENANCE.json. The source vendor
README is retained. Existing robot metadata and upstream license notices are
preserved where provided. No new blanket license is asserted for upstream code,
robot descriptions/meshes, cuRobo, or the NVIDIA Docker image. A future public
release should retain the original licensors' terms.
