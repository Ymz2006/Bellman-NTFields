# Frozen implementation dependencies

Only the import dependency closure needed by the current model, planner, physical-label audit and scorer is retained here. There are no historical experiment result directories or runnable ablation entry points.

`ntrl/` preserves the upstream constructor, feature encoder, serialization layout and MPPI model adapter. Those generic classes contain historical options; `method/checkpoint_compat.py` and `method/model.py` constrain the active model to the selected configuration. Keeping upstream code intact preserves parameter registration order and random-number consumption.

`planner/` preserves the MPPI implementation, geometric checks and independent dense scorer. Its generic CLI is not the reproduction contract: use the fixed `reproduce.py evaluate` entry point. `physical_geometry.py` is the original offline geometry helper used only to audit the frozen speed labels.

Original source paths and hashes are in `repro/vendor_sources.json`. The generated `ntrl/train/__init__.py` establishes an explicit package so the research file `method/train.py` cannot shadow it. Empty mount directories are container compatibility placeholders.
