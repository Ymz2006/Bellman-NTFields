# Validation

Results validated before migration:

- Full evaluation of the frozen deployed checkpoint: MPPI 532/533/532, RK4 549/600.
- Clean retraining from scratch: MPPI 532/533/532, RK4 548/600.
- The checkpoint file from the repeated training on 2026-09-29 has the same
  SHA256 as the previous clean retraining:
  `6a4f28557b1ec655a0eb813f00774f6ed02586ad93410a282efe705ee3294cab`.
  All 1800 MPPI and 600 RK4 success flags are identical.
- Deployed checkpoint SHA256:
  `572d74fed6384d5e0211315147e8276cff35c2bcff5fd3da645bec1c4ea68a85`.
- Retraining versus the deployed checkpoint: MPPI rescued 1 execution and
  newly failed 1; RK4 rescued 1 query and newly failed 2.

This historical evidence is in `evidence/`. Old absolute paths in the original
records indicate provenance only; the current entry points and the baseline
pointer do not depend on them.

## Standalone repository migration validation: passed

Registered run: `20260930T002043Z-6d56aa43` (UTC 2026-09-30, New York local
time 2026-09-29); registry status succeeded.

- The new directory trained from scratch for 5 rounds and 15,620 steps in
  255.39 seconds.
- MPPI 532/533/532, 1597/1800 (88.72%); RK4 548/600.
- The retrained checkpoint SHA256 is identical to the pre-migration clean
  retraining.
- Every per-query success flag for the 1800 MPPI and 600 RK4 runs matches the
  pre-migration retraining.
- All Docker mounts come from the new directory; there are no dependencies on
  the old repository or release paths, or on symlinks.
- After cloning Git into another directory and unpacking the assets, all 113
  frozen asset files passed SHA256 verification.
- The GUI could read trajectories and scores for every task and seed, robot FK
  passed, and a temporary localhost server initialized and returned HTTP 200;
  the temporary server was shut down after the check.

Evidence: [migration checks](evidence/migration/migration_validation.json),
[packaging checks](evidence/migration/packaging_checks.json),
[full scores](evidence/migration/evaluation.json).

This change only touched the directory migration, entry points, report wording
and GUI display; the network, losses and training loop are unchanged. It still
only demonstrates reproduction on fixed hardware and a fixed image; it does not
claim bitwise equality across environments.

## No-wall preparation reconstruction

See [full reconstruction evidence](evidence/no_wall_rebuild/REPORT.md).

## Full source sampling and colleague delivery

[Verified full pipeline](evidence/from_source/REPORT.md): newly sampled training
poses/labels, retrained speed/bank, identical final checkpoint and all success
flags; clean-clone runtime-only smoke and real GUI passed.
