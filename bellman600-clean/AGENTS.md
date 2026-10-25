# Bellman600 Clean

Read README.md and VALIDATION.md first. This is a standalone offline baseline.

- Preserve the338177-parameter network, three losses6/2/2, seed87358,5 rounds,
  15620 updates, and fixed fifth-round evaluation unless the user authorizes research changes.
- Keep assets and runtime frozen. Use run.py to register research runs; outputs
  must go into new results directories. Do not bypass the registry via launch.py.
- The current user decision is offline transitions. Do not restart online
  experiments or introduce architecture/loss/planner tuning without direction.
- Planning seeds0/1/2 are not training seeds. There are600 distinct queries.
- Preserve original provenance and frozen evaluation evidence. New results must
  not replace assets/checkpoint.pt or baseline.json automatically.
- Large assets are excluded from Git; verify their hashes with tools/assets.py.
- Do not push or create a remote repository without user authorization.
