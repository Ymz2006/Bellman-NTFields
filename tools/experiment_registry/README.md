# Experiment registry

Purpose: **look for existing results before launching; after an experiment exits, be able to find its config, code, logs and outputs.** It does not change any method and never starts training on its own.

## The four everyday commands

Run from the repository root:

```bash
# 1. Search history, including directories not yet verified
/usr/bin/python3 experiments.py list --search routing

# 2. Check an experiment you are about to run: identical content, or history that already tested the same checkpoint, is reported and returns 3
/usr/bin/python3 experiments.py check tools/experiment_registry/specs/nobias52k_eval600.json

# 3. Run only when you are sure: checks for duplicates again and allocates a new output directory automatically
/usr/bin/python3 experiments.py run path/to/my_spec.json

# 4. Check registered outputs for missing files, changed content or vanished manager processes
/usr/bin/python3 experiments.py doctor
```

`run` launches the program with an argument array, not through a shell; stdout/stderr are written to `runs/<ID>/console.log` in the registry. The log and result directories are printed when the command starts. The original `reproduce.py` can still be used directly, but bypassing this tool loses the pre-launch duplicate protection.

The registry defaults to `~/.local/share/ntfields-experiments`, so the same user shares records across branches and working directories. Use `NTFIELDS_REGISTRY` or `--home /path/to/registry` before the subcommand to choose another location. The database uses SQLite transactions, so if two processes register the same experiment at once, only one succeeds. Sharing one SQLite database on a network file system between several people is not recommended.

## What the fingerprint includes

- The explicitly declared config, including training seed, planning seeds, step count, query-suite protocol, and so on.
- The SHA256 of every real file in `inputs`: code, data, initialization, model weights, query suites, scorer, etc.; directories are expanded recursively.
- Command arguments, the declared environment, output requirements and the validation mode; a few inherited environment variables that affect computation are also recorded.
- The Git commit is stored separately, but duplicates are not detected by commit alone, so that a documentation-only change is not treated as a new experiment.

The experiment name and output directory do not determine identity; changing code, data or config changes the fingerprint. The input set is declared by the spec, so **a missing dependency weakens duplicate detection**. Inputs are re-checked before and after the run; this is no substitute for a read-only snapshot, so do not modify dependent code or data while a run is in progress. Source and config files smaller than 1 MiB are also copied; large original weights and data are tracked by path and hash. When backing up after the source has changed, prefer the copy saved at registration whose hash matches, rather than passing off current code as historical code.

Three templates are provided: `specs/nobias52k_eval600.json`, `specs/compact_eval600.json` and `specs/train52k.json`. Copy one and edit it, keeping command, config and inputs in sync; `root` is resolved relative to the directory containing the spec file. The templates use the immutable Docker image ID verified on this machine; a recipient should build the environment and fill in their own image ID, and must not treat a mutable tag as a fixed environment. The compact-model template requires generating the compact checkpoint first.

Exclusive GPU access is not guaranteed by this tool; latency experiments still need to record hardware, concurrent load and timing boundaries. Undeclared system dependencies are not frozen either.

## Do not mistake "it ran" for "it has a conclusion"

| Status | Meaning |
|---|---|
| reserved / running | Fingerprint claimed / executing |
| succeeded | Exit code 0, inputs unchanged, declared output checks passed; does not by itself mean the method works |
| failed / interrupted | Failed or interrupted; the reason and logs are kept |
| needs_review | The manager was interrupted and child processes may still be running; nothing is restarted and no Docker containers are killed automatically |
| linked_existing | An old result linked by hand; the available source snapshot and outputs were checked, but not every historical command/environment is proven automatically |
| discovered_unverified | A historical directory was found; unknown whether training finished or whether only one factor was varied |
| recovered_outputs | Outputs passed re-checking after an interruption; the original exit status and proof of unchanged inputs are not fabricated |

The `evaluation600` validation mode requires six tasks × three planning seeds, 100 queries per task, complete flags and a matching checkpoint hash; the templates also require each of the 18 trajectory and score files to exist. `training_steps` checks that the training status reached the number of steps the spec requires, so 2000 steps cannot stand in for 52000. The generic `files` mode only checks declared outputs and infers nothing about SR or training completion.

Any existing identical fingerprint (including failed runs) is blocked, and the same checkpoint in the historical index must also be reviewed first. Those historical records may use different protocols, so the warning is not a judgement that it is a "strict repeat". When you really need to re-run:

```bash
python3 experiments.py run my_spec.json --repeat-reason 'Last run hit a storage error; retrying after the fix; confirmed the old process and container have exited'
```

Record the purpose and conclusion so they can be found later:

```bash
python3 experiments.py note RUN_ID 'Only changed the branch supervision mask; SR results not yet summarized, must not be written up as a method contribution'
python3 experiments.py show RUN_ID
```

## Recovering existing experiments

```bash
python3 experiments.py discover /path/to/results
python3 experiments.py list --search uniform
python3 experiments.py link-existing my_spec.json /path/to/old_output --reason 'Explain how the config corresponds and which original evidence was checked'
```

`discover` reads only result groups one level below the root directory, recursively lists the location and size of their JSON, reports, weights, trajectories and logs, does not modify the original files, and does not treat a directory count as a number of independent training runs. Historical files are only checked for existence; they are not yet uniformly content-hashed. For results containing `evaluation.json`, the checkpoint hash is also extracted for the pre-launch warnings; other historical formats are checked by hand via keywords.

`link-existing` requires an existing source_snapshot, matching hashes for every file that overlaps between it and the currently declared inputs, and passing output validation. It refuses even when the source differs only in formatting, to avoid silently loosening identity checks; such results can be kept in the historical index with a note about the existing equivalence evidence.

If the manager exits unexpectedly, first check the logs, PID and Docker. Once you are sure it has finished, use `recover RUN_ID --reason ...` to re-check the outputs; the tool refuses to recover a local record that is still running and never restarts an experiment automatically.

## Backups and loss prevention

```bash
# Searchable standalone JSON snapshot
python3 experiments.py export /backup/experiments-20260916.json
# Database, small source snapshots, registry logs
python3 experiments.py backup /backup/registry-20260916.tar.gz
# Also include formally registered/linked result directories, with their weights, trajectories and scores
python3 experiments.py backup /backup/registry-with-results-20260916.tar.gz --include-results
# Also include the input assets declared by registered experiments: data, initialization and model weights, deduplicated by hash
python3 experiments.py backup /backup/registry-full-20260916.tar.gz --include-results --include-inputs
```

Backups never overwrite existing files. SQLite produces a consistent snapshot through the backup API; a result directory that is still changing is not a transactional snapshot, so formal archives should be made after the run finishes. Discovered historical directories are not packed automatically. Original training data, initialization and weight assets are only packed with `--include-inputs` and only when declared in a registry record; otherwise you still need the original Release or a separate backup. `--include-results` includes only formally registered/linked result directories. **A backup on the same disk only protects against accidental deletion, not disk failure; copy archives to another disk or a colleague's machine.**

Restoring: unpack the archive into a new directory and use the database in it with `--home /restored`. Result archives are placed in `results/<RUN_ID>`, while the original records keep their original absolute paths; viewing and searching work directly, but a full doctor or later runs require restoring the original output locations first or remapping paths using the mapping in the original record. With `--include-inputs`, input files are stored by content hash under `inputs/` and can be restored to their original paths or the matching locations in a new checkout using `input_paths.json`. Do not overwrite a database that is in use.

## Testing

```bash
/usr/bin/python3 -m unittest discover -s tools/experiment_registry -p 'test_*.py' -v
```

All tests are small CPU programs in temporary directories: concurrent duplicate checks, saving failures, input changes, missing/tampered file detection, warnings about historical identical checkpoints, training that falls short of the required steps, rejection of incomplete 600-query results, and database and output recovery from backups. They do not start model training or planning.
