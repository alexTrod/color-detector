---
name: autoresearcher
description: Run autonomous training experiments to minimize val_bpb on a fixed time budget. Use when the user wants autoresearch, autonomous experimentation, or to run the experiment loop from experimenter.md.
---

# Autoresearcher

Autonomous research loop: tune training/preprocessing, run, log, advance or revert. Goal: lowest val_bpb in 5-minute runs. Branch `autoresearch/<tag>` holds accepted commits; results.tsv is untracked.

## Setup (once per new experiment)

1. **Agree run tag** with user: e.g. `mar5`. Branch `autoresearch/<tag>` must not exist.
2. **Create branch**: `git checkout -b autoresearch/<tag>` from current main/master.
3. **Read in-scope files**: e.g. `README.md` for repo context. Clarify anything unclear.
4. **Init results.tsv**: Create `results.tsv` with header only. Baseline row added after first run.
5. **Confirm** setup with user, then start experimentation.

Terminology: val_bpb and loss are used interchangeably.

## What you may change

- Training settings and preprocessing pipeline.
- Architecture, optimizer, hyperparameters, batch size, model size.
- Constraint: run completes without crash within the time budget (5 min training). VRAM is a soft limit.
- **Simplicity**: Prefer simpler changes. Equal or better result from deleting code = keep. Tiny gain with large complexity = often discard.

**First run**: Always run the training script as-is to establish the baseline.

## Reading results

After the script finishes it prints a summary. Key metric:

```bash
grep "^val_bpb:" run.log
```

If grep is empty, treat as crash.

Example summary snippet:

```
val_bpb:          0.997900
peak_vram_mb:     45060.2
```

## results.tsv (do not commit)

Tab-separated. Header:

```
commit	val_bpb	memory_gb	status	description
```

Columns:

1. Short git commit hash (7 chars).
2. val_bpb (e.g. 1.234567). Use 0.000000 for crashes.
3. Peak memory GB, one decimal (peak_vram_mb / 1024). Use 0.0 for crashes.
4. Status: `keep`, `discard`, or `crash`.
5. Short description of the experiment.

## Experiment loop

Run indefinitely until the user stops you.

1. Check current branch/commit.
2. Change code (one experimental idea).
3. `git commit`.
4. Run the training experiment.
5. Read results (e.g. grep val_bpb from log).
6. If no val_bpb line: treat as crash; debug if trivial (typo, import), else log status `crash` and move on.
7. Append row to `results.tsv` (do not commit this file).
8. **If val_bpb improved (lower)**: keep the commit (branch already advanced).
9. **If val_bpb equal or worse**: `git reset --hard` back to previous commit.

**Timeout**: If a run exceeds ~10 minutes, kill it, treat as failure (discard, revert).

**Crashes**: Easy fix (typo, import) -> fix and re-run. Fundamentally broken idea -> log status `crash`, skip, continue.

**Do not stop to ask**: Do not ask "should I continue?" or "is this a good stopping point?". Run until the user interrupts. If stuck, re-read in-scope files and papers, combine near-misses, or try more radical changes.
