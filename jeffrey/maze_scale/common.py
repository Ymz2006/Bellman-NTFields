"""Shared helpers: distance bins, logging, json, summaries by bin."""
import json
import time
from pathlib import Path
import numpy as np

# query bins by true travel distance D = T* / pitch (cells at full speed)
BINS = [0.5, 2, 4, 8, 16, 32, 64, 128, 256]
PLAN_SEEDS = [0, 1, 2]
TRAIN_SEEDS = [87358, 1, 2]


def bin_labels():
    return [f'{BINS[i]:g}-{BINS[i + 1]:g}' for i in range(len(BINS) - 1)]


def bin_of(d_cells):
    return np.digitize(np.asarray(d_cells), BINS) - 1           # -1 below, len(BINS)-1 above


class Logger:
    def __init__(self, path):
        self.f = open(path, 'a')

    def __call__(self, msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        self.f.write(line + '\n')
        self.f.flush()


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=float) + '\n')
    tmp.replace(path)


def by_bin(rows, bins, key='success'):
    """{bin label: [count of rows with key true, rows]} for rows aligned with bins (bin indices)."""
    out = {}
    labels = bin_labels()
    for b in sorted(set(int(x) for x in bins)):
        idx = [i for i, x in enumerate(bins) if int(x) == b]
        out[labels[b]] = [int(sum(bool(rows[i][key]) for i in idx)), len(idx)]
    return out


def save_paths(path, paths):
    arr = np.empty(len(paths), dtype=object)
    arr[:] = [p.detach().cpu().numpy().astype(np.float32) for p in paths]
    np.save(path, arr, allow_pickle=True)
