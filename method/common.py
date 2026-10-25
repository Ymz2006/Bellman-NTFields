"""Paths and serialization shared by clean training and evaluation."""
import hashlib
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RELEASE = Path('/repo')
# Frozen planner/scorer compatibility; no research or archive imports.
sys.path.extend([str(RELEASE/'vendor/ntrl'), str(RELEASE/'method'), '/workspace'])


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str)+'\n')
