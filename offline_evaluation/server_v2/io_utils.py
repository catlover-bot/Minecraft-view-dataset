"""Scoped, atomic research output helpers. No credentials are recorded."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_ROOTS = (ROOT, Path('/root/minecraft-runs'), Path('/root/minecraft-checkpoints'))

def writable(path):
    path = Path(path).resolve()
    if not any(path == root or path.is_relative_to(root) for root in ALLOWED_ROOTS):
        raise ValueError('Output must be inside an authorized research directory')
    return path

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(path, value):
    path = writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    fd, name = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def read_json(path):
    return json.loads(Path(path).read_text())
