"""Allowlisted, public-safe snapshots and verified extraction. Never publishes."""
from __future__ import annotations
import gzip
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from contextlib import ExitStack
from .io_utils import ROOT, atomic_json, read_json, sha256, writable

FORBIDDEN = {'.env', '.ssh', '.config', '.codex', 'credentials', '.git', '__pycache__', '.pytest_cache'}
PATTERNS = [re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----'),
            re.compile(rb'(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{30,})'),
            re.compile(rb'(?i)(?:authorization\s*[":=]+\s*["\x27]?\s*(?:bearer|token)\s+)[A-Za-z0-9_.-]{20,}')]

def safe_relative(name):
    p = PurePosixPath(name)
    if (not name or p.is_absolute() or '..' in p.parts or '\\' in name or
            any(part in FORBIDDEN or part.startswith(('.venv', '.env', 'credentials')) for part in p.parts)):
        raise ValueError('Unsafe or forbidden checkpoint path')
    return p

def scan_file(path):
    """Stream scan actual contents before compression; never print detected values."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Only regular non-symlink source files are allowed')
    with path.open('rb') as f:
        carry = b''
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            data = carry + chunk
            if any(p.search(data) for p in PATTERNS):
                raise ValueError('Secret-like content detected; publication blocked (value suppressed)')
            carry = data[-2048:]

def _archive(out, entries, metadata):
    out = writable(out)
    if out.exists():
        raise FileExistsError('Checkpoint files are immutable; choose a unique ID')
    if shutil.disk_usage(out.parent).free < 10 * 1024**3:
        raise ValueError('At least 10GiB free required before archive creation')
    manifest = []
    for name, path in entries:
        safe_relative(name)
        scan_file(path)
        manifest.append({'path': name, 'size': path.stat().st_size, 'sha256': sha256(path)})
    if len({r['path'] for r in manifest}) != len(manifest):
        raise ValueError('Duplicate archive paths')
    with tempfile.TemporaryDirectory(prefix='metadata-', dir=out.parent) as temporary:
        mp = Path(temporary) / 'FILES.json'
        mp.write_text(json.dumps({'metadata': metadata, 'files': manifest}, indent=2) + '\n')
        scan_file(mp)
        with out.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz, tarfile.open(fileobj=gz, mode='w') as archive:
            expected = {r['path']: r for r in manifest}
            for name, path in entries + [('FILES.json', mp)]:
                if name != 'FILES.json' and (path.stat().st_size != expected[name]['size'] or sha256(path) != expected[name]['sha256']):
                    raise ValueError('Snapshot source changed during archive creation')
                info = archive.gettarinfo(str(path), arcname=name)
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                info.mtime = 0
                info.mode = 0o644
                with path.open('rb') as content:
                    archive.addfile(info, content)
    # A single asset defaults to <=1GiB, below GitHub's official <2GiB limit.
    if out.stat().st_size > 1024**3:
        raise ValueError('Archive exceeds autonomous 1GiB asset limit; split before upload')
    return {'name': out.name, 'sha256': sha256(out), 'size': out.stat().st_size,
            'file_count': len(manifest), 'manifest': manifest}

def create_checkpoint(destination, run_dir, source_paths, commit, checkpoint_id, source_reference=None):
    """Freeze all cooperative Builder writers for the duration of the snapshot."""
    run_dir = writable(run_dir)
    with ExitStack() as stack:
        for lock in [run_dir/'checkpoint.lock'] + sorted(run_dir.glob('builder_*/run.lock')):
            stream = stack.enter_context(lock.open('a+b'))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('Run task is active; checkpoint refused') from None
        return _create_checkpoint(destination,run_dir,source_paths,commit,checkpoint_id,source_reference)

def _create_checkpoint(destination, run_dir, source_paths, commit, checkpoint_id, source_reference=None):
    destination = writable(destination)
    destination.mkdir(parents=True, exist_ok=False)
    run_dir = writable(run_dir)
    if not (run_dir / 'completed.json').is_file():
        raise ValueError('Only completed tasks may be snapshotted')
    completed = read_json(run_dir / 'completed.json')
    if completed.get('status') != 'COMPLETE' or completed.get('active_tasks') != 0:
        raise ValueError('Completed-task manifest must assert no active tasks')
    for name, digest in completed.get('files', {}).items():
        safe_relative(name)
        if sha256(run_dir / name) != digest:
            raise ValueError('Completed task changed after freeze')
    original_records = {row['path']: row for row in
                        map(json.loads, (ROOT/'reports/server_only/originals_manifest.jsonl').read_text().splitlines())}
    source_refs = sorted({original_records[name]['ref'] for name in source_paths})
    source_entries = []
    for name in sorted(set(source_paths)):
        rel = safe_relative(name)
        path = ROOT.joinpath(*rel.parts)
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError('Source escaped repository')
        if sha256(path) != original_records[name]['sha256']:
            raise ValueError('Original file does not match audited source/ref/hash')
        source_entries.append(('originals/' + name, path))
    metadata = {'schema': 'server_v2_checkpoint_v1', 'checkpoint_id': checkpoint_id,
                'commit': commit, 'source_repo': 'catlover-bot/Minecraft-view-dataset', 'source_refs': source_refs,
                'original_manifest_sha256': sha256(ROOT/'reports/server_only/originals_manifest.jsonl'), 'completeness': 'PARTIAL',
                'reason': 'ARR/fixed_builder originals absent in reachable repository; available source and completed CPU tasks preserved',
                'research_generation_requests': 0}
    assets = []
    if source_reference is None:
        source = _archive(destination / 'sources.tar.gz', source_entries, dict(metadata, kind='immutable_originals'))
        assets.append(source)
        source_reference = {'sha256': source['sha256'], 'name': source['name'], 'source_id': 'sha256:' + source['sha256']}
    snapshot = []
    allowed_run_dirs = {'oracle','reevaluation','builder_smoke','builder_evaluation',
                        'builder_smoke_verified','builder_evaluation_verified'}
    for path in sorted(run_dir.rglob('*')):
        if path.is_file():
            relative = path.relative_to(run_dir).as_posix()
            parts = path.relative_to(run_dir).parts
            selected = parts[0] in allowed_run_dirs or (len(parts)==1 and path.suffix in {'.json','.log','.xml','.txt','.time','.sh'})
            if selected and not any(part.startswith('.pending-') for part in path.parts):
                snapshot.append(('run/' + relative, path))
    for folder in (ROOT / 'offline_evaluation/server_v2', ROOT / 'reports/server_only'):
        for path in sorted(folder.rglob('*')):
            if path.is_file() and not any(part in FORBIDDEN for part in path.parts):
                snapshot.append(('code/' + path.relative_to(ROOT).as_posix(), path))
    assets.append(_archive(destination / 'checkpoint.tar.gz', snapshot, dict(metadata, sources=source_reference)))
    for asset in assets:
        atomic_json(destination / (asset['name'] + '.files.json'), asset.pop('manifest'))
    sums = ''.join(f"{a['sha256']}  {a['name']}\n" for a in assets)
    (destination / 'SHA256SUMS').write_text(sums)
    (destination / 'RESTORE.md').write_text(
        '# Restore\n\nKeep this draft private to authorized repository users, but treat all contents as public-safe.\n'
        'Download assets with gh release download TAG --repo catlover-bot/Minecraft-view-dataset --dir NEW_DIRECTORY.\n'
        'Verify SHA256SUMS, then use python -m offline_evaluation.server_v2.cli restore --archive ARCHIVE --sha256 HASH --destination EMPTY_DIRECTORY.\n'
        'Extract sources and snapshot to separate new directories. The extractor rejects traversal and links and checks every file hash.\n'
        'For runnable reconstruction create another NEW directory under /root/minecraft-checkpoints/. Copy originals/datasets and originals/outputs from the sources extraction into that directory, and copy code/offline_evaluation from the snapshot extraction beside them. Copy run/ into a new directory under /root/minecraft-runs/. Run Python from the reconstruction directory so its offline_evaluation package resolves there. All copies are regular files; do not overwrite any existing checkout or run.\n'
        'Recreate a dedicated venv with code/offline_evaluation/server_v2/requirements.lock; never install into system Python.\n'
        'Historical ARR replay remains BLOCKED; this snapshot preserves the available source cohort and new implementation CPU results.\n')
    result = dict(metadata, assets=assets, sources=source_reference, status='LOCAL_VERIFIED_PENDING_UPLOAD')
    atomic_json(destination / 'checkpoint.json', result)
    return result

def restore_archive(archive_path, destination, expected_sha256):
    archive_path = Path(archive_path)
    destination = writable(destination)
    if sha256(archive_path) != expected_sha256:
        raise ValueError('Archive SHA-256 mismatch')
    if destination.exists():
        raise FileExistsError('Restore destination must be new; no overwrites')
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = archive.getmembers()
        names = set()
        for member in members:
            safe_relative(member.name)
            if not member.isfile() or member.name in names:
                raise ValueError('Links, devices, directories and duplicate paths rejected')
            names.add(member.name)
        if 'FILES.json' not in names:
            raise ValueError('Missing archive file manifest')
        manifest = json.load(archive.extractfile('FILES.json'))
        if names != {'FILES.json'} | {r['path'] for r in manifest['files']}:
            raise ValueError('Archive entries differ from allowlist manifest')
        destination.mkdir(parents=True)
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)
        for row in manifest['files']:
            target = destination / row['path']
            if target.stat().st_size != row['size'] or sha256(target) != row['sha256']:
                raise ValueError('Restored file checksum mismatch')
    return {'status': 'VERIFIED', 'archive_sha256': expected_sha256, 'files': len(manifest['files'])}
