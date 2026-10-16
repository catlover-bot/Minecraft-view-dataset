"""Copy completed, source-bound Goal010 aggregates; never upload or generate."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import json

from .. import pipeline, source_guard as guard
from ....server_v2.checkpoint import scan_file, safe_relative

VERSION = 'goal010_public_aggregate_copy_v1'
MAX_FILE_BYTES = 10 * 1024**2
EXTENSIONS = frozenset(('.json', '.csv', '.md'))
OWN_FILES = ('offline_evaluation/analysis/goal010/publication/__init__.py',
             'offline_evaluation/analysis/goal010/publication/publish.py')


def destination_path(value):
    raw = Path(value)
    if '..' in raw.parts:
        raise ValueError('publication destination cannot contain parent traversal')
    path = raw.absolute()
    allowed = guard.ROOT / 'reports/server_only/goal010'
    if path == allowed or not path.is_relative_to(allowed):
        raise ValueError('publication destination must be below reports/server_only/goal010')
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('publication destination cannot contain symlinks')
    if path.exists():
        raise FileExistsError('fresh publication destination required; preserve prior copies')
    if not path.parent.is_dir():
        raise ValueError('publication destination parent must already exist')
    return path


def _receipt_chain(root):
    first = guard.read_json(guard.regular(root / 'PHASE0_RECEIPT.json'))
    binding = first['binding']
    actual = pipeline.code_binding(binding['code_commit'], binding['specification_commit'])
    if binding != actual:
        raise ValueError('analysis receipt differs from current committed code/specification')
    previous = pipeline.verify_previous(root, 6, binding)
    gate = guard.read_json(guard.regular(root / 'phase0_protocol/source_gate.json'))
    guard.require_gate(gate, binding['specification_sha256'])
    if (gate.get('input_kind') != 'verified_sources' or gate.get('code_commit') != binding['code_commit']
            or gate.get('goal007_manifest_sha256') != guard.audit8.MANIFEST_SHA256):
        raise ValueError('canonical Phase0 code/source gate mismatch')
    initial = guard.read_json(guard.regular(root / 'phase0_protocol/source_inventory_receipt.json'))
    if initial.get('status') != 'PASS':
        raise ValueError('Phase0 source inventory must have passed')
    receipts = []
    for number in range(6):
        if (root / f'PHASE{number}_STOP.json').exists():
            raise ValueError('a stopped phase cannot be published')
        receipt = guard.read_json(root / f'PHASE{number}_RECEIPT.json')
        if (receipt.get('version') != 'goal010_phase_receipt_v1'
                or receipt.get('source_before_after_unchanged') is not True
                or receipt.get('source_package_sha256') != initial['source_package_sha256']
                or any(receipt.get(key) != 0 for key in pipeline.ZERO_QUERIES)):
            raise ValueError('completed phase source or zero-operation evidence differs')
        receipts.append(receipt)
    return binding, previous, initial, receipts


def select_files(root, receipts, receipt_descriptors):
    """Explicit paths from six known phase dirs; no recursive raw artifacts."""
    selected = {}
    final_products = receipts[-1]['products']
    for directory in pipeline.DIRECTORIES:
        base = root / directory
        if (not base.is_dir() or any(p.is_symlink() for p in (base, *base.parents))):
            raise ValueError('all six canonical nonsymlink phase directories required')
        for path in sorted(base.iterdir()):
            if path.is_symlink():
                raise ValueError('phase output symlink rejected')
            if not path.is_file() or path.suffix not in EXTENSIONS:
                continue
            relative = path.relative_to(root).as_posix()
            safe_relative(relative)
            if relative not in final_products:
                raise ValueError('top-level aggregate is not bound by the final phase receipt')
            selected[relative] = dict(final_products[relative])
    for number in range(6):
        name = f'PHASE{number}_RECEIPT.json'
        descriptor = receipt_descriptors[str(number)]
        selected[name] = {key: descriptor[key] for key in ('sha256', 'bytes')}
    for name, descriptor in selected.items():
        if type(descriptor.get('bytes')) is not int or not 0 <= descriptor['bytes'] <= MAX_FILE_BYTES:
            raise ValueError('aggregate exceeds the10MiB publication file limit')
        path = guard.safe_child(root, name)
        if path.stat().st_size != descriptor['bytes'] or guard.sha256_file(path) != descriptor['sha256']:
            raise ValueError('allowlisted aggregate changed')
        scan_file(path)
    return dict(sorted(selected.items()))


def _copy(source, destination, descriptor):
    raw = guard.regular(source).read_bytes()
    if len(raw) != descriptor['bytes'] or hashlib.sha256(raw).hexdigest() != descriptor['sha256']:
        raise ValueError('source bytes changed before aggregate copy')
    destination.parent.mkdir(exist_ok=True)
    if any(p.is_symlink() for p in (destination, *destination.parents)):
        raise ValueError('copy destination acquired a symlink')
    with destination.open('xb') as stream:
        stream.write(raw)
    if destination.stat().st_size != descriptor['bytes'] or guard.sha256_file(destination) != descriptor['sha256']:
        raise ValueError('copied aggregate bytes disagree')
    scan_file(destination)


def publish(run_root, destination, *, code_commit):
    root = pipeline.safe_root(run_root, 6)
    output = destination_path(destination)
    own_hashes = guard.verify_commit(code_commit, OWN_FILES)
    with guard.locked_sources() as locks:
        binding, receipt_descriptors, initial, receipts = _receipt_chain(root)
        before = guard.load_verified_sources()
        source_digest = pipeline.digest(before)
        if source_digest != initial['source_package_sha256']:
            raise ValueError('current immutable sources differ from initial Phase0 package')
        selected = select_files(root, receipts, receipt_descriptors)
        output.mkdir(exist_ok=False)
        copied = {}
        for name, descriptor in selected.items():
            source, target = root / name, output / name
            _copy(source, target, descriptor)
            copied[name] = {'canonical_source': str(source), 'copy_path': str(target),
                'source_sha256': descriptor['sha256'], 'copy_sha256': guard.sha256_file(target), 'bytes': descriptor['bytes']}
        after = guard.load_verified_sources()
        if pipeline.digest(after) != source_digest:
            raise ValueError('immutable source package changed during publication copying')
        final_binding, final_receipts, final_initial, final_chain = _receipt_chain(root)
        if (final_binding != binding or final_receipts != receipt_descriptors or final_initial != initial
                or select_files(root, final_chain, final_receipts) != selected):
            raise ValueError('canonical completed analysis changed during publication copying')
        guard.verify_products(output, selected)
        output_files = set()
        for path in output.rglob('*'):
            if path.is_symlink():
                raise ValueError('unexpected symlink in public aggregate copy')
            if path.is_file():
                output_files.add(path.relative_to(output).as_posix())
        if output_files != set(selected):
            raise ValueError('public copy contains files outside the explicit allowlist')
        if guard.verify_commit(code_commit, OWN_FILES) != own_hashes:
            raise ValueError('publication copier code changed during execution')
        receipt = {'version': VERSION, 'status': 'PUBLIC_COPY_VERIFIED', 'created_at_utc': pipeline.now(),
            'run_root': str(root), 'destination': str(output), 'analysis_binding': binding,
            'copier_code_commit': code_commit, 'copier_code_hashes': own_hashes,
            'copier_implementation_sha256': guard.sha256_file(__file__),
            'phase_receipts': receipt_descriptors, 'completed_phases': list(range(6)),
            'source_package_sha256': source_digest, 'initial_and_before_after_source_packages_identical': True,
            'source_lock_count': len(locks), 'python_network_connections_disabled': True,
            'OS_level_isolation_guaranteed': False, 'file_count': len(copied), 'files': copied,
            'explicit_allowlist': list(selected), 'maximum_individual_file_bytes': MAX_FILE_BYTES,
            'secret_scanner': 'offline_evaluation.server_v2.checkpoint.scan_file',
            'all_source_and_copy_bytes_sha256_identical': True,
            'retained_artifacts': {'canonical_run_root': str(root),
                'note': 'Prediction directories, raw actions, NPY arrays and other nested/non-aggregate artifacts remain only in the canonical run. No original model responses are copied.'},
            'new_inference_runs': 0, 'github_mutations': 0, 'draft_release_updates': 0, 'network_downloads': 0,
            **pipeline.ZERO_QUERIES}
    # Complete only after locked_sources' exit process audit succeeds.
    path = output / 'PUBLIC_COPY_RECEIPT.json'
    pipeline.write_json(path, receipt)
    try:
        scan_file(path)
    except Exception:
        path.unlink()  # Remove only this newly-created incomplete receipt.
        raise
    return {'status': receipt['status'], 'receipt': str(path), 'receipt_sha256': guard.sha256_file(path),
            'files': len(copied), 'research_api_requests_sent': 0, 'github_mutations': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--code-commit', required=True, help='Full committed copier SHA; file bytes are checked before and after copying.')
    args = parser.parse_args(argv)
    try:
        result = publish(args.run_root, args.destination, code_commit=args.code_commit)
    except Exception as exc:
        print(json.dumps({'status': 'STOP', 'error_type': type(exc).__name__,
            'safe_reason': 'Aggregate copy did not complete; any partial files are preserved without a completion receipt.',
            'research_api_requests_sent': 0, 'github_mutations': 0}))
        return 5
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
