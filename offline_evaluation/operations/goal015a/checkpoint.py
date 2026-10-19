"""Independent Goal015A packages; immutable parent files are only read/copied."""
from __future__ import annotations
import ast
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from urllib.parse import quote
import uuid

from ...experiments.goal015 import checkpoint as frozen
from ...experiments.goal015 import runtime as original
from ...server_v2.checkpoint import safe_relative, scan_file
from . import evaluate

descriptor, path_safe, read_json = original.descriptor, original.path_safe, original.read_json
require, _save = original.require, original._save
Github = frozen.Github
REPOSITORY, BRANCH, MAX_BYTES = frozen.REPOSITORY, frozen.BRANCH, frozen.MAX_BYTES


def verify_archive(path, expected_sha256, *, destination=None):
    from . import runtime
    require(descriptor(path)['sha256'] == expected_sha256, 'archive SHA differs')
    target = path_safe(destination) if destination is not None else None
    if target is not None:
        require(not target.exists(), 'new verification destination required')
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        names = [item.name for item in members]
        require(len(names) == len(set(names)) and 'FILES.json' in names, 'archive duplicate/missing manifest')
        total = 0
        for item in members:
            safe_relative(item.name)
            require(item.isfile() and PurePosixPath(item.name).as_posix() == item.name, 'archive regular canonical files only')
            total += item.size
            require(total <= MAX_BYTES, 'archive expanded limit')
        def document(name):
            return json.loads(archive.extractfile(name).read())
        manifest = document('FILES.json')
        require(manifest.get('schema') == 'goal015a_checkpoint_files_v1', 'Goal015A archive manifest required')
        require(set(names) == set(manifest['files']) | {'FILES.json'}, 'archive membership differs')
        required = {'plan.json', 'parent/plan.json', 'parent/enabled/ledger.json',
            'parent/enabled/approval.json', 'parent/enabled/authority.json',
            'child/enabled/ledger.json', 'child/enabled/approval.json', 'child/enabled/authority.json'}
        require(required <= set(names), 'parent/child execution controls missing')
        child, parent = document('plan.json'), document('parent/plan.json')
        child_ledger, parent_ledger = document('child/enabled/ledger.json'), document('parent/enabled/ledger.json')
        require(child['plan_id'] == original.identity(child) == manifest['plan_id']
                and parent['plan_id'] == original.identity(parent) == child['parent_plan_id'], 'archive plan identity differs')
        calls, provenance = evaluate.select_calls(parent, parent_ledger, child, child_ledger)
        mappings = manifest['original_paths']
        require(set(mappings) == set(manifest['files']) and len(set(mappings.values())) == len(mappings),
                'archive original path mapping differs')
        reverse = {value: key for key, value in mappings.items()}
        def bound(desc):
            require(desc['path'] in reverse, 'selected artifact absent from archive')
            member = reverse[desc['path']]
            require(manifest['files'][member] == {key: desc[key] for key in ('bytes', 'sha256')}, 'selected artifact hash differs')
            return member
        child_approval = document('child/enabled/approval.json')
        child_authority = document('child/enabled/authority.json')
        parent_approval = document('parent/enabled/approval.json')
        parent_authority = document('parent/enabled/authority.json')
        require(child.get('schema') == 'goal015a_disabled_plan_v1'
                and child.get('api_allowed') is False and child.get('budget_approved') is False
                and child.get('proposed_budget_usd') == '7.00' and child.get('planned_calls') == 8
                and child.get('total_planned_calls') == 80 and child.get('max_retries') == 0,
                'archived child fixed scope differs')
        require(child_ledger['plan_id'] == child_approval['plan_id'] == child['plan_id']
                and child_approval['parent_plan_id'] == child['parent_plan_id']
                and child_approval['canonical_root'] == child['canonical_root']
                and child_approval['approved_combined_budget_usd'] == '7.00'
                and child_approval['scope_calls'] == 8
                and child_approval['parent_known_cost_usd'] == child['parent_known_cost_usd']
                and child_approval['status'] == 'APPROVED'
                and child_approval.get('approval_text') == runtime.approval_text(child),
                'archived child approval differs')
        require(child_authority.get('status') == 'APPROVED'
                and child_authority.get('plan_id') == child['plan_id']
                and child_authority.get('parent_plan_id') == parent['plan_id']
                and child_authority.get('canonical_root') == child['canonical_root']
                and child_authority.get('scope_calls') == 8 and child_authority.get('max_retries') == 0
                and child_authority.get('parent_ledger') == child['parent']['ledger'],
                'archived child authority differs')
        require(bound(child_authority['plan']) == 'plan.json'
                and bound(child_authority['approval']) == 'child/enabled/approval.json',
                'archived child authority binding differs')
        require(parent_ledger['plan_id'] == parent_approval['plan_id'] == parent['plan_id']
                and parent_approval['status'] == 'APPROVED' and parent_approval['scope_calls'] == 80
                and parent_approval['approved_budget_usd'] == '5.00'
                and parent_approval['canonical_root'] == parent['canonical_root'],
                'archived parent approval differs')
        require(parent_authority.get('status') == 'APPROVED'
                and parent_authority.get('plan_id') == parent['plan_id']
                and parent_authority.get('canonical_root') == parent['canonical_root']
                and parent_authority.get('scope_calls') == 80 and parent_authority.get('max_retries') == 0
                and parent_authority.get('plan') == child['parent']['plan']
                and parent_authority.get('approval') == child['parent']['approval'],
                'archived parent authority differs')
        for key in ('plan', 'approval', 'authority', 'ledger', 'evaluation', 'github_verification', 'checkpoint'):
            bound(child['parent'][key])
        for row in child_ledger['calls']:
            state = row['state']
            require(state in ('NOT_SENT', 'SENT_PENDING', 'UNKNOWN_USAGE', 'SETTLED'), 'archived child state differs')
            reserve = original._decimal(row['reservation_usd'])
            require(reserve in (original._decimal('0'), original._decimal(row['reserve_usd'])),
                    'archived child reservation differs')
            if state == 'NOT_SENT':
                require(row.get('raw') is None and row.get('response_text_path') is None
                        and row.get('send_status') == 'NOT_SENT' and row.get('response_received') is False
                        and row.get('completed_response') is False
                        and 'started_at_utc' not in row and 'finished_at_utc' not in row,
                        'archived NOT_SENT has transmission evidence')
                require(not any('child/enabled/responses/' + row['call_id'] + suffix in names
                                for suffix in ('.raw.json', '.text.txt')),
                        'archived NOT_SENT has response files')
            else:
                require(reserve == original._decimal(row['reserve_usd']), 'archived attempted reservation differs')
            if state == 'SETTLED':
                require(row.get('usage_known') is True and row.get('known_cost_usd') is not None
                        and row.get('raw') is not None, 'archived settled evidence missing')
                original._decimal(row['known_cost_usd'])
            else:
                require(row.get('usage_known') is False and row.get('known_cost_usd') is None,
                        'archived unresolved usage differs')
            for key in ('raw', 'response_text_path'):
                if row.get(key) is not None:
                    require(bound(row[key]).startswith('child/enabled/responses/'),
                            'archived child response escaped canonical subtree')
        for row in calls:
            for key in ('raw', 'response_text_path'):
                if row.get(key) is not None:
                    bound(row[key])
        merged = manifest.get('merged_evaluation')
        if merged is not None:
            data = document(bound(merged))
            require(len(data['rows']) == 80 and data['plan_binding']['child_plan_id'] == child['plan_id'], 'merged frame differs')
            binding = data['plan_binding']
            require(binding['plan_id'] == parent['plan_id'] and binding['plan'] == child['parent']['plan']
                    and binding['ledger'] == child['parent']['ledger'], 'merged parent binding differs')
            require(bound(binding['child_plan']) == 'plan.json'
                    and bound(binding['child_ledger']) == 'child/enabled/ledger.json', 'merged child binding differs')
            sources = document(bound(binding['selected_sources']))
            require(sources['rows'] == provenance, 'merged selected sources differ')
            parent_evaluation = document(bound(child['parent']['evaluation']))
            require(len(parent_evaluation['rows']) == 80, 'archived parent evaluation frame differs')
            for row, saved, source, original_row in zip(data['rows'], calls, provenance,
                                                       parent_evaluation['rows']):
                require(all(row[key] == saved[key] for key in evaluate.frozen.IDENTITY)
                        and row['state'] == saved['state'] and row['selected_source'] == source,
                        'merged selected row differs')
                if source['parent_original_state'] == 'SETTLED':
                    require({key: value for key, value in row.items() if key != 'selected_source'} == original_row,
                            'archived reused parent evaluation row changed')
                require(row['artifacts']['request_payload'] == saved['payload'], 'merged payload reference differs')
                for key in ('raw', 'response_text_path'):
                    require(row['artifacts'].get(key) == saved.get(key), 'merged raw source differs')
                for key, desc in row['artifacts'].items():
                    if key != 'request_payload':
                        bound(desc)
                if row['metrics'] is not None:
                    require(document(bound(row['artifacts']['metrics.json'])) == row['metrics'], 'merged metric bytes differ')
        for item in members:
            raw = archive.extractfile(item).read()
            if item.name != 'FILES.json':
                require(manifest['files'][item.name] == {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}, 'member hash differs')
        if target is not None:
            target.mkdir(parents=True)
            for item in members:
                out = target / item.name
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open('xb') as stream:
                    stream.write(archive.extractfile(item).read())
                scan_file(out)
    return {'status': 'VERIFIED', 'archive_sha256': expected_sha256, 'files': len(manifest['files']),
            'plan_id': child['plan_id'], 'selected_slots': 80, 'parent_slots': 72, 'child_slots': 8,
            'merged_evaluation_included': merged is not None, 'geometry_or_generation_reexecuted': False,
            'external_dependencies_included': False}


def create_checkpoint(plan_path, child_ledger_path=None, *, reason, target_commit=None):
    """Called under existing paired locks; create_save_only is the public wrapper."""
    from . import runtime
    plan = runtime.verify_plan(plan_path)
    parent = runtime.verify_parent(plan)
    root = Path(plan['canonical_root'])
    ledger_path = path_safe(child_ledger_path or root / 'enabled/ledger.json')
    require(ledger_path == root / 'enabled/ledger.json', 'canonical child ledger required')
    ledger = runtime.verify_enabled(plan)['ledger']
    evaluate.parent_evaluation(plan, parent)
    summary = evaluate.find_existing_evaluation(plan_path)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    name = 'goal015a-' + plan['plan_id'][:12] + '-' + stamp + '-' + uuid.uuid4().hex[:6]
    directory = root / 'checkpoints' / name
    directory.mkdir(parents=True, exist_ok=False)
    entries = {'plan.json': path_safe(plan_path)}
    parent_root = Path(parent['plan']['canonical_root'])
    def add(path, prefix, source_root):
        path = path_safe(path)
        require(path.is_file(), 'regular checkpoint source required')
        member = prefix + '/' + path.relative_to(source_root).as_posix()
        safe_relative(member)
        entries[member] = path
    def tree(source, prefix, source_root):
        source = path_safe(source)
        for path in sorted(source.rglob('*')):
            require(not path.is_symlink(), 'checkpoint symlink forbidden')
            if path.is_dir():
                continue
            if not path.name.startswith('.pending-'):
                add(path, prefix, source_root)
    tree(root / 'enabled', 'child', root)
    add(parent_root / 'plan.json', 'parent', parent_root)
    tree(parent_root / 'enabled', 'parent', parent_root)
    for row in parent['ledger']['calls']:
        for key in ('raw', 'response_text_path'):
            if row.get(key) is not None:
                add(Path(row[key]['path']), 'parent', parent_root)
    tree(Path(plan['parent']['evaluation']['path']).parent, 'parent', parent_root)
    entries['parent/github_verification.json'] = path_safe(plan['parent']['github_verification']['path'])
    entries['parent/original_checkpoint.json'] = path_safe(plan['parent']['checkpoint']['path'])
    if summary is not None:
        tree(Path(summary['evaluation_artifact']['path']).parent, 'parent', parent_root)
    files = {}
    for member, path in entries.items():
        scan_file(path)
        desc = descriptor(path)
        files[member] = {key: desc[key] for key in ('bytes', 'sha256')}
    manifest = {'schema': 'goal015a_checkpoint_files_v1', 'plan_id': plan['plan_id'], 'files': files,
        'original_paths': {name: str(path) for name, path in entries.items()},
        'merged_evaluation': summary['evaluation_artifact'] if summary else None,
        'external_dependencies': {'bundled': False, 'categories': [
            'Original request payloads and description sources', 'GT voxel/bbox exports',
            'Frozen implementation/environment', 'Referenced official pricing document bodies'],
            'request_payloads': [row['payload'] for row in parent['plan']['schedule']],
            'gt_sources': parent['plan']['gt_sources']}}
    archive_path = directory / 'goal015a-state.tar.gz'
    with archive_path.open('xb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as gz, tarfile.open(fileobj=gz, mode='w') as archive:
        for member, path in entries.items():
            raw = path.read_bytes()
            require(files[member] == {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}, 'source changed while checkpointing')
            item = tarfile.TarInfo(member); item.size = len(raw); item.mode = 0o600
            archive.addfile(item, io.BytesIO(raw))
        raw = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
        item = tarfile.TarInfo('FILES.json'); item.size = len(raw); item.mode = 0o600
        archive.addfile(item, io.BytesIO(raw))
    arc = descriptor(archive_path)
    require(arc['bytes'] <= MAX_BYTES, 'checkpoint exceeds1GiB')
    restoration = verify_archive(archive_path, arc['sha256'], destination=directory / 'verified-restoration')
    if target_commit is None:
        target_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[3]).decode().strip()
    require(len(target_commit) == 40 and all(c in '0123456789abcdef' for c in target_commit), 'full commit identity required')
    package = {'schema': 'goal015a_checkpoint_v1', 'status': 'LOCAL_VERIFIED', 'github_status': 'NOT_UPLOADED',
        'checkpoint_id': name, 'directory': str(directory), 'canonical_root': str(root), 'plan_id': plan['plan_id'],
        'parent_plan_id': plan['parent_plan_id'], 'tag': 'checkpoint-' + name, 'target_commitish': target_commit,
        'reason': reason, 'assets': [{'name': archive_path.name, 'bytes': arc['bytes'], 'sha256': arc['sha256']}],
        'archive_sha256': arc['sha256'], 'restoration': restoration, 'fixture_only': plan.get('fixture_only') is True,
        'external_dependencies_included': False, 'research_requests_sent_by_save': 0,
        'merged_evaluation': manifest['merged_evaluation']}
    _save(directory / 'checkpoint.json', package, exclusive=True)
    return {'status': 'LOCAL_VERIFIED', 'path': str(directory), 'checkpoint': descriptor(directory / 'checkpoint.json'),
            'archive_sha256': arc['sha256'], 'research_api_requests_sent': 0}


def create_save_only(plan_path, *, reason='SAVE_ONLY_RETRY'):
    from . import runtime
    plan = runtime.verify_plan(plan_path)
    with runtime.paired_locks(plan, create_child=False):
        return create_checkpoint(plan_path, reason=reason)


@lru_cache(maxsize=1)
def _uploader():
    """Reuse the frozen upload algorithm with only package namespace changed.

    The copied function keeps numeric release/asset IDs, Draft identity checks,
    no overwrite, paginated lookup, redownload hashing and safe restoration.
    No original module/function is patched, and no provider module is imported.
    """
    tree = ast.parse(Path(frozen.__file__).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'upload_checkpoint']
    require(len(functions) == 1, 'frozen upload function missing')
    class Namespace(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                replacements = {'goal015_checkpoint_v1': 'goal015a_checkpoint_v1',
                    'goal015-': 'goal015a-', 'goal015-state.tar.gz': 'goal015a-state.tar.gz'}
                node.value = replacements.get(node.value, node.value)
                node.value = node.value.replace('Goal015 immutable', 'Goal015A immutable')
            return node
    module = ast.fix_missing_locations(Namespace().visit(ast.Module(body=functions, type_ignores=[])))
    namespace = {name: globals()[name] for name in ('path_safe', 'read_json', 'require', 'descriptor',
        'verify_archive', '_save', 'Github', 'REPOSITORY', 'BRANCH', 'quote')}
    exec(compile(module, str(frozen.__file__), 'exec'), namespace)
    return namespace['upload_checkpoint']


def upload_checkpoint(directory, verification_directory, *, release_id=None, allow_create=False, github=None):
    return _uploader()(directory, verification_directory, release_id=release_id,
                       allow_create=allow_create, github=github)
