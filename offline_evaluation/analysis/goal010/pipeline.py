"""Committed, phase-ordered CPU diagnostics; no generation or network requests."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import resource
import shutil
import time

from . import baselines, exact_oracle, factors, protocol, robustness, source_guard as guard
from ..goal008.pipeline import augment_slots

RUNS = Path('/root/minecraft-runs')
SPECIFICATION = 'reports/server_only/goal010/posthoc_specification.md'
PRODUCTION = tuple('offline_evaluation/analysis/goal010/' + name + '.py' for name in (
    '__init__', 'baselines', 'exact_oracle', 'factors', 'independent_geometry', 'pipeline',
    'protocol', 'robustness', 'source_guard'))
DIRECTORIES = ('phase0_protocol', 'phase1_baselines', 'phase2_factors',
               'phase3_exact_oracle', 'phase4_visibility', 'phase5_robustness')
ZERO_QUERIES = dict(research_api_requests_sent=0, api_key_queries=0, model_queries=0,
                   token_count_api_requests=0, human_experiments=0, checkpoint_mutations=0)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_text(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(value)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n'); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
                if isinstance(value, (list, dict)) else value for key, value in row.items()})


def safe_root(value, phase):
    path = Path(value)
    if (not path.is_absolute() or '..' in path.parts or path.parent != RUNS
            or not re.fullmatch(r'goal010-[a-zA-Z0-9_-]*\d{8}T\d{6}Z(?:-[a-zA-Z0-9_-]+)?', path.name)):
        raise ValueError('new direct Goal010 UTC research-run path required')
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('run path must not contain symlinks')
    if phase == 0 and path.exists():
        raise FileExistsError('Phase0 requires a fresh run root')
    if phase != 0 and not path.is_dir():
        raise ValueError('prior phase run root required')
    if not path.parent.is_dir() or shutil.disk_usage(path.parent).free < 10 * 1024**3:
        raise ValueError('existing run parent with at least10 GiB free required')
    return path


def code_binding(code_commit, specification_commit):
    current = {p.relative_to(guard.ROOT).as_posix() for p in (guard.ROOT / 'offline_evaluation/analysis/goal010').glob('*.py')}
    if current != set(PRODUCTION):
        raise ValueError('explicit production-code allowlist differs from current module files')
    code = guard.verify_commit(code_commit, [*PRODUCTION, SPECIFICATION])
    specification = guard.verify_commit(specification_commit, [SPECIFICATION])
    if code[SPECIFICATION] != specification[SPECIFICATION]:
        raise ValueError('code and specification commits bind different definitions')
    return {'code_commit': code_commit, 'specification_commit': specification_commit,
            'specification_sha256': specification[SPECIFICATION], 'code_and_specification_hashes': code}


def products(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('output inventory contains a symlink')
        if path.is_file() and path.suffix != '.npy':
            result[path.relative_to(root).as_posix()] = {'sha256': guard.sha256_file(path), 'bytes': path.stat().st_size}
    return result


def verify_predictions(root):
    path = root / 'phase1_baselines/baseline_predictions_manifest.json'
    if not path.exists():
        return
    manifest = guard.read_json(guard.regular(path))
    for row in manifest['predictions']:
        guard.verify_products(path.parent / 'predictions',
            {item['path']: {'sha256': item['sha256'], 'bytes': item['bytes']} for item in row['artifacts'].values()})


def verify_previous(root, phase, binding):
    previous = {}
    for number in range(phase):
        path = guard.regular(root / f'PHASE{number}_RECEIPT.json')
        receipt = guard.read_json(path)
        if (receipt.get('status') != 'PASS' or receipt.get('phase') != number
                or receipt.get('binding') != binding or receipt.get('run_root') != str(root)):
            raise ValueError('prior phase receipt is incomplete or has different code/specification/root binding')
        guard.verify_products(root, receipt['products'])
        previous[str(number)] = {'path': path.name, 'sha256': guard.sha256_file(path), 'bytes': path.stat().st_size}
    verify_predictions(root)
    return previous


def package_receipt(package):
    manifest = package['goal007_manifest']; slots = package['slots']
    if (len(slots) != 960 or [r['global_schedule_index'] for r in slots] != list(range(960))
            or Counter(r['status'] for r in slots) != {'success': 934, 'failed': 25, 'operational_unmeasurable': 1}
            or any(slots[942].get(key) is not None for key in ('metrics', 'usage', 'response'))
            or slots[942]['status'] != 'operational_unmeasurable'):
        raise ValueError('frozen slot counts or global942 null semantics changed')
    selected = slots[940]
    if (selected['status'] != 'success' or selected.get('original_attempt_id') in (None, selected['attempt_id'])
            or selected.get('source_attempt_id') != selected['original_attempt_id']):
        raise ValueError('global940 selected retry provenance changed')
    if (manifest['fixed_scope_attempts'] != 961 or manifest['historical_calls'] != 991
            or manifest['cost_validation']['unknown_usage_holds_released'] is not False):
        raise ValueError('historical counts or held unknown usage changed')
    return {'status': 'PASS', 'source_package_sha256': digest(package),
        'original_source_inventory': manifest['source_files'],
        'original_source_inventory_count': len(manifest['source_files']),
        'original_source_inventory_sha256': digest(manifest['source_files']),
        'github_checkpoint_proof_digest': digest(manifest['github_source_checkpoints']),
        'github_proofs_reused_without_download_or_remote_request': True,
        'goal007_manifest_sha256': guard.audit8.MANIFEST_SHA256,
        'goal008_manifest_sha256': guard.GOAL008_MANIFEST_SHA256,
        'fixed_slots': 960, 'valid': 934, 'model_failures': 25, 'operational_missing': 1,
        'fixed_scope_attempts': 961, 'historical_calls': 991,
        'missing942': {key: slots[942].get(key) for key in ('global_schedule_index', 'attempt_id', 'status', 'usage_known', 'completed_request_outcome', 'metrics', 'usage', 'response')},
        'selected940': {key: selected.get(key) for key in ('global_schedule_index', 'attempt_id', 'original_attempt_id', 'source_attempt_id', 'source_label', 'selection_reason', 'status')},
        'unknown_usage_holds': manifest['cost_validation']['unknown_usage_holds'],
        'unknown_usage_holds_released': False, 'budget_usd_not_invoice': manifest['cost_validation']['combined_budget_usd'],
        **ZERO_QUERIES}


def phase0(package, root, binding):
    out = root / DIRECTORIES[0]
    result = protocol.audit_protocol(package, out)
    if result.get('status') != 'PASS':
        raise ValueError('protocol audit did not pass')
    write_json(out / 'covariates.json', guard.metadata_rows(package))
    write_json(out / 'source_inventory_receipt.json', package_receipt(package))
    gate = {'status': 'PASS', 'input_kind': 'verified_sources', 'phase0_binding': 'PASS',
        'definitions_frozen': True, 'specification_sha256': binding['specification_sha256'],
        'code_commit': binding['code_commit'], 'goal007_manifest_sha256': guard.audit8.MANIFEST_SHA256}
    write_json(out / 'source_gate.json', gate)
    write_text(out / 'PROTOCOL_EVALUATION_SUPPLEMENT_JA.md', '''# 入力・評価・履歴の補足監査

Oは保存された人間可読フィールド付きJSON、Jは同じtyped treeを再シリアライズした入力である。両者は別々の単発Builder生成で、sampling条件を制御した表記だけの因果比較ではない。Pは開示GTのworld_bbox・voxel count・material set等の部分属性、Eはresmoke_v2_exact_cuboids_v1のsorted exact paletteとinclusive cuboidsであり、自然言語品質の母集団上限とは呼ばない。

GTはstorage bboxとNPY YXZ配列をworld XYZへ変換する。storage paddingとoccupied bbox、world originと撮影generation originを区別する。server_v2.2_strict_translationは占有最適の厳密平行移動だけを使い、その同じshiftで材質を評価する。rotation・scale・GT補修・指標別shift再選択は行わない。材質集合F1は座標上の材質一致とは異なる。既知失敗でもMAE/repairはnullであり、unknown-usageをfailure0へ変換しない。

960固定slotsは934 valid・25既知出力失敗・global942の520欠測1件。固定範囲961 attempts、歴史的991 callsを維持する。global940は元401と明示承認retryの選択関係を保存し、成功を1回だけ数える。unknown-usage予約2件を解除しない。USD21.56083625は保守的予算計算値で請求書ではない。過去の20比較、180 proxy、360探索結果と全既存保存証拠は再利用し改変しない。

元checkpointのローカル復元・GitHub検証証拠、旧manifestと全source inventoryを再照合した。今回GitHubから再取得していない。source_inventory_receipt.jsonとsource_gate.jsonにhash・選択provenance・定義commitを束縛する。研究API・キー・モデル・token-count照会は0件。Pythonの接続禁止と読取りlockはOSレベル隔離の保証ではない。
''')
    return {'status': 'PASS', 'protocol_manifest': 'phase0_protocol/PROTOCOL_MANIFEST.json'}


def execute_phase(number, package, root, gate, binding):
    out = root / DIRECTORIES[number]; spec = binding['specification_sha256']
    if number == 0:
        return phase0(package, root, binding)
    if number == 1:
        result = baselines.run_baselines(package, out, gate=gate, specification_sha256=spec)
        if result.get('status') != 'BASELINE_PHASE_COMPLETED': raise ValueError('baseline phase incomplete')
        return result
    if number == 2:
        result = factors.analyze_factors(package['slots'], audit_gate=gate, specification_sha256=spec)
        if len(result['records']) != 36: raise ValueError('all36 factor contrasts required')
        out.mkdir(exist_ok=False)
        write_json(out / 'description_builder_analysis.json', result)
        write_csv(out / 'description_builder_matched.csv', result['records'])
        write_csv(out / 'description_builder_pairs.csv', result['pairs'])
        write_csv(out / 'description_builder_exclusions.csv', result['exclusions'])
        request = guard.read_json(root / 'phase0_protocol/view_and_generation_request_audit.json')
        write_json(out / 'description_builder_confounds.json', request['summary'])
        write_text(out / 'FACTOR_INTERPRETATION_JA.md', '# 説明生成元の探索的比較\n\nPOST_HOC_EXPLORATORY。全36対比を保存した。同じ建築・Builder・O/J条件のcommon-valid pairのみを比較し、1建築1差の等重み平均を示す。差はAnthropic−OpenAI、Gemini−OpenAI、Gemini−Anthropic。IoUは高いほど、MAEは低いほど良い。\n\nbootstrap frameは各対比のcommon-valid建築であり、baselineのplanned frameとは異なる。2000 building drawsと95% percentile CIを使用し、新しいp値は計算しない。モデルversion・実画像・prompt・出力上限・時期を含む観測条件の交絡がある。固有能力や純粋な表記変更の因果効果を主張しない。全pairs・除外理由・NをJSON/CSVに残し、2 BuildersのNを合算しない。\n')
        return {'status': 'PASS', 'contrasts': len(result['records'])}
    if number == 3:
        result = exact_oracle.run_exact_oracle(package['slots'], guard.audit8.GOAL007 / 'MANIFEST.json', out,
                                               gate=gate, specification_sha256=spec)
        if result.get('status') != 'E_AUDIT_PASS': raise ValueError('E diagnostic incomplete')
        return result
    if number == 4:
        result = protocol.visibility_feasibility(root / 'phase0_protocol/PROTOCOL_MANIFEST.json', out)
        if result.get('status') != 'NOT_VALIDATED_NO_VISIBILITY_CONCLUSION':
            raise ValueError('unexpected visibility verdict for frozen feasibility protocol')
        return result
    metadata = guard.read_json(root / 'phase0_protocol/covariates.json')
    augmented = augment_slots(package['slots'], package['descriptions'])
    result = robustness.analyze_robustness(package['proxy_scores'], augmented, metadata,
                                          audit_gate=gate, specification_sha256=spec)
    if len(result['records']) != 1350: raise ValueError('all1350 robustness records required')
    out.mkdir(exist_ok=False)
    write_json(out / 'proxy_robustness.json', result)
    write_csv(out / 'proxy_robustness.csv', result['records'])
    write_text(out / 'PROXY_VALIDITY_JA.md', '# CPU属性proxyの頑健性\n\nPOST_HOC_EXPLORATORY。元5特徴と4個の固定component除外感度、5観測共変量について全1350行を保存した。元の180 scoresと360結果は変更していない。定数・不足・欠損・不安定CIもstatus付きで残した。\n\n独立単位は最大60建築で、180記述や2 Buildersを独立Nとして扱わない。層境界はplanned cohortから固定し、同値を分割せず、空層も残す。未知usageは0へ補完しない。両Builder平均には双方のpolicy適格性を要求する。bootstrapは層内planned building clusterを再標本化し、pooled rankは複製後に再計算する。within法は建築内center/rank後にclusterを再標本化する。\n\ncomponent除外は算術的感度で、新しい品質尺度や独立因果寄与ではない。文字数・GT量・密度による層別も因果調整ではなく、共有GTとvalid選択、多数の探索を考慮する。新p値・Holm・fit・CVを加えない。従来RQ2はNOT_TESTABLE_WITH_CURRENT_MATCHED_SCORESのままである。\n')
    return {'status': 'PASS', 'records': len(result['records']), 'status_counts': result['status_counts']}


def run_phase(phase, run_root, code_commit, specification_commit):
    if type(phase) is not int or phase not in range(6): raise ValueError('phase must be0..5')
    root = safe_root(run_root, phase)
    binding = code_binding(code_commit, specification_commit)
    previous = verify_previous(root, phase, binding) if phase else {}
    targets = [root / DIRECTORIES[phase], root / f'PHASE{phase}_RECEIPT.json',
               root / f'PHASE{phase}_STARTED.json', root / f'PHASE{phase}_STOP.json']
    if any(path.exists() for path in targets): raise FileExistsError('phase output already exists; preserve partial or completed work')
    started = time.monotonic(); cpu = time.process_time(); created = False
    try:
        with guard.locked_sources() as locks:
            before = guard.load_verified_sources(); before_digest = digest(before)
            if phase == 0: root.mkdir(exist_ok=False)
            write_json(root / f'PHASE{phase}_STARTED.json', {'phase': phase, 'status': 'STARTED', 'created_at_utc': now(), 'binding': binding})
            created = True
            gate = None
            if phase:
                gate = guard.read_json(root / 'phase0_protocol/source_gate.json')
                guard.require_gate(gate, binding['specification_sha256'])
                if gate.get('code_commit') != code_commit or gate.get('goal007_manifest_sha256') != guard.audit8.MANIFEST_SHA256:
                    raise ValueError('Phase0 gate code/source mismatch')
                initial = guard.read_json(root / 'phase0_protocol/source_inventory_receipt.json')
                if initial['source_package_sha256'] != before_digest:
                    raise ValueError('source package differs from Phase0 receipt')
            result = execute_phase(phase, before, root, gate, binding)
            after = guard.load_verified_sources()
            if digest(after) != before_digest: raise ValueError('immutable sources changed during phase')
            if code_binding(code_commit, specification_commit) != binding: raise ValueError('code changed during phase')
            verify_previous(root, phase, binding)
            verify_predictions(root)
            receipt = {'version': 'goal010_phase_receipt_v1', 'status': 'PASS', 'phase': phase,
                'run_root': str(root), 'created_at_utc': now(), 'binding': binding, 'prior_receipts': previous,
                'source_package_sha256': before_digest, 'source_before_after_unchanged': True,
                'source_lock_count': len(locks), 'python_network_connections_disabled': True,
                'OS_level_isolation_guaranteed': False, 'phase_result_status': result.get('status'),
                'products': products(root), 'baseline_arrays_bound_through_prediction_manifest': phase >= 1,
                'performance': {'elapsed_seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu,
                    'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'free_disk_bytes': shutil.disk_usage(root).free,
                    'run_disk_bytes': sum(p.stat().st_size for p in root.rglob('*') if p.is_file())}, **ZERO_QUERIES}
        # locked_sources' exit process audit must also pass before completion is recorded.
        write_json(root / f'PHASE{phase}_RECEIPT.json', receipt)
        return receipt
    except Exception as exc:
        if created:
            stop = root / f'PHASE{phase}_STOP.json'
            if not stop.exists():
                write_json(stop, {'version': 'goal010_phase_stop_v1', 'status': 'STOP', 'phase': phase,
                    'created_at_utc': now(), 'binding': binding, 'error_type': type(exc).__name__,
                    'safe_reason': 'Phase failed; partial outputs retained. No automatic retry or phase completion.',
                    'performance': {'elapsed_seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu}, **ZERO_QUERIES})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, type=int, choices=range(6))
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--specification-commit', required=True)
    args = parser.parse_args(argv)
    try:
        receipt = run_phase(args.phase, args.run_root, args.code_commit, args.specification_commit)
    except Exception as exc:
        print(json.dumps({'status': 'STOP', 'error_type': type(exc).__name__,
            'safe_reason': 'Phase did not complete; inspect preserved local receipts.', **ZERO_QUERIES}), flush=True)
        return 5
    print(json.dumps({'status': receipt['status'], 'phase': args.phase,
                      'receipt': str(Path(args.run_root) / f'PHASE{args.phase}_RECEIPT.json'),
                      'performance': receipt['performance'], **ZERO_QUERIES}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
