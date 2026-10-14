"""Suffix-weighted descriptive forecasts from recorded, settled attempts only."""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import csv

from ..server_v2.generation import object_hash, read_json, utc_now, write_json, GenerationError
from ..server_v2.data import sha256_file


def _stamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.utcoffset().total_seconds() != 0:
        raise GenerationError('attempt_time_must_be_UTC')
    return result


def execution_intervals(entries, gap_seconds=60):
    """Keep every gap and label >60s pauses; elapsed is not provider compute time."""
    intervals, gaps, active = [], [], Decimal(0)
    last = None
    for entry in entries:
        if not entry.get('finished_at_utc'):
            raise GenerationError('settled_finished_attempts_required_for_forecast')
        start, end = _stamp(entry['started_at_utc']), _stamp(entry['finished_at_utc'])
        if end < start or (last and start < last):
            raise GenerationError('nonsequential_or_invalid_attempt_times')
        elapsed = Decimal(str((end - start).total_seconds()))
        active += elapsed
        gap = Decimal(str((start - last).total_seconds())) if last else Decimal(0)
        if last:
            gaps.append({'before_schedule_index': entry['schedule_index'], 'seconds': float(gap),
                         'classified_pause': gap > gap_seconds})
        if not intervals or gap > gap_seconds:
            intervals.append({'start_utc': entry['started_at_utc'], 'end_utc': entry['finished_at_utc'],
                              'first_schedule_index': entry['schedule_index'],
                              'last_schedule_index': entry['schedule_index'], 'attempts': 1})
        else:
            intervals[-1].update(end_utc=entry['finished_at_utc'], last_schedule_index=entry['schedule_index'])
            intervals[-1]['attempts'] += 1
        last = end
    wall = (_stamp(entries[-1]['finished_at_utc']) - _stamp(entries[0]['started_at_utc'])).total_seconds() if entries else 0
    return {'active_attempt_seconds': float(active), 'calendar_wall_seconds': wall,
            'pause_threshold_seconds': gap_seconds, 'execution_intervals': intervals,
            'between_attempt_gaps': gaps, 'classified_pause_seconds': sum(g['seconds'] for g in gaps if g['classified_pause']),
            'interpretation': 'Active duration is recorded start-to-finish request/parse/execution time, not isolated model compute. Calendar wall includes downtime; pauses are not extrapolated.'}


def forecast(schedule, ledger, remaining, pricing, cumulative_usage):
    entries = ledger['entries']
    if len(entries) != len(schedule) or any(e.get('usage') is None or e['status'] == 'reserved' for e in entries):
        raise GenerationError('complete_known_usage_sample_required')
    key = lambda r: (r['provider'], r['condition'], r.get('description_model'))
    remaining_counts = Counter(key(r) for r in remaining)
    observed = defaultdict(list)
    seen = set()
    for record, entry in zip(schedule, entries):
        pair = (record['provider'], record['input_id'])
        if pair in seen or (entry['provider'], entry['input_id']) != pair or entry['schedule_index'] != record['schedule_index']:
            raise GenerationError('forecast_sample_identity_mismatch')
        seen.add(pair)
        observed[key(record)].append(entry)
    suffix_pairs = [(r['provider'], r['input_id']) for r in remaining]
    if len(set(suffix_pairs)) != len(suffix_pairs) or seen & set(suffix_pairs):
        raise GenerationError('forecast_suffix_duplicates_or_overlaps_sample')
    if not remaining_counts.keys() <= observed.keys():
        raise GenerationError('forecast_missing_observed_stratum')
    groups, projected_cost, projected_tokens, projected_seconds = [], Decimal(0), Decimal(0), Decimal(0)
    for group in sorted(observed, key=lambda k: tuple('' if x is None else x for x in k)):
        es = observed[group]
        n, target = len(es), remaining_counts[group]
        cost = sum((Decimal(str(e['charged_cost_usd'])) for e in es), Decimal(0))
        inp, out = sum(e['charged_input_tokens'] for e in es), sum(e['charged_output_tokens'] for e in es)
        seconds = sum((Decimal(str((_stamp(e['finished_at_utc']) - _stamp(e['started_at_utc'])).total_seconds())) for e in es), Decimal(0))
        factor = Decimal(target) / n
        projected_cost += cost * factor
        projected_tokens += Decimal(inp + out) * factor
        projected_seconds += seconds * factor
        groups.append({'provider': group[0], 'condition': group[1], 'description_model': group[2],
                       'observed_calls': n, 'remaining_calls': target,
                       'observed_input_tokens': inp, 'observed_output_tokens': out,
                       'observed_cost_usd_upper_bound': float(cost), 'observed_active_seconds': float(seconds),
                       'projected_input_tokens': float(Decimal(inp) * factor),
                       'projected_output_tokens': float(Decimal(out) * factor),
                       'projected_cost_usd': float(cost * factor), 'projected_active_seconds': float(seconds * factor)})
    context_tokens = sum(pricing[r['provider']]['context_window_tokens'] + 32768 for r in remaining)
    reservation_cost = sum((Decimal(str(pricing[r['provider']]['input_usd_per_million'])) * pricing[r['provider']]['context_window_tokens']
                            + Decimal(str(pricing[r['provider']]['output_usd_per_million'])) * 32768) / 1000000 for r in remaining)
    total = Decimal(str(cumulative_usage['cost_usd_upper_bound'])) + projected_cost
    return {'version': 'goal006_suffix_weighted_forecast_v1', 'created_at_utc': utc_now(), 'research_api_requests_sent': 0,
            'groups': groups, 'observed_calls': len(entries), 'remaining_calls': len(remaining),
            'projected_remaining_cost_usd': float(projected_cost), 'projected_total_cost_usd': float(total),
            'projected_remaining_tokens': float(projected_tokens), 'projected_remaining_active_seconds': float(projected_seconds),
            'projected_remaining_active_hours': float(projected_seconds / 3600),
            'observed_time': execution_intervals(entries), 'prior_cumulative_usage': cumulative_usage,
            'reservation': {'new_max_tokens': context_tokens, 'cumulative_max_tokens': cumulative_usage['tokens'] + context_tokens,
                            'sum_all_remaining_full_context_and_output_usd': float(reservation_cost),
                            'proposed_cumulative_cost_cap_usd': 30.0, 'approved': False,
                            'interpretation': 'Each next request is reserved durably and settled on known usage; sum is neither prepayment nor predicted actual cost.'},
            'uncertainty': 'Descriptive forecast from first10 fixed buildings, weighted to frozen suffix composition. Building/output-length differences and future throughput/availability remain uncertain. No completion guarantee; server charges excluded.'}


def save(parent_plan, manifest, ledger, destination, cumulative_usage):
    """Caller supplies once-validated originals; only new report files are written."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    schedule = parent_plan['identity']['schedule']
    full = manifest['identity']['full_schedule']
    if full[:160] != schedule or len(full) != 960:
        raise GenerationError('forecast_full_manifest_prefix_mismatch')
    result = forecast(schedule, ledger, full[160:], parent_plan['identity']['pricing'], cumulative_usage)
    result['source'] = {'parent_plan_identity_hash': parent_plan['identity_hash'],
                        'manifest_identity_hash': manifest['identity_hash'],
                        'ledger_sha256': sha256_file(Path(parent_plan['identity']['enabled_directory']) / 'ledger.json')}
    write_json(destination / 'costs.json', result)
    with (destination / 'provider_condition_description.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['groups'][0])); writer.writeheader(); writer.writerows(result['groups'])
    return result
