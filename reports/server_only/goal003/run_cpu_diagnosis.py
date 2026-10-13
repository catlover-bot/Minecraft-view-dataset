"""Reproduce Goal003 CPU diagnosis with every research transport disabled."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from offline_evaluation.server_v2 import generation, providers
from offline_evaluation.server_v2.data import load_scene
from offline_evaluation.server_v2.goal003_diagnosis import build_diagnosis, diagnose

CAMPAIGN = Path('/root/minecraft-runs/goal002-approved-smoke-001')
LOGS = Path('/root/minecraft-runs/goal002-manual-review.WkFtAAr6')


def forbidden(*_args, **_kwargs):
    raise AssertionError('Goal003 diagnosis prohibits generation and network transport')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--verify-existing', action='store_true',
                        help='Recompute in memory and verify all saved JSON without replacing outputs')
    args = parser.parse_args()
    records = generation.read_json(CAMPAIGN / 'runs/openai/inputs.json')
    gt = {}
    for scene in sorted({r['scene_id'] for r in records}):
        dataset, building = scene.rsplit('_building_', 1)
        gt[scene] = load_scene(ROOT / 'datasets' / dataset / f'building_{building}').grid
    with patch.object(generation, 'generate', forbidden), patch.object(providers, 'send_request', forbidden), \
            patch.object(providers, 'http_transport', forbidden), patch.object(urllib.request, 'urlopen', forbidden), \
            patch.object(socket, 'create_connection', forbidden):
        if args.verify_existing:
            result = build_diagnosis(CAMPAIGN, LOGS, gt_by_scene=gt)
            for name, payload in result.items():
                assert generation.read_json(args.destination / f'{name}.json') == json.loads(json.dumps(payload)), name
        else:
            result = diagnose(CAMPAIGN, LOGS, args.destination, gt_by_scene=gt)
    print(json.dumps({'destination': str(args.destination), 'verified_existing': args.verify_existing,
                      'used': result['ledger_reconciliation']['used'],
                      'providers': result['ledger_reconciliation']['providers'],
                      'metric_exact_matches': len(result['metric_verification']),
                      'original_hashes_unchanged': result['original_hashes']['unchanged'],
                      'additional_research_api_requests': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
