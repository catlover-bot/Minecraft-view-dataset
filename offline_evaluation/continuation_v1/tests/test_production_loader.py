"""Exercise the real prepare/load pipeline with a workspace-local disabled copy.

Only the destination root and authority path are redirected to this fixture.
The loader, source readers, completed-parent proof and immutable manifest checks
are real. No production plan or state is written; no transport is available.
"""
from pathlib import Path
import socket

import pytest

from offline_evaluation.continuation_v1 import plan as planning, runner
from offline_evaluation.server_v2.generation import GenerationError
from offline_evaluation.server_v2.io_utils import ROOT


def test_real_production_loader_accepts_string_descriptors_and_keeps_plan_disabled(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail('network forbidden during real-loader regression fixture')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(planning, 'CANONICAL_ROOT', tmp_path)
    monkeypatch.setattr(planning, 'AUTHORITY_PATH', tmp_path / 'fixture-authority.json')
    pricing = tmp_path / 'verified-pricing.json'
    pricing.write_bytes((ROOT / 'reports/server_only/goal005/pricing_refresh_20261014T084535Z/official_pricing.json').read_bytes())
    references = tmp_path / 'checkpoint-sources.json'
    references.write_bytes((ROOT / 'reports/server_only/goal006/checkpoint_sources.json').read_bytes())
    destination = tmp_path / 'disabled-fixture'
    plan = planning.prepare(str(destination), pricing_file=str(pricing), commit='1' * 40,
                            checkpoint_sources=str(references))
    path = destination / 'plan.json'
    assert isinstance(plan['identity']['pricing_source_file_path'], str)
    assert isinstance(plan['identity']['checkpoint_sources_path'], str)
    assert planning.load_plan(str(path)) == plan
    assert planning.load_plan(path) == plan
    state = runner.status(str(path))
    assert state['attempted_inputs'] == 0 and state['not_run_inputs'] == 800
    assert state['used']['combined']['calls'] == 190
    assert not state['approved'] and not (destination / 'enabled').exists()
    assert not planning.AUTHORITY_PATH.exists()
    assert not planning.AUTHORITY_PATH.with_suffix('.json.lock').exists()
    before = path.read_bytes()
    copied = tmp_path / 'copied.json'
    copied.write_bytes(before)
    with pytest.raises(GenerationError, match='canonical_plan_path'):
        planning.load_plan(str(copied))
    references.write_bytes(references.read_bytes() + b'\n')
    with pytest.raises(GenerationError, match='checkpoint_source_reference_changed'):
        planning.load_plan(str(path))
    assert path.read_bytes() == before
