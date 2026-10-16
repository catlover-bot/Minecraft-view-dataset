from pathlib import Path
from types import SimpleNamespace

import pytest

from offline_evaluation.analysis.goal007 import cli


def test_invalid_output_cannot_create_or_touch_sources(tmp_path):
    target=tmp_path/'outside_research_directory'
    with pytest.raises(cli.GenerationError,match='new_research_output'):
        cli.run(SimpleNamespace(destination=str(target)))
    assert not target.exists()


def test_failed_checkpoint_gate_stops_before_output_or_statistics(monkeypatch):
    target=Path('/root/minecraft-runs/goal007-fixture-must-never-be-created')
    assert not target.exists()
    monkeypatch.setattr(cli,'verify_code_commit',lambda _: {})
    def failed(_): raise cli.audit.AuditError('checkpoint_corrupt_fixture')
    monkeypatch.setattr(cli.audit,'verify_gate',failed)
    def forbidden(*a,**k): raise AssertionError('statistics must not be called')
    monkeypatch.setattr(cli.statistics,'analyze_slots',forbidden)
    with pytest.raises(cli.audit.AuditError,match='checkpoint_corrupt'):
        cli.run(SimpleNamespace(destination=str(target),code_commit='a'*40,audit_gate='fixture'))
    assert not target.exists()


def test_explicit_implementation_commit_required():
    with pytest.raises(cli.GenerationError,match='explicit_code_commit'):
        cli.verify_code_commit('HEAD')


def test_failure_layers_never_merge_attempts_and_selected_slots():
    rows=[{'provider':'openai','condition':'O','status':'failed','failure_class':'operational_unknown','failure_kind':'auth_failure'},
          {'provider':'openai','condition':'J','status':'failed','failure_class':'operational_unknown','failure_kind':'http_failure'},
          {'provider':'anthropic','condition':'E','status':'failed','failure_class':'model_output_failure','failure_kind':'truncation'}]
    selected=[dict(rows[1],status='operational_unmeasurable'),rows[2]]
    taxonomy,model,operational=cli.failure_tables({'attempts':rows,'rows':selected})
    assert len(model)==1 and len(operational)==2
    assert sum(r['count'] for r in taxonomy if r['record_type']=='attempt')==3
    assert sum(r['count'] for r in taxonomy if r['record_type']=='selected_slot')==2
