import fcntl
import pytest
from offline_evaluation.server_v2.checkpoint import freeze_completed


def test_budget_writer_lock_blocks_freeze(tmp_path):
    (tmp_path/'task.json').write_text('{"done":true}')
    with (tmp_path/'budget.lock').open('a+b') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(ValueError,match='active'):freeze_completed(tmp_path)
    assert not (tmp_path/'completed.json').exists()
    assert freeze_completed(tmp_path)['files']==1


def test_freeze_preserves_not_run_and_refuses_changed_terminal_data(tmp_path):
    (tmp_path/'state.json').write_text('{"attempts":[]}')
    assert freeze_completed(tmp_path)['experiment_completion']=='NOT_RUN'
    assert freeze_completed(tmp_path)['status']=='ALREADY_FROZEN'
    (tmp_path/'state.json').write_text('{"attempts":[1]}')
    with pytest.raises(ValueError,match='changed'):freeze_completed(tmp_path)
