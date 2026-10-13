"""Zero-send preview/refusal and fixed scope controls; CPU fixture only."""
from copy import deepcopy
import errno
from pathlib import Path

import pytest

from offline_evaluation.batch_v1 import cli, plan, runner
from offline_evaluation.batch_v1.tests.test_runner import fixture_plan
from offline_evaluation.server_v2.generation import GenerationError, read_json, write_json


def test_preview_no_keys_no_tty_no_activation_even_stale(fixture_plan, monkeypatch, capsys):
    monkeypatch.delenv('OPENAI_API_KEY')
    monkeypatch.delenv('ANTHROPIC_API_KEY')
    monkeypatch.setattr(cli, '_confirm_from_tty', lambda *_: pytest.fail('preview opened TTY'))
    monkeypatch.setattr(runner, 'initialize_enabled', lambda *_a, **_k: pytest.fail('preview enabled'))
    assert cli.launch(fixture_plan['path'], preview_only=True) == 0
    assert '160' in capsys.readouterr().out
    assert not Path(fixture_plan['plan']['identity']['enabled_directory']).exists()


@pytest.mark.parametrize('answer', [False, 'io_error'])
def test_tty_refusal_or_io_error_cannot_enable(fixture_plan, monkeypatch, answer):
    monkeypatch.setattr(plan, 'verify_parent', lambda *_: None)
    def confirm(expected):
        assert '160 SENDS 0 RETRIES USD10 TOTAL' in expected
        if answer == 'io_error':
            raise cli.TTYConfirmationError('TTY_UNAVAILABLE', OSError(errno.ENXIO, 'PRIVATE'))
        return False
    monkeypatch.setattr(cli, '_confirm_from_tty', confirm)
    monkeypatch.setattr(runner, 'initialize_enabled', lambda *_a, **_k: pytest.fail('refusal enabled'))
    monkeypatch.setattr(runner, 'run', lambda *_a, **_k: pytest.fail('refusal sent'))
    assert cli.main(['launch', '--plan', str(fixture_plan['path'])]) == (5 if answer == 'io_error' else 2)
    assert not Path(fixture_plan['plan']['identity']['authority_path']).exists()
    assert not Path(fixture_plan['plan']['identity']['enabled_directory']).exists()


def test_complete_and_partial_cli_exit_codes_are_distinct(fixture_plan, monkeypatch):
    monkeypatch.setattr(plan, 'verify_parent', lambda *_: None)
    monkeypatch.setattr(cli, '_confirm_from_tty', lambda *_: True)
    monkeypatch.setattr(runner, 'initialize_enabled', lambda *_a, **_k: None)
    monkeypatch.setattr(runner, 'run', lambda *_a, **_k: {})
    for complete, code in [(True, 0), (False, 4)]:
        monkeypatch.setattr(runner, 'evaluate_and_summarize', lambda *_: {'experiment_complete': complete})
        assert cli.launch(fixture_plan['path']) == code


def test_budget_derivation_and_pricing_changes_require_new_approval(fixture_plan):
    pricing = fixture_plan['plan']['identity']['pricing']
    budgets = plan.budgets(pricing)
    assert budgets['max_tokens'] == 53242880
    assert budgets['max_total_tokens'] == 53596968
    assert budgets['max_total_calls'] == 190 and budgets['max_cost_usd'] == 10
    assert budgets['concurrency'] == 1 and budgets['max_new_retries'] == 0
    for field, value in [('context_window_tokens', 1000), ('accepted_response_model_ids', ['other']),
                         ('output_usd_per_million', 99), ('model_max_output_tokens', 65536)]:
        changed = deepcopy(pricing)
        changed['openai'][field] = value
        with pytest.raises(GenerationError, match='scope_reapproval'):
            plan._unchanged_pricing(pricing, changed)


def test_relocated_plan_rejected_before_preview_or_activation(fixture_plan):
    copied = fixture_plan['path'].with_name('copied.json')
    copied.write_bytes(fixture_plan['path'].read_bytes())
    with pytest.raises(GenerationError, match='canonical_plan_path'):
        cli.launch(copied, preview_only=True)
