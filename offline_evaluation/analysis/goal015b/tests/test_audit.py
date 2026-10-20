import json
import pytest

from offline_evaluation.analysis.goal015b.audit import decode, extract, prefix_observations, openai_echo_matches, write_csv


def test_duplicate_keys_rejected_at_any_depth():
    for raw in ['{"output":[],"output":[]}', '{"a":{"x":1,"x":2}}']:
        with pytest.raises(ValueError, match="duplicate"):
            decode(raw)


def test_openai_multiple_blocks_order_and_nontext_recorded():
    raw = json.dumps({"output": [
        {"type": "reasoning"},
        {"type": "message", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "A"},
                     {"type": "refusal", "refusal": "fixture"},
                     {"type": "output_text", "text": "B"}]},
    ]})
    _, text, meta = extract("openai", raw)
    assert text == "AB"
    assert meta["text_block_locations"] == [[1, 0], [1, 2]]
    assert set(meta["envelope_events"]) == {"nontext_output:reasoning", "refusal", "multiple_text_blocks"}


def test_anthropic_missing_text_is_not_empty_program():
    _, text, meta = extract("anthropic", '{"role":"assistant","content":[{"type":"thinking"}]}')
    assert text is None and "missing_text" in meta["envelope_events"]


@pytest.mark.parametrize("raw", [
    '{"output":[{"type":"message","content":null}]}',
    '{"output":[{"type":"message","content":[{"type":"output_text","text":1}]}]}',
    '{"output":[0]}',
])
def test_invalid_envelope_types_rejected(raw):
    with pytest.raises(ValueError):
        extract("openai", raw)


def test_prefix_observation_does_not_repair_or_execute():
    text = '{"operations":[{"op":"set"},{"op":"set"},{"op":'
    out = prefix_observations(text)
    assert out["prefix_operations_observed"] == 2
    assert out["prefix_longest_identical_run"] == 2
    assert out["text_ends_with_operations_close"] is False
    with pytest.raises(ValueError):
        decode(text)


def test_explicit_empty_text_and_no_text_distinguished():
    _, text, meta = extract("anthropic", '{"role":"assistant","content":[{"type":"text","text":""}]}')
    assert text == "" and meta["text_block_count"] == 1


def test_response_format_metadata_is_not_schema_change():
    format_ = {"type": "json_schema", "name": "fixture", "strict": True,
               "schema": {"type": "object"}}
    payload = {"text": {"format": format_}}
    envelope = {"text": {"format": {**format_, "description": None}}}
    result = openai_echo_matches(envelope, payload)
    assert result["raw_echo_format_exact_matches_payload"] is False
    assert result["raw_echo_schema_matches_payload"] is True
    assert result["raw_echo_format_contract_matches_payload"] is True
    envelope["text"]["format"]["schema"] = {"type": "array"}
    assert openai_echo_matches(envelope, payload)["raw_echo_schema_matches_payload"] is False


def test_csv_missing_union_columns_are_explicit_null(tmp_path):
    import csv
    path = tmp_path / "rows.csv"
    write_csv(path, [{"state": "NOT_SENT"}, {"state": "SETTLED", "n": 1}])
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["n"] == "NULL" and rows[1]["n"] == "1"
