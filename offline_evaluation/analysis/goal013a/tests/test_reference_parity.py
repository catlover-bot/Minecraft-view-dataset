"""AST extraction mechanics only; these are not real-model parity receipts."""
import hashlib

import pytest

from offline_evaluation.analysis.goal013a import reference_parity as r
from offline_evaluation.analysis.goal013a.scorer import ScorerError


def author_shaped_fixture():
    return '''import deliberately_unavailable_caption_library
raise RuntimeError("unrelated top-level code must never run")
class CLIPCapDataset:
    pass
class CLIPImageDataset:
    pass
def extract_all_captions(*args):
    return sentinel
def extract_all_images(*args):
    return sentinel
def get_clip_score(*args):
    return extract_all_captions()
def main():
    raise RuntimeError("CLI must not run")
'''


def test_extraction_executes_original_functions_and_excludes_top_level_code(tmp_path):
    path = tmp_path/'author_fixture.py'; raw = author_shaped_fixture().encode(); path.write_bytes(raw)
    sentinel = object()
    namespace, metadata = r.extract_reference(path, hashlib.sha256(raw).hexdigest(), {'sentinel': sentinel})
    assert namespace['get_clip_score']() is sentinel
    assert metadata['original_function_bodies_modified'] is False
    assert {d['name'] for d in metadata['definitions']} == set(r.DEFINITIONS)
    assert 'main' not in namespace and metadata['omitted_top_level_imports']


def test_reference_hash_mismatch_never_executes_source(tmp_path):
    path = tmp_path/'source.py'; path.write_text(author_shaped_fixture())
    with pytest.raises(ScorerError, match='hash mismatch'):
        r.extract_reference(path, '0'*64, {})


def test_missing_or_duplicate_reference_definition_is_rejected(tmp_path):
    path = tmp_path/'source.py'
    raw = (author_shaped_fixture() + '\ndef get_clip_score():\n    return None\n').encode()
    path.write_bytes(raw)
    with pytest.raises(ScorerError, match='definition set'):
        r.extract_reference(path, hashlib.sha256(raw).hexdigest(), {})
