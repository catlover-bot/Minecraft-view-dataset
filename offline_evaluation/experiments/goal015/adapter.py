"""Goal015 exact-snapshot, one-request native structured-output adapter.

This module performs no credential lookup or network call on import, payload
construction, parsing or accounting. The runtime must establish its canonical
TTY approval, lock, freshness and durable reservation before enabling send.
Frozen resmoke_v2/schema and server_v2 helpers remain unchanged.

Official contracts (dated server evidence is bound by the plan):
https://developers.openai.com/api/docs/models/gpt-5-mini
https://developers.openai.com/api/docs/guides/reasoning
https://platform.claude.com/docs/en/models/haiku-4-5/overview
https://platform.claude.com/docs/en/api/messages/create
https://platform.claude.com/docs/en/build-with-claude/structured-outputs
"""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, InvalidOperation
import hashlib
import http.client
import json
import math
from pathlib import Path
import socket
from types import MappingProxyType
from typing import Callable
import urllib.error
import urllib.request

from offline_evaluation.resmoke_v2 import adapter as frozen_adapter
from offline_evaluation.resmoke_v2.schema import action_schema, validate_document
from offline_evaluation.server_v2 import providers
from offline_evaluation.server_v2.cohort_v4 import strict_saved_json, _serialize_exact
from offline_evaluation.server_v2.core import Limits, ValidationError, strict_json

MODELS = MappingProxyType({'openai': 'gpt-5-mini-2025-08-07',
                          'anthropic': 'claude-haiku-4-5-20251001'})
MAX_OUTPUT_TOKENS = 32768
MAX_RESPONSE_BYTES = providers.MAX_RESPONSE_BYTES
FIELDS = ('summary', 'building_type', 'shape', 'dimensions_estimate', 'materials',
          'elements', 'rebuild_hints', 'uncertainties')
PROMPT_PATH = Path(__file__).resolve().parents[2]/'resmoke_v2/prompts/builder.txt'
PROMPT_SHA256 = 'ba7a555860628c1ad55ba5b65200ac3cf0d528be86c42f46dbb5bbdd61e6ccac'
ProviderError = providers.ProviderError
ProviderResponse = providers.ProviderResponse


def _configuration(ok):
    if not ok:
        raise ProviderError('configuration_failure')


def canonical_json(value):
    """Exact type-preserving comparison; bool must not compare equal to int."""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ProviderError('configuration_failure') from None


def fixed_prompt():
    raw = PROMPT_PATH.read_bytes()
    _configuration(hashlib.sha256(raw).hexdigest() == PROMPT_SHA256)
    return raw.decode('utf-8')


def validate_input(input_text):
    """Only S1 or F8 human-readable fields; no coercion, stripping or repair."""
    _configuration(isinstance(input_text, str) and bool(input_text))
    try:
        document = strict_saved_json(input_text)
    except (ValidationError, ValueError):
        raise ProviderError('configuration_failure') from None
    _configuration(isinstance(document, dict) and set(document) in ({'summary'}, set(FIELDS)))
    _configuration(isinstance(document['summary'], str) and bool(document['summary'].strip()))
    _configuration(_serialize_exact(document) == input_text)
    return document


def build_payload(provider: str, input_text: str) -> dict:
    _configuration(provider in MODELS)
    validate_input(input_text)
    payload = providers.build_payload(provider, MODELS[provider], fixed_prompt(), input_text, MAX_OUTPUT_TOKENS)
    if provider == 'openai':
        payload['reasoning'] = {'effort': 'medium'}
        payload['text'] = {'format': {'type': 'json_schema', 'name': 'minecraft_actions_v2',
                                      'strict': True, 'schema': action_schema()}}
    else:
        payload['temperature'] = 1.0
        payload['thinking'] = {'type': 'disabled'}
        payload['output_config'] = {'format': {'type': 'json_schema', 'schema': action_schema()}}
    return payload


def validate_payload(provider: str, payload: dict) -> None:
    _configuration(provider in MODELS and isinstance(payload, dict))
    if provider == 'openai':
        input_text = payload.get('input')
    else:
        messages = payload.get('messages')
        _configuration(isinstance(messages, list) and len(messages) == 1 and isinstance(messages[0], dict))
        input_text = messages[0].get('content')
    expected = build_payload(provider, input_text)
    _configuration(canonical_json(payload) == canonical_json(expected))


class TransportError(ProviderError):
    """Safe classification plus private body bytes for durable local recording.

    str/repr contain no body, endpoint, headers or credentials. A received HTTP
    error may still have unknown usage and is never a model-output failure.
    """
    def __init__(self, kind, http_status=None, *, http_response_received=False,
                 raw_response=None, raw_complete=False):
        self.http_status = http_status
        self.http_response_received = http_response_received
        self.raw_response = raw_response
        self.raw_complete = raw_complete
        self.failure_kind = kind
        super().__init__(kind, http_status)


def classify_http_error(status_code: int, raw: bytes = b'') -> str:
    return frozen_adapter.classify_http_error(status_code, raw)


def http_transport(request: urllib.request.Request, timeout: float) -> bytes:
    """One POST; no redirect, retry, SDK, hidden repair or model fallback."""
    opener = urllib.request.build_opener(providers._NoRedirect())
    status, received = None, False
    try:
        with opener.open(request, timeout=timeout) as response:
            status = response.getcode(); received = True
            body = response.read(MAX_RESPONSE_BYTES+1)
    except urllib.error.HTTPError as exc:
        raw, complete = None, False
        try:
            raw = exc.read(MAX_RESPONSE_BYTES+1)
            complete = len(raw) <= MAX_RESPONSE_BYTES
        except (OSError, http.client.HTTPException):
            pass
        raise TransportError(classify_http_error(exc.code, raw or b''), exc.code,
                             http_response_received=True, raw_response=raw,
                             raw_complete=complete) from None
    except ProviderError as exc:
        raise TransportError(exc.kind, exc.status_code,
                             http_response_received=exc.status_code is not None) from None
    except http.client.IncompleteRead as exc:
        raise TransportError('transport_failure', status, http_response_received=received,
                             raw_response=exc.partial, raw_complete=False) from None
    except (TimeoutError, socket.timeout):
        raise TransportError('timeout', status, http_response_received=received) from None
    except urllib.error.URLError as exc:
        kind = 'timeout' if isinstance(exc.reason, (TimeoutError, socket.timeout)) else 'transport_failure'
        raise TransportError(kind, status, http_response_received=received) from None
    except (OSError, ConnectionError, http.client.HTTPException):
        raise TransportError('transport_failure', status, http_response_received=received) from None
    if len(body) > MAX_RESPONSE_BYTES:
        raise TransportError('response_too_large', status, http_response_received=True,
                             raw_response=body, raw_complete=False)
    if type(status) is not int or status != 200:
        raise TransportError('http_failure', status, http_response_received=True,
                             raw_response=body, raw_complete=True)
    return body


def send_request(provider: str, payload: dict, *, api_allowed: bool = False,
                 permission: bool = False, timeout: float = 600.0,
                 transport: Callable | None = None) -> bytes:
    if api_allowed is not True or permission is not True:
        raise ProviderError('api_disabled')
    validate_payload(provider, payload)
    _configuration(type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0)
    # Old helper reads only the selected environment key, after both gates.
    # Copy the validated JSON tree so callers cannot mutate in-flight settings.
    frozen = json.loads(canonical_json(payload))
    return providers.send_request(provider, frozen, api_allowed=True, permission=True,
                                  timeout=timeout, transport=transport or http_transport)


def parse_response(provider: str, raw: bytes) -> ProviderResponse:
    _configuration(provider in MODELS)
    if not isinstance(raw, bytes) or len(raw) > MAX_RESPONSE_BYTES:
        raise ProviderError('response_parse_failure')
    # Reuse duplicate-key rejection, response extraction, refusal/truncation.
    response = frozen_adapter.parse_response(provider, raw)
    if response.model != MODELS[provider]:
        return replace(response, complete=False, failure_kind='response_model_mismatch')
    data = strict_json(raw.decode('utf-8'), Limits(max_json_bytes=MAX_RESPONSE_BYTES))
    if provider == 'openai':
        unexpected = any(item.get('type') not in ('reasoning', 'message') for item in data.get('output', []))
        unexpected |= any(block.get('type') not in ('output_text', 'refusal')
                          for item in data.get('output', []) if item.get('type') == 'message'
                          for block in item.get('content', []))
    else:
        unexpected = any(block.get('type') != 'text' for block in data.get('content', []))
    if unexpected:
        return replace(response, complete=False, failure_kind='unexpected_response_feature')
    if not response.complete and response.failure_kind is None:
        return replace(response, failure_kind='incomplete_response')
    return response


def validate_output(text: str) -> list[dict]:
    """Shape-only protocol validation; semantic limits/execution belong to CPU evaluation."""
    if not isinstance(text, str):
        raise ValidationError('missing output text')
    return validate_document(strict_json(text))


def usage_upper_cost(provider: str, usage: dict | None, pricing: dict) -> tuple[int, int, float]:
    """No cache discount; output usage already includes OpenAI reasoning tokens."""
    _configuration(provider in MODELS and isinstance(pricing, dict))
    try:
        for name in ('input_usd_per_million', 'output_usd_per_million'):
            rate = Decimal(str(pricing[name]))
            _configuration(rate.is_finite() and rate >= 0)
    except (KeyError, InvalidOperation, ValueError):
        raise ProviderError('configuration_failure') from None
    return providers.usage_upper_cost(provider, usage, pricing)


def classify_usage(provider: str, usage: dict | None, pricing: dict) -> dict:
    try:
        inputs, outputs, cost = usage_upper_cost(provider, usage, pricing)
    except ProviderError as exc:
        if exc.kind != 'unknown_usage':
            raise
        return {'usage_known': False, 'input_tokens': None, 'output_tokens': None,
                'cost_usd': None, 'failure_kind': 'unknown_usage'}
    return {'usage_known': True, 'input_tokens': inputs, 'output_tokens': outputs,
            'cost_usd': cost, 'failure_kind': None}
