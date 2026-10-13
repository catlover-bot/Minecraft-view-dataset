"""Goal004 native structured-output protocol, with no retries or fallback.

Build on the old explicit request, response and accounting primitives without
changing the implementation hashes of the original campaign.
"""
from __future__ import annotations

import http.client
import json
import socket
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Callable

from offline_evaluation.server_v2 import providers
from offline_evaluation.server_v2.core import Limits, ValidationError, strict_json
from .schema import action_schema


MODELS = {"openai": "gpt-5-mini", "anthropic": "claude-haiku-4-5-20251001"}
MAX_OUTPUT_TOKENS = 32768
ProviderError = providers.ProviderError
ProviderResponse = providers.ProviderResponse


def build_payload(provider: str, model: str, prompt: str, input_text: str,
                  max_output_tokens: int = MAX_OUTPUT_TOKENS) -> dict:
    if provider not in MODELS or model != MODELS[provider]:
        raise ProviderError("configuration_failure")
    if not isinstance(prompt, str) or not isinstance(input_text, str):
        raise ProviderError("configuration_failure")
    if type(max_output_tokens) is not int or max_output_tokens != MAX_OUTPUT_TOKENS:
        raise ProviderError("configuration_failure")
    payload = providers.build_payload(provider, model, prompt, input_text, max_output_tokens)
    if provider == "openai":
        payload["text"] = {"format": {"type": "json_schema", "name": "minecraft_actions_v2",
                                      "strict": True, "schema": action_schema()}}
    else:
        payload["output_config"] = {"format": {"type": "json_schema", "schema": action_schema()}}
    return payload


def classify_http_error(status_code: int, raw: bytes = b"") -> str:
    """Retain no server text; distinguish native format/schema rejections."""
    base = providers.classify_http_error(status_code, raw)
    if base != "http_failure" or status_code not in {400, 422}:
        return base
    try:
        error = json.loads(raw.decode("utf-8")).get("error", {})
        if not isinstance(error, dict):
            return base
        code = str(error.get("code") or error.get("type") or "").lower()
        param = str(error.get("param") or "").lower()
        message = str(error.get("message") or "").lower()
        if code in {"invalid_json_schema", "unsupported_schema", "schema_too_complex"} or any(
            word in param + " " + message for word in
            ("json_schema", "json schema", "output_config", "text.format", "structured output")
        ):
            return "schema_unsupported"
    except (ValueError, UnicodeError, AttributeError):
        pass
    return base


def http_transport(request: urllib.request.Request, timeout: float) -> bytes:
    """One POST, original redirect/size policy, plus schema error classification.

This narrow transport extension is necessary because the old transport erases
HTTP bodies before native schema errors can be distinguished. Bodies and keys
are never attached to exceptions or logged here.
"""
    opener = urllib.request.build_opener(providers._NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read(providers.MAX_RESPONSE_BYTES + 1)
    except ProviderError:
        raise
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read(16_384)
        except (OSError, http.client.HTTPException):
            raw = b""
        raise ProviderError(classify_http_error(exc.code, raw), exc.code) from None
    except (TimeoutError, socket.timeout):
        raise ProviderError("timeout") from None
    except urllib.error.URLError as exc:
        raise ProviderError("timeout" if isinstance(exc.reason, (TimeoutError, socket.timeout))
                            else "transport_failure") from None
    except (OSError, ConnectionError, http.client.HTTPException):
        raise ProviderError("transport_failure") from None
    if len(body) > providers.MAX_RESPONSE_BYTES:
        raise ProviderError("response_too_large")
    return body


def _validate_native_payload(provider: str, payload: dict) -> None:
    """Reject format/model changes and chargeable features absent from the plan."""
    if not isinstance(payload, dict) or payload.get("model") != MODELS.get(provider):
        raise ProviderError("configuration_failure")
    if provider == "openai":
        prompt, input_text = payload.get("instructions"), payload.get("input")
    elif provider == "anthropic":
        messages = payload.get("messages")
        if (not isinstance(messages, list) or len(messages) != 1
                or not isinstance(messages[0], dict)):
            raise ProviderError("configuration_failure")
        prompt, input_text = payload.get("system"), messages[0].get("content")
    else:
        raise ProviderError("configuration_failure")
    expected = build_payload(provider, MODELS[provider], prompt, input_text)
    if payload != expected:
        raise ProviderError("configuration_failure")


def send_request(provider: str, payload: dict, *, api_allowed: bool = False,
                 permission: bool = False, timeout: float = 60.0,
                 transport: Callable | None = None) -> bytes:
    if api_allowed is not True or permission is not True:
        raise ProviderError("api_disabled")
    _validate_native_payload(provider, payload)
    return providers.send_request(provider, payload, api_allowed=api_allowed,
                                  permission=permission, timeout=timeout,
                                  transport=transport or http_transport)


def parse_response(provider: str, raw: bytes) -> ProviderResponse:
    try:
        data = strict_json(raw.decode("utf-8"), Limits(max_json_bytes=providers.MAX_RESPONSE_BYTES))
    except (UnicodeError, ValidationError):
        raise ProviderError("response_parse_failure") from None
    response = providers.parse_response(provider, raw)
    if provider == "openai":
        refused = any(block.get("type") == "refusal"
                      for item in data.get("output", []) if item.get("type") == "message"
                      for block in item.get("content", []))
    else:
        refused = data.get("stop_reason") == "refusal"
    if refused:
        return replace(response, complete=False, failure_kind="refusal")
    if response.failure_kind:
        return replace(response, complete=False)
    return response
