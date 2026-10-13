"""Explicit, single-request HTTP adapters. No SDK, hidden retries, or model fallback.

Official API contracts checked 2026-10-13 (documentation reads, no generation):
https://developers.openai.com/api/reference/resources/responses/methods/create
https://developers.openai.com/api/docs/guides/text
https://platform.claude.com/docs/en/api/messages/create
https://platform.claude.com/docs/en/about-claude/pricing

Pricing is deliberately NOT embedded: a future operator must supply a dated,
model-specific verified pricing/context-window record before generation.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from typing import Callable


MAX_RESPONSE_BYTES = 16_000_000
ENDPOINTS = {
    "openai": "https://api.openai.com/v1/responses",
    "anthropic": "https://api.anthropic.com/v1/messages",
}
KEY_NAMES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


class ProviderError(RuntimeError):
    """Safe error classification: never contains headers, keys, or server bodies."""

    def __init__(self, kind: str, status_code: int | None = None):
        self.kind = kind
        self.status_code = status_code
        super().__init__(kind if status_code is None else f"{kind} (HTTP {status_code})")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward credentials to a redirected host or retry a POST.
        raise ProviderError("http_failure", code)


def http_transport(request: urllib.request.Request, timeout: float) -> bytes:
    """Perform exactly one POST. urllib has no automatic retry loop."""
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except ProviderError:
        raise
    except urllib.error.HTTPError as exc:
        raise ProviderError("rate_limit" if exc.code == 429 else "http_failure", exc.code) from None
    except (TimeoutError, socket.timeout):
        raise ProviderError("timeout") from None
    except urllib.error.URLError as exc:
        kind = "timeout" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "transport_failure"
        raise ProviderError(kind) from None
    except (OSError, ConnectionError):
        raise ProviderError("transport_failure") from None
    if len(body) > MAX_RESPONSE_BYTES:
        raise ProviderError("response_too_large")
    return body


def build_payload(provider: str, model: str, prompt: str, input_text: str, max_output_tokens: int) -> dict:
    if provider not in ENDPOINTS:
        raise ValueError("provider must be openai or anthropic")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("explicit model is required")
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    if provider == "openai":
        return {
            "model": model,
            "instructions": prompt,
            "input": input_text,
            "max_output_tokens": max_output_tokens,
            "stream": False,
            "store": False,
            "truncation": "disabled",
            "service_tier": "default",
        }
    return {
        "model": model,
        "system": prompt,
        "messages": [{"role": "user", "content": input_text}],
        "max_tokens": max_output_tokens,
        "stream": False,
        "service_tier": "standard_only",
    }


def send_request(
    provider: str,
    payload: dict,
    *,
    api_allowed: bool = False,
    permission: bool = False,
    timeout: float = 60.0,
    transport: Callable | None = None,
) -> bytes:
    """Both durable configuration and an explicit invocation must authorize POST."""
    if api_allowed is not True or permission is not True:
        raise ProviderError("api_disabled")
    if provider not in ENDPOINTS:
        raise ProviderError("configuration_failure")
    # Only the selected key is read, after the gate. Its value is never persisted.
    key = os.environ.get(KEY_NAMES[provider])
    if not key:
        raise ProviderError("missing_api_key")
    headers = {"Content-Type": "application/json"}
    if provider == "openai":
        headers["Authorization"] = f"Bearer {key}"
    else:
        headers["x-api-key"] = key
        headers["anthropic-version"] = "2023-06-01"
    request = urllib.request.Request(
        ENDPOINTS[provider],
        data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    # An injected transport is intended only for fixture_only unit tests.
    return (transport or http_transport)(request, timeout)


@dataclass(frozen=True)
class ProviderResponse:
    text: str | None
    response_id: str | None
    model: str | None
    usage: dict | None
    complete: bool


def cost_ceiling(input_tokens: int, output_tokens: int, pricing: dict) -> float:
    """Round each persisted USD upper bound upward to one nanodollar."""
    cost = (
        Decimal(input_tokens) * Decimal(str(pricing["input_usd_per_million"]))
        + Decimal(output_tokens) * Decimal(str(pricing["output_usd_per_million"]))
    ) / Decimal(1_000_000)
    return float(cost.quantize(Decimal("0.000000001"), rounding=ROUND_CEILING))


def parse_response(provider: str, raw: bytes) -> ProviderResponse:
    try:
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError
        content = data.get("output", []) if provider == "openai" else data.get("content", [])
        if provider == "openai":
            parts = [
                block["text"]
                for item in content if item.get("type") == "message"
                for block in item.get("content", []) if block.get("type") == "output_text"
            ]
            complete = data.get("status") == "completed"
        elif provider == "anthropic":
            parts = [block["text"] for block in content if block.get("type") == "text"]
            complete = data.get("stop_reason") == "end_turn"
        else:
            raise ValueError
        if any(not isinstance(part, str) for part in parts):
            raise ValueError
        usage = data.get("usage")
        response_id, model = data.get("id"), data.get("model")
        return ProviderResponse(
            "".join(parts) if parts else None,
            response_id if isinstance(response_id, str) else None,
            model if isinstance(model, str) else None,
            usage if isinstance(usage, dict) else None,
            complete,
        )
    except (UnicodeError, ValueError, KeyError, TypeError, AttributeError):
        raise ProviderError("response_parse_failure") from None


def usage_upper_cost(provider: str, usage: dict | None, pricing: dict) -> tuple[int, int, float]:
    """Known billable token usage and conservative USD, with no cached discount.

Plain text requests do not enable prompt cache writes. Unexpected cache writes
or charged features stop billing reconciliation instead of silently underpricing.
Input cache reads are conservatively charged at the normal input rate.
"""
    if not isinstance(usage, dict):
        raise ProviderError("unknown_usage")
    try:
        input_tokens, output_tokens = usage["input_tokens"], usage["output_tokens"]
        if any(type(value) is not int or value < 0 for value in (input_tokens, output_tokens)):
            raise ValueError
        if provider == "anthropic":
            for name in ("cache_read_input_tokens", "cache_creation_input_tokens"):
                value = usage.get(name, 0)
                if type(value) is not int or value < 0:
                    raise ValueError
                if name == "cache_creation_input_tokens" and value:
                    raise ValueError
                input_tokens += value
            if usage.get("server_tool_use"):
                raise ValueError
        elif provider == "openai":
            if usage.get("input_tokens_details", {}).get("cache_write_tokens", 0):
                raise ValueError
        else:
            raise ValueError
        cost = cost_ceiling(input_tokens, output_tokens, pricing)
        return input_tokens, output_tokens, cost
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ProviderError("unknown_usage") from None
