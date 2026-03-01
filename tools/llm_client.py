#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import mimetypes
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tools.llm_config import LLMConfig


class LLMError(RuntimeError):
    pass


@dataclass
class LLMCompletion:
    text: str
    provider: str
    model: str
    endpoint: str
    usage: Dict[str, Any]
    raw_response: Dict[str, Any]


def _openai_supports_custom_temperature(model: str) -> bool:
    m = (model or "").strip().lower()
    return not m.startswith("gpt-5")


def _is_openai_gpt5_family(model: str) -> bool:
    return (model or "").strip().lower().startswith("gpt-5")


def _read_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _guess_media_type(path: Path) -> str:
    media, _ = mimetypes.guess_type(str(path))
    if media:
        return media
    return "image/png"


def _http_post_json(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_sec: float = 180.0,
) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec, context=ctx) as resp:
            text = resp.read().decode("utf-8")
            return json.loads(text)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)
        raise LLMError(f"HTTP {exc.code} for {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Network error for {url}: {exc}") from exc


def _extract_openai_responses_text(payload: Dict[str, Any]) -> Optional[str]:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = payload.get("output")
    if not isinstance(output, list):
        return None

    chunks: List[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"output_text", "text"}:
                txt = part.get("text")
                if isinstance(txt, str) and txt.strip():
                    chunks.append(txt.strip())
    if chunks:
        return "\n".join(chunks)
    return None


def _extract_openai_chat_text(payload: Dict[str, Any]) -> Optional[str]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            txt = part.get("text")
            if isinstance(txt, str) and txt.strip():
                chunks.append(txt.strip())
        if chunks:
            return "\n".join(chunks)
    return None


def _extract_anthropic_text(payload: Dict[str, Any]) -> Optional[str]:
    content = payload.get("content")
    if not isinstance(content, list):
        return None
    chunks: List[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            txt = part.get("text")
            if isinstance(txt, str) and txt.strip():
                chunks.append(txt.strip())
    if chunks:
        return "\n".join(chunks)
    return None


def _usage_from_openai_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "raw": {},
        }

    input_tokens = usage.get("input_tokens")
    if input_tokens is None:
        input_tokens = usage.get("prompt_tokens", 0)

    output_tokens = usage.get("output_tokens")
    if output_tokens is None:
        output_tokens = usage.get("completion_tokens", 0)

    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)

    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(total_tokens or 0),
        "raw": usage,
    }


def _usage_from_anthropic_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "raw": {},
        }
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "raw": usage,
    }


def _openai_v1_responses(
    cfg: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    image_paths: Sequence[Path],
    temperature: float,
    max_tokens: int,
) -> LLMCompletion:
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
    for path in image_paths:
        b64 = _read_base64(path)
        media = _guess_media_type(path)
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{media};base64,{b64}",
            }
        )

    payload = {
        "model": cfg.openai_model,
        "max_output_tokens": int(max_tokens),
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ],
    }
    if _openai_supports_custom_temperature(cfg.openai_model):
        payload["temperature"] = float(temperature)
    if _is_openai_gpt5_family(cfg.openai_model):
        payload["reasoning"] = {"effort": "minimal"}
    headers = {
        "Authorization": f"Bearer {cfg.openai_api_key}",
        "Content-Type": "application/json",
    }
    data = _http_post_json("https://api.openai.com/v1/responses", headers=headers, payload=payload)
    text = _extract_openai_responses_text(data)
    if not text:
        raise LLMError("OpenAI responses endpoint returned no text output.")
    return LLMCompletion(
        text=text,
        provider="openai",
        model=cfg.openai_model,
        endpoint="/v1/responses",
        usage=_usage_from_openai_payload(data),
        raw_response=data,
    )


def _openai_chat_completions(
    cfg: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    image_paths: Sequence[Path],
    temperature: float,
    max_tokens: int,
) -> LLMCompletion:
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for path in image_paths:
        b64 = _read_base64(path)
        media = _guess_media_type(path)
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media};base64,{b64}",
                },
            }
        )

    payload = {
        "model": cfg.openai_model,
        "max_completion_tokens": int(max_tokens),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    if _openai_supports_custom_temperature(cfg.openai_model):
        payload["temperature"] = float(temperature)
    headers = {
        "Authorization": f"Bearer {cfg.openai_api_key}",
        "Content-Type": "application/json",
    }
    data = _http_post_json("https://api.openai.com/v1/chat/completions", headers=headers, payload=payload)
    text = _extract_openai_chat_text(data)
    if not text:
        raise LLMError("OpenAI chat completions endpoint returned no text output.")
    return LLMCompletion(
        text=text,
        provider="openai",
        model=cfg.openai_model,
        endpoint="/v1/chat/completions",
        usage=_usage_from_openai_payload(data),
        raw_response=data,
    )


def _anthropic_messages(
    cfg: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    image_paths: Sequence[Path],
    temperature: float,
    max_tokens: int,
) -> LLMCompletion:
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for path in image_paths:
        b64 = _read_base64(path)
        media = _guess_media_type(path)
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media,
                    "data": b64,
                },
            }
        )

    payload = {
        "model": cfg.anthropic_model,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "system": system_prompt,
        "messages": [{"role": "user", "content": content}],
    }
    headers = {
        "x-api-key": cfg.anthropic_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    data = _http_post_json("https://api.anthropic.com/v1/messages", headers=headers, payload=payload)
    text = _extract_anthropic_text(data)
    if not text:
        raise LLMError("Anthropic messages endpoint returned no text output.")
    return LLMCompletion(
        text=text,
        provider="anthropic",
        model=cfg.anthropic_model,
        endpoint="/v1/messages",
        usage=_usage_from_anthropic_payload(data),
        raw_response=data,
    )


def complete_multimodal_with_meta(
    cfg: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    image_paths: Optional[Sequence[Path]] = None,
    temperature: float = 0.2,
    max_tokens: int = 1600,
) -> LLMCompletion:
    provider = (cfg.provider or "").strip().lower()
    images = list(image_paths or [])

    if provider == "openai":
        last_exc: Optional[Exception] = None
        for fn in (_openai_v1_responses, _openai_chat_completions):
            try:
                return fn(
                    cfg=cfg,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_paths=images,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue
        raise LLMError(f"OpenAI request failed: {last_exc}")

    if provider == "anthropic":
        return _anthropic_messages(
            cfg=cfg,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_paths=images,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "mock":
        return LLMCompletion(
            text="{\"summary\": \"mock response\"}",
            provider="mock",
            model="mock-model",
            endpoint="mock",
            usage={
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "raw": {},
            },
            raw_response={},
        )

    raise LLMError(f"Unsupported provider: {provider}")


def complete_multimodal(
    cfg: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    image_paths: Optional[Sequence[Path]] = None,
    temperature: float = 0.2,
    max_tokens: int = 1600,
) -> str:
    res = complete_multimodal_with_meta(
        cfg=cfg,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_paths=image_paths,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return res.text


def extract_json_object(text: str) -> Dict[str, Any]:
    src = text.strip()
    try:
        obj = json.loads(src)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = src.find("{")
    while start >= 0:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(src)):
            ch = src[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = src[start : i + 1]
                    try:
                        obj = json.loads(cand)
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = src.find("{", start + 1)

    raise LLMError("No valid JSON object found in model response.")
