#!/usr/bin/env python3
from __future__ import annotations

import base64
import ast
import json
import mimetypes
import re
import ssl
import urllib.error
import urllib.parse
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


def _extract_gemini_text(payload: Dict[str, Any]) -> Optional[str]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return None
    candidate_texts: List[str] = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        content = cand.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        chunks: List[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            txt = part.get("text")
            if isinstance(txt, str) and txt.strip():
                chunks.append(txt.strip())
        if chunks:
            candidate_texts.append("\n".join(chunks))
    if candidate_texts:
        # Prefer the richest candidate text if multiple are returned.
        candidate_texts.sort(key=len, reverse=True)
        return candidate_texts[0]
    return None


def _strip_markdown_fence(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    return raw


def _strip_json_comments(text: str) -> str:
    out: List[str] = []
    i = 0
    n = len(text)
    in_str = False
    esc = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            i += 2
            while i < n and text[i] not in "\r\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i = min(n, i + 2)
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _remove_trailing_commas(text: str) -> str:
    prev = text
    while True:
        curr = re.sub(r",(\s*[}\]])", r"\1", prev)
        if curr == prev:
            return curr
        prev = curr


def _to_python_literal_jsonish(text: str) -> str:
    out = text
    out = re.sub(r"\btrue\b", "True", out, flags=re.IGNORECASE)
    out = re.sub(r"\bfalse\b", "False", out, flags=re.IGNORECASE)
    out = re.sub(r"\bnull\b", "None", out, flags=re.IGNORECASE)
    return out


def _try_load_jsonish(text: str) -> Optional[Any]:
    raw = _strip_markdown_fence(text.strip())
    if not raw:
        return None

    candidates = [raw, _strip_json_comments(raw)]
    candidates.append(_remove_trailing_commas(candidates[-1]))
    candidates.append(candidates[-1].replace("“", '"').replace("”", '"').replace("’", "'"))

    seen = set()
    uniq: List[str] = []
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            uniq.append(cand)

    for cand in uniq:
        try:
            return json.loads(cand)
        except Exception:
            continue

    for cand in uniq:
        try:
            obj = ast.literal_eval(_to_python_literal_jsonish(cand))
        except Exception:
            continue
        if isinstance(obj, (dict, list)):
            return obj

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


def _usage_from_gemini_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    usage = payload.get("usageMetadata")
    if not isinstance(usage, dict):
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "raw": {},
        }
    input_tokens = int(usage.get("promptTokenCount", 0) or 0)
    output_tokens = int(usage.get("candidatesTokenCount", 0) or 0)
    total_tokens = int(usage.get("totalTokenCount", input_tokens + output_tokens) or (input_tokens + output_tokens))
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
    llm_seed: Optional[int] = None,
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
    if llm_seed is not None and int(llm_seed) >= 0:
        payload["seed"] = int(llm_seed)
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
    llm_seed: Optional[int] = None,
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
    if llm_seed is not None and int(llm_seed) >= 0:
        payload["seed"] = int(llm_seed)
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
    llm_seed: Optional[int] = None,
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


def _gemini_generate_content(
    cfg: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    image_paths: Sequence[Path],
    temperature: float,
    max_tokens: int,
    llm_seed: Optional[int] = None,
) -> LLMCompletion:
    del llm_seed  # Gemini REST does not currently support deterministic seeding via this path.

    user_parts: List[Dict[str, Any]] = [{"text": user_prompt}]
    for path in image_paths:
        b64 = _read_base64(path)
        media = _guess_media_type(path)
        user_parts.append(
            {
                "inline_data": {
                    "mime_type": media,
                    "data": b64,
                }
            }
        )

    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": user_parts}],
        "generationConfig": {
            "temperature": float(temperature),
            "max_output_tokens": int(max_tokens),
            "response_mime_type": "application/json",
        },
    }
    if system_prompt.strip():
        payload["system_instruction"] = {"parts": [{"text": system_prompt}]}

    model_name = urllib.parse.quote(cfg.gemini_model, safe="")
    api_key = urllib.parse.quote(cfg.gemini_api_key, safe="")
    endpoint = f"/v1beta/models/{cfg.gemini_model}:generateContent"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = _http_post_json(url, headers=headers, payload=payload)
    text = _extract_gemini_text(data)
    if not text:
        raise LLMError("Gemini generateContent endpoint returned no text output.")
    return LLMCompletion(
        text=text,
        provider="gemini",
        model=cfg.gemini_model,
        endpoint=endpoint,
        usage=_usage_from_gemini_payload(data),
        raw_response=data,
    )


def complete_multimodal_with_meta(
    cfg: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    image_paths: Optional[Sequence[Path]] = None,
    temperature: float = 0.2,
    max_tokens: int = 1600,
    llm_seed: Optional[int] = None,
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
                    llm_seed=llm_seed,
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
            llm_seed=llm_seed,
        )

    if provider == "gemini":
        return _gemini_generate_content(
            cfg=cfg,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_paths=images,
            temperature=temperature,
            max_tokens=max_tokens,
            llm_seed=llm_seed,
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
    llm_seed: Optional[int] = None,
) -> str:
    res = complete_multimodal_with_meta(
        cfg=cfg,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_paths=image_paths,
        temperature=temperature,
        max_tokens=max_tokens,
        llm_seed=llm_seed,
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

    relaxed = _try_load_jsonish(src)
    if isinstance(relaxed, dict):
        return relaxed

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
                        loaded = _try_load_jsonish(cand)
                        if isinstance(loaded, dict):
                            return loaded
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = src.find("{", start + 1)

    raise LLMError("No valid JSON object found in model response.")
