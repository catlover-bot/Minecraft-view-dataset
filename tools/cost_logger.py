#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def _to_float(v: object) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _default_pricing_per_mtok(provider: str, model: str) -> Tuple[Optional[float], Optional[float], str]:
    p = (provider or "").strip().lower()
    m = (model or "").strip().lower()

    # OpenAI official pricing (used in this project).
    if p == "openai" and "gpt-5-mini" in m:
        return 0.25, 2.0, "default_openai_gpt5mini"

    # Anthropic defaults are heuristic unless overridden by env.
    if p == "anthropic" and "haiku" in m:
        return 0.8, 4.0, "heuristic_anthropic_haiku"
    if p == "anthropic" and "sonnet" in m:
        return 3.0, 15.0, "heuristic_anthropic_sonnet"

    return None, None, "unknown"


def _env_pricing_override(provider: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    p = (provider or "").strip().lower()

    in_common = _to_float(os.environ.get("LLM_PRICE_INPUT_PER_MTOK"))
    out_common = _to_float(os.environ.get("LLM_PRICE_OUTPUT_PER_MTOK"))

    if p == "openai":
        in_p = _to_float(os.environ.get("OPENAI_PRICE_INPUT_PER_MTOK"))
        out_p = _to_float(os.environ.get("OPENAI_PRICE_OUTPUT_PER_MTOK"))
    elif p == "anthropic":
        in_p = _to_float(os.environ.get("ANTHROPIC_PRICE_INPUT_PER_MTOK"))
        out_p = _to_float(os.environ.get("ANTHROPIC_PRICE_OUTPUT_PER_MTOK"))
    else:
        in_p = None
        out_p = None

    in_final = in_p if in_p is not None else in_common
    out_final = out_p if out_p is not None else out_common
    if in_final is None or out_final is None:
        return None, None, None
    return in_final, out_final, "env_override"


def estimate_usage_cost(provider: str, model: str, usage: Dict[str, Any]) -> Dict[str, Any]:
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens))

    in_rate, out_rate, source = _default_pricing_per_mtok(provider, model)
    env_in, env_out, env_source = _env_pricing_override(provider)
    if env_source is not None:
        in_rate, out_rate, source = env_in, env_out, env_source

    if in_rate is None or out_rate is None:
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_price_per_mtok": None,
            "output_price_per_mtok": None,
            "input_cost_usd": None,
            "output_cost_usd": None,
            "estimated_cost_usd": None,
            "pricing_source": source,
        }

    input_cost = (input_tokens / 1_000_000.0) * in_rate
    output_cost = (output_tokens / 1_000_000.0) * out_rate
    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_price_per_mtok": in_rate,
        "output_price_per_mtok": out_rate,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "estimated_cost_usd": total_cost,
        "pricing_source": source,
    }


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def summarize_cost_events(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(events)

    total_calls = len(rows)
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_cost = 0.0
    known_cost_calls = 0

    by_provider: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0, "known_cost_calls": 0})
    by_model: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0, "known_cost_calls": 0})
    by_stage: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0, "known_cost_calls": 0})

    for e in rows:
        usage = e.get("usage", {}) if isinstance(e.get("usage"), dict) else {}
        cost = e.get("cost", {}) if isinstance(e.get("cost"), dict) else {}
        provider = str(e.get("provider", "unknown"))
        model = str(e.get("model", "unknown"))
        stage = str(e.get("stage", "unknown"))

        it = int(usage.get("input_tokens", 0) or 0)
        ot = int(usage.get("output_tokens", 0) or 0)
        tt = int(usage.get("total_tokens", it + ot) or (it + ot))
        ec = cost.get("estimated_cost_usd")
        ec_val = float(ec) if isinstance(ec, (int, float)) else None

        total_input_tokens += it
        total_output_tokens += ot
        total_tokens += tt
        if ec_val is not None:
            total_cost += ec_val
            known_cost_calls += 1

        for bucket in (by_provider[provider], by_model[model], by_stage[stage]):
            bucket["calls"] += 1
            bucket["input_tokens"] += it
            bucket["output_tokens"] += ot
            bucket["total_tokens"] += tt
            if ec_val is not None:
                bucket["estimated_cost_usd"] += ec_val
                bucket["known_cost_calls"] += 1

    return {
        "total_calls": total_calls,
        "known_cost_calls": known_cost_calls,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": total_cost if known_cost_calls > 0 else None,
        "by_provider": by_provider,
        "by_model": by_model,
        "by_stage": by_stage,
    }


def append_cost_event(dataset_root: Path, event: Dict[str, Any]) -> Dict[str, Any]:
    logs_dir = dataset_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "llm_costs.jsonl"
    summary_path = logs_dir / "llm_cost_summary.json"

    event = dict(event)
    event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    summary = summarize_cost_events(_read_jsonl(log_path))
    summary_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "log_path": str(log_path),
        **summary,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_payload
