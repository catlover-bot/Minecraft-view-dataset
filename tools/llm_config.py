#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class LLMConfig:
    provider: str
    openai_api_key: str
    anthropic_api_key: str
    gemini_api_key: str
    openai_model: str
    anthropic_model: str
    gemini_model: str


def _parse_dotenv(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.is_file():
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def load_llm_config(dotenv_path: Optional[str] = None) -> LLMConfig:
    env_path = Path(dotenv_path).expanduser().resolve() if dotenv_path else Path(".env").resolve()
    dotenv_values = _parse_dotenv(env_path)

    def pick(name: str, default: str = "") -> str:
        return os.environ.get(name, dotenv_values.get(name, default))

    return LLMConfig(
        provider=pick("LLM_PROVIDER", "openai"),
        openai_api_key=pick("OPENAI_API_KEY"),
        anthropic_api_key=pick("ANTHROPIC_API_KEY"),
        gemini_api_key=pick("GEMINI_API_KEY"),
        openai_model=pick("OPENAI_MODEL", "gpt-5-mini"),
        anthropic_model=pick("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        gemini_model=pick("GEMINI_MODEL", "gemini-3.1-pro-preview"),
    )


def require_provider_key(config: LLMConfig) -> None:
    provider = config.provider.lower().strip()
    if provider == "openai" and not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is empty. Set it in .env or environment variables.")
    if provider == "anthropic" and not config.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is empty. Set it in .env or environment variables.")
    if provider == "gemini" and not config.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is empty. Set it in .env or environment variables.")
    if provider not in {"openai", "anthropic", "gemini", "mock"}:
        raise RuntimeError("LLM_PROVIDER must be one of: openai, anthropic, gemini, mock")


def model_for_provider(config: LLMConfig, provider: Optional[str] = None) -> str:
    p = (provider or config.provider or "").strip().lower()
    if p == "openai":
        return config.openai_model or "openai_model"
    if p == "anthropic":
        return config.anthropic_model or "anthropic_model"
    if p == "gemini":
        return config.gemini_model or "gemini_model"
    if p == "mock":
        return "mock-model"
    return "unknown-model"


if __name__ == "__main__":
    cfg = load_llm_config()
    require_provider_key(cfg)
    print(
        f"provider={cfg.provider} "
        f"openai_model={cfg.openai_model} "
        f"anthropic_model={cfg.anthropic_model} "
        f"gemini_model={cfg.gemini_model}"
    )
