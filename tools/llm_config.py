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
    openai_model: str
    anthropic_model: str


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
        openai_model=pick("OPENAI_MODEL", "gpt-5-mini"),
        anthropic_model=pick("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
    )


def require_provider_key(config: LLMConfig) -> None:
    provider = config.provider.lower().strip()
    if provider == "openai" and not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is empty. Set it in .env or environment variables.")
    if provider == "anthropic" and not config.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is empty. Set it in .env or environment variables.")
    if provider not in {"openai", "anthropic"}:
        raise RuntimeError("LLM_PROVIDER must be one of: openai, anthropic")


if __name__ == "__main__":
    cfg = load_llm_config()
    require_provider_key(cfg)
    print(
        f"provider={cfg.provider} "
        f"openai_model={cfg.openai_model} "
        f"anthropic_model={cfg.anthropic_model}"
    )
