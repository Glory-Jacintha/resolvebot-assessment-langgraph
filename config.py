"""Settings, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date

try:  # python-dotenv is optional; real environment variables always win
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4o",
    "gemini": "gemini-3.5-flash-lite",
}


@dataclass(frozen=True)
class Settings:
    policies_path: str
    manifest_path: str
    db_path: str
    app_today: date
    llm_provider: str
    llm_model: str
    max_tool_calls: int
    llm_timeout: float
    reply_time_budget: float


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def get_settings() -> Settings:
    """Read settings fresh each call (cheap), so tests can change env vars."""
    today_raw = (os.getenv("APP_TODAY") or "2026-07-01").strip()
    try:
        today = date.fromisoformat(today_raw)
    except ValueError as exc:
        raise RuntimeError(f"APP_TODAY must look like 2026-07-01, got {today_raw!r}") from exc

    provider = (os.getenv("LLM_PROVIDER") or "anthropic").strip().lower()
    model = (os.getenv("LLM_MODEL") or "").strip() or DEFAULT_MODELS.get(provider, "")

    return Settings(
        policies_path=os.getenv("POLICIES_PATH") or "./data/policies.md",
        manifest_path=os.getenv("MANIFEST_PATH") or "./data/manifest.json",
        db_path=os.getenv("DB_PATH") or "./data/shopsphere.db",
        app_today=today,
        llm_provider=provider,
        llm_model=model,
        max_tool_calls=_int("MAX_TOOL_CALLS", 6),
        llm_timeout=_float("LLM_TIMEOUT_SECONDS", 20.0),
        reply_time_budget=_float("REPLY_TIME_BUDGET_SECONDS", 26.0),
    )
