"""Chat models (LangChain) with the five tools attached. Temperature is 0.

Providers: "anthropic", "openai" (also a local OpenAI-compatible server via
OPENAI_BASE_URL) and "gemini" (Google's OpenAI-compatible endpoint).

The agent only needs one method: ``llm.invoke(list_of_messages) -> AIMessage``.
"""
from __future__ import annotations

import os
from typing import Any

from config import get_settings
from tools import TOOL_SPECS

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class _TemperatureFallback:
    """Some newer models reject a temperature setting: retry once without it."""

    def __init__(self, build):
        self._build = build
        self._model = build(True)

    def invoke(self, messages: list[Any]) -> Any:
        try:
            return self._model.invoke(messages)
        except Exception as exc:
            if "temperature" in str(exc).lower():
                self._model = self._build(False)
                return self._model.invoke(messages)
            raise


def create_llm() -> _TemperatureFallback:
    s = get_settings()
    provider = s.llm_provider

    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        from langchain_anthropic import ChatAnthropic

        def build(with_temperature: bool):
            kwargs: dict[str, Any] = dict(
                model=s.llm_model, api_key=key, timeout=s.llm_timeout, max_retries=1, max_tokens=1500
            )
            if with_temperature:
                kwargs["temperature"] = 0
            return ChatAnthropic(**kwargs).bind_tools(_lc_tools())

    elif provider in ("openai", "gemini"):
        from langchain_openai import ChatOpenAI

        if provider == "gemini":
            key, base_url = os.getenv("GOOGLE_API_KEY"), GEMINI_BASE_URL
            if not key:
                raise RuntimeError("GOOGLE_API_KEY is not set")
        else:
            base_url = os.getenv("OPENAI_BASE_URL") or None
            key = os.getenv("OPENAI_API_KEY") or ("not-needed" if base_url else "")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is not set")

        def build(with_temperature: bool):
            kwargs: dict[str, Any] = dict(
                model=s.llm_model, api_key=key, base_url=base_url, timeout=s.llm_timeout, max_retries=1
            )
            if with_temperature:
                kwargs["temperature"] = 0
            return ChatOpenAI(**kwargs).bind_tools(_lc_tools())

    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER {provider!r} (use anthropic, openai or gemini)")

    return _TemperatureFallback(build)


def _lc_tools() -> list[dict[str, Any]]:
    """The tool schemas in OpenAI function format, which LangChain accepts for every provider."""
    return [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
        for t in TOOL_SPECS
    ]
