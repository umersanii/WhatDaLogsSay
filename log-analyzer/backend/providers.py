"""
Multi-provider AI client abstraction.
Supports Groq, OpenRouter (OpenAI-compatible), and Google Gemini.
"""
import os
from dataclasses import dataclass
from typing import Literal

Provider = Literal["groq", "openrouter", "gemini"]

@dataclass
class ModelInfo:
    id: str
    name: str
    provider: Provider
    supports_tools: bool = True
    context_k: int = 32


MODELS: list[ModelInfo] = [
    # Groq models
    ModelInfo("llama-3.3-70b-versatile",         "Llama 3.3 70B",           "groq",        True,  128),
    ModelInfo("llama-3.1-8b-instant",            "Llama 3.1 8B (fast)",     "groq",        True,  128),
    ModelInfo("mixtral-8x7b-32768",              "Mixtral 8x7B",            "groq",        True,   32),
    ModelInfo("gemma2-9b-it",                    "Gemma 2 9B",              "groq",        False,   8),
    # Google Gemini (native)
    ModelInfo("gemini-2.0-flash",                "Gemini 2.0 Flash",        "gemini",      True,  128),
    ModelInfo("gemini-1.5-pro",                  "Gemini 1.5 Pro",          "gemini",      True,  128),
    ModelInfo("gemini-1.5-flash",                "Gemini 1.5 Flash",        "gemini",      True,  128),
    # OpenRouter models
    ModelInfo("openai/gpt-4o",                   "GPT-4o",                  "openrouter",  True,  128),
    ModelInfo("openai/gpt-4o-mini",              "GPT-4o Mini",             "openrouter",  True,  128),
    ModelInfo("anthropic/claude-3.5-sonnet",     "Claude 3.5 Sonnet",       "openrouter",  True,  200),
    ModelInfo("anthropic/claude-3-haiku",        "Claude 3 Haiku",          "openrouter",  True,  200),
    ModelInfo("google/gemini-flash-1.5",         "Gemini Flash 1.5 (OR)",   "openrouter",  True,  128),
    ModelInfo("meta-llama/llama-3.3-70b-instruct","Llama 3.3 70B (OR)",     "openrouter",  True,  128),
    ModelInfo("mistralai/mistral-large",         "Mistral Large",           "openrouter",  True,  128),
]

DEFAULT_PROVIDER: Provider = "groq"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def get_models_list() -> list[dict]:
    available = []
    groq_key = os.environ.get("GROQ") or os.environ.get("GROQ_API_KEY")
    or_key   = os.environ.get("OPEN_ROUTER_API_KEY")
    gem_key  = os.environ.get("GEMINI_API_KEY")
    for m in MODELS:
        if m.provider == "groq"        and not groq_key: continue
        if m.provider == "openrouter"  and not or_key:   continue
        if m.provider == "gemini"      and not gem_key:  continue
        available.append({
            "id":             m.id,
            "name":           m.name,
            "provider":       m.provider,
            "supports_tools": m.supports_tools,
            "context_k":      m.context_k,
        })
    return available


def make_async_client(provider: Provider):
    """Return an async client for the given provider."""
    if provider == "groq":
        from groq import AsyncGroq
        api_key = os.environ.get("GROQ") or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ or GROQ_API_KEY environment variable must be set.")
        return AsyncGroq(api_key=api_key)

    if provider == "openrouter":
        from openai import AsyncOpenAI
        api_key = os.environ.get("OPEN_ROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPEN_ROUTER_API_KEY environment variable must be set.")
        return AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    if provider == "gemini":
        from openai import AsyncOpenAI
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable must be set.")
        return AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    raise ValueError(f"Unknown provider: {provider}")


def make_sync_client(provider: Provider):
    """Return a sync client for the given provider (used in background thread)."""
    if provider == "groq":
        from groq import Groq
        api_key = os.environ.get("GROQ") or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ or GROQ_API_KEY environment variable must be set.")
        return Groq(api_key=api_key)

    if provider == "openrouter":
        from openai import OpenAI
        api_key = os.environ.get("OPEN_ROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPEN_ROUTER_API_KEY environment variable must be set.")
        return OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    if provider == "gemini":
        from openai import OpenAI
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable must be set.")
        return OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    raise ValueError(f"Unknown provider: {provider}")
