"""Resolves the configured VLM/LLM provider name (Settings) to a concrete instance.

Add a new backend by implementing base.VLMProvider/LLMProvider and
registering it in the maps below — callers never change.
"""
from __future__ import annotations

from common.config import get_settings

from .base import LLMProvider, VLMProvider
from .stub import StubLLMProvider, StubVLMProvider

_VLM_PROVIDERS: dict[str, type[VLMProvider]] = {
    "stub": StubVLMProvider,
}

_LLM_PROVIDERS: dict[str, type[LLMProvider]] = {
    "stub": StubLLMProvider,
}


def get_vlm_provider() -> VLMProvider:
    settings = get_settings()
    try:
        provider_cls = _VLM_PROVIDERS[settings.vlm_provider]
    except KeyError:
        raise ValueError(f"Unknown VLM provider: {settings.vlm_provider!r}") from None
    return provider_cls()


def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    try:
        provider_cls = _LLM_PROVIDERS[settings.llm_provider]
    except KeyError:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider!r}") from None
    return provider_cls()
