"""Tests for TEI LLM provider layer."""

import os
import pytest
from tei_loop.llm_provider import (
    PROVIDER_MODELS,
    detect_available_providers,
    estimate_cost,
    resolve_models,
    resolve_provider,
)
from tei_loop.models import LLMConfig


def test_provider_models_structure():
    assert "openai" in PROVIDER_MODELS
    assert "anthropic" in PROVIDER_MODELS
    assert "google" in PROVIDER_MODELS
    for provider in PROVIDER_MODELS:
        assert "eval_recommended" in PROVIDER_MODELS[provider]
        assert "improve_recommended" in PROVIDER_MODELS[provider]
        assert "models" in PROVIDER_MODELS[provider]


def test_detect_no_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert detect_available_providers() == []


def test_detect_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    providers = detect_available_providers()
    assert "openai" in providers


def test_resolve_provider_explicit(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    config = LLMConfig(provider="anthropic")
    assert resolve_provider(config) == "anthropic"


def test_resolve_provider_auto_prefers_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    config = LLMConfig(provider="auto")
    assert resolve_provider(config) == "openai"


def test_resolve_provider_no_keys_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    config = LLMConfig(provider="auto")
    with pytest.raises(ValueError, match="No LLM API keys detected"):
        resolve_provider(config)


def test_resolve_models_auto(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = LLMConfig(provider="auto", eval_model="auto", improve_model="auto")
    eval_m, improve_m = resolve_models(config)
    assert eval_m == "gpt-5.2"
    assert improve_m == "gpt-5.1"


def test_resolve_models_explicit(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = LLMConfig(provider="openai", eval_model="gpt-5.2", improve_model="gpt-5-mini")
    eval_m, improve_m = resolve_models(config)
    assert eval_m == "gpt-5.2"
    assert improve_m == "gpt-5-mini"


def test_estimate_cost():
    cost = estimate_cost("gpt-5.2", "gpt-5-mini", "openai")
    assert cost["provider"] == "openai"
    assert cost["eval_model"] == "gpt-5.2"
    assert cost["total_estimate_usd"] > 0
    assert cost["eval_cost_usd"] > cost["improve_cost_usd"]


def test_estimate_cost_google():
    cost = estimate_cost(
        "gemini-3-pro-preview", "gemini-3-flash-preview", "google"
    )
    assert cost["provider"] == "google"
    assert cost["total_estimate_usd"] > 0
