"""
TEI LLM Provider Layer.

Auto-detects available API keys from environment, recommends models,
estimates costs, and provides a unified interface to OpenAI / Anthropic / Google.

Key design decisions (as agreed):
- Evaluation LLM = smartest available (this is the critical judge)
- Improvement LLM = capable but can be cheaper (generates fixes based on diagnosis)
- Auto-detect from env vars (zero friction, zero-friction pattern)
- Minimum model versions enforced:
    OpenAI:    never below GPT-5.1
    Anthropic: never below Claude Sonnet 4.6
    Google:    never below Gemini 3 Flash
- Cost estimate shown before run (user sees what they will pay)
- One-line config change to swap any model
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from .models import LLMConfig


PROVIDER_MODELS: dict[str, dict[str, dict[str, Any]]] = {
    "openai": {
        "eval_recommended": "gpt-5.2",
        "improve_recommended": "gpt-5.1",
        "models": {
            "gpt-5.2": {
                "input_cost_per_1m": 3.00,
                "output_cost_per_1m": 12.00,
                "tier": "smart",
            },
            "gpt-5.1": {
                "input_cost_per_1m": 2.00,
                "output_cost_per_1m": 8.00,
                "tier": "balanced",
            },
            "gpt-5-mini": {
                "input_cost_per_1m": 0.50,
                "output_cost_per_1m": 2.00,
                "tier": "fast",
            },
        },
    },
    "anthropic": {
        "eval_recommended": "claude-opus-4-6",
        "improve_recommended": "claude-sonnet-4-6",
        "models": {
            "claude-opus-4-6": {
                "input_cost_per_1m": 5.00,
                "output_cost_per_1m": 25.00,
                "tier": "smart",
            },
            "claude-sonnet-4-6": {
                "input_cost_per_1m": 3.00,
                "output_cost_per_1m": 15.00,
                "tier": "balanced",
            },
            "claude-haiku-4-5": {
                "input_cost_per_1m": 0.80,
                "output_cost_per_1m": 4.00,
                "tier": "fast",
            },
        },
    },
    "google": {
        "eval_recommended": "gemini-3-pro-preview",
        "improve_recommended": "gemini-3-flash-preview",
        "models": {
            "gemini-3-pro-preview": {
                "input_cost_per_1m": 1.25,
                "output_cost_per_1m": 10.00,
                "tier": "smart",
            },
            "gemini-3-flash-preview": {
                "input_cost_per_1m": 0.15,
                "output_cost_per_1m": 0.60,
                "tier": "fast",
            },
        },
    },
}

ENV_KEY_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def detect_available_providers() -> list[str]:
    """Check which LLM providers have API keys in the environment."""
    available = []
    for provider, env_var in ENV_KEY_MAP.items():
        if os.environ.get(env_var):
            available.append(provider)
    return available


def resolve_provider(config: LLMConfig) -> str:
    """Determine which provider to use. Priority: explicit > openai > anthropic > google."""
    if config.provider != "auto":
        if config.provider not in PROVIDER_MODELS:
            raise ValueError(
                f"Unknown provider '{config.provider}'. "
                f"Supported: {list(PROVIDER_MODELS.keys())}"
            )
        env_var = ENV_KEY_MAP[config.provider]
        if not (config.api_key or os.environ.get(env_var)):
            raise ValueError(
                f"Provider '{config.provider}' selected but no API key found. "
                f"Set {env_var} in your environment or pass api_key in config."
            )
        return config.provider

    available = detect_available_providers()
    if not available:
        raise ValueError(
            "No LLM API keys detected. TEI needs an LLM for evaluation.\n"
            "Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY\n"
            "These are the same keys your agent already uses."
        )
    priority = ["openai", "anthropic", "google"]
    for p in priority:
        if p in available:
            return p
    return available[0]


def resolve_models(config: LLMConfig) -> tuple[str, str]:
    """Return (eval_model, improve_model) based on config and available provider."""
    provider = resolve_provider(config)
    provider_info = PROVIDER_MODELS[provider]

    eval_model = config.eval_model
    if eval_model == "auto":
        eval_model = provider_info["eval_recommended"]

    improve_model = config.improve_model
    if improve_model == "auto":
        improve_model = provider_info["improve_recommended"]

    return eval_model, improve_model


def estimate_cost(
    eval_model: str,
    improve_model: str,
    provider: str,
    num_eval_calls: int = 4,
    avg_input_tokens: int = 2000,
    avg_output_tokens: int = 800,
    num_improve_calls: int = 2,
) -> dict[str, Any]:
    """Estimate the cost of a single TEI run before executing."""
    models = PROVIDER_MODELS.get(provider, {}).get("models", {})

    eval_info = models.get(eval_model, {"input_cost_per_1m": 2.0, "output_cost_per_1m": 10.0})
    improve_info = models.get(improve_model, {"input_cost_per_1m": 0.5, "output_cost_per_1m": 2.0})

    eval_input_cost = (num_eval_calls * avg_input_tokens / 1_000_000) * eval_info["input_cost_per_1m"]
    eval_output_cost = (num_eval_calls * avg_output_tokens / 1_000_000) * eval_info["output_cost_per_1m"]
    eval_total = eval_input_cost + eval_output_cost

    improve_input_cost = (num_improve_calls * avg_input_tokens / 1_000_000) * improve_info["input_cost_per_1m"]
    improve_output_cost = (num_improve_calls * avg_output_tokens / 1_000_000) * improve_info["output_cost_per_1m"]
    improve_total = improve_input_cost + improve_output_cost

    return {
        "provider": provider,
        "eval_model": eval_model,
        "improve_model": improve_model,
        "eval_cost_usd": round(eval_total, 6),
        "improve_cost_usd": round(improve_total, 6),
        "total_estimate_usd": round(eval_total + improve_total, 6),
        "note": "Estimate assumes 1 improvement cycle. Actual cost depends on retries.",
    }


def get_api_key(provider: str, config: LLMConfig) -> str:
    """Get the API key for a provider from config or environment."""
    if config.api_key:
        return config.api_key
    env_var = ENV_KEY_MAP.get(provider, "")
    key = os.environ.get(env_var, "")
    if not key:
        raise ValueError(f"No API key for {provider}. Set {env_var} in your environment.")
    return key


class BaseLLMProvider(ABC):
    """Abstract LLM provider interface."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.1, max_tokens: int = 4096):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...

    @abstractmethod
    async def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        ...

    def get_cost(self, provider: str) -> float:
        models = PROVIDER_MODELS.get(provider, {}).get("models", {})
        info = models.get(self.model, {"input_cost_per_1m": 2.0, "output_cost_per_1m": 10.0})
        input_cost = (self.total_input_tokens / 1_000_000) * info["input_cost_per_1m"]
        output_cost = (self.total_output_tokens / 1_000_000) * info["output_cost_per_1m"]
        return input_cost + output_cost


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""

    def _build_params(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": self.max_tokens,
        }
        if self.temperature != 1.0:
            params["temperature"] = self.temperature
        return params

    _FALLBACK_MODELS = ["gpt-4o", "gpt-4o-mini"]

    async def _call(self, params: dict[str, Any]) -> Any:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.api_key)

        try:
            return await client.chat.completions.create(**params)
        except Exception as e:
            err = str(e)
            if "temperature" in err:
                params.pop("temperature", None)
                return await client.chat.completions.create(**params)
            if "model_not_found" in err or "does not exist" in err:
                for fb in self._FALLBACK_MODELS:
                    if fb != params["model"]:
                        params["model"] = fb
                        try:
                            return await client.chat.completions.create(**params)
                        except Exception:
                            continue
            raise

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("pip install 'tei-loop[openai]' to use OpenAI models")

        params = self._build_params(system_prompt, user_prompt)
        response = await self._call(params)
        usage = response.usage
        if usage:
            self.total_input_tokens += usage.prompt_tokens
            self.total_output_tokens += usage.completion_tokens
        return response.choices[0].message.content or ""

    async def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("pip install 'tei-loop[openai]' to use OpenAI models")

        params = self._build_params(system_prompt, user_prompt)
        params["response_format"] = {"type": "json_object"}
        response = await self._call(params)
        usage = response.usage
        if usage:
            self.total_input_tokens += usage.prompt_tokens
            self.total_output_tokens += usage.completion_tokens
        text = response.choices[0].message.content or "{}"
        return json.loads(text)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider."""

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("pip install 'tei-loop[anthropic]' to use Anthropic models")

        client = AsyncAnthropic(api_key=self.api_key)
        response = await client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.temperature,
        )
        if response.usage:
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
        return response.content[0].text if response.content else ""

    async def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        full_prompt = user_prompt + "\n\nRespond with valid JSON only. No other text."
        text = await self.generate(system_prompt, full_prompt)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        return json.loads(text)


class GoogleProvider(BaseLLMProvider):
    """Google Generative AI provider."""

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("pip install 'tei-loop[google]' to use Google models")

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(
            self.model,
            system_instruction=system_prompt,
        )
        response = await asyncio.to_thread(
            model.generate_content,
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            self.total_input_tokens += getattr(response.usage_metadata, "prompt_token_count", 0)
            self.total_output_tokens += getattr(response.usage_metadata, "candidates_token_count", 0)
        return response.text or ""

    async def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        full_prompt = user_prompt + "\n\nRespond with valid JSON only. No other text."
        text = await self.generate(system_prompt, full_prompt)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        return json.loads(text)


PROVIDER_CLASSES = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
}


def create_provider(
    provider: str,
    model: str,
    config: LLMConfig,
) -> BaseLLMProvider:
    """Create an LLM provider instance."""
    cls = PROVIDER_CLASSES.get(provider)
    if not cls:
        raise ValueError(f"Unknown provider: {provider}")
    api_key = get_api_key(provider, config)
    return cls(
        api_key=api_key,
        model=model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )


def build_providers(config: LLMConfig) -> tuple[str, BaseLLMProvider, BaseLLMProvider]:
    """Build both eval and improve providers from config."""
    provider = resolve_provider(config)
    eval_model, improve_model = resolve_models(config)
    eval_llm = create_provider(provider, eval_model, config)
    improve_llm = create_provider(provider, improve_model, config)
    return provider, eval_llm, improve_llm


TIER_OPTIONS: dict[str, dict[str, dict[str, str]]] = {
    "openai": {
        "fast":     {"eval": "gpt-5-mini",    "improve": "gpt-5-mini"},
        "balanced": {"eval": "gpt-5.2",          "improve": "gpt-5.1"},
        "quality":  {"eval": "gpt-5.2",          "improve": "gpt-5.2"},
    },
    "anthropic": {
        "fast":     {"eval": "claude-sonnet-4-6", "improve": "claude-haiku-4-5"},
        "balanced": {"eval": "claude-opus-4-6",   "improve": "claude-sonnet-4-6"},
        "quality":  {"eval": "claude-opus-4-6",   "improve": "claude-opus-4-6"},
    },
    "google": {
        "fast":     {"eval": "gemini-3-flash-preview", "improve": "gemini-3-flash-preview"},
        "balanced": {"eval": "gemini-3-pro-preview",   "improve": "gemini-3-flash-preview"},
        "quality":  {"eval": "gemini-3-pro-preview",   "improve": "gemini-3-pro-preview"},
    },
}


def print_model_recommendation(config: LLMConfig) -> None:
    """Print model selection with 3 tier options and cost estimates."""
    provider = resolve_provider(config)
    eval_model, improve_model = resolve_models(config)
    cost = estimate_cost(eval_model, improve_model, provider)

    tiers = TIER_OPTIONS.get(provider, {})

    print("\n" + "=" * 60)
    print("TEI Loop  Model Selection")
    print("=" * 60)
    print(f"  Provider: {provider}")

    if tiers:
        for tier_name, tier_label in [("fast", "Fast & Economical"), ("balanced", "Balanced (selected)"), ("quality", "Maximum Quality")]:
            tier = tiers[tier_name]
            tier_cost = estimate_cost(tier["eval"], tier["improve"], provider)
            marker = " <--" if tier_name == "balanced" else ""
            print(f"  {tier_label}:{marker}")
            print(f"    Eval: {tier['eval']}  Improve: {tier['improve']}")
            print(f"    Est. cost/run: ${tier_cost['total_estimate_usd']:.4f}")

    print(f"  Active config:")
    print(f"    Eval model:     {eval_model}")
    print(f"    Improve model:  {improve_model}")
    print(f"    Est. cost/run:  ${cost['total_estimate_usd']:.4f}")
    print("=" * 60 + "\n")
