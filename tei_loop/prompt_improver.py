"""
TEI Prompt Improver.

The core prompt improvement engine:
1. Extract prompts from agent source code
2. Evaluate agent output with original prompts
3. Generate improved prompts based on evaluation failures
4. Patch the agent to use improved prompts
5. Re-run agent, compare before/after

This is REAL improvement — modifying the agent's prompts,
not rewriting the output with a different LLM.
"""

from __future__ import annotations

import re
import copy
import inspect
import textwrap
from typing import Any, Callable, Optional
from pathlib import Path

from .llm_provider import BaseLLMProvider
from .models import EvalResult, Failure


def extract_prompts(agent_fn: Callable, agent_file: Optional[str] = None) -> dict[str, str]:
    """Extract system and user prompt templates from agent source code."""
    source = ""
    if agent_file:
        try:
            source = Path(agent_file).read_text()
        except Exception:
            pass
    if not source:
        try:
            source = inspect.getsource(agent_fn)
        except (OSError, TypeError):
            pass

    if not source:
        return {}

    prompts = {}

    # Try triple-quoted variable assignments first (e.g. SYSTEM_PROMPT = """...""")
    var_patterns = [
        r'(?:SYSTEM_PROMPT|system_prompt|SYS_PROMPT)\s*=\s*"""(.*?)"""',
        r"(?:SYSTEM_PROMPT|system_prompt|SYS_PROMPT)\s*=\s*'''(.*?)'''",
        r'(?:SYSTEM_PROMPT|system_prompt|SYS_PROMPT)\s*=\s*"([^"]+)"',
        r"(?:SYSTEM_PROMPT|system_prompt|SYS_PROMPT)\s*=\s*'([^']+)'",
    ]
    for pat in var_patterns:
        match = re.search(pat, source, re.DOTALL)
        if match:
            prompts["system_prompt"] = match.group(1).strip()
            break

    if "system_prompt" not in prompts:
        sys_matches = re.findall(
            r'"role":\s*"system",\s*"content":\s*"([^"]+)"', source
        )
        if sys_matches:
            prompts["system_prompt"] = sys_matches[-1]

    # Also check for variable reference in messages (e.g. "content": SYSTEM_PROMPT)
    if "system_prompt" not in prompts:
        var_ref = re.search(
            r'"role":\s*"system",\s*"content":\s*(\w+)', source
        )
        if var_ref:
            var_name = var_ref.group(1)
            for pat in [
                rf'{var_name}\s*=\s*"""(.*?)"""',
                rf"{var_name}\s*=\s*'''(.*?)'''",
                rf'{var_name}\s*=\s*"([^"]+)"',
            ]:
                m = re.search(pat, source, re.DOTALL)
                if m:
                    prompts["system_prompt"] = m.group(1).strip()
                    break

    user_var = re.search(
        r'(summary_prompt|user_prompt|prompt|instruction)\s*=\s*\((.*?)\)',
        source, re.DOTALL
    )
    if user_var:
        raw = user_var.group(2)
        clean = re.sub(r'f?"([^"]*)"', r'\1', raw)
        clean = clean.replace('\\n', '\n').replace('\n            ', '\n')
        clean = re.sub(r'\{[^}]+\}', '{input}', clean)
        prompts["user_prompt_template"] = clean.strip()
        prompts["_user_var_name"] = user_var.group(1)

    if not user_var:
        content_matches = re.findall(
            r'"role":\s*"user",\s*"content":\s*(?:f?)?"([^"]{20,})"', source
        )
        if content_matches:
            prompts["user_prompt_template"] = content_matches[-1]

    chat_block = re.search(r'chat\.completions\.create\(.*?\)', source, re.DOTALL)
    if chat_block:
        model_in_chat = re.search(r'model\s*=\s*"([^"]+)"', chat_block.group(0))
        if model_in_chat:
            prompts["model"] = model_in_chat.group(1)
    if "model" not in prompts:
        for m in re.finditer(r'model\s*=\s*"([^"]+)"', source):
            if m.group(1) not in ("whisper-1", "tts-1", "tts-1-hd"):
                prompts["model"] = m.group(1)
                break

    temp_match = re.search(r'temperature\s*=\s*([\d.]+)', source)
    if temp_match:
        prompts["temperature"] = float(temp_match.group(1))

    return prompts


async def generate_improved_prompts(
    original_prompts: dict[str, str],
    eval_result: EvalResult,
    failures: list[Failure],
    improve_llm: BaseLLMProvider,
) -> dict[str, str]:
    """Generate improved versions of the agent's prompts based on evaluation."""
    failure_details = []
    for f in failures[:4]:
        failure_details.append(
            f"- {f.dimension.value}: {f.description[:300]}"
        )

    dim_scores = []
    for dim, ds in eval_result.dimension_scores.items():
        dim_scores.append(f"  {dim.value}: {ds.score:.2f} - {ds.reasoning[:150]}")

    prompt = (
        "You are a prompt engineer. An AI agent uses the prompts below and scored "
        f"{eval_result.aggregate_score:.2f}/1.00 on a structured evaluation.\n\n"
        "CURRENT SYSTEM PROMPT:\n"
        f'"{original_prompts.get("system_prompt", "none")}"\n\n'
        "CURRENT USER PROMPT TEMPLATE:\n"
        f'"{original_prompts.get("user_prompt_template", "none")}"\n\n'
        "EVALUATION SCORES:\n" + "\n".join(dim_scores) + "\n\n"
        "SPECIFIC WEAKNESSES:\n" + "\n".join(failure_details) + "\n\n"
        "Generate IMPROVED versions of both prompts that will fix these weaknesses. "
        "The improved prompts should:\n"
        "1. Be more specific about what to include\n"
        "2. Add constraints that prevent the identified failures\n"
        "3. Request evidence-based claims only\n"
        "4. Maintain the same general purpose\n\n"
        "Return JSON:\n"
        '{\n'
        '  "improved_system_prompt": "the new system prompt",\n'
        '  "improved_user_prompt_template": "the new user prompt template (use {input} for the data placeholder)",\n'
        '  "changes_made": ["list of specific changes and why"]\n'
        '}'
    )

    try:
        result = await improve_llm.generate_json(
            "You are an expert prompt engineer. Return valid JSON only.",
            prompt,
        )
        return {
            "system_prompt": result.get("improved_system_prompt", original_prompts.get("system_prompt", "")),
            "user_prompt_template": result.get("improved_user_prompt_template", original_prompts.get("user_prompt_template", "")),
            "changes_made": result.get("changes_made", []),
        }
    except Exception as e:
        print(f"  Prompt improvement failed: {e}")
        return {}


def _detect_provider(fn: Callable, agent_file: Optional[str] = None) -> str:
    """Detect which LLM provider the agent uses by inspecting source code."""
    source = ""
    if agent_file:
        try:
            source = Path(agent_file).read_text()
        except Exception:
            pass
    if not source:
        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError):
            pass
    lower = source.lower()
    if "anthropic" in lower or "messages.create" in lower:
        return "anthropic"
    if "google" in lower or "generativeai" in lower or "generate_content" in lower:
        return "google"
    return "openai"


def create_patched_agent(
    original_fn: Callable,
    original_prompts: dict[str, str],
    improved_prompts: dict[str, str],
    agent_file: Optional[str] = None,
) -> Callable:
    """Create a patched version of the agent that uses improved prompts."""
    new_sys = improved_prompts.get("system_prompt", "")
    new_user_template = improved_prompts.get("user_prompt_template", "")
    provider = _detect_provider(original_fn, agent_file)
    model = original_prompts.get("model", "")
    temp = original_prompts.get("temperature", 0.3)

    def _build_user_content(query: str) -> str:
        if new_user_template and "{input}" in new_user_template:
            return new_user_template.replace("{input}", str(query))
        elif new_user_template:
            return new_user_template + "\n\n" + str(query)
        return str(query)

    def patched_agent(query: str) -> str:
        import os
        user_content = _build_user_content(query)

        if provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError:
                return original_fn(query)
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            mdl = model or "claude-sonnet-4-20250514"
            try:
                response = client.messages.create(
                    model=mdl,
                    max_tokens=1500,
                    system=new_sys or "You are a helpful assistant.",
                    messages=[{"role": "user", "content": user_content}],
                )
                return response.content[0].text if response.content else ""
            except Exception as e:
                return f"Error: {e}"

        try:
            from openai import OpenAI
        except ImportError:
            return original_fn(query)
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        mdl = model or "gpt-4o-mini"
        try:
            response = client.chat.completions.create(
                model=mdl,
                messages=[
                    {"role": "system", "content": new_sys or "You are a helpful assistant."},
                    {"role": "user", "content": user_content},
                ],
                temperature=temp,
                max_tokens=1500,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"Error: {e}"

    return patched_agent
