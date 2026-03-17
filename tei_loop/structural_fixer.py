"""
Proposes and applies structural fixes to agent code based on TEI evaluation results.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import Checkpoint, CheckpointResult, EvalResult, StructuralFix
from .llm_provider import BaseLLMProvider


def _read_code_context(file_path: str, line_number: int, context_lines: int = 10) -> str:
    """Read source code around a specific line number."""
    path = Path(file_path)
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    if not lines:
        return ""
    start = max(0, line_number - context_lines - 1)
    end = min(len(lines), line_number + context_lines)
    snippet_lines = lines[start:end]
    return "\n".join(f"{i + 1:4d}| {ln}" for i, ln in enumerate(snippet_lines, start=start + 1))


class StructuralFixer:
    def __init__(self, improve_llm: BaseLLMProvider):
        self.improve_llm = improve_llm

    async def propose_fixes(
        self,
        checkpoints: list[Checkpoint],
        checkpoint_results: list[CheckpointResult],
        eval_result: EvalResult,
        agent_files: list[str],
    ) -> list[StructuralFix]:
        """Propose structural fixes for checkpoints. Works on any checkpoint, not just weak/fail."""
        if not checkpoint_results:
            return []

        fixes: list[StructuralFix] = []
        base_dirs = [Path(f).resolve().parent for f in agent_files] if agent_files else [Path.cwd()]

        for cr in checkpoint_results:
            cp = cr.checkpoint
            file_path = self._resolve_path(cp.file_path, base_dirs)
            code_context = _read_code_context(file_path, cp.line_number, context_lines=10)
            dim_score = eval_result.dimension_scores.get(cp.dimension)

            system_prompt = """You are a code improvement assistant. Given a checkpoint in an AI agent that scored below perfect in a TEI evaluation, propose a concrete structural code fix to improve the agent's Target Alignment, Reasoning Soundness, Execution Accuracy, or Output Integrity.

The fix must be a REAL code change — not a suggestion. Provide the exact code patch.

Respond with valid JSON only. Use this exact schema:
{
  "issue": "what specifically is wrong or suboptimal in the code at this checkpoint",
  "proposed_fix": "concrete description of the code change",
  "expected_impact": "which dimension(s) this improves and why",
  "code_patch": "REPLACE:\n<exact old code from the source>\nWITH:\n<new code> OR INSERT_AT_LINE <line_number>:\n<code to insert>"
}

IMPORTANT: The code_patch field MUST contain an actionable patch using REPLACE or INSERT_AT_LINE format. Do not leave it empty."""

            fail_summary = f"- Dimension failure summary: {dim_score.failure_summary}" if dim_score and dim_score.failure_summary else ""
            user_prompt = f"""Checkpoint:
- File: {cp.file_path}
- Line: {cp.line_number}
- Type: {cp.checkpoint_type}
- Dimension: {cp.dimension.value}
- Description: {cp.description}

Evaluation:
- Score: {cr.score:.2f}
- Status: {cr.status}
- Reasoning: {cr.reasoning}
{fail_summary}

Source code (20 lines around the checkpoint):
```
{code_context}
```

Propose a structural fix. Respond with JSON only."""

            try:
                out = await self.improve_llm.generate_json(system_prompt, user_prompt)
                if isinstance(out, dict):
                    fix = StructuralFix(
                        checkpoint_id=cp.checkpoint_id,
                        file_path=str(Path(file_path).resolve()),
                        issue=out.get("issue", ""),
                        proposed_fix=out.get("proposed_fix", ""),
                        expected_impact=out.get("expected_impact", ""),
                        code_patch=out.get("code_patch", ""),
                    )
                    fixes.append(fix)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        return fixes

    async def propose_holistic_fix(
        self,
        checkpoints: list[Checkpoint],
        checkpoint_results: list[CheckpointResult],
        eval_result: EvalResult,
        agent_files: list[str],
    ) -> list[StructuralFix]:
        """Propose ONE fix that improves ALL dimensions without degrading any."""
        base_dirs = [Path(f).resolve().parent for f in agent_files] if agent_files else [Path.cwd()]

        dim_lines = []
        for dim, ds in sorted(eval_result.dimension_scores.items(), key=lambda kv: kv[1].score):
            label = dim.value.replace("_", " ").title()
            fail_info = f" -- {ds.failure_summary}" if ds.failure_summary else ""
            dim_lines.append(f"  {label}: {ds.score:.2f}{fail_info}")
        dim_block = "\n".join(dim_lines)

        source_blocks = []
        for fp in agent_files[:3]:
            p = Path(fp)
            if p.exists():
                try:
                    src = p.read_text(encoding="utf-8", errors="replace")
                    if len(src) > 8000:
                        src = src[:8000] + "\n... (truncated)"
                    source_blocks.append(f"=== {p.name} ===\n{src}")
                except OSError:
                    pass
        source_text = "\n\n".join(source_blocks) if source_blocks else "(no source available)"

        system_prompt = """You are an expert AI agent architect. You propose ONE structural code fix that improves the agent across ALL four evaluation dimensions simultaneously:
- Target Alignment: does the agent pursue the correct objective?
- Reasoning Soundness: is the reasoning logical?
- Execution Accuracy: are tools and APIs called correctly?
- Output Integrity: is the output complete and accurate?

CRITICAL RULES:
1. The fix MUST improve the weakest dimension(s) WITHOUT degrading the stronger ones.
2. The fix must be a REAL code change with an exact code_patch.
3. Focus on structural improvements: better prompts, validation, error handling, output formatting, grounding instructions.

Respond with valid JSON only:
{
  "issue": "what is wrong across the dimensions",
  "proposed_fix": "concrete description of the single code change",
  "expected_impact": "how this fix improves ALL dimensions, not just one",
  "target_file": "which file to modify (filename only)",
  "code_patch": "REPLACE:\\n<exact old code>\\nWITH:\\n<new code>"
}

The code_patch MUST use REPLACE format with exact old code from the source."""

        user_prompt = f"""CURRENT EVALUATION SCORES (all 4 dimensions):
{dim_block}
Aggregate: {eval_result.aggregate_score:.2f}

AGENT SOURCE CODE:
{source_text}

Propose ONE structural fix that improves the weakest dimension(s) WITHOUT hurting the others. The fix should address the root cause that limits multiple dimensions at once.

Return valid JSON only."""

        try:
            out = await self.improve_llm.generate_json(system_prompt, user_prompt)
        except Exception:
            return []

        if not isinstance(out, dict) or not out.get("code_patch"):
            return []

        target_name = out.get("target_file", "")
        fix_file = ""
        for fp in agent_files:
            if Path(fp).name == target_name:
                fix_file = fp
                break
        if not fix_file and agent_files:
            fix_file = agent_files[0]

        cp_id = checkpoints[0].checkpoint_id if checkpoints else ""
        fix = StructuralFix(
            checkpoint_id=cp_id,
            file_path=fix_file,
            issue=out.get("issue", ""),
            proposed_fix=out.get("proposed_fix", ""),
            expected_impact=out.get("expected_impact", ""),
            code_patch=out.get("code_patch", ""),
        )
        return [fix]

    def _resolve_path(self, file_path: str, base_dirs: list[Path]) -> str:
        p = Path(file_path)
        if p.is_absolute() and p.exists():
            return str(p)
        for base in base_dirs:
            candidate = (base / file_path).resolve()
            if candidate.exists():
                return str(candidate)
        return file_path

    async def generate_fix_details(self, fix: StructuralFix) -> StructuralFix:
        """Generate the actual code patch for an approved fix."""
        if fix.code_patch:
            return fix

        code_context = _read_code_context(fix.file_path, 1, context_lines=100)
        system_prompt = """You are a code patching assistant. Given a proposed fix description, produce an exact code patch.

Respond with valid JSON only:
{
  "code_patch": "REPLACE:\n<exact old code from file>\nWITH:\n<new code> OR INSERT_AT_LINE <line_number>:\n<code to insert>"
}

The code_patch must be directly applicable: REPLACE uses exact string match, INSERT_AT_LINE inserts after the given line."""

        user_prompt = f"""File: {fix.file_path}
Issue: {fix.issue}
Proposed fix: {fix.proposed_fix}

Relevant source:
```
{code_context}
```

Produce the code_patch. JSON only."""

        try:
            out = await self.improve_llm.generate_json(system_prompt, user_prompt)
            if isinstance(out, dict) and out.get("code_patch"):
                fix.code_patch = out["code_patch"]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return fix

    def apply_fix(self, fix: StructuralFix) -> bool:
        """Apply an approved structural fix to the agent source file. Returns True if successful."""
        content = fix.user_alternative if fix.user_alternative else fix.code_patch
        if not content:
            return False

        path = Path(fix.file_path)
        if not path.exists():
            return False

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines(keepends=True)

            insert_match = re.match(r"INSERT_AT_LINE\s+(\d+)\s*:\s*\n?(.*)", content, re.DOTALL)
            if insert_match:
                line_num = int(insert_match.group(1))
                code_to_insert = insert_match.group(2).rstrip()
                if not code_to_insert.endswith("\n"):
                    code_to_insert += "\n"
                idx = min(max(0, line_num - 1), len(lines))
                lines.insert(idx, code_to_insert)
                path.write_text("".join(lines), encoding="utf-8")
                return True

            if "REPLACE:\n" in content and "\nWITH:\n" in content:
                parts = content.split("\nWITH:\n", 1)
                old_part = parts[0].replace("REPLACE:\n", "", 1).strip()
                new_part = parts[1].strip()
                if old_part in text:
                    new_text = text.replace(old_part, new_part, 1)
                    path.write_text(new_text, encoding="utf-8")
                    return True

            return False
        except OSError:
            return False
