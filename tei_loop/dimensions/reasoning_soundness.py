"""
Dimension 2: Reasoning Soundness

Evaluates whether the agent's reasoning and planning were logical, non-contradictory,
and well-structured. Checks the quality of the thought process, not just the final answer.
"""

from ..models import Dimension, Trace
from .base import BaseJudge


class ReasoningSoundnessJudge(BaseJudge):
    dimension = Dimension.REASONING_SOUNDNESS

    def system_prompt(self) -> str:
        return """You are a TEI (Target-Evaluate-Improve) evaluation judge.
Your role: assess whether the agent's reasoning was logically sound.

You must produce verifiable assertions about the reasoning quality.

Scoring rubric (0.0 to 0.97, never 1.0):
  0.85-0.97: Reasoning is clear, logically valid, no contradictions, well-structured
  0.70-0.84: Reasoning is mostly sound with minor logical gaps
  0.50-0.69: Reasoning has noticeable flaws, unsupported jumps, or contradictions
  0.30-0.49: Reasoning is poorly structured with major logical errors
  0.00-0.29: No coherent reasoning visible, or reasoning contradicts itself fundamentally

Check for:
- Internal contradictions (agent says X then later says not-X)
- Unsupported claims (conclusions without evidence)
- Logical jumps (skipping necessary reasoning steps)
- Circular reasoning
- Appropriate use of evidence and facts

Return JSON with this exact structure:
{
  "score": <float 0.0-0.97>,
  "assertions": [
    {
      "claim": "The agent's reasoning about [topic] is [sound/flawed]",
      "evidence": "In the output: [quote showing reasoning]",
      "verdict": "pass|fail|partial",
      "explanation": "Why this reasoning step is sound or flawed"
    }
  ],
  "reasoning": "Overall assessment of reasoning quality",
  "failure_summary": "If score < threshold, what specific reasoning flaws exist"
}

Generate 3-6 assertions. Each must reference specific reasoning in the output."""

    def build_user_prompt(self, trace: Trace) -> str:
        agent_input = str(trace.agent_input or "")[:3000]
        agent_output = str(trace.agent_output or "")[:5000]

        steps_text = ""
        if trace.steps:
            step_summaries = []
            for s in trace.steps[:10]:
                out_preview = str(s.output_data or "")[:500]
                summary = f"  Step '{s.name}': {out_preview}"
                step_summaries.append(summary)
            steps_text = "\nAgent reasoning steps:\n" + "\n".join(step_summaries)

        return f"""Evaluate REASONING SOUNDNESS for this agent run.

USER QUERY:
{agent_input}

AGENT OUTPUT:
{agent_output}
{steps_text}

Assess: Is the agent's reasoning logically sound? Are there contradictions,
unsupported claims, or logical gaps? Generate verifiable assertions."""
