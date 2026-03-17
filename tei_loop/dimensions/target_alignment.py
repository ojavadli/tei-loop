"""
Dimension 1: Target Alignment

Evaluates whether the agent pursued the correct objective given the user query.
Checks: Did the agent understand what was asked? Did it stay on target throughout?
Did it address the actual need, not a hallucinated or tangential one?
"""

from ..models import Dimension, Trace
from .base import BaseJudge


class TargetAlignmentJudge(BaseJudge):
    dimension = Dimension.TARGET_ALIGNMENT

    def system_prompt(self) -> str:
        return """You are a TEI (Target-Evaluate-Improve) evaluation judge.
Your role: assess whether the agent correctly understood and pursued the user's objective.

You must produce verifiable assertions, not subjective opinions.

Scoring rubric (0.0 to 0.97, never 1.0):
  0.85-0.97: Agent fully understood the objective and every part of its output addresses it
  0.70-0.84: Agent understood the main objective but missed secondary requirements
  0.50-0.69: Agent partially addressed the objective, notable gaps or misinterpretation
  0.30-0.49: Agent misunderstood the core objective
  0.00-0.29: Agent pursued a completely different objective or produced irrelevant output

Return JSON with this exact structure:
{
  "score": <float 0.0-0.97>,
  "assertions": [
    {
      "claim": "The agent addressed [specific aspect of the query]",
      "evidence": "In the output: [quote or reference]",
      "verdict": "pass|fail|partial",
      "explanation": "Why this assertion holds or fails"
    }
  ],
  "reasoning": "Overall assessment of target alignment",
  "failure_summary": "If score < threshold, what specifically went wrong"
}

Generate 3-6 assertions. Each must reference specific content from input or output."""

    def build_user_prompt(self, trace: Trace) -> str:
        agent_input = str(trace.agent_input or "")[:3000]
        agent_output = str(trace.agent_output or "")[:5000]

        steps_text = ""
        if trace.steps:
            step_summaries = []
            for s in trace.steps[:10]:
                summary = f"  Step '{s.name}' ({s.step_type})"
                if s.error:
                    summary += f" ERROR: {s.error}"
                step_summaries.append(summary)
            steps_text = "\nAgent execution steps:\n" + "\n".join(step_summaries)

        return f"""Evaluate TARGET ALIGNMENT for this agent run.

USER QUERY (what the agent was asked to do):
{agent_input}

AGENT OUTPUT (what the agent produced):
{agent_output}
{steps_text}

Assess: Did the agent correctly identify and pursue the user's objective?
Generate verifiable assertions about alignment between query and output."""
