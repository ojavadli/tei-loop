"""
Dimension 4: Output Integrity

Evaluates the quality of the agent's final output: completeness, factual accuracy,
internal consistency, and format correctness. This is the "deliverable check."
"""

from ..models import Dimension, Trace
from .base import BaseJudge


class OutputIntegrityJudge(BaseJudge):
    dimension = Dimension.OUTPUT_INTEGRITY

    def system_prompt(self) -> str:
        return """You are a TEI (Target-Evaluate-Improve) evaluation judge.
Your role: assess the integrity of the agent's final output.

You must produce verifiable assertions about output quality.

Scoring rubric (0.0 to 0.97, never 1.0):
  0.85-0.97: Output is complete, factually accurate, internally consistent, well-formatted
  0.70-0.84: Output is mostly complete with minor gaps or formatting issues
  0.50-0.69: Output has noticeable gaps, some inaccuracies, or inconsistencies
  0.30-0.49: Output is substantially incomplete or contains significant errors
  0.00-0.29: Output is empty, gibberish, or fundamentally wrong

Check for:
- Completeness: Does the output address all parts of the query?
- Factual accuracy: Are stated facts correct and verifiable?
- Internal consistency: Does the output contradict itself?
- Format correctness: Is the output in the expected format?
- Missing information: Are there obvious gaps?
- Hallucinations: Are there fabricated facts not supported by the input?

Return JSON with this exact structure:
{
  "score": <float 0.0-0.97>,
  "assertions": [
    {
      "claim": "The output [correctly/incorrectly] [specific aspect]",
      "evidence": "[Quote or reference from the output]",
      "verdict": "pass|fail|partial",
      "explanation": "Why this aspect of the output passes or fails"
    }
  ],
  "reasoning": "Overall assessment of output integrity",
  "failure_summary": "If score < threshold, what integrity issues exist"
}

Generate 4-8 assertions. Each must reference specific content in the output."""

    def build_user_prompt(self, trace: Trace) -> str:
        agent_input = str(trace.agent_input or "")[:3000]
        agent_output = str(trace.agent_output or "")[:6000]

        return f"""Evaluate OUTPUT INTEGRITY for this agent run.

USER QUERY:
{agent_input}

AGENT OUTPUT:
{agent_output}

Assess: Is the output complete, factually accurate, internally consistent,
and properly formatted? Check for hallucinations, missing information,
and contradictions. Generate verifiable assertions."""
