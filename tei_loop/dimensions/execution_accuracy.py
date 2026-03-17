"""
Dimension 3: Execution Accuracy

Evaluates whether the agent executed its plan correctly: right tools with right
parameters, proper API calls, correct data transformations, no skipped steps.
"""

from ..models import Dimension, Trace
from .base import BaseJudge


class ExecutionAccuracyJudge(BaseJudge):
    dimension = Dimension.EXECUTION_ACCURACY

    def system_prompt(self) -> str:
        return """You are a TEI (Target-Evaluate-Improve) evaluation judge.
Your role: assess whether the agent executed its actions correctly.

You must produce verifiable assertions about execution quality.

Scoring rubric (0.0 to 0.97, never 1.0):
  0.85-0.97: All actions executed correctly, right tools, right parameters, no errors
  0.70-0.84: Mostly correct execution with minor parameter issues or inefficiencies
  0.50-0.69: Some execution errors, wrong tool choices, or incorrect parameters
  0.30-0.49: Significant execution failures, multiple wrong actions
  0.00-0.29: Agent failed to execute meaningful actions or crashed

Check for:
- Were the right tools/functions selected for the task?
- Were parameters correct and complete?
- Were API responses handled properly?
- Were there any errors or exceptions?
- Were all necessary steps executed (nothing skipped)?
- Was the execution efficient (no redundant calls)?

Return JSON with this exact structure:
{
  "score": <float 0.0-0.97>,
  "assertions": [
    {
      "claim": "The agent [correctly/incorrectly] [specific action]",
      "evidence": "[What happened in the execution]",
      "verdict": "pass|fail|partial",
      "explanation": "Why this execution step was correct or not"
    }
  ],
  "reasoning": "Overall assessment of execution accuracy",
  "failure_summary": "If score < threshold, what execution errors occurred"
}

Generate 3-6 assertions. Reference specific actions or tool calls."""

    def build_user_prompt(self, trace: Trace) -> str:
        agent_input = str(trace.agent_input or "")[:3000]
        agent_output = str(trace.agent_output or "")[:5000]

        steps_text = ""
        if trace.steps:
            step_details = []
            for s in trace.steps[:15]:
                detail = f"  [{s.step_type}] '{s.name}'"
                if s.input_data:
                    detail += f"\n    Input: {str(s.input_data)[:300]}"
                if s.output_data:
                    detail += f"\n    Output: {str(s.output_data)[:300]}"
                if s.error:
                    detail += f"\n    ERROR: {s.error}"
                detail += f"\n    Duration: {s.duration_ms:.0f}ms"
                step_details.append(detail)
            steps_text = "\nExecution trace:\n" + "\n".join(step_details)

        return f"""Evaluate EXECUTION ACCURACY for this agent run.

USER QUERY:
{agent_input}

AGENT OUTPUT:
{agent_output}
{steps_text}

Assess: Did the agent execute its actions correctly? Were the right tools used
with the right parameters? Were there errors? Generate verifiable assertions."""
