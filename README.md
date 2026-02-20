# TEI Loop

**Target, Evaluate, Improve** — a self-improving loop for any AI agent.

## Get Started

```bash
pip install tei-loop
```

```bash
python3 -m tei_loop your_agent.py
```

TEI does the rest — 8 steps, fully automated with interactive approval gates.

## The 8-Step Flow

| Step | What happens |
|------|-------------|
| 1. Scan & Place Checkmarks | Scans all agent files, finds LLM calls, tool calls, outputs. Places evaluation checkmarks. |
| 2. Baseline Evaluation | Runs agent, scores 4 dimensions. "Before" snapshot. |
| 3. Structural Fixes | Proposes code fixes in terminal. You approve: Y/N/Propose Other. |
| 4. Middle Evaluation | Re-evaluates after fixes. Shows delta. |
| 5. Propose Metrics | Proposes task-specific objective metrics. You approve: Y/N/Propose Other. |
| 6. Baseline Efficiency | Measures current prompts against confirmed metrics. |
| 7. Iterative Optimization | 20-50 iterations: mutate prompts, track Pareto front, select best. |
| 8. Final Report | Baseline -> Middle -> Final comparison. Applies best prompt to source. |

## 4 Evaluation Dimensions

| Dimension | What it checks |
|---|---|
| **Target Alignment** | Did the agent pursue the correct objective? |
| **Reasoning Soundness** | Was the reasoning logical? |
| **Execution Accuracy** | Were tools called correctly? |
| **Output Integrity** | Is the output complete and accurate? |

## Options

```bash
python3 -m tei_loop agent.py                        # Auto everything
python3 -m tei_loop agent.py --iterations 50         # 50 optimization runs
python3 -m tei_loop agent.py --query "custom input"  # Custom test query
python3 -m tei_loop agent.py --non-interactive       # Auto-approve all
python3 -m tei_loop agent.py --verbose               # Detailed output
```

## Python API

```python
import asyncio
from tei_loop import TEILoop

def my_agent(query: str) -> str:
    # your agent code
    return result

async def main():
    loop = TEILoop(
        agent=my_agent,
        agent_file="my_agent.py",
        num_iterations=30,
    )
    result = await loop.run(
        "test query",
        test_queries=["query1", "query2", "query3", "query4", "query5"],
    )

asyncio.run(main())
```

## Multi-File Agents

TEI scans all imported local Python files automatically:

```
$ python3 -m tei_loop agent/main.py

Scanning agent files...
  Found: agent/main.py (200 lines)
  Found: agent/tools.py (150 lines)
  Found: agent/prompts.py (80 lines)

Identified 8 checkpoint locations across 3 files.
```

## Iterative Optimization

TEI:
- Maintains a **Pareto front** of non-dominated prompt candidates
- Uses **reflective mutation** (reflect on trace failures, propose improvements)
- Uses **system-aware merge** (combine lessons from two strong candidates)
- Runs **all configured iterations** (no early stopping)
- Selects the **best prompt by composite score**

## Works With Any Agent

Any Python callable. No framework lock-in: LangGraph, CrewAI, custom Python, FastAPI, anything.

## License

MIT
