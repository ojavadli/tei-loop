# TEI Loop

**Target, Evaluate, Improve** — a self-improving loop for AI agents.

TEI connects structured evaluation to automated improvement: it identifies *what* failed, *why* it failed, and applies the right fix (structural code change or prompt optimization) based on the failure type.

## Quick Start

```bash
# One command — install + run:
export OPENAI_API_KEY="sk-..."   # or ANTHROPIC_API_KEY or GOOGLE_API_KEY
pip3 install git+https://github.com/ojavadli/tei-loop.git && python3 -m tei_loop your_agent.py
```

TEI auto-detects your agent function, clones the file, runs the 8-step pipeline, and saves results to `tei-results/`. Your original file is never modified.

## The 8-Step Pipeline

| Step | What happens | Output |
|------|-------------|--------|
| 1. **Scan** | AST-parse agent files, find LLM calls, tool calls, outputs | Checkpoint locations |
| 2. **Baseline eval** | Run agent, score 4 dimensions via LLM-as-judge | Scores + failure diagnosis |
| 3. **Structural fixes** | 20-iteration batch: propose code patches, apply, eval, **rollback if worse** | Best structural improvement |
| 4. **Middle eval** | Re-evaluate after fixes, show delta from baseline | Before/after comparison |
| 5. **Metric proposal** | LLM proposes task-specific objective metrics | Approved metrics + weights |
| 6. **Prompt baseline** | Measure current prompts against confirmed metrics | Composite efficiency score |
| 7. **Prompt optimization** | Pareto-front optimization: mutation + merge over N iterations | Best prompt candidate |
| 8. **Final report** | Baseline → Middle → Final comparison across all dimensions | JSON report + optimized prompt |

## Evaluation Methodology

### 4 Dimensions (LLM-as-Judge)

| Dimension | Failure class | What the judge checks |
|---|---|---|
| **Target Alignment** | Drift from objective | Did the agent pursue what the user actually asked for? |
| **Reasoning Soundness** | Logic errors | Are intermediate steps coherent and grounded? |
| **Execution Accuracy** | Wrong tool calls / API errors | Were external calls made correctly with valid parameters? |
| **Output Integrity** | Hallucination / incompleteness | Is the final output accurate, complete, and non-fabricated? |

Each dimension is scored 0.00–1.00 by a dedicated judge prompt. The aggregate is the mean of all 4 scores.

### Failure → Fix Routing

| Failure type | Fix strategy |
|---|---|
| Target drift | Re-anchor system prompt to stated objective |
| Bad reasoning chain | Decompose task, add intermediate validation |
| Wrong tool call | Schema fix, parameter validation, error handling |
| Hallucinated output | Grounding instructions, structured validation |

### Structural Fix Safety

- TEI **clones** the original agent file (e.g. `agent.py` → `agentCLONE1.py`) in the same directory. The original is never touched.
- If a clone already exists, TEI auto-increments (`agentCLONE2.py`, etc.)
- After each patch, the modified clone is **reloaded and re-executed** to verify the fix actually works
- If the score drops, **the patch is automatically reverted**
- Only the best-performing version is kept across all iterations

### Prompt Optimization

- **Pareto front**: maintains non-dominated candidates across multiple metrics
- **Reflective mutation**: LLM reflects on trace failures, proposes targeted prompt changes
- **System-aware merge**: combines lessons from two strong candidates
- **Composite scoring**: weighted combination of all approved metrics
- Only applies the optimized prompt if it **actually improves** the composite score

## Python API

```python
import asyncio
from tei_loop import TEILoop

def my_agent(query: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
    )
    return r.choices[0].message.content

async def main():
    loop = TEILoop(
        agent=my_agent,
        agent_file="my_agent.py",   # Required for structural fixes + prompt extraction
        num_iterations=10,
        interactive=False,           # Auto-approve all (or True for Y/N prompts)
    )
    result = await loop.run(
        "What is the capital of France?",
        test_queries=["query1", "query2", "query3"],
    )
    print(result.summary())

asyncio.run(main())
```

## CLI Options

```bash
python3 -m tei_loop agent.py                        # Auto everything
python3 -m tei_loop agent.py --iterations 50         # 50 optimization runs
python3 -m tei_loop agent.py --query "custom input"  # Custom test query
python3 -m tei_loop agent.py --non-interactive       # Auto-approve all
python3 -m tei_loop agent.py --verbose               # Detailed output
```

## Supported Providers

| Provider | Eval model | Improve model | Env var |
|---|---|---|---|
| OpenAI | gpt-5.2 | gpt-5.1 | `OPENAI_API_KEY` |
| Anthropic | claude-opus-4-6 | claude-sonnet-4-6 | `ANTHROPIC_API_KEY` |
| Google | gemini-3-pro-preview | gemini-3-flash-preview | `GOOGLE_API_KEY` |

Auto-detected from environment. Falls back to available models if primary is unavailable.

## Output

TEI saves to `tei-results/`:
- `run_YYYYMMDD_HHMMSS.json` — full structured result (all scores, fixes, metrics, Pareto front)
- `optimized_prompts.json` — original and optimized prompt text
- `latest.json` — most recent run

## Current Status

**Alpha** — the core pipeline works end-to-end, but this is a research project, not a production tool.

What works:
- Full 8-step pipeline with real LLM calls
- 4-dimension evaluation with dedicated judges
- Structural fixes with automatic rollback
- Pareto-front prompt optimization
- Multi-provider support (OpenAI, Anthropic, Google)
- CLI + Python API

Known limitations:
- LLM-as-judge variance means scores can fluctuate between runs
- Structural fixes are most effective on simple, single-file agents
- Prompt optimization quality depends heavily on the metrics proposed

## How TEI Protects Your Code

TEI never modifies your original agent file. It creates a clone (`agentCLONE1.py`) in the same directory and experiments only on the clone. Each structural fix is applied, **reloaded at runtime**, and evaluated. If it doesn't improve the score, it's rolled back instantly. After optimization, you get:
- The improved clone file with structural fixes
- An optimized prompt saved to `TEI-work/optimized_prompt.txt`
- Full JSON results in `tei-results/`

## License

MIT
