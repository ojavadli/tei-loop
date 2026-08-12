# TEI Loop

**Target, Evaluate, Improve** — a self-improving loop for AI agents.

TEI connects structured evaluation to automated improvement: it identifies *what* failed, *why* it failed, and applies the right fix (structural code change or prompt optimization) based on the failure type.

## Results

Applied to **30 real SWE-bench leaderboard agents** (TEI-SWE study). Pre-repair confirmatory result: 21/26 patched agents (105/130 votes); after repairing the one import-time defect, the adaptive retest shows 22/26 strict majorities, 17 unanimous (110/130 votes), with 4 agents preferring the baseline — 6,000 audited candidate versions (100 structural + 100 prompt per system) at ~$1.79 per agent, a pre-registered sham placebo rejecting the style explanation, and two pre-registered execution arms both null (gpt-4o-mini 3/36 vs 3/36; funded gpt-5.6-luna 47/100 vs 50/100). Full study, datasets, and paper: [tei-swe](https://github.com/ojavadli/tei-swe) · controlled benchmark program: [tei-bench](https://github.com/ojavadli/tei-bench).

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

### Step-7 Optimizer Modes: `pareto` · `cubic` · `hybrid`

Step 7 (prompt optimization only — never Step-3 structural code patching) has
three modes, selected with `--optimizer-mode`:

| Mode | Behavior |
|---|---|
| `pareto` (default) | The historical Pareto-front loop above. Fully backward compatible. |
| `cubic` | **D-ARC** — Discrete Adaptive Cubic Regularization: incumbent selection on the scalar loss `F(p) = 1 − U(p)`. |
| `hybrid` | D-ARC controls step acceptance and the regularization strength while a Pareto archive preserves every evaluated candidate's non-dominated metric vector and supplies the final options (ranked by scalar utility, safety-checked in rank order, baseline fallback). |

**What D-ARC is.** An *ARC-inspired, candidate-restricted local-surrogate
method for discrete prompt optimization*. With normalized metric scores
`z_j(p) = clamp(score_j(p)/100, 0, 1)` and normalized nonnegative weights
`w_j`, the scalar utility is `U(p) = Σ_j w_j z_j(p)` and D-ARC minimizes
`F(p) = 1 − U(p)`. At incumbent `p_k` it fits, by locality-weighted ridge
regression over previously evaluated prompts, a local quadratic surrogate in
an SVD subspace of a deterministic prompt-feature space `phi` (signed
blake2b-hashed character 3/4/5-grams, 256 dims, + 8 structural features,
L2-normalized), then scores each of up to 4 LLM-proposed candidate prompts
with the cubic model

```
m_k(p) = F(p_k) + g_kᵀ z + ½ zᵀ B_k z + (σ_k/3)·‖s‖₂³ ,   s = phi(p) − phi(p_k),  z = Q_kᵀ s
```

(the cubic penalty uses the FULL feature-space step norm, so novel directions
outside the fitted subspace are penalized), evaluates the model minimizer,
and accepts/rejects by `ρ_k = ared/pred` with the classical thresholds
(η₁ = 0.1, η₂ = 0.9; σ halves on very successful steps, doubles on failures,
clipped to [1e-4, 1e4]). The subproblem is solved only over the finite
LLM-proposed candidate set — no pretend decoding of feature vectors to text.

**Limitations (stated, not fine print).** `g_k` and `B_k` are *fitted
surrogate coefficients* in the feature space — not true derivatives of an
LLM; prompt text is discrete, so D-ARC does **not** inherit classical ARC's
convergence or O(ε^{-3/2}) complexity guarantees. Foundations of the
continuous method it adapts: Nesterov & Polyak (2006),
https://doi.org/10.1007/s10107-006-0706-8, and Cartis, Gould & Toint (ARC I),
https://people.maths.ox.ac.uk/cartis/papers/ARCpI.pdf.

```bash
# D-ARC needs numpy:  pip install 'tei-loop[cubic]'
python3 -m tei_loop agent.py --optimizer-mode cubic \
    --cubic-sigma 1.0 --cubic-warmup 5 --cubic-proposals 4 \
    --cubic-window 20 --cubic-patience 3
python3 -m tei_loop agent.py --optimizer-mode hybrid   # D-ARC + Pareto archive
```

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
