# Reference implementation — the exact TEI used in the 2026 paper

These files are the **verbatim instrument and loop** behind *"TEI (Target–Evaluate–Improve)
Loop: A Joint Advantage over Agent-Improvement Methods with Low-Cost Adaptive Bottleneck
Optimization of Harnesses and Prompts for Self-Improving Agents"* (Zimina, Denisov-Blanch,
Javadli, 2026). They are published unmodified so every paper number can be traced to the
code that produced it; frozen study data lives in the artifacts repo
([tei-bench](https://github.com/ojavadli/tei-bench)).

| File | Role in the paper |
|---|---|
| `tei_pipeline.py` | The scoring instrument and TEI search: 4-dimension rubric (target_alignment, reasoning_soundness, execution_accuracy, output_integrity), `clamp(s)=min(max(s,0),0.999)`, aggregate = mean of clamped dimensions, weakest-dimension diagnosis, structural + prompt candidate generation, AST apply pre-gate, paraphrase-orbit null control (Sec. 2, Algorithm 1) |
| `blind_reval.py` | Blinded validation: direction-hidden A/B judging, k=5 independent votes per pair (Sec. 2.7) |
| `run_arm.py` | The preregistered 31-iteration comparison harness: adapted GEPA / ACE / AHE / MIPRO proposers and the Algorithm-1-faithful `teiv2` TEI proposer (re-diagnosis from best-so-far each batch, structural batches then prompt batches), do-no-harm gate = mean >= reference AND sign-test losses <= wins (Sec. 4, Appendix J) |

All model calls read `OPENAI_API_KEY` from the environment; every proposer/judge call in the
study ran on `gpt-5.6-luna`. The pip-installable `tei_loop/` package in this repo is the
interactive tool; where the two differ, **these files are what the paper measured.**
