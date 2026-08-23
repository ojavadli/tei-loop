#!/usr/bin/env python3
"""Matched-budget comparator arms vs TEI on the seed-11 TEN, all on gpt-5.6-luna.

Each arm reimplements a 2026 optimizer's PUBLISHED algorithm as a proposer
strategy: same agents, same probe set, same 4-dim rubric evaluator, same AST
apply pre-gate, same blinded protocol, same iteration budget. Only proposal +
selection differ. Endpoints (prereg): PRIMARY blinded A/B (k=5) TEI-best vs
comparator-best; SECONDARY deployed rubric delta, improvement per $/call.
Append-only to comparison/results/<arm>.jsonl; meters comparison/spend.json.

  set -a; . ~/.config/tei_run/openai.env; set +a
  ~/swebench-agents/_venv_aider/bin/python comparison/run_arm.py --arm gepa --only 05_joycode --smoke
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.expanduser("~/swebench-agents"))
import tei_pipeline as T
from tei_pipeline import call_json, BUDGET, clamp_score, aggregate, evidence_pack
import blind_reval as BR

ROOT = os.path.expanduser("~/swebench-agents")
AGENTS = os.path.join(ROOT, "agents")
RESULTS = os.path.join(ROOT, "comparison", "results")
SPEND = os.path.join(ROOT, "comparison", "spend.json")
os.makedirs(RESULTS, exist_ok=True)
TEN = ['05_joycode', '06_lingxi', '07_moatlesstools', '09_agentscope',
       '11_experepair', '15_composioswekit', '18_codefusecgm',
       '19_agentlesslite', '22_orcaloca', '24_swefixer']
K = 5


def sh(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace")


def show(repo, ref, path, n=2400):
    r = sh(["git", "show", f"{ref}:{path}"], repo)
    return r.stdout[:n] if r.returncode == 0 else ""


# ---------------- proposer strategies (faithful to each method) ----------------
def _ctx(ob, base, ev):
    return (f"SYSTEM: {ob['system']} ({ob.get('resolve_rate','?')}% resolved). "
            f"BASELINE dims {json.dumps(base.get('dimensions'))} (agg {base.get('aggregate')}). "
            f"DIAGNOSED FAILURE MODES: {json.dumps([f.get('name') for f in base.get('failure_modes', [])])}.\n"
            f"REPOSITORY CONTEXT:\n{ev}")

GROUND = ("CRITICAL: each edit is an exact find/replace and its 'find' MUST be copied VERBATIM from a "
          "file's content shown in REPOSITORY CONTEXT above (<=15 lines, must occur exactly in that "
          "file). Do NOT invent snippets; only edit code you can see verbatim above.")
JSON_EDITS = (GROUND + '\nJSON: {"candidates":[{"file":"<path from context>","find":"<verbatim snippet>",'
              '"replace":"<new text>","note":"one line"}]}')


def gepa(ob, base, ev, st, n):
    # reflective prompt evolution + Pareto: reflect on best-so-far, mutate/merge
    front = st.get("front", [])
    seed_txt = ("Best candidates so far (Pareto front by rubric dims):\n" +
                "\n".join(f"- dims {json.dumps(c['dims'])}: {c['note'][:90]}" for c in front[-4:])) if front else ""
    p = (f"You are GEPA, a reflective optimizer. Reflect on the diagnosed failures and the weakest "
         f"dimension, then propose {n} DISTINCT targeted edits (prefer instruction / decision-rule / "
         f"prompt strings; otherwise the logic implementing the weak dimension). Merge complementary "
         f"lessons from the front.\n{_ctx(ob,base,ev)}\n{seed_txt}\n{JSON_EDITS}")
    return call_json(p, max_out=8000).get("candidates", [])[:n], "reflective edit"


def ace(ob, base, ev, st, n):
    # agentic context engineering: grow an itemized playbook via delta updates
    pb = st.setdefault("playbook", [])
    pbtxt = "\n".join(f"- {b}" for b in pb[-8:]) if pb else "(empty)"
    p = (f"You are ACE (Agentic Context Engineering). Maintain an evolving PLAYBOOK of strategy bullets "
         f"from execution feedback; add/refine bullets (incremental delta, no wholesale rewrite) and emit "
         f"{n} edits that inject the improved guidance into the agent's instruction/context strings or "
         f"comments.\nCURRENT PLAYBOOK:\n{pbtxt}\n{_ctx(ob,base,ev)}\n"
         f'{GROUND}\nReturn JSON: {{"new_bullets":["..."],"candidates":[{{"file":"<path from context>",'
         f'"find":"<verbatim snippet>","replace":"<new text>","note":"one line"}}]}}')
    r = call_json(p, max_out=8000)
    pb.extend(r.get("new_bullets", [])[:3])
    return r.get("candidates", [])[:n], "playbook edit"


def ahe(ob, base, ev, st, n):
    # harness engineering: structural edits, each a falsifiable contract
    p = (f"You are AHE (Agentic Harness Engineering). Propose {n} DISTINCT STRUCTURAL edits to the agent "
         f"HARNESS (control flow, tool wiring, retry/recovery, error handling, middleware) — not prose "
         f"rewording. Each edit carries a 'contract': a one-line falsifiable prediction of the dimension "
         f"it improves.\n{_ctx(ob,base,ev)}\n{GROUND}\n"
         f'JSON: {{"candidates":[{{"file":"<path from context>","find":"<verbatim snippet>","replace":'
         f'"<new text>","note":"one line","contract":"predicts +X on <dim>"}}]}}')
    return call_json(p, max_out=8000).get("candidates", [])[:n], "structural harness edit"


def maestro(ob, base, ev, st, n):
    # joint graph+config AND prompt search under budget
    p = (f"You are Maestro, a holistic optimizer that JOINTLY searches the agent graph/config (control "
         f"flow, model/tool wiring) AND prompt text under a tight budget. Propose {n} DISTINCT edits; "
         f"each may change structure, config, or prompt — pick the highest expected-value change for the "
         f"diagnosed weakness.\n{_ctx(ob,base,ev)}\n{JSON_EDITS}")
    return call_json(p, max_out=8000).get("candidates", [])[:n], "joint graph+config+prompt edit"


def mipro(ob, base, ev, st, n):
    # instruction + few-shot demo optimization (anchor)
    p = (f"You are MIPRO, optimizing free-form INSTRUCTIONS and few-shot DEMONSTRATIONS (no "
         f"gradients/labels). Propose {n} DISTINCT edits that rewrite an instruction string and/or insert "
         f"a grounded demonstration into the agent's prompt/instruction text.\n{_ctx(ob,base,ev)}\n{JSON_EDITS}")
    return call_json(p, max_out=8000).get("candidates", [])[:n], "instruction+demo edit"


def tei(ob, base, ev, st, n):
    # TEI (Target-Evaluate-Improve): diagnose the weakest rubric dimension and
    # propose concrete edits that DIRECTLY raise it -- structural fixes where code
    # is editable, targeted prompt/instruction edits otherwise; ledger-informed.
    # Same harness as every other arm (rubric, AST gate, blinded, metered, 31 iters):
    # only the proposal strategy is TEI's, so its cost/dollar is commensurable.
    front = st.get("front", [])
    ledger = ("Ledger of scored candidates so far (best dims first):\n" +
              "\n".join(f"- dims {json.dumps(c['dims'])}: {c['note'][:80]}" for c in front[-4:])) if front else ""
    dims = base.get("dimensions") or {}
    weak = min(dims, key=dims.get) if dims else "execution_accuracy"
    p = (f"You are TEI (Target-Evaluate-Improve). TARGET the single weakest rubric dimension "
         f"('{weak}') and propose {n} DISTINCT edits that DIRECTLY raise it. Prefer concrete STRUCTURAL "
         f"fixes when code is editable -- add a verification/self-check test, decompose a planning step, "
         f"tighten an output contract, add localization or an invariant guard, repair a retry/recovery "
         f"path -- and fall back to a targeted instruction/prompt edit only when no structural surface "
         f"applies. Name the failure mode each edit removes.\n{_ctx(ob,base,ev)}\n{ledger}\n{JSON_EDITS}")
    return call_json(p, max_out=8000).get("candidates", [])[:n], f"TEI targeted fix ({weak})"


def _tei_best(st):
    """Best-so-far ledger entry (Algorithm 1: A* = argmax_L S); None before batch 1."""
    best, bs = None, -1.0
    for c in st.get("front", []):
        vals = [v for v in (c.get("dims") or {}).values() if isinstance(v, (int, float))]
        if vals and sum(vals) / len(vals) > bs:
            bs, best = sum(vals) / len(vals), c
    return best


def teiv2(ob, base, ev, st, n):
    # TEI per the paper's Algorithm 1, FAITHFUL (supersedes the v1 arm, which was
    # not TEI: static baseline diagnosis, no best-so-far building, prompt phase
    # suppressed). Here: (i) per-batch RE-DIAGNOSIS of the weakest dimension from
    # the best-so-far ledger entry; (ii) proposals BUILD ON the best-so-far change;
    # (iii) two phases like the deployed pipeline -- structural first (batches 1-3,
    # 18 cands), then prompt/instruction refinement (batches 4-6, 13 cands).
    st["tei_batch"] = st.get("tei_batch", 0) + 1
    phase = "structural" if st["tei_batch"] <= 3 else "prompt"
    bestc = _tei_best(st)
    dims = (bestc or {}).get("dims") or base.get("dimensions") or {}
    dims = {k: v for k, v in dims.items() if isinstance(v, (int, float))}
    weak = min(dims, key=dims.get) if dims else "execution_accuracy"
    seed = (f"BEST-SO-FAR candidate (BUILD ON IT; do not regress it): dims "
            f"{json.dumps(bestc['dims'])}; change: {(bestc.get('note') or '')[:110]}"
            if bestc else "No scored candidate yet: improve from the baseline.")
    if phase == "structural":
        ask = (f"propose {n} DISTINCT STRUCTURAL fixes aimed at '{weak}': add a verification/self-check "
               f"test, decompose a planning step, tighten an output contract, add localization or an "
               f"invariant guard, repair a retry/recovery path. Name the failure mode each edit removes.")
    else:
        ask = (f"the structural phase is done; now propose {n} DISTINCT PROMPT/INSTRUCTION refinements "
               f"aimed at '{weak}': rewrite the operative instruction, decision rule, or output-format "
               f"text inside the agent's prompt/instruction strings shown in the context.")
    p = (f"You are TEI (Target-Evaluate-Improve), iteration batch {st['tei_batch']}, {phase} phase. "
         f"TARGET: the current weakest rubric dimension is '{weak}' (re-diagnosed from the best-so-far); "
         f"{ask}\n{seed}\n{_ctx(ob,base,ev)}\n{JSON_EDITS}")
    return call_json(p, max_out=8000).get("candidates", [])[:n], f"TEI {phase} fix ({weak})"


PROPOSERS = {"gepa": gepa, "ace": ace, "ahe": ahe, "maestro": maestro, "mipro": mipro,
             "tei": tei, "teiv2": teiv2}


def score_batch(ob, base, cands, notes):
    slim = [{"i": i, "technique": notes, "change": (str(c.get("replace") or c.get("note", ""))[:300])}
            for i, c in enumerate(cands)]
    p = (f"Strictly score each candidate VERSION of this SWE-bench agent on the TEI rubric.\n"
         f"SYSTEM: {ob['system']} ({ob.get('resolve_rate','?')}% resolved). Phase: structural.\n"
         f"BASELINE dimensions: {json.dumps(base.get('dimensions'))} (aggregate {base.get('aggregate')})\n"
         f"DIAGNOSED FAILURE MODES: {json.dumps([f.get('name') for f in base.get('failure_modes', [])])}\n"
         f"CANDIDATE VERSIONS:\n{json.dumps(slim, indent=1)}\n"
         f"Be strict and differentiating: most targeted changes move a dimension by a small amount "
         f"(+/-0.01-0.06); some make things worse and MUST score below baseline. No value may exceed 0.99.\n"
         f'JSON: {{"results":[{{"i":0,"dimensions":{{"target_alignment":0.0,"reasoning_soundness":0.0,'
         f'"execution_accuracy":0.0,"output_integrity":0.0}},"why":"one sentence"}}]}}')
    return call_json(p, max_out=6000).get("results", [])


def blind_pair(ob, probes, pairs, rng):
    """k=5 direction-hidden A/B. pairs=[(path, challenger, other)]. -> {challenger,other,tie}."""
    w = {"challenger": 0, "other": 0, "tie": 0}
    for _ in range(K):
        if not pairs:
            break
        flip = rng.random() < 0.5
        blocks = []
        for p, ch, ot in pairs:
            v1, v2 = (ot, ch) if flip else (ch, ot)
            blocks.append(f"### file: {p}\n--- VERSION 1 ---\n{v1}\n--- VERSION 2 ---\n{v2}")
        prompt = (f"You are comparing two versions of the same SWE agent ({ob['system']}; it resolves real "
                  f"GitHub issues). For each shown file are the two versions in differing regions. You are NOT "
                  f"told which is which.\nRepresentative instances: {', '.join(probes)}.\n\n" + "\n".join(blocks) +
                  '\n\nWhich version is more likely to resolve such issues correctly end-to-end? If too '
                  'trivial/ambiguous, say tie.\nReturn ONLY JSON: {"better":"1"|"2"|"tie","confidence":0.0,"reason":"one sentence"}')
        try:
            v = call_json(prompt, max_out=1200)
        except Exception:
            w["tie"] += 1
            continue
        pick = str(v.get("better", "")).strip()
        if pick not in ("1", "2"):
            w["tie"] += 1
        else:
            chose_ch = (pick == "2") if flip else (pick == "1")
            w["challenger" if chose_ch else "other"] += 1
    return w


PRIOR_SPENT = 0.0  # spend from previous processes (cap accumulates across relaunches)


def meter(arm):
    d = json.load(open(SPEND))
    total = round(PRIOR_SPENT + BUDGET.conservative, 4)
    d["spent_usd"] = total
    d.setdefault("arms", {})[arm] = {"proc_spent_usd": round(BUDGET.conservative, 4),
                                     "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(d, open(SPEND, "w"), indent=1)
    return total


def run_agent(arm, d, iters, rng):
    repo = os.path.join(AGENTS, d)
    ob = json.load(open(os.path.join(repo, "tei", "onboarding.json")))
    base = json.load(open(os.path.join(repo, "tei", "baseline_eval.json")))
    res = json.load(open(os.path.join(repo, "tei", "result.json")))
    base_sha = ob["repo_sha"]
    cur = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo).stdout.strip()
    t0 = time.time()
    sh(["git", "checkout", "-q", "-B", f"{arm}-v1", base_sha], repo)
    row = {}
    try:
        ev = evidence_pack(repo, ob)
        st, applied, scores, calls0, cost0 = {}, 0, [], BUDGET.calls, BUDGET.conservative
        for b in range(0, iters, 6):
            n = min(6, iters - b)
            try:
                cands, notes = PROPOSERS[arm](ob, base, ev, st, n)
            except Exception as e:
                print(f"  gen fail {d}/{arm}: {str(e)[:80]}"); continue
            for c in cands:
                if all(k in c for k in ("file", "find", "replace")):
                    ok, _ = T.apply_patch(repo, c, f"{arm}-{b}")
                    applied += bool(ok)
            try:
                for s in score_batch(ob, base, cands, notes):
                    agg = aggregate({k: clamp_score(v) for k, v in (s.get("dimensions") or {}).items()})
                    if agg is not None:
                        scores.append(agg)
                        st.setdefault("front", []).append({"dims": s.get("dimensions", {}), "note": s.get("why", "")})
            except Exception as e:
                print(f"  score fail {d}/{arm}: {str(e)[:80]}")
        best = max(scores) if scores else None
        best_delta = round(best - base["aggregate"], 4) if best is not None else None
        probes = [p["instance_id"] for p in base.get("probes", [])][:4]
        # blinded vs baseline: challenger = arm (after), other = baseline (before)
        vb = blind_pair(ob, probes,
                        [(p, e[1], e[0]) for p in BR.changed_files(repo, base_sha)
                         if (e := BR.excerpt_pair(repo, base_sha, p))], rng)
        # PRIMARY: comparator-best vs TEI-best (only if the arm actually changed the agent)
        tei_ref = "tei-v7"
        armfiles = [x for x in sh(["git", "diff", "--name-only", base_sha, f"{arm}-v1"], repo).stdout.split("\n") if x.strip()]
        vtei = None
        if armfiles and sh(["git", "rev-parse", "--verify", "-q", tei_ref], repo).returncode == 0:
            allf = set(armfiles)
            allf.update(x for x in sh(["git", "diff", "--name-only", base_sha, tei_ref], repo).stdout.split("\n") if x.strip())
            tpairs = [(p, show(repo, f"{arm}-v1", p), show(repo, tei_ref, p)) for p in list(allf)[:6]]
            tpairs = [t for t in tpairs if t[1] or t[2]]
            if tpairs:
                vtei = blind_pair(ob, probes, tpairs, rng)
        row = {"arm": arm, "agent": d, "model": T.MODEL, "iters": iters, "n_applied": applied,
               "best_delta": best_delta, "tei_shipped_delta": res.get("shipped_delta"),
               "blind_vs_baseline": {"arm": vb["challenger"], "baseline": vb["other"], "tie": vb["tie"]},
               "blind_vs_tei": ({"comparator": vtei["challenger"], "tei": vtei["other"], "tie": vtei["tie"]} if vtei else None),
               "calls": BUDGET.calls - calls0, "cost_usd": round(BUDGET.conservative - cost0, 4),
               "seconds": round(time.time() - t0, 1), "scores": scores, "seed": 11,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        assert row["model"] == "gpt-5.6-luna", row["model"]
    finally:
        sh(["git", "checkout", "-q", cur or "tei-v7"], repo)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", required=True, choices=list(PROPOSERS))
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")   # results dir under comparison/
    ap.add_argument("--spend", default="spend.json")  # spend meter file under comparison/
    ap.add_argument("--all", action="store_true")    # all agents/ with tei/ data (30), not seed-11 TEN
    ap.add_argument("--cap", type=float, default=50.0)
    a = ap.parse_args()
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY missing from environment"
    global PRIOR_SPENT, RESULTS, SPEND
    RESULTS = os.path.join(ROOT, "comparison", a.outdir)
    SPEND = os.path.join(ROOT, "comparison", a.spend)
    os.makedirs(RESULTS, exist_ok=True)
    if not os.path.exists(SPEND):
        json.dump({"cap_usd": a.cap, "spent_usd": 0.0, "model": T.MODEL, "arms": {}},
                  open(SPEND, "w"), indent=1)
    PRIOR_SPENT = json.load(open(SPEND)).get("spent_usd", 0.0)
    ALL30 = sorted(x for x in os.listdir(AGENTS)
                   if os.path.isdir(os.path.join(AGENTS, x, "tei")))
    agents = (a.only or ([TEN[0]] if a.smoke else (ALL30 if a.all else TEN)))
    rng = random.Random(11)
    print(f"ARMS {a.arm} | {len(agents)} agents | iters {a.iters} | model {T.MODEL} | cap ${a.cap}", flush=True)
    stop = False
    for arm in a.arm:
        if stop:
            break
        out = os.path.join(RESULTS, f"{arm}.jsonl")
        done = ({json.loads(l)["agent"] for l in open(out) if l.strip()}
                if os.path.exists(out) else set())
        for d in agents:
            if d in done:
                print(f"  [{arm}] {d} exists, skip", flush=True)
                continue
            row = run_agent(arm, d, a.iters, rng)
            if not row:
                continue
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            spent = meter(arm)
            print(f"  [{arm}] {d:24s} appl {row['n_applied']}/{a.iters} bestΔ {row['best_delta']} "
                  f"vsTEI {row['blind_vs_tei']} vsBase {row['blind_vs_baseline']} | "
                  f"{row['calls']}c ${row['cost_usd']:.2f} {row['seconds']}s [spent ${spent:.2f}]", flush=True)
            if spent > a.cap:
                print("SPEND CAP HIT — stopping", flush=True)
                stop = True
                break


if __name__ == "__main__":
    main()
