#!/usr/bin/env python3
"""TEI v7 application driver for the frozen 30-agent SWE-bench set.

Per agent: baseline -> 30 structural versions -> select best -> 30 prompt
versions -> best overall, with a do-no-harm confirmation from the deployed
methodology (tei_loop.gate.verify_candidate) plus its MDE/noise-floor
diagnostics. Every version gets a score and a why-record.

ALL experiment LLM calls go to api.openai.com with model gpt-5.6-luna. No
fallback, no other model, and never the Anthropic key.

Score substrate:
  VERIFIED - real execution outcomes on the fixed paired instance set (Tier A)
  PROXY    - judge rubric scores of the version against the diagnosed failure
             modes and fixed probe instances (agents that cannot be executed)
Every number is labelled; the two are never mixed.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.expanduser("~/swebench-agents")
AGENTS = os.path.join(ROOT, "agents")
ARCHIVE = os.path.join(ROOT, "archive")
STATE_PATH = os.path.join(ROOT, "_run_state.json")
MODEL = "gpt-5.6-luna"
API_URL = "https://api.openai.com/v1/responses"

# Cost: this key lacks api.usage.read, so billed cost cannot be read back.
# Tokens are metered exactly; dollars are an ASSUMPTION, stated everywhere.
NOMINAL_IN, NOMINAL_OUT = 1.25, 10.00      # $/Mtok, assumed list price
CONSERV_IN, CONSERV_OUT = 2.50, 20.00      # $/Mtok, 2x safety factor for the cap
BUDGET_CAP = 25.00

DIMS = ["target_alignment", "reasoning_soundness", "execution_accuracy", "output_integrity"]


# ----------------------------------------------------------------- api layer
EMB_IN, EMB_IN_CONS = 0.02, 0.04           # $/Mtok text-embedding-3-small, list / 2x


class Budget:
    def __init__(self):
        self.calls = 0
        self.tin = 0
        self.tout = 0
        self.emb_calls = 0
        self.emb_tokens = 0
        self.scale_downs = []

    @property
    def nominal(self):
        return (self.tin / 1e6 * NOMINAL_IN + self.tout / 1e6 * NOMINAL_OUT
                + self.emb_tokens / 1e6 * EMB_IN)

    @property
    def conservative(self):
        return (self.tin / 1e6 * CONSERV_IN + self.tout / 1e6 * CONSERV_OUT
                + self.emb_tokens / 1e6 * EMB_IN_CONS)

    def add(self, u):
        self.calls += 1
        self.tin += u.get("input_tokens", 0)
        self.tout += u.get("output_tokens", 0)

    def add_emb(self, n):
        self.emb_calls += 1
        self.emb_tokens += n

    def as_dict(self):
        return {"calls": self.calls, "input_tokens": self.tin, "output_tokens": self.tout,
                "embedding_calls": self.emb_calls, "embedding_tokens": self.emb_tokens,
                "cost_nominal_usd": round(self.nominal, 4),
                "cost_conservative_usd": round(self.conservative, 4),
                "pricing_assumption": f"nominal ${NOMINAL_IN}/${NOMINAL_OUT} per Mtok, "
                                      f"conservative ${CONSERV_IN}/${CONSERV_OUT}; "
                                      f"embeddings ${EMB_IN}/${EMB_IN_CONS} per Mtok; "
                                      "billed cost unreadable (key lacks api.usage.read)",
                "scale_downs": self.scale_downs}


BUDGET = Budget()


def call_llm(prompt, max_out=3000, retries=4, json_mode=False, model=None):
    """One gpt-5.6-luna call. Returns text. Never falls back to another model."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set")
    mdl = model or MODEL
    payload = {"model": mdl, "input": prompt, "max_output_tokens": max_out}
    if json_mode:
        # Enforced at the API layer: some agents' prompts reliably drew markdown
        # lists back instead of JSON, and retrying the same free-form ask never fixed it.
        payload["text"] = {"format": {"type": "json_object"}}
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(API_URL, data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
            if d.get("model") and not str(d["model"]).startswith(mdl):
                raise SystemExit(f"model mandate violated: server returned {d['model']}")
            BUDGET.add(d.get("usage", {}))
            txt = "".join(c["text"] for it in d.get("output", [])
                          for c in (it.get("content") or []) if c.get("type") == "output_text")
            if txt.strip():
                return txt
            last = f"empty output (status={d.get('status')})"
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200].decode(errors='ignore')}"
            if e.code in (400, 401, 403, 404):
                raise SystemExit(f"fatal API error: {last}")
        except Exception as e:  # timeouts, transient 5xx
            last = str(e)[:200]
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"gpt-5.6-luna call failed after {retries} attempts: {last}")


EMB_URL = "https://api.openai.com/v1/embeddings"
EMB_MODEL = "text-embedding-3-small"


def call_embed(texts, retries=4):
    """Embed texts with OpenAI text-embedding-3-small (the only non-luna call the
    mission authorizes). Chunked at 512 inputs; tokens metered into BUDGET."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set")
    out = []
    for i in range(0, len(texts), 512):
        chunk = [(t or " ")[:4000] for t in texts[i:i + 512]]
        body = json.dumps({"model": EMB_MODEL, "input": chunk}).encode()
        last = None
        for attempt in range(retries):
            req = urllib.request.Request(EMB_URL, data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    d = json.loads(r.read())
                BUDGET.add_emb((d.get("usage") or {}).get("prompt_tokens", 0))
                out.extend(e["embedding"] for e in d["data"])
                break
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read()[:200].decode(errors='ignore')}"
                if e.code in (400, 401, 403, 404):
                    raise SystemExit(f"fatal embeddings error: {last}")
            except Exception as e:
                last = str(e)[:200]
            time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"embeddings failed after {retries} attempts: {last}")
    return out


def call_json(prompt, max_out=3000, model=None):
    """Call the model in enforced JSON mode and parse the reply."""
    txt = call_llm(prompt + "\n\nReturn ONLY a valid JSON object. No prose, no code fences.",
                   max_out, json_mode=True, model=model)
    t = txt.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise ValueError(f"unparseable JSON: {t[:200]}")


# ------------------------------------------------------------------ helpers
def clamp_score(x):
    """No perfect scores: aggregates must be strictly below 1.000."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(v, 0.999))


def aggregate(dims):
    vals = [clamp_score(dims.get(d)) for d in DIMS]
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def sh(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def read_head(path, n=1800):
    try:
        return open(path, errors="ignore").read()[:n]
    except OSError:
        return ""


def evidence_pack(agent_dir, ob):
    """Compact, real evidence about this agent: README + prompt-surface excerpts."""
    parts = []
    for name in ("README.md", "README_EN.md", "readme.md"):
        p = os.path.join(agent_dir, name)
        if os.path.isfile(p):
            parts.append(f"--- {name} (head) ---\n{read_head(p, 1500)}")
            break
    for h in ob.get("prompt_surface", [])[:3]:
        p = os.path.join(agent_dir, h["path"])
        parts.append(f"--- prompt surface: {h['path']} (head) ---\n{read_head(p, 1200)}")
    if not ob.get("prompt_surface"):
        parts.append("--- NOTE: no prompt surface found in this repository ---")
    return "\n\n".join(parts)[:9000]


def probe_instances(ob, k=6, seed=0):
    """Fixed-seed probe set: resolved + unresolved instances this system attempted."""
    rng = random.Random(seed)
    resolved = list(ob.get("resolved_instance_ids") or [])
    split = ob["split"]
    all_ids = set()
    base = os.path.join(ARCHIVE, "evaluation", split)
    for folder in os.listdir(base):
        rp = os.path.join(base, folder, "results", "results.json")
        if os.path.isfile(rp):
            try:
                r = json.load(open(rp))["resolved"]
            except Exception:
                continue
            if isinstance(r, list):
                all_ids.update(r)
    unresolved = sorted(all_ids - set(resolved))
    rng.shuffle(resolved)
    rng.shuffle(unresolved)
    half = max(1, k // 2)
    picks = [{"instance_id": i, "archive_outcome": "resolved"} for i in resolved[:half]]
    picks += [{"instance_id": i, "archive_outcome": "unresolved_by_this_system"}
              for i in unresolved[:k - len(picks)]]
    return picks


# ------------------------------------------------------------------- phases
def baseline_eval(agent_dir, ob, probes):
    ev = evidence_pack(agent_dir, ob)
    prompt = f"""You are a strict evaluator applying the TEI rubric to a real SWE-bench agent system.

SYSTEM: {ob['system']}
SWE-bench split: {ob['split']}   officially resolved: {ob['resolved']} ({ob['resolve_rate']}%)
Repo: {ob['repo_url']} @ {ob['repo_sha'][:12]}

REAL EVIDENCE FROM THE REPOSITORY:
{ev}

FIXED PROBE INSTANCES (from the archive's recorded per-instance outcomes for THIS system):
{json.dumps(probes, indent=1)}

Score the DEFAULT (unmodified) system on the four TEI dimensions, calibrated against
its real {ob['resolve_rate']}% resolve rate. Be strict and calibrated: a system that
fails ~{round(100 - ob['resolve_rate'], 1)}% of instances must not score near 1.0.
No dimension may be 1.0; the maximum allowed is 0.99.

Also give per-probe scores in [0,1] for how well the default system handles each probe,
and the top 3 recurring failure modes with the probe ids that evidence them.

JSON shape:
{{"dimensions":{{"target_alignment":0.0,"reasoning_soundness":0.0,"execution_accuracy":0.0,"output_integrity":0.0}},
 "probe_scores":[{{"instance_id":"...","score":0.0}}],
 "weakest_dimension":"...",
 "failure_modes":[{{"name":"...","description":"...","evidence_instance_ids":["..."]}}],
 "why":"2-3 sentences grounded in the evidence above"}}"""
    d = call_json(prompt, max_out=6000)
    d["dimensions"] = {k: clamp_score(v) for k, v in (d.get("dimensions") or {}).items()}
    d["aggregate"] = aggregate(d["dimensions"])
    for p in d.get("probe_scores", []):
        p["score"] = clamp_score(p.get("score"))
    return d


def gen_candidates(ob, phase, baseline, best_so_far, n, agent_dir, has_code):
    ev = evidence_pack(agent_dir, ob)
    fms = json.dumps(baseline.get("failure_modes", []), indent=1)
    if phase == "structural":
        what = ("concrete STRUCTURAL code fixes. Where the repository contains code, give an exact "
                "replacement: 'file' (path relative to repo root), 'find' (an exact snippet that "
                "occurs verbatim, <=15 lines), 'replace' (the new text). If the repository contains "
                "no patchable source code, set file/find/replace to null and say so in 'why'.")
    else:
        what = ("PROMPT-SURFACE rewrites applied on top of the best structural version. Give 'file' "
                "(a prompt-surface file), 'find' (exact snippet of prompt text) and 'replace'. If no "
                "prompt surface exists, set them null and describe the intended rewrite in 'why'.")
    prompt = f"""Propose {n} DISTINCT {phase} improvement candidates for this SWE-bench agent.

SYSTEM: {ob['system']}  ({ob['resolve_rate']}% resolved on {ob['split']})
Repo has patchable source code: {has_code}
Baseline TEI aggregate: {baseline.get('aggregate')}   weakest: {baseline.get('weakest_dimension')}
Best version so far: {best_so_far}

DIAGNOSED FAILURE MODES:
{fms}

REPOSITORY EVIDENCE:
{ev}

Each candidate must target ONE diagnosed failure mode and be {what}

JSON: {{"candidates":[{{"technique":"...","target_failure_mode":"...","expected_dimension":"one of {DIMS}",
"file":"...|null","find":"...|null","replace":"...|null","why":"why this should help"}}]}}"""
    out = call_json(prompt, max_out=8000)
    out = out if isinstance(out, list) else out.get("candidates", [])
    return [c for c in out if isinstance(c, dict)]   # luna occasionally emits bare strings


FAMILY_GLOSS = {
    "verification-tests": "add/strengthen verification, tests, regression oracles, reproduction steps",
    "patch-format": "patch/diff formatting, parsing, extraction, normalization, apply robustness",
    "localization": "fault localization, code search, root-cause tracing",
    "retrieval-context": "retrieval, repo maps, context construction, indexing",
    "prompt-structure": "prompt/instruction structure, roles, templates, system messages",
    "invariants-guards": "invariants, guards, validation checks, sanity pre/post-conditions",
    "retry-recovery": "retry, recovery, fallbacks, timeouts, budget loops",
    "output-contract": "output contracts, schemas, JSON validity, final submission formatting",
    "decomposition-planning": "task decomposition, planning, hypotheses, acceptance criteria",
    "other": "any other targeted improvement grounded in a diagnosed failure mode",
}


def gen_candidates_bcl(ob, phase, baseline, best_so_far, n, agent_dir, has_code, ledger, fams):
    """BCL generation contract (Phase B): the technique family for each slot is
    CHOSEN by the ledger's Thompson draw over the two-head posteriors and STATED
    to the model; the model instantiates the concrete fix. The mathematics
    steers, the model writes."""
    ev = evidence_pack(agent_dir, ob)
    fms = json.dumps(baseline.get("failure_modes", []), indent=1)
    if phase == "structural":
        what = ("concrete STRUCTURAL code fixes. Where the repository contains code, give an exact "
                "replacement: 'file' (path relative to repo root), 'find' (an exact snippet that "
                "occurs verbatim, <=15 lines), 'replace' (the new text). If the repository contains "
                "no patchable source code, set file/find/replace to null and say so in 'why'.")
    else:
        what = ("PROMPT-SURFACE rewrites applied on top of the best structural version. Give 'file' "
                "(a prompt-surface file), 'find' (exact snippet of prompt text) and 'replace'. If no "
                "prompt surface exists, set them null and describe the intended rewrite in 'why'.")
    diag = (f"phase={phase}; weakest dimension={baseline.get('weakest_dimension')}; failure modes: "
            + "; ".join(str(f.get("name")) for f in baseline.get("failure_modes", [])))
    ledger_block = ledger.render(diag)
    slots = "\n".join(f"- candidate {i + 1}: technique family '{f}' ({FAMILY_GLOSS.get(f, '')})"
                      for i, f in enumerate(fams))
    prompt = f"""Propose {n} DISTINCT {phase} improvement candidates for this SWE-bench agent.

SYSTEM: {ob['system']}  ({ob['resolve_rate']}% resolved on {ob['split']})
Repo has patchable source code: {has_code}
Baseline TEI aggregate: {baseline.get('aggregate')}   weakest: {baseline.get('weakest_dimension')}
Best version so far: {best_so_far}

DIAGNOSED FAILURE MODES:
{fms}

{ledger_block}

SLOT ASSIGNMENTS (technique family per slot, chosen by Thompson sampling over the Bayesian
credit posteriors above; implement each candidate WITHIN its assigned family):
{slots}

REPOSITORY EVIDENCE:
{ev}

Each candidate must implement its ASSIGNED technique family, target ONE diagnosed failure
mode, and be {what}

JSON: {{"candidates":[{{"technique":"...","target_failure_mode":"...","expected_dimension":"one of {DIMS}",
"file":"...|null","find":"...|null","replace":"...|null","why":"why this should help"}}]}}"""
    out = call_json(prompt, max_out=8000)
    out = out if isinstance(out, list) else out.get("candidates", [])
    return [c for c in out if isinstance(c, dict)]   # luna occasionally emits bare strings


def eval_candidates(ob, phase, baseline, cands, probes):
    slim = [{"i": i, "technique": c.get("technique"), "target_failure_mode": c.get("target_failure_mode"),
             "change": (str(c.get("replace"))[:300] if c.get("replace") else c.get("why", ""))[:300]}
            for i, c in enumerate(cands)]
    prompt = f"""Strictly score each candidate VERSION of this SWE-bench agent on the TEI rubric.

SYSTEM: {ob['system']} ({ob['resolve_rate']}% resolved). Phase: {phase}.
BASELINE dimensions: {json.dumps(baseline.get('dimensions'))} (aggregate {baseline.get('aggregate')})
BASELINE per-probe: {json.dumps(baseline.get('probe_scores'))}
DIAGNOSED FAILURE MODES: {json.dumps([f.get('name') for f in baseline.get('failure_modes', [])])}

CANDIDATE VERSIONS:
{json.dumps(slim, indent=1)}

For each candidate score the four TEI dimensions AND the same probe instances.
Be strict and differentiating: most targeted changes move a dimension by a small
amount (±0.01-0.06); some make things worse and MUST score below baseline. Do not
give every candidate the same score. No value may exceed 0.99.

JSON: {{"results":[{{"i":0,"dimensions":{{"target_alignment":0.0,"reasoning_soundness":0.0,"execution_accuracy":0.0,"output_integrity":0.0}},
"probe_scores":[{{"instance_id":"...","score":0.0}}],"why":"what changed and why it scored above/below the previous best"}}]}}"""
    out = call_json(prompt, max_out=9000)
    return out if isinstance(out, list) else out.get("results", [])


def apply_patch(agent_dir, cand, tag):
    """Apply an exact-match replacement and commit. Returns (applied, note)."""
    f, find, repl = cand.get("file"), cand.get("find"), cand.get("replace")
    if not (f and find and repl):
        return False, "no concrete patch (null file/find/replace)"
    path = os.path.join(agent_dir, str(f).lstrip("/"))
    if not os.path.isfile(path):
        return False, f"file not found: {f}"
    try:
        text = open(path, errors="ignore").read()
    except OSError as e:
        return False, f"unreadable: {e}"
    if text.count(find) != 1:
        return False, f"find-block matched {text.count(find)}x (need exactly 1)"
    new_text = text.replace(find, repl, 1)
    # Deterministic compile pre-gate (added after the syntax audit found 6 damaged
    # files): a patch that breaks parsing is auto-rejected at zero LLM cost, and
    # the file is left untouched.
    if path.endswith(".py"):
        import ast as _ast
        try:
            _ast.parse(new_text)
        except SyntaxError as e:
            return False, f"REJECTED by compile pre-gate: SyntaxError line {e.lineno}: {e.msg}"
    open(path, "w").write(new_text)
    # Stage ONLY the patched file: `git add -A` would sweep our own tei/ artifacts
    # into the agent's history, and a later reset would then delete them.
    sh(["git", "add", "--", os.path.relpath(path, agent_dir)], cwd=agent_dir)
    r = sh(["git", "commit", "-m", f"tei-v7 {tag}: {str(cand.get('technique'))[:60]}"], cwd=agent_dir)
    return True, "applied+committed" if r.returncode == 0 else f"applied, commit said: {r.stdout[-80:]}"


def run_phase(agent_dir, ob, phase, baseline, probes, iters, batch, has_code, jsonl, tier,
              start=0, ledger=None, best0=None, arm=None):
    """One phase. `iters` is the TOTAL target for the phase; `start` is how many
    versions the immutable store already holds (extension mode resumes there).
    With a ledger, proposals follow the BCL generation contract."""
    versions = []
    best = best0 if best0 is not None else baseline["aggregate"]
    done = start
    while done < iters:
        n = min(batch, iters - done)
        cands = scored = None
        fams = None
        for attempt in range(3):
            try:
                if ledger is not None:
                    fams = ledger.choose_slots(n)
                    cands = gen_candidates_bcl(ob, phase, baseline, best, n, agent_dir,
                                               has_code, ledger, fams)[:n]
                else:
                    cands = gen_candidates(ob, phase, baseline, best, n, agent_dir, has_code)[:n]
                if not cands:
                    continue
                scored = eval_candidates(ob, phase, baseline, cands, probes)
                break
            except (ValueError, RuntimeError, AttributeError, TypeError, KeyError) as e:
                print(f"    .. batch attempt {attempt+1}/3 failed ({str(e)[:80]})", flush=True)
                cands = scored = None
        if not cands or scored is None:
            print(f"    !! batch failed 3x; stopping {phase} at {done}", flush=True)
            break
        by_i = {s.get("i"): s for s in scored if isinstance(s, dict)}
        n_written = 0
        for j, c in enumerate(cands):
            s = by_i.get(j) or {}
            dims = {k: clamp_score(v) for k, v in (s.get("dimensions") or {}).items()}
            agg = aggregate(dims)
            if agg is None:
                continue
            done += 1
            vid = f"{phase[:6]}-{done:02d}"
            applied, note = apply_patch(agent_dir, c, vid)
            rec = {
                "version_id": vid, "phase": phase, "technique": c.get("technique"),
                "target_failure_mode": c.get("target_failure_mode"),
                "expected_dimension": c.get("expected_dimension"),
                "evidence_instance_ids": [p["instance_id"] for p in probes],
                "patch_or_prompt": {"file": c.get("file"),
                                    "find": (c.get("find") or "")[:400],
                                    "replace": (c.get("replace") or "")[:400]},
                "dimensions": dims, "aggregate": agg,
                "probe_scores": [{**p, "score": clamp_score(p.get("score"))}
                                 for p in (s.get("probe_scores") or [])],
                "score_label": tier, "decision": "applied" if applied else "proposed_not_applied",
                "apply_note": note, "delta_vs_baseline": round(agg - baseline["aggregate"], 4),
                "why": s.get("why") or c.get("why") or "no why returned (recorded as such)",
            }
            if arm:
                rec["arm"] = arm
                rec["proposer"] = "bcl-thompson" if ledger is not None else "best-so-far"
            if fams:
                rec["bcl_family"] = fams[j] if j < len(fams) else None
            if ledger is not None:
                # Bayesian surprise computed at insertion, stored on the record (L3 input)
                rec["bayesian_surprise"] = round(ledger.ingest(rec), 5)
            versions.append(rec)
            jsonl.write(json.dumps(rec) + "\n")
            jsonl.flush()
            n_written += 1
            best = max(best, agg)
        if ledger is not None and n_written:
            ledger.update_lessons(ledger.records[-n_written:])  # one small luna call per batch
            ledger.save()
        print(f"    {phase}: {done}/{iters} versions, best={best:.4f} "
              f"[${BUDGET.nominal:.2f} nom/${BUDGET.conservative:.2f} cons]", flush=True)
    return versions


def confirm(best_rec, baseline):
    """Do-no-harm confirmation using the deployed methodology's gate."""
    sys.path.insert(0, os.path.expanduser("~/Documents/STANFORD/tei-loop"))
    from tei_loop.gate import verify_candidate, preflight_power  # noqa: E402
    ref = [p["score"] for p in baseline.get("probe_scores", []) if p.get("score") is not None]
    cand_map = {p["instance_id"]: p["score"] for p in (best_rec or {}).get("probe_scores", [])
                if p.get("score") is not None}
    cand = [cand_map.get(p["instance_id"]) for p in baseline.get("probe_scores", [])]
    pairs = [(c, r) for c, r in zip(cand, ref) if c is not None and r is not None]
    if not pairs:
        return {"accept": False, "reason": "no paired probe scores", "n_queries": 0}
    v = verify_candidate([c for c, _ in pairs], [r for _, r in pairs])
    v["preflight"] = preflight_power(len(pairs))
    return v


def noise_floor(ob, baseline, probes, agent_dir):
    """Paraphrase orbit: undirected rewordings, scored the same way."""
    ev = evidence_pack(agent_dir, ob)[:3000]
    # The orbit MUST be elicited with the same anchoring as candidate scoring, or the
    # floor is measured on a different scale than the gains it is meant to bound.
    prompt = f"""Produce 3 UNDIRECTED PARAPHRASES of this agent's operating instructions: reworded
only, targeting no failure mode, adding no capability, changing no behaviour. Then score each
on the TEI rubric using EXACTLY the calibration below.

SYSTEM: {ob['system']} ({ob['resolve_rate']}% resolved on {ob['split']}).
BASELINE dimensions (the unmodified system): {json.dumps(baseline.get('dimensions'))}
BASELINE aggregate: {baseline.get('aggregate')}

Calibration rules, identical to those used for real candidates:
- Score relative to that baseline, anchored to the real {ob['resolve_rate']}% resolve rate.
- A pure rewording changes behaviour very little: expect deltas near 0.00 (about -0.02..+0.02).
- No value may exceed 0.99, and none may approach 1.0.

EVIDENCE:\n{ev}

JSON: {{"paraphrases":[{{"paraphrase":"short description","dimensions":{{"target_alignment":0.0,"reasoning_soundness":0.0,"execution_accuracy":0.0,"output_integrity":0.0}}}}]}}"""
    try:
        out = call_json(prompt, max_out=4000)
    except (ValueError, RuntimeError):
        return None
    items = out if isinstance(out, list) else (out.get("paraphrases") or [])
    aggs = [aggregate({k: clamp_score(v) for k, v in (o.get("dimensions") or {}).items()})
            for o in items]
    aggs = [a for a in aggs if a is not None]
    return {"paraphrase_aggregates": aggs,
            "noise_floor": round(max(aggs) - baseline["aggregate"], 4) if aggs else None}


# -------------------------------------------------- Phase B: 100+100 extension
VAR0_DEFAULT = 0.00043609   # empirical global variance of delta_vs_baseline over
                            # the 1,140 prefix records (see BUDGET_100.md)
ABLATION_AGENTS = ["01_livesweagent", "04_acoder", "09_agentscope", "21_swerl",
                   "24_swefixer"]   # fixed: random.Random(42).sample(sorted(slugs), 5)


def load_candidates(tei_dir):
    """Load the immutable store. A process killed mid-write can leave ONE
    truncated final line; that line is quarantined to candidates.trimmed (the
    unfinished version simply re-runs). A bad line anywhere else is corruption
    and raises."""
    p = os.path.join(tei_dir, "candidates.jsonl")
    recs = []
    if not os.path.isfile(p):
        return recs
    lines = open(p).read().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                open(p + ".trimmed", "a").write(line + "\n")
                open(p, "w").write("\n".join(lines[:i]) + ("\n" if i else ""))
                print(f"    !! trimmed truncated final line in {p} (quarantined)", flush=True)
                break
            raise RuntimeError(f"corrupt non-final line {i + 1} in {p}")
    return recs


def run_extension(agent_dir, d, struct_target, prompt_target, batch, var0, use_bcl,
                  arm, result_name="result.json"):
    """Continue an agent's existing run (its immutable prefix) to the per-phase
    targets. Baseline, probes, and judge instrument are REUSED unchanged; only
    the proposer differs (BCL generation contract vs original best-so-far)."""
    tei_dir = os.path.join(agent_dir, "tei")
    ob = json.load(open(os.path.join(tei_dir, "onboarding.json")))
    base = json.load(open(os.path.join(tei_dir, "baseline_eval.json")))
    probes = base.get("probes") or probe_instances(ob, k=6)
    t = base.get("runnability") or tier_of(agent_dir, ob)
    tier = base.get("score_label", "PROXY")
    existing = load_candidates(tei_dir)
    ns = sum(1 for r in existing if r.get("phase") == "structural")
    np_ = sum(1 for r in existing if r.get("phase") == "prompt")
    if ns >= struct_target and np_ >= prompt_target:
        print(f"  {d}: already at target ({ns}+{np_}), skipping", flush=True)
        return None
    best0 = max([r["aggregate"] for r in existing if r.get("aggregate") is not None]
                + [base["aggregate"]])
    ledger = None
    if use_bcl:
        sys.path.insert(0, ROOT)
        from bcl import BayesianCreditLedger  # noqa: E402
        ledger = BayesianCreditLedger(
            tei_dir, weak_dim=base.get("weakest_dimension") or "execution_accuracy",
            var0=var0, seed=ob["rank"], call_json=call_json, embed=call_embed)
        for r in existing:          # deterministic bootstrap replay of the prefix
            ledger.ingest(r, log=False)
    print(f"  {d}: prefix {ns}+{np_} -> targets {struct_target}+{prompt_target} "
          f"(proposer={'bcl-thompson' if use_bcl else 'best-so-far'}, arm={arm})", flush=True)
    with open(os.path.join(tei_dir, "candidates.jsonl"), "a") as jl:
        sv = run_phase(agent_dir, ob, "structural", base, probes, struct_target, batch,
                       t["has_code"], jl, tier, start=ns, ledger=ledger, best0=best0, arm=arm)
        best0p = max([r["aggregate"] for r in sv] + [best0])
        pv = run_phase(agent_dir, ob, "prompt", base, probes, prompt_target, batch,
                       t["has_code"], jl, tier, start=np_, ledger=ledger, best0=best0p, arm=arm)
    if ledger is not None:
        ledger.save()

    allv = load_candidates(tei_dir)
    prior_path = os.path.join(tei_dir, "result.json")
    prior = json.load(open(prior_path)) if os.path.isfile(prior_path) else {}
    pre_path = os.path.join(tei_dir, "result_v7pre100.json")
    if prior and not os.path.isfile(pre_path):
        json.dump(prior, open(pre_path, "w"), indent=2)   # superseded result kept as labeled history
    svall = [r for r in allv if r.get("phase") == "structural" and r.get("aggregate") is not None]
    scorable = [r for r in allv if r.get("aggregate") is not None]
    best_struct = max(svall, key=lambda r: r["aggregate"]) if svall else None
    best = max(scorable, key=lambda r: r["aggregate"]) if scorable else None
    conf = confirm(best, base) if best else {"accept": False, "reason": "no versions"}
    shipped = best if (best and conf.get("accept")) else None
    result = {
        "agent": d, "system": ob["system"], "rank": ob["rank"],
        "resolve_rate": ob["resolve_rate"], "score_label": tier,
        "design": f"extension to {struct_target}+{prompt_target} ({arm})",
        "proposer": "bcl-thompson" if use_bcl else "best-so-far",
        "runnability": t,
        "baseline": base["aggregate"],
        "best_structural": best_struct["aggregate"] if best_struct else base["aggregate"],
        "best_structural_id": best_struct["version_id"] if best_struct else None,
        "best_final": best["aggregate"] if best else base["aggregate"],
        "best_final_id": best["version_id"] if best else None,
        "shipped": (shipped or {}).get("version_id") or "baseline (not confirmed better)",
        "confirmation": conf, "noise_floor": prior.get("noise_floor"),
        "n_versions": len(allv),
        "n_applied": sum(1 for r in allv if r.get("decision") == "applied"),
        "budget_after": BUDGET.as_dict(),
    }
    json.dump(result, open(os.path.join(tei_dir, result_name), "w"), indent=2)
    return result


def extend_main(a):
    state = json.load(open(STATE_PATH)) if os.path.isfile(STATE_PATH) else {"agents": {}}
    dirs = sorted(d for d in os.listdir(AGENTS) if os.path.isdir(os.path.join(AGENTS, d)))
    if a.only:
        dirs = [d for d in dirs if a.only in d]
    if a.range:
        lo, hi = (int(x) for x in a.range.split("-"))
        dirs = [d for d in dirs if lo <= int(d.split("_")[0]) <= hi]
    print(f"extension shard: {len(dirs)} agents -> {a.struct_iters}+{a.prompt_iters}, "
          f"BCL={a.bcl}, state {os.path.basename(STATE_PATH)}", flush=True)
    for d in dirs:
        try:
            result = run_extension(os.path.join(AGENTS, d), d, a.struct_iters, a.prompt_iters,
                                   a.batch, a.var0, use_bcl=(a.bcl == "on"),
                                   arm="bcl-extension" if a.bcl == "on" else "extension-bestsofar")
        except SystemExit:
            raise                       # fatal API/key errors must stop the shard
        except Exception as e:          # one agent's malformed output must not kill the shard
            print(f"  !! {d} extension failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
            continue
        if result:
            state["agents"][d] = {"done": True, **{k: result[k] for k in
                                  ("baseline", "best_structural", "best_final", "shipped", "score_label")}}
        state["budget"] = BUDGET.as_dict()
        json.dump(state, open(STATE_PATH, "w"), indent=2)
        print(f"  == shard spend so far: ${BUDGET.nominal:.2f} nom / "
              f"${BUDGET.conservative:.2f} cons ({BUDGET.calls} calls, "
              f"{BUDGET.emb_calls} embed)", flush=True)
    print("\nSHARD BUDGET:", json.dumps(BUDGET.as_dict(), indent=2), flush=True)


def ablation_setup():
    """Create isolated git worktrees for the 5 ablation agents BEFORE any main
    shard extends them, freezing the shared prefix (repo state + candidates)."""
    abl_root = os.path.join(ROOT, "ablation")
    os.makedirs(abl_root, exist_ok=True)
    for d in ABLATION_AGENTS:
        src = os.path.join(AGENTS, d)
        wt = os.path.join(abl_root, d)
        tei_src, tei_wt = os.path.join(src, "tei"), os.path.join(wt, "tei")
        if not os.path.isdir(wt):
            r = sh(["git", "worktree", "add", "--detach", wt, "HEAD"], cwd=src)
            if r.returncode != 0:
                print(f"  !! worktree add failed for {d}: {(r.stderr or '')[-200:]}", flush=True)
                continue
        if not os.path.isfile(os.path.join(tei_wt, "prefix_counts.json")):
            os.makedirs(tei_wt, exist_ok=True)
            for fn in ("onboarding.json", "baseline_eval.json", "candidates.jsonl", "result.json"):
                p = os.path.join(tei_src, fn)
                if os.path.isfile(p):
                    open(os.path.join(tei_wt, fn), "w").write(open(p).read())
            ex = load_candidates(tei_wt)
            json.dump({"structural": sum(1 for r in ex if r.get("phase") == "structural"),
                       "prompt": sum(1 for r in ex if r.get("phase") == "prompt")},
                      open(os.path.join(tei_wt, "prefix_counts.json"), "w"))
        print(f"  ablation worktree ready: {d}", flush=True)


def ablation_main(a):
    """BCL-off comparison arm: 5 fixed-seed agents, +30 structural +30 prompt on
    top of the same prefix, original best-so-far proposer, isolated worktrees."""
    ablation_setup()
    if a.setup_only:
        return
    state = json.load(open(STATE_PATH)) if os.path.isfile(STATE_PATH) else {"agents": {}}
    for d in ABLATION_AGENTS:
        wt = os.path.join(ROOT, "ablation", d)
        pc_path = os.path.join(wt, "tei", "prefix_counts.json")
        if not os.path.isfile(pc_path):
            print(f"  !! {d}: no worktree/prefix counts; skipping", flush=True)
            continue
        pc = json.load(open(pc_path))
        try:
            result = run_extension(wt, d, pc["structural"] + 30, pc["prompt"] + 30, a.batch,
                                   a.var0, use_bcl=False, arm="ablation-bestsofar",
                                   result_name="ablation_result.json")
        except (ValueError, RuntimeError, OSError) as e:
            print(f"  !! {d} ablation failed: {str(e)[:160]}", flush=True)
            continue
        if result:
            state["agents"][d] = {"done": True, "arm": "ablation", **{k: result[k] for k in
                                  ("baseline", "best_structural", "best_final", "shipped", "score_label")}}
        state["budget"] = BUDGET.as_dict()
        json.dump(state, open(STATE_PATH, "w"), indent=2)
        print(f"  == ablation spend so far: ${BUDGET.nominal:.2f} nom / "
              f"${BUDGET.conservative:.2f} cons", flush=True)
    print("\nABLATION BUDGET:", json.dumps(BUDGET.as_dict(), indent=2), flush=True)


# --------------------------------------------------------------------- main
def tier_of(agent_dir, ob):
    """Honest runnability check: no Docker, no GPU, a real entry point, code present."""
    has_code = any(os.path.splitext(f)[1] in (".py", ".ts", ".java")
                   for _, dr, fs in os.walk(agent_dir) for f in fs
                   if ".git" not in _ ) if os.path.isdir(agent_dir) else False
    blob = ""
    for n in ("README.md", "README_EN.md"):
        blob += read_head(os.path.join(agent_dir, n), 6000).lower()
    needs_docker = "docker" in blob
    needs_gpu = any(w in blob for w in ("cuda", "gpu", "vllm", "a100", "h100"))
    return {"has_code": has_code, "needs_docker": needs_docker, "needs_gpu": needs_gpu,
            "tier": "A-candidate" if (has_code and not needs_docker and not needs_gpu) else "PROXY",
            "reason": ("docker required" if needs_docker else
                       "gpu/local-weights required" if needs_gpu else
                       "no source code in linked repo" if not has_code else
                       "cli entry point plausible; execution still requires SWE-bench task infra")}


def main():
    global BUDGET_CAP, STATE_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct-iters", type=int, default=30)
    ap.add_argument("--prompt-iters", type=int, default=30)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--probes", type=int, default=6)
    ap.add_argument("--only", default=None)
    ap.add_argument("--state", default=STATE_PATH,
                    help="state file (shards use separate files to avoid clobbering)")
    ap.add_argument("--cap", type=float, default=BUDGET_CAP,
                    help="this process's slice of the global budget cap")
    ap.add_argument("--range", default=None, help="inclusive rank range, e.g. 02-11")
    ap.add_argument("--extend", action="store_true",
                    help="Phase B: continue existing runs (prefix) to the per-phase targets")
    ap.add_argument("--ablation-arm", action="store_true",
                    help="Phase B mini-ablation: 5 fixed agents, +30/+30, BCL off, worktrees")
    ap.add_argument("--setup-only", action="store_true",
                    help="with --ablation-arm: create the worktrees (freeze the prefix) and exit")
    ap.add_argument("--bcl", choices=["on", "off"], default="on",
                    help="proposer for --extend: BCL generation contract vs original best-so-far")
    ap.add_argument("--var0", type=float, default=VAR0_DEFAULT,
                    help="BCL magnitude-head prior scale (empirical global delta variance)")
    a = ap.parse_args()

    BUDGET_CAP, STATE_PATH = a.cap, a.state
    if a.ablation_arm:
        return ablation_main(a)
    if a.extend:
        return extend_main(a)

    state = json.load(open(STATE_PATH)) if os.path.isfile(STATE_PATH) else {"agents": {}}
    dirs = sorted(d for d in os.listdir(AGENTS) if os.path.isdir(os.path.join(AGENTS, d)))
    if a.only:
        dirs = [d for d in dirs if a.only in d]
    if a.range:
        lo, hi = (int(x) for x in a.range.split("-"))
        dirs = [d for d in dirs if lo <= int(d.split("_")[0]) <= hi]
    # already-finished agents (any shard) are skipped
    dirs = [d for d in dirs if not os.path.isfile(os.path.join(AGENTS, d, "tei", "result.json"))]
    print(f"shard: {len(dirs)} agents, cap ${BUDGET_CAP:.2f}, state {os.path.basename(STATE_PATH)}", flush=True)

    struct_iters, prompt_iters, probes_n = a.struct_iters, a.prompt_iters, a.probes
    for idx, d in enumerate(dirs):
        agent_dir = os.path.join(AGENTS, d)
        ob = json.load(open(os.path.join(agent_dir, "tei", "onboarding.json")))
        if state["agents"].get(d, {}).get("done"):
            print(f"[{idx+1}/{len(dirs)}] {d}: already done, skipping", flush=True)
            continue

        # budget projection & uniform scale-down
        spent = BUDGET.conservative
        remaining_agents = len(dirs) - idx
        if idx > 0:
            per = spent / idx
            projected = spent + per * remaining_agents
            # Step down as many levels as the projection needs, in the pre-registered
            # order (instances per evaluation first, then iterations per phase), and
            # apply the reduced setting uniformly to all remaining agents.
            while projected > BUDGET_CAP:
                scale = None
                if probes_n > 4:
                    probes_n, scale = 4, f"probes 6->4 (projected ${projected:.2f})"
                elif struct_iters > 18:
                    struct_iters = prompt_iters = 18
                    scale = f"iters 30->18 (projected ${projected:.2f})"
                elif struct_iters > 12:
                    struct_iters = prompt_iters = 12
                    scale = f"iters 18->12 (projected ${projected:.2f})"
                elif struct_iters > 6:
                    struct_iters = prompt_iters = 6
                    scale = f"iters 12->6 (projected ${projected:.2f})"
                if scale is None:
                    break
                BUDGET.scale_downs.append(f"{scale} before {d}")
                print(f"  ** scale-down: {BUDGET.scale_downs[-1]}", flush=True)
                # each level roughly halves the per-agent version count
                projected = spent + (per * remaining_agents) * 0.55 ** len(BUDGET.scale_downs)
            if spent >= BUDGET_CAP:
                print(f"!! budget cap reached (${spent:.2f}); stopping before {d}", flush=True)
                break

        t = tier_of(agent_dir, ob)
        tier = "PROXY"  # execution substrate is only VERIFIED if the agent actually runs
        probes = probe_instances(ob, k=probes_n)
        print(f"[{idx+1}/{len(dirs)}] {d}  ({ob['resolve_rate']}%)  tier={t['tier']} ({t['reason']})", flush=True)

        try:
            base = baseline_eval(agent_dir, ob, probes)
        except (ValueError, RuntimeError) as e:
            print(f"  !! baseline failed: {e}", flush=True)
            continue
        base.update(score_label=tier, probes=probes, runnability=t)
        json.dump(base, open(os.path.join(agent_dir, "tei", "baseline_eval.json"), "w"), indent=2)
        print(f"  baseline={base['aggregate']} weakest={base.get('weakest_dimension')}", flush=True)

        with open(os.path.join(agent_dir, "tei", "candidates.jsonl"), "w") as jl:
            sv = run_phase(agent_dir, ob, "structural", base, probes, struct_iters,
                           a.batch, t["has_code"], jl, tier)
            best_struct = max(sv, key=lambda r: r["aggregate"]) if sv else None
            bs_agg = best_struct["aggregate"] if best_struct else base["aggregate"]
            pv = run_phase(agent_dir, ob, "prompt", base, probes, prompt_iters,
                           a.batch, t["has_code"], jl, tier)

        allv = sv + pv
        best = max(allv, key=lambda r: r["aggregate"]) if allv else None
        conf = confirm(best, base) if best else {"accept": False, "reason": "no versions"}
        nf = noise_floor(ob, base, probes, agent_dir)
        shipped = best if (best and conf.get("accept")) else None

        result = {
            "agent": d, "system": ob["system"], "rank": ob["rank"],
            "resolve_rate": ob["resolve_rate"], "score_label": tier,
            "runnability": t,
            "baseline": base["aggregate"],
            "best_structural": bs_agg,
            "best_structural_id": best_struct["version_id"] if best_struct else None,
            "best_final": best["aggregate"] if best else base["aggregate"],
            "best_final_id": best["version_id"] if best else None,
            "shipped": (shipped or {}).get("version_id") or "baseline (not confirmed better)",
            "confirmation": conf, "noise_floor": nf,
            "n_versions": len(allv),
            "n_applied": sum(1 for r in allv if r["decision"] == "applied"),
            "budget_after": BUDGET.as_dict(),
        }
        json.dump(result, open(os.path.join(agent_dir, "tei", "result.json"), "w"), indent=2)
        state["agents"][d] = {"done": True, **{k: result[k] for k in
                              ("baseline", "best_structural", "best_final", "shipped", "score_label")}}
        state["budget"] = BUDGET.as_dict()
        json.dump(state, open(STATE_PATH, "w"), indent=2)
        print(f"  => baseline {result['baseline']} -> struct {result['best_structural']} "
              f"-> final {result['best_final']} | ship={result['shipped']} "
              f"| ${BUDGET.conservative:.2f} cons / ${BUDGET.nominal:.2f} nom", flush=True)

    state["budget"] = BUDGET.as_dict()
    json.dump(state, open(STATE_PATH, "w"), indent=2)
    print("\nBUDGET:", json.dumps(BUDGET.as_dict(), indent=2))


if __name__ == "__main__":
    main()
