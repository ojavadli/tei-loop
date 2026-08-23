#!/usr/bin/env python3
"""Blinded A/B validation of the applied TEI-v7 changes.

For each agent with >=1 applied patch: reconstruct the REAL before/after file
states (baseline SHA vs tei-v7 HEAD), present both to gpt-5.6-luna in random
order with NO why-record narrative, no scores, and no statement of which is
modified, and ask which state would more effectively resolve SWE-bench issues.
k independent repeats per agent -> per-agent blinded preference + retest
consistency. This controls the two biases the rubric pass could not:
narrative anchoring and knowing-which-is-new.
"""
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.expanduser("~/swebench-agents"))
from tei_pipeline import call_json, BUDGET  # same OpenAI-only, gpt-5.6-luna path

ROOT = os.path.expanduser("~/swebench-agents")
AGENTS = os.path.join(ROOT, "agents")
K = 5
MAX_FILES = 4
CTX = 12  # context lines around changes
rng = random.Random(0)


def sh(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def changed_files(repo, base_sha):
    out = sh(["git", "diff", "--numstat", f"{base_sha}..HEAD"], repo).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0].isdigit():
            rows.append((int(parts[0]) + int(parts[1]), parts[2]))
    rows.sort()  # smallest churn first: fits context, hardest to spot by size
    return [p for _, p in rows[:MAX_FILES]]


def excerpt_pair(repo, base_sha, path):
    """Aligned before/after excerpts around the changed hunks of one file."""
    diff = sh(["git", "diff", f"-U{CTX}", f"{base_sha}..HEAD", "--", path], repo).stdout
    if not diff:
        return None
    before, after = [], []
    for line in diff.splitlines():
        if line.startswith(("---", "+++", "diff ", "index ", "@@")):
            if line.startswith("@@"):
                before.append("  [...]")
                after.append("  [...]")
            continue
        if line.startswith("-"):
            before.append(line[1:])
        elif line.startswith("+"):
            after.append(line[1:])
        else:
            before.append(line[1:] if line else line)
            after.append(line[1:] if line else line)
    return "\n".join(before)[:2600], "\n".join(after)[:2600]


def main():
    only = set(sys.argv[1:])          # agent dirs to (re-)run; empty = all
    prior = {}
    out_path = os.path.join(ROOT, "blind_reval.json")
    if only and os.path.isfile(out_path):
        prior = {r["agent"]: r for r in json.load(open(out_path))["results"]}
    results = []
    dirs = sorted(d for d in os.listdir(AGENTS) if os.path.isdir(os.path.join(AGENTS, d)))
    if only:
        dirs = [d for d in dirs if d in only]
    for d in dirs:
        repo = os.path.join(AGENTS, d)
        res = json.load(open(os.path.join(repo, "tei", "result.json")))
        ob = json.load(open(os.path.join(repo, "tei", "onboarding.json")))
        base_sha = ob["repo_sha"]
        if res.get("n_applied", 0) == 0:
            results.append({"agent": d, "skipped": "no applied patches (nothing real to compare)"})
            print(f"{d:30s} SKIP no applied patches", flush=True)
            continue
        files = changed_files(repo, base_sha)
        pairs = []
        for p in files:
            e = excerpt_pair(repo, base_sha, p)
            if e:
                pairs.append((p, e[0], e[1]))
        if not pairs:
            results.append({"agent": d, "skipped": "diff empty/unreadable"})
            continue

        probes = [p["instance_id"] for p in (json.load(open(os.path.join(repo, "tei", "baseline_eval.json")))
                                             .get("probes") or [])][:4]
        votes = []
        for k in range(K):
            flip = rng.random() < 0.5  # True: version 1 = patched
            blocks = []
            for path, bef, aft in pairs:
                v1, v2 = (aft, bef) if flip else (bef, aft)
                blocks.append(f"### file: {path}\n--- VERSION 1 ---\n{v1}\n--- VERSION 2 ---\n{v2}")
            prompt = f"""You are comparing two versions of the same software-engineering agent system
({ob['system']}; its job: resolve real GitHub issues, SWE-bench style). Below, for each shown file,
are the two versions' contents in the regions where they differ. You are NOT told which version is
older; judge only what you see.

Representative task instances this agent faces: {', '.join(probes)}.

{chr(10).join(blocks)}

Which version is more likely to resolve such issues correctly end-to-end? Consider robustness,
verification behavior, and failure handling actually visible in the text. If the differences are
too trivial or ambiguous to matter, say tie.

Return ONLY JSON: {{"better": "1" | "2" | "tie", "confidence": 0.0, "reason": "one sentence"}}"""
            try:
                v = call_json(prompt, max_out=1200)
            except Exception as e:
                votes.append({"error": str(e)[:80]})
                continue
            pick = str(v.get("better", "")).strip()
            patched_pick = ("1" if flip else "2")
            votes.append({"vote": ("patched" if pick == patched_pick else
                                   "baseline" if pick in ("1", "2") else "tie"),
                          "confidence": v.get("confidence"),
                          "reason": str(v.get("reason", ""))[:160],
                          "patched_shown_as": "1" if flip else "2"})
        pv = sum(1 for v in votes if v.get("vote") == "patched")
        bv = sum(1 for v in votes if v.get("vote") == "baseline")
        tv = sum(1 for v in votes if v.get("vote") == "tie")
        results.append({"agent": d, "system": ob["system"], "files_shown": [p for p, _, _ in pairs],
                        "n_applied": res["n_applied"], "rubric_delta": res.get("shipped_delta"),
                        "votes": votes, "patched": pv, "baseline": bv, "tie": tv})
        print(f"{d:30s} patched {pv} / baseline {bv} / tie {tv}   [${BUDGET.conservative:.2f} cons]",
              flush=True)

    if only:                          # merge re-runs into the existing record
        for r in results:
            r["post_repair"] = True
            prior[r["agent"]] = r
        results = [prior[k] for k in sorted(prior)]
    json.dump({"k": K, "results": results, "budget": BUDGET.as_dict()},
              open(os.path.join(ROOT, "blind_reval.json"), "w"), indent=2)
    done = [r for r in results if "votes" in r]
    maj_p = sum(1 for r in done if r["patched"] > r["baseline"] + r["tie"])
    maj_b = sum(1 for r in done if r["baseline"] > r["patched"] + r["tie"])
    print(f"\nagents evaluated: {len(done)} | strict-majority patched: {maj_p} | "
          f"strict-majority baseline: {maj_b} | rest mixed/tie")
    print(f"budget: {json.dumps(BUDGET.as_dict())}")


if __name__ == "__main__":
    main()
