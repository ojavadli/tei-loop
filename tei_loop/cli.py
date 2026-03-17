"""
TEI CLI.

Usage:
    pip install tei-loop
    python3 -m tei_loop your_agent.py

TEI handles everything else.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

SKIP_NAMES = {
    "load_dotenv", "OpenAI", "AsyncOpenAI", "FastAPI", "Request", "Response",
    "File", "Form", "Cookie", "HTMLResponse", "JSONResponse", "RedirectResponse",
    "UploadFile", "HTTPException", "Base", "asynccontextmanager", "contextmanager",
    "create_engine", "sessionmaker", "declarative_base", "DictLoader", "Environment",
    "app", "router", "client", "session", "engine", "render_template",
    "init_db", "lifespan", "health", "home", "login", "logout",
    "verify_session", "verify_credentials", "create_session_token",
    "save_interview_to_db", "get_interview_from_db", "get_all_completed_interviews",
}

FASTAPI_PARAM_TYPES = {"Request", "UploadFile", "Cookie", "File", "Form", "Response"}


def _find_agent_function(agent_file: str) -> tuple[Callable, str]:
    """Find the actual agent function — the one that calls an LLM."""
    path = Path(agent_file).resolve()
    if not path.exists():
        print(f"{RED}Error: File not found: {agent_file}{RESET}")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("agent_module", path)
    module = importlib.util.module_from_spec(spec)

    orig_dir = os.getcwd()
    os.chdir(path.parent)
    orig_argv = sys.argv
    sys.argv = [str(path)]
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass
    except Exception as e:
        print(f"{YELLOW}Note: Module raised {type(e).__name__} during import{RESET}")
    finally:
        sys.argv = orig_argv
        os.chdir(orig_dir)

    all_fns = []
    for name, obj in vars(module).items():
        if (not callable(obj) or name.startswith("_") or isinstance(obj, type)
                or inspect.isclass(obj) or name in SKIP_NAMES):
            continue

        source = ""
        try:
            source = inspect.getsource(obj)
        except (OSError, TypeError):
            continue

        source_lower = source.lower()

        calls_llm = any(kw in source_lower for kw in [
            "chat.completions.create", "messages.create",
            "generate_content",
        ])
        if not calls_llm:
            continue

        try:
            sig = inspect.signature(obj)
        except (ValueError, TypeError):
            continue

        params = list(sig.parameters.values())
        required = [p for p in params
                    if p.default is inspect.Parameter.empty and p.name != "self"]

        has_fastapi_params = any(
            p.annotation.__name__ in FASTAPI_PARAM_TYPES
            for p in params
            if hasattr(p.annotation, "__name__")
        )
        if has_fastapi_params:
            continue

        has_return = "return " in source and "return None" not in source

        score = 10
        if len(required) == 1:
            score += 15
        elif len(required) == 0:
            score += 5
        else:
            score -= len(required) * 3
        if has_return:
            score += 5
        if name in ("agent", "run", "main", "invoke", "execute", "summarize",
                     "analyze", "predict", "generate"):
            score += 8

        all_fns.append((name, obj, score, required))

    if not all_fns:
        print(f"{RED}Error: No LLM-calling function found in {agent_file}{RESET}")
        print(f"{DIM}TEI looks for functions that call OpenAI/Anthropic/Google.{RESET}")
        sys.exit(1)

    all_fns.sort(key=lambda x: x[2], reverse=True)
    best_name, best_fn, best_score, best_required = all_fns[0]

    if len(best_required) > 1:
        wrapper = _create_wrapper(best_fn, best_name, best_required, agent_file)
        if wrapper:
            return wrapper, best_name
        print(f"{YELLOW}Function '{best_name}' takes {len(best_required)} params: "
              f"{[p.name for p in best_required]}{RESET}")
        print(f"{YELLOW}TEI will try to call it with a single text input.{RESET}")

    return best_fn, best_name


def _create_wrapper(fn: Callable, fn_name: str, required_params: list, agent_file: str) -> Optional[Callable]:
    """Create a wrapper that maps a single text input to a multi-param function."""
    source = ""
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return None

    param_names = [p.name for p in required_params]

    if "session" in param_names and ("responses" in source or "transcript" in source):
        def wrapper(query: str) -> Any:
            lines = query.strip().split("\n")
            responses = []
            q_num = 0
            current_q = ""
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.upper().startswith(("Q", "QUESTION")):
                    q_num += 1
                    current_q = re.sub(r'^Q\d*[:\.]?\s*', '', line, flags=re.IGNORECASE)
                elif line.upper().startswith(("A", "ANSWER")):
                    answer = re.sub(r'^A\d*[:\.]?\s*', '', line, flags=re.IGNORECASE)
                    responses.append({
                        "question_number": q_num,
                        "question": current_q or f"Question {q_num}",
                        "answer": answer,
                    })

            if not responses:
                responses = [{"question_number": 1, "question": "Input", "answer": query}]

            session = {
                "id": "tei-test-001",
                "responses": responses,
                "interview_id": "tei-test-001",
                "status": "completed",
                "participant_name": "TEI Test",
                "completed": True,
                "current_question": len(responses),
                "summary": None,
            }

            id_param = next((p for p in param_names if "id" in p.lower()), None)
            session_param = next((p for p in param_names if "session" in p.lower()), None)

            kwargs = {}
            for p in required_params:
                if "id" in p.name.lower():
                    kwargs[p.name] = "tei-test-001"
                elif "session" in p.name.lower():
                    kwargs[p.name] = session
                else:
                    kwargs[p.name] = query

            try:
                if inspect.iscoroutinefunction(fn):
                    import asyncio as _aio
                    _loop = _aio.new_event_loop()
                    try:
                        result = _loop.run_until_complete(fn(**kwargs))
                    finally:
                        _loop.close()
                else:
                    result = fn(**kwargs)
            except Exception:
                pass

            summary = kwargs.get(next((p for p in param_names if "session" in p.lower()), ""), {}).get("summary")
            if summary:
                return summary
            if isinstance(result, str):
                return result
            if isinstance(result, dict):
                return result.get("summary", str(result))
            return str(result) if result else summary or "No output generated"

        return wrapper

    if len(required_params) == 2:
        first_name = required_params[0].name.lower()
        second_name = required_params[1].name.lower()

        if any(kw in first_name for kw in ["id", "key", "name", "type"]):
            def wrapper(query: str) -> Any:
                kwargs = {required_params[0].name: "tei-test", required_params[1].name: query}
                if inspect.iscoroutinefunction(fn):
                    import asyncio as _aio
                    loop = _aio.new_event_loop()
                    try:
                        return loop.run_until_complete(fn(**kwargs))
                    finally:
                        loop.close()
                return fn(**kwargs)
            return wrapper

    return None


def _auto_generate_query(agent_fn: Callable, fn_name: str, agent_file: str) -> str:
    source = ""
    try:
        source = inspect.getsource(agent_fn)
    except (OSError, TypeError):
        try:
            source = Path(agent_file).read_text()
        except Exception:
            pass
    source_lower = source.lower()

    if "interview" in source_lower or "transcript" in source_lower:
        return (
            "Q1: Tell me about yourself.\n"
            "A1: I'm Alex, 30, software engineer. I enjoy cooking and hiking.\n\n"
            "Q2: What does a typical day look like?\n"
            "A2: Wake up at 7, coffee, work from 9-5, gym after work, dinner at 7pm.\n\n"
            "Q3: What are your main goals right now?\n"
            "A3: Getting better at system design and training for a half marathon.\n\n"
            "Q4: What are your go-to snacks?\n"
            "A4: Protein bars, almonds, Greek yogurt. Sometimes chips when I'm stressed.\n\n"
            "Q5: How do you shop for food?\n"
            "A5: Mostly Whole Foods, I check nutrition labels. If protein is high, I buy it."
        )
    if "summar" in source_lower:
        return (
            "The global renewable energy market reached $1.2 trillion in 2025, "
            "driven by solar and wind installations. China led with 45% of new capacity. "
            "Key challenges include grid storage and supply chain constraints."
        )
    if "classif" in source_lower or "sentiment" in source_lower:
        return "The product arrived damaged and customer support has been unresponsive for three days."
    return (
        "Analyze the following: AI agents are becoming mainstream in enterprise workflows. "
        "Companies reported a 40% increase in automation but also cited quality and reliability "
        "as top concerns. What are the key implications?"
    )


def _generate_test_queries(
    base_query: str, agent_fn: Callable, fn_name: str, agent_file: str
) -> list[str]:
    queries = [base_query]
    variations = [
        base_query + "\n\nConsider edge cases and alternative interpretations.",
        "Alternative scenario: " + base_query[:200] + ("..." if len(base_query) > 200 else ""),
        "Focus on clarity and completeness: " + base_query[:150] + ("..." if len(base_query) > 150 else ""),
        "Stress test: " + base_query[:150] + ("..." if len(base_query) > 150 else ""),
    ]
    for v in variations:
        if v not in queries and len(v) > 10:
            queries.append(v)
        if len(queries) >= 5:
            break
    while len(queries) < 5:
        queries.append(base_query)
        if len(queries) >= 5:
            break
    return queries[:5]


def _print_full_result(result: "TEIFullResult") -> None:
    from .models import TEIFullResult, Dimension

    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}TEI Full Result (8-Step Flow){RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}")

    print(f"\n  {BOLD}8 Steps Summary:{RESET}")
    print(f"    1. Checkpoints: {len(result.checkpoints)} placed")
    print(f"    2. Baseline eval: {'done' if result.baseline_eval else 'N/A'}")
    print(f"    3. Structural fixes: {len(result.structural_fixes)} proposed")
    print(f"    4. Middle eval: {'done' if result.middle_eval else 'N/A'}")
    print(f"    5. Metrics: {len(result.metrics)} approved")
    print(f"    6. Baseline prompt scores: {len(result.baseline_prompt_scores)} measured")
    print(f"    7. Optimization: {result.optimization.total_iterations if result.optimization else 0} iterations")
    print(f"    8. Final eval: {'done' if result.final_eval else 'N/A'}")

    if result.baseline_eval and result.final_eval:
        print(f"\n  {BOLD}Final Comparison:{RESET}")
        print(f"    {'Dimension':<28} {'Baseline':<10} {'Final':<10} {'Delta':<8}")
        print(f"    {'-'*56}")
        for dim in Dimension:
            base_ds = result.baseline_eval.dimension_scores.get(dim)
            final_ds = result.final_eval.dimension_scores.get(dim)
            base_s = base_ds.score if base_ds else 0
            final_s = final_ds.score if final_ds else 0
            delta = final_s - base_s
            dc = GREEN if delta > 0 else RED if delta < 0 else YELLOW
            bc = GREEN if base_s >= 0.7 else YELLOW if base_s >= 0.5 else RED
            fc = GREEN if final_s >= 0.7 else YELLOW if final_s >= 0.5 else RED
            print(f"    {dim.value:<28} {bc}{base_s:.2f}{RESET}      {fc}{final_s:.2f}{RESET}      {dc}{delta:+.2f}{RESET}")

        base_agg = result.baseline_eval.aggregate_score
        final_agg = result.final_eval.aggregate_score
        imp = final_agg - base_agg
        impc = GREEN if imp > 0 else RED if imp < 0 else YELLOW
        print(f"    {'AGGREGATE':<28} {base_agg:.2f}      {final_agg:.2f}      {impc}{imp:+.2f}{RESET}")

    if result.optimization:
        opt = result.optimization
        print(f"\n  {BOLD}Optimization Improvement:{RESET}")
        if opt.baseline_scores and opt.final_scores:
            for name, base in opt.baseline_scores.items():
                final = opt.final_scores.get(name, base)
                d = final - base
                c = GREEN if d > 0 else RED if d < 0 else YELLOW
                print(f"    {name}: {base:.2f} -> {final:.2f} ({c}{d:+.2f}{RESET})")
        print(f"    Iterations: {opt.total_iterations}")

    print(f"\n  {BOLD}Cost & Duration:{RESET}")
    print(f"    Duration: {result.total_duration_ms / 1000:.1f}s")
    print(f"    Cost:     ${result.total_cost_usd:.4f}")

    print(f"{CYAN}{'=' * 60}{RESET}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tei",
        description="TEI: Target -> Evaluate -> Improve your AI agent",
        usage="python3 -m tei_loop <agent_file.py>",
    )
    parser.add_argument("agent_file", nargs="?", help="Path to your agent Python file")
    parser.add_argument("--query", "-q", help="Custom test query")
    parser.add_argument("--function", "-f", help="Specific function name to test")
    parser.add_argument("--iterations", "-i", type=int, default=30, help="Optimization iterations")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--non-interactive", action="store_true", help="Skip Y/N prompts, auto-approve")

    args = parser.parse_args()

    if not args.agent_file:
        py_files = [f for f in Path(".").glob("*.py")
                    if not f.name.startswith("_") and not f.name.startswith("test_")]
        if len(py_files) == 1:
            args.agent_file = str(py_files[0])
        else:
            print(f"\n{BOLD}Usage:{RESET} python3 -m tei_loop <agent_file.py>\n")
            if py_files:
                print("Python files in current directory:")
                for f in sorted(py_files):
                    print(f"  {f.name}")
            sys.exit(0)

    from .loop import TEILoop

    agent_path = Path(args.agent_file).resolve()

    for env_file in [agent_path.parent / ".env", Path.cwd() / ".env"]:
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file, override=True)
            except ImportError:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, _, val = line.partition("=")
                            val = val.strip().strip("'\"")
                            os.environ.setdefault(key.strip(), val)
            break

    results_dir = agent_path.parent / "tei-results"
    results_dir.mkdir(exist_ok=True)

    print(f"\n{BOLD}{CYAN}TEI Loop{RESET} - Target, Evaluate, Improve\n")

    api_keys_found = []
    for provider, env_var in [("OpenAI", "OPENAI_API_KEY"), ("Anthropic", "ANTHROPIC_API_KEY"), ("Google", "GOOGLE_API_KEY")]:
        key = os.environ.get(env_var, "")
        if key:
            api_keys_found.append(f"{provider} ({env_var}={key[:8]}...)")
    if api_keys_found:
        print(f"  {GREEN}API keys: {', '.join(api_keys_found)}{RESET}")
    else:
        print(f"  {RED}No API keys found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY{RESET}")
        sys.exit(1)

    if args.function:
        spec = importlib.util.spec_from_file_location("agent_module", agent_path)
        module = importlib.util.module_from_spec(spec)
        orig_dir = os.getcwd()
        os.chdir(agent_path.parent)
        if str(agent_path.parent) not in sys.path:
            sys.path.insert(0, str(agent_path.parent))
        try:
            spec.loader.exec_module(module)
        except (SystemExit, Exception):
            pass
        finally:
            os.chdir(orig_dir)
        fn = getattr(module, args.function, None)
        if not fn:
            print(f"{RED}Function '{args.function}' not found.{RESET}")
            sys.exit(1)
        agent_fn, fn_name = fn, args.function
    else:
        agent_fn, fn_name = _find_agent_function(args.agent_file)

    base_query = args.query or _auto_generate_query(agent_fn, fn_name, args.agent_file)
    if args.query:
        test_queries = [args.query]
    else:
        test_queries = _generate_test_queries(base_query, agent_fn, fn_name, args.agent_file)

    print(f"  Agent:      {agent_path.name}")
    print(f"  Function:   {fn_name}()")
    print(f"  Query:      {base_query[:60]}{'...' if len(base_query) > 60 else ''}")
    print(f"  Test batch: {len(test_queries)} queries")
    print(f"  Results:    tei-results/")
    print()

    interactive = not args.non_interactive
    loop = TEILoop(
        agent=agent_fn,
        verbose=args.verbose,
        agent_file=str(agent_path),
        interactive=interactive,
        num_iterations=args.iterations,
    )
    result = asyncio.run(loop.run(base_query, test_queries=test_queries))
    _print_full_result(result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"run_{timestamp}.json"
    result_file.write_text(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
    print(f"{GREEN}Results saved to tei-results/run_{timestamp}.json{RESET}")

    (results_dir / "latest.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, default=str)
    )


if __name__ == "__main__":
    main()
