"""
AST-based multi-file scanner for TEI checkpoints.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

from .models import Checkpoint, Dimension


_PROMPT_VAR_NAMES = frozenset(
    {"system_prompt", "user_prompt", "instruction", "system_message", "system_msg"}
)

_LLM_CALL_ATTRS = frozenset(
    {"create", "acreate", "generate_content", "invoke", "generate"}
)

_LLM_CLIENT_NAMES = frozenset(
    {"chat", "completions", "messages", "client", "model", "llm", "openai", "anthropic"}
)

_TOOL_DECORATORS = frozenset({"tool", "tools", "function_tool"})

_DB_METHODS = frozenset({"execute", "commit", "query", "cursor", "fetchall", "fetchone"})

_HTTP_CLIENTS = frozenset({"requests", "httpx", "aiohttp", "urllib"})

_FILE_OPS = frozenset({"open", "read", "write", "read_text", "write_text", "read_bytes", "write_bytes"})

_RESPONSE_TYPES = frozenset({"HTMLResponse", "JSONResponse", "PlainTextResponse", "Response"})

_OUTPUT_ASSEMBLY = frozenset({"join", "format", "render", "render_template", "render_template_string"})

_PROMPT_PHRASES = re.compile(
    r"(?:You are|Your role|Your task)",
    re.IGNORECASE,
)

_ROLE_SYSTEM_PATTERN = re.compile(
    r'["\']role["\']\s*:\s*["\']system["\']',
    re.IGNORECASE,
)


def _get_source_line(source: str, line_no: int) -> str:
    lines = source.splitlines()
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1].strip()
    return ""


def _make_checkpoint(
    file_path: str,
    line_number: int,
    checkpoint_type: str,
    dimension: Dimension,
    code_snippet: str = "",
    description: str = "",
) -> Checkpoint:
    return Checkpoint(
        file_path=file_path,
        line_number=line_number,
        code_snippet=code_snippet or "",
        checkpoint_type=checkpoint_type,
        dimension=dimension,
        description=description or checkpoint_type,
    )


def _resolve_local_imports(file_path: str) -> list[str]:
    """Find all locally imported Python files from the given file."""
    resolved: list[str] = []
    base_dir = Path(file_path).resolve().parent
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return resolved
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return resolved

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                candidates = [
                    base_dir / f"{mod}.py",
                    base_dir / mod / "__init__.py",
                ]
                for c in candidates:
                    if c.exists() and str(c.resolve()) not in {str(Path(p).resolve()) for p in resolved}:
                        resolved.append(str(c.resolve()))
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            parts = node.module.split(".")
            mod_dir = base_dir
            for part in parts:
                mod_dir = mod_dir / part
            candidates = [
                mod_dir / "__init__.py",
                mod_dir.with_suffix(".py") if "." not in node.module else mod_dir.parent / f"{parts[-1]}.py",
            ]
            if node.level > 0:
                parent = base_dir
                for _ in range(node.level - 1):
                    parent = parent.parent
                mod_dir = parent / node.module.replace(".", os.sep) if node.module else parent
                candidates = [
                    mod_dir / "__init__.py",
                    mod_dir.with_suffix(".py"),
                ]
            for c in candidates:
                if c.exists():
                    r = str(c.resolve())
                    if r not in resolved:
                        resolved.append(r)
                    break

    return resolved


def _scan_single_file(file_path: str) -> list[Checkpoint]:
    """Scan a single Python file for checkpoints."""
    checkpoints: list[Checkpoint] = []
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return checkpoints

    if _ROLE_SYSTEM_PATTERN.search(source):
        for i, line in enumerate(source.splitlines(), 1):
            if _ROLE_SYSTEM_PATTERN.search(line):
                checkpoints.append(
                    _make_checkpoint(
                        file_path, i, "system_role_dict", Dimension.TARGET_ALIGNMENT,
                        code_snippet=line.strip()[:200],
                    )
                )

    if _PROMPT_PHRASES.search(source):
        for i, line in enumerate(source.splitlines(), 1):
            if _PROMPT_PHRASES.search(line):
                checkpoints.append(
                    _make_checkpoint(
                        file_path, i, "prompt_phrase", Dimension.TARGET_ALIGNMENT,
                        code_snippet=line.strip()[:200],
                    )
                )

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return checkpoints

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.checkpoints: list[Checkpoint] = []

        def _add(self, node: ast.AST, ck_type: str, dim: Dimension, desc: str = "") -> None:
            line = getattr(node, "lineno", 0) or 0
            if line:
                snip = _get_source_line(source, line)
                self.checkpoints.append(
                    _make_checkpoint(file_path, line, ck_type, dim, snip, desc)
                )

        def visit_Assign(self, node: ast.Assign) -> None:
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in _PROMPT_VAR_NAMES:
                    self._add(
                        node, "prompt_var_assignment", Dimension.TARGET_ALIGNMENT,
                        f"Prompt variable: {t.id}",
                    )
            self.generic_visit(node)

        def visit_Constant(self, node: ast.Constant) -> None:
            if isinstance(node.value, str) and _PROMPT_PHRASES.search(node.value):
                self._add(
                    node, "prompt_string", Dimension.TARGET_ALIGNMENT,
                    "Prompt content with role/task phrasing",
                )
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            attr_chain: list[str] = []

            if isinstance(func, ast.Attribute):
                n = func.value
                while isinstance(n, ast.Attribute):
                    attr_chain.insert(0, n.attr)
                    n = n.value
                if isinstance(n, ast.Name):
                    attr_chain.insert(0, n.id)
                else:
                    attr_chain.insert(0, "")
            elif isinstance(func, ast.Name):
                attr_chain = [func.id]

            full_name = ".".join(attr_chain) if attr_chain else ""

            if any(x in full_name for x in ("completions.create", "chat.completions.create", "messages.create")):
                self._add(node, "llm_call", Dimension.REASONING_SOUNDNESS, "LLM completion call")
            elif any(x in full_name for x in ("generate_content", "generate")):
                base = full_name.split(".")[0] if "." in full_name else full_name
                if base in {"model", "client", "genai", "vertexai"} or "generate" in full_name:
                    self._add(node, "llm_call", Dimension.REASONING_SOUNDNESS, "LLM generate call")

            for kw in node.keywords or []:
                if isinstance(kw.arg, str) and kw.arg == "tools":
                    self._add(node, "tool_invocation", Dimension.EXECUTION_ACCURACY, "Tool call")
                    break

            if attr_chain:
                first = attr_chain[0] if attr_chain else ""
                last = attr_chain[-1] if attr_chain else ""
                if last in _DB_METHODS or (first in _DB_METHODS):
                    self._add(node, "db_operation", Dimension.EXECUTION_ACCURACY, f"DB: {last or first}")
                elif first in _HTTP_CLIENTS and last in {"get", "post", "put", "delete", "request", "patch"}:
                    self._add(node, "http_call", Dimension.EXECUTION_ACCURACY, f"HTTP: {first}.{last}")
                elif last in _FILE_OPS or (len(attr_chain) == 1 and first == "open"):
                    self._add(node, "file_io", Dimension.EXECUTION_ACCURACY, f"File I/O: {last or first}")

            if attr_chain and attr_chain[-1] in _RESPONSE_TYPES:
                self._add(node, "response_construction", Dimension.OUTPUT_INTEGRITY, f"Response: {attr_chain[-1]}")

            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            for d in node.decorator_list:
                name = ""
                if isinstance(d, ast.Name):
                    name = d.id
                elif isinstance(d, ast.Attribute):
                    name = d.attr
                elif isinstance(d, ast.Call):
                    if isinstance(d.func, ast.Name):
                        name = d.func.id
                    elif isinstance(d.func, ast.Attribute):
                        name = d.func.attr
                if name in _TOOL_DECORATORS:
                    self._add(node, "tool_function", Dimension.EXECUTION_ACCURACY, f"Tool: {node.name}")
                    break
            self.generic_visit(node)

        def visit_Return(self, node: ast.Return) -> None:
            if node.value is not None:
                self._add(node, "return_statement", Dimension.OUTPUT_INTEGRITY, "Return in agent")
            self.generic_visit(node)

    v = Visitor()
    v.visit(tree)

    seen: set[tuple[int, str]] = set()
    for cp in v.checkpoints:
        key = (cp.line_number, cp.checkpoint_type)
        if key not in seen:
            seen.add(key)
            checkpoints.append(cp)

    for i, line in enumerate(source.splitlines(), 1):
        if "chain" in line.lower() and ("thought" in line.lower() or "of" in line.lower()):
            key = (i, "chain_of_thought")
            if key not in seen:
                seen.add(key)
                checkpoints.append(
                    _make_checkpoint(
                        file_path, i, "chain_of_thought", Dimension.REASONING_SOUNDNESS,
                        code_snippet=line.strip()[:200],
                    )
                )

    return checkpoints


def scan_agent(entry_file: str) -> tuple[list[str], list[Checkpoint]]:
    """Scan agent files and return (list_of_files_found, list_of_checkpoints)."""
    entry = Path(entry_file).resolve()
    if not entry.exists():
        return [], []
    entry_str = str(entry)
    to_scan: list[str] = [entry_str]
    scanned: set[str] = set()
    files_found: list[str] = []

    while to_scan:
        path = to_scan.pop()
        path_resolved = str(Path(path).resolve())
        if path_resolved in scanned:
            continue
        scanned.add(path_resolved)
        files_found.append(path_resolved)
        for imp in _resolve_local_imports(path_resolved):
            if imp not in scanned:
                to_scan.append(imp)

    checkpoints: list[Checkpoint] = []
    for f in files_found:
        checkpoints.extend(_scan_single_file(f))

    return files_found, checkpoints
