"""Guardrails: what the agent may touch, and how to undo anything else."""
from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path

ALLOWED_AGENT_FILES = {"kernel/attention.py", "notes.md", "experiments/current.json"}
# Written by the driver between invocations (never during one). Excluded from the git-status
# check; the driver protects them with a byte snapshot around each invocation instead.
DRIVER_FILES = {"results.tsv", "results/run_state.json", "results/clock.json", "results/tokens.json"}
# Appended by the agent's own `harness.check` runs (ruling R3); committed as the raw record.
HARNESS_WRITTEN = {"runs.log"}

ALLOWED_IMPORTS = {"torch", "triton", "math", "typing", "__future__"}
# Ways to reach native code / dynamic imports without an import statement.
BANNED_TOKENS = ["inline_asm", "__import__", "importlib", "cpp_extension", "load_inline",
                 "load_library", "jiterator", "ctypes", "exec(", "eval("]


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True).stdout


def harness_hash(root: Path) -> str:
    root = Path(root)
    files = sorted(p for p in (root / "harness").rglob("*.py"))
    files += [root / n for n in ("program.md", "pyproject.toml", "uv.lock")]
    h = hashlib.sha256()
    for p in files:
        if not p.is_file():  # R5: skip files that do not exist yet
            continue
        h.update(p.relative_to(root).as_posix().encode() + b"\0")
        h.update(p.read_bytes() + b"\0")
    return h.hexdigest()


def changed_files(root: Path) -> set[str]:
    """Paths reported by `git status` (modified, staged, deleted, untracked files; both sides
    of a rename). Gitignored paths are not reported."""
    out = _git(Path(root), "status", "--porcelain=v1", "-z", "-uall")
    parts = out.split("\0")
    files: set[str] = set()
    i = 0
    while i < len(parts):
        e = parts[i]
        i += 1
        if len(e) < 4:
            continue
        files.add(e[3:])
        if e[0] in "RC":  # rename/copy: next field is the source path
            if i < len(parts) and parts[i]:
                files.add(parts[i])
            i += 1
    return files


# Builtins that can rebind/introspect module state or reach outside the kernel.
BANNED_CALLS = {"setattr", "delattr", "open", "exec", "eval", "compile", "globals", "vars",
                "locals", "__import__", "breakpoint", "input"}


def _root_name(node: ast.AST) -> str | None:
    """Root Name of an attribute/subscript chain (`torch.cuda.Event.x` -> 'torch')."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _targets(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.Assign):
        ts = node.targets
    elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        ts = [node.target]
    elif isinstance(node, ast.Delete):
        ts = node.targets
    elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
        ts = [node.target]
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        ts = [i.optional_vars for i in node.items if i.optional_vars is not None]
    else:
        return []
    out: list[ast.AST] = []
    stack = list(ts)
    while stack:
        t = stack.pop()
        if isinstance(t, (ast.Tuple, ast.List)):
            stack.extend(t.elts)
        elif isinstance(t, ast.Starred):
            stack.append(t.value)
        else:
            out.append(t)
    return out


def kernel_violation(src: str) -> str | None:
    """Policy check for kernel/attention.py: allowed imports only, no native code, and nothing
    that could rebind torch/triton/math/typing attributes (e.g. timing functions)."""
    for tok in BANNED_TOKENS:
        if tok in src:
            return f"kernel/attention.py contains banned token {tok!r}"
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None  # not a policy breach; the bench will report it as crashed
    modules: set[str] = set()  # every name bound by an import (modules and their aliases)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"kernel/attention.py imports {a.name!r}"
                modules.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                return "kernel/attention.py uses a relative import"
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"kernel/attention.py imports {node.module!r}"
            for a in node.names:
                if a.name == "*":
                    return "kernel/attention.py uses a star import"
                modules.add(a.asname or a.name)
    # Names bound from module-rooted expressions are aliases too (`ev = torch.cuda.Event`).
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and node.value is not None \
                    and _root_name(node.value) in modules:
                for t in (_targets(node) if not isinstance(node, ast.NamedExpr) else [node.target]):
                    if isinstance(t, ast.Name) and t.id not in modules:
                        modules.add(t.id)
                        changed = True
    for node in ast.walk(tree):
        for t in _targets(node):
            # Any attribute (re)binding is rejected outright: a kernel has no need for it, and it
            # is how timing functions would be monkeypatched (possibly through an alias).
            if isinstance(t, ast.Attribute) or (isinstance(t, ast.Subscript) and _root_name(t) in modules):
                return (f"kernel/attention.py modifies module state "
                        f"({ast.unparse(t)}, line {getattr(t, 'lineno', '?')})")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            f = node.func.id
            if f in BANNED_CALLS:
                return f"kernel/attention.py calls {f}() (line {node.lineno})"
            if f in ("getattr", "hasattr"):
                if len(node.args) < 2 or not (isinstance(node.args[1], ast.Constant)
                                              and isinstance(node.args[1].value, str)):
                    return f"kernel/attention.py calls {f}() with a non-literal name (line {node.lineno})"
                if "asm" in node.args[1].value.lower():
                    return f"kernel/attention.py reaches {node.args[1].value!r} via {f}()"
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "asm" in node.value.lower():
            return f"kernel/attention.py has a string literal containing 'asm' (line {node.lineno})"
    return None


def check_agent_changes(root: Path, head_before: str) -> str | None:
    root = Path(root)
    head = _git(root, "rev-parse", "HEAD").strip()
    if head != head_before:
        return f"HEAD moved from {head_before[:10]} to {head[:10]}"
    outside = sorted(changed_files(root) - ALLOWED_AGENT_FILES - DRIVER_FILES - HARNESS_WRITTEN)
    if outside:
        return "changed files outside the allowed set: " + ", ".join(outside)
    k = root / "kernel" / "attention.py"
    if not k.is_file():
        return "kernel/attention.py was deleted"
    return kernel_violation(k.read_text(errors="replace"))


def revert_to(root: Path, commit: str) -> None:
    """Hard-reset tracked files and delete untracked (non-ignored) files anywhere in the tree.
    No -x: .venv, results/bench, progress.log, .superpowers are never touched."""
    _git(Path(root), "reset", "-q", "--hard", commit)
    _git(Path(root), "clean", "-q", "-fd")
