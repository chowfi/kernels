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


def kernel_violation(src: str) -> str | None:
    for tok in BANNED_TOKENS:
        if tok in src:
            return f"kernel/attention.py contains banned token {tok!r}"
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None  # not a policy breach; the bench will report it as crashed
    for node in ast.walk(tree):
        mods: list[str] = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                return "kernel/attention.py uses a relative import"
            mods = [node.module or ""]
        for m in mods:
            if m.split(".")[0] not in ALLOWED_IMPORTS:
                return f"kernel/attention.py imports {m!r}"
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
