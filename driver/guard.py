"""Guardrails: what the agent may touch, and how to undo anything else."""
from __future__ import annotations

import ast
import hashlib
import re
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
                 "load_library", "jiterator", "ctypes", "exec(", "eval(",
                 # I1: delegating attention to non-Triton kernels, data-keyed caches, graphs/compile
                 "scaled_dot_product_attention", "flash_attn", "torch.ops", "data_ptr", "CUDAGraph",
                 "cuda.graph", "torch.compile", "_dynamo", "_inductor"]
# I1: attribute names (and imported names / literal getattr names) that are rejected anywhere:
# torch math that could replace the Triton kernel, op registries, streams, graphs, data identity.
BANNED_ATTRS = {"nn", "functional", "ops", "matmul", "bmm", "baddbmm", "addbmm", "mm", "addmm",
                "einsum", "softmax", "log_softmax", "tensordot", "linalg",
                "scaled_dot_product_attention", "Stream", "stream", "ExternalStream", "set_stream",
                "CUDAGraph", "graph", "graphs", "data_ptr", "_version", "compile", "_C",
                "untyped_storage", "storage",
                # host reads of tensor contents (data-keyed caches) and Triton launch-hook spoofing
                "item", "tolist", "cpu", "numpy", "knobs", "launch_enter_hook", "launch_exit_hook"}


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
# Builtins that can rebind/introspect state, run dynamic code or reach outside the kernel.
# Any bare-name LOAD of these is rejected (calling, aliasing `e = exec`, passing as a value).
BANNED_NAMES = {"setattr", "delattr", "open", "exec", "eval", "compile", "globals", "vars",
                "locals", "__import__", "breakpoint", "input", "__builtins__"}
# getattr/hasattr are allowed only as a direct call with a literal attribute name.
LITERAL_ONLY = {"getattr", "hasattr"}
# Dunder attributes that expose or rebind object/module internals: accessing them is rejected.
BANNED_DUNDERS = {"__dict__", "__setattr__", "__delattr__", "__class__", "__builtins__",
                  "__globals__", "__code__", "__closure__", "__subclasses__",
                  "__getattribute__", "__mro__", "__bases__"}
ASM_RE = re.compile(r"inline_asm|\basm\b", re.I)
WRITE_ROOT_NAMES = {"self", "ctx", "cls"}
CONTAINER_CALLS = {"dict", "list", "set"}

GUARD_RULES = """\
Guard rules enforced by `driver/guard.py` (a breach reverts the whole experiment as `violation`):
- `kernel/attention.py` may import only torch, triton, math and typing. No relative or star imports.
- No inline PTX or assembly: nothing containing `inline_asm`, and no string literal containing the
  standalone word "asm" (never write it in a string; avoid it in comments too). Also banned anywhere in
  the file: `cpp_extension`, `load_inline`, `load_library`, `jiterator`, `ctypes`, `importlib`.
- All attention math must run in your own Triton kernels; torch is only for allocation, views,
  reshapes, strides and device queries. Banned anywhere in the file (substring match, comments
  included): `scaled_dot_product_attention`, `flash_attn`, `torch.ops`, `data_ptr`, `CUDAGraph`,
  `cuda.graph`, `torch.compile`, `_dynamo`, `_inductor`. Banned as an attribute (`x.name`), an
  imported name or module component, or a literal `getattr` name: `nn`, `functional`, `ops`,
  `matmul`, `bmm`, `baddbmm`, `addbmm`, `mm`, `addmm`, `einsum`, `softmax`, `log_softmax`,
  `tensordot`, `linalg`, `scaled_dot_product_attention`, `Stream`, `stream`, `ExternalStream`,
  `set_stream`, `CUDAGraph`, `graph`, `graphs`, `data_ptr`, `_version`, `compile`, `_C`,
  `untyped_storage`, `storage`, `item`, `tolist`, `cpu`, `numpy`, `knobs`, `launch_enter_hook`,
  `launch_exit_hook`. The `@` operator is banned (use `tl.dot` inside the kernel).
  So: no side streams, no CUDA graphs, no torch/cuBLAS/FA/SDPA attention, and no caches keyed on
  tensor identity or contents.
- These builtins may not appear at all (not called, not aliased, not passed as a value): `setattr`,
  `delattr`, `open`, `exec`, `eval`, `compile`, `globals`, `vars`, `locals`, `__import__`,
  `breakpoint`, `input`, `__builtins__`. `getattr`/`hasattr` only as a direct call with a literal
  string attribute name, e.g. `getattr(tl, "dot")`.
- No access to object internals: `__dict__`, `__setattr__`, `__delattr__`, `__class__`, `__builtins__`,
  `__globals__`, `__code__`, `__closure__`, `__subclasses__`, `__getattribute__`, `__mro__`, `__bases__`;
  no attribute or item write through any dunder attribute.
- Attribute and item writes (`x.a = ...`, `x.a += ...`, `del x.a`, `x[i] = ...`, and loop/with targets
  like `for x.a in ...`) are allowed ONLY in these forms:
  - exactly one level under `self`, `ctx` or `cls` (`self.a = ...`, `ctx.sm_scale = ...`, `self[k] = ...`),
    and only if that name is never bound in the file other than as the FIRST positional parameter of a
    function (no `self = ...`, no `def f(q, self)`, no loop/with/except target or global of that name);
  - exactly one level under a function or class defined in the file with `def`/`class` and never rebound
    anywhere (no assignment, import, parameter, loop/with target of the same name),
    e.g. `attention._cache = {}`;
  - item writes (`x[k] = v`, `x[k][j] = v`, no attribute in between) on a name bound exactly once, by
    `x = {}` / `[]` / `{...}` / `dict()` / `list()` / `set()`, and never otherwise rebound,
    e.g. `_CACHE = {}` then `_CACHE[key] = kernel` (use this for caches, not `attention._cache[key]`).
  Everything else is rejected: deeper chains (`self.m.x = ...`, `attention._cache[k] = ...`) and writes
  through parameters, module names, aliases, call results, subscripts or conditional expressions. In
  particular you cannot write into tensors with `t[...] = ...` in Python: use torch ops (`t.zero_()`,
  `t.copy_(x)`, `torch.empty(...)`) or `tl.store` inside the kernel.
- The official benchmark verifies that `do_bench`, `torch.cuda.Event` (and `elapsed_time`/`record`),
  `torch.cuda.synchronize` and the harness timer are untouched after your kernel is imported and run;
  tampering fails the experiment.
"""


def _chain(node: ast.AST) -> tuple[ast.AST, list[ast.AST]]:
    """(root, links) of an attribute/subscript chain; links run outermost first."""
    links = []
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        links.append(node)
        node = node.value
    return node, links


def _chain_has_dunder(node: ast.AST) -> bool:
    return any(isinstance(l, ast.Attribute) and l.attr.startswith("__") and l.attr.endswith("__")
               for l in _chain(node)[1])


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
    elif isinstance(node, ast.NamedExpr):
        ts = [node.target]
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


def _is_container_literal(v: ast.AST | None) -> bool:
    if isinstance(v, (ast.Dict, ast.List, ast.Set)):
        return True
    return (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
            and v.func.id in CONTAINER_CALLS)


def _bindings(tree: ast.AST) -> dict[str, list[str]]:
    """Every way each name is bound in the file: 'def', 'first_param' (first positional parameter
    of a function), 'container' (single-name assignment from a container literal) or 'other'
    (assignment, import, other parameter, loop/with/except/match target, global, ...)."""
    b: dict[str, list[str]] = {}

    def add(name: str, kind: str) -> None:
        b.setdefault(name, []).append(kind)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(node.name, "def")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            positional = [*a.posonlyargs, *a.args]
            for i, arg in enumerate([*positional, *a.kwonlyargs, a.vararg, a.kwarg]):
                if arg is not None:
                    add(arg.arg, "first_param" if i == 0 and positional else "other")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                add(al.asname or al.name.split(".")[0], "other")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for n in node.names:
                add(n, "other")
        elif isinstance(node, ast.ExceptHandler) and node.name:
            add(node.name, "other")
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            add(node.name, "other")
        elif isinstance(node, ast.MatchMapping) and node.rest:
            add(node.rest, "other")
        single_container = (
            (isinstance(node, ast.Assign) and len(node.targets) == 1
             and isinstance(node.targets[0], ast.Name) and _is_container_literal(node.value))
            or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and _is_container_literal(node.value)))
        for t in _targets(node):
            if isinstance(t, ast.Name):
                add(t.id, "container" if single_container else "other")
    return b


def kernel_violation(src: str) -> str | None:
    """Policy check for kernel/attention.py (see GUARD_RULES)."""
    for tok in BANNED_TOKENS:
        if tok in src:
            return f"kernel/attention.py contains banned token {tok!r}"
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None  # not a policy breach; the bench will report it as crashed
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"kernel/attention.py imports {a.name!r}"
                if BANNED_ATTRS & set(a.name.split(".")):
                    return f"kernel/attention.py imports {a.name!r} (non-Triton math/streams/graphs)"
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                return "kernel/attention.py uses a relative import"
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"kernel/attention.py imports {node.module!r}"
            if any(a.name == "*" for a in node.names):
                return "kernel/attention.py uses a star import"
            for a in node.names:
                if BANNED_ATTRS & {*(node.module or "").split("."), a.name}:
                    return (f"kernel/attention.py imports {a.name!r} from {node.module!r} "
                            f"(non-Triton math/streams/graphs)")

    b = _bindings(tree)
    def_roots = {n for n, kinds in b.items() if set(kinds) == {"def"}}
    # self/ctx/cls are trusted only when every binding is as a function's first positional param.
    self_roots = {n for n in WRITE_ROOT_NAMES if b.get(n) and set(b[n]) == {"first_param"}}
    container_ok = not (CONTAINER_CALLS & set(b))
    container_roots = {n for n, kinds in b.items() if kinds == ["container"]} if container_ok else set()

    literal_calls: set[int] = set()  # ids of getattr/hasattr Name nodes used correctly
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in LITERAL_ONLY):
            if len(node.args) < 2 or not (isinstance(node.args[1], ast.Constant)
                                          and isinstance(node.args[1].value, str)):
                return (f"kernel/attention.py calls {node.func.id}() with a non-literal name "
                        f"(line {node.lineno})")
            name = node.args[1].value
            if ASM_RE.search(name) or name in BANNED_DUNDERS or name in BANNED_ATTRS:
                return f"kernel/attention.py reaches {name!r} via {node.func.id}()"
            literal_calls.add(id(node.func))

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in BANNED_NAMES:
                return f"kernel/attention.py uses {node.id} (line {node.lineno})"
            if node.id in LITERAL_ONLY and id(node) not in literal_calls:
                return f"kernel/attention.py uses {node.id} other than as a literal-name call (line {node.lineno})"
        if isinstance(node, ast.Attribute) and node.attr in BANNED_DUNDERS:
            return f"kernel/attention.py accesses {node.attr} (line {node.lineno})"
        if isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            return (f"kernel/attention.py uses .{node.attr} (line {node.lineno}): attention math "
                    f"must run in Triton; no torch math ops, streams, graphs or data_ptr")
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.MatMult):
            return f"kernel/attention.py uses the @ operator (line {node.lineno}); use tl.dot in Triton"
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and ASM_RE.search(node.value):
            return f"kernel/attention.py has a string literal mentioning asm (line {node.lineno})"
        for t in _targets(node):
            if not isinstance(t, (ast.Attribute, ast.Subscript)):
                continue
            where = f"{ast.unparse(t)}, line {getattr(t, 'lineno', '?')}"
            if _chain_has_dunder(t):
                return f"kernel/attention.py writes through a dunder attribute ({where})"
            root, links = _chain(t)
            ok = isinstance(root, ast.Name) and (
                # exactly one link: `self.x = ...` / `attention._cache = ...`, never `self.m.x`
                ((root.id in self_roots or root.id in def_roots) and len(links) == 1)
                or (root.id in container_roots and all(isinstance(l, ast.Subscript) for l in links)))
            if not ok:
                return (f"kernel/attention.py writes an attribute/item that is not one level under "
                        f"self/ctx/cls or an own def/class, nor an item of an own container literal "
                        f"({where})")
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
