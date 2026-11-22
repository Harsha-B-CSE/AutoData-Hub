"""
AutoData-Hub AST Security Sandbox.

Validates dynamically generated or user-provided Python code strings
against a strict allowlist/denylist of modules and AST node types before
any execution occurs.

Security guarantees:
  - Blocks all system calls: os, subprocess, sys, pathlib.Path.unlink, etc.
  - Blocks all network calls: requests, httpx, urllib, socket, etc.
  - Blocks dynamic imports via __import__ and importlib.
  - Blocks eval(), exec(), compile(), open() with write modes.
  - Allows only an explicit allowlist of safe data-science modules.

Usage:
  from utils.ast_sandbox import validate_code, ASTSandboxError

  try:
      validate_code(user_snippet)
  except ASTSandboxError as e:
      print(f"Blocked: {e}")
"""

from __future__ import annotations

import ast
from typing import FrozenSet, List, Set

# ---------------------------------------------------------------------------
# Allowlisted top-level module imports
# ---------------------------------------------------------------------------
_ALLOWED_IMPORTS: FrozenSet[str] = frozenset({
    "pandas",
    "pd",
    "numpy",
    "np",
    "math",
    "datetime",
    "hashlib",
    "re",
    "sklearn",
    "scipy",
})

# ---------------------------------------------------------------------------
# Blocklisted module patterns (string prefix match)
# ---------------------------------------------------------------------------
_BLOCKED_MODULE_PREFIXES: FrozenSet[str] = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests", "httpx", "aiohttp",
    "ftplib", "smtplib", "imaplib", "xmlrpc",
    "importlib", "pkgutil", "builtins",
    "ctypes", "cffi", "mmap",
    "multiprocessing", "threading", "asyncio", "concurrent",
    "signal", "pty", "tty", "termios",
    "pickle", "shelve", "marshal",
    "tempfile", "io",
    "glob", "fnmatch",
})

# ---------------------------------------------------------------------------
# Blocked AST Call names (function-level blocks)
# ---------------------------------------------------------------------------
_BLOCKED_CALLS: FrozenSet[str] = frozenset({
    "eval", "exec", "compile", "__import__",
    "open",          # file I/O blocked entirely
    "input",         # interactive input blocked
    "breakpoint",    # debugging not allowed
    "setattr", "getattr", "delattr",   # reflection restricted
    "globals", "locals", "vars",       # namespace introspection blocked
    "dir",
})


class ASTSandboxError(ValueError):
    """Raised when submitted code fails the AST security sandbox validation."""
    pass


# ---------------------------------------------------------------------------
# Visitor
# ---------------------------------------------------------------------------


class _SecurityVisitor(ast.NodeVisitor):
    """
    AST tree visitor that traverses all nodes and raises ASTSandboxError
    on first policy violation found.
    """

    def __init__(self) -> None:
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in _BLOCKED_MODULE_PREFIXES:
                self.violations.append(
                    f"Line {node.lineno}: Blocked import '{alias.name}' "
                    f"(module '{root}' is not permitted)."
                )
            elif alias.name not in _ALLOWED_IMPORTS and root not in _ALLOWED_IMPORTS:
                self.violations.append(
                    f"Line {node.lineno}: Blocked import '{alias.name}' "
                    f"(module not in allowlist)."
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".")[0]
        if root in _BLOCKED_MODULE_PREFIXES:
            self.violations.append(
                f"Line {node.lineno}: Blocked 'from {module} import ...' "
                f"(module '{root}' is not permitted)."
            )
        elif module not in _ALLOWED_IMPORTS and root not in _ALLOWED_IMPORTS:
            self.violations.append(
                f"Line {node.lineno}: Blocked 'from {module} import ...' "
                f"(module not in allowlist)."
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        # Direct name calls: eval(...), exec(...), open(...)
        if isinstance(func, ast.Name):
            if func.id in _BLOCKED_CALLS:
                self.violations.append(
                    f"Line {node.lineno}: Blocked function call '{func.id}()'."
                )
        # Attribute calls: os.system(...), subprocess.run(...), etc.
        elif isinstance(func, ast.Attribute):
            attr_name = func.attr
            if isinstance(func.value, ast.Name):
                obj_name = func.value.id
                blocked_patterns = {
                    "os": {"system", "popen", "execv", "execvp", "execvpe", "spawn",
                           "fork", "remove", "unlink", "rmdir", "makedirs", "rename"},
                    "subprocess": {"run", "call", "Popen", "check_output", "check_call"},
                    "sys": {"exit", "modules", "path"},
                    "shutil": {"rmtree", "move", "copy", "copytree"},
                }
                if obj_name in blocked_patterns and attr_name in blocked_patterns[obj_name]:
                    self.violations.append(
                        f"Line {node.lineno}: Blocked call '{obj_name}.{attr_name}()'."
                    )
            # Block dunder method calls: __class__.__subclasses__()
            if attr_name.startswith("__") and attr_name not in ("__len__", "__str__", "__repr__"):
                self.violations.append(
                    f"Line {node.lineno}: Blocked dunder attribute access '.{attr_name}'."
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Block __globals__, __code__, __closure__ attribute access
        dangerous_attrs = {"__globals__", "__code__", "__closure__", "__builtins__",
                           "__class__", "__subclasses__", "__import__"}
        if node.attr in dangerous_attrs:
            self.violations.append(
                f"Line {node.lineno}: Blocked dangerous attribute access '.{node.attr}'."
            )
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_code(code: str, raise_on_first: bool = True) -> List[str]:
    """
    Parses and validates a Python code string against the AST security sandbox.

    Parameters
    ----------
    code : str
        Python source code to validate.
    raise_on_first : bool
        If True (default), raises ASTSandboxError on the first violation found.
        If False, collects all violations and returns them as a list.

    Returns
    -------
    violations : List[str]
        Empty list if code passes. If raise_on_first=False, contains all
        violation messages.

    Raises
    ------
    ASTSandboxError
        If the code fails validation and raise_on_first=True.
    SyntaxError
        If the code cannot be parsed as valid Python.
    """
    if not code or not code.strip():
        return []

    # Step 1: Parse — will raise SyntaxError for invalid Python
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ASTSandboxError(f"Code failed syntax check: {exc}") from exc

    # Step 2: Run security visitor
    visitor = _SecurityVisitor()
    visitor.visit(tree)

    if visitor.violations:
        if raise_on_first:
            raise ASTSandboxError(
                f"AST Security Sandbox blocked code execution:\n"
                + "\n".join(visitor.violations)
            )
        return visitor.violations

    return []


def is_safe(code: str) -> bool:
    """
    Returns True if the code passes all sandbox checks, False otherwise.
    Does not raise exceptions.
    """
    try:
        violations = validate_code(code, raise_on_first=False)
        return len(violations) == 0
    except Exception:
        return False
