"""The policy gate (R2): a pure function that reads code as an AST and never runs it.

evaluate_code_policy(code) returns {"verdict", "reasons", "findings"}:
  allow           -> may run in the sandbox
  needs_approval  -> touches files or the network; an instructor must approve
  deny            -> never runs

The lists below ARE the policy. Moving "os" from DENY_IMPORTS to ALLOW_IMPORTS
turns the golden gate red (that is the live red -> green demo).
"""

import ast

ALLOW_IMPORTS = {
    "math", "cmath", "statistics", "functools", "itertools", "operator", "collections", "heapq",
    "bisect", "array", "re", "string", "textwrap", "unicodedata", "typing", "dataclasses", "enum",
    "abc", "numbers", "decimal", "fractions", "random", "datetime", "calendar", "time", "json",
    "copy", "__future__",
}
# Allowed in the generated test file only. unittest.mock is common in model-written tests.
TEST_ONLY_IMPORTS = {"pytest", "solution", "unittest"}

NEEDS_APPROVAL_IMPORTS = {
    # files
    "pathlib", "glob", "tempfile", "fileinput", "csv", "zipfile", "tarfile", "gzip", "sqlite3", "io",
    # network
    "socket", "ssl", "urllib", "http", "requests", "httpx", "aiohttp", "asyncio", "smtplib", "ftplib",
}

DENY_IMPORTS = {
    "os", "posix", "sys", "subprocess", "shutil", "ctypes", "cffi", "multiprocessing", "concurrent",
    "threading", "_thread", "signal", "resource", "pty", "fcntl", "mmap", "importlib", "builtins",
    "inspect", "gc", "code", "codeop", "pickle", "marshal", "shelve", "runpy", "pkgutil", "site",
    "sysconfig", "platform", "getpass", "pwd", "webbrowser",
}

DENY_BUILTINS = {"eval", "exec", "compile", "__import__", "globals", "locals", "vars", "breakpoint"}
NEEDS_APPROVAL_BUILTINS = {"open"}

# Method names that destroy files, start processes or reach the home folder.
DENY_METHODS = {
    "unlink", "rmdir", "rmtree", "removedirs", "chmod", "chown", "system", "popen", "kill",
    "killpg", "fork", "execv", "execve", "spawnv", "expanduser", "home", "rename", "renames", "symlink",
    "truncate",
}
NEEDS_APPROVAL_METHODS = {"listdir", "scandir", "read_text", "write_text", "read_bytes", "write_bytes", "iterdir", "mkdir", "touch"}

# Attribute names used to escape a sandbox from pure Python.
DENY_DUNDERS = {"__globals__", "__builtins__", "__subclasses__", "__bases__", "__mro__", "__code__", "__closure__",
                "__dict__", "__loader__", "__spec__"}

# Allowed modules re-export dangerous ones: random._os, dataclasses.sys, typing.sys ...
# Reading one of these names as an attribute is a way round the import rules.
ESCAPE_ATTRS = {"os", "sys", "subprocess", "shutil", "builtins", "posix", "nt", "importlib", "ctypes", "modules"}

ORDER = {"allow": 0, "needs_approval": 1, "deny": 2}


def _finding(node, level: str, rule: str, detail: str) -> dict:
    """One thing the gate noticed, with its line number."""
    return {"line": getattr(node, "lineno", 0), "level": level, "rule": rule, "detail": detail}


def _check_import(node, module: str, is_test: bool) -> dict | None:
    """Classify one imported module."""
    top = (module or "").split(".")[0]
    if top in ALLOW_IMPORTS or (is_test and top in TEST_ONLY_IMPORTS):
        return None
    if top in NEEDS_APPROVAL_IMPORTS:
        return _finding(node, "needs_approval", "import", f"imports `{top}` (file or network access)")
    if top in DENY_IMPORTS:
        return _finding(node, "deny", "import", f"imports `{top}` (blocked module)")
    # Not malicious, just unknown (numpy, pandas ...): still denied, but marked so the graph can
    # ask the model to rewrite with the standard library instead of blaming the student.
    return _finding(node, "deny", "unlisted-import", f"imports `{top or '?'}` (not on the allow-list)")


def _check_call(node: ast.Call) -> dict | None:
    """Classify one function or method call."""
    func = node.func
    if isinstance(func, ast.Name) and func.id in {"getattr", "setattr", "delattr"} and len(node.args) >= 2:
        name = node.args[1]
        if not isinstance(name, ast.Constant):
            return _finding(node, "deny", "builtin", f"calls `{func.id}` with a computed name (hides what it reads)")
        attr = str(name.value)
        if attr.startswith("_") or attr in DENY_BUILTINS | DENY_METHODS | ESCAPE_ATTRS:
            return _finding(node, "deny", "builtin", f"calls `{func.id}` on `{attr}` (sandbox escape)")
    return None


def _check_name(node: ast.Name) -> dict | None:
    """Any use of a dangerous builtin, called or not: `r = open; r(p)` is the same as `open(p)`."""
    if node.id in DENY_BUILTINS:
        return _finding(node, "deny", "builtin", f"uses `{node.id}` (runs arbitrary code)")
    if node.id in NEEDS_APPROVAL_BUILTINS:
        return _finding(node, "needs_approval", "builtin", f"uses `{node.id}` (file access)")
    if node.id in DENY_DUNDERS:
        return _finding(node, "deny", "dunder", f"uses `{node.id}` (sandbox escape)")
    return None


def _check_attribute(node: ast.Attribute, modules: set) -> dict | None:
    """Attribute reads, called or not: `u = p.unlink; u()` is the same as `p.unlink()`."""
    attr = node.attr
    if attr in DENY_DUNDERS:
        return _finding(node, "deny", "dunder", f"uses `{attr}` (sandbox escape)")
    if attr in DENY_METHODS:
        return _finding(node, "deny", "method", f"uses `.{attr}` (destructive or escapes the sandbox)")
    if attr in ESCAPE_ATTRS:
        return _finding(node, "deny", "module-escape", f"reads `.{attr}` (reaches a blocked module)")
    base = node.value
    if isinstance(base, ast.Name) and base.id in modules and attr.startswith("_"):
        return _finding(node, "deny", "module-escape", f"reads private `{base.id}.{attr}` (reaches module internals)")
    if attr in NEEDS_APPROVAL_METHODS:
        return _finding(node, "needs_approval", "method", f"uses `.{attr}` (file access)")
    return None


def _imported_names(tree) -> set:
    """Local names bound to imported modules (`import random as r` -> {"r"})."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return names


def evaluate_code_policy(code: str, is_test: bool = False) -> dict:
    """Read the code's AST and return allow / needs_approval / deny with reasons. Never runs it."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError as err:
        finding = {"line": err.lineno or 0, "level": "deny", "rule": "syntax", "detail": "code does not parse"}
        return {"verdict": "deny", "reasons": ["code does not parse"], "findings": [finding]}

    findings = []
    modules = _imported_names(tree)
    for node in ast.walk(tree):
        found = None
        if isinstance(node, ast.Import):
            for alias in node.names:
                found = _check_import(node, alias.name, is_test)
                if found:
                    findings.append(found)
            continue
        if isinstance(node, ast.ImportFrom):
            module = node.module if node.level == 0 else ""  # relative imports are not allowed
            found = _check_import(node, module, is_test)
        elif isinstance(node, ast.Call):
            found = _check_call(node)
        elif isinstance(node, ast.Attribute):
            found = _check_attribute(node, modules)
        elif isinstance(node, ast.Name):
            found = _check_name(node)
        if found:
            findings.append(found)

    verdict = "allow"
    for f in findings:
        if ORDER[f["level"]] > ORDER[verdict]:
            verdict = f["level"]
    reasons = [f"line {f['line']}: {f['detail']}" for f in findings]
    return {"verdict": verdict, "reasons": reasons, "findings": findings}


def is_model_fixable(result: dict) -> bool:
    """True when every deny comes from the model's own choices, not from what the student asked for.

    That is: an unknown (not blocked) import, or anything in the model-written test file. The graph
    then asks the model to rewrite instead of refusing, and no policy strike is counted.
    Deny findings in the solution itself (os, eval, unlink ...) are never fixable this way.
    """
    denies = [f for f in result.get("findings", []) if f["level"] == "deny"]
    return bool(denies) and all(f.get("file") == "tests" or f["rule"] == "unlisted-import" for f in denies)


def evaluate_code_and_tests(code: str, tests: str) -> dict:
    """Check the solution and the test file; the worst verdict wins and reasons are merged."""
    a = evaluate_code_policy(code)
    b = evaluate_code_policy(tests, is_test=True)
    verdict = a["verdict"] if ORDER[a["verdict"]] >= ORDER[b["verdict"]] else b["verdict"]
    reasons = a["reasons"] + [f"tests, {r}" for r in b["reasons"]]
    findings = a["findings"] + [dict(f, file="tests") for f in b["findings"]]
    return {"verdict": verdict, "reasons": reasons, "findings": findings}
