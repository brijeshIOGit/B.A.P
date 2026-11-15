"""Run untrusted code + tests in a sandbox (R2).

Separate process, fresh temp folder, timeout, no network:
  - macOS: sandbox-exec profile that denies all network and only allows writes in the temp folder
  - Linux: unshare --net (a network namespace with no network), when the kernel allows it
  - everywhere: a conftest.py that makes socket connections raise an error
The result reports which layer was active, so we never claim more than we did.
"""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from . import config
from .masking import mask_text

# Loaded by pytest before the test module, so it runs before the student's code.
NETWORK_GUARD = '''import socket

def _blocked(*args, **kwargs):
    raise PermissionError("network is disabled in the CodeForge sandbox")

socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
'''

MAC_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{folder}") (literal "/dev/null") (literal "/dev/tty"))
"""

_LINUX_UNSHARE_OK = None  # checked once

MEMORY_LIMIT_BYTES = 1024 * 1024 * 1024  # 1 GB of address space for pytest + the student's code


def _linux_unshare_works() -> bool:
    """True if `unshare --net` is allowed for normal users on this machine."""
    global _LINUX_UNSHARE_OK
    if _LINUX_UNSHARE_OK is None:
        try:
            probe = subprocess.run(["unshare", "--user", "--map-root-user", "--net", "true"],
                                   capture_output=True, timeout=5)
            _LINUX_UNSHARE_OK = probe.returncode == 0
        except Exception:
            _LINUX_UNSHARE_OK = False
    return _LINUX_UNSHARE_OK


def _wrap_command(cmd: list, folder: str) -> tuple[list, str]:
    """Add the OS-level network block for this platform. Returns (command, isolation name)."""
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        return ["sandbox-exec", "-p", MAC_PROFILE.format(folder=folder)] + cmd, "sandbox-exec"
    if sys.platform.startswith("linux") and shutil.which("unshare") and _linux_unshare_works():
        return ["unshare", "--user", "--map-root-user", "--net"] + cmd, "unshare"
    return cmd, "python-guard"


def _limits(timeout: float):
    """Return a function that sets CPU and file-size limits in the child (POSIX only)."""

    def apply():
        try:
            import resource
        except ImportError:
            return
        cpu = int(timeout) + 2
        limits = [(resource.RLIMIT_CPU, cpu), (resource.RLIMIT_FSIZE, 10_000_000)]
        # Memory cap so `[0] * 10**10` fails with MemoryError instead of eating the host's RAM.
        # Linux enforces RLIMIT_AS; macOS ignores it (there a huge malloc fails on its own).
        if sys.platform.startswith("linux"):
            limits.append((resource.RLIMIT_AS, MEMORY_LIMIT_BYTES))
        for which, value in limits:
            try:  # set each limit on its own, so one the OS refuses does not skip the others
                resource.setrlimit(which, (value, value))
            except (ValueError, OSError):
                pass

    return apply


def _read_junit(path: str) -> dict:
    """Counts and the first failing test from pytest's JUnit XML report."""
    result = {"total": 0, "failed": 0, "errors": 0, "skipped": 0, "first_failure": ""}
    if not os.path.exists(path):
        return result
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    for suite in suites:
        result["total"] += int(suite.get("tests", 0))
        result["failed"] += int(suite.get("failures", 0))
        result["errors"] += int(suite.get("errors", 0))
        result["skipped"] += int(suite.get("skipped", 0))
        for case in suite.iter("testcase"):
            bad = case.find("failure")
            if bad is None:
                bad = case.find("error")
            if bad is not None and not result["first_failure"]:
                message = (bad.get("message") or "").strip().splitlines()
                result["first_failure"] = f"{case.get('name')}: {message[0] if message else 'failed'}"[:300]
    return result


def run_in_sandbox(code: str, tests: str, timeout: float | None = None) -> dict:
    """Write solution.py + test_solution.py to a temp folder and run pytest once, safely."""
    timeout = timeout or config.SETTINGS["sandbox_timeout"]
    folder = os.path.realpath(tempfile.mkdtemp(prefix="codeforge-"))
    try:
        files = {"solution.py": code, "test_solution.py": tests, "conftest.py": NETWORK_GUARD}
        for name, text in files.items():
            with open(os.path.join(folder, name), "w") as f:
                f.write(text)

        pytest_cmd = [sys.executable, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                      "--junitxml=result.xml", "test_solution.py"]
        cmd, isolation = _wrap_command(pytest_cmd, folder)
        # A fresh, tiny environment: no API keys can leak into the child.
        env = {"PATH": "/usr/bin:/bin", "HOME": folder, "TMPDIR": folder, "LANG": "C.UTF-8",
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}

        started = time.perf_counter()
        proc = subprocess.Popen(cmd, cwd=folder, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True, preexec_fn=_limits(timeout))
        timed_out = False
        try:
            output, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)  # kill pytest and anything it started
            except ProcessLookupError:
                pass
            output, _ = proc.communicate()
        duration = round(time.perf_counter() - started, 2)

        counts = _read_junit(os.path.join(folder, "result.xml"))
        exit_code = None if timed_out else proc.returncode
        passed = max(counts["total"] - counts["failed"] - counts["errors"] - counts["skipped"], 0)
        tail = "\n".join(output.decode(errors="replace").splitlines()[-40:])
        return {
            "exit_code": exit_code,
            "timed_out": timed_out,
            "passed": 0 if timed_out else passed,
            "failed": counts["failed"] + counts["errors"],
            "total": counts["total"],
            "first_failure": "" if timed_out else counts["first_failure"],
            "duration": duration,
            "timeout": timeout,
            "isolation": isolation,
            "output": mask_text(tail),
        }
    finally:
        shutil.rmtree(folder, ignore_errors=True)
