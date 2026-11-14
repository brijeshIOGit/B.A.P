"""R2: the sandbox runs pytest in a separate process with a timeout and no network."""

import os

from codeforge.sandbox import run_in_sandbox

CODE = "def add(a, b):\n    return a + b\n"


def test_passing_tests_exit_zero():
    r = run_in_sandbox(CODE, "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    assert r["exit_code"] == 0 and r["passed"] == 1 and r["total"] == 1


def test_failing_tests_are_counted_and_named():
    tests = "from solution import add\n\ndef test_ok():\n    assert add(1, 1) == 2\n\ndef test_bad():\n    assert add(1, 1) == 3\n"
    r = run_in_sandbox(CODE, tests)
    assert r["exit_code"] == 1 and r["passed"] == 1 and r["failed"] == 1
    assert r["first_failure"].startswith("test_bad")


def test_infinite_loop_is_killed_at_the_timeout():
    r = run_in_sandbox("def spin():\n    while True:\n        pass\n",
                       "from solution import spin\n\ndef test_spin():\n    spin()\n", timeout=2)
    assert r["timed_out"] and r["exit_code"] is None and r["passed"] == 0
    assert r["duration"] < 6


def test_network_is_blocked():
    code = "import socket\n\ndef call():\n    socket.create_connection(('1.1.1.1', 80), timeout=2)\n"
    tests = ("import pytest\nfrom solution import call\n\n"
             "def test_blocked():\n    with pytest.raises((PermissionError, OSError)):\n        call()\n")
    r = run_in_sandbox(code, tests)
    assert r["exit_code"] == 0
    assert r["isolation"] in ("sandbox-exec", "unshare", "python-guard")


def test_no_api_keys_reach_the_child(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-SHOULDNOTLEAK123456")
    code = "import json\n\ndef env_size():\n    return 0\n"
    tests = ("from solution import env_size\n\ndef test_env():\n"
             "    import os\n    assert 'OPENAI_API_KEY' not in os.environ\n")
    assert run_in_sandbox(code, tests)["exit_code"] == 0


def test_temp_folder_is_removed():
    before = set(os.listdir(os.path.realpath(__import__("tempfile").gettempdir())))
    run_in_sandbox(CODE, "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    after = set(os.listdir(os.path.realpath(__import__("tempfile").gettempdir())))
    assert not [d for d in after - before if d.startswith("codeforge-")]
