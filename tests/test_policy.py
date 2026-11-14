"""R2: the policy gate reads the AST and never runs the code."""

import pytest

from codeforge.policy import evaluate_code_and_tests, evaluate_code_policy


def verdict(code, **kw):
    return evaluate_code_policy(code, **kw)["verdict"]


@pytest.mark.parametrize("code", [
    "def f(xs):\n    return sum(x for x in xs)\n",
    "import math\n\ndef area(r):\n    return math.pi * r * r\n",
    "from functools import reduce\n\ndef total(t):\n    return reduce(lambda a, b: a + b, t, 0)\n",
    "def spin():\n    while True:\n        pass\n",  # allowed on purpose: the sandbox timeout handles it
    "def f(xs):\n    xs.remove(1)\n    return xs\n",  # list.remove is fine
])
def test_pure_code_is_allowed(code):
    assert verdict(code) == "allow"


@pytest.mark.parametrize("code", [
    "import os\n", "import os.path\n", "from os import remove\n", "import subprocess\n", "import shutil\n",
    "import ctypes\n", "import pickle\n", "import sys\n", "import numpy\n",  # numpy: not on the allow-list
    "x = eval('1+1')\n", "exec('print(1)')\n", "__import__('os')\n",
    "def f():\n    return ().__class__.__bases__[0].__subclasses__()\n",
    "import pathlib\npathlib.Path.home()\n",
    "def f(p):\n    p.unlink()\n",
    "getattr(object, '__subclasses__')\n",
])
def test_dangerous_code_is_denied(code):
    assert verdict(code) == "deny"


@pytest.mark.parametrize("code", [
    "def f(p):\n    return open(p).read()\n",
    "import socket\n",
    "import urllib.request\n",
    "import pathlib\n\ndef f(p):\n    return pathlib.Path(p).read_text()\n",
    "import csv\n",
])
def test_file_and_network_need_approval(code):
    assert verdict(code) == "needs_approval"


def test_student_b_code_is_denied_with_reasons():
    code = ("import os\n\ndef delete_home_files():\n    home = os.path.expanduser('~')\n"
            "    for root, dirs, files in os.walk(home):\n        for n in files:\n            os.remove(n)\n")
    result = evaluate_code_policy(code)
    assert result["verdict"] == "deny"
    assert any("os" in r for r in result["reasons"]) and any("expanduser" in r for r in result["reasons"])


def test_syntax_error_is_denied():
    assert verdict("def broken(:\n") == "deny"


def test_pytest_is_allowed_only_in_tests():
    assert verdict("import pytest\n") == "deny"
    assert verdict("import pytest\nfrom solution import f\n", is_test=True) == "allow"


def test_worst_verdict_wins_across_code_and_tests():
    result = evaluate_code_and_tests("def f():\n    return 1\n", "import os\nfrom solution import f\n")
    assert result["verdict"] == "deny" and result["reasons"][0].startswith("tests,")


def test_policy_never_runs_the_code(tmp_path):
    marker = tmp_path / "ran.txt"
    evaluate_code_policy(f"open({str(marker)!r}, 'w').write('x')\n")
    assert not marker.exists()
