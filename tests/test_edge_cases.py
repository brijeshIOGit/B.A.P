"""Regression tests for the 11 edge cases found in the instructor review.

Each test names the finding it guards (E1 ... E11). If one goes red, that hole is open again.
"""

import sys

import pytest

from codeforge import agents, config, db
from codeforge.escalation import combine_signals, higher_priority, merge_ticket, priority_for
from codeforge.graph import chat, make_config
from codeforge.masking import mask
from codeforge.policy import evaluate_code_and_tests, evaluate_code_policy, is_model_fixable
from codeforge.sandbox import run_in_sandbox
from codeforge.templates import has_extra_spec, match_template

FAKE_KEY = "sk-proj-ABCDEFGHIJKLMNOP4321"


def verdict(code, **kw):
    return evaluate_code_policy(code, **kw)["verdict"]


# ── E1: a builtin stored under another name is still caught ─────────────────

def test_e1_aliased_eval_is_denied():
    assert verdict("def f(s):\n    e = eval\n    return e(s)\n") == "deny"


def test_e1_aliased_open_needs_approval():
    assert verdict("def f(p):\n    r = open\n    return r(p).read()\n") == "needs_approval"


def test_e1_aliased_method_is_denied():
    assert verdict("def f(p):\n    u = p.unlink\n    u()\n") == "deny"


def test_e1_getattr_with_computed_name_is_denied():
    assert verdict("def f(o, n):\n    return getattr(o, n)\n") == "deny"


# ── E2: no reaching os / sys through an allowed module ──────────────────────

@pytest.mark.parametrize("code", [
    "import random\n\ndef f(p):\n    random._os.remove(p)\n",
    "import dataclasses\n\ndef f():\n    return dataclasses.sys.modules['os']\n",
    "import typing\n\ndef f():\n    return typing.sys\n",
    "import random as r\n\ndef f():\n    return r._inst\n",
    "import json\n\ndef f():\n    return json.__dict__\n",
])
def test_e2_module_escape_is_denied(code):
    assert verdict(code) == "deny"


def test_e2_private_attributes_on_own_objects_are_fine():
    code = "class Stack:\n    def __init__(self):\n        self._items = []\n\n    def push(self, x):\n" \
           "        self._items.append(x)\n"
    assert verdict(code) == "allow"


# ── E3: a memory bomb is stopped by the sandbox (Linux enforces RLIMIT_AS) ──

@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS is only enforced on Linux")
def test_e3_memory_bomb_fails_inside_the_sandbox():
    code = "def bomb():\n    return [0] * (10 ** 10)\n"
    tests = "import pytest\nfrom solution import bomb\n\ndef test_bomb():\n    with pytest.raises(MemoryError):\n" \
            "        bomb()\n"
    result = run_in_sandbox(code, tests, timeout=20)
    assert not result["timed_out"] and result["exit_code"] == 0


# ── E4: the model's own bad import is rewritten, not a student strike ───────

def test_e4_unittest_mock_is_allowed_in_tests():
    tests = "from unittest.mock import patch\nfrom solution import f\n\ndef test_a():\n    assert f() == 1\n"
    assert evaluate_code_and_tests("def f():\n    return 1\n", tests)["verdict"] == "allow"


def test_e4_unknown_import_and_bad_tests_are_model_fixable():
    assert is_model_fixable(evaluate_code_and_tests("import numpy\n", "from solution import f\n"))
    assert is_model_fixable(evaluate_code_and_tests("def f():\n    return 1\n", "import sys\nfrom solution import f\n"))


def test_e4_blocked_module_in_solution_is_not_model_fixable():
    assert not is_model_fixable(evaluate_code_and_tests("import os\n", "from solution import f\n"))


def test_e4_policy_gate_retries_once_then_refuses_without_a_strike():
    cfg = make_config("u_alice", "e4")
    state = {"code": "import numpy as np\n\ndef mean(x):\n    return float(np.mean(x))\n",
             "tests": "from solution import mean\n\ndef test_m():\n    assert mean([1, 3]) == 2\n",
             "served_by": {"code_gen": "primary"}, "request": "mean of a list"}
    first = agents.policy_gate(state, cfg)
    assert first["policy_retry"] and "numpy" in first["policy_feedback"]
    second = agents.policy_gate({**state, "policy_feedback": first["policy_feedback"]}, cfg)
    assert second["refusal"]["kind"] == "policy_deny" and second["refusal"]["strike"] is False
    refused = agents.refuse({**state, **second, "strikes": 1}, cfg)
    assert "strikes" not in refused and "no policy strike" in refused["answer"]


# ── E5: the raw key never reaches the checkpoint history ────────────────────

def test_e5_checkpoint_history_holds_no_raw_key(app):
    chat(app, f"my key {FAKE_KEY} please help", user_id="u_alice", thread="e5")
    history = list(app.get_state_history(make_config("u_alice", "e5")))
    assert history and all(FAKE_KEY not in str(snapshot.values) for snapshot in history)


def test_e5_pre_masked_text_still_reports_pii(app):
    state = chat(app, f"my key {FAKE_KEY} write a palindrome checker", user_id="u_alice", thread="e5b")
    assert "secret" in state["pii_found"]


# ── E6: more key formats and international phones; code values survive ─────

@pytest.mark.parametrize("text", [
    "sk_live_abcdefghijklmnop1234", "ghp_abcdefghijklmnopqrstuvwxyz0123456789", "AKIAIOSFODNN7EXAMPLE",
    "sk-ant-api03-abcdefghijklmnopqrstuv", "AIzaSyA1234567890abcdefghijklmnopqrstuv",  # 39 chars, like a real key
])
def test_e6_more_secret_formats_are_masked(text):
    assert mask(f"key {text} here")[0] == "key [SECRET] here"


def test_e6_international_phone_is_masked():
    assert mask("call +1 415 555 0134 now")[0] == "call [PHONE] now"


def test_e6_ten_digit_value_in_code_is_not_a_phone():
    assert mask("return 9876543210")[0] == "return 9876543210"
    assert mask("assert f(x) == 9876543210")[0] == "assert f(x) == 9876543210"


def test_e6_phone_assigned_to_a_variable_is_still_masked():
    assert mask("phone = 9876543210")[0] == "phone = [PHONE]"


# ── E7: quota limits sandbox runs only, and is checked before every run ─────

def test_e7_student_at_quota_can_still_reach_a_human(app):
    config.SETTINGS["quota"] = {"student": 0, "instructor": 0}
    state = chat(app, "THIS IS NOT WORKING!!! get me a human", user_id="u_alice", thread="e7")
    assert state["ticket"]["priority"] == "P1"


def test_e7_instructor_at_quota_can_still_ask_sql(app):
    config.SETTINGS["quota"] = {"student": 0, "instructor": 0}
    state = chat(app, "total sales by region", user_id="u_carol", thread="e7s")
    assert state["sql"]["ok"]


def test_e7_quota_blocks_the_run_and_nothing_executes(app):
    config.SETTINGS["quota"] = {"student": 0, "instructor": 0}
    state = chat(app, "write a function to reverse a string", user_id="u_alice", thread="e7r")
    assert "quota" in state["answer"] and db.my_runs(make_config("u_alice", "x")) == []


# ── E8: idempotency survives a race and a failing request sent twice ────────

def test_e8_second_insert_of_the_same_key_returns_the_stored_run():
    result = {"exit_code": 0, "passed": 1, "failed": 0, "total": 1, "timed_out": False, "duration": 0.1,
              "first_failure": "", "isolation": "test"}
    first = db.save_run("u_alice", "k" * 64, "req", "code", "tests", result, {})
    second = db.save_run("u_alice", "k" * 64, "req", "code", "tests", result, {})
    assert second["reused"] and second["run_id"] == first["run_id"]


def test_e8_failing_request_twice_runs_once(app):
    config.SETTINGS["sandbox_timeout"] = 2
    first = chat(app, "write a function that loops forever", user_id="u_alice", thread="e8a")
    second = chat(app, "write a function that loops forever", user_id="u_alice", thread="e8b")
    assert first["run"]["timed_out"] and second["run"]["reused"]
    assert len(db.my_runs(make_config("u_alice", "x"))) == 1


# ── E9: P1 does not depend on the model's exact mood word; one ticket per chat

def test_e9_rules_raise_a_mild_model_reading():
    signals = combine_signals({"sentiment": "negative", "urgency": 1, "wants_human": False},
                              {"sentiment": "angry", "urgency": 1, "wants_human": True})
    assert signals["sentiment"] == "angry" and signals["wants_human"]


def test_e9_rules_never_lower_the_model_reading():
    signals = combine_signals({"sentiment": "angry", "urgency": 5, "wants_human": True},
                              {"sentiment": "neutral", "urgency": 1, "wants_human": False})
    assert signals == {"sentiment": "angry", "urgency": 5, "wants_human": True}


def test_e9_negative_and_asks_for_human_is_p1():
    assert priority_for({"sentiment": "negative", "urgency": 1, "wants_human": True}, "")[0] == "P1"


def test_e9_ticket_merge_rules():
    assert merge_ticket(None) == "new" and merge_ticket({"id": "T-1001"}) == "update"
    assert higher_priority("P2", "P1") == "P1" and higher_priority("P1", "P3") == "P1"


def test_e9_second_escalation_in_a_chat_updates_the_open_ticket(app):
    first = chat(app, "get me a human please", user_id="u_alice", thread="e9")
    second = chat(app, "THIS IS NOT WORKING!!! get me a human", user_id="u_alice", thread="e9")
    assert second["ticket"]["id"] == first["ticket"]["id"] and second["ticket"]["priority"] == "P1"
    assert len(db.all_tickets()) == 1 and "already open" in second["answer"]


# ── E10: a template is never used for a request with extra rules ────────────

@pytest.mark.parametrize("request_text", [
    "Write a case-sensitive palindrome checker that does not ignore spaces",
    "factorial that raises ValueError for negatives",
    "fibonacci where fib(1)=1, 1-indexed",
    "sort the list in place and return None",
    "return a list of the first n fibonacci numbers",
])
def test_e10_extra_rules_never_match_a_template(request_text):
    assert match_template(request_text) is None


def test_e10_whole_words_only():
    assert not has_extra_spec("praise this palindrome checker")


def test_e10_extra_rule_in_an_outage_gets_an_honest_refusal(app):
    state = chat(app, "Write a case-sensitive palindrome checker", user_id="u_alice", thread="e10")
    assert "Cannot generate now" in state["answer"] and state["code"] == ""


# ── E11: classes are generated and tested, not refused ──────────────────────

def test_e11_class_request_runs_and_passes(app):
    state = chat(app, "Write a Python class Stack with push and pop", user_id="u_alice", thread="e11")
    assert "class Stack" in state["code"] and state["run"]["exit_code"] == 0


def test_e11_defines_function_accepts_classes_and_async():
    assert agents.defines_function("class Stack:\n    pass\n", "Stack")
    assert agents.defines_function("async def fetch():\n    return 1\n", "fetch")
