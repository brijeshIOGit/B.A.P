"""R5: retries only for transient errors, full jitter, circuit breaker, backup, template, honest refusal."""

from langchain_core.messages import HumanMessage

from codeforge import config, models
from codeforge.agents import GeneratedCode, Triage
from codeforge.models import CALL_LOG, CHAOS, SimulatedAPIError, ask_model, breaker_allows, is_transient

PALINDROME = [HumanMessage(content="REQUEST: Write a Python function that checks if a string is a palindrome.")]
PANGRAM = [HumanMessage(content="REQUEST: write a function to check if a sentence is a pangram")]


def test_transient_errors_are_recognised():
    assert is_transient(SimulatedAPIError(429)) and is_transient(SimulatedAPIError(503))
    assert not is_transient(SimulatedAPIError(400)) and not is_transient(ValueError("bad output"))


def test_normal_call_is_served_by_primary():
    answer, tier = ask_model("code_gen", PALINDROME, schema=GeneratedCode, size="big")
    assert tier == "primary" and "def is_palindrome" in answer.code


def test_primary_down_falls_back_to_backup():
    CHAOS.configure(rate=1.0, tiers=("primary",))
    answer, tier = ask_model("code_gen", PALINDROME, schema=GeneratedCode, size="big")
    assert tier == "backup" and answer is not None


def test_total_outage_uses_the_template_tier():
    CHAOS.configure(rate=1.0)
    template = lambda: GeneratedCode(code="def is_palindrome(t):\n    return t == t[::-1]\n")  # noqa: E731
    answer, tier = ask_model("code_gen", PALINDROME, schema=GeneratedCode, size="big", template=template)
    assert tier == "template" and "is_palindrome" in answer.code


def test_unknown_task_in_outage_gets_no_invented_code():
    CHAOS.configure(rate=1.0)
    answer, tier = ask_model("code_gen", PANGRAM, schema=GeneratedCode, size="big", template=lambda: None)
    assert answer is None and tier == "none"


def test_retries_wait_with_full_jitter(monkeypatch):
    waits = []
    monkeypatch.setattr(config, "SLEEP", waits.append)
    CHAOS.configure(rate=1.0, tiers=("primary",))
    ask_model("triage", PALINDROME, schema=Triage)
    caps = [min(config.SETTINGS["retry_cap"], config.SETTINGS["retry_base"] * 2 ** k) for k in range(2)]
    assert len(waits) == 2  # 3 tries -> 2 waits
    assert all(0 <= w <= cap for w, cap in zip(waits, caps))


def test_non_transient_error_is_not_retried():
    CALL_LOG.clear()
    ask_model("code_gen", PANGRAM, schema=GeneratedCode, size="big")
    primary = [c for c in CALL_LOG if c["tier"] == "primary"]
    assert len(primary) == 1 and primary[0]["outcome"] == "error"


def test_breaker_opens_after_three_failures_and_closes_after_cooldown(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(config, "CLOCK", lambda: now[0])
    model = "test-model"
    for _ in range(3):
        models.breaker_failure(model)
    assert not breaker_allows(model)
    now[0] += config.SETTINGS["breaker_cooldown"] + 1
    assert breaker_allows(model)


def test_chaos_at_60_percent_still_answers_every_request():
    CHAOS.configure(rate=0.6, seed=1)
    tiers = [ask_model("code_gen", PALINDROME, schema=GeneratedCode, size="big",
                       template=lambda: GeneratedCode(code="def is_palindrome(t):\n    return True\n"))[1]
             for _ in range(20)]
    assert all(t in ("primary", "backup", "template") for t in tiers)
    assert CHAOS.injected > 0


def test_prompts_are_masked_before_the_model():
    from codeforge.cost import Meter

    meter = Meter()
    secret = "sk-proj-AbCdEfGhIjKlMnOp1234"
    ask_model("triage", [HumanMessage(content=f"REQUEST: my key {secret}")], schema=Triage,
              cfg={"callbacks": [meter]})
    assert meter.prompts and not any(secret in p for p in meter.prompts)
    assert any("[SECRET]" in p for p in meter.prompts)
