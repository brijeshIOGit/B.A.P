"""The agents: Pydantic schemas, prompts, and the node functions of the graph.

Every node is a plain function  node(state, config) -> dict  that returns only the
fields it changed. Model nodes call ask_model(); everything else is plain Python.
"""

import ast
import hashlib
import json
import keyword
import re
from typing import ClassVar, Literal

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from .config import PROVIDER, SETTINGS
from .brain import guess_function_name, rule_requirements, rule_route, rule_sql, rule_triage
from .cache import CACHE, is_shareable
from .db import create_ticket, find_run, update_ticket, log_event, quota_left, run_lock, save_run, who_is
from .escalation import (REASON_CANNOT_GENERATE, REASON_DENIALS, REASON_REPAIRS, combine_signals,
                         denials_need_human, higher_priority, merge_ticket, next_unhappy_streak, priority_for,
                         repairs_need_human, should_escalate)
from .masking import mask, mask_code, mask_text
from .models import ask_model
from .policy import evaluate_code_and_tests, is_model_fixable
from .sandbox import run_in_sandbox
from .sql_agent import run_sql, validate_sql
from .templates import match_template, template_requirements


# ── structured outputs ───────────────────────────────────────────────────────

class Triage(BaseModel):
    """Signals about the latest message. The model reads the mood; code decides what happens."""
    task_type: Literal["code_request", "sql_question", "question", "complaint", "other"]
    language: str = Field(default="python", description="programming language asked for; python if none")
    sentiment: Literal["positive", "neutral", "negative", "angry"]
    urgency: int = Field(ge=1, le=5, description="1 = no rush ... 5 = deadline in minutes or tonight")
    wants_human: bool = Field(description="True only if the user asks for a person, mentor or human")


class RouteDecision(BaseModel):
    """What kind of code is needed."""
    kind: Literal["python_function", "sql_query", "other"]
    language: str = "python"
    reasoning: str = ""


class Requirements(BaseModel):
    """The contract that code and tests are both built from."""
    function_name: str = Field(description="snake_case name of the one function, or CamelCase name of the one class")
    params: list[str] = Field(default_factory=list, description='e.g. ["text: str"]')
    returns: str = ""
    edge_cases: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list, description='e.g. ["is_palindrome(\'abba\') -> True"]')


class GeneratedCode(BaseModel):
    text_field: ClassVar[str] = "code"  # asked for as one ```python block, not a JSON tool call
    code: str = Field(description="the full content of solution.py")


class GeneratedTests(BaseModel):
    text_field: ClassVar[str] = "tests"  # asked for as one ```python block, not a JSON tool call
    tests: str = Field(description="the full content of test_solution.py")


class Handoff(BaseModel):
    summary: str = Field(description="two sentences a busy mentor can read in five seconds")
    user_ask: str
    suggested_next_step: str


class SQLQuery(BaseModel):
    sql: str
    explanation: str = ""


# ── prompts ──────────────────────────────────────────────────────────────────

TRIAGE_PROMPT = ("Classify the student's latest message for a coding-help service. Judge sentiment from tone, "
                 "punctuation and wording; urgency from deadlines they mention; wants_human only if they ask for a "
                 "person. You do NOT decide what happens next.")
ROUTER_PROMPT = ("Decide what the student needs: python_function (they want code: a function or small program), "
                 "sql_query (a question about data in a database), or other (anything else, including code in "
                 "another programming language). Default to Python when no language is named.")
REQUIREMENTS_PROMPT = ("Extract the requirements for ONE Python function (or ONE class, if the request asks for a "
                       "class) from the request: function_name is the snake_case function name or the CamelCase "
                       "class name, then its params, what it returns, edge cases and 2-4 examples.")
CODE_PROMPT = ("You write the content of solution.py. Define a top-level function (or class, if one is asked for) "
               "named `{name}` that meets the requirements. "
               "Use only the Python standard library. No file access, no network, no input(), no printing, no "
               "top-level code except imports and definitions. Return only code.")
TESTS_PROMPT = ("You write the content of test_solution.py for pytest. Start with `from solution import {name}`. "
                "Write 4-8 small, correct tests that check the behaviour in the requirements and edge cases. Use "
                "only pytest and the standard library. No file or network access. Return only code.")
HANDOFF_PROMPT = ("Write a short handoff note for the human mentor taking over. Be factual. Never include keys, "
                  "emails, phone numbers, card or Aadhaar numbers.")
SQL_PROMPT = ("Write ONE SQLite SELECT query that answers the question. Tables: "
              "sales(id, region, product, amount, sale_date, employee_id), "
              "employees(id, name, department, region, hire_date). Use only these tables.")
SUMMARY_PROMPT = "Summarise this coding-help chat in 3 short sentences. Keep task names and decisions. No personal data."

# Fields that belong to one turn. guard_in resets them at the start of every message.
TURN_RESET = {"request": "", "pii_found": [], "blocked": "", "triage": {}, "escalate_reason": "", "route": "",
              "requirements": {}, "code": "", "tests": "", "served_by": {}, "cache_hit": "", "cache_source": "", "policy": {},
              "approved": False, "run": {}, "attempts": 0, "repair": False, "refusal": {}, "ticket": {},
              "sql": {}, "answer": "", "quota_left": 0,
              "policy_retry": False, "policy_feedback": ""}


# ── small helpers ────────────────────────────────────────────────────────────

def clean_code(text: str) -> str:
    """Remove ``` fences a model may add around code."""
    text = (text or "").strip()
    fence = re.match(r"^```[a-zA-Z]*\n(.*?)\n?```$", text, re.S)
    return (fence.group(1) if fence else text).strip() + "\n"


DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def defines_function(code: str, name: str) -> bool:
    """True if the code parses and defines a function or class with this name (top level)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(n, DEFINITIONS) and n.name == name for n in tree.body)


def run_key(user_id: str, code: str, tests: str) -> str:
    """Idempotency key: the same user, code and tests always give the same key."""
    return hashlib.sha256(f"{user_id}\n{code}\n{tests}".encode()).hexdigest()


def _with_tier(state: dict, agent: str, tier: str) -> dict:
    """served_by with one more agent -> tier entry."""
    return {**state.get("served_by", {}), agent: tier}


# ── nodes ────────────────────────────────────────────────────────────────────

def guard_in(state, config):
    """Mask the new message, read identity from config, check the quota, reset turn fields."""
    last = state["messages"][-1]
    clean, found = mask(str(last.content))
    user_id, role = who_is(config)
    left = quota_left(user_id, role)
    log_event("turn_in", user=user_id, role=role, text=clean, pii=found)
    update = {**TURN_RESET, "request": clean, "pii_found": found, "quota_left": left}
    if clean != last.content:
        update["messages"] = [HumanMessage(content=clean, id=last.id)]  # same id: replaces the raw message
    # The quota is NOT checked here: it limits sandbox runs, so it is enforced in run_tests.
    # A student at the limit can still ask for a human, ask an SQL question or chat.
    return update


def _compact(state, config) -> dict:
    """Fold old messages into a summary when the chat gets long (R4)."""
    messages = state["messages"]
    if not SETTINGS["compact"] or len(messages) <= SETTINGS["compact_after"]:
        return {}
    old = messages[:-SETTINGS["keep_recent"]]
    lines = [f"{'user' if isinstance(m, HumanMessage) else 'assistant'}: {str(m.content)[:300]}" for m in old]
    text = (state.get("summary", "") + "\n" + "\n".join(lines)).strip()

    def template():
        asks = [ln[6:60] for ln in lines if ln.startswith("user: ")]
        return AIMessage(content="The student asked about: " + "; ".join(asks[-5:]) + ".")

    answer, _ = ask_model("compact", [SystemMessage(content=SUMMARY_PROMPT), HumanMessage(content=text)],
                          size="small", template=template, cfg=config)
    return {"summary": mask_text(str(answer.content)), "messages": [RemoveMessage(id=m.id) for m in old]}


def triage(state, config):
    """One triage call for the mood signals, then the escalation rules in Python (R3)."""
    update = _compact(state, config)
    summary = update.get("summary", state.get("summary", ""))
    recent = state["messages"][-SETTINGS["keep_recent"]:] if update else state["messages"][-6:]
    context = ([SystemMessage(content=f"Earlier in this conversation: {summary}")] if summary else []) + list(recent)
    prompt = [SystemMessage(content=TRIAGE_PROMPT)] + context + [HumanMessage(content=f"REQUEST: {state['request']}")]
    answer, tier = ask_model("triage", prompt, schema=Triage, size="small",
                             template=lambda: Triage(**rule_triage(state["request"])), cfg=config)
    signals = combine_signals(answer.model_dump(), rule_triage(state["request"]))
    streak = next_unhappy_streak(signals["sentiment"], state.get("unhappy_streak", 0))
    reason = should_escalate(signals, streak)
    log_event("triage", **signals, unhappy_streak=streak, escalate=reason)
    update.update({"triage": signals, "unhappy_streak": streak, "escalate_reason": reason,
                   "served_by": _with_tier(state, "triage", tier)})
    return update


def cache_lookup(state, config):
    """Reuse code + tests: the same user repeating a request, or another user's SAME plain problem (R4)."""
    if not SETTINGS["use_cache"]:
        return {}
    user_id, _ = who_is(config)
    request = state["request"]
    if config.get("configurable", {}).get("fresh"):
        CACHE.forget(user_id, request)  # "Regenerate fresh": do not trust the cache for this request
        log_event("cache_skipped", user=user_id, reason="regenerate fresh")
        return {}
    payload, kind = CACHE.find_repeat(user_id, request), "repeat"
    if payload is None and is_shareable(request, state.get("pii_found", [])):
        payload, kind = CACHE.lookup(request), "semantic"
    if payload is None:
        return {}
    log_event("cache_hit", kind=kind, user=user_id)
    return {"cache_hit": kind, "cache_source": payload.get("matched", ""), "requirements": payload["requirements"],
            "code": payload["code"], "tests": payload["tests"],
            "served_by": {**state.get("served_by", {}), "code_gen": "cache", "test_gen": "cache"}}


def router(state, config):
    """Router/Parser agent: python_function, sql_query or other (structured output)."""
    hint = state.get("triage", {}).get("task_type", "")
    prompt = [SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=f"REQUEST: {state['request']}\nHINT: {hint}")]
    answer, tier = ask_model("router", prompt, schema=RouteDecision, size="small",
                             template=lambda: RouteDecision(**rule_route(state["request"])), cfg=config)
    kind = answer.kind
    if kind == "python_function" and answer.language.strip().lower() not in ("python", "py", ""):
        kind = "other"  # we can only test Python
    update = {"route": kind, "served_by": _with_tier(state, "router", tier)}
    if kind == "other":
        language = answer.language if answer.language.lower() not in ("python", "") else ""
        reason = f"You asked for {language} code, which I cannot test." if language else "This is not a coding request."
        update["refusal"] = {"kind": "out_of_scope", "reason": reason}
    return update


def requirements(state, config):
    """Requirements agent: function name, params, return value, edge cases, examples."""
    request = state["request"]
    prompt = [SystemMessage(content=REQUIREMENTS_PROMPT), HumanMessage(content=f"REQUEST: {request}")]
    answer, tier = ask_model("requirements", prompt, schema=Requirements, size="small",
                             template=lambda: Requirements(**rule_requirements(request)), cfg=config)
    req = answer.model_dump()
    name = req["function_name"].strip()
    if not name.isidentifier() or keyword.iskeyword(name):
        req["function_name"] = guess_function_name(request)
    return {"requirements": req, "served_by": _with_tier(state, "requirements", tier)}


def code_gen(state, config):
    """Code Generation agent: writes solution.py. On a repair it also sees the failing test."""
    request, req = state["request"], state["requirements"]
    name = req["function_name"]
    prompt = f"REQUEST: {request}\nREQUIREMENTS: {json.dumps(req)}"
    if state.get("policy_feedback"):
        prompt += (f"\n\nYOUR PREVIOUS CODE OR TESTS WERE REJECTED BY THE SAFETY POLICY: {state['policy_feedback']}\n"
                   f"Use only the Python standard library (math, re, functools, itertools, collections ...). "
                   f"Do not import sys, os or third-party packages, in the code or in the tests.")
    if state.get("attempts", 0) > 0 and state.get("run"):
        run = state["run"]
        prompt += (f"\n\nYOUR PREVIOUS CODE FAILED ITS TESTS.\nFirst failure: {run.get('first_failure')}\n"
                   f"pytest output:\n{run.get('output', '')[-1500:]}\nPrevious code:\n{state.get('code', '')}")
    template = match_template(request)

    def check(answer):
        code = clean_code(answer.code)
        return GeneratedCode(code=code) if defines_function(code, name) else None

    answer, tier = ask_model("code_gen", [SystemMessage(content=CODE_PROMPT.format(name=name)), HumanMessage(content=prompt)],
                             schema=GeneratedCode, size="big", check=check, cfg=config,
                             template=lambda: GeneratedCode(code=template["code"]) if template else None)
    served = _with_tier(state, "code_gen", tier)
    if answer is None:
        return {"code": "", "served_by": served,
                "refusal": {"kind": "cannot_generate", "reason": "no model answered and no template matches"}}
    update = {"code": mask_code(answer.code), "served_by": served}
    if tier == "template":
        update["requirements"] = template_requirements(template)  # keep code and tests on the same contract
    return update


def test_gen(state, config):
    """Test Generation agent: writes test_solution.py for pytest."""
    request, req = state["request"], state["requirements"]
    name = req["function_name"]
    template = match_template(request)
    if state.get("served_by", {}).get("code_gen") == "template" and template:
        return {"tests": template["tests"], "served_by": _with_tier(state, "test_gen", "template")}

    def check(answer):
        tests = clean_code(answer.tests)
        try:
            ast.parse(tests)
        except SyntaxError:
            return None
        ok = "def test_" in tests and "from solution import" in tests
        return GeneratedTests(tests=tests) if ok else None

    prompt = f"REQUEST: {request}\nREQUIREMENTS: {json.dumps(req)}\nCODE UNDER TEST:\n{state['code']}"
    if state.get("policy_feedback"):
        prompt += (f"\n\nYOUR PREVIOUS TESTS WERE REJECTED BY THE SAFETY POLICY: {state['policy_feedback']}\n"
                   f"Import only pytest, unittest.mock and the standard library. Never import sys or os.")
    use_template = template is not None and template["function_name"] == name
    answer, tier = ask_model("test_gen", [SystemMessage(content=TESTS_PROMPT.format(name=name)), HumanMessage(content=prompt)],
                             schema=GeneratedTests, size="big", check=check, cfg=config,
                             template=lambda: GeneratedTests(tests=template["tests"]) if use_template else None)
    served = _with_tier(state, "test_gen", tier)
    if answer is None:
        return {"served_by": served,
                "refusal": {"kind": "cannot_generate", "reason": "no model answered and no template matches"}}
    return {"tests": mask_code(answer.tests), "served_by": served}


def policy_gate(state, config):
    """Run evaluate_code_policy on the code and the tests (reads the AST, never runs it)."""
    verdict = evaluate_code_and_tests(state["code"], state["tests"])
    user_id, _ = who_is(config)
    log_event("policy", user=user_id, verdict=verdict["verdict"], reasons="; ".join(verdict["reasons"]))
    update = {"policy": verdict, "policy_retry": False}
    if verdict["verdict"] != "deny":
        return update
    fixable = is_model_fixable(verdict)
    # The model chose an unknown import or a bad test file: ask it to rewrite once (no strike).
    # Never for cached or template answers: those would come back the same.
    if fixable and not state.get("policy_feedback") and not state.get("cache_hit") \
            and state.get("served_by", {}).get("code_gen") not in ("template", "cache"):
        log_event("policy_retry", user=user_id, reasons="; ".join(verdict["reasons"]))
        update.update(policy_retry=True, policy_feedback="; ".join(verdict["reasons"]))
        return update
    update["refusal"] = {"kind": "policy_deny", "reason": "; ".join(verdict["reasons"]), "strike": not fixable}
    return update


def human_approval(state, config):
    """Pause with interrupt() until a human decides. Only an instructor's approval counts."""
    decision = interrupt({"type": "code_approval", "request": state["request"], "code": state["code"],
                          "tests": state["tests"], "reasons": state["policy"]["reasons"]})
    if not isinstance(decision, dict):
        decision = {"approved": bool(decision)}
    conf = config.get("configurable", {})
    approver_role = conf.get("approver_role", "")
    approved = bool(decision.get("approved")) and approver_role == "instructor"
    log_event("approval", approved=approved, approver=str(conf.get("approver_id", "")), approver_role=approver_role)
    if approved:
        return {"approved": True}
    reason = "an instructor rejected it" if approver_role == "instructor" else "only an instructor can approve it"
    return {"approved": False, "refusal": {"kind": "approval_rejected", "reason": reason}}


def run_tests(state, config):
    """Idempotency check, then the sandbox, then decide whether to repair."""
    user_id, role = who_is(config)
    key = run_key(user_id, state["code"], state["tests"])
    with run_lock(key):  # a double-click sends two identical jobs: the second waits, then reuses
        existing = find_run(key)
        if existing:
            log_event("run_reused", user=user_id, key=key[:12])
            return {"run": {**existing, "reused": True, "output": ""}, "repair": False}
        # Quota counts every sandbox run, repairs included, and is checked before EACH run.
        if quota_left(user_id, role) <= 0:
            return {"repair": False, "refusal": {"kind": "quota",
                                                 "reason": f"daily quota of {SETTINGS['quota'][role]} runs used"}}
        result = run_in_sandbox(state["code"], state["tests"])
        saved = save_run(user_id, key, state["request"], state["code"], state["tests"], result,
                         state.get("served_by", {}))
    if saved.get("reused"):  # another process stored the same key first: report that run, honestly
        return {"run": {**saved, "output": ""}, "repair": False}
    run = {**saved, "reused": False, "output": result["output"], "timeout": result["timeout"]}
    log_event("run", user=user_id, key=key[:12], exit_code=run["exit_code"], timed_out=run["timed_out"])
    failed = run["exit_code"] != 0 and not run["timed_out"]  # a timeout is reported, not repaired
    attempts = state.get("attempts", 0)
    repair = failed and attempts < SETTINGS["max_repairs"]
    return {"run": run, "repair": repair, "attempts": attempts + 1 if repair else attempts}


PYTEST_EXIT_MEANING = {2: "interrupted", 3: "internal error", 4: "usage error", 5: "no tests collected"}


def report(state, config):
    """The final answer: a template filled from the REAL pytest result. Never model prose."""
    run, req = state["run"], state["requirements"]
    if run.get("timed_out"):
        head = f"⏱ Timed out after {run.get('timeout', SETTINGS['sandbox_timeout']):g} s. The run was killed. No test result."
    elif run.get("exit_code") == 0:
        head = f"✅ Tests passed: {run['passed']} of {run['total']} (pytest exit code 0, {run['duration']} s)."
    elif run.get("exit_code") in PYTEST_EXIT_MEANING:
        head = f"⚠️ The tests could not run (pytest exit code {run['exit_code']}: {PYTEST_EXIT_MEANING[run['exit_code']]})."
    else:
        head = (f"❌ Tests failed: {run['passed']} of {run['total']} passed (pytest exit code {run['exit_code']}). "
                f"First failure: {run.get('first_failure') or 'see output'}.")
    served = state.get("served_by", {})
    tiers = ", ".join(f"{a}={served[a]}" for a in ("code_gen", "test_gen") if a in served)
    lines = [head, f"Function: `{req.get('function_name', '?')}` · Served by: {tiers} · Provider: {PROVIDER}",
             f"Sandbox: {run.get('isolation', '?')} · Run key: {run.get('key', '')[:12]}"]
    if run.get("reused"):
        lines.append("↻ Same code and tests as an earlier run: reused that result, nothing was executed again.")
    if state.get("cache_hit"):
        lines.append(f"♻ Code and tests came from the {state['cache_hit']} cache; the policy gate still checked them. "
                     f"Not what you asked? Use “Regenerate fresh” or 👎.")
    if state.get("attempts"):
        lines.append(f"Repairs used: {state['attempts']} of {SETTINGS['max_repairs']}.")
    passed = run.get("exit_code") == 0
    update = {}
    user_id, _ = who_is(config)
    payload = {"requirements": req, "code": state["code"], "tests": state["tests"]}
    # Same user, same request again (even a failing one): reuse the same code + tests, so the
    # idempotency key matches and nothing runs twice. "Regenerate fresh" skips this.
    CACHE.remember_repeat(user_id, state["request"], payload)
    if passed:
        if SETTINGS["use_cache"] and state.get("cache_hit") != "semantic" and \
                is_shareable(state["request"], state.get("pii_found", [])):
            CACHE.store(state["request"], payload)  # only plain requests are shared with other users
    elif not run.get("timed_out") and repairs_need_human(state.get("attempts", 0), SETTINGS["max_repairs"], passed):
        update["escalate_reason"] = REASON_REPAIRS
        lines.append("I could not fix it in two repairs, so I am handing this to a person.")
    answer = "\n".join(lines) + f"\n\n```python\n{state['code'].rstrip()}\n```"
    update.update({"answer": answer, "messages": [AIMessage(content=answer)]})
    return update


REFUSAL_TEXT = {
    "quota": "⛔ You have used today's run quota ({reason}). Ask an instructor to reset it, or try again tomorrow.",
    "policy_deny": "⛔ Denied by the policy gate.\n{bullets}\nNothing was executed.",
    "approval_rejected": "⛔ Not approved: {reason}. Nothing was executed.",
    "out_of_scope": "{reason} I write and test Python functions, and answer SQL questions for instructors. "
                    "Your own past runs are in the History tab.",
    "cannot_generate": "⚠️ Cannot generate now: no model could produce working code and no built-in template "
                       "matches this request. I have not invented any code.",
}


def refuse(state, config):
    """Every 'no' in one place. Only policy denials count as strikes."""
    refusal = state.get("refusal") or {"kind": "out_of_scope", "reason": ""}
    kind = refusal["kind"]
    bullets = "\n".join(f"- {r}" for r in state.get("policy", {}).get("reasons", []))
    text = REFUSAL_TEXT.get(kind, "⛔ {reason}").format(reason=refusal.get("reason", ""), bullets=bullets)
    strikes = state.get("strikes", 0)
    update = {}
    if kind == "policy_deny" and refusal.get("strike", True):
        strikes += 1
        text += f" (Policy strike {strikes} of 2.)"
        update["strikes"] = strikes
        if denials_need_human(strikes):
            update["escalate_reason"] = REASON_DENIALS
    elif kind == "policy_deny":
        text += " (The model chose this, not you: no policy strike.)"
    if kind == "cannot_generate":
        update["escalate_reason"] = REASON_CANNOT_GENERATE
    log_event("refused", kind=kind, reason=refusal.get("reason", ""))
    update.update({"answer": text, "messages": [AIMessage(content=text)]})
    return update


def ticket(state, config):
    """Open a ticket: the model writes the note, Python sets priority + SLA and masks it again (R3)."""
    user_id, _ = who_is(config)
    signals = state.get("triage") or {}
    reason = state.get("escalate_reason", "")
    priority, sla = priority_for(signals, reason)
    run = state.get("run") or {}
    facts = (f"Reason: {reason}. Request: {state['request']}. Repairs used: {state.get('attempts', 0)}. "
             f"Last failing test: {run.get('first_failure') or 'none'}. "
             f"Policy: {'; '.join(state.get('policy', {}).get('reasons', [])) or 'n/a'}.")
    prompt = [SystemMessage(content=HANDOFF_PROMPT),
              HumanMessage(content=f"REQUEST: {state['request']}\nFACTS: {facts}\nCHAT SUMMARY: {state.get('summary', '')}")]
    note, tier = ask_model("ticket", prompt, schema=Handoff, size="small", cfg=config,
                           template=lambda: Handoff(summary=f"Student needs help. {reason}.", user_ask=state["request"][:200],
                                                    suggested_next_step="Read the request and reply personally."))
    text_note = f"{note.summary} | Ask: {note.user_ask} | Next: {note.suggested_next_step} | {facts}"
    open_ticket = state.get("open_ticket") or None
    if merge_ticket(open_ticket) == "update":
        priority = higher_priority(open_ticket["priority"], priority)
        sla = {"P1": 30, "P2": 240, "P3": 1440}[priority]
        saved = update_ticket(open_ticket["id"], priority, sla, reason, text_note)
    else:
        saved = create_ticket(user_id, priority, sla, reason, text_note)
    log_event("ticket", user=user_id, ticket=saved["id"], priority=priority, reason=reason,
              updated=bool(open_ticket))
    when = {30: "30 minutes", 240: "4 hours", 1440: "24 hours"}[sla]
    if open_ticket:
        message = (f"Your ticket {saved['id']} is already open, so I added this to it (priority {priority}); "
                   f"a person will reply within {when}.")
    elif state.get("answer"):
        message = f"I've also opened ticket {saved['id']} ({priority}): a person will reply within {when}."
    else:
        message = (f"I'm sorry this is frustrating. I've passed it to a person as ticket {saved['id']} "
                   f"(priority {priority}); you'll hear back within {when}. A masked summary is attached, so you "
                   f"won't need to repeat yourself.")
    answer = (state.get("answer", "") + "\n\n" + message).strip()
    return {"ticket": saved, "open_ticket": {"id": saved["id"], "priority": priority, "sla_minutes": sla},
            "answer": answer, "messages": [AIMessage(content=message)],
            "served_by": _with_tier(state, "ticket", tier)}


def sql_agent(state, config):
    """Instructor-only SQL agent: model writes SQL, validate_sql decides, read-only DB runs it (R7)."""
    user_id, role = who_is(config)
    request = state["request"]
    if role != "instructor":
        text = "🔒 The SQL agent is for instructors only. Your role comes from your sign-in, not from the chat."
        return {"sql": {"ok": False, "reason": "instructor only"}, "answer": text,
                "messages": [AIMessage(content=text)]}
    answer, tier = ask_model("sql_agent", [SystemMessage(content=SQL_PROMPT), HumanMessage(content=f"REQUEST: {request}")],
                             schema=SQLQuery, size="big", cfg=config,
                             template=lambda: SQLQuery(sql=rule_sql(request)))
    proposed = answer.sql.strip()
    verdict = validate_sql(proposed)
    log_event("sql", user=user_id, ok=verdict["ok"], reason=verdict["reason"])
    info = {"question": request, "proposed": proposed, **verdict, "columns": [], "rows": []}
    if not verdict["ok"]:
        text = f"⛔ Blocked SQL: {verdict['reason']}.\n```sql\n{proposed}\n```"
    else:
        try:
            info.update(run_sql(verdict["sql"]))
            preview = "\n".join(" | ".join(str(v) for v in row) for row in info["rows"][:10])
            text = (f"```sql\n{verdict['sql']}\n```\n({verdict['reason']}) {len(info['rows'])} rows.\n"
                    f"{' | '.join(info['columns'])}\n{preview}")
        except Exception as err:  # the read-only layer refused it
            info.update(ok=False, reason=f"read-only database refused it: {err}")
            text = f"⛔ Blocked by the read-only database: {err}"
    return {"sql": info, "answer": text, "messages": [AIMessage(content=text)],
            "served_by": _with_tier(state, "sql_agent", tier)}
