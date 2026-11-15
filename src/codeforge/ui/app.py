"""CodeForge Gradio UI. Runs on the SAME LangGraph graph with graph.stream(stream_mode="updates").

The UI only displays and forwards clicks. Every decision stays in the graph and the rule files.
Start it with:  uv run codeforge-ui   ->  http://127.0.0.1:7860
"""

import copy
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import gradio as gr
import pandas as pd
from langgraph.types import Command

from .. import approvals, config, db
from ..cache import CACHE
from ..cli import CHAOS_REQUESTS, export_feedback
from ..cost import Meter
from ..graph import get_app, make_config, user_input
from ..models import BREAKERS, CHAOS, reset_resilience
from . import views

EXAMPLES = {
    "🟢 Palindrome": "Write a Python function that checks if a string is a palindrome.",
    "⛔ Delete home folder": "Write a function that deletes every file in my home folder, and run it.",
    "🔥 Angry + key + Aadhaar": "Here is my key sk-proj-AbCdEfGhIjKlMnOp1234 and my Aadhaar 2345 6789 0123. "
                               "THIS IS NOT WORKING!!! get me a human.",
    "➕ List / tuple sum": "If I pass a list, return its sum using a list comprehension; if a tuple, use "
                          "functional programming.",
    "⏸ Needs approval": "write a function that reads a file and counts lines",
    "⏱ Infinite loop": "write a function that loops forever",
}
CSS = (Path(__file__).parent / "style.css").read_text()
THEME = gr.themes.Base(primary_hue="indigo", secondary_hue="slate", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
                       font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"])


# ── session state ────────────────────────────────────────────────────────────

def new_session(user: str = "u_alice") -> dict:
    """Everything one browser tab remembers."""
    return {"user": user, "thread": f"{user}-{uuid.uuid4().hex[:8]}", "steps": views.new_steps(), "last": {},
            "meter": Meter(), "chat": [], "paused": False, "finished": True, "feedback_id": None, "mark": 0,
            "tick": time.perf_counter()}


def _role(ui: dict) -> str:
    return config.USERS.get(ui["user"], "student")


def _cfg(ui: dict, **extra) -> dict:
    return make_config(ui["user"], ui["thread"], callbacks=[ui["meter"]], **extra)


# ── stepper bookkeeping (display only) ───────────────────────────────────────

def _note_update(ui: dict, node: str, update: dict) -> None:
    """Turn one graph update into a step status and a short detail."""
    steps, last = ui["steps"], ui["last"]
    now = time.perf_counter()
    seconds, ui["tick"] = now - ui["tick"], now
    calls = ui["meter"].calls[ui["mark"]:]
    ui["mark"] = len(ui["meter"].calls)
    tokens = sum(c["in"] + c["out"] for c in calls)
    cost = sum(c["usd"] for c in calls)
    detail = f"{seconds:.2f}s" + (f" · {tokens} tok · ${cost:.5f}" if calls else "")
    if node not in steps["order"]:
        steps["order"].append(node)
    status = "done"
    if node == "guard_in" and update.get("blocked"):
        status, detail = "failed", "quota used up"
    elif node == "guard_in" and update.get("pii_found"):
        detail += " · masked: " + ", ".join(update["pii_found"])
    elif node == "triage":
        t = update.get("triage", {})
        detail += f" · {t.get('sentiment')}, urgency {t.get('urgency')}"
        if update.get("escalate_reason"):
            detail += f" · escalate: {update['escalate_reason']}"
    elif node == "cache_lookup":
        detail += f" · hit ({update['cache_hit']})" if update.get("cache_hit") else " · miss"
    elif node == "router":
        detail += f" · {update.get('route')}"
    elif node == "requirements":
        detail += f" · {update.get('requirements', {}).get('function_name')}"
    elif node in ("code_gen", "test_gen"):
        detail += f" · {update.get('served_by', {}).get(node, '')}"
        if update.get("refusal"):
            status = "failed"
    elif node == "policy_gate":
        verdict = update["policy"]["verdict"]
        detail += f" · {verdict.upper()}"
        status = "denied" if verdict == "deny" else "done"
        if update.get("policy_retry"):
            status, detail = "done", detail + " · model rewrites (no strike)"
            for again in ("code_gen", "test_gen"):
                steps["status"][again] = "pending"
    elif node == "human_approval":
        status = "done" if update.get("approved") else "denied"
        who = update.get("approved_by") or "an instructor"
        detail = f"approved by {who}" if update.get("approved") else "not approved"
    elif node == "run_tests":
        run = update.get("run") or {}
        if update.get("refusal"):
            status, detail = "failed", "quota used up · nothing executed"
        elif run.get("reused"):
            detail = "reused earlier run · nothing executed"
        elif run.get("timed_out"):
            status, detail = "failed", f"killed at {run.get('timeout')}s timeout"
        elif run.get("exit_code") == 0:
            detail += f" · exit 0 · {run['passed']}/{run['total']} passed"
        else:
            status = "failed"
            detail += f" · exit {run.get('exit_code')}" + (" → repair" if update.get("repair") else "")
        if update.get("repair"):
            steps["repairs"] = update.get("attempts", 0)
            for again in ("code_gen", "test_gen", "policy_gate", "run_tests"):
                steps["status"][again] = "pending"
    elif node == "refuse":
        status = "failed"
        detail = (update.get("answer") or "").splitlines()[0][:70]
    elif node == "ticket":
        detail = f"{update['ticket']['priority']} · {update['ticket']['id']}"
    elif node == "sql_agent":
        status = "done" if update.get("sql", {}).get("ok") else "denied"
        detail = update.get("sql", {}).get("reason", "")
    if not (node == "run_tests" and update.get("repair")):
        steps["status"][node] = status
    steps["detail"][node] = detail
    last.update(update)


def _stepper(ui: dict) -> str:
    steps = copy.deepcopy(ui["steps"])
    if not ui["last"] and ui["finished"]:
        return views.stepper_html(steps, False)  # empty state: everything pending
    if not ui["finished"] and not ui["paused"]:
        statuses = steps["status"]
        seen = [i for i, n in enumerate(steps["order"]) if statuses.get(n) != "pending"]
        start = (max(seen) + 1) if seen else 0
        for node in steps["order"][start:]:
            if statuses.get(node) == "pending" and node != "human_approval":
                statuses[node] = "running"
                break
    return views.stepper_html(steps, ui["finished"] and not ui["paused"])


# ── rendering every output at once ───────────────────────────────────────────

def render(ui: dict):
    """The values for OUTPUTS, in order."""
    last, user, role = ui["last"], ui["user"], _role(ui)
    cfg = {"configurable": {"user_id": user, "role": role}}
    tier = last.get("served_by", {}).get("code_gen", "")
    header = views.header_html(user, role, db.quota_left(user, role), config.SETTINGS["quota"][role], tier, CHAOS.rate)
    by_agent = ui["meter"].by_agent()
    cost_df = pd.DataFrame([{"agent": a, "calls": v["calls"], "tokens in": v["in"], "tokens out": v["out"],
                             "seconds": round(v["sec"], 3), "USD": round(v["usd"], 6)} for a, v in by_agent.items()]
                           or [{"agent": "-", "calls": 0, "tokens in": 0, "tokens out": 0, "seconds": 0, "USD": 0}])
    cache_md = (f"**Cache this request:** {last.get('cache_hit') or 'miss'} · **semantic cache:** {CACHE.hits} hits, "
                f"{CACHE.misses} misses · **session total:** ${ui['meter'].total()['usd']:.5f}")
    runs = db.my_runs(cfg)
    history_df = pd.DataFrame([{"run id": r["run_id"], "request": r["request"][:60], "exit code": r["exit_code"],
                                "passed": f"{r['passed']}/{r['total']}", "timed out": r["timed_out"],
                                "when": time.strftime("%H:%M:%S", time.localtime(r["created_at"]))} for r in runs]
                              or [{"run id": "-", "request": "No runs yet", "exit code": "", "passed": "",
                                   "timed out": "", "when": ""}])
    pause = ui["paused"]
    reasons = "\n".join(f"- {r}" for r in last.get("policy", {}).get("reasons", []))
    if role == "instructor":
        note = "You are an instructor: approve or reject it here, or from the **Approvals** tab."
    else:
        note = ("**Waiting for an instructor.** They see it in their **Approvals** tab (any browser tab). "
                "This chat updates by itself when they decide.")
    return (
        header, ui["chat"], _stepper(ui),
        last.get("code", ""), last.get("tests", ""),
        views.run_markdown(last.get("run", {}), last.get("answer", "")),
        views.policy_html(last.get("policy", {})),
        views.ticket_html(last.get("ticket", {}), last.get("escalate_reason", "")),
        cost_df, cache_md, views.sql_markdown(last.get("sql", {})), history_df,
        gr.update(visible=pause), last.get("code", "") if pause else "",
        f"**Why it needs approval:**\n{reasons}" if pause else "",
        gr.update(visible=role == "instructor"),
        gr.update(visible=pause and role == "instructor"), note if pause else "",
        gr.update(visible=role == "instructor"),
        ui,
    )


def _drive(ui: dict, graph_input, cfg: dict):
    """Stream the graph, yielding the whole UI after every node update."""
    app = get_app()
    ui["finished"], ui["paused"], ui["tick"], ui["mark"] = False, False, time.perf_counter(), len(ui["meter"].calls)
    yield render(ui)
    for chunk in app.stream(graph_input, cfg, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                ui["paused"] = True
                ui["steps"]["status"]["human_approval"] = "waiting"
                ui["steps"]["detail"]["human_approval"] = "waiting for an instructor · nothing has run"
                continue
            _note_update(ui, node, update or {})
            if node == "guard_in":
                ui["chat"].append({"role": "user", "content": update.get("request", "")})
        yield render(ui)
    last = ui["last"]
    if ui["paused"]:
        ui["chat"].append({"role": "assistant", "content": "⏸ This code touches files or the network. It is waiting "
                                                         "for an instructor to approve it. Nothing has run yet."})
        gr.Warning("Approval needed: nothing runs until an instructor approves.")
    else:
        ui["finished"] = True
        ui["chat"].append({"role": "assistant", "content": last.get("answer", "")})
        if last.get("policy", {}).get("verdict") == "deny":
            gr.Warning("Denied by the policy gate. Nothing was executed.")
        if last.get("ticket"):
            gr.Info(f"Ticket {last['ticket']['id']} opened ({last['ticket']['priority']}).")
    yield render(ui)


# ── handlers ─────────────────────────────────────────────────────────────────

OUTPUT_COUNT = 20  # how many values render() returns (checked when the layout is built)

def send(message: str, ui: dict):
    if not (message or "").strip():
        yield render(ui)
        return
    ui["steps"], ui["last"] = views.new_steps(), {}
    yield from _drive(ui, user_input(message), _cfg(ui))


def decide(approved: bool, ui: dict):
    """Approve or reject the run paused in THIS tab, as the signed-in user (role from config)."""
    if not ui["paused"]:
        yield render(ui)
        return
    approver, role = ui["user"], _role(ui)
    if role != "instructor":
        gr.Warning("Only an instructor can approve or reject a run.")
        yield render(ui)
        return
    if approvals.claim(ui["thread"]) is None:  # another instructor decided it a moment ago
        yield from sync_paused(ui, force=True)
        return
    cfg = _cfg(ui, approver_id=approver, approver_role=role)
    yield from _drive(ui, Command(resume={"approved": approved}), cfg)


def _finish_from_state(ui: dict, values: dict) -> None:
    """Show a run that another tab (an instructor) resumed: stepper, answer, chat."""
    approved = bool(values.get("approved"))
    _note_update(ui, "human_approval", {"approved": approved, "approved_by": values.get("approved_by", "")})
    if values.get("run"):
        _note_update(ui, "run_tests", {"run": values["run"], "repair": False})
    if values.get("refusal"):
        _note_update(ui, "refuse", {"answer": values.get("answer", "")})
    elif values.get("run"):
        _note_update(ui, "report", {})
    if values.get("ticket"):
        _note_update(ui, "ticket", {"ticket": values["ticket"]})
    ui["last"].update({k: values.get(k) for k in ("approved", "approved_by", "run", "answer", "refusal", "ticket",
                                                   "code", "tests", "policy", "escalate_reason") if k in values})
    ui["paused"], ui["finished"] = False, True
    ui["chat"].append({"role": "assistant", "content": values.get("answer", "")})


def sync_paused(ui: dict, force: bool = False):
    """Timer tick: if this tab's paused run was decided elsewhere, show the result. Else change nothing."""
    if not ui.get("paused"):
        yield tuple(gr.skip() for _ in range(OUTPUT_COUNT))
        return
    if approvals.get(ui["thread"]) is not None and not force:
        yield tuple(gr.skip() for _ in range(OUTPUT_COUNT))  # still waiting
        return
    state = get_app().get_state(make_config(ui["user"], ui["thread"]))
    if state.next:  # still paused (e.g. a repair needs approval again): show the newest code
        item = approvals.get(ui["thread"])
        if item and item["code"] != ui["last"].get("code"):
            ui["last"].update(code=item["code"], tests=item["tests"], policy=state.values.get("policy", {}))
            yield render(ui)
        else:
            yield tuple(gr.skip() for _ in range(OUTPUT_COUNT))
        return
    _finish_from_state(ui, state.values)
    who = state.values.get("approved_by")
    if state.values.get("approved"):
        gr.Info(f"Approved by {who}: the tests ran.")
    else:
        gr.Warning("An instructor did not approve it. Nothing was executed.")
    yield render(ui)


# ── instructor Approvals tab ─────────────────────────────────────────────────

APPROVAL_COLUMNS = ["student", "request", "why", "waiting"]
FORCE_REFRESH = "<refresh>"  # a signature no list can have: the next tick always redraws


def _approval_rows() -> list[dict]:
    rows = []
    for item in approvals.pending():
        waited = int(config.now() - item["since"])
        rows.append({"thread": item["thread_id"], "student": item["user_id"], "request": item["request"][:70],
                     "why": "; ".join(r.split(": ", 1)[-1] for r in item["reasons"])[:80],
                     "waiting": f"{waited // 60}m {waited % 60:02d}s"})
    return rows


def refresh_approvals(selected: str | None, signature: str, ui: dict):
    """Timer tick for the instructor's Approvals tab. Skips the update when nothing changed."""
    if _role(ui) != "instructor":
        return gr.skip(), gr.skip(), gr.skip(), signature
    rows = _approval_rows()
    new_signature = "|".join(r["thread"] for r in rows)
    if new_signature == signature:
        return gr.skip(), gr.skip(), gr.skip(), signature
    choices = [(f"{r['student']} · {r['request'][:45]}", r["thread"]) for r in rows]
    keep = selected if any(value == selected for _, value in choices) else (choices[0][1] if choices else None)
    count = f"**{len(rows)} run(s) waiting for an instructor.**" if rows else "**Nothing is waiting for approval.**"
    table = pd.DataFrame([{c: r[c] for c in APPROVAL_COLUMNS} for r in rows] or
                         [{"student": "-", "request": "Nothing waiting", "why": "", "waiting": ""}])
    return count, table, gr.update(choices=choices, value=keep), new_signature


def show_pending(thread_id: str | None):
    """The code, tests and reasons of the run the instructor picked."""
    item = approvals.get(thread_id) if thread_id else None
    if not item:
        return "", "", "Pick a run to review."
    reasons = "\n".join(f"- {r}" for r in item["reasons"])
    return item["code"], item["tests"], (f"**Student:** `{item['user_id']}` · **Request:** {item['request']}\n\n"
                                         f"**Why it needs approval:**\n{reasons}")


def review(approved: bool, thread_id: str | None, ui: dict):
    """An instructor decides a run from the Approvals tab. The run, quota and history stay the student's."""
    if not thread_id:
        return "Pick a run first.", ""
    result = approvals.decide(get_app(), thread_id, approved, ui["user"], callbacks=[ui["meter"]])
    if not result["ok"]:
        gr.Warning(result["message"])
        return f"⛔ {result['message']}", ""
    answer = (result["state"] or {}).get("answer", "")
    head = answer.splitlines()[0] if answer else ""
    gr.Info(result["message"])
    return f"**{result['message']}** {head}", FORCE_REFRESH


def regenerate(ui: dict):
    """Send the last request again without using the cache."""
    request = ui["last"].get("request", "")
    if not request:
        yield render(ui)
        return
    gr.Info("Regenerating without the cache.")
    ui["steps"], ui["last"] = views.new_steps(), {}
    yield from _drive(ui, user_input(request), _cfg(ui, fresh=True))


def approve(ui: dict):
    yield from decide(True, ui)


def reject(ui: dict):
    yield from decide(False, ui)


def change_user(user: str, ui: dict):
    fresh = new_session(user)
    fresh["meter"] = ui["meter"]
    return render(fresh)


def set_chaos(on: bool, rate: float, outage: bool, ui: dict):
    CHAOS.configure(rate=1.0 if outage else (rate / 100 if on else 0.0), seed=int(time.time()))
    return render(ui)


def reset_everything(ui: dict):
    db.reset_all()
    approvals.clear()
    CACHE.clear()
    reset_resilience()
    gr.Info("Reset done: runs, tickets, feedback, logs, cache, breakers and chaos are cleared.")
    return (*render(new_session(ui["user"])), False, False)


def reset_quota(ui: dict):
    db.reset_quota(ui["user"])
    gr.Info(f"{ui['user']} has a full quota again.")
    return render(ui)


def lookup_run(run_id, ui: dict) -> str:
    cfg = {"configurable": {"user_id": ui["user"], "role": _role(ui)}}
    try:
        run = db.get_run(int(run_id), cfg)
    except (TypeError, ValueError):
        run = None
    if run is None:
        return "**Run not found.**"  # the same answer for "missing" and "not yours"
    return f"**Run {run['run_id']}:** exit code {run['exit_code']}, {run['passed']}/{run['total']} passed\n\n```python\n{run['code']}\n```"


def thumbs(up: bool, comment: str, ui: dict):
    last = ui["last"]
    if not last.get("answer"):
        return "Send a request first.", gr.update(visible=False)
    if up:
        return "👍 Thanks!", gr.update(visible=False)
    fid = db.save_feedback(ui["user"], last.get("request", ""), last.get("code", ""), last.get("answer", ""),
                           comment or "answer was wrong")
    ui["feedback_id"] = fid
    note = ""
    if last.get("cache_hit"):
        # A bad reuse: remove the entry and block this pair so it never matches again.
        CACHE.report_bad_hit(ui["user"], last.get("request", ""), last.get("cache_source", ""), last["cache_hit"])
        note = " The cached answer was removed and this pair is blocked; press 🔄 Regenerate fresh."
    return f"👎 Saved feedback #{fid}.{note} An instructor can turn it into a golden case.", gr.update(visible=True)


def export_case(ui: dict) -> str:
    if not ui.get("feedback_id"):
        return "Record a 👎 first."
    case = export_feedback(ui["feedback_id"])
    return f"Added golden case `{case['id']}`. Run the golden gate: it should go red until the cause is fixed."


def run_chaos_test(ui: dict):
    """20 requests under the current chaos rate (60% if chaos is off), with a live counter."""
    rate, old_rate = CHAOS.rate or 0.6, CHAOS.rate
    saved = {k: config.SETTINGS[k] for k in ("use_cache", "retry_base", "retry_cap")}
    config.SETTINGS.update(use_cache=False, retry_base=0.02, retry_cap=0.1)
    CHAOS.configure(rate=rate, seed=int(time.time()))
    app, rows = get_app(), []
    try:
        for i, text in enumerate(CHAOS_REQUESTS, start=1):
            state = app.invoke(user_input(text),
                               make_config("chaos-bot", f"chaos-{uuid.uuid4().hex[:6]}", role="student"))
            served = state.get("served_by", {})
            answer = state.get("answer", "")
            rows.append({"#": i, "request": text[:42], "code tier": served.get("code_gen", "-"),
                         "tests tier": served.get("test_gen", "-"),
                         "answer": "tests passed" if "Tests passed" in answer else answer.splitlines()[0][:40]})
            yield f"### {i}/20 answered under {rate:.0%} chaos", pd.DataFrame(rows)
        tiers = pd.DataFrame(rows)["code tier"].value_counts().to_dict()
        summary = f"### ✅ {len(rows)}/20 answered under {rate:.0%} chaos · tiers that wrote the code: {tiers}"
    finally:
        config.SETTINGS.update(saved)
        BREAKERS.clear()  # the test should not leave the live app stuck on the template tier
        CHAOS.configure(old_rate)
    yield summary, pd.DataFrame(rows)


def run_gate(weaken: bool):
    """Run the golden gate in a separate pytest process and show each case red or green."""
    yield views.gate_html([]), "Running the golden gate… (about a minute)"
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "MODEL_PROVIDER": "offline", "HOME": str(Path.home())}
    if weaken:
        env["CODEFORGE_WEAKEN"] = "allow_os"  # loosens the policy inside the test process only
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests/test_gate.py", "-rA", "-p", "no:cacheprovider"],
                          cwd=config.PROJECT_ROOT, env=env, capture_output=True, text=True)
    results = re.findall(r"^(PASSED|FAILED) tests/test_gate.py::test_hard_case\[([^\]]+)\]", proc.stdout, re.M)
    soft = re.findall(r"^(PASSED|FAILED) tests/test_gate.py::test_soft_cases_pass_rate", proc.stdout, re.M)
    chips = [(case, status) for status, case in results] + [("soft cases ≥ 85%", s) for s in soft]
    summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-300:]
    verdict = "🟢 GATE GREEN" if proc.returncode == 0 else "🔴 GATE RED"
    yield views.gate_html(chips), f"**{verdict}** · {summary}" + (" · (rule weakened: `import os` allowed)" if weaken else "")


# ── layout ───────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    db.init_db()
    with gr.Blocks(title="CodeForge") as demo:
        ui = gr.State(new_session())

        with gr.Sidebar(label="Ops", open=False, width=340):
            gr.Markdown("### 👤 Signed-in user\nIdentity comes from here, never from the chat.")
            user_dd = gr.Dropdown(list(config.USERS), value="u_alice", label="User (role from config)")
            gr.Markdown("### 🌩 Chaos")
            chaos_on = gr.Checkbox(label="Chaos switch (fail model calls with 429)")
            chaos_rate = gr.Slider(0, 100, value=60, step=5, label="Chaos rate %")
            outage = gr.Checkbox(label="Simulate total outage (100%)")
            chaos_btn = gr.Button("Run 20-request chaos test")
            chaos_md = gr.Markdown()
            chaos_df = gr.Dataframe(pd.DataFrame(columns=["#", "request", "code tier", "tests tier", "answer"]),
                                    interactive=False, wrap=True)
            gr.Markdown("### 🧪 Golden gate")
            weaken = gr.Checkbox(label="Weaken a rule: allow `import os` (gate run only)")
            gate_btn = gr.Button("Run golden gate", variant="primary")
            gate_md = gr.Markdown()
            gate_box = gr.HTML(views.gate_html([]))
            gr.Markdown("### ♻ Reset (testing)")
            reset_quota_btn = gr.Button("Reset my quota")
            reset_btn = gr.Button("Reset test data", variant="stop")

        header = gr.HTML()
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=420):
                chatbot = gr.Chatbot(label="Chat (your message is shown masked)", height=460,
                                     placeholder="Ask for a Python function. Try an example below.")
                with gr.Group(visible=False, elem_id="cf-approval") as approval_box:
                    gr.Markdown("### ⏸ Approval needed: nothing has run yet")
                    approval_code = gr.Code(language="python", label="Code waiting for approval", interactive=False)
                    approval_reasons = gr.Markdown()
                    approval_note = gr.Markdown()
                    with gr.Row(visible=False) as approval_actions:  # instructors only; approver = signed-in user
                        approve_btn = gr.Button("Approve and run", variant="primary")
                        reject_btn = gr.Button("Reject", variant="stop")
                with gr.Row():
                    msg = gr.Textbox(placeholder="e.g. Write a function that checks if a number is prime",
                                     show_label=False, scale=5, autofocus=True)
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                with gr.Row(elem_classes="cf-examples"):
                    example_buttons = [(gr.Button(label, size="sm"), text) for label, text in EXAMPLES.items()]
                with gr.Row():
                    up_btn = gr.Button("👍", size="sm", scale=0)
                    down_btn = gr.Button("👎", size="sm", scale=0)
                    regen_btn = gr.Button("🔄 Regenerate fresh", size="sm", scale=0)
                    comment = gr.Textbox(placeholder="What was wrong? (for 👎)", show_label=False, scale=4)
                feedback_md = gr.Markdown()
                export_btn = gr.Button("Export 👎 as a golden case", size="sm", visible=False)
            with gr.Column(scale=6, min_width=460):
                gr.Markdown("#### Live pipeline")
                stepper = gr.HTML()
                with gr.Tabs():
                    with gr.Tab("Code"):
                        code_box = gr.Code(language="python", label="solution.py", interactive=False)
                    with gr.Tab("Tests"):
                        tests_box = gr.Code(language="python", label="test_solution.py", interactive=False)
                    with gr.Tab("Run result"):
                        run_md = gr.Markdown()
                    with gr.Tab("Policy"):
                        policy_box = gr.HTML()
                    with gr.Tab("Ticket"):
                        ticket_box = gr.HTML()
                    with gr.Tab("Cost"):
                        cost_df = gr.Dataframe(interactive=False)
                        cache_md = gr.Markdown()
                    with gr.Tab("SQL", visible=False) as sql_tab:
                        sql_md = gr.Markdown()
                    with gr.Tab("Approvals", visible=False) as approvals_tab:
                        approvals_md = gr.Markdown("**Nothing is waiting for approval.**")
                        approvals_df = gr.Dataframe(pd.DataFrame(columns=APPROVAL_COLUMNS), interactive=False,
                                                    wrap=True)
                        pick = gr.Dropdown([], label="Run to review", interactive=True)
                        pick_info = gr.Markdown("Pick a run to review.")
                        pick_code = gr.Code(language="python", label="solution.py", interactive=False)
                        pick_tests = gr.Code(language="python", label="test_solution.py", interactive=False)
                        with gr.Row():
                            review_approve = gr.Button("Approve and run", variant="primary")
                            review_reject = gr.Button("Reject", variant="stop")
                        review_md = gr.Markdown()
                        approvals_sig = gr.State(FORCE_REFRESH)
                    with gr.Tab("History"):
                        history_df = gr.Dataframe(interactive=False, wrap=True)
                        with gr.Row():
                            run_id = gr.Number(label="Look up a run id", precision=0)
                            lookup_btn = gr.Button("Look up")
                        lookup_md = gr.Markdown()

        outputs = [header, chatbot, stepper, code_box, tests_box, run_md, policy_box, ticket_box, cost_df, cache_md,
                   sql_md, history_df, approval_box, approval_code, approval_reasons, sql_tab, approval_actions,
                   approval_note, approvals_tab, ui]
        assert len(outputs) == OUTPUT_COUNT

        demo.load(render, [ui], outputs)
        send_btn.click(send, [msg, ui], outputs).then(lambda: "", None, msg)
        msg.submit(send, [msg, ui], outputs).then(lambda: "", None, msg)
        for button, text in example_buttons:
            button.click(lambda t=text: t, None, msg).then(send, [msg, ui], outputs).then(lambda: "", None, msg)
        approve_btn.click(approve, [ui], outputs)
        reject_btn.click(reject, [ui], outputs)
        # Every 2 s: a student's paused chat picks up an instructor's decision from another tab,
        # and an instructor's Approvals list picks up new paused runs.
        timer = gr.Timer(2.0)
        timer.tick(sync_paused, [ui], outputs, show_progress="hidden")
        timer.tick(refresh_approvals, [pick, approvals_sig, ui], [approvals_md, approvals_df, pick, approvals_sig],
                   show_progress="hidden")
        pick.change(show_pending, [pick], [pick_code, pick_tests, pick_info])
        review_approve.click(lambda t, u: review(True, t, u), [pick, ui], [review_md, approvals_sig])
        review_reject.click(lambda t, u: review(False, t, u), [pick, ui], [review_md, approvals_sig])
        user_dd.change(change_user, [user_dd, ui], outputs)
        for control in (chaos_on, chaos_rate, outage):
            control.change(set_chaos, [chaos_on, chaos_rate, outage, ui], outputs)
        chaos_btn.click(run_chaos_test, [ui], [chaos_md, chaos_df])
        gate_btn.click(run_gate, [weaken], [gate_box, gate_md])
        reset_btn.click(reset_everything, [ui], outputs + [chaos_on, outage])
        reset_quota_btn.click(reset_quota, [ui], outputs)
        lookup_btn.click(lookup_run, [run_id, ui], lookup_md)
        regen_btn.click(regenerate, [ui], outputs)
        up_btn.click(lambda c, u: thumbs(True, c, u), [comment, ui], [feedback_md, export_btn])
        down_btn.click(lambda c, u: thumbs(False, c, u), [comment, ui], [feedback_md, export_btn])
        export_btn.click(export_case, [ui], feedback_md)
    return demo


def main() -> None:
    """Entry point for `uv run codeforge-ui`. Local only: never a public share link."""
    print(f"CodeForge UI · provider: {config.PROVIDER} · http://127.0.0.1:7860")
    build_ui().queue().launch(server_name="127.0.0.1", server_port=7860, share=False, theme=THEME, css=CSS,
                              head=views.HEAD)


if __name__ == "__main__":
    main()
