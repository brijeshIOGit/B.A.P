"""Small HTML builders for the UI. Display only: no decisions are made here."""

import html
import time

# The pipeline as the stepper shows it: (graph node, label).
MAIN_STEPS = [
    ("guard_in", "Guard in"), ("triage", "Triage"), ("cache_lookup", "Cache"), ("router", "Router"),
    ("requirements", "Requirements"), ("code_gen", "Code"), ("test_gen", "Tests"), ("policy_gate", "Policy gate"),
    ("human_approval", "Approval"), ("run_tests", "Sandbox"), ("report", "Report"),
]
EXTRA_STEPS = {"refuse": "Refusal", "ticket": "Ticket", "sql_agent": "SQL agent"}
ICONS = {"pending": "○", "running": "◉", "done": "✓", "failed": "✕", "denied": "✕", "waiting": "⏸", "skipped": "–"}


def esc(text) -> str:
    return html.escape(str(text if text is not None else ""))


def new_steps() -> dict:
    """Every main step starts pending; the approval step is optional."""
    return {"order": [n for n, _ in MAIN_STEPS], "status": {n: "pending" for n, _ in MAIN_STEPS},
            "detail": {}, "repairs": 0}


def stepper_html(steps: dict, finished: bool) -> str:
    """The vertical timeline with a progress bar on top."""
    labels = dict(MAIN_STEPS) | EXTRA_STEPS
    status = dict(steps["status"])
    if finished:
        for node, state in status.items():
            if state in ("pending", "running"):
                status[node] = "skipped"
    done = sum(1 for s in status.values() if s in ("done", "failed", "denied", "waiting", "skipped"))
    percent = 100 if finished else int(100 * done / max(len(status), 1))
    rows = []
    for node in steps["order"]:
        state = status.get(node, "pending")
        detail = steps["detail"].get(node, "")
        badge = f'<span class="cf-badge-mini">↻ repair {steps["repairs"]}/2</span>' \
            if node == "code_gen" and steps["repairs"] else ""
        rows.append(f'<div class="cf-step cf-{state}"><span class="cf-dot">{ICONS[state]}</span>'
                    f'<span class="cf-label">{esc(labels.get(node, node))}{badge}</span>'
                    f'<span class="cf-detail">{esc(detail)}</span></div>')
    return (f'<div class="cf-progress"><div class="cf-bar" style="width:{percent}%"></div></div>'
            f'<div class="cf-stepper">{"".join(rows)}</div>')


def chip(text: str, kind: str) -> str:
    return f'<span class="cf-chip cf-chip-{kind}">{esc(text)}</span>'


def header_html(user: str, role: str, quota_left: int, quota: int, tier: str, chaos_rate: float) -> str:
    """Brand, tagline and the status badges."""
    tier_kind = {"primary": "ok", "backup": "warn", "template": "alert", "cache": "info"}.get(tier, "muted")
    chaos = chip(f"Chaos {chaos_rate:.0%}", "alert") if chaos_rate else chip("Chaos OFF", "muted")
    return (
        '<div class="cf-header"><div class="cf-brand"><span class="cf-logo">⚒</span>'
        '<div><div class="cf-name">CodeForge</div>'
        '<div class="cf-tagline">The model supplies judgement. Code makes the decision.</div></div></div>'
        f'<div class="cf-badges">{chip(f"{user} · {role}", "info")}'
        f'{chip(f"Quota {quota_left}/{quota}", "ok" if quota_left else "alert")}'
        f'{chip(f"Tier: {tier or "–"}", tier_kind)}{chaos}</div></div>'
    )


def policy_html(policy: dict) -> str:
    if not policy:
        return '<div class="cf-empty">Run a request to see the policy verdict and the AST findings.</div>'
    kind = {"allow": "ok", "needs_approval": "warn", "deny": "alert"}[policy["verdict"]]
    label = {"allow": "ALLOW", "needs_approval": "NEEDS APPROVAL", "deny": "DENY"}[policy["verdict"]]
    items = "".join(f"<li>{esc(r)}</li>" for r in policy.get("reasons", [])) or "<li>No risky imports or calls found.</li>"
    return f'<div class="cf-card">{chip(label, kind)}<ul class="cf-list">{items}</ul></div>'


def ticket_html(ticket: dict, reason: str) -> str:
    if not ticket:
        return '<div class="cf-empty">No ticket for this request. Angry users, requests for a human, repeated ' \
               'denials and failed repairs open one.</div>'
    kind = {"P1": "alert", "P2": "warn", "P3": "info"}[ticket["priority"]]
    deadline = int((ticket.get("created_at") or time.time()) + ticket["sla_minutes"] * 60)
    return (f'<div class="cf-card"><div class="cf-row">{chip(ticket["priority"], kind)}'
            f'<b>{esc(ticket["id"])}</b><span>SLA {ticket["sla_minutes"]} min · due in '
            f'<span class="sla-countdown" data-deadline="{deadline}">--:--</span></span></div>'
            f'<p><b>Reason:</b> {esc(reason or ticket.get("reason", ""))}</p>'
            f'<p class="cf-note"><b>Handoff note (masked):</b> {esc(ticket["note"])}</p></div>')


def run_markdown(run: dict, answer: str) -> str:
    if not run:
        return "_Run a request to see the sandbox result here._"
    exit_code = "none (killed at timeout)" if run.get("timed_out") else run.get("exit_code")
    lines = [f"**{answer.splitlines()[0] if answer else ''}**", "",
             "| | |", "|---|---|",
             f"| Passed / total | {run.get('passed', 0)} / {run.get('total', 0)} |",
             f"| pytest exit code | {exit_code} |",
             f"| Duration | {run.get('duration', 0)} s |",
             f"| Sandbox isolation | {run.get('isolation', '')} |",
             f"| Idempotency key | `{run.get('key', '')[:16]}…` |",
             f"| Reused previous run | {'yes, nothing executed again' if run.get('reused') else 'no'} |"]
    if run.get("first_failure"):
        lines.append(f"| First failure | {run['first_failure']} |")
    return "\n".join(lines)


def sql_markdown(sql: dict) -> str:
    if not sql:
        return "_Instructors can ask data questions in the chat, e.g. “total sales by region”._"
    if not sql.get("ok"):
        return f"**⛔ Blocked:** {sql.get('reason', '')}\n\n```sql\n{sql.get('proposed', '')}\n```"
    head = " | ".join(sql["columns"])
    sep = " | ".join("---" for _ in sql["columns"])
    rows = "\n".join(" | ".join(str(v) for v in r) for r in sql["rows"][:50])
    return (f"**Question:** {sql.get('question', '')}\n\n**Validated SQL** ({sql['reason']}):\n```sql\n{sql['sql']}\n```\n\n"
            f"| {head} |\n| {sep} |\n" + "\n".join(f"| {r} |" for r in rows.splitlines()))


def gate_html(results: list[tuple[str, str]]) -> str:
    """Red/green chips, one per golden case."""
    if not results:
        return '<div class="cf-empty">Press “Run golden gate”.</div>'
    chips = "".join(chip(f"{'✓' if status == 'PASSED' else '✕'} {case}", "ok" if status == "PASSED" else "alert")
                    for case, status in results)
    return f'<div class="cf-gate">{chips}</div>'


# Updates every SLA countdown once a second.
HEAD = """<script>
setInterval(() => {
  document.querySelectorAll('.sla-countdown[data-deadline]').forEach(el => {
    const left = Math.max(0, parseInt(el.dataset.deadline) - Math.floor(Date.now() / 1000));
    const m = Math.floor(left / 60), s = left % 60;
    el.textContent = m + ':' + String(s).padStart(2, '0');
  });
}, 1000);
</script>"""
