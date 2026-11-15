"""Pending approvals shared by every chat (R2 human approval).

When a run pauses at human_approval, the node registers it here. Any instructor, in any browser
tab, sees the list and decides; the decision resumes the STUDENT's thread, so the run, quota and
history stay the student's while the approval is recorded as the instructor's.

The registry lives in memory, like the MemorySaver checkpointer it mirrors: a restart clears both.
"""

import threading

from . import config

_PENDING: dict = {}
_LOCK = threading.Lock()


def register(thread_id: str, user_id: str, role: str, request: str, code: str, tests: str, reasons: list) -> None:
    """Record a paused run. Called again when the node re-runs on resume: keeps the first timestamp."""
    with _LOCK:
        since = _PENDING.get(thread_id, {}).get("since", config.now())
        _PENDING[thread_id] = {"thread_id": thread_id, "user_id": user_id, "role": role, "request": request,
                               "code": code, "tests": tests, "reasons": list(reasons), "since": since}


def discard(thread_id: str) -> None:
    """Forget a run once it has been decided."""
    with _LOCK:
        _PENDING.pop(thread_id, None)


def pending() -> list[dict]:
    """Every run waiting for an instructor, oldest first."""
    with _LOCK:
        return sorted((dict(item) for item in _PENDING.values()), key=lambda item: item["since"])


def get(thread_id: str) -> dict | None:
    with _LOCK:
        item = _PENDING.get(thread_id)
        return dict(item) if item else None


def claim(thread_id: str) -> dict | None:
    """Take a run off the list. Only the first of two instructors clicking at once gets it."""
    with _LOCK:
        return _PENDING.pop(thread_id, None)


def clear() -> None:
    with _LOCK:
        _PENDING.clear()


def decide(app, thread_id: str, approved: bool, approver_id: str, callbacks=None) -> dict:
    """An instructor approves or rejects a paused run from anywhere. Returns {ok, message, state}.

    The approver's role comes from config.USERS. A student is refused BEFORE the run is claimed,
    so a student can neither approve nor reject someone's run (the node checks the role again).
    """
    from .graph import resume  # imported here: graph -> agents -> approvals would be circular

    role = config.USERS.get(approver_id, "student")
    if role != "instructor":
        return {"ok": False, "message": "Only an instructor can approve or reject a run.", "state": None}
    item = claim(thread_id)
    if item is None:
        return {"ok": False, "message": "This run was already decided, or is no longer waiting.", "state": None}
    try:
        state = resume(app, approved, user_id=item["user_id"], thread=thread_id, approver_id=approver_id,
                       approver_role=role, role=item["role"], callbacks=callbacks)
    except Exception:
        with _LOCK:
            _PENDING.setdefault(thread_id, item)  # nothing was decided: put it back
        raise
    verb = "Approved" if approved else "Rejected"
    return {"ok": True, "message": f"{verb} {item['user_id']}'s run.", "state": state}
