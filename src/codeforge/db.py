"""SQLite storage, identity, run history, quota, tickets, feedback and masked logs (R1).

Identity always comes from config["configurable"], never from the chat text.
History functions take NO user argument: they read the user from config.
"""

import json
from pathlib import Path

from sqlalchemy import Boolean, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from . import config
from .masking import mask_text


class Base(DeclarativeBase):
    pass


class Run(Base):
    """One sandbox execution. key = sha256(user, code, tests) is unique: same job runs once."""

    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(64), unique=True)
    user_id: Mapped[str] = mapped_column(String(40), index=True)
    request: Mapped[str] = mapped_column(Text, default="")
    code: Mapped[str] = mapped_column(Text, default="")
    tests: Mapped[str] = mapped_column(Text, default="")
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    passed: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int] = mapped_column(Integer, default=0)
    timed_out: Mapped[bool] = mapped_column(Boolean, default=False)
    duration: Mapped[float] = mapped_column(Float, default=0.0)
    first_failure: Mapped[str] = mapped_column(Text, default="")
    isolation: Mapped[str] = mapped_column(String(20), default="")
    served_by: Mapped[str] = mapped_column(Text, default="{}")
    counts_for_quota: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[float] = mapped_column(Float)


class Ticket(Base):
    __tablename__ = "tickets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(40))
    priority: Mapped[str] = mapped_column(String(4))
    sla_minutes: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(Text)
    note: Mapped[str] = mapped_column(Text)
    created_at: Mapped[float] = mapped_column(Float)


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(40))
    request: Mapped[str] = mapped_column(Text)
    code: Mapped[str] = mapped_column(Text, default="")
    answer: Mapped[str] = mapped_column(Text, default="")
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[float] = mapped_column(Float)


class EventLog(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    t: Mapped[float] = mapped_column(Float)
    kind: Mapped[str] = mapped_column(String(40))
    data: Mapped[str] = mapped_column(Text)


_ENGINE = None
EVENTS: list = []  # in-memory copy of the log, for tests and the UI


def init_db(path: str | None = None):
    """Open (and create) the SQLite database. Tests pass a temp path."""
    global _ENGINE
    path = path or config.SETTINGS["db_path"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _ENGINE = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(_ENGINE)
    return _ENGINE


def session() -> Session:
    """A new database session (opens the default database on first use)."""
    if _ENGINE is None:
        init_db()
    return Session(_ENGINE, expire_on_commit=False)


# ── identity ─────────────────────────────────────────────────────────────────

def who_is(cfg) -> tuple[str, str]:
    """(user_id, role) from config. An unknown role is treated as student (fail closed)."""
    conf = (cfg or {}).get("configurable", {})
    user_id = str(conf.get("user_id", "anonymous"))
    role = conf.get("role", "student")
    return user_id, role if role in ("student", "instructor") else "student"


# ── masked logging ───────────────────────────────────────────────────────────

def log_event(event: str, **fields) -> dict:
    """Save a structured log row. Every text field is masked first."""
    row = {"t": round(config.now(), 3), "event": event}
    for key, value in fields.items():
        row[key] = mask_text(value) if isinstance(value, str) else value
    EVENTS.append(row)
    with session() as s:
        s.add(EventLog(t=row["t"], kind=event, data=json.dumps(row, default=str)))
        s.commit()
    return row


# ── runs, history, quota ─────────────────────────────────────────────────────

def _run_dict(run: Run) -> dict:
    """A run as a plain dict for the graph and the UI."""
    return {
        "run_id": run.id, "key": run.key, "request": run.request, "code": run.code, "tests": run.tests,
        "exit_code": run.exit_code, "passed": run.passed, "failed": run.failed, "total": run.total,
        "timed_out": run.timed_out, "duration": run.duration, "first_failure": run.first_failure,
        "isolation": run.isolation, "served_by": json.loads(run.served_by or "{}"), "created_at": run.created_at,
    }


def find_run(key: str) -> dict | None:
    """The stored run for this idempotency key, if any."""
    with session() as s:
        run = s.scalar(select(Run).where(Run.key == key))
        return _run_dict(run) if run else None


def save_run(user_id: str, key: str, request: str, code: str, tests: str, result: dict, served_by: dict) -> dict:
    """Store one sandbox run and return it with its id."""
    with session() as s:
        run = Run(key=key, user_id=user_id, request=mask_text(request), code=code, tests=tests,
                  exit_code=result["exit_code"], passed=result["passed"], failed=result["failed"],
                  total=result["total"], timed_out=result["timed_out"], duration=result["duration"],
                  first_failure=result["first_failure"], isolation=result["isolation"],
                  served_by=json.dumps(served_by), created_at=config.now())
        s.add(run)
        s.commit()
        return _run_dict(run)


def my_runs(cfg) -> list[dict]:
    """The signed-in user's own runs, newest first. No user argument on purpose."""
    user_id, _ = who_is(cfg)
    with session() as s:
        rows = s.scalars(select(Run).where(Run.user_id == user_id).order_by(Run.id.desc())).all()
        return [_run_dict(r) for r in rows]


def get_run(run_id: int, cfg) -> dict | None:
    """One run, only if it is yours. Missing and not-yours both return None (same answer)."""
    user_id, _ = who_is(cfg)
    with session() as s:
        run = s.scalar(select(Run).where(Run.id == run_id, Run.user_id == user_id))
        return _run_dict(run) if run else None


def runs_used_today(user_id: str) -> int:
    """Sandbox runs this user made in the last 24 hours (what the quota counts)."""
    since = config.now() - 24 * 3600
    with session() as s:
        rows = s.scalars(select(Run).where(Run.user_id == user_id, Run.created_at > since,
                                           Run.counts_for_quota.is_(True))).all()
        return len(rows)


def quota_left(user_id: str, role: str) -> int:
    """How many runs this user may still start today."""
    return max(config.SETTINGS["quota"][role] - runs_used_today(user_id), 0)


# ── tickets and feedback ─────────────────────────────────────────────────────

def create_ticket(user_id: str, priority: str, sla_minutes: int, reason: str, note: str) -> dict:
    """Open a ticket. The note is masked again here, as a last line of defence."""
    with session() as s:
        ticket = Ticket(user_id=user_id, priority=priority, sla_minutes=sla_minutes, reason=reason,
                        note=mask_text(note), created_at=config.now())
        s.add(ticket)
        s.commit()
        return {"id": f"T-{1000 + ticket.id}", "priority": priority, "sla_minutes": sla_minutes,
                "reason": reason, "note": ticket.note, "created_at": ticket.created_at}


def all_tickets() -> list[dict]:
    """Every ticket (for tests and the ops view)."""
    with session() as s:
        return [{"id": f"T-{1000 + t.id}", "user_id": t.user_id, "priority": t.priority,
                 "sla_minutes": t.sla_minutes, "reason": t.reason, "note": t.note}
                for t in s.scalars(select(Ticket)).all()]


def save_feedback(user_id: str, request: str, code: str, answer: str, comment: str) -> int:
    """Save a thumbs-down (masked). Returns its id."""
    with session() as s:
        fb = Feedback(user_id=user_id, request=mask_text(request), code=code, answer=mask_text(answer),
                      comment=mask_text(comment), created_at=config.now())
        s.add(fb)
        s.commit()
        return fb.id


def get_feedback(feedback_id: int) -> dict | None:
    """One feedback row as a dict."""
    with session() as s:
        fb = s.get(Feedback, feedback_id)
        if not fb:
            return None
        return {"id": fb.id, "user_id": fb.user_id, "request": fb.request, "code": fb.code,
                "answer": fb.answer, "comment": fb.comment}


# ── reset (for testing and demos) ────────────────────────────────────────────

def reset_quota(user_id: str | None = None) -> None:
    """Give users their full quota back. Runs stay, so idempotency still works."""
    with session() as s:
        query = select(Run)
        if user_id:
            query = query.where(Run.user_id == user_id)
        for run in s.scalars(query).all():
            run.counts_for_quota = False
        s.commit()


def reset_all() -> None:
    """Delete every run, ticket, feedback row and log line."""
    with session() as s:
        for table in (Run, Ticket, Feedback, EventLog):
            s.query(table).delete()
        s.commit()
    EVENTS.clear()
