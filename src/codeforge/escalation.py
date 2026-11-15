"""Escalation rules (R3). The model reads the mood; these plain rules decide.

Each function is pure and has unit tests in tests/test_escalation.py.
"""

REASON_ASKED_HUMAN = "user asked for a human"
REASON_ANGRY = "user is angry"
REASON_UNHAPPY_URGENT = "unhappy and urgent"
REASON_TWO_UNHAPPY = "two unhappy turns in a row"
REASON_DENIALS = "repeated policy denials"
REASON_REPAIRS = "two repair attempts failed"
REASON_CANNOT_GENERATE = "cannot generate now (models unavailable, no template)"


SENTIMENT_RANK = {"positive": 0, "neutral": 1, "negative": 2, "angry": 3}


def combine_signals(model: dict, rules: dict) -> dict:
    """Model signals, raised (never lowered) by the keyword rules. Code has the last word.

    A model may read "THIS IS NOT WORKING!!! get me a human" as merely "negative" or miss the
    request for a person; shouting and asking for a human are facts plain Python can see.
    """
    out = dict(model)
    if SENTIMENT_RANK.get(rules.get("sentiment"), 0) > SENTIMENT_RANK.get(out.get("sentiment"), 0):
        out["sentiment"] = rules["sentiment"]
    out["urgency"] = max(int(out.get("urgency", 1)), int(rules.get("urgency", 1)))
    out["wants_human"] = bool(out.get("wants_human")) or bool(rules.get("wants_human"))
    return out


def next_unhappy_streak(sentiment: str, streak: int) -> int:
    """Count unhappy turns in a row. A calm or happy turn resets the count."""
    return streak + 1 if sentiment in ("negative", "angry") else 0


def should_escalate(triage: dict, unhappy_streak: int) -> str:
    """Return the reason to hand this turn to a human, or "" to carry on. First match wins."""
    if triage.get("wants_human"):
        return REASON_ASKED_HUMAN
    if triage.get("sentiment") == "angry":
        return REASON_ANGRY
    if triage.get("sentiment") == "negative" and triage.get("urgency", 1) >= 4:
        return REASON_UNHAPPY_URGENT
    if unhappy_streak >= 2:
        return REASON_TWO_UNHAPPY
    return ""


PRIORITY_RANK = {"P1": 1, "P2": 2, "P3": 3}


def merge_ticket(open_ticket: dict | None) -> str:
    """"new" when this chat has no open ticket, "update" when it has one (one ticket per chat).

    A later escalation in the same chat adds to the open ticket (and may raise its priority)
    instead of opening a duplicate.
    """
    return "update" if open_ticket else "new"


def higher_priority(a: str, b: str) -> str:
    """The more urgent of two priorities."""
    return a if PRIORITY_RANK.get(a, 9) <= PRIORITY_RANK.get(b, 9) else b


def denials_need_human(strikes: int) -> bool:
    """Two policy denials in the same chat go to a human."""
    return strikes >= 2


def repairs_need_human(attempts: int, max_repairs: int, passed: bool) -> bool:
    """Every repair was used and the tests still fail."""
    return (not passed) and attempts >= max_repairs


def priority_for(triage: dict, reason: str) -> tuple[str, int]:
    """Return (priority, SLA in minutes). P1 = 30 min, P2 = 4 h, P3 = 24 h.

    P1: unhappy (negative or angry) AND asks for a person, or urgent AND (angry or asks for a person).
    """
    angry = triage.get("sentiment") == "angry"
    unhappy = triage.get("sentiment") in ("angry", "negative")
    wants_human = bool(triage.get("wants_human"))
    urgent = triage.get("urgency", 1) >= 4
    if (unhappy and wants_human) or (urgent and (angry or wants_human)):
        return "P1", 30
    if unhappy or wants_human or reason in (REASON_DENIALS, REASON_REPAIRS):
        return "P2", 240
    return "P3", 1440
