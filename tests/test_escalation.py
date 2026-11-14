"""R3: the escalation rules are plain Python, tested here (not in a prompt)."""

from codeforge.escalation import (REASON_DENIALS, REASON_REPAIRS, denials_need_human, next_unhappy_streak,
                                  priority_for, repairs_need_human, should_escalate)


def t(sentiment="neutral", urgency=1, wants_human=False):
    return {"sentiment": sentiment, "urgency": urgency, "wants_human": wants_human}


def test_calm_request_does_not_escalate():
    assert should_escalate(t(), 0) == ""


def test_asking_for_a_human_escalates():
    assert should_escalate(t(wants_human=True), 0) == "user asked for a human"


def test_angry_user_escalates():
    assert should_escalate(t("angry"), 1) == "user is angry"


def test_one_mildly_unhappy_turn_does_not_escalate():
    streak = next_unhappy_streak("negative", 0)
    assert streak == 1 and should_escalate(t("negative"), streak) == ""


def test_two_mildly_unhappy_turns_escalate():
    streak = next_unhappy_streak("negative", next_unhappy_streak("negative", 0))
    assert should_escalate(t("negative"), streak) == "two unhappy turns in a row"


def test_calm_turn_resets_the_streak():
    assert next_unhappy_streak("neutral", 1) == 0


def test_unhappy_and_urgent_escalates():
    assert should_escalate(t("negative", urgency=5), 1) == "unhappy and urgent"


def test_two_denials_need_a_human():
    assert not denials_need_human(1) and denials_need_human(2)


def test_two_failed_repairs_need_a_human():
    assert repairs_need_human(2, 2, passed=False)
    assert not repairs_need_human(1, 2, passed=False)
    assert not repairs_need_human(2, 2, passed=True)


def test_student_c_is_p1():
    assert priority_for(t("angry", wants_human=True), "user asked for a human") == ("P1", 30)


def test_urgent_human_request_is_p1():
    assert priority_for(t("neutral", urgency=5, wants_human=True), "") == ("P1", 30)


def test_unhappy_or_repeated_denials_are_p2():
    assert priority_for(t("negative"), "")[0] == "P2"
    assert priority_for(t(), REASON_DENIALS)[0] == "P2"
    assert priority_for(t(), REASON_REPAIRS)[0] == "P2"


def test_everything_else_is_p3():
    assert priority_for(t(), "cannot generate now") == ("P3", 1440)
