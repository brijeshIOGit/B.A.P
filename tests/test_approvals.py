"""R2 approval across tabs: a student's paused run is decided by an instructor from anywhere."""

from codeforge import approvals, db
from codeforge.graph import chat, is_paused, make_config

FILE_TASK = "write a function that reads a file and counts lines"


def runs_of(user):
    return db.my_runs(make_config(user, "any"))


def test_paused_run_is_listed_for_instructors(app):
    state = chat(app, FILE_TASK, user_id="u_alice", thread="alice-1")
    assert is_paused(state)
    [item] = approvals.pending()
    assert item["thread_id"] == "alice-1" and item["user_id"] == "u_alice" and "open" in " ".join(item["reasons"])
    assert runs_of("u_alice") == []  # nothing ran yet


def test_instructor_approves_from_another_tab(app):
    chat(app, FILE_TASK, user_id="u_alice", thread="alice-2")
    result = approvals.decide(app, "alice-2", True, approver_id="u_carol")
    assert result["ok"] and result["state"]["run"]["exit_code"] == 0
    assert result["state"]["approved_by"] == "u_carol"
    assert approvals.pending() == []
    # The run belongs to the student (their history and quota), not to the instructor.
    assert len(runs_of("u_alice")) == 1 and runs_of("u_carol") == []
    # The student's own thread now shows the finished result.
    student_view = app.get_state(make_config("u_alice", "alice-2"))
    assert not student_view.next and "Tests passed" in student_view.values["answer"]


def test_instructor_rejects_and_nothing_runs(app):
    chat(app, FILE_TASK, user_id="u_alice", thread="alice-3")
    result = approvals.decide(app, "alice-3", False, approver_id="u_carol")
    assert result["ok"] and "Not approved" in result["state"]["answer"]
    assert runs_of("u_alice") == [] and approvals.pending() == []


def test_student_cannot_approve_or_reject_someone_elses_run(app):
    chat(app, FILE_TASK, user_id="u_alice", thread="alice-4")
    for approved in (True, False):
        result = approvals.decide(app, "alice-4", approved, approver_id="u_bob")
        assert not result["ok"] and "Only an instructor" in result["message"]
    assert len(approvals.pending()) == 1 and runs_of("u_alice") == []  # still waiting, untouched


def test_second_instructor_click_is_refused(app):
    chat(app, FILE_TASK, user_id="u_alice", thread="alice-5")
    assert approvals.decide(app, "alice-5", True, approver_id="u_carol")["ok"]
    again = approvals.decide(app, "alice-5", True, approver_id="u_carol")
    assert not again["ok"] and "already decided" in again["message"]
    assert len(runs_of("u_alice")) == 1


def test_claim_is_first_come_first_served():
    approvals.register("t", "u_alice", "student", "req", "code", "tests", ["r"])
    assert approvals.claim("t") is not None and approvals.claim("t") is None
