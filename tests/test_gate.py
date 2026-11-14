"""R6: the golden gate. Hard cases must pass 100%; soft cases at least 85%.

A failing hard case is named in the test id, e.g. test_hard_case[H-policy-deny-os].
"""

import pytest

from codeforge.evals import load_cases, run_case, soft_pass_rate

CASES = load_cases()
HARD = [c for c in CASES if c["kind"] == "hard"]
SOFT = [c for c in CASES if c["kind"] == "soft"]


def test_golden_set_is_big_enough():
    assert len(CASES) >= 20 and len(HARD) >= 10 and len(SOFT) >= 10


@pytest.mark.parametrize("case", HARD, ids=[c["id"] for c in HARD])
def test_hard_case(case):
    ok, detail = run_case(case)
    assert ok, f"{case['id']}: {detail}"


def test_soft_cases_pass_rate():
    results = []
    failing = []
    for case in SOFT:
        ok, detail = run_case(case)
        results.append((case["id"], ok))
        if not ok:
            failing.append(f"{case['id']}: {detail}")
    rate = soft_pass_rate(results)
    assert rate >= 0.85, f"soft pass rate {rate:.0%} < 85%. Failing: {failing}"
