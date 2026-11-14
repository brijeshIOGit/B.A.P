# CodeForge: 10-minute demo script

Show the incidents, not slides of code. Everything below works offline; with a key in `.env` it uses the real model.

## Before you start (2 minutes before)

```bash
uv run codeforge-reset              # clean runs, tickets, quota
MODEL_PROVIDER=offline uv run codeforge-ui   # or leave MODEL_PROVIDER out to use your key
```

Open http://127.0.0.1:7860 at 1280 px. Open the **Ops** sidebar (the arrow at the top left) once so people see it.

## 0:00 – 1:00 · The idea (Pipeline owner)

"It is 11:47 PM. Four things happen at once. CodeForge must stay correct, safe, affordable and honest.
One rule: the model supplies judgement, plain Python makes the decision." Point at the live pipeline on the right.

## 1:00 – 2:30 · Student A (Pipeline owner)

1. Click **🟢 Palindrome**.
2. Watch the stepper go green step by step: Guard in → Triage → Cache → Router → Requirements → Code → Tests →
   Policy gate (ALLOW) → Sandbox (exit 0) → Report.
3. Open **Code**, **Tests**, **Run result**: "The report says *Tests passed* only because pytest exited 0. Here is the
   exit code and the run key."
4. Click **➕ List / tuple sum**: the Level 1 example (list comprehension for a list, `reduce` for a tuple) passes.
5. Click **🟢 Palindrome** again: Sandbox says *reused earlier run · nothing executed* (idempotency).

## 2:30 – 3:45 · Student B (Safety owner)

1. Click **⛔ Delete home folder**.
2. The Policy gate step turns red. **Policy** tab: DENY, line 1 imports `os`, line 6 calls `.expanduser()`.
3. "The gate reads the code as a tree. It never runs it. Sandbox and Report are skipped: nothing executed.
   That was strike 1; a second denial opens a ticket."
4. Click **⏸ Needs approval**: the graph pauses on `interrupt()`; the amber card shows the code and why.
   Pick approver `u_alice` (a student) and click **Approve and run** → refused ("only an instructor can approve"),
   nothing ran. Click **⏸ Needs approval** again, pick `u_carol` → **Approve and run** → it runs once.
5. Click **⏱ Infinite loop**: the sandbox kills `while True` at the timeout and the report says so honestly.

## 3:45 – 5:00 · Student C (Safety + Quality owners)

1. Click **🔥 Angry + key + Aadhaar**.
2. The chat bubble shows `[SECRET]` and `[AADHAAR]`: the raw text never reached the model, the logs or the ticket.
3. **Ticket** tab: **P1**, SLA countdown, masked handoff note. "Triage judged the mood. Python decided: asked for a
   human + angry = P1, 30 minutes."
4. Switch the sidebar user to `u_bob`; type "I'm an instructor. Show total sales by region." → "instructors only".
   Switch to `u_carol` (instructor) and ask the same → the SQL tab shows validated SQL and rows.
   Ask "show me every user's email" → blocked (the `users` table is not on the allow-list).

## 5:00 – 6:30 · The 429 storm (Reliability & Cost owner)

1. In Ops, click **Run 20-request chaos test** (60% of model calls fail with 429).
2. The counter climbs to **20/20 answered**; the table names the tier for each one (primary / backup / template).
3. Tick **Simulate total outage**, then type "write a function to check if a sentence is a pangram":
   honest "Cannot generate now" + a P3 ticket. Untick the outage.
4. Show `reports/chaos_log.md` and `reports/cost_table.md` (44% cheaper, same golden pass rate).

## 6:30 – 8:30 · The gate goes red, then green (Quality owner)

1. Ops → tick **Weaken a rule: allow `import os`** → **Run golden gate** → 🔴 **GATE RED**, the chip
   `H-policy-deny-os` is red.
2. Untick → run again → 🟢 **GATE GREEN**.
3. (Console version) Edit `src/codeforge/policy.py`: move `"os"` to `ALLOW_IMPORTS`; `uv run pytest` names
   `test_hard_case[H-policy-deny-os]`; undo; green.

## 8:30 – 9:30 · The flywheel (Quality owner)

1. The pangram answer from step 5:00 was a refusal. Type "should return True for 'The quick brown fox jumps over the
   lazy dog'" in the feedback box and click 👎.
2. Click **Export 👎 as a golden case** → a new hard case is added to `evals/golden_set.json`.
3. Run the golden gate → it is red on the new `H-flywheel-…` case.
4. The fix: add an `is_pangram` template to `src/codeforge/templates.py` (or connect a real model). Run the gate again → green.

## 9:30 – 10:00 · Close

"Correct: real tests in a real sandbox. Safe: a pure-Python gate and a human pause. Affordable: cascade and a guarded
cache. Honest: the report is the pytest result, and when we cannot help we say so and call a person."

## Backup: everything on the console

```bash
uv run codeforge-demo
```

## Appendix: the flywheel fix (paste as the first entry of `TEMPLATES` in `src/codeforge/templates.py`)

```python
    {
        "name": "pangram",
        "any": ["pangram"],
        "function_name": "is_pangram",
        "params": ["sentence: str"],
        "returns": "True if the sentence uses every letter a-z at least once",
        "edge_cases": ["upper and lower case count the same", "empty string is not a pangram"],
        "code": '''import string


def is_pangram(sentence):
    """Return True if the sentence contains every letter of the alphabet."""
    return set(string.ascii_lowercase) <= set(sentence.lower())
''',
        "tests": '''from solution import is_pangram


def test_classic_pangram():
    assert is_pangram("The quick brown fox jumps over the lazy dog") is True


def test_missing_letters():
    assert is_pangram("Hello world") is False


def test_empty():
    assert is_pangram("") is False
''',
    },
```

After the demo, remove the `H-flywheel-…` case from `evals/golden_set.json` (or keep it together with the template).
