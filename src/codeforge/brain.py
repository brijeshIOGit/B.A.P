"""The offline brain: a rule-based stand-in for a chat model (pattern from the bootcamp notebook).

It is used when no API key is set, and in every test. It behaves like a model:
it reports token counts (so metering works), can be failed by the chaos switch,
and fills the same Pydantic schemas. For code it only knows the template library
plus a few "proposals" that let us demo the policy gate (Student B, file access,
an infinite loop). Anything else: it says it cannot answer, and the system falls
back honestly.
"""

import re
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from .templates import match_template, template_requirements


class OfflineCannotAnswer(Exception):
    """The offline brain has no answer for this request (not a transient error)."""


# What a careless model might write for these requests. The policy gate must catch them.
OFFLINE_PROPOSALS = [
    {
        "match": lambda t: "delete" in t and ("file" in t or "folder" in t),
        "function_name": "delete_home_files",
        "code": '''import os


def delete_home_files():
    """Delete every file in the user's home folder."""
    home = os.path.expanduser("~")
    for root, dirs, files in os.walk(home):
        for name in files:
            os.remove(os.path.join(root, name))
''',
        "tests": '''from solution import delete_home_files


def test_is_callable():
    assert callable(delete_home_files)
''',
    },
    {
        "match": lambda t: ("read" in t or "count" in t) and ("file" in t or "csv" in t) and "line" in t,
        "function_name": "count_lines",
        "code": '''def count_lines(path):
    """Count the lines in a text file."""
    with open(path) as handle:
        return sum(1 for _ in handle)
''',
        "tests": '''from solution import count_lines


def test_counts_lines(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("a\\nb\\nc\\n")
    assert count_lines(str(sample)) == 3
''',
    },
    {
        "match": lambda t: "forever" in t or "infinite loop" in t or "while true" in t,
        "function_name": "spin_forever",
        "code": '''def spin_forever():
    """Loop forever (the sandbox timeout must stop this)."""
    while True:
        pass
''',
        "tests": '''from solution import spin_forever


def test_spin():
    spin_forever()
''',
    },
]


def find_proposal(request: str) -> dict | None:
    """An offline 'model proposal' for a demo request, if any."""
    text = (request or "").lower()
    for proposal in OFFLINE_PROPOSALS:
        if proposal["match"](text):
            return proposal
    return None


# ── rules that imitate model judgement ───────────────────────────────────────

ANGRY_WORDS = ("useless", "worst", "ridiculous", "furious", "terrible", "fed up", "hate this", "unacceptable")
NEGATIVE_WORDS = ("not working", "doesn't work", "does not work", "still failing", "still broken", "frustrat",
                  "annoyed", "disappointed", "wrong again", "not happy", "broken")
URGENT_WORDS = ("deadline", "due tonight", "tonight", "asap", "urgent", "right now", "in 10 minutes",
                "immediately", "minutes left")
HUMAN_RE = re.compile(r"\b(human|person|agent|someone real|mentor|teaching assistant|real teacher)\b", re.I)
SQL_RE = re.compile(r"\b(sql|select|query|table|database|sales|employees|revenue|join)\b", re.I)
OTHER_LANG_RE = re.compile(r"\b(javascript|typescript|java|c\+\+|rust|golang|ruby|php|kotlin|swift|c#)\b", re.I)
CODE_RE = re.compile(r"\b(function|def|write|implement|return|code|program|python|script|check|compute|"
                     r"calculate|convert|sort|count|list|string|number)\b", re.I)


def rule_triage(text: str) -> dict:
    """Keyword triage: task type, language, sentiment, urgency, wants_human."""
    t = (text or "").lower()
    letters = [c for c in text or "" if c.isalpha()]
    shouty = len(letters) > 8 and sum(c.isupper() for c in letters) / len(letters) > 0.5
    caps_words = re.findall(r"\b[A-Z]{3,}\b", text or "")
    if shouty or "!!!" in t or "???" in t or len(caps_words) >= 3 or any(w in t for w in ANGRY_WORDS):
        sentiment = "angry"
    elif any(w in t for w in NEGATIVE_WORDS):
        sentiment = "negative"
    elif any(w in t for w in ("thanks", "thank you", "great", "awesome", "perfect")):
        sentiment = "positive"
    else:
        sentiment = "neutral"
    urgency = 5 if any(w in t for w in URGENT_WORDS) else (3 if "soon" in t or "today" in t else 1)
    route = rule_route(text)
    task_type = {"python_function": "code_request", "sql_query": "sql_question"}.get(route["kind"], "other")
    if sentiment in ("angry", "negative") and route["kind"] == "other":
        task_type = "complaint"
    return {"task_type": task_type, "language": route["language"], "sentiment": sentiment,
            "urgency": urgency, "wants_human": bool(HUMAN_RE.search(t))}


def rule_route(text: str) -> dict:
    """Keyword router: python_function, sql_query or other."""
    t = text or ""
    lang = OTHER_LANG_RE.search(t)
    if lang:
        return {"kind": "other", "language": lang.group(1).lower(), "reasoning": "asks for another language"}
    if match_template(t) or find_proposal(t):
        return {"kind": "python_function", "language": "python", "reasoning": "known Python task"}
    if SQL_RE.search(t) and not re.search(r"\bfunction\b", t, re.I):
        return {"kind": "sql_query", "language": "sql", "reasoning": "asks a question about data"}
    if CODE_RE.search(t):
        return {"kind": "python_function", "language": "python", "reasoning": "asks for code"}
    return {"kind": "other", "language": "python", "reasoning": "not a coding request"}


def guess_function_name(request: str) -> str:
    """A snake_case name built from the request, e.g. 'check pangram' -> 'check_pangram'."""
    stop = {"write", "a", "an", "the", "python", "function", "that", "which", "to", "for", "me", "please",
            "returns", "return", "of", "in", "and", "if", "is", "it", "can", "you", "my", "create", "make"}
    words = [w for w in re.findall(r"[a-z]+", (request or "").lower()) if w not in stop]
    name = "_".join(words[:3]) or "solve"
    return name if name.isidentifier() else "solve"


def rule_requirements(request: str) -> dict:
    """Requirements from the template library, a demo proposal, or a simple guess."""
    template = match_template(request)
    if template:
        return template_requirements(template)
    proposal = find_proposal(request)
    name = proposal["function_name"] if proposal else guess_function_name(request)
    return {"function_name": name, "params": [], "returns": "see request", "edge_cases": [], "examples": []}


def rule_sql(question: str) -> str:
    """Keyword SQL for the offline SQL agent."""
    t = (question or "").lower()
    if "user" in t or "email" in t or "aadhaar" in t:
        return "SELECT name, email FROM users"  # the validator must block this
    if "region" in t:
        return "SELECT region, SUM(amount) AS total FROM sales GROUP BY region ORDER BY total DESC"
    if "product" in t:
        return "SELECT product, COUNT(*) AS n, SUM(amount) AS total FROM sales GROUP BY product ORDER BY total DESC"
    if "department" in t or "employee" in t:
        return "SELECT department, COUNT(*) AS headcount FROM employees GROUP BY department"
    if "top" in t or "best" in t:
        return ("SELECT e.name, SUM(s.amount) AS total FROM sales s JOIN employees e ON e.id = s.employee_id "
                "GROUP BY e.name ORDER BY total DESC LIMIT 5")
    return "SELECT * FROM sales"


# ── the fake chat model ──────────────────────────────────────────────────────

def extract_request(messages) -> str:
    """The user's request: the 'REQUEST:' line of the prompt, else the last human message."""
    text = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            text = m.content if isinstance(m.content, str) else str(m.content)
            break
    found = re.search(r"^REQUEST:\s*(.+)$", text, re.M)
    return found.group(1).strip() if found else text


def _structured_answer(schema, messages):
    """Fill one of our Pydantic schemas with rules."""
    request = extract_request(messages)
    name = schema.__name__
    if name == "Triage":
        return schema(**rule_triage(request))
    if name == "RouteDecision":
        return schema(**rule_route(request))
    if name == "Requirements":
        return schema(**rule_requirements(request))
    if name in ("GeneratedCode", "GeneratedTests"):
        source = match_template(request) or find_proposal(request)
        if not source:
            raise OfflineCannotAnswer(f"offline brain has no code for: {request[:60]}")
        if name == "GeneratedCode":
            return schema(code=source["code"])
        return schema(tests=source["tests"])
    if name == "Handoff":
        return schema(summary=f"Student wrote: {request[:160]}", user_ask=request[:160],
                      suggested_next_step="Read the request and reply personally.")
    if name == "SQLQuery":
        return schema(sql=rule_sql(request), explanation="keyword-based query")
    raise OfflineCannotAnswer(f"no rule for schema {name}")


class RuleBasedChat(BaseChatModel):
    """Deterministic stand-in for an LLM. `label` is the model it pretends to be (for prices)."""

    label: str = "offline"

    @property
    def _llm_type(self) -> str:
        return "rule-based"

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.label}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Plain text answers. Also echoes a structured answer so its size is metered."""
        last = messages[-1] if messages else None
        if isinstance(last, SystemMessage) and str(last.content).startswith("#answer "):
            return self._result(messages[:-1], str(last.content)[len("#answer "):])
        first = messages[0].content if messages and isinstance(messages[0], SystemMessage) else ""
        if str(first).startswith("Summarise"):
            asks = [ln.split(": ", 1)[1][:50] for ln in str(messages[-1].content).splitlines() if ln.startswith("user: ")]
            content = "The student asked about: " + "; ".join(asks[-5:]) + "."
        else:
            content = "OK."
        return self._result(messages, content)

    def _result(self, messages, content: str) -> ChatResult:
        """Wrap text with rough token counts (4 characters ~ 1 token), like a real model reports."""
        n_in = sum(len(str(m.content)) for m in messages) // 4
        n_out = max(len(content) // 4, 1)
        msg = AIMessage(content=content, id=f"run-{uuid.uuid4().hex[:8]}")
        msg.usage_metadata = {"input_tokens": n_in, "output_tokens": n_out, "total_tokens": n_in + n_out}
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def with_structured_output(self, schema, **kwargs):
        """Return a runnable that fills `schema`. It makes a real (fake) model call so metering sees it."""

        def run(messages, config=None):
            messages = messages if isinstance(messages, list) else [HumanMessage(content=str(messages))]
            answer = _structured_answer(schema, messages)  # may raise OfflineCannotAnswer
            # Report the answer's size as output tokens, like a real model call.
            self._meter_call(messages, answer.model_dump_json(), config)
            return answer

        return RunnableLambda(run)

    def _meter_call(self, messages, content: str, config):
        """Send the call through the normal callback path so the Meter records tokens."""
        self.invoke(messages + [SystemMessage(content="#answer " + content)], config)
