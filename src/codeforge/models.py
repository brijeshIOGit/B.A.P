"""Every model call goes through ask_model() (R4 + R5).

ask_model does, in order:
  1. mask every message again (R1: secrets never reach a model)
  2. tag the call with agent + model so the Meter can count tokens, seconds and dollars (R4)
  3. walk down the tiers (R5):
       primary model  -> retries with full jitter on 429 / 5xx / timeouts, circuit breaker
       backup model   -> same
       template       -> plain Python, no model
       none           -> return None; the node gives an honest "cannot generate now"
The chaos switch can fail a share of primary/backup calls on purpose.
"""

import json
import random
import re

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool

from . import config
from .brain import OfflineCannotAnswer, RuleBasedChat
from .masking import mask_text


# ── chaos switch (from the bootcamp notebook) ────────────────────────────────

class SimulatedAPIError(Exception):
    """A fake provider error, e.g. 429 Too Many Requests."""

    def __init__(self, status_code: int = 429, message: str = "simulated provider failure"):
        super().__init__(f"{status_code}: {message}")
        self.status_code = status_code


class Chaos:
    """Fail a share of model calls on purpose, to rehearse an outage."""

    def __init__(self):
        self.configure(0.0)

    def configure(self, rate: float = 0.0, tiers=("primary", "backup"), status: int = 429, seed: int = 7,
                  models=None):
        """models: only fail these model names (e.g. "the 120b model is down"); None = any model."""
        self.rate, self.tiers, self.status, self.models = rate, tiers, status, models
        self.rng = random.Random(seed)
        self.injected = 0

    def maybe_fail(self, tier: str, model: str = ""):
        """Raise a fake 429 for `rate` of the calls on the chosen tiers (and models, if set)."""
        if self.models is not None and model not in self.models:
            return
        if self.rate and tier in self.tiers and self.rng.random() < self.rate:
            self.injected += 1
            raise SimulatedAPIError(self.status)


CHAOS = Chaos()
CHAOS.configure(float(config._env("CODEFORGE_CHAOS", "0") or 0))

# Every attempt is logged here (agent, tier, model, outcome). The chaos report reads it.
CALL_LOG: list = []


# ── circuit breaker: one small dict per model ────────────────────────────────

BREAKERS: dict = {}


def breaker_allows(model: str) -> bool:
    """False while this model's breaker is open (too many recent failures)."""
    state = BREAKERS.get(model)
    return not state or config.now() >= state["open_until"]


def breaker_failure(model: str) -> None:
    """Count a failure; open the breaker after N in a row."""
    state = BREAKERS.setdefault(model, {"fails": 0, "open_until": 0.0})
    state["fails"] += 1
    if state["fails"] >= config.SETTINGS["breaker_fails"]:
        state["open_until"] = config.now() + config.SETTINGS["breaker_cooldown"]
        state["fails"] = 0


def breaker_success(model: str) -> None:
    """A success closes the breaker again."""
    BREAKERS[model] = {"fails": 0, "open_until": 0.0}


def breaker_states() -> dict:
    """'open' or 'closed' for each model we have called (for the UI)."""
    return {m: ("open" if not breaker_allows(m) else "closed") for m in BREAKERS}


def reset_resilience() -> None:
    """Close every breaker, clear the call log and turn chaos off."""
    BREAKERS.clear()
    CALL_LOG.clear()
    CHAOS.configure(0.0)


# ── providers ────────────────────────────────────────────────────────────────

_CHATS: dict = {}


def make_chat(model: str):
    """Create the chat model for the configured provider (cached). Offline: the rule-based brain."""
    provider = config.PROVIDER
    key = (provider, model)
    if key in _CHATS:
        return _CHATS[key]
    common = {"temperature": 0, "max_retries": 0}  # our ladder retries, the SDK must not
    if provider == "groq":
        from langchain_groq import ChatGroq

        chat = ChatGroq(model=model, timeout=30, **common)
    elif provider == "xai":
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(model=model, base_url="https://api.x.ai/v1", api_key=config._env("XAI_API_KEY"),
                          timeout=30, **common)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(model=model, timeout=30, **common)
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        chat = ChatOllama(model=model, temperature=0)
    else:
        chat = RuleBasedChat(label=model)
    _CHATS[key] = chat
    return chat


def _snake(name: str) -> str:
    """ActionProposal -> action_proposal (Groq checks tool names exactly)."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


CODE_BLOCK_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
TEXT_MODE_RULE = ("Reply with exactly one ```python code block that contains the whole file, "
                  "and nothing else: no explanation before or after it.")


def extract_code_block(text: str) -> str:
    """The longest ```python block in a reply (or the whole reply if it has no fence)."""
    blocks = CODE_BLOCK_RE.findall(text or "")
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return (text or "").strip() + "\n"


def _recover_failed_generation(err, schema):
    """Groq sends back what the model wrote when it rejects a tool call ('failed_generation').
    If those arguments still fit the schema, use them instead of failing (from the notebook)."""
    body = getattr(err, "body", None)
    if not isinstance(body, dict):
        return None
    gen = (body.get("error") or body).get("failed_generation")
    try:
        data = json.loads(gen) if isinstance(gen, str) else gen
        if isinstance(data, list):
            data = data[0] if data else {}
        args = data.get("arguments", data.get("parameters", data))
        args = json.loads(args) if isinstance(args, str) else args
        return schema.model_validate(args)
    except Exception:
        return None


def structured(llm, schema):
    """Ask the model to fill `schema`.

    Long code (GeneratedCode / GeneratedTests) is asked for as plain text in one ```python block:
    on Groq, forcing a big code file into a JSON tool call often fails ("did not call a tool",
    "failed to parse tool call arguments"). Everything else uses one forced tool call, with the
    notebook's failed_generation recovery.
    """
    if isinstance(llm, RuleBasedChat):
        return llm.with_structured_output(schema)
    text_field = getattr(schema, "text_field", None)
    if text_field:
        def run_text(messages, cfg=None):
            reply = llm.invoke(list(messages) + [SystemMessage(content=TEXT_MODE_RULE)], cfg)
            return schema(**{text_field: extract_code_block(str(reply.content))})

        return RunnableLambda(run_text)

    name = _snake(schema.__name__)
    tool = convert_to_openai_tool(schema)
    tool["function"]["name"] = name
    chain = llm.bind_tools([tool], tool_choice=name) | JsonOutputKeyToolsParser(key_name=name, first_tool_only=True)

    def run(messages, cfg=None):
        try:
            args = chain.invoke(messages, cfg)
        except Exception as err:
            rescued = _recover_failed_generation(err, schema)
            if rescued is not None:
                return rescued
            raise
        if args is None:
            raise ValueError("model returned no structured answer")
        return schema.model_validate(args)

    return RunnableLambda(run)


def is_transient(err: Exception) -> bool:
    """Retry only 429, 5xx and timeouts / connection drops."""
    status = getattr(err, "status_code", None) or getattr(getattr(err, "response", None), "status_code", None)
    if status in (429, 500, 502, 503, 504):
        return True
    name = type(err).__name__.lower()
    return any(word in name for word in ("timeout", "ratelimit", "connection", "unavailable"))


def _mask_messages(messages: list) -> list:
    """Copies of the messages with every text masked."""
    clean = []
    for m in messages:
        if isinstance(m, BaseMessage) and isinstance(m.content, str):
            clean.append(m.model_copy(update={"content": mask_text(m.content)}))
        else:
            clean.append(m)
    return clean


def _tiers(size: str) -> list[tuple[str, str]]:
    """[(tier, model)] to try in order. Cascade: small model for easy steps."""
    small, big = config.MODELS["small"], config.MODELS["big"]
    primary = small if (size == "small" and config.SETTINGS["cascade"]) else big
    backup = config.MODELS["backup"]
    if backup == primary:  # a backup must be a different model: the two gpt-oss sizes back each other up
        backup = big if primary == small else small
    return [("primary", primary), ("backup", backup)]


def _try_model(agent, tier, model, messages, schema, check, callbacks):
    """Try one model with retries. Returns the answer, or None to move to the next tier."""
    tries = config.SETTINGS["retry_tries"]
    for attempt in range(tries):
        if not breaker_allows(model):
            CALL_LOG.append({"agent": agent, "tier": tier, "model": model, "outcome": "breaker open"})
            return None
        try:
            CHAOS.maybe_fail(tier, model)
            llm = make_chat(model)
            runnable = structured(llm, schema) if schema else llm
            run_config = {"callbacks": callbacks, "metadata": {"agent": agent, "tier": tier, "model": model}}
            answer = runnable.invoke(messages, run_config)
            if check is not None:
                answer = check(answer)
                if answer is None:
                    raise ValueError("answer failed our checks")
            breaker_success(model)
            CALL_LOG.append({"agent": agent, "tier": tier, "model": model, "outcome": "ok"})
            return answer
        except Exception as err:
            transient = is_transient(err)
            CALL_LOG.append({"agent": agent, "tier": tier, "model": model,
                             "outcome": "transient error" if transient else "error", "error": str(err)[:120]})
            if not transient:
                return None  # bad output, auth error, offline brain has no answer: next tier
            breaker_failure(model)
            if attempt < tries - 1:
                cap = min(config.SETTINGS["retry_cap"], config.SETTINGS["retry_base"] * 2 ** attempt)
                config.sleep(random.uniform(0, cap))  # full jitter
    return None


def ask_model(agent: str, messages: list, schema=None, size: str = "small", template=None, check=None,
              cfg=None):
    """Ask a model through the whole ladder. Returns (answer or None, tier)."""
    messages = _mask_messages(messages)
    callbacks = (cfg or {}).get("callbacks")
    for tier, model in _tiers(size):
        answer = _try_model(agent, tier, model, messages, schema, check, callbacks)
        if answer is not None:
            return answer, tier
    if template is not None:
        answer = template()
        if answer is not None:
            CALL_LOG.append({"agent": agent, "tier": "template", "model": "-", "outcome": "ok"})
            return answer, "template"
    CALL_LOG.append({"agent": agent, "tier": "none", "model": "-", "outcome": "no answer"})
    return None, "none"


__all__ = ["ask_model", "CHAOS", "CALL_LOG", "SimulatedAPIError", "OfflineCannotAnswer", "make_chat",
           "breaker_states", "reset_resilience", "is_transient"]
