"""Metering (R4): a LangChain callback that counts tokens, seconds and dollars per agent.

It also keeps every prompt it saw, so a test can prove no secret ever reached a model.
"""

import time

from langchain_core.callbacks import BaseCallbackHandler

from . import config


class Meter(BaseCallbackHandler):
    """Counts every model call made under it (pass it in config["callbacks"])."""

    def __init__(self):
        self.calls: list = []
        self.prompts: list = []
        self._open: dict = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        meta = metadata or {}
        for batch in messages:
            for m in batch:
                self.prompts.append(str(m.content))
        self._open[run_id] = (time.perf_counter(), meta.get("agent", "?"), meta.get("model", "?"),
                              meta.get("tier", "?"))

    def on_llm_end(self, response, *, run_id, **kwargs):
        started, agent, model, tier = self._open.pop(run_id, (time.perf_counter(), "?", "?", "?"))
        usage = {}
        try:
            usage = response.generations[0][0].message.usage_metadata or {}
        except Exception:
            pass
        n_in, n_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        price_in, price_out = config.PRICES.get(model, (0.0, 0.0))
        self.calls.append({"agent": agent, "model": model, "tier": tier, "in": n_in, "out": n_out,
                           "sec": time.perf_counter() - started, "usd": (n_in * price_in + n_out * price_out) / 1e6})

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._open.pop(run_id, None)

    def by_agent(self) -> dict:
        """Totals per agent: calls, tokens in/out, seconds, dollars."""
        out: dict = {}
        for c in self.calls:
            row = out.setdefault(c["agent"], {"calls": 0, "in": 0, "out": 0, "sec": 0.0, "usd": 0.0})
            row["calls"] += 1
            row["in"] += c["in"]
            row["out"] += c["out"]
            row["sec"] += c["sec"]
            row["usd"] += c["usd"]
        return out

    def total(self) -> dict:
        """Totals over every call."""
        return {"calls": len(self.calls), "in": sum(c["in"] for c in self.calls),
                "out": sum(c["out"] for c in self.calls), "sec": sum(c["sec"] for c in self.calls),
                "usd": sum(c["usd"] for c in self.calls)}
