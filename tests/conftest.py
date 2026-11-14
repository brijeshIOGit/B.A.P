"""Shared test setup: offline brain, a fresh temp database, no real waiting.

The tests need NO API key: MODEL_PROVIDER is forced to "offline" before codeforge is imported.
"""

import os

os.environ["MODEL_PROVIDER"] = "offline"
os.environ["CODEFORGE_EMBEDDINGS"] = "hash"

import copy  # noqa: E402

import pytest  # noqa: E402

from codeforge import cache, config, db, models, policy  # noqa: E402
from codeforge.graph import build_graph  # noqa: E402

# Live demo hook for the UI "weaken a rule" switch: CODEFORGE_WEAKEN=allow_os loosens the
# policy INSIDE THIS TEST PROCESS ONLY, so the gate goes red. The app's policy is never changed.
if os.environ.get("CODEFORGE_WEAKEN") == "allow_os":
    policy.DENY_IMPORTS.discard("os")
    policy.ALLOW_IMPORTS.add("os")


@pytest.fixture(autouse=True)
def fresh_system(tmp_path):
    """Every test starts with an empty database, cache, breakers and chaos switch."""
    saved = copy.deepcopy(config.SETTINGS)
    config.SETTINGS["sample_db_path"] = str(tmp_path / "sample.db")
    db.init_db(str(tmp_path / "codeforge.db"))
    db.EVENTS.clear()
    cache.CACHE.clear()
    models.reset_resilience()
    config.SLEEP = lambda seconds: None  # retries do not really wait in tests
    yield
    config.SETTINGS.clear()
    config.SETTINGS.update(saved)
    config.CLOCK = __import__("time").time
    models.reset_resilience()


@pytest.fixture
def app():
    """A fresh graph with its own memory."""
    return build_graph()
