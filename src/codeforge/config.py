"""Settings for CodeForge, read once from the environment (.env).

Everything a rule depends on is a plain value here, so it is easy to read and to change.
Tests change SETTINGS directly (for example a shorter sandbox timeout).
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
EVALS_DIR = PROJECT_ROOT / "evals"
REPORTS_DIR = PROJECT_ROOT / "reports"

# .env lives in the project root and is gitignored. It never overrides variables already set.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env(name: str, default: str = "") -> str:
    """Read an env var; an empty value counts as missing."""
    value = os.environ.get(name, "").strip()
    return value or default


def detect_provider() -> str:
    """Pick the model provider: MODEL_PROVIDER if set, else the first key we find, else offline."""
    chosen = _env("MODEL_PROVIDER").lower()
    if chosen:
        return chosen
    if _env("GROQ_API_KEY"):
        return "groq"
    if _env("XAI_API_KEY"):
        return "xai"
    if _env("OPENAI_API_KEY"):
        return "openai"
    return "offline"


# Default model names per provider. Override them in .env (CODEFORGE_BIG_MODEL, ...).
# "offline" reuses the Groq names as labels so the cost table still has real prices.
DEFAULT_MODELS = {
    # Groq + open-weight gpt-oss models: the same choice as the bootcamp notebook.
    "groq": {"big": "openai/gpt-oss-120b", "small": "openai/gpt-oss-20b", "backup": "openai/gpt-oss-20b"},
    "xai": {"big": "grok-4", "small": "grok-3-mini", "backup": "grok-3"},
    "openai": {"big": "gpt-4.1", "small": "gpt-4.1-mini", "backup": "gpt-4.1-mini"},
    "ollama": {"big": "qwen2.5-coder", "small": "llama3.1", "backup": "llama3.1"},
    "offline": {"big": "openai/gpt-oss-120b", "small": "openai/gpt-oss-20b", "backup": "openai/gpt-oss-20b"},
}


def model_names(provider: str) -> dict:
    """Big, small and backup model names for this provider (env overrides win)."""
    names = dict(DEFAULT_MODELS.get(provider, DEFAULT_MODELS["offline"]))
    if provider == "ollama" and _env("OLLAMA_MODEL"):
        names["big"] = _env("OLLAMA_MODEL")
    names["big"] = _env("CODEFORGE_BIG_MODEL", names["big"])
    names["small"] = _env("CODEFORGE_SMALL_MODEL", names["small"])
    names["backup"] = _env("CODEFORGE_BACKUP_MODEL", names["backup"])
    return names


# Illustrative USD per 1M tokens (input, output). Check your provider's price page and edit freely.
PRICES = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "grok-4": (3.00, 15.00),
    "grok-3": (3.00, 15.00),
    "grok-3-mini": (0.30, 0.50),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
}

KEY_FOR = {"groq": "GROQ_API_KEY", "xai": "XAI_API_KEY", "openai": "OPENAI_API_KEY"}


def usable_provider(provider: str) -> str:
    """A provider that needs a key but has none falls back to offline, with a clear message."""
    key = KEY_FOR.get(provider)
    if key and not _env(key):
        print(f"[codeforge] MODEL_PROVIDER={provider} but {key} is empty in .env: running OFFLINE "
              f"(rule-based brain + templates) until you add the key.")
        return "offline"
    return provider


PROVIDER = usable_provider(detect_provider())
MODELS = model_names(PROVIDER)

# All the numbers the rules use. Change them here (or in a test) and the whole system follows.
SETTINGS = {
    "sandbox_timeout": float(_env("CODEFORGE_SANDBOX_TIMEOUT", "10")),
    "quota": {"student": 20, "instructor": 100},  # sandbox runs per user per 24 h
    "max_repairs": 2,  # first try + 2 repairs, then a ticket
    "compact_after": 12,  # messages before we summarise the chat
    "keep_recent": 4,  # newest messages kept as they are
    "cache_ttl": 3600,  # semantic cache entry life, seconds
    "repeat_ttl": 600,  # same user, same request: reuse within 10 min
    "retry_tries": 3,  # tries per model before moving to the next tier
    "retry_base": 0.5,  # full jitter: wait random(0, min(cap, base * 2**try))
    "retry_cap": 4.0,
    "breaker_fails": 3,  # failures in a row that open the circuit breaker
    "breaker_cooldown": 30,  # seconds the breaker stays open
    "embeddings": _env("CODEFORGE_EMBEDDINGS", "hash"),
    "db_path": str(DATA_DIR / "codeforge.db"),
    "sample_db_path": str(DATA_DIR / "sample.db"),
    # Request pairs a thumbs-down said must never share a cache entry (also used as test data).
    "cache_blocked_path": str(EVALS_DIR / "cache_blocked_pairs.json"),
    # Production levers (R4). The cost script turns these off for the "before" run.
    "use_cache": True,
    "cascade": True,
    "compact": True,
}

# Seeded demo users. Identity always comes from config, never from the chat text.
USERS = {"u_alice": "student", "u_bob": "student", "u_carol": "instructor"}

# Clock and sleep are indirections so tests can freeze time and skip waiting.
CLOCK = time.time
SLEEP = time.sleep


def now() -> float:
    """Current time in seconds (goes through CLOCK so tests can freeze it)."""
    return CLOCK()


def sleep(seconds: float) -> None:
    """Wait (goes through SLEEP so tests and the chaos run can skip waiting)."""
    SLEEP(seconds)
