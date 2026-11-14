"""Two caches that reuse code + tests (R4, and R2 "same request twice runs once").

1. Repeat cache: same user, exact same (masked) request, within 10 minutes.
2. Semantic cache: any user, a request that MEANS the same thing, within 1 hour.
   Two locks: embedding similarity >= threshold AND the same guard words.

A cache hit only skips the model steps. The policy gate and the sandbox still run.
"""

import hashlib
import re

import numpy as np
from langchain_core.embeddings import Embeddings

from . import config


class HashingEmbeddings(Embeddings):
    """Offline embeddings: character 3-grams + words hashed into 512 numbers (from the notebook)."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> list:
        v = np.zeros(self.dim)
        t = " " + re.sub(r"[^a-z0-9 ]", " ", text.lower()) + " "
        grams = [t[i:i + 3] for i in range(len(t) - 2)] + t.split()
        for g in grams:
            v[int(hashlib.md5(g.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def get_embedder():
    """MiniLM if asked for and installed, else the offline hashing embeddings."""
    if config.SETTINGS["embeddings"] == "minilm":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            pass
    return HashingEmbeddings()


# Requests about the user's own context must never be shared with someone else.
PERSONAL_RE = re.compile(r"\b(my|mine|our|this|above|previous|earlier|again|fix)\b|```", re.I)

# Words that change the answer even when two requests look alike.
GUARD_STEMS = ("ascend", "descend", "asc", "desc", "revers", "recurs", "iterat", "min", "max", "small", "larg",
               "even", "odd", "first", "last", "prime", "case", "insensitiv", "sensitiv", "ignor", "uniq", "duplic",
               "negat", "inclusiv", "exclusiv", "list", "tuple", "dict", "set", "string", "float", "sort", "not",
               "without", "word", "char", "sum", "product", "count", "vowel", "consonant")

# Default similarity thresholds, picked with threshold_sweep() on LABELLED_PAIRS (see tests).
THRESHOLDS = {"hash": 0.6, "minilm": 0.75}


def is_cacheable(request: str, pii: list) -> bool:
    """Only context-free requests with no personal data may be shared between users."""
    return not pii and not PERSONAL_RE.search(request or "")


def guard_terms(text: str) -> frozenset:
    """The answer-changing words (and every number) in this text."""
    t = (text or "").lower()
    stems = {s for s in GUARD_STEMS if re.search(rf"\b{s}", t)}
    numbers = set(re.findall(r"\d+", t))
    return frozenset(stems | numbers)


def _normal(text: str) -> str:
    """Lower case, single spaces: the key for the repeat cache."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


class SemanticCache:
    """Reuse code + tests for a request that means the same thing as an earlier one."""

    def __init__(self, embedder=None, threshold: float | None = None):
        self.embedder = embedder or get_embedder()
        kind = "hash" if isinstance(self.embedder, HashingEmbeddings) else "minilm"
        self.threshold = threshold if threshold is not None else THRESHOLDS[kind]
        self.items: list = []
        self.repeats: dict = {}
        self.hits = self.misses = 0
        self._memo: dict = {}

    def _embed(self, text: str):
        if text not in self._memo:
            v = np.array(self.embedder.embed_query(text), dtype=float)
            self._memo[text] = v / (np.linalg.norm(v) or 1.0)
        return self._memo[text]

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity of two texts (1.0 = same meaning)."""
        return float(self._embed(a) @ self._embed(b))

    # repeat cache (same user)
    def remember_repeat(self, user_id: str, request: str, payload: dict) -> None:
        self.repeats[(user_id, _normal(request))] = {"payload": payload, "t": config.now()}

    def find_repeat(self, user_id: str, request: str) -> dict | None:
        item = self.repeats.get((user_id, _normal(request)))
        if item and config.now() - item["t"] <= config.SETTINGS["repeat_ttl"]:
            return item["payload"]
        return None

    # semantic cache (any user)
    def store(self, request: str, payload: dict) -> None:
        if len(self.items) >= 500:
            self.items.pop(0)
        self.items.append({"request": request, "payload": payload, "t": config.now()})

    def lookup(self, request: str) -> dict | None:
        """The best stored payload with the same guard words and similarity >= threshold."""
        now, guards = config.now(), guard_terms(request)
        best, best_score = None, -1.0
        for item in self.items:
            if now - item["t"] > config.SETTINGS["cache_ttl"]:
                continue
            if guard_terms(item["request"]) != guards:
                continue
            score = self.similarity(request, item["request"])
            if score > best_score:
                best, best_score = item, score
        if best is not None and best_score >= self.threshold:
            self.hits += 1
            return dict(best["payload"], score=round(best_score, 3), matched=best["request"])
        self.misses += 1
        return None

    def clear(self) -> None:
        self.items.clear()
        self.repeats.clear()
        self.hits = self.misses = 0


CACHE = SemanticCache()

# (request A, request B, should they share an answer?)
LABELLED_PAIRS = [
    ("Write a Python function that checks if a string is a palindrome.",
     "Write a python function to check whether a string is a palindrome", True),
    ("write a function to compute the factorial of a number", "function that computes factorial of a number", True),
    ("write a function that returns the nth fibonacci number", "write a function returning the nth fibonacci number", True),
    ("write a function to reverse a string", "write a python function that reverses a string", True),
    ("count the vowels in a string", "write a function to count vowels in a string", True),
    ("write fizzbuzz in python", "python fizzbuzz function", True),
    ("sort a list in ascending order", "sort a list in descending order", False),
    ("factorial using recursion", "factorial using iteration", False),
    ("sum of even numbers in a list", "sum of odd numbers in a list", False),
    ("first 10 fibonacci numbers", "first 20 fibonacci numbers", False),
    ("find the maximum of a list", "find the minimum of a list", False),
    ("reverse a string", "reverse a list", False),
    ("check if a number is prime", "check if a number is even", False),
    ("count vowels in a string", "count consonants in a string", False),
]


def threshold_sweep(cache: SemanticCache, pairs=LABELLED_PAIRS, thresholds=(0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9),
                    use_guard: bool = True) -> list[dict]:
    """For each threshold: true hits (same answer) and false hits (different answer)."""
    rows = []
    for th in thresholds:
        hits = false_hits = 0
        for a, b, same in pairs:
            ok = cache.similarity(a, b) >= th and (not use_guard or guard_terms(a) == guard_terms(b))
            hits += ok and same
            false_hits += ok and not same
        rows.append({"threshold": th, "hits": hits, "false_hits": false_hits,
                     "same_pairs": sum(p[2] for p in pairs), "different_pairs": sum(not p[2] for p in pairs)})
    return rows
