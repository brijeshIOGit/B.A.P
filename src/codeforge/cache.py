"""Two caches that reuse code + tests (R4, and R2 "same request twice runs once").

1. Repeat cache: same user, exact same (masked) request, within 10 minutes. Always safe.
2. Shared (semantic) cache: another user's request that is THE SAME PROBLEM, within 1 hour.
   It must pass four locks, all plain Python:
     a. plain request   - short, no extra rules ("and", "but", "treat", quotes...), no PII, not personal
     b. same guard words - "ascending" vs "descending", numbers, "recursive"...
     c. same detail words - every meaningful word matches; only filler wording may differ,
                           so "...and returns yes or no" never reuses a True/False answer
     d. similarity >= threshold on the embeddings
   Plus two escape hatches: "Regenerate fresh" skips the cache, and a thumbs-down on a cached
   answer removes the entry and blocks that pair from matching again.

A cache hit only skips the model steps. The policy gate and the sandbox still run.
"""

import hashlib
import json
import re
from pathlib import Path

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

# Words that add an extra rule or condition. A request with one of these is too specific to share.
CONNECTOR_RE = re.compile(r"\b(and|but|except|unless|also|then|only|instead|treat|treating|however|while|"
                          r"where|plus|otherwise|rather)\b|[\"“”`]", re.I)
MAX_SHARED_DETAIL_WORDS = 6  # longer requests carry details nobody else is likely to need

# Words that change the answer even when two requests look alike.
GUARD_STEMS = ("ascend", "descend", "asc", "desc", "revers", "recurs", "iterat", "min", "max", "small", "larg",
               "even", "odd", "first", "last", "prime", "case", "insensitiv", "sensitiv", "ignor", "uniq", "duplic",
               "negat", "inclusiv", "exclusiv", "list", "tuple", "dict", "set", "string", "float", "sort", "not",
               "without", "word", "char", "sum", "product", "count", "vowel", "consonant")

# Filler: words that only change the wording, never the problem.
FILLER = {"write", "a", "an", "the", "python", "function", "func", "program", "code", "script", "method", "that",
          "which", "to", "for", "me", "please", "can", "could", "would", "you", "i", "want", "need", "make",
          "create", "implement", "give", "of", "in", "on", "is", "it", "its", "if", "whether", "check", "checker",
          "tell", "find", "get", "return", "compute", "calculate", "given", "some", "simple", "small", "just",
          "how", "do", "does", "will", "should", "using", "use", "with", "from", "by", "be", "are", "value"}

# Default similarity thresholds, picked with threshold_sweep() on LABELLED_PAIRS (see tests).
THRESHOLDS = {"hash": 0.6, "minilm": 0.75}


def _stem(word: str) -> str:
    """A tiny stemmer so 'checks', 'checking' and 'check' count as one word."""
    for suffix in ("ing", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    return word[:-1] if len(word) > 3 and word.endswith("e") else word


def detail_words(text: str) -> frozenset:
    """The meaningful words of a request (stemmed, filler removed, numbers kept)."""
    words = re.findall(r"[a-z]+|\d+", (text or "").lower())
    out = set()
    for w in words:
        stem = _stem(w)
        if w not in FILLER and stem not in FILLER:
            out.add(stem)
    return frozenset(out)


def guard_terms(text: str) -> frozenset:
    """The answer-changing words (and every number) in this text."""
    t = (text or "").lower()
    stems = {s for s in GUARD_STEMS if re.search(rf"\b{s}", t)}
    numbers = set(re.findall(r"\d+", t))
    return frozenset(stems | numbers)


def is_cacheable(request: str, pii: list) -> bool:
    """No personal data and nothing about the user's own context."""
    return not pii and not PERSONAL_RE.search(request or "")


def is_shareable(request: str, pii: list) -> bool:
    """Lock a: only plain, context-free requests may be shared with OTHER users."""
    return (is_cacheable(request, pii) and not CONNECTOR_RE.search(request or "")
            and len(detail_words(request)) <= MAX_SHARED_DETAIL_WORDS)


def _normal(text: str) -> str:
    """Lower case, single spaces: the key for the repeat cache and for blocked pairs."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


class SemanticCache:
    """Reuse code + tests, but only for the same user's repeat or for the SAME problem."""

    def __init__(self, embedder=None, threshold: float | None = None):
        self.embedder = embedder or get_embedder()
        kind = "hash" if isinstance(self.embedder, HashingEmbeddings) else "minilm"
        self.threshold = threshold if threshold is not None else THRESHOLDS[kind]
        self.items: list = []
        self.repeats: dict = {}
        self.blocked: set = set()  # (request, stored request) pairs a thumbs-down said must not match
        self.hits = self.misses = 0
        self._memo: dict = {}
        self.load_blocked()

    def _embed(self, text: str):
        if text not in self._memo:
            v = np.array(self.embedder.embed_query(text), dtype=float)
            self._memo[text] = v / (np.linalg.norm(v) or 1.0)
        return self._memo[text]

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity of two texts (1.0 = same meaning)."""
        return float(self._embed(a) @ self._embed(b))

    def same_problem(self, a: str, b: str) -> bool:
        """Locks b, c and d: same guard words, same detail words, similar enough, not blocked."""
        if (_normal(a), _normal(b)) in self.blocked or (_normal(b), _normal(a)) in self.blocked:
            return False
        return (guard_terms(a) == guard_terms(b) and detail_words(a) == detail_words(b)
                and self.similarity(a, b) >= self.threshold)

    # ── repeat cache (same user) ──
    def remember_repeat(self, user_id: str, request: str, payload: dict) -> None:
        self.repeats[(user_id, _normal(request))] = {"payload": payload, "t": config.now()}

    def find_repeat(self, user_id: str, request: str) -> dict | None:
        item = self.repeats.get((user_id, _normal(request)))
        if item and config.now() - item["t"] <= config.SETTINGS["repeat_ttl"]:
            return dict(item["payload"], matched=request)
        return None

    # ── shared cache (any user) ──
    def store(self, request: str, payload: dict) -> None:
        if len(self.items) >= 500:
            self.items.pop(0)
        self.items = [it for it in self.items if _normal(it["request"]) != _normal(request)]
        self.items.append({"request": request, "payload": payload, "t": config.now()})

    def lookup(self, request: str) -> dict | None:
        """The best stored payload for the same problem, or None."""
        now = config.now()
        best, best_score = None, -1.0
        for item in self.items:
            if now - item["t"] > config.SETTINGS["cache_ttl"]:
                continue
            if not self.same_problem(request, item["request"]):
                continue
            score = self.similarity(request, item["request"])
            if score > best_score:
                best, best_score = item, score
        if best is not None:
            self.hits += 1
            return dict(best["payload"], score=round(best_score, 3), matched=best["request"])
        self.misses += 1
        return None

    # ── escape hatches ──
    def forget(self, user_id: str, request: str) -> None:
        """'Regenerate fresh': drop this user's repeat entry and every shared entry that would match."""
        self.repeats.pop((user_id, _normal(request)), None)
        self.items = [it for it in self.items if not self.same_problem(request, it["request"])]

    def report_bad_hit(self, user_id: str, request: str, matched: str, kind: str) -> None:
        """A thumbs-down on a cached answer: remove the entry and block this pair for good."""
        if kind == "repeat":
            self.repeats.pop((user_id, _normal(request)), None)
            return
        self.items = [it for it in self.items if _normal(it["request"]) != _normal(matched)]
        if _normal(request) != _normal(matched):
            self.blocked.add((_normal(request), _normal(matched)))
            self.save_blocked(request, matched)

    def _blocked_path(self) -> Path:
        return Path(config.SETTINGS["cache_blocked_path"])

    def load_blocked(self) -> None:
        """Pairs recorded by earlier thumbs-downs (they are also golden test data)."""
        path = self._blocked_path()
        if path.exists():
            for pair in json.loads(path.read_text()):
                self.blocked.add((_normal(pair["request"]), _normal(pair["matched"])))

    def save_blocked(self, request: str, matched: str) -> None:
        path = self._blocked_path()
        pairs = json.loads(path.read_text()) if path.exists() else []
        if not any(_normal(p["request"]) == _normal(request) and _normal(p["matched"]) == _normal(matched)
                   for p in pairs):
            pairs.append({"request": request, "matched": matched})
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(pairs, indent=2) + "\n")

    def clear(self) -> None:
        """Empty the caches (blocked pairs stay: they are lessons, not cache entries)."""
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
    # Same topic, one extra rule: these broke the first version of the cache.
    ("Write a Python function that checks if a string is a palindrome.",
     "Write a Python function that checks if a string is a palindrome and returns yes or no", False),
    ("write a function to count the vowels in a string",
     "write a function to count the vowels in a string, treating y as a vowel", False),
    ("write a function that returns the nth fibonacci number",
     "write a function that returns the nth fibonacci number starting from 1", False),
    ("write a function to reverse a string", "write a function to reverse each word in a string", False),
    ("write a function to check if a string is a palindrome", "check if a sentence is a palindrome ignoring spaces", False),
    ("write a function to compute the factorial of a number", "compute the factorial of a number modulo 1000000007", False),
]


def threshold_sweep(cache: SemanticCache, pairs=LABELLED_PAIRS, thresholds=(0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9),
                    use_guard: bool = True) -> list[dict]:
    """For each threshold: true hits (same answer) and false hits (different answer).
    use_guard=False shows what embeddings alone would do (that is why the locks exist)."""
    rows = []
    for th in thresholds:
        hits = false_hits = 0
        for a, b, same in pairs:
            similar = cache.similarity(a, b) >= th
            locked = guard_terms(a) == guard_terms(b) and detail_words(a) == detail_words(b)
            ok = similar and (not use_guard or locked)
            hits += ok and same
            false_hits += ok and not same
        rows.append({"threshold": th, "hits": hits, "false_hits": false_hits,
                     "same_pairs": sum(p[2] for p in pairs), "different_pairs": sum(not p[2] for p in pairs)})
    return rows
