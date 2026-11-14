"""R4: guarded semantic cache. 0 false hits, at least 3 hits, sort ascending != sort descending."""

from codeforge import config
from codeforge.cache import LABELLED_PAIRS, SemanticCache, guard_terms, is_cacheable, threshold_sweep

PAYLOAD = {"requirements": {}, "code": "x", "tests": "y"}


def test_labelled_pairs_are_enough():
    assert len(LABELLED_PAIRS) >= 12


def test_chosen_threshold_has_no_false_hits_and_at_least_three_hits():
    cache = SemanticCache()
    row = [r for r in threshold_sweep(cache, thresholds=(cache.threshold,))][0]
    assert row["false_hits"] == 0 and row["hits"] >= 3


def test_without_guard_words_there_would_be_false_hits():
    rows = threshold_sweep(SemanticCache(), thresholds=(0.6,), use_guard=False)
    assert rows[0]["false_hits"] > 0  # this is why the guard exists


def test_sort_ascending_never_matches_sort_descending():
    cache = SemanticCache(threshold=0.0)  # even with no similarity bar
    cache.store("sort a list in ascending order", PAYLOAD)
    assert cache.lookup("sort a list in descending order") is None


def test_paraphrase_hits():
    cache = SemanticCache()
    cache.store("Write a Python function that checks if a string is a palindrome.", PAYLOAD)
    hit = cache.lookup("Write a python function to check whether a string is a palindrome")
    assert hit is not None and hit["code"] == "x"


def test_numbers_are_guard_terms():
    assert guard_terms("first 10 fibonacci") != guard_terms("first 20 fibonacci")


def test_personal_or_pii_requests_are_not_cacheable():
    assert not is_cacheable("fix my code", [])
    assert not is_cacheable("write a palindrome checker", ["secret"])
    assert is_cacheable("write a palindrome checker", [])


def test_entries_expire_after_the_ttl(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(config, "CLOCK", lambda: now[0])
    cache = SemanticCache()
    cache.store("write a function to reverse a string", PAYLOAD)
    now[0] += config.SETTINGS["cache_ttl"] + 1
    assert cache.lookup("write a function to reverse a string") is None


def test_repeat_cache_is_per_user():
    cache = SemanticCache()
    cache.remember_repeat("u_alice", "Reverse a string", PAYLOAD)
    assert cache.find_repeat("u_alice", "reverse a  string") is not None
    assert cache.find_repeat("u_bob", "reverse a string") is None
