"""R4: the shared cache reuses only the SAME plain problem. 0 false hits, at least 3 true hits."""

from codeforge import config
from codeforge.cache import (LABELLED_PAIRS, SemanticCache, detail_words, guard_terms, is_cacheable, is_shareable,
                             threshold_sweep)
from codeforge.graph import chat

PAYLOAD = {"requirements": {}, "code": "x", "tests": "y"}
PALINDROME = "Write a Python function that checks if a string is a palindrome."


def test_labelled_pairs_are_enough():
    assert len(LABELLED_PAIRS) >= 12


def test_chosen_threshold_has_no_false_hits_and_at_least_three_hits():
    cache = SemanticCache()
    row = threshold_sweep(cache, thresholds=(cache.threshold,))[0]
    assert row["false_hits"] == 0 and row["hits"] >= 3


def test_without_the_locks_there_would_be_false_hits():
    rows = threshold_sweep(SemanticCache(), thresholds=(0.6,), use_guard=False)
    assert rows[0]["false_hits"] > 0  # this is why the locks exist


# ── the chats from the design discussion ──

def test_chat1_plain_paraphrase_is_reused():
    cache = SemanticCache()
    cache.store(PALINDROME, PAYLOAD)
    assert cache.lookup("Write a python function to check whether a string is a palindrome") is not None


def test_chat5_extra_output_rule_is_not_reused():
    cache = SemanticCache()
    cache.store(PALINDROME, PAYLOAD)
    bob = "Write a Python function that checks if a string is a palindrome and returns yes or no"
    assert cache.lookup(bob) is None
    assert not is_shareable(bob, [])  # "and" adds a rule: never shared at all


def test_chat6_extra_rule_is_not_reused():
    cache = SemanticCache()
    cache.store("write a function to count the vowels in a string", PAYLOAD)
    assert cache.lookup("write a function to count the vowels in a string, treating y as a vowel") is None


def test_sort_ascending_never_matches_sort_descending():
    cache = SemanticCache(threshold=0.0)  # even with no similarity bar
    cache.store("sort a list in ascending order", PAYLOAD)
    assert cache.lookup("sort a list in descending order") is None


def test_detail_words_ignore_filler_only():
    assert detail_words("Write a python function to check whether a string is a palindrome") == \
        detail_words("checks if a string is a palindrome")
    assert detail_words("palindrome returns yes or no") != detail_words("palindrome")


def test_numbers_are_guard_terms():
    assert guard_terms("first 10 fibonacci") != guard_terms("first 20 fibonacci")


def test_personal_pii_or_specific_requests_are_not_shared():
    assert not is_cacheable("fix my code", [])
    assert not is_shareable("write a palindrome checker", ["secret"])
    assert not is_shareable('palindrome that prints "ok"', [])
    assert is_shareable("write a palindrome checker", [])


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


# ── escape hatches ──

def test_thumbs_down_removes_the_entry_and_blocks_the_pair():
    cache = SemanticCache()
    bob = "Write a python function to check whether a string is a palindrome"
    cache.store(PALINDROME, PAYLOAD)
    hit = cache.lookup(bob)
    cache.report_bad_hit("u_bob", bob, hit["matched"], "semantic")
    assert cache.lookup(bob) is None
    cache.store(PALINDROME, PAYLOAD)  # even if stored again, this pair stays blocked
    assert cache.lookup(bob) is None
    # A brand-new cache (e.g. after a restart) reloads the blocked pair from disk.
    assert (bob.lower().strip(), PALINDROME.lower().strip()) in SemanticCache().blocked


def test_regenerate_fresh_skips_the_cache(app):
    chat(app, PALINDROME, user_id="u_alice", thread="r1")
    second = chat(app, PALINDROME, user_id="u_alice", thread="r2")
    assert second["cache_hit"] == "repeat"
    from codeforge.graph import make_config
    from langchain_core.messages import HumanMessage

    fresh = app.invoke({"messages": [HumanMessage(content=PALINDROME)]},
                       make_config("u_alice", "r3", fresh=True))
    assert fresh["cache_hit"] == "" and fresh["served_by"]["code_gen"] != "cache"


def test_other_users_get_a_plain_request_from_the_shared_cache(app):
    chat(app, PALINDROME, user_id="u_alice", thread="s1")
    bob = chat(app, "Write a python function to check whether a string is a palindrome", user_id="u_bob", thread="s2")
    assert bob["cache_hit"] == "semantic" and bob["run"]["exit_code"] == 0
