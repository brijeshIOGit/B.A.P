"""Hidden reference tests: written by us, never shown to a model."""
import inspect

import solution


def pick(name):
    """The function under test: the expected name, or the only public function in solution.py."""
    if hasattr(solution, name):
        return getattr(solution, name)
    found = [f for n, f in vars(solution).items()
             if inspect.isfunction(f) and f.__module__ == "solution" and not n.startswith("_")]
    assert len(found) == 1, "could not find the function under test"
    return found[0]


is_palindrome = pick("is_palindrome")


def test_basic():
    assert is_palindrome("racecar")
    assert not is_palindrome("python")


def test_case_insensitive():
    assert is_palindrome("Madam")


def test_ignores_spaces_and_punctuation():
    assert is_palindrome("Was it a car or a cat I saw?")


def test_empty_and_single():
    assert is_palindrome("")
    assert is_palindrome("a")
