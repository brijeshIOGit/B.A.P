"""Hidden reference tests: written by us, never shown to a model."""
import pytest

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


factorial = pick("factorial")


def test_values():
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(6) == 720


def test_negative_raises():
    with pytest.raises(ValueError):
        factorial(-2)
