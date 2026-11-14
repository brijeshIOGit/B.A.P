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


fibonacci = pick("fibonacci")


def test_sequence():
    assert [fibonacci(n) for n in range(10)] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


def test_large():
    assert fibonacci(40) == 102334155
