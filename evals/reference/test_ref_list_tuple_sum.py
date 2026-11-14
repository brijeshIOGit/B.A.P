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


sum_collection = pick("sum_collection")


def test_list():
    assert sum_collection([1, 2, 3, 4]) == 10


def test_tuple():
    assert sum_collection((10, 20, 30)) == 60


def test_empty():
    assert sum_collection([]) == 0
    assert sum_collection(()) == 0


def test_list_uses_comprehension_and_tuple_uses_functional_style():
    source = inspect.getsource(sum_collection)
    assert " for " in source and "[" in source
    assert "reduce" in source or "map(" in source or "lambda" in source
