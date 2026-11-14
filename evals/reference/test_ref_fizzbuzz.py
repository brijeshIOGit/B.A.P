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


fizzbuzz = pick("fizzbuzz")


def test_first_fifteen():
    assert fizzbuzz(15) == ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"]
