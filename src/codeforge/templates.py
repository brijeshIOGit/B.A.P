"""Template tier (R5): hand-written code + tests for common tasks. No model needed.

When every model is down, these tasks still get correct code and tests.
Anything not here gets an honest "cannot generate now", never invented code.

Matching is plain keywords:
  any  - at least one of these phrases must appear
  all  - every one of these must appear
  none - none of these may appear (keeps "sort ascending" and "sort descending" apart)
"""

import re

TEMPLATES = [
    {
        "name": "list_tuple_sum",
        "any": ["sum"],
        "all": ["list", "tuple"],
        "function_name": "sum_collection",
        "params": ["data: list | tuple"],
        "returns": "the sum of the numbers",
        "edge_cases": ["empty list or tuple gives 0", "other types raise TypeError"],
        "code": '''from functools import reduce


def sum_collection(data):
    """Sum a list with a list comprehension, and a tuple with functional programming."""
    if isinstance(data, list):
        return sum([item for item in data])
    if isinstance(data, tuple):
        return reduce(lambda total, item: total + item, data, 0)
    raise TypeError("data must be a list or a tuple")
''',
        "tests": '''import pytest
from solution import sum_collection


def test_list_sum():
    assert sum_collection([1, 2, 3]) == 6


def test_tuple_sum():
    assert sum_collection((4, 5, 6)) == 15


def test_empty_list_and_tuple():
    assert sum_collection([]) == 0
    assert sum_collection(()) == 0


def test_negative_and_float_values():
    assert sum_collection([-1, 2.5]) == 1.5


def test_other_types_raise():
    with pytest.raises(TypeError):
        sum_collection("123")
''',
    },
    {
        "name": "palindrome",
        "any": ["palindrome", "reads the same backward", "same forwards and backwards", "same backwards"],
        "none": ["sentence is a pangram"],
        "function_name": "is_palindrome",
        "params": ["text: str"],
        "returns": "True if text reads the same backwards, ignoring case, spaces and punctuation",
        "edge_cases": ["empty string is a palindrome", "mixed case", "spaces and punctuation are ignored"],
        "code": '''def is_palindrome(text):
    """Return True if text reads the same backwards (ignores case, spaces and punctuation)."""
    cleaned = [ch.lower() for ch in text if ch.isalnum()]
    return cleaned == cleaned[::-1]
''',
        "tests": '''from solution import is_palindrome


def test_simple_palindrome():
    assert is_palindrome("racecar") is True


def test_not_a_palindrome():
    assert is_palindrome("hello") is False


def test_mixed_case():
    assert is_palindrome("Level") is True


def test_sentence_with_spaces_and_punctuation():
    assert is_palindrome("A man, a plan, a canal: Panama") is True


def test_empty_string():
    assert is_palindrome("") is True


def test_single_character():
    assert is_palindrome("x") is True
''',
    },
    {
        "name": "factorial",
        "any": ["factorial"],
        "function_name": "factorial",
        "params": ["n: int"],
        "returns": "n! as an int",
        "edge_cases": ["0! is 1", "negative numbers raise ValueError"],
        "code": '''def factorial(n):
    """Return n! for a non-negative integer n."""
    if n < 0:
        raise ValueError("factorial is not defined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
''',
        "tests": '''import pytest
from solution import factorial


def test_zero():
    assert factorial(0) == 1


def test_small_numbers():
    assert factorial(1) == 1
    assert factorial(5) == 120


def test_larger_number():
    assert factorial(10) == 3628800


def test_negative_raises():
    with pytest.raises(ValueError):
        factorial(-1)
''',
    },
    {
        "name": "fibonacci",
        "any": ["fibonacci"],
        "none": ["list of", "first n", "sequence up to", "series up to", "up to n"],  # those want a list
        "function_name": "fibonacci",
        "params": ["n: int"],
        "returns": "the n-th Fibonacci number (fibonacci(0) = 0, fibonacci(1) = 1)",
        "edge_cases": ["n = 0 and n = 1", "negative n raises ValueError"],
        "code": '''def fibonacci(n):
    """Return the n-th Fibonacci number, with fibonacci(0) = 0 and fibonacci(1) = 1."""
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
''',
        "tests": '''import pytest
from solution import fibonacci


def test_first_values():
    assert [fibonacci(i) for i in range(7)] == [0, 1, 1, 2, 3, 5, 8]


def test_bigger_value():
    assert fibonacci(30) == 832040


def test_negative_raises():
    with pytest.raises(ValueError):
        fibonacci(-3)
''',
    },
    {
        "name": "fizzbuzz",
        "any": ["fizzbuzz", "fizz buzz"],
        "function_name": "fizzbuzz",
        "params": ["n: int"],
        "returns": "a list of strings for 1..n with Fizz, Buzz and FizzBuzz",
        "edge_cases": ["n = 0 gives an empty list", "multiples of 15 give FizzBuzz"],
        "code": '''def fizzbuzz(n):
    """Return the FizzBuzz strings for 1..n."""
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result
''',
        "tests": '''from solution import fizzbuzz


def test_first_fifteen():
    out = fizzbuzz(15)
    assert out[:5] == ["1", "2", "Fizz", "4", "Buzz"]
    assert out[14] == "FizzBuzz"


def test_length():
    assert len(fizzbuzz(30)) == 30


def test_zero():
    assert fizzbuzz(0) == []
''',
    },
    {
        "name": "reverse_string",
        "any": ["reverse"],
        "all": ["string"],
        "none": ["list", "sort"],
        "function_name": "reverse_string",
        "params": ["text: str"],
        "returns": "the text reversed",
        "edge_cases": ["empty string"],
        "code": '''def reverse_string(text):
    """Return the text reversed."""
    return text[::-1]
''',
        "tests": '''from solution import reverse_string


def test_word():
    assert reverse_string("hello") == "olleh"


def test_empty():
    assert reverse_string("") == ""


def test_spaces_kept():
    assert reverse_string("ab c") == "c ba"
''',
    },
    {
        "name": "is_prime",
        "any": ["prime"],
        "function_name": "is_prime",
        "params": ["n: int"],
        "returns": "True if n is a prime number",
        "edge_cases": ["0 and 1 are not prime", "negative numbers are not prime"],
        "code": '''def is_prime(n):
    """Return True if n is a prime number."""
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True
''',
        "tests": '''from solution import is_prime


def test_small_primes():
    assert [n for n in range(20) if is_prime(n)] == [2, 3, 5, 7, 11, 13, 17, 19]


def test_not_prime():
    assert is_prime(1) is False
    assert is_prime(21) is False


def test_negative():
    assert is_prime(-7) is False
''',
    },
    {
        "name": "sort_descending",
        "any": ["sort", "order"],
        "all": ["desc"],
        "function_name": "sort_descending",
        "params": ["numbers: list"],
        "returns": "a new list sorted from largest to smallest",
        "edge_cases": ["empty list", "the input list is not changed"],
        "code": '''def sort_descending(numbers):
    """Return a new list sorted from largest to smallest."""
    return sorted(numbers, reverse=True)
''',
        "tests": '''from solution import sort_descending


def test_sorts_descending():
    assert sort_descending([3, 1, 2]) == [3, 2, 1]


def test_empty():
    assert sort_descending([]) == []


def test_input_unchanged():
    data = [1, 3, 2]
    sort_descending(data)
    assert data == [1, 3, 2]
''',
    },
    {
        "name": "sort_ascending",
        "any": ["sort"],
        "none": ["desc"],
        "function_name": "sort_ascending",
        "params": ["numbers: list"],
        "returns": "a new list sorted from smallest to largest",
        "edge_cases": ["empty list", "the input list is not changed"],
        "code": '''def sort_ascending(numbers):
    """Return a new list sorted from smallest to largest."""
    return sorted(numbers)
''',
        "tests": '''from solution import sort_ascending


def test_sorts_ascending():
    assert sort_ascending([3, 1, 2]) == [1, 2, 3]


def test_empty():
    assert sort_ascending([]) == []


def test_input_unchanged():
    data = [2, 1]
    sort_ascending(data)
    assert data == [2, 1]
''',
    },
    {
        "name": "find_max",
        "any": ["maximum", "largest", "biggest", "max"],
        "none": ["sort"],
        "function_name": "find_max",
        "params": ["numbers: list"],
        "returns": "the largest number",
        "edge_cases": ["empty list raises ValueError", "negative numbers"],
        "code": '''def find_max(numbers):
    """Return the largest number in a non-empty list."""
    if not numbers:
        raise ValueError("the list is empty")
    largest = numbers[0]
    for value in numbers[1:]:
        if value > largest:
            largest = value
    return largest
''',
        "tests": '''import pytest
from solution import find_max


def test_basic():
    assert find_max([3, 9, 2]) == 9


def test_negative():
    assert find_max([-5, -2, -9]) == -2


def test_empty_raises():
    with pytest.raises(ValueError):
        find_max([])
''',
    },
    {
        "name": "count_vowels",
        "any": ["vowel"],
        "function_name": "count_vowels",
        "params": ["text: str"],
        "returns": "how many vowels (a, e, i, o, u, any case) are in text",
        "edge_cases": ["empty string", "upper case vowels"],
        "code": '''def count_vowels(text):
    """Count the vowels a, e, i, o, u (any case) in text."""
    return sum(1 for ch in text.lower() if ch in "aeiou")
''',
        "tests": '''from solution import count_vowels


def test_word():
    assert count_vowels("education") == 5


def test_upper_case():
    assert count_vowels("AEIOU") == 5


def test_empty():
    assert count_vowels("") == 0
''',
    },
    {
        "name": "gcd",
        "any": ["gcd", "greatest common divisor", "hcf", "highest common factor"],
        "function_name": "gcd",
        "params": ["a: int", "b: int"],
        "returns": "the greatest common divisor of a and b",
        "edge_cases": ["one number is 0", "negative numbers"],
        "code": '''def gcd(a, b):
    """Return the greatest common divisor of a and b (Euclid's algorithm)."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a
''',
        "tests": '''from solution import gcd


def test_basic():
    assert gcd(12, 18) == 6


def test_with_zero():
    assert gcd(0, 7) == 7


def test_negative():
    assert gcd(-8, 12) == 4
''',
    },
    {
        "name": "celsius_to_fahrenheit",
        "any": ["celsius", "fahrenheit"],
        "function_name": "celsius_to_fahrenheit",
        "params": ["celsius: float"],
        "returns": "the temperature in Fahrenheit",
        "edge_cases": ["freezing and boiling points", "negative values"],
        "code": '''def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32
''',
        "tests": '''from solution import celsius_to_fahrenheit


def test_freezing():
    assert celsius_to_fahrenheit(0) == 32


def test_boiling():
    assert celsius_to_fahrenheit(100) == 212


def test_minus_forty():
    assert celsius_to_fahrenheit(-40) == -40
''',
    },
    {
        "name": "stack",
        "any": ["stack"],
        "none": ["overflow"],
        "function_name": "Stack",
        "params": [],
        "returns": "a Stack class with push, pop, peek, is_empty and len()",
        "edge_cases": ["pop or peek on an empty stack raises IndexError"],
        "code": '''class Stack:
    """A last-in, first-out stack backed by a list."""

    def __init__(self):
        self._items = []

    def push(self, item):
        """Put an item on top."""
        self._items.append(item)

    def pop(self):
        """Remove and return the top item. Raises IndexError when empty."""
        if not self._items:
            raise IndexError("pop from an empty stack")
        return self._items.pop()

    def peek(self):
        """Return the top item without removing it. Raises IndexError when empty."""
        if not self._items:
            raise IndexError("peek at an empty stack")
        return self._items[-1]

    def is_empty(self):
        return not self._items

    def __len__(self):
        return len(self._items)
''',
        "tests": '''import pytest
from solution import Stack


def test_push_then_pop_is_lifo():
    s = Stack()
    s.push(1)
    s.push(2)
    assert s.pop() == 2 and s.pop() == 1


def test_peek_does_not_remove():
    s = Stack()
    s.push("a")
    assert s.peek() == "a" and len(s) == 1


def test_empty_stack():
    s = Stack()
    assert s.is_empty() and len(s) == 0
    with pytest.raises(IndexError):
        s.pop()
    with pytest.raises(IndexError):
        s.peek()
''',
    },
]


def _has(text: str, phrase: str) -> bool:
    """True if the phrase appears in the text (lower-cased)."""
    return phrase in text


# Words that add a rule the templates do not follow. A request with one of them never gets a
# template: in an outage it gets an honest "cannot generate now" instead of code that passes
# the template's tests but ignores what the student asked for.
SPEC_PHRASES = (
    "case-sensitive", "case sensitive", "in place", "in-place", "recursion", "recursive", "recursively",
    "iterative", "iteratively",
    "1-indexed", "0-indexed", "one-indexed", "zero-indexed", "1 indexed", "0 indexed", "raise", "raises", "raising", "exception", "exceptions",
    "error", "errors", "without using", "don't use", "do not use", "dont use", "not ignore", "doesn't ignore",
    "dont ignore", "don't ignore", "generator", "yield", "memo", "memoize", "memoized", "memoization", "lru_cache", "decorator", "async", "return none",
    "one line", "one-liner", "big o", "o(1)", "o(n)", "o(log", "dictionary", "dict", "type hint",
    "docstring", "unicode", "bigint", "modulo", "mod 10",
)


def has_extra_spec(request: str) -> bool:
    """True when the request adds a rule (see SPEC_PHRASES) that no template is built for."""
    text = re.sub(r"\s+", " ", (request or "").lower())
    # Whole words only: "raise" must not fire on "praise", nor "memo" on "memory".
    return any(re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", text) for p in SPEC_PHRASES)


def match_template(request: str) -> dict | None:
    """The first template whose keywords fit this request, or None."""
    text = re.sub(r"\s+", " ", (request or "").lower())
    if has_extra_spec(text):
        return None
    for template in TEMPLATES:
        if not any(_has(text, p) for p in template.get("any", [])):
            continue
        if not all(_has(text, p) for p in template.get("all", [])):
            continue
        if any(_has(text, p) for p in template.get("none", [])):
            continue
        return template
    return None


def template_requirements(template: dict) -> dict:
    """The requirements dict that this template satisfies."""
    return {
        "function_name": template["function_name"],
        "params": template["params"],
        "returns": template["returns"],
        "edge_cases": template["edge_cases"],
        "examples": [],
    }


def template_names() -> list[str]:
    """Names of every template (for the README and the UI)."""
    return [t["name"] for t in TEMPLATES]
