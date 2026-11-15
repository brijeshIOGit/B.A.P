# CodeForge chaos log

Provider: `offline` · chaos rate: **60%** of primary/backup model calls fail with 429 · seed 7

**Answered: 20 of 20.** Injected failures: 65. Simulated retry waiting (full jitter): 20.3 s. Tiers that wrote the code: {'template': 16, 'primary': 3, 'backup': 1}.

## 20 requests under chaos

|   # | request                                          | code_gen tier   | test_gen tier   |   injected 429s |   breaker skips | status                 |
|----:|:-------------------------------------------------|:----------------|:----------------|----------------:|----------------:|:-----------------------|
|   1 | Write a Python function that checks if a string  | template        | template        |               8 |               4 | answered: tests passed |
|   2 | write a function to compute the factorial of n   | template        | template        |               0 |               8 | answered: tests passed |
|   3 | write a function that returns the nth fibonacci  | template        | template        |               0 |               8 | answered: tests passed |
|   4 | write fizzbuzz as a python function              | primary         | primary         |               9 |               2 | answered: tests passed |
|   5 | If I pass a list, return its sum using a list co | template        | template        |               3 |               7 | answered: tests passed |
|   6 | write a function to reverse a string             | template        | template        |               0 |               8 | answered: tests passed |
|   7 | write a function to check if a number is prime   | template        | template        |               6 |               3 | answered: tests passed |
|   8 | sort a list of numbers in ascending order        | template        | template        |               5 |               6 | answered: tests passed |
|   9 | sort a list of numbers in descending order       | template        | template        |               0 |               8 | answered: tests passed |
|  10 | find the maximum value in a list                 | template        | template        |               5 |               3 | answered: tests passed |
|  11 | count the vowels in a string                     | primary         | template        |               6 |               4 | answered: tests passed |
|  12 | compute the gcd of two numbers                   | template        | template        |               0 |               8 | answered: tests passed |
|  13 | convert celsius to fahrenheit                    | backup          | template        |               5 |               2 | answered: tests passed |
|  14 | python function: is this word a palindrome       | primary         | primary         |               2 |               3 | answered: tests passed |
|  15 | factorial of a number please                     | template        | template        |               5 |               4 | answered: tests passed |
|  16 | give me fizz buzz up to n                        | template        | template        |               5 |               5 | answered: tests passed |
|  17 | nth fibonacci number function                    | template        | template        |               0 |               8 | answered: tests passed |
|  18 | reverse the characters of a string               | template        | template        |               3 |               6 | answered: tests passed |
|  19 | prime number checker                             | template        | template        |               3 |               1 | answered: tests passed |
|  20 | largest number in a list                         | template        | template        |               0 |               8 | answered: tests passed |

## Total outage (100% of model calls fail)

Template tasks still return code and tests. Unknown tasks get an honest refusal plus a ticket; no code is invented.

|   # | request                                          | code_gen tier   | test_gen tier   |   injected 429s |   breaker skips | status                                             | ticket   |
|----:|:-------------------------------------------------|:----------------|:----------------|----------------:|----------------:|:---------------------------------------------------|:---------|
|   1 | Write a Python function that checks if a string  | template        | template        |               3 |               7 | answered: tests passed                             |          |
|   2 | write fizzbuzz as a python function              | template        | template        |               0 |               8 | answered: tests passed                             |          |
|   3 | write a function to compute the factorial of n   | template        | template        |               0 |               8 | answered: tests passed                             |          |
|   4 | write a function to check if a sentence is a pan | none            | -               |               0 |              10 | answered: ⚠️ Cannot generate now: no model could p | T-1001   |
|   5 | write a function that reads a file and counts li | none            | -               |               0 |              10 | answered: ⚠️ Cannot generate now: no model could p | T-1002   |

## The degraded tier never auto-approves risky code

Big model down, request needs file access: served by `backup`, paused for approval: **True**, runs executed: **0**. An instructor must still resume the graph.
