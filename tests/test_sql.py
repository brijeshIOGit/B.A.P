"""R7: validate_sql blocks 10 standard attacks + 5 of our own, allows CTE and UNION, clamps LIMIT."""

import sqlite3

import pytest

from codeforge.sql_agent import ensure_sample_db, run_sql, validate_sql

STANDARD_ATTACKS = [
    "DROP TABLE sales",
    "DELETE FROM sales",
    "UPDATE employees SET name = 'x'",
    "INSERT INTO sales (id, amount) VALUES (999, 1)",
    "SELECT * FROM users",
    "SELECT 1; DROP TABLE sales",
    "SELECT name FROM employees UNION SELECT email FROM users",
    "ATTACH DATABASE 'other.db' AS other",
    "PRAGMA table_info(users)",
    "SELECT * FROM sqlite_master",
]

OUR_ATTACKS = [
    "WITH u AS (SELECT * FROM users) SELECT * FROM u",  # PII hidden inside a CTE
    'SELECT * FROM sales WHERE id IN (SELECT id FROM "UsErS")',  # quoted, mixed-case table name
    "SELECT load_extension('evil')",  # function not on the allow-list
    "SELECT 1 -- harmless\n; DELETE FROM sales",  # stacked query hidden after a comment
    "SELECT * FROM sales LIMIT (SELECT 10000000)",  # LIMIT that is not a plain number
]

LEGIT = [
    "SELECT region, SUM(amount) AS total FROM sales GROUP BY region",
    "SELECT e.name, SUM(s.amount) FROM sales s JOIN employees e ON e.id = s.employee_id GROUP BY e.name",
    "WITH totals AS (SELECT region, SUM(amount) AS t FROM sales GROUP BY region) SELECT * FROM totals ORDER BY t DESC",
    "SELECT region FROM sales UNION SELECT region FROM employees",
]


@pytest.mark.parametrize("sql", STANDARD_ATTACKS + OUR_ATTACKS)
def test_attack_is_blocked(sql):
    assert validate_sql(sql)["ok"] is False


@pytest.mark.parametrize("sql", LEGIT)
def test_legitimate_query_is_allowed_and_runs(sql):
    verdict = validate_sql(sql)
    assert verdict["ok"], verdict["reason"]
    assert run_sql(verdict["sql"])["columns"]


def test_huge_limit_is_clamped_not_passed():
    verdict = validate_sql("SELECT * FROM sales LIMIT 10000000")
    assert verdict["ok"] and verdict["sql"].endswith("LIMIT 1000") and "clamped" in verdict["reason"]


def test_missing_limit_is_added():
    assert validate_sql("SELECT * FROM sales")["sql"].endswith("LIMIT 100")


def test_read_only_connection_refuses_writes_even_if_validation_were_skipped():
    with pytest.raises(sqlite3.Error):
        run_sql("DELETE FROM sales")


def test_users_table_cannot_be_read_even_if_validation_were_skipped():
    ensure_sample_db()
    with pytest.raises(sqlite3.Error):
        run_sql("SELECT email FROM users")
