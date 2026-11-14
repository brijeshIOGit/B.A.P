"""Safe SQL (R7): a sample database, validate_sql() with sqlglot, and a read-only runner.

The model writes the SQL. Plain Python decides if it may run:
  one statement, SELECT only (CTE and UNION allowed), allow-listed tables and functions,
  LIMIT added or clamped. Then it runs on a read-only connection (second layer) that also
  refuses to read the users table (third layer).
"""

import random
import sqlite3
import time
from pathlib import Path

import sqlglot
from sqlglot import exp

from . import config

ALLOWED_TABLES = {"sales", "employees"}
ALLOWED_FUNCTIONS = {"COUNT", "SUM", "AVG", "MIN", "MAX", "ROUND", "ABS", "LOWER", "UPPER", "LENGTH",
                     "COALESCE", "SUBSTRING", "TRIM", "CAST", "CASE", "IF", "NULLIF", "DATE", "STRFTIME",
                     "TIME_TO_STR", "STR_TO_TIME", "TS_OR_DS_TO_DATE", "TS_OR_DS_TO_TIMESTAMP", "IFNULL",
                     "DISTINCT"}  # some names are sqlglot's internal names for SQLite date functions
DEFAULT_LIMIT = 100
MAX_LIMIT = 1000

# Statement types that change data or reach outside the database. Blocked anywhere in the tree.
FORBIDDEN = tuple(getattr(exp, name) for name in
                  ("Insert", "Update", "Delete", "Drop", "Create", "Alter", "Command", "Pragma", "Attach",
                   "Detach", "Transaction", "Merge", "TruncateTable")
                  if hasattr(exp, name))


def _block(reason: str, sql: str = "") -> dict:
    return {"ok": False, "reason": reason, "sql": sql}


def _function_name(node) -> str:
    """Upper-case SQL name of a function node."""
    if isinstance(node, exp.Anonymous):
        return str(node.name).upper()
    return node.sql_name().upper()


def validate_sql(sql: str) -> dict:
    """Check one query. Returns {"ok", "reason", "sql"}; "sql" is the safe, rewritten query."""
    try:
        statements = [s for s in sqlglot.parse(sql or "", read="sqlite") if s is not None]
    except sqlglot.errors.ParseError:
        return _block("the SQL does not parse")
    if len(statements) != 1:
        return _block(f"exactly one statement is allowed (found {len(statements)})")
    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        return _block(f"only SELECT queries are allowed (got {tree.key.upper()})")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN):
            return _block(f"{node.key.upper()} is not allowed")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        name = str(table.name).lower()
        if name not in ALLOWED_TABLES and name not in cte_names:
            return _block(f"table `{name}` is not on the allow-list (allowed: sales, employees)")

    for func in tree.find_all(exp.Func):
        name = _function_name(func)
        if name not in ALLOWED_FUNCTIONS:
            return _block(f"function `{name}` is not on the allow-list")

    note = ""
    limit = tree.args.get("limit")
    if limit is None:
        tree = tree.limit(DEFAULT_LIMIT)
        note = f"LIMIT {DEFAULT_LIMIT} added"
    else:
        value = limit.expression
        if not (isinstance(value, exp.Literal) and value.is_int):
            return _block("LIMIT must be a plain number")
        if int(value.this) > MAX_LIMIT:
            note = f"LIMIT {int(value.this)} clamped to {MAX_LIMIT}"
            tree = tree.limit(MAX_LIMIT)
    return {"ok": True, "reason": note or "ok", "sql": tree.sql(dialect="sqlite")}


# ── sample database ──────────────────────────────────────────────────────────

def ensure_sample_db(path: str | None = None) -> str:
    """Create data/sample.db (sales, employees, users with fake PII) if it does not exist."""
    path = path or config.SETTINGS["sample_db_path"]
    if Path(path).exists():
        return path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, region TEXT, hire_date TEXT);
        CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, product TEXT, amount REAL, sale_date TEXT,
                            employee_id INTEGER REFERENCES employees(id));
        CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, phone TEXT, aadhaar TEXT,
                            password_hash TEXT);
    """)
    staff = [("Asha Rao", "Sales", "North"), ("Vikram Singh", "Sales", "South"), ("Meera Iyer", "Sales", "West"),
             ("Kabir Das", "Sales", "East"), ("Nisha Jain", "Support", "North"), ("Rahul Mehta", "Engineering", "West"),
             ("Priya Nair", "Engineering", "South"), ("Arjun Sethi", "Finance", "East")]
    for i, (name, dept, region) in enumerate(staff, start=1):
        con.execute("INSERT INTO employees VALUES (?,?,?,?,?)", (i, name, dept, region, f"202{i % 5}-0{1 + i % 9}-15"))
    products = ["Laptop", "Mouse", "Keyboard", "Monitor", "Headset"]
    for i in range(1, 61):
        emp = rng.randint(1, 4)
        con.execute("INSERT INTO sales VALUES (?,?,?,?,?,?)",
                    (i, staff[emp - 1][2], rng.choice(products), round(rng.uniform(500, 60000), 2),
                     f"2026-{rng.randint(1, 10):02d}-{rng.randint(1, 28):02d}", emp))
    people = [("Ravi Kumar", "ravi@example.com", "9876543210", "2345 6789 0123"),
              ("Sara Khan", "sara@example.com", "9123456780", "3456 7890 1234"),
              ("John Mathew", "john@example.com", "9988776655", "4567 8901 2345")]
    for i, (name, email, phone, aadhaar) in enumerate(people, start=1):
        con.execute("INSERT INTO users VALUES (?,?,?,?,?,?)", (i, name, email, phone, aadhaar, "x" * 32))
    con.commit()
    con.close()
    return path


def run_sql(sql: str, path: str | None = None, max_seconds: float = 2.0) -> dict:
    """Run a validated query on a READ-ONLY connection. Returns columns and rows."""
    path = ensure_sample_db(path)
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        con.execute("PRAGMA query_only = ON")

        def authorizer(action, arg1, arg2, db_name, source):
            # Third layer: never read the users table, whatever the SQL says.
            if action == sqlite3.SQLITE_READ and str(arg1).lower() == "users":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        con.set_authorizer(authorizer)
        deadline = time.monotonic() + max_seconds
        con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        cur = con.execute(sql)
        columns = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(MAX_LIMIT)
        return {"columns": columns, "rows": [list(r) for r in rows]}
    finally:
        con.close()
