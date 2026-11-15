# CodeForge: decisions and assumptions

Every default chosen during the build, with the reason. Newest at the bottom of each section.

## Your answers to the Phase 1 questions

| # | You said | What we did |
|---|---|---|
| D1 | Build in the `Codeforge/` folder (its own git clone with a GitHub remote) so you can push it | The project lives in `Codeforge/` on a new local branch `codeforge` (the clone's default branch is `master`). Nothing was pushed. The outer `Agentic AI` repo was not touched. |
| D2 | Use "grok" with your keys, in a local `.env` that is never pushed | "Grok" (xAI, `XAI_API_KEY`) and "Groq" (groq.com, `GROQ_API_KEY`, the bootcamp provider) are different services, so both are supported. `.env` was created from `.env.example` with empty values; `.env` is in `.gitignore` (checked with `git check-ignore`). The provider is chosen from whichever key is filled (Groq, then xAI, then OpenAI) or forced with `MODEL_PROVIDER`. |
| D3 | 10 s timeout, 20/100 quota, **plus a reset button** | Reset exists three ways: UI button "Reset test data" (runs, tickets, feedback, logs, cache, breakers, chaos), UI button "Reset my quota", and `uv run codeforge-reset`. Quota reset keeps old runs so idempotency still works. |
| D4 | Only instructors approve | The approval counts only when the resuming `config` says `approver_role = instructor`. |
| D5 | Deny unknown imports | Any module not on a list is denied ("not on the allow-list"). |

## Environment

- **Models: Groq + open-weight gpt-oss only (your decision, same as the bootcamp notebook).** `.env` sets
  `MODEL_PROVIDER=groq`, big `openai/gpt-oss-120b`, small and backup `openai/gpt-oss-20b`. Because the circuit
  breaker is per model, a backup equal to the primary would be useless, so **the two sizes back each other up**:
  small tasks fall back to 120b, big tasks to 20b (`models._tiers`). If `GROQ_API_KEY` is empty, start-up prints a
  warning and runs offline instead of failing every call. The chaos switch can now fail one named model
  (`CHAOS.configure(models=…)`), which is how Groq rate limits really behave; `H-degraded-no-auto-approve` uses
  "the 120b model is down". The xAI / OpenAI / Ollama wiring stays in the code but is not used.

- **`OPENAI_API_KEY` is set in your shell environment** (not in a file in this repo). With no Groq/xAI key in `.env`, CodeForge picks OpenAI automatically. One early manual test run did call OpenAI (a few small requests). After that, every manual run used `MODEL_PROVIDER=offline`, and the tests force offline mode. To use Groq or Grok, fill the key in `.env` (or set `MODEL_PROVIDER=groq` / `xai`).
- Default model names are only defaults: Groq `openai/gpt-oss-120b` (big), `openai/gpt-oss-20b` (small), `llama-3.3-70b-versatile` (backup); xAI `grok-4` / `grok-3-mini` / `grok-3`. Change them in `.env` (`CODEFORGE_BIG_MODEL`, ...). xAI is reached through its OpenAI-compatible endpoint `https://api.x.ai/v1`, so no extra package is needed.
- Prices in `config.PRICES` are illustrative. The cost table prints the prices it used.
- `langchain-huggingface` / `sentence-transformers` (MiniLM) are **not installed** by default: they pull in PyTorch (~1 GB). The cache uses the offline `HashingEmbeddings`. To use MiniLM: `uv add langchain-huggingface sentence-transformers` and set `CODEFORGE_EMBEDDINGS=minilm` (threshold 0.75 is a starting point; rerun `threshold_sweep`).

## Design changes made while building (small, recorded here)

- `log_event`'s first argument is named `event` (a field called `kind` is common and clashed).
- Node functions take a parameter literally named `config` because LangGraph passes the run config only to a parameter with that name. Settings are imported as `SETTINGS` in `agents.py` to avoid the name clash.
- The offline brain lives in `brain.py` (not inside `models.py`) to keep `models.py` short.
- **Timeouts are not repaired.** A run killed at the timeout goes straight to an honest report. Repairing would re-run an infinite loop up to 3 times (30 s) for no benefit. Repairs are for failing tests only.
- After a repair, `code_gen` → `test_gen` → `policy_gate` run again (the repaired code is re-checked and gets a new run key).
- When `code_gen` falls back to the template tier, the requirements and tests are taken from the same template, so code and tests always share one contract.
- The policy treats `io` and `csv` as file access (needs approval). `list.remove` stays allowed; `os.remove` is blocked because `os` is denied.
- "Personal words" that make a request uncacheable: `my, mine, our, this, above, previous, earlier, again, fix` and pasted code blocks. Ordinary words like "that" and "it" were removed after they blocked normal requests from the cache.
- **Cache redesign (after the "multiple faces of a program" review).** The first version could reuse Alice's
  `is_palindrome` (True/False) for Bob's "…palindrome and returns yes or no": the extra rule was not a guard word,
  and cached tests always agree with cached code, so the report would say "passed" for the wrong spec. Fixes:
  (1) **detail words**: every meaningful word must match, only filler may differ; (2) **plain-only sharing**:
  requests with connectors (`and`, `but`, `treat`, quotes…) or more than 6 detail words are never shared across
  users (the same user's repeat cache still works); (3) **🔄 Regenerate fresh** button (`config.fresh`) skips and
  clears the cache for that request; (4) **👎 on a cached answer** removes the entry and writes the pair to
  `evals/cache_blocked_pairs.json`, so it can never match again (a cache flywheel). Labelled pairs grew from 14 to
  20 (6 new "same topic, one extra rule" negatives). New hard golden case `H-cache-no-false-reuse`. Cost saving
  went from 44% to 41%, golden pass rate unchanged.
- Cache threshold for the offline hashing embeddings: 0.6 (sweep: 6/6 true hits, 0/8 false hits with guard words; without guard words there are false hits).
- Escalation rule "two policy denials" lives in `refuse` (it is raised the moment the second denial happens), not in `triage`, so a chat is not escalated again on every later message.
- The repeat cache and semantic cache can both be switched off (`SETTINGS["use_cache"]`) for the cost "before" run. Idempotency still holds without them, because the run key is a hash of user + code + tests.
- Golden checks use a fresh user id per case so run counts never mix between cases.
- The chaos script uses a simulated clock: retry waits are added to the clock instead of slept, so the run takes seconds while breakers still cool down realistically.
- The approval demo request ("reads a file and counts lines") and Student B's code come from the offline brain's `OFFLINE_PROPOSALS`: what a careless model might write. They are not in the template library (templates are the trusted last-resort tier and never contain risky code).
- Flywheel cases are exported as **hard** cases, so the new case turns the gate red until it is fixed.

## Build process

- Milestones M1–M5 share most modules (`agents.py` imports the policy, sandbox, cache and SQL modules), so the code was written together and committed in milestone slices. `uv run pytest` was green before every commit.
