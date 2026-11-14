# CodeForge cost table (R4)

Provider: `offline` · fixed 20-request script (two students, paraphrases and one exact repeat).

* **Before:** big model for every agent, no cache, no compaction.
* **After:** cascade (small model for triage, router, requirements, ticket, compaction), guarded semantic cache + same-user repeat cache, chat compaction.

| agent        |   calls before |   calls after |   tokens before |   tokens after |   sec before |   sec after |   USD before |   USD after |
|:-------------|---------------:|--------------:|----------------:|---------------:|-------------:|------------:|-------------:|------------:|
| code_gen     |             20 |            13 |            3805 |           2442 |        0.001 |       0     |     0.001092 |    0.000701 |
| compact      |              0 |             2 |               0 |            927 |        0     |       0     |     0        |    9.7e-05  |
| requirements |             20 |            13 |            1956 |           1250 |        0.001 |       0     |     0.000714 |    0.000227 |
| router       |             20 |            13 |            2041 |           1320 |        0.001 |       0     |     0.000477 |    0.000155 |
| test_gen     |             20 |            13 |            5558 |           3527 |        0.001 |       0     |     0.001598 |    0.001001 |
| triage       |             20 |            20 |            7405 |           8034 |        0.002 |       0.001 |     0.001336 |    0.000715 |
| TOTAL        |            100 |            74 |           20765 |          17500 |        0.004 |       0.003 |     0.005217 |    0.002896 |

**Saving: 44% cheaper** (target ≥ 30%).

Golden soft-case pass rate: before **100%**, after **100%** (must not drop).

Prices used (USD per 1M tokens, illustrative, edit in `src/codeforge/config.py`):

| model | input | output |
|---|---|---|
| `openai/gpt-oss-120b` | 0.15 | 0.6 |
| `openai/gpt-oss-20b` | 0.075 | 0.3 |
| `llama-3.3-70b-versatile` | 0.59 | 0.79 |

Offline, token counts come from the rule-based brain (about 4 characters per token), priced as the model it stands in for. With a real key the same script meters real usage.
