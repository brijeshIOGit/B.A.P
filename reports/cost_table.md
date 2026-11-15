# CodeForge cost table (R4)

Provider: `offline` · fixed 20-request script (two students, paraphrases and one exact repeat).

* **Before:** big model for every agent, no cache, no compaction.
* **After:** cascade (small model for triage, router, requirements, ticket, compaction), guarded semantic cache + same-user repeat cache, chat compaction.

| agent        |   calls before |   calls after |   tokens before |   tokens after |   sec before |   sec after |   USD before |   USD after |
|:-------------|---------------:|--------------:|----------------:|---------------:|-------------:|------------:|-------------:|------------:|
| code_gen     |             20 |            14 |            3805 |           2630 |        0     |       0     |     0.001092 |    0.000759 |
| compact      |              0 |             2 |               0 |            927 |        0     |       0     |     0        |    9.7e-05  |
| requirements |             20 |            14 |            1956 |           1338 |        0.001 |       0     |     0.000714 |    0.000242 |
| router       |             20 |            14 |            2041 |           1422 |        0.001 |       0     |     0.000477 |    0.000166 |
| test_gen     |             20 |            14 |            5558 |           3808 |        0.001 |       0     |     0.001598 |    0.001084 |
| triage       |             20 |            20 |            7405 |           8134 |        0.002 |       0.001 |     0.001336 |    0.000723 |
| TOTAL        |            100 |            78 |           20765 |          18259 |        0.004 |       0.003 |     0.005217 |    0.003072 |

**Saving: 41% cheaper** (target ≥ 30%).

Golden soft-case pass rate: before **100%**, after **100%** (must not drop).

Prices used (USD per 1M tokens, illustrative, edit in `src/codeforge/config.py`):

| model | input | output |
|---|---|---|
| `openai/gpt-oss-120b` | 0.15 | 0.6 |
| `openai/gpt-oss-20b` | 0.075 | 0.3 |
| `llama-3.3-70b-versatile` | 0.59 | 0.79 |

Offline, token counts come from the rule-based brain (about 4 characters per token), priced as the model it stands in for. With a real key the same script meters real usage.
