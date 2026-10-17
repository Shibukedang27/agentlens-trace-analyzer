# agentlens report

> Synthetic traces, example prices: numbers illustrate the method, not real billing.

## Overview

| Metric | Value |
|---|---:|
| Runs | 600 |
| Spans | 5,676 |
| Agents | 5 |
| Total tokens | 9,644,207 |
| Total cost | $41.98 |
| Estimated savings | $6.98 (16.6%) |
| Runs flagged by detectors | 173 (187 findings) |
| Span error rate | 6.6% |
| LLM latency p50 / p95 | 2.5s / 6.8s |
| Run duration p50 / p95 | 16.9s / 32.5s |

## Waste by detector

| Detector | Findings | Runs | Wasted tokens | Est. savings |
|---|---:|---:|---:|---:|
| Tool-call loop | 39 | 39 | 511,916 | $2.22 |
| Retry storm | 19 | 19 | 151,514 | $0.6371 |
| Idle / no-op run | 54 | 54 | 186,480 | $0.5398 |
| Prompt bloat | 30 | 30 | 769,651 | $3.35 |
| Oversized model | 45 | 45 | 0 | $0.2314 |

Per-detector savings are not capped; the overview total caps each run at its own cost.

## Cost by agent

| Agent | Runs | Total cost | Avg cost/run | p50 run | p95 run | p95 LLM | Error rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| support-agent | 173 | $15.88 | $0.0918 | 19.1s | 30.7s | 6.9s | 7.4% |
| sql-analyst | 105 | $11.58 | $0.1103 | 20.3s | 33.2s | 6.8s | 5.2% |
| research-agent | 113 | $10.34 | $0.0915 | 23.0s | 36.6s | 6.8s | 4.8% |
| ops-heartbeat | 146 | $4.18 | $0.0286 | 12.7s | 22.2s | 6.9s | 9.3% |
| faq-bot | 63 | $0.0070 | $0.0001 | 2.7s | 5.5s | 5.4s | 0.0% |

## Top 10 findings by estimated savings

| Detector | Agent | Trace | Finding | Est. savings |
|---|---|---|---|---:|
| prompt_bloat | support-agent | `071a8d5363e0` | Prompt grew 9.9x over 8 LLM calls | $0.2160 |
| prompt_bloat | support-agent | `cbf9c222c669` | Prompt grew 11.7x over 8 LLM calls | $0.2068 |
| prompt_bloat | sql-analyst | `759a97252e77` | Prompt grew 10.1x over 8 LLM calls | $0.2059 |
| prompt_bloat | support-agent | `4ba408a8d15d` | Prompt grew 12.4x over 7 LLM calls | $0.1917 |
| prompt_bloat | research-agent | `502d35938480` | Prompt grew 8.8x over 11 LLM calls | $0.1800 |
| prompt_bloat | support-agent | `332066bf0060` | Prompt grew 8.4x over 8 LLM calls | $0.1797 |
| tool_loop | support-agent | `c36f62b6ab3c` | get_order called 8x with identical arguments | $0.1593 |
| prompt_bloat | support-agent | `af3ae9ef663b` | Prompt grew 8.3x over 7 LLM calls | $0.1575 |
| prompt_bloat | sql-analyst | `d43baff3cef9` | Prompt grew 6.8x over 7 LLM calls | $0.1569 |
| prompt_bloat | sql-analyst | `27af99958e5a` | Prompt grew 8.1x over 7 LLM calls | $0.1545 |
