# Oracle Document Quality Report

## Summary Table

| setting_name | role | accuracy_mean | delta_vs_baseline | invalid_rate_mean | latency_ms_mean |
|---|---|---:|---:|---:|---:|
| oracle_gold_support | Upper bound (perfect document) | 0.8727 | 0.0633 | 0.0000 | 54.05 |
| baseline_k3_full_context | Normal RAG | 0.8093 | 0.0000 | 0.0000 | 89.84 |
| A_remove_answer_sentence | Lower bound (corrupted document) | 0.4823 | -0.3270 | 0.0000 | 77.92 |
| B_only_answer_sentence | Minimal ideal evidence | 0.8187 | 0.0093 | 0.0000 | 41.83 |

## Interpretation

- Observed ordering: oracle_gold_support=0.8727, baseline_k3_full_context=0.8093, A_remove_answer_sentence=0.4823.
- Expected ordering Upper >= Normal >= Lower is satisfied.
- B_only_answer_sentence is a compressed, answer-focused evidence condition rather than a standard oracle paragraph, so it should be interpreted as minimal ideal evidence instead of a full-document oracle.
