# Figures Report

## Source Files Used

- `/Users/yuanyuan/Documents/nlp_project/Jing/outputs/context_interventions/aggregate_context_interventions_20260309_222105.csv`
- `/Users/yuanyuan/Documents/nlp_project/Jing/outputs/oracle_doc_quality/aggregate_oracle_doc_quality_20260312_001625.csv`
- `/Users/yuanyuan/Documents/nlp_project/Jing/outputs/seed_eval_n1000/aggregate_seed_eval_n1000_20260309_115552.csv`

## Settings Found

- `A_remove_answer_sentence`
- `B_only_answer_sentence`
- `C_k1_plus_random_distractor`
- `D_shuffle_top3_order`
- `baseline_k3_full_context`
- `llm_only`
- `oracle_gold_support`

## Figure 1 Values

| label | raw_setting | accuracy | error_bar |
|---|---|---:|---:|
| LLM-only | `llm_only` | 0.4373 | 0.0101 |
| RAG baseline | `baseline_k3_full_context` | 0.8093 | 0.0038 |
| Oracle gold support | `oracle_gold_support` | 0.8727 | 0.0057 |
| Remove answer sentence | `A_remove_answer_sentence` | 0.4823 | 0.0006 |
| Only answer sentence | `B_only_answer_sentence` | 0.8187 | 0.0074 |
| Add distractor context | `C_k1_plus_random_distractor` | 0.7660 | 0.0052 |
| Shuffle context order | `D_shuffle_top3_order` | 0.8187 | 0.0091 |

## Figure 2 Values

| label | raw_setting | delta_vs_baseline | error_bar |
|---|---|---:|---:|
| Remove answer sentence | `A_remove_answer_sentence` | -0.3270 | 0.0038 |
| Only answer sentence | `B_only_answer_sentence` | 0.0093 | 0.0083 |
| Add distractor context | `C_k1_plus_random_distractor` | -0.0433 | 0.0064 |
| Shuffle context order | `D_shuffle_top3_order` | 0.0093 | 0.0098 |
| Oracle gold support | `oracle_gold_support` | 0.0633 | 0.0068 |

## Figure 3 Values

| label | raw_setting | accuracy |
|---|---|---:|
| Perfect document | `oracle_gold_support` | 0.8727 |
| Normal document | `baseline_k3_full_context` | 0.8093 |
| Corrupted document | `A_remove_answer_sentence` | 0.4823 |

## Oracle Handling

- Oracle gold support was found and used for the ladder figure.