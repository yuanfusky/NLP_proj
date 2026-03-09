# Context Intervention Experiments

- A: Remove answer sentence
- B: Only answer sentence
- C: Add distractor context
- D: Shuffle context order

## Accuracy Mean ± Std (95% CI)

- A_remove_answer_sentence: 0.4823 ± 0.0006 (95% CI 0.4809 to 0.4838)
- B_only_answer_sentence: 0.8187 ± 0.0074 (95% CI 0.8004 to 0.8370)
- C_baseline_k1: 0.7750 ± 0.0036 (95% CI 0.7660 to 0.7840)
- C_k1_plus_random_distractor: 0.7660 ± 0.0052 (95% CI 0.7531 to 0.7789)
- D_shuffle_top3_order: 0.8187 ± 0.0091 (95% CI 0.7961 to 0.8412)
- baseline_k3_full_context: 0.8093 ± 0.0038 (95% CI 0.7999 to 0.8187)

## Delta Accuracy by Seed

 seed  delta_A_vs_baseline_k3  delta_B_vs_baseline_k3  delta_C_noise_vs_baseline_k1  delta_D_shuffle_vs_baseline_k3
    0                  -0.329                   0.016                        -0.010                           0.016
    1                  -0.330                   0.004                        -0.014                           0.008
    2                  -0.322                   0.008                        -0.003                           0.004