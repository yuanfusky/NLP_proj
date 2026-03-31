## Case 1: baseline correct -> remove-answer wrong

- Category: Case 1
- QID: `validation_252`
- Question: The distribution of thermal speeds depends strongly on temperature. as temperature increases, the speeds are shifted to higher values and the distribution is what?
- Answer options: A) decreased | B) improved | C) broadened | D) removed
- Gold answer: C
- Baseline prediction (baseline_k3_full_context): C
- Intervention prediction (A_remove_answer_sentence): B
- Baseline context preview: the distribution of thermal speeds depends strongly on temperature. as temperature increases, the speeds are shifted... || the speed of a gas particle is directly proportional to the temperature of the system.
- Intervention context preview: the distribution of thermal speeds depends strongly on temperature. || the speed of a gas particle is directly proportional to the temperature of the system.
- Interpretation: Under baseline_k3_full_context, the model predicts C, which matches the gold answer C. After removing the answer-bearing sentence, the prediction changes to B, suggesting that the removed sentence carried the decisive evidence.

## Case 2: baseline correct -> only-answer still correct

- Category: Case 2
- QID: `validation_486`
- Question: What type of organisms carry out their life processes through division of labor and have specialized cells that do specific job?
- Answer options: A) CompoundCellular | B) multicellular | C) dermal | D) biomolecular
- Gold answer: B
- Baseline prediction (baseline_k3_full_context): B
- Intervention prediction (B_only_answer_sentence): B
- Baseline context preview: multicellular organisms carry out their life processes through division of labor. they have specialized cells that do... || the activity of an organism depends on the total activity of independent cells,.
- Intervention context preview: multicellular organisms carry out their life processes through division of labor.
- Interpretation: The model predicts B in both baseline_k3_full_context and B_only_answer_sentence. This indicates that the answer-bearing sentence alone is enough to preserve the correct decision on this example.

## Case 3: baseline correct -> distractor-added wrong

- Category: Case 3
- QID: `validation_276`
- Question: What kind of mixture consists of two or more phases, exemplified when a combination of oil and water forms layers?
- Answer options: A) simple mixture | B) heterogeneous | C) homogeneous | D) complex miture
- Gold answer: B
- Baseline prediction (C_baseline_k1): B
- Intervention prediction (C_k1_plus_random_distractor): C
- Baseline context preview: a phase is any part of a sample that has a uniform composition and properties. by definition, a pure substance or a homogeneous mixture consists of a single phase. a heterogeneous mixture consists of two or more phase...
- Intervention context preview: a phase is any part of a sample that has a uniform composition and properties. by definition, a pure substance or a h... || Global air currents affect precipitation. How they affect it varies with latitude ( Figure below...
- baseline_k3_full_context status: correct
- Interpretation: The model is correct with C_baseline_k1 but changes to C after adding a random distractor paragraph. This example illustrates that irrelevant extra context can interfere with an otherwise sufficient evidence signal. The corresponding baseline_k3_full_context row is correct.
