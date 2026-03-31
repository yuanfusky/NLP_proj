# Significance Test: Baseline vs Best RAG

- Baseline run: `llm_only_validation_n200_seed0_20260305_191421`
- Best run: `rag_all-MiniLM-L6-v2_chunk256_k3_validation_n200_seed0_20260305_191421`
- Paired questions: `200`

## Accuracy
- Baseline: `0.4500` (95% CI `0.3800` to `0.5200`)
- Best: `0.8400` (95% CI `0.7900` to `0.8900`)
- Delta (Best - Baseline): `0.3900` (95% CI `0.3150` to `0.4650`)

## McNemar (Exact)
- b (baseline correct, best wrong): `5`
- c (baseline wrong, best correct): `83`
- Discordant total: `88`
- p-value: `2.68971e-19`