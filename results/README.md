# Results

Each file is the raw output of one run. File names carry the configuration.

| File | Script | Backs |
|---|---|---|
| `frequency.csv` | `frequency.py` (defaults) | Fig. 1, rollout exposure. A fresh run reproduces it byte-for-byte. |
| `quality_qwen2.5-0.5b-instruct_fp32_kv_to_bf16_server.json` | `kv_quality.py --mode dtype --model Qwen/Qwen2.5-0.5B-Instruct` | Table I, row 1 (0.925 → 0.942, 0 breaks / 2 fixes) |
| `quality_qwen2.5-0.5b_kv_to_qwen2.5-0.5b-instruct_server.json` | `kv_quality.py --mode checkpoint --model-a Qwen/Qwen2.5-0.5B --model-b Qwen/Qwen2.5-0.5B-Instruct` | Table I, row 2 (0.942 → 0.925, 4 breaks / 2 fixes, p = 0.34) |
| `quality_qwen2-0.5b_kv_to_qwen2.5-0.5b-instruct_server.json` | `kv_quality.py --mode checkpoint --model-a Qwen/Qwen2-0.5B --model-b Qwen/Qwen2.5-0.5B-Instruct` | Table I, row 3 (0.942 → 0.033, 109 breaks / 0 fixes) |
| `interp_sweep.json` | `interp_sweep.py --limit 60` | Table II, Fig. 2–3 |
| `distance_sweep_pythia-160m.json` | `distance_sweep.py --model EleutherAI/pythia-160m` | Section V-E, replication attempt 1 |
| `family_matrix.json` | `family_matrix.py` | Section V-E, replication attempt 2 |
| `severity_qwen2.5-0.5b_kv_to_qwen2.5-0.5b-instruct_server.json` | `kv_severity.py --mode checkpoint` | Early token-divergence probe (mean agreement 0.69); not in the paper's tables |

## superseded/

Runs of the earlier quality probes, kept because the README's measurement
history refers to them. Do not cite these numbers.

- `quality_v1_qwen2.5_base_to_instruct.json`: 15 tasks, control at ceiling.
- `quality_v2_dtype_fp32_to_bf16.json`: 120 tasks with the v2 scorer
  (accent and truncation bugs; this pair happens to be unaffected).
