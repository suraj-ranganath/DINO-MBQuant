# Metrics JSON Schema

Per-run metrics file path:
`results/wall/<variant>/opt_steps_<X>/seed_<S>/metrics.json`

Required fields:
- `variant` (string)
- `opt_steps` (int)
- `seed` (int)
- `n_evals` (int)
- `success_count` (int)
- `success_rate` (float)
- `avg_plan_time_seconds` (float)
- `peak_gpu_mem_mb` (float)
- `model_size_mb` (float)
- `run_id` (string)
- `timestamp_utc` (ISO-8601 UTC)
- `source_output_dir` (string path)
- `logs_path` (string path)
- `quant_backend_effective` (string)
