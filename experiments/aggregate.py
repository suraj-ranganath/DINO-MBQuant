from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-run metrics JSON into CSV summaries.")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--summary-out", default="results/summary.csv")
    parser.add_argument("--grouped-out", default="results/summary_grouped.csv")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    wall_root = Path(cfg["paths"]["wall_root"]).resolve()

    metrics_files = sorted(wall_root.glob("*/opt_steps_*/seed_*/metrics.json"))
    if not metrics_files:
        raise SystemExit(f"No metrics found under {wall_root}")

    rows: List[Dict[str, Any]] = []
    for path in metrics_files:
        rows.append(pd.read_json(path, typ="series").to_dict())

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No rows parsed from metrics JSON files.")

    wanted_cols = [
        "variant",
        "quant_bits",
        "quant_bits_desc",
        "opt_steps",
        "seed",
        "success_rate",
        "avg_plan_time_seconds",
        "peak_gpu_mem_mb",
        "model_size_mb",
        "run_id",
        "source_output_dir",
        "timestamp_utc",
    ]
    keep_cols = [c for c in wanted_cols if c in df.columns]
    summary = df[keep_cols].sort_values(by=["variant", "opt_steps", "seed"]).reset_index(drop=True)

    grouped = (
        summary.groupby(["variant", "opt_steps"], as_index=False)
        .agg(
            success_rate_mean=("success_rate", "mean"),
            success_rate_std=("success_rate", "std"),
            avg_time_mean=("avg_plan_time_seconds", "mean"),
            avg_time_std=("avg_plan_time_seconds", "std"),
            peak_mem_mean=("peak_gpu_mem_mb", "mean"),
            peak_mem_std=("peak_gpu_mem_mb", "std"),
            model_size_mean=("model_size_mb", "mean"),
            model_size_std=("model_size_mb", "std"),
            quant_bits_mean=("quant_bits", "mean"),
            quant_bits_desc=("quant_bits_desc", "first"),
            num_runs=("seed", "count"),
        )
        .sort_values(by=["variant", "opt_steps"])
        .reset_index(drop=True)
    )

    # 95% CI using normal approximation over run-level means.
    # For n==1 we keep CI equal to the point estimate.
    def _ci_bounds(row: pd.Series) -> tuple[float, float]:
        mu = float(row["success_rate_mean"])
        std = row["success_rate_std"]
        n = int(row["num_runs"])
        if n <= 1 or pd.isna(std):
            return (mu, mu)
        half = 1.96 * float(std) / math.sqrt(n)
        lo = max(0.0, mu - half)
        hi = min(1.0, mu + half)
        return (lo, hi)

    ci = grouped.apply(_ci_bounds, axis=1)
    grouped["success_rate_ci95_lo"] = [x[0] for x in ci]
    grouped["success_rate_ci95_hi"] = [x[1] for x in ci]

    summary_path = Path(args.summary_out).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_path = Path(args.grouped_out).resolve()
    grouped_path.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(summary_path, index=False)
    grouped.to_csv(grouped_path, index=False)

    print(f"Wrote {summary_path}")
    print(f"Wrote {grouped_path}")


if __name__ == "__main__":
    main()
