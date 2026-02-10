from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List

import pandas as pd
import yaml

from experiments.run_paths import resolve_path, run_scoped_file


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-run metrics JSON into CSV summaries.")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run folder name appended under configured output roots.",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        default=False,
        help="Allow overwriting existing summary files for the same run-name.",
    )
    parser.add_argument("--summary-out", default="results/summary.csv")
    parser.add_argument("--grouped-out", default="results/summary_grouped.csv")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    wall_root = resolve_path(cfg, key="wall_root", run_name=args.run_name)

    metrics_files = sorted(wall_root.rglob("metrics.json"))
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
        "budget_id",
        "quant_bits",
        "quant_bits_desc",
        "opt_steps",
        "seed",
        "goal_source",
        "goal_H",
        "planner_max_iter",
        "goal_file_path",
        "success_rate",
        "episode_success_count",
        "episode_successes_path",
        "avg_plan_time_seconds",
        "peak_gpu_mem_mb",
        "model_size_mb",
        "run_id",
        "source_output_dir",
        "timestamp_utc",
    ]
    keep_cols = [c for c in wanted_cols if c in df.columns]
    summary = df[keep_cols].sort_values(by=["variant", "opt_steps", "seed"]).reset_index(drop=True)

    group_keys = [k for k in ["variant", "budget_id", "opt_steps", "goal_H", "planner_max_iter"] if k in summary.columns]
    grouped = (
        summary.groupby(group_keys, dropna=False, as_index=False)
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
        .sort_values(by=group_keys)
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

    summary_path = run_scoped_file(args.summary_out, run_name=args.run_name)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_path = run_scoped_file(args.grouped_out, run_name=args.run_name)
    grouped_path.parent.mkdir(parents=True, exist_ok=True)
    if (summary_path.exists() or grouped_path.exists()) and not args.allow_existing:
        raise SystemExit(
            "Summary output already exists for this run-name.\n"
            f"- {summary_path}\n- {grouped_path}\n"
            "Choose a new --run-name, new output paths, or pass --allow-existing."
        )

    summary.to_csv(summary_path, index=False)
    grouped.to_csv(grouped_path, index=False)

    print(f"Wrote {summary_path}")
    print(f"Wrote {grouped_path}")


if __name__ == "__main__":
    main()
