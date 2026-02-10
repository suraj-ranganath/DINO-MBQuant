from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from experiments.run_paths import run_scoped_file


def _bootstrap_ci(values: np.ndarray, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(low=0, high=values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired success deltas and bootstrap CI.")
    parser.add_argument("--outcomes", default="results/episode_outcomes.csv")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--budget-id", required=True)
    parser.add_argument("--variant-a", required=True, help="Improved variant (e.g., mixed_int4).")
    parser.add_argument("--variant-b", required=True, help="Reference variant (e.g., uniform_int4).")
    parser.add_argument("--out", default="results/paired_delta.json")
    args = parser.parse_args()

    outcomes_path = run_scoped_file(args.outcomes, run_name=args.run_name)
    if not outcomes_path.exists():
        raise SystemExit(f"Missing outcomes csv: {outcomes_path}")
    df = pd.read_csv(outcomes_path)
    if df.empty:
        raise SystemExit("Outcomes csv is empty.")

    df = df[df["budget_id"] == args.budget_id].copy()
    df = df[df["variant"].isin([args.variant_a, args.variant_b])].copy()
    if df.empty:
        raise SystemExit("No rows after filtering by budget and variants.")

    pivot = (
        df.pivot_table(
            index=["pair_id", "seed", "episode_id"],
            columns="variant",
            values="success",
            aggfunc="first",
        )
        .dropna()
        .reset_index()
    )
    if args.variant_a not in pivot.columns or args.variant_b not in pivot.columns:
        raise SystemExit("Paired variants are missing after pivot.")

    deltas = pivot[args.variant_a].to_numpy(dtype=float) - pivot[args.variant_b].to_numpy(dtype=float)
    delta_mean = float(deltas.mean())
    lo, hi = _bootstrap_ci(deltas, n_boot=4000, alpha=0.05, seed=0)

    out_payload: Dict[str, Any] = {
        "run_name": args.run_name,
        "budget_id": args.budget_id,
        "variant_a": args.variant_a,
        "variant_b": args.variant_b,
        "paired_units": int(deltas.size),
        "delta_mean": delta_mean,
        "delta_ci95_lo": lo,
        "delta_ci95_hi": hi,
        "variant_a_success": float(pivot[args.variant_a].mean()),
        "variant_b_success": float(pivot[args.variant_b].mean()),
    }

    out_path = run_scoped_file(args.out, run_name=args.run_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(json.dumps(out_payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

