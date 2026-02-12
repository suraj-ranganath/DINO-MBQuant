from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from experiments.run_paths import run_scoped_file


def _parse_budget_ids(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _compute_counts(df: pd.DataFrame, variant_a: str, variant_b: str) -> Dict[str, int]:
    a = df[df["variant"] == variant_a][["pair_id", "success"]].rename(columns={"success": "a"})
    b = df[df["variant"] == variant_b][["pair_id", "success"]].rename(columns={"success": "b"})
    m = a.merge(b, on="pair_id", how="inner")

    a_win = int(((m["a"] == 1) & (m["b"] == 0)).sum())
    b_win = int(((m["a"] == 0) & (m["b"] == 1)).sum())
    both_win = int(((m["a"] == 1) & (m["b"] == 1)).sum())
    both_fail = int(((m["a"] == 0) & (m["b"] == 0)).sum())
    overlap = int(len(m))

    return {
        "paired_episodes": overlap,
        "a_wins": a_win,
        "b_wins": b_win,
        "both_win": both_win,
        "both_fail": both_fail,
        "a_minus_b_winrate": float((a_win - b_win) / overlap) if overlap > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Episode-level paired win/loss contingency for two variants.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--outcomes", default="results/episode_outcomes.csv")
    parser.add_argument("--budget-ids", required=True, help="Comma-separated budget ids.")
    parser.add_argument("--variant-a", required=True)
    parser.add_argument("--variant-b", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    outcomes_path = run_scoped_file(args.outcomes, run_name=args.run_name)
    out_path = run_scoped_file(args.out, run_name=args.run_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(outcomes_path)
    budget_ids = _parse_budget_ids(args.budget_ids)

    by_budget: Dict[str, Dict[str, int]] = {}
    for budget_id in budget_ids:
        sub = df[df["budget_id"] == budget_id].copy()
        by_budget[budget_id] = _compute_counts(sub, variant_a=args.variant_a, variant_b=args.variant_b)

    pooled = _compute_counts(df[df["budget_id"].isin(budget_ids)].copy(), variant_a=args.variant_a, variant_b=args.variant_b)

    payload = {
        "run_name": args.run_name,
        "variant_a": args.variant_a,
        "variant_b": args.variant_b,
        "budget_ids": budget_ids,
        "by_budget": by_budget,
        "pooled": pooled,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
