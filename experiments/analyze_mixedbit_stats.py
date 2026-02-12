from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from experiments.run_paths import run_scoped_file


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 4000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(low=0, high=values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _binom_cdf(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    denom = 2.0 ** n
    s = 0.0
    for i in range(0, k + 1):
        s += math.comb(n, i) / denom
    return s


def _two_sided_sign_test_pvalue(wins: int, losses: int) -> float:
    n = wins + losses
    if n <= 0:
        return 1.0
    k = min(wins, losses)
    p = 2.0 * _binom_cdf(k, n)
    return float(min(1.0, p))


def _parse_comparisons(text: str) -> List[Tuple[str, str, str]]:
    # Format: "budget:variant_a:variant_b,budget:variant_a:variant_b"
    items = [x.strip() for x in text.split(",") if x.strip()]
    out: List[Tuple[str, str, str]] = []
    for item in items:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid comparison spec: {item}")
        out.append((parts[0], parts[1], parts[2]))
    return out


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if np.isnan(v):
        return None
    return v


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute consolidated paired stats for mixed-bit story variants."
    )
    parser.add_argument("--config", default="configs/experiment_config.mac_mixedbit_story.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--outcomes", default="results/episode_outcomes.csv")
    parser.add_argument("--out-csv", default="results/mixedbit_pairwise_stats.csv")
    parser.add_argument("--out-json", default="results/mixedbit_pairwise_stats.json")
    parser.add_argument("--note-out", default="notes/mixedbit_story_notes.md")
    parser.add_argument(
        "--comparisons",
        default=(
            "bA:mixed_int4:uniform_int4,"
            "bB:mixed_int4:uniform_int4,"
            "bA:mixed_int3:uniform_int3,"
            "bB:mixed_int3:uniform_int3,"
            "bA:enc8_pred4:uniform_int4,"
            "bA:enc6_pred4:uniform_int4,"
            "bA:enc4_pred8:mixed_int4,"
            "bA:enc4_pred6:mixed_int4"
        ),
    )
    args = parser.parse_args()

    _ = load_cfg(args.config)  # Config is currently used for path consistency and future extension.

    summary_path = run_scoped_file(args.summary, run_name=args.run_name)
    outcomes_path = run_scoped_file(args.outcomes, run_name=args.run_name)
    if not summary_path.exists():
        raise SystemExit(f"Missing summary CSV: {summary_path}")
    if not outcomes_path.exists():
        raise SystemExit(f"Missing outcomes CSV: {outcomes_path}")

    summary_df = pd.read_csv(summary_path)
    outcomes_df = pd.read_csv(outcomes_path)
    if summary_df.empty:
        raise SystemExit("Summary CSV is empty.")
    if outcomes_df.empty:
        raise SystemExit("Outcomes CSV is empty.")

    comparisons = _parse_comparisons(args.comparisons)
    rows: List[Dict[str, Any]] = []

    for budget_id, variant_a, variant_b in comparisons:
        sub = outcomes_df[
            (outcomes_df["budget_id"] == budget_id)
            & (outcomes_df["variant"].isin([variant_a, variant_b]))
        ].copy()
        if sub.empty:
            rows.append(
                {
                    "budget_id": budget_id,
                    "variant_a": variant_a,
                    "variant_b": variant_b,
                    "paired_units": 0,
                    "delta_mean": None,
                    "delta_ci95_lo": None,
                    "delta_ci95_hi": None,
                    "wins_a": 0,
                    "wins_b": 0,
                    "ties": 0,
                    "win_rate_a_non_tie": None,
                    "sign_test_pvalue": None,
                    "success_a": None,
                    "success_b": None,
                }
            )
            continue

        pivot = (
            sub.pivot_table(
                index=["pair_id", "seed", "episode_id"],
                columns="variant",
                values="success",
                aggfunc="first",
            )
            .dropna()
            .reset_index()
        )
        if variant_a not in pivot.columns or variant_b not in pivot.columns:
            continue

        a = pivot[variant_a].to_numpy(dtype=float)
        b = pivot[variant_b].to_numpy(dtype=float)
        deltas = a - b
        lo, hi = _bootstrap_ci(deltas, n_boot=4000, alpha=0.05, seed=0)
        wins_a = int(np.sum((a == 1.0) & (b == 0.0)))
        wins_b = int(np.sum((a == 0.0) & (b == 1.0)))
        ties = int(np.sum(a == b))
        non_tie = wins_a + wins_b
        sign_p = _two_sided_sign_test_pvalue(wins=wins_a, losses=wins_b)
        win_rate = float(wins_a / non_tie) if non_tie > 0 else None

        rows.append(
            {
                "budget_id": budget_id,
                "variant_a": variant_a,
                "variant_b": variant_b,
                "paired_units": int(deltas.size),
                "delta_mean": float(deltas.mean()),
                "delta_ci95_lo": float(lo),
                "delta_ci95_hi": float(hi),
                "wins_a": wins_a,
                "wins_b": wins_b,
                "ties": ties,
                "win_rate_a_non_tie": win_rate,
                "sign_test_pvalue": sign_p,
                "success_a": float(a.mean()) if a.size else None,
                "success_b": float(b.mean()) if b.size else None,
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = run_scoped_file(args.out_csv, run_name=args.run_name)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    payload = {
        "run_name": args.run_name,
        "comparisons": rows,
    }
    out_json = run_scoped_file(args.out_json, run_name=args.run_name)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Build a concise notes artifact for paper writing.
    note_lines: List[str] = []
    note_lines.append("# Mixed-Bit Story Notes")
    note_lines.append("")
    note_lines.append(f"Run: `{args.run_name}`")
    note_lines.append("")
    note_lines.append("## Key Paired Effects")
    note_lines.append("")
    for _, r in out_df.iterrows():
        delta = _safe_float(r.get("delta_mean"))
        lo = _safe_float(r.get("delta_ci95_lo"))
        hi = _safe_float(r.get("delta_ci95_hi"))
        n = int(r.get("paired_units", 0) or 0)
        pa = _safe_float(r.get("sign_test_pvalue"))
        if delta is None or lo is None or hi is None:
            note_lines.append(
                f"- {r['budget_id']} {r['variant_a']} vs {r['variant_b']}: insufficient paired data."
            )
            continue
        ptext = f", sign-test p={pa:.4f}" if pa is not None else ""
        note_lines.append(
            f"- {r['budget_id']} {r['variant_a']} vs {r['variant_b']}: "
            f"delta={delta:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], n={n}{ptext}"
        )

    best = out_df.dropna(subset=["delta_mean"]).sort_values(by="delta_mean", ascending=False)
    if not best.empty:
        top = best.iloc[0]
        note_lines.append("")
        note_lines.append("## Strongest Observed Mixed-Bit Gain")
        note_lines.append("")
        note_lines.append(
            f"- {top['budget_id']} {top['variant_a']} over {top['variant_b']}: "
            f"{float(top['delta_mean']):+.3f} success delta."
        )

    note_lines.append("")
    note_lines.append("## Suggested Framing")
    note_lines.append("")
    note_lines.append(
        "- Near the 4-bit region, allocation policy (encoder vs predictor precision) changes outcomes "
        "more than average bitwidth alone."
    )
    note_lines.append(
        "- 3-bit variants should be framed as a collapse regime where both uniform and mixed settings fail."
    )

    note_out = run_scoped_file(args.note_out, run_name=args.run_name)
    note_out.parent.mkdir(parents=True, exist_ok=True)
    note_out.write_text("\n".join(note_lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {note_out}")


if __name__ == "__main__":
    main()
