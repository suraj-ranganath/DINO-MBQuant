from __future__ import annotations

import argparse
import re
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from experiments.run_paths import resolve_path, run_scoped_file


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _family(name: str) -> str:
    if name == "fp16":
        return "fp16"
    if name.startswith("uniform_"):
        return "uniform"
    if name.startswith("mixed_"):
        return "mixed"
    return "other"


def _color(family: str) -> str:
    return {
        "fp16": "#4D4D4D",
        "uniform": "#D55E00",
        "mixed": "#009E73",
        "other": "#0072B2",
    }.get(family, "#0072B2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create transition-study figures.")
    parser.add_argument("--config", default="configs/experiment_config.mac_transition_study.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--grouped", default="results/summary_grouped.csv")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    labels = cfg.get("labels", {})
    grouped_path = run_scoped_file(args.grouped, run_name=args.run_name)
    if not grouped_path.exists():
        raise SystemExit(f"Missing grouped summary file: {grouped_path}")
    df = pd.read_csv(grouped_path)
    if df.empty:
        raise SystemExit("Grouped summary is empty.")

    fig_root = resolve_path(cfg, key="figures_root", run_name=args.run_name)
    fig_root.mkdir(parents=True, exist_ok=True)

    # Frontier figure (core uniform/mixed bit ladders; layerwise variants shown separately).
    ladder_variants = sorted(
        [
            str(v)
            for v in df["variant"].dropna().unique()
            if re.match(r"^(uniform|mixed)_int\d+$", str(v))
        ]
    )
    core_frontier_variants = ["fp16"] + ladder_variants
    frontier_df = df[df["variant"].isin(core_frontier_variants)].copy()
    if frontier_df.empty:
        frontier_df = df.copy()

    # Frontier figure (one panel per budget if multiple budgets exist).
    budgets = (
        list(frontier_df["budget_id"].dropna().unique())
        if "budget_id" in frontier_df.columns
        else ["default"]
    )
    n_panels = max(1, len(budgets))
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 4.0), squeeze=False)
    for i, budget_id in enumerate(budgets):
        ax = axes[0, i]
        sub = (
            frontier_df[frontier_df["budget_id"] == budget_id]
            if "budget_id" in frontier_df.columns
            else frontier_df
        )
        for _, r in sub.iterrows():
            fam = _family(str(r["variant"]))
            x = float(r["model_size_mean"])
            y = float(r["success_rate_mean"])
            y_lo = float(r.get("success_rate_ci95_lo", y))
            y_hi = float(r.get("success_rate_ci95_hi", y))
            ax.errorbar(
                [x],
                [y],
                yerr=[[max(0.0, y - y_lo)], [max(0.0, y_hi - y)]],
                fmt="o",
                color=_color(fam),
                ecolor=_color(fam),
                capsize=3,
                markersize=6,
                alpha=0.95,
            )
            label = labels.get(str(r["variant"]), str(r["variant"]))
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7)
        ax.set_xlabel("Model Size (MB)")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(f"Frontier ({budget_id})")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    frontier_path = fig_root / "transition_frontier.pdf"
    fig.savefig(frontier_path, bbox_inches="tight")
    plt.close(fig)

    # Budget sensitivity for key variants.
    key_variants = [
        v
        for v in ["fp16", "uniform_int6", "mixed_int6", "uniform_int4", "mixed_int4", "uniform_int3", "mixed_int3"]
        if v in set(df["variant"].astype(str))
    ]
    plot_df = df[df["variant"].isin(key_variants)].copy()
    if not plot_df.empty and "budget_id" in plot_df.columns:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        xcats = list(plot_df["budget_id"].dropna().unique())
        x_idx = np.arange(len(xcats))
        width = 0.14
        for j, v in enumerate(key_variants):
            sub = plot_df[plot_df["variant"] == v]
            ys = []
            errs = []
            for b in xcats:
                row = sub[sub["budget_id"] == b]
                if row.empty:
                    ys.append(np.nan)
                    errs.append(0.0)
                else:
                    y = float(row.iloc[0]["success_rate_mean"])
                    y_lo = float(row.iloc[0].get("success_rate_ci95_lo", y))
                    y_hi = float(row.iloc[0].get("success_rate_ci95_hi", y))
                    ys.append(y)
                    errs.append(max(y - y_lo, y_hi - y))
            ax.bar(
                x_idx + (j - (len(key_variants) - 1) / 2.0) * width,
                ys,
                width=width,
                yerr=errs,
                capsize=2.5,
                label=labels.get(v, v),
                color=_color(_family(v)),
                alpha=0.88 if v in {"uniform_int4", "mixed_int4"} else 0.6,
            )
        ax.set_xticks(x_idx)
        ax.set_xticklabels(xcats)
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("Success Rate")
        ax.set_xlabel("Planner Budget")
        ax.set_title("Budget sensitivity at low-bit frontier")
        ax.legend(fontsize=8, frameon=True)
        fig.tight_layout()
        budget_path = fig_root / "budget_sensitivity.pdf"
        fig.savefig(budget_path, bbox_inches="tight")
        plt.close(fig)

    # Encoder retention curve (INT4 predictor fixed), variants encfp16_{0,25,50,75,100}
    enc_rows = []
    for _, r in df.iterrows():
        m = re.match(r"encfp16_(\d+)$", str(r["variant"]))
        if not m:
            continue
        enc_rows.append(
            {
                "enc_pct": int(m.group(1)),
                "success_rate_mean": float(r["success_rate_mean"]),
                "success_rate_ci95_lo": float(r.get("success_rate_ci95_lo", r["success_rate_mean"])),
                "success_rate_ci95_hi": float(r.get("success_rate_ci95_hi", r["success_rate_mean"])),
                "budget_id": str(r.get("budget_id", "default")),
            }
        )
    if enc_rows:
        enc_df = pd.DataFrame(enc_rows).sort_values(by=["budget_id", "enc_pct"]).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for budget_id, sub in enc_df.groupby("budget_id"):
            y = sub["success_rate_mean"].to_numpy()
            y_lo = sub["success_rate_ci95_lo"].to_numpy()
            y_hi = sub["success_rate_ci95_hi"].to_numpy()
            ax.errorbar(
                sub["enc_pct"],
                y,
                yerr=[np.maximum(0.0, y - y_lo), np.maximum(0.0, y_hi - y)],
                marker="o",
                linewidth=2.0,
                capsize=3,
                label=f"{budget_id}",
            )
        ax.set_xlabel("% Encoder Kept FP16 (predictor INT4)")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0.0, 1.02)
        ax.set_title("Where bits matter inside the encoder")
        ax.legend(title="Budget", frameon=True)
        fig.tight_layout()
        enc_curve_path = fig_root / "encoder_retention_curve.pdf"
        fig.savefig(enc_curve_path, bbox_inches="tight")
        plt.close(fig)

    # Difficulty-conditioned success (computed from episode outcomes).
    outcomes_path = run_scoped_file("results/episode_outcomes.csv", run_name=args.run_name)
    if outcomes_path.exists():
        out_df = pd.read_csv(outcomes_path)
        need_cols = {"budget_id", "variant", "goal_distance_init", "success"}
        if need_cols.issubset(set(out_df.columns)):
            sub = out_df[
                (out_df["budget_id"] == "bA")
                & (out_df["variant"].isin(["fp16", "uniform_int4", "mixed_int4"]))
            ].copy()
            sub = sub.dropna(subset=["goal_distance_init"])
            if not sub.empty:
                sub["difficulty_bin"] = pd.qcut(
                    sub["goal_distance_init"], q=3, labels=["easy", "medium", "hard"]
                )
                agg = (
                    sub.groupby(["variant", "difficulty_bin"], dropna=False, as_index=False)["success"]
                    .mean()
                    .rename(columns={"success": "success_rate"})
                )
                fig, ax = plt.subplots(figsize=(6.8, 4.0))
                xcats = ["easy", "medium", "hard"]
                x = np.arange(len(xcats))
                keys = ["fp16", "uniform_int4", "mixed_int4"]
                width = 0.22
                for i, key in enumerate(keys):
                    ys = []
                    for c in xcats:
                        row = agg[(agg["variant"] == key) & (agg["difficulty_bin"] == c)]
                        ys.append(float(row.iloc[0]["success_rate"]) if not row.empty else np.nan)
                    ax.bar(
                        x + (i - 1) * width,
                        ys,
                        width=width,
                        label=labels.get(key, key),
                        color=_color(_family(key)),
                        alpha=0.9 if key != "fp16" else 0.75,
                    )
                ax.set_xticks(x)
                ax.set_xticklabels(xcats)
                ax.set_ylim(0.0, 1.02)
                ax.set_ylabel("Success Rate")
                ax.set_xlabel("Goal-Difficulty Tercile (initial goal distance)")
                ax.set_title("Difficulty-conditioned success (Budget bA, paired episodes)")
                ax.legend(frameon=True, fontsize=8)
                fig.tight_layout()
                diff_path = fig_root / "difficulty_conditioned_success.pdf"
                fig.savefig(diff_path, bbox_inches="tight")
                plt.close(fig)

    print(f"Wrote {frontier_path}")
    if (fig_root / "budget_sensitivity.pdf").exists():
        print(f"Wrote {fig_root / 'budget_sensitivity.pdf'}")
    if (fig_root / "encoder_retention_curve.pdf").exists():
        print(f"Wrote {fig_root / 'encoder_retention_curve.pdf'}")
    if (fig_root / "difficulty_conditioned_success.pdf").exists():
        print(f"Wrote {fig_root / 'difficulty_conditioned_success.pdf'}")


if __name__ == "__main__":
    main()
