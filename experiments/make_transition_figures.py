from __future__ import annotations

import argparse
import re
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

from experiments.plot_theme import (
    FAMILY_BASE_COLORS,
    apply_paper_theme,
    variant_color,
    variant_marker,
)
from experiments.run_paths import resolve_path, run_scoped_file


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pareto_mask_size_success(df: pd.DataFrame, *, size_col: str, success_col: str) -> pd.Series:
    """Return True for non-dominated points (min size, max success)."""
    mask = []
    sizes = df[size_col].to_numpy(dtype=float)
    succs = df[success_col].to_numpy(dtype=float)
    for i in range(len(df)):
        dominated = False
        for j in range(len(df)):
            if i == j:
                continue
            no_worse = sizes[j] <= sizes[i] and succs[j] >= succs[i]
            strictly_better = sizes[j] < sizes[i] or succs[j] > succs[i]
            if no_worse and strictly_better:
                dominated = True
                break
        mask.append(not dominated)
    return pd.Series(mask, index=df.index)
def main() -> None:
    parser = argparse.ArgumentParser(description="Create transition-study figures.")
    parser.add_argument("--config", default="configs/experiment_config.mac_transition_study.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--grouped", default="results/summary_grouped.csv")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    apply_paper_theme()
    labels = cfg.get("labels", {})
    grouped_path = run_scoped_file(args.grouped, run_name=args.run_name)
    if not grouped_path.exists():
        raise SystemExit(f"Missing grouped summary file: {grouped_path}")
    df = pd.read_csv(grouped_path)
    if df.empty:
        raise SystemExit("Grouped summary is empty.")

    fig_root = resolve_path(cfg, key="figures_root", run_name=args.run_name)
    fig_root.mkdir(parents=True, exist_ok=True)

    # Figure 1 frontier uses broad variant coverage for stronger evidence:
    # fp16 + uniform/mixed ladders + asymmetric encoder/predictor allocations.
    frontier_df = df.copy()
    frontier_df = frontier_df[
        ~frontier_df["variant"].astype(str).str.match(r"^encfp16_\d+$", na=False)
    ].copy()
    if frontier_df.empty:
        frontier_df = df.copy()

    # Frontier figure (one panel per budget if multiple budgets exist).
    budgets = (
        list(frontier_df["budget_id"].dropna().unique())
        if "budget_id" in frontier_df.columns
        else ["default"]
    )
    n_panels = max(1, len(budgets))
    fig, axes = plt.subplots(1, n_panels, figsize=(6.6 * n_panels, 4.35), squeeze=False)
    for i, budget_id in enumerate(budgets):
        ax = axes[0, i]
        sub = (
            frontier_df[frontier_df["budget_id"] == budget_id]
            if "budget_id" in frontier_df.columns
            else frontier_df
        )
        if sub.empty:
            continue

        # Plot all data points with uncertainty bars.
        for _, r in sub.iterrows():
            variant = str(r["variant"])
            x = float(r["model_size_mean"])
            y = float(r["success_rate_mean"])
            y_lo = float(r.get("success_rate_ci95_lo", y))
            y_hi = float(r.get("success_rate_ci95_hi", y))
            ax.errorbar(
                [x],
                [y],
                yerr=[[max(0.0, y - y_lo)], [max(0.0, y_hi - y)]],
                fmt=variant_marker(variant),
                color=variant_color(variant),
                ecolor=variant_color(variant),
                capsize=3,
                markersize=8 if variant == "fp16" else 6.5,
                markeredgecolor="white",
                markeredgewidth=0.7,
                zorder=3,
            )

        # Compute Pareto-optimal points and overlay frontier with stars.
        pareto_mask = _pareto_mask_size_success(
            sub,
            size_col="model_size_mean",
            success_col="success_rate_mean",
        )
        psub = sub[pareto_mask].sort_values(by="model_size_mean").copy()
        ax.plot(
            psub["model_size_mean"],
            psub["success_rate_mean"],
            color="#111111",
            linestyle="--",
            linewidth=1.8,
            zorder=4,
        )
        ax.scatter(
            psub["model_size_mean"],
            psub["success_rate_mean"],
            marker="*",
            s=190,
            color="#F2C94C",
            edgecolors="#111111",
            linewidths=0.8,
            zorder=5,
            label="Pareto frontier",
        )

        # Annotate only Pareto labels to avoid clutter.
        for _, r in psub.iterrows():
            v = str(r["variant"])
            ax.annotate(
                labels.get(v, v),
                (float(r["model_size_mean"]), float(r["success_rate_mean"])),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#DDDDDD", alpha=0.8),
            )
        ax.set_xlabel("Model Size (MB)")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(f"Pareto Frontier: Success vs Model Size ({budget_id})")
        ax.grid(alpha=0.2, linestyle="-")
        legend_handles = [
            Line2D([0], [0], marker="o", color=FAMILY_BASE_COLORS["uniform"], linestyle="None", markersize=6, label="Uniform"),
            Line2D([0], [0], marker="s", color=FAMILY_BASE_COLORS["mixed"], linestyle="None", markersize=6, label="Mixed (enc FP16)"),
            Line2D([0], [0], marker="v", color=FAMILY_BASE_COLORS["asymmetric"], linestyle="None", markersize=6, label="Asymmetric"),
            Line2D([0], [0], marker="D", color=FAMILY_BASE_COLORS["fp16"], linestyle="None", markersize=7, label="FP16"),
            Line2D([0], [0], marker="*", color="#F2C94C", markeredgecolor="#111111", linestyle="--", markersize=11, label="Pareto frontier"),
        ]
        ax.legend(handles=legend_handles, loc="lower right")
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
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        xcats = list(plot_df["budget_id"].dropna().unique())
        x_idx = np.arange(len(xcats))
        for v in key_variants:
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
            ax.errorbar(
                x_idx,
                ys,
                yerr=errs,
                capsize=3,
                label=labels.get(v, v),
                color=variant_color(v),
                marker=variant_marker(v),
                markersize=6.5,
                linewidth=2.1 if v in {"uniform_int4", "mixed_int4"} else 1.6,
                alpha=0.95 if v in {"uniform_int4", "mixed_int4"} else 0.8,
            )
        ax.set_xticks(x_idx)
        ax.set_xticklabels(xcats)
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("Success Rate")
        ax.set_xlabel("Planner Budget")
        ax.set_title("Budget Sensitivity at the Low-Bit Frontier")
        ax.legend(fontsize=8, ncol=2, frameon=True)
        ax.grid(alpha=0.2)
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
        fig, ax = plt.subplots(figsize=(6.6, 4.1))
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
                color=variant_color("encfp16_50"),
                markerfacecolor="white",
            )
        ax.set_xlabel("% Encoder Kept FP16 (predictor INT4)")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0.0, 1.02)
        ax.set_title("Encoder-Retention Curve (Predictor INT4)")
        ax.legend(title="Budget", frameon=True)
        ax.grid(alpha=0.2)
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
                    sub.groupby(["variant", "difficulty_bin"], dropna=False, observed=False, as_index=False)["success"]
                    .mean()
                    .rename(columns={"success": "success_rate"})
                )
                fig, ax = plt.subplots(figsize=(6.8, 4.1))
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
                        color=variant_color(key),
                        alpha=0.9 if key != "fp16" else 0.75,
                        edgecolor="white",
                        linewidth=0.7,
                    )
                ax.set_xticks(x)
                ax.set_xticklabels(xcats)
                ax.set_ylim(0.0, 1.02)
                ax.set_ylabel("Success Rate")
                ax.set_xlabel("Goal-Difficulty Tercile (initial goal distance)")
                ax.set_title("Difficulty-Conditioned Success (Budget bA)")
                ax.legend(frameon=True, fontsize=8)
                ax.grid(alpha=0.2, axis="y")
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
