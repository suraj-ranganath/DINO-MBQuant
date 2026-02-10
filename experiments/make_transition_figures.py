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

    # Frontier figure (one panel per budget if multiple budgets exist).
    budgets = list(df["budget_id"].dropna().unique()) if "budget_id" in df.columns else ["default"]
    n_panels = max(1, len(budgets))
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 4.0), squeeze=False)
    for i, budget_id in enumerate(budgets):
        ax = axes[0, i]
        sub = df[df["budget_id"] == budget_id] if "budget_id" in df.columns else df
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
    key_variants = ["fp16", "uniform_int4", "mixed_int4", "uniform_int3", "mixed_int3"]
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

    print(f"Wrote {frontier_path}")
    if (fig_root / "budget_sensitivity.pdf").exists():
        print(f"Wrote {fig_root / 'budget_sensitivity.pdf'}")
    if (fig_root / "encoder_retention_curve.pdf").exists():
        print(f"Wrote {fig_root / 'encoder_retention_curve.pdf'}")


if __name__ == "__main__":
    main()

