from __future__ import annotations

import argparse
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from experiments.run_paths import resolve_path, run_scoped_file


def load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "font.family": "DejaVu Sans",
        }
    )


def variant_family(v: str) -> str:
    if v == "fp16":
        return "fp16"
    if v.startswith("uniform_"):
        return "uniform"
    if v.startswith("mixed_int"):
        return "mixed_preserve_encoder_fp16"
    if v.startswith("mixed_e"):
        return "mixed_two_quantized_modules"
    return "other"


def family_color(fam: str) -> str:
    colors = {
        "fp16": "#4D4D4D",
        "uniform": "#D55E00",
        "mixed_preserve_encoder_fp16": "#009E73",
        "mixed_two_quantized_modules": "#0072B2",
        "other": "#999999",
    }
    return colors.get(fam, "#999999")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-ready bitsweep figures.")
    parser.add_argument("--config", default="configs/experiment_config.mac_bitsweep_h9_mixed.yaml")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run folder name appended under configured output roots.",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        default=False,
        help="Allow overwriting existing figure files for the same run-name.",
    )
    parser.add_argument("--summary", default="results/summary_bitsweep_h9_mixed.csv")
    parser.add_argument("--grouped", default="results/summary_bitsweep_h9_mixed_grouped.csv")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    labels = cfg.get("labels", {})
    style()

    summary_path = run_scoped_file(args.summary, run_name=args.run_name)
    grouped_path = run_scoped_file(args.grouped, run_name=args.run_name)
    if not summary_path.exists() or not grouped_path.exists():
        raise SystemExit("Missing summary/grouped CSV for bitsweep figures.")

    summary = pd.read_csv(summary_path)
    grouped = pd.read_csv(grouped_path)
    if summary.empty or grouped.empty:
        raise SystemExit("Bitsweep summary/grouped CSV is empty.")

    grouped = grouped.copy()
    grouped["family"] = grouped["variant"].map(variant_family)
    grouped["label"] = grouped["variant"].map(lambda x: labels.get(x, x))
    grouped["success_err"] = grouped["success_rate_std"].fillna(0.0)
    grouped["size"] = grouped["model_size_mean"]
    grouped = grouped.sort_values(["success_rate_mean", "size"], ascending=[False, True]).reset_index(drop=True)

    fig_root = resolve_path(cfg, key="figures_root", run_name=args.run_name)
    if fig_root.exists() and not args.allow_existing:
        preexisting = [
            fig_root / "bitsweep_success_bars.pdf",
            fig_root / "bitsweep_frontier.pdf",
            fig_root / "bitsweep_efficiency_table.pdf",
        ]
        if any(p.exists() for p in preexisting):
            raise SystemExit(
                f"Figure outputs already exist in {fig_root}\n"
                "Choose a new --run-name or pass --allow-existing."
            )
    fig_root.mkdir(parents=True, exist_ok=True)

    # Figure 1: success bars with family color coding.
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    x = np.arange(len(grouped))
    bar_colors = [family_color(f) for f in grouped["family"]]
    ax.bar(
        x,
        grouped["success_rate_mean"],
        yerr=grouped["success_err"],
        capsize=3,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Success Rate")
    ax.set_xlabel("Variant")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["label"], rotation=32, ha="right")
    ax.set_title("Wall planning success under low-bit quantization (mean ± std over seeds)")
    ax.grid(axis="y", alpha=0.25)

    # Draw FP16 reference line.
    fp16_rows = grouped[grouped["variant"] == "fp16"]
    if not fp16_rows.empty:
        fp16_sr = float(fp16_rows.iloc[0]["success_rate_mean"])
        ax.axhline(fp16_sr, color="#4D4D4D", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(
            len(grouped) - 0.6,
            min(0.98, fp16_sr + 0.02),
            f"FP16={fp16_sr:.2f}",
            fontsize=8,
            color="#4D4D4D",
            ha="right",
        )

    # Legend proxies.
    from matplotlib.patches import Patch

    legend_items = [
        Patch(facecolor=family_color("uniform"), label="Uniform"),
        Patch(facecolor=family_color("mixed_preserve_encoder_fp16"), label="Mixed (encoder FP16)"),
        Patch(facecolor=family_color("mixed_two_quantized_modules"), label="Mixed (encoder/predictor bits differ)"),
        Patch(facecolor=family_color("fp16"), label="FP16"),
    ]
    ax.legend(handles=legend_items, loc="upper right", frameon=True)
    fig.tight_layout()
    fig1 = fig_root / "bitsweep_success_bars.pdf"
    fig.savefig(fig1, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: model size vs success frontier.
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    for _, r in grouped.iterrows():
        fam = r["family"]
        color = family_color(fam)
        ax.scatter(
            r["model_size_mean"],
            r["success_rate_mean"],
            s=80,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.95,
        )
        ax.annotate(
            r["label"],
            (r["model_size_mean"], r["success_rate_mean"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7.5,
        )
    ax.set_xlabel("Model Size (MB, effective)")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Accuracy-memory frontier on Wall (Mac, bounded planning)")
    ax.grid(alpha=0.28)
    fig.tight_layout()
    fig2 = fig_root / "bitsweep_frontier.pdf"
    fig.savefig(fig2, bbox_inches="tight")
    plt.close(fig)

    # Table figure for compact camera-ready rendering.
    table_df = grouped[
        [
            "label",
            "success_rate_mean",
            "avg_time_mean",
            "model_size_mean",
            "quant_bits_desc",
        ]
    ].copy()
    table_df.columns = ["Variant", "Success", "Avg Time (s)", "Model Size (MB)", "Quant Spec"]
    table_df["Success"] = table_df["Success"].map(lambda x: f"{x:.3f}")
    table_df["Avg Time (s)"] = table_df["Avg Time (s)"].map(lambda x: f"{x:.2f}")
    table_df["Model Size (MB)"] = table_df["Model Size (MB)"].map(lambda x: f"{x:.2f}")

    fig, ax = plt.subplots(figsize=(8.2, 3.1))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.2)
    tbl.scale(1.0, 1.35)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E8EEF6")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F8FAFC")
    fig.tight_layout()
    fig3 = fig_root / "bitsweep_efficiency_table.pdf"
    fig.savefig(fig3, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {fig1}")
    print(f"Wrote {fig2}")
    print(f"Wrote {fig3}")


if __name__ == "__main__":
    main()
