from __future__ import annotations

import argparse
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.ticker import MaxNLocator

from experiments.run_paths import resolve_path, run_scoped_file


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
        }
    )


def _variant_palette(variants: list[str]) -> Dict[str, str]:
    # Colorblind-safe base colors for canonical variants.
    base = {
        "fp16": "#0072B2",
        "uniform_int8": "#D55E00",
        "mixed_int8": "#009E73",
        "uniform_int4": "#CC79A7",
        "mixed_int4": "#56B4E9",
        "uniform_int3": "#E69F00",
        "mixed_int3": "#332288",
    }
    palette = dict(base)
    extras = [v for v in variants if v not in palette]
    if extras:
        cmap = plt.get_cmap("tab20")
        for i, name in enumerate(extras):
            r, g, b, _ = cmap(i % cmap.N)
            palette[name] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return palette


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from aggregated results.")
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
        help="Allow overwriting existing figure files for the same run-name.",
    )
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--grouped", default="results/summary_grouped.csv")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    labels = cfg.get("labels", {})
    _style()

    summary_path = run_scoped_file(args.summary, run_name=args.run_name)
    if not summary_path.exists():
        raise SystemExit(f"Missing summary file: {summary_path}")
    grouped_path = run_scoped_file(args.grouped, run_name=args.run_name)
    if not grouped_path.exists():
        raise SystemExit(f"Missing grouped summary file: {grouped_path}")

    df = pd.read_csv(summary_path)
    if df.empty:
        raise SystemExit("Summary CSV is empty.")
    grouped = pd.read_csv(grouped_path)
    if grouped.empty:
        raise SystemExit("Grouped summary CSV is empty.")
    palette = _variant_palette(list(grouped["variant"].unique()))

    figures_root = resolve_path(cfg, key="figures_root", run_name=args.run_name)
    if figures_root.exists() and not args.allow_existing:
        preexisting = [
            figures_root / "success_vs_opt_steps.pdf",
            figures_root / "efficiency_tradeoff.pdf",
            figures_root / "efficiency_table.pdf",
        ]
        if any(p.exists() for p in preexisting):
            raise SystemExit(
                f"Figure outputs already exist in {figures_root}\n"
                "Choose a new --run-name or pass --allow-existing."
            )
    figures_root.mkdir(parents=True, exist_ok=True)

    grouped = grouped.sort_values(by=["variant", "opt_steps"]).copy()
    if "success_rate_ci95_lo" not in grouped.columns or "success_rate_ci95_hi" not in grouped.columns:
        grouped["success_rate_ci95_lo"] = grouped["success_rate_mean"]
        grouped["success_rate_ci95_hi"] = grouped["success_rate_mean"]

    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    for variant in grouped["variant"].unique():
        sub = grouped[grouped["variant"] == variant]
        label = labels.get(variant, variant)
        color = palette.get(variant, None)
        ax.errorbar(
            sub["opt_steps"],
            sub["success_rate_mean"],
            yerr=[
                (sub["success_rate_mean"] - sub["success_rate_ci95_lo"]).fillna(0.0),
                (sub["success_rate_ci95_hi"] - sub["success_rate_mean"]).fillna(0.0),
            ],
            marker="o",
            linewidth=2.2,
            markersize=5,
            capsize=4,
            label=label,
            color=color,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['success_rate_mean']:.2f}",
                (row["opt_steps"], row["success_rate_mean"]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color=color if color else "black",
            )

    ax.set_xlabel("Planning optimization steps")
    ax.set_ylabel("Planning success rate (mean, 95% CI)")
    ax.set_ylim(0.0, 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Wall task planning success")
    ax.legend(frameon=True, edgecolor="#BBBBBB")
    fig.tight_layout()

    fig1_path = figures_root / "success_vs_opt_steps.pdf"
    fig.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig)

    eff = (
        df.groupby("variant", as_index=False)
        .agg(
            model_size_mb=("model_size_mb", "mean"),
            avg_plan_time_seconds=("avg_plan_time_seconds", "mean"),
            success_rate=("success_rate", "mean"),
            peak_gpu_mem_mb=("peak_gpu_mem_mb", "mean"),
        )
        .sort_values(by="variant")
    )
    eff["variant_label"] = eff["variant"].map(lambda x: labels.get(x, x))

    # Tradeoff plot: model size vs planning latency (bubble = success)
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    for _, row in eff.iterrows():
        variant = row["variant"]
        color = palette.get(variant, "#333333")
        size = 180 + 260 * float(row["success_rate"])
        ax.scatter(
            row["model_size_mb"],
            row["avg_plan_time_seconds"],
            s=size,
            color=color,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.8,
        )
        ax.annotate(
            row["variant_label"],
            (row["model_size_mb"], row["avg_plan_time_seconds"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8.5,
        )
    ax.set_xlabel("Model size (MB)")
    ax.set_ylabel("Average planning time per episode (s)")
    ax.set_title("Efficiency tradeoff (Wall, Mac MPS)")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    tradeoff_path = figures_root / "efficiency_tradeoff.pdf"
    fig.savefig(tradeoff_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 2.6))
    ax.axis("off")
    table_df = eff[["variant_label", "model_size_mb", "avg_plan_time_seconds", "peak_gpu_mem_mb"]].copy()
    table_df.columns = ["Variant", "Model Size (MB)", "Avg Plan Time (s)", "Peak GPU Mem (MB)"]
    for col in ["Model Size (MB)", "Avg Plan Time (s)", "Peak GPU Mem (MB)"]:
        table_df[col] = table_df[col].map(lambda x: f"{x:.2f}")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)

    # Header + zebra stripes for readability.
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E8EEF6")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F8FAFC")
    fig.tight_layout()

    table_path = figures_root / "efficiency_table.pdf"
    fig.savefig(table_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {fig1_path}")
    print(f"Wrote {tradeoff_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
