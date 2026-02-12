from __future__ import annotations

import argparse
import re
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from experiments.run_paths import resolve_path, run_scoped_file


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _variant_bits(name: str) -> int | None:
    m = re.search(r"int(\d+)$", name)
    if not m:
        return None
    return int(m.group(1))


def _parse_asymmetric(name: str) -> tuple[int, int] | None:
    m = re.match(r"enc(\d+)_pred(\d+)$", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create appendix figures for mixed-bit study.")
    parser.add_argument("--config", default="configs/experiment_config.mac_mixedbit_story.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--grouped", default="results/summary_grouped.csv")
    parser.add_argument("--stats", default="results/mixedbit_pairwise_stats.csv")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    labels = cfg.get("labels", {})
    grouped_path = run_scoped_file(args.grouped, run_name=args.run_name)
    if not grouped_path.exists():
        raise SystemExit(f"Missing grouped summary: {grouped_path}")
    grouped = pd.read_csv(grouped_path)
    if grouped.empty:
        raise SystemExit("Grouped summary is empty.")

    fig_root = resolve_path(cfg, key="figures_root", run_name=args.run_name)
    fig_root.mkdir(parents=True, exist_ok=True)

    # Figure A1: Bit ladder (uniform vs mixed encoder-preserving).
    ladder_rows: List[Dict[str, Any]] = []
    for _, r in grouped.iterrows():
        variant = str(r["variant"])
        budget = str(r.get("budget_id", "default"))
        bits = _variant_bits(variant)
        if bits is None:
            continue
        family = None
        if variant.startswith("uniform_int"):
            family = "uniform"
        elif variant.startswith("mixed_int"):
            family = "mixed"
        if family is None:
            continue
        ladder_rows.append(
            {
                "budget_id": budget,
                "variant": variant,
                "family": family,
                "bits": bits,
                "success": float(r["success_rate_mean"]),
                "ci_lo": float(r.get("success_rate_ci95_lo", r["success_rate_mean"])),
                "ci_hi": float(r.get("success_rate_ci95_hi", r["success_rate_mean"])),
            }
        )

    if ladder_rows:
        ladder = pd.DataFrame(ladder_rows)
        budgets = list(ladder["budget_id"].dropna().unique())
        fig, axes = plt.subplots(
            1, max(1, len(budgets)), figsize=(6.2 * max(1, len(budgets)), 4.0), squeeze=False
        )
        colors = {"uniform": "#D55E00", "mixed": "#009E73"}
        for i, budget in enumerate(budgets):
            ax = axes[0, i]
            sub = ladder[ladder["budget_id"] == budget]
            for fam in ["uniform", "mixed"]:
                fam_sub = sub[sub["family"] == fam].sort_values(by="bits", ascending=False)
                if fam_sub.empty:
                    continue
                x = fam_sub["bits"].to_numpy()
                y = fam_sub["success"].to_numpy()
                lo = fam_sub["ci_lo"].to_numpy()
                hi = fam_sub["ci_hi"].to_numpy()
                ax.errorbar(
                    x,
                    y,
                    yerr=[np.maximum(0.0, y - lo), np.maximum(0.0, hi - y)],
                    marker="o",
                    linewidth=2.0,
                    capsize=3,
                    color=colors[fam],
                    label=fam.capitalize(),
                )
            ax.set_xticks([8, 6, 4, 3])
            ax.set_ylim(0.0, 1.02)
            ax.set_xlabel("Bitwidth")
            ax.set_ylabel("Success Rate")
            ax.set_title(f"Bit ladder ({budget})")
            ax.grid(alpha=0.25)
            ax.legend(frameon=True)
        fig.tight_layout()
        out = fig_root / "appendix_bit_ladder.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")

    # Figure A2: Asymmetric allocation map.
    asym_rows: List[Dict[str, Any]] = []
    for _, r in grouped.iterrows():
        variant = str(r["variant"])
        parsed = _parse_asymmetric(variant)
        if parsed is None:
            continue
        enc_bits, pred_bits = parsed
        asym_rows.append(
            {
                "budget_id": str(r.get("budget_id", "default")),
                "variant": variant,
                "enc_bits": enc_bits,
                "pred_bits": pred_bits,
                "success": float(r["success_rate_mean"]),
                "size_mb": float(r["model_size_mean"]),
            }
        )
    if asym_rows:
        asym = pd.DataFrame(asym_rows)
        focus_budget = "bA" if "bA" in set(asym["budget_id"]) else str(asym.iloc[0]["budget_id"])
        sub = asym[asym["budget_id"] == focus_budget].copy()
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        sc = ax.scatter(
            sub["enc_bits"],
            sub["pred_bits"],
            c=sub["success"],
            s=np.maximum(80, sub["size_mb"] * 1.5),
            cmap="viridis",
            alpha=0.9,
            edgecolors="white",
            linewidths=0.8,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                labels.get(str(r["variant"]), str(r["variant"])),
                (r["enc_bits"], r["pred_bits"]),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=8,
            )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Success Rate")
        ax.set_xlabel("Encoder bits")
        ax.set_ylabel("Predictor bits")
        ax.set_title(f"Asymmetric bit allocation map ({focus_budget})")
        ax.set_xticks(sorted(sub["enc_bits"].unique()))
        ax.set_yticks(sorted(sub["pred_bits"].unique()))
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out = fig_root / "appendix_asymmetric_allocation_map.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")

    # Figure A3: Pairwise delta forest from consolidated stats.
    stats_path = run_scoped_file(args.stats, run_name=args.run_name)
    if stats_path.exists():
        stats_df = pd.read_csv(stats_path)
        stats_df = stats_df.dropna(subset=["delta_mean", "delta_ci95_lo", "delta_ci95_hi"])
        if not stats_df.empty:
            stats_df = stats_df.copy()
            stats_df["label"] = stats_df.apply(
                lambda r: f"{r['budget_id']} {r['variant_a']} - {r['variant_b']}", axis=1
            )
            stats_df = stats_df.sort_values(by="delta_mean", ascending=True).reset_index(drop=True)
            y = np.arange(len(stats_df))
            fig, ax = plt.subplots(figsize=(8.2, max(3.0, 0.45 * len(stats_df) + 1.5)))
            means = stats_df["delta_mean"].to_numpy(dtype=float)
            lo = stats_df["delta_ci95_lo"].to_numpy(dtype=float)
            hi = stats_df["delta_ci95_hi"].to_numpy(dtype=float)
            ax.errorbar(
                means,
                y,
                xerr=[np.maximum(0.0, means - lo), np.maximum(0.0, hi - means)],
                fmt="o",
                capsize=3,
                color="#0072B2",
            )
            ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
            ax.set_yticks(y)
            ax.set_yticklabels(stats_df["label"])
            ax.set_xlabel("Paired success delta (variant_a - variant_b)")
            ax.set_title("Paired mixed-bit effects with 95% bootstrap CI")
            ax.grid(alpha=0.25, axis="x")
            fig.tight_layout()
            out = fig_root / "appendix_pairwise_deltas.pdf"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {out}")


if __name__ == "__main__":
    main()
