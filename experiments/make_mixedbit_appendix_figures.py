from __future__ import annotations

import argparse
import re
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from experiments.plot_theme import (
    FAMILY_BASE_COLORS,
    apply_paper_theme,
)
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


def _pareto_mask_size_success(df: pd.DataFrame, *, size_col: str, success_col: str) -> pd.Series:
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
    parser = argparse.ArgumentParser(description="Create appendix figures for mixed-bit study.")
    parser.add_argument("--config", default="configs/experiment_config.mac_mixedbit_story.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--grouped", default="results/summary_grouped.csv")
    parser.add_argument("--stats", default="results/mixedbit_pairwise_stats.csv")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    apply_paper_theme()
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
                "size_mb": float(r["model_size_mean"]),
                "ci_lo": float(r.get("success_rate_ci95_lo", r["success_rate_mean"])),
                "ci_hi": float(r.get("success_rate_ci95_hi", r["success_rate_mean"])),
            }
        )

    if ladder_rows:
        ladder = pd.DataFrame(ladder_rows)
        budgets = list(ladder["budget_id"].dropna().unique())
        fig, axes = plt.subplots(
            1, max(1, len(budgets)), figsize=(6.3 * max(1, len(budgets)), 4.1), squeeze=False
        )
        colors = {"uniform": FAMILY_BASE_COLORS["uniform"], "mixed": FAMILY_BASE_COLORS["mixed"]}
        for i, budget in enumerate(budgets):
            ax = axes[0, i]
            sub = ladder[ladder["budget_id"] == budget]
            pareto_mask = _pareto_mask_size_success(sub, size_col="size_mb", success_col="success")
            pareto_variants = set(sub[pareto_mask]["variant"].astype(str).tolist())
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
                    linewidth=2.1,
                    capsize=3,
                    color=colors[fam],
                    label=fam.capitalize(),
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                )
                psub = fam_sub[fam_sub["variant"].astype(str).isin(pareto_variants)]
                if not psub.empty:
                    ax.scatter(
                        psub["bits"].to_numpy(),
                        psub["success"].to_numpy(),
                        marker="*",
                        s=160,
                        color="#F2C94C",
                        edgecolors="#111111",
                        linewidths=0.7,
                        zorder=5,
                    )
            ax.set_xticks([8, 6, 4, 3])
            ax.set_ylim(0.0, 1.02)
            ax.set_xlabel("Bitwidth")
            ax.set_ylabel("Success Rate")
            ax.set_title(f"Bit Ladder ({budget})")
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
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        sc = ax.scatter(
            sub["enc_bits"],
            sub["pred_bits"],
            c=sub["success"],
            s=np.maximum(80, sub["size_mb"] * 1.5),
            cmap="cividis",
            alpha=0.9,
            edgecolors="white",
            linewidths=0.8,
        )
        pareto_mask = _pareto_mask_size_success(sub, size_col="size_mb", success_col="success")
        psub = sub[pareto_mask].copy()
        ax.scatter(
            psub["enc_bits"],
            psub["pred_bits"],
            marker="*",
            s=200,
            color="#F2C94C",
            edgecolors="#111111",
            linewidths=0.8,
            zorder=5,
        )
        for _, r in psub.iterrows():
            ax.annotate(
                labels.get(str(r["variant"]), str(r["variant"])),
                (r["enc_bits"], r["pred_bits"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#DDDDDD", alpha=0.8),
            )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Success Rate")
        ax.set_xlabel("Encoder bits")
        ax.set_ylabel("Predictor bits")
        ax.set_title(f"Asymmetric Bit Allocation Map ({focus_budget})")
        ax.set_xticks(sorted(sub["enc_bits"].unique()))
        ax.set_yticks(sorted(sub["pred_bits"].unique()))
        ax.grid(alpha=0.25)
        ax.text(
            0.02,
            0.02,
            "* Pareto frontier in (model size, success)",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#DDDDDD", alpha=0.85),
        )
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
            fig, ax = plt.subplots(figsize=(8.4, max(3.0, 0.45 * len(stats_df) + 1.5)))
            means = stats_df["delta_mean"].to_numpy(dtype=float)
            lo = stats_df["delta_ci95_lo"].to_numpy(dtype=float)
            hi = stats_df["delta_ci95_hi"].to_numpy(dtype=float)
            point_colors = [
                FAMILY_BASE_COLORS["mixed"] if m >= 0 else FAMILY_BASE_COLORS["uniform"] for m in means
            ]
            for idx, m in enumerate(means):
                ax.errorbar(
                    [m],
                    [y[idx]],
                    xerr=[[max(0.0, m - lo[idx])], [max(0.0, hi[idx] - m)]],
                    fmt="o",
                    capsize=3,
                    color=point_colors[idx],
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                )
            ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
            ax.set_yticks(y)
            ax.set_yticklabels(stats_df["label"])
            ax.set_xlabel("Paired success delta (variant_a - variant_b)")
            ax.set_title("Paired Mixed-Bit Effects (95% bootstrap CI)")
            ax.grid(alpha=0.25, axis="x")
            fig.tight_layout()
            out = fig_root / "appendix_pairwise_deltas.pdf"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {out}")


if __name__ == "__main__":
    main()
