from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from experiments.run_paths import resolve_path


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_run_dir(cfg: Dict[str, Any], run_name: str) -> Path:
    wall_root = resolve_path(cfg, key="wall_root", run_name=None)
    exact = wall_root / run_name
    if exact.exists():
        return exact

    cands = sorted(wall_root.glob(f"*{run_name}*"))
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        names = ", ".join(str(c.name) for c in cands)
        raise SystemExit(f"Multiple run dirs match '{run_name}': {names}")
    raise SystemExit(f"Run directory not found for '{run_name}' under {wall_root}")


def load_metrics(run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for p in sorted(run_dir.rglob("metrics.json")):
        with p.open("r", encoding="utf-8") as f:
            r = json.load(f)
        rows.append(
            {
                "variant": str(r.get("variant")),
                "budget_id": str(r.get("budget_id")),
                "seed": int(r.get("seed", -1)),
                "success_rate": float(r.get("success_rate", np.nan)),
                "avg_plan_time_seconds": float(r.get("avg_plan_time_seconds", np.nan)),
                "model_size_mb": float(r.get("model_size_mb", np.nan)),
                "n_evals": int(r.get("n_evals", 0)),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No metrics.json found in {run_dir}")
    return df


def color_for_variant(name: str) -> str:
    if name == "fp16":
        return "#4D4D4D"
    if name.startswith("mixed"):
        return "#009E73"
    if name.startswith("uniform"):
        return "#D55E00"
    return "#0072B2"


def plot_frontier(df: pd.DataFrame, labels: Dict[str, str], out: Path) -> None:
    g = (
        df.groupby(["variant", "budget_id"], as_index=False)
        .agg(
            success_mean=("success_rate", "mean"),
            success_std=("success_rate", "std"),
            size_mean=("model_size_mb", "mean"),
            time_mean=("avg_plan_time_seconds", "mean"),
        )
        .fillna(0.0)
    )

    budgets = list(g["budget_id"].dropna().unique())
    markers = ["o", "s", "^", "D", "P", "X"]
    budget_marker = {b: markers[i % len(markers)] for i, b in enumerate(budgets)}

    plt.figure(figsize=(7.2, 4.6))
    for _, r in g.iterrows():
        v = str(r["variant"])
        b = str(r["budget_id"])
        x = float(r["size_mean"])
        y = float(r["success_mean"])
        plt.scatter(
            x,
            y,
            s=120,
            color=color_for_variant(v),
            marker=budget_marker[b],
            alpha=0.92,
            edgecolors="black",
            linewidths=0.4,
        )
        lbl = labels.get(v, v)
        plt.annotate(f"{lbl} ({b})", (x, y), xytext=(5, 4), textcoords="offset points", fontsize=7)

    plt.xlabel("Model Size (MB)")
    plt.ylabel("Success Rate")
    plt.ylim(0.0, 1.02)
    plt.title("Accuracy-Memory Frontier (Interim)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def plot_budget_bars(df: pd.DataFrame, labels: Dict[str, str], out: Path) -> None:
    key = ["fp16", "uniform_int8", "mixed_int8", "uniform_int4", "mixed_int4"]
    d = df[df["variant"].isin(key)].copy()
    if d.empty:
        return

    budgets = list(d["budget_id"].dropna().unique())
    x = np.arange(len(key))
    width = 0.34 if len(budgets) == 2 else 0.24

    plt.figure(figsize=(8.8, 4.8))
    for i, b in enumerate(budgets):
        sub = d[d["budget_id"] == b]
        means = []
        stds = []
        for v in key:
            vals = sub[sub["variant"] == v]["success_rate"].to_numpy()
            means.append(float(np.mean(vals)) if len(vals) else np.nan)
            stds.append(float(np.std(vals)) if len(vals) else 0.0)
        xpos = x + (i - (len(budgets) - 1) / 2) * width
        plt.bar(xpos, means, width=width, yerr=stds, capsize=3, alpha=0.82, label=f"Budget {b}")

    # Overlay seed-level points.
    for idx, v in enumerate(key):
        sub = d[d["variant"] == v]
        jit = np.linspace(-0.06, 0.06, num=max(1, len(sub)))
        for j, (_, r) in enumerate(sub.reset_index(drop=True).iterrows()):
            plt.scatter(idx + jit[j], float(r["success_rate"]), color=color_for_variant(v), s=26, zorder=3)

    plt.xticks(x, [labels.get(v, v) for v in key], rotation=20, ha="right")
    plt.ylabel("Success Rate")
    plt.ylim(0.0, 1.02)
    plt.title("Success by Variant and Budget (Seed-level Points)")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def plot_mixed_uniform_delta(df: pd.DataFrame, out: Path) -> None:
    rows = []
    for b in sorted(df["budget_id"].dropna().unique()):
        for s in sorted(df["seed"].dropna().unique()):
            m = df[(df["variant"] == "mixed_int4") & (df["budget_id"] == b) & (df["seed"] == s)]
            u = df[(df["variant"] == "uniform_int4") & (df["budget_id"] == b) & (df["seed"] == s)]
            if m.empty or u.empty:
                continue
            rows.append({"budget_id": b, "seed": int(s), "delta": float(m.iloc[0]["success_rate"] - u.iloc[0]["success_rate"])})

    ddf = pd.DataFrame(rows)
    if ddf.empty:
        return

    plt.figure(figsize=(6.8, 4.2))
    budgets = list(ddf["budget_id"].unique())
    xpos = np.arange(len(budgets))

    means = []
    for i, b in enumerate(budgets):
        vals = ddf[ddf["budget_id"] == b]["delta"].to_numpy()
        means.append(float(np.mean(vals)))
        jit = np.linspace(-0.06, 0.06, num=max(1, len(vals)))
        for j, y in enumerate(vals):
            plt.scatter(i + jit[j], y, color="#0072B2", s=36, zorder=3)

    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    plt.bar(xpos, means, width=0.42, alpha=0.55, color="#56B4E9", label="Mean delta")
    plt.xticks(xpos, budgets)
    plt.ylabel("Delta Success (mixed_int4 - uniform_int4)")
    plt.title("Per-budget Mixed-vs-Uniform INT4 Delta")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def plot_heatmap(df: pd.DataFrame, labels: Dict[str, str], out: Path) -> None:
    d = (
        df.groupby(["variant", "budget_id"], as_index=False)
        .agg(success_mean=("success_rate", "mean"))
        .copy()
    )
    if d.empty:
        return
    pivot = d.pivot(index="variant", columns="budget_id", values="success_mean")
    order = [v for v in ["fp16", "uniform_int8", "mixed_int8", "uniform_int4", "mixed_int4", "encfp16_50", "encheadfp16_50"] if v in pivot.index]
    if order:
        pivot = pivot.loc[order]

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    im = ax.imshow(pivot.to_numpy(), vmin=0.0, vmax=1.0, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([labels.get(v, v) for v in pivot.index])
    ax.set_xlabel("Budget")
    ax.set_title("Success Heatmap (Variant x Budget)")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success Rate")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_paired_delta_ci(run_dir: Path, out: Path) -> None:
    files = [
        run_dir / "paired_delta_bA_mixed_vs_uniform_int4.json",
        run_dir / "paired_delta_bB_mixed_vs_uniform_int4.json",
        run_dir / "paired_delta_bA_tail50_vs_head50_int4.json",
    ]
    rows = []
    for p in files:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
        rows.append(
            {
                "label": f"{d.get('budget_id', 'NA')}:{d.get('variant_a', 'A')}-{d.get('variant_b', 'B')}",
                "mean": float(d.get("delta_mean", 0.0)),
                "lo": float(d.get("delta_ci95_lo", 0.0)),
                "hi": float(d.get("delta_ci95_hi", 0.0)),
            }
        )

    if not rows:
        return
    ddf = pd.DataFrame(rows)
    x = np.arange(len(ddf))
    y = ddf["mean"].to_numpy()
    err_lo = np.maximum(0.0, y - ddf["lo"].to_numpy())
    err_hi = np.maximum(0.0, ddf["hi"].to_numpy() - y)

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.errorbar(x, y, yerr=[err_lo, err_hi], fmt="o", capsize=4, markersize=7, color="#0072B2")
    ax.set_xticks(x)
    ax.set_xticklabels(ddf["label"].tolist(), rotation=15, ha="right")
    ax.set_ylabel("Paired Delta in Success")
    ax.set_title("Paired Delta with 95% CI")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create interim figures directly from metrics.json files.")
    parser.add_argument("--config", default="configs/experiment_config.mac_m3_main2_4h_strict.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    labels = cfg.get("labels", {})

    run_dir = resolve_run_dir(cfg, args.run_name)
    df = load_metrics(run_dir)

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        fig_root = resolve_path(cfg, key="figures_root", run_name=None)
        out_dir = (fig_root / run_dir.name / "interim_metrics").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_frontier(df, labels, out_dir / "interim_frontier_size_vs_success.pdf")
    plot_budget_bars(df, labels, out_dir / "interim_budget_variant_bars.pdf")
    plot_mixed_uniform_delta(df, out_dir / "interim_mixed_vs_uniform_int4_delta.pdf")
    plot_heatmap(df, labels, out_dir / "interim_success_heatmap.pdf")
    plot_paired_delta_ci(run_dir, out_dir / "interim_paired_delta_ci.pdf")

    csv_out = out_dir / "interim_metrics_flat.csv"
    df.sort_values(["variant", "budget_id", "seed"]).to_csv(csv_out, index=False)

    print(f"Run dir: {run_dir}")
    print(f"Wrote {out_dir / 'interim_frontier_size_vs_success.pdf'}")
    print(f"Wrote {out_dir / 'interim_budget_variant_bars.pdf'}")
    print(f"Wrote {out_dir / 'interim_mixed_vs_uniform_int4_delta.pdf'}")
    print(f"Wrote {out_dir / 'interim_success_heatmap.pdf'}")
    print(f"Wrote {out_dir / 'interim_paired_delta_ci.pdf'}")
    print(f"Wrote {csv_out}")


if __name__ == "__main__":
    main()
