from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from experiments.run_paths import resolve_path, run_scoped_file


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _last_final_eval(logs_path: Path) -> Dict[str, Any]:
    if not logs_path.exists():
        return {}
    final_row: Dict[str, Any] = {}
    for line in logs_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if any(k.startswith("final_eval/") for k in row.keys()):
            final_row = row
    return final_row


def _variant_family(name: str) -> str:
    if name.startswith("uniform_"):
        return "uniform"
    if name.startswith("mixed_"):
        return "mixed"
    if name == "fp16":
        return "fp16"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze mechanistic metrics from run logs.")
    parser.add_argument("--config", default="configs/experiment_config.mac_transition_study.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--out-csv", default="results/mechanistic_metrics.csv")
    parser.add_argument("--out-json", default="results/mechanistic_correlations.json")
    parser.add_argument("--fig-out", default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    wall_root = resolve_path(cfg, key="wall_root", run_name=args.run_name)
    if not wall_root.exists():
        raise SystemExit(f"Missing wall root: {wall_root}")

    summary_path = run_scoped_file(args.summary, run_name=args.run_name)
    if not summary_path.exists():
        raise SystemExit(f"Missing summary file: {summary_path}")
    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        raise SystemExit("Summary is empty.")

    rows: List[Dict[str, Any]] = []
    for _, row in summary_df.iterrows():
        run_dir = Path(str(row.get("source_output_dir", "")))
        if not run_dir.exists():
            continue
        final = _last_final_eval(run_dir / "logs.json")
        rows.append(
            {
                "variant": row["variant"],
                "budget_id": row.get("budget_id", "default"),
                "seed": int(row["seed"]),
                "success_rate": float(row["success_rate"]),
                "model_size_mb": float(row["model_size_mb"]),
                "mean_state_dist": final.get("final_eval/mean_state_dist"),
                "mean_div_visual_emb": final.get("final_eval/mean_div_visual_emb"),
                "mean_div_proprio_emb": final.get("final_eval/mean_div_proprio_emb"),
                "family": _variant_family(str(row["variant"])),
            }
        )

    mech_df = pd.DataFrame(rows)
    if mech_df.empty:
        raise SystemExit("No mechanistic rows were parsed from logs.")

    corr_targets = ["mean_state_dist", "mean_div_visual_emb", "mean_div_proprio_emb"]
    corrs = {}
    for c in corr_targets:
        sub = mech_df[["success_rate", c]].dropna()
        if sub.empty:
            corrs[c] = {"spearman": None}
            continue
        value = float(sub["success_rate"].corr(sub[c], method="spearman"))
        corrs[c] = {"spearman": value if pd.notna(value) else None}

    out_csv = run_scoped_file(args.out_csv, run_name=args.run_name)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mech_df.to_csv(out_csv, index=False)

    out_json = run_scoped_file(args.out_json, run_name=args.run_name)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(corrs, indent=2, allow_nan=False), encoding="utf-8")

    fig_path = run_scoped_file(
        args.fig_out if args.fig_out else f"{cfg['paths']['figures_root']}/mechanistic_success_vs_divergence.pdf",
        run_name=args.run_name,
    )
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = {"fp16": "#4D4D4D", "uniform": "#D55E00", "mixed": "#009E73", "other": "#666666"}
    for fam, g in mech_df.groupby("family"):
        ax.scatter(
            g["mean_div_visual_emb"],
            g["success_rate"],
            s=55,
            alpha=0.9,
            label=fam,
            color=colors.get(fam, "#666666"),
            edgecolors="white",
            linewidths=0.8,
        )
    for _, r in mech_df.iterrows():
        ax.annotate(str(r["variant"]), (r["mean_div_visual_emb"], r["success_rate"]), textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.set_xlabel("Final mean divergence in visual embedding")
    ax.set_ylabel("Success rate")
    ax.set_title("Mechanistic signal: embedding divergence vs planning success")
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
