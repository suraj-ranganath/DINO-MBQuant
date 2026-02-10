from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate copy-ready markdown snippets for paper writing.")
    parser.add_argument("--config", default="configs/experiment_config.mac.yaml")
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--grouped", default="results/summary_grouped.csv")
    parser.add_argument("--out", default="notes/paper_numbers.md")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    labels = cfg.get("labels", {})

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise SystemExit(f"Missing summary file: {summary_path}")
    grouped_path = Path(args.grouped)
    if not grouped_path.exists():
        raise SystemExit(f"Missing grouped summary file: {grouped_path}")

    df = pd.read_csv(summary_path)
    if df.empty:
        raise SystemExit("Summary is empty.")
    grouped_df = pd.read_csv(grouped_path)
    if grouped_df.empty:
        raise SystemExit("Grouped summary is empty.")

    grouped = (
        df.groupby(["variant", "opt_steps"], as_index=False)
        .agg(
            success_rate_mean=("success_rate", "mean"),
            success_rate_std=("success_rate", "std"),
            avg_time_mean=("avg_plan_time_seconds", "mean"),
            model_size_mean=("model_size_mb", "mean"),
            peak_mem_mean=("peak_gpu_mem_mb", "mean"),
        )
        .sort_values(by=["opt_steps", "variant"])
    )

    focus = grouped[grouped["opt_steps"] == grouped["opt_steps"].max()].copy()
    focus = focus.set_index("variant")

    def get_sr(variant: str) -> float:
        return float(focus.loc[variant, "success_rate_mean"]) if variant in focus.index else float("nan")

    fp16 = get_sr("fp16")
    uni = get_sr("uniform_int8")
    mix = get_sr("mixed_int8")
    gap = mix - uni

    lines = []
    lines.append("# Paper Numbers (Auto-generated)\n")
    lines.append(f"Summary source: `{summary_path}`\n")
    lines.append(f"Grouped source: `{grouped_path}`\n")
    lines.append("## Headline (highest opt_steps)\n")
    lines.append(f"- opt_steps used for headline: `{int(grouped['opt_steps'].max())}`")
    lines.append(f"- FP16 success rate: `{fp16:.3f}`")
    lines.append(f"- Uniform INT8 success rate: `{uni:.3f}`")
    lines.append(f"- Mixed INT8 success rate: `{mix:.3f}`")
    lines.append(f"- Mixed - Uniform gap: `{gap:+.3f}`\n")

    lines.append("## Mixed vs Uniform by opt_steps\n")
    lines.append("| opt_steps | Uniform mean | Mixed mean | Gap (Mixed-Uniform) | Uniform CI95 | Mixed CI95 |")
    lines.append("|---:|---:|---:|---:|---:|---:|")

    uni_map = grouped_df[grouped_df["variant"] == "uniform_int8"].set_index("opt_steps")
    mix_map = grouped_df[grouped_df["variant"] == "mixed_int8"].set_index("opt_steps")
    common_opt = sorted(set(uni_map.index).intersection(set(mix_map.index)))
    for opt in common_opt:
        u = uni_map.loc[opt]
        m = mix_map.loc[opt]
        u_ci = f"[{float(u.get('success_rate_ci95_lo', u['success_rate_mean'])):.3f}, {float(u.get('success_rate_ci95_hi', u['success_rate_mean'])):.3f}]"
        m_ci = f"[{float(m.get('success_rate_ci95_lo', m['success_rate_mean'])):.3f}, {float(m.get('success_rate_ci95_hi', m['success_rate_mean'])):.3f}]"
        lines.append(
            f"| {int(opt)} | {float(u['success_rate_mean']):.3f} | {float(m['success_rate_mean']):.3f} | {(float(m['success_rate_mean']) - float(u['success_rate_mean'])):+.3f} | {u_ci} | {m_ci} |"
        )

    lines.append("## Per-setting results\n")
    lines.append("| Variant | opt_steps | Success mean | Success std | Avg plan time (s) | Model size (MB) | Peak mem (MB) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for _, r in grouped.iterrows():
        label = labels.get(r["variant"], r["variant"])
        std = 0.0 if pd.isna(r["success_rate_std"]) else float(r["success_rate_std"])
        lines.append(
            f"| {label} | {int(r['opt_steps'])} | {float(r['success_rate_mean']):.3f} | {std:.3f} | {float(r['avg_time_mean']):.3f} | {float(r['model_size_mean']):.2f} | {float(r['peak_mem_mean']):.2f} |"
        )

    lines.append("\n## Suggested abstract sentence\n")
    lines.append(
        "- On the Wall task at opt_steps={} (Mac-only preliminary setting), mixed INT8 improved success over uniform INT8 by {} absolute while retaining similar local efficiency footprint.".format(
            int(grouped["opt_steps"].max()),
            f"{gap:+.3f}",
        )
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
