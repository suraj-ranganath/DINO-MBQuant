from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt


def load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mac quantization benchmark results.")
    parser.add_argument(
        "--json",
        nargs="+",
        required=True,
        help="One or more benchmark JSON outputs from benchmark_predictor_coreml.py",
    )
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    rows = []
    for path in args.json:
        obj = load(path)
        bsz = int(obj["shape"]["batch_size"])
        runs = obj["runs"]
        for key in ["torch_mps_fp16", "torch_mps_fake_int8", "coreml_fp16", "coreml_int8"]:
            if runs.get(key) is None:
                continue
            rows.append((bsz, key, float(runs[key]["per_iter_ms"])))

    if not rows:
        raise SystemExit("No valid benchmark rows found.")

    rows.sort(key=lambda x: (x[0], x[1]))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    method_labels = {
        "torch_mps_fp16": "PyTorch MPS FP16",
        "torch_mps_fake_int8": "PyTorch MPS fake INT8",
        "coreml_fp16": "CoreML FP16",
        "coreml_int8": "CoreML INT8",
    }
    colors = {
        "torch_mps_fp16": "#0072B2",
        "torch_mps_fake_int8": "#56B4E9",
        "coreml_fp16": "#009E73",
        "coreml_int8": "#D55E00",
    }

    batches = sorted({r[0] for r in rows})
    methods = ["torch_mps_fp16", "torch_mps_fake_int8", "coreml_fp16", "coreml_int8"]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    width = 0.18
    x0 = range(len(batches))
    for i, m in enumerate(methods):
        ys = []
        for b in batches:
            val = next((v for bb, mm, v in rows if bb == b and mm == m), None)
            ys.append(val if val is not None else float("nan"))
        xs = [x + (i - 1.5) * width for x in x0]
        ax.bar(xs, ys, width=width, label=method_labels[m], color=colors[m])

    ax.set_xticks(list(x0))
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Latency per forward (ms)")
    ax.set_title("DINO-WM predictor latency on M4 MacBook Pro")
    ax.legend(frameon=True, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_path = out_dir / "mac_predictor_latency.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
