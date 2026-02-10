from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import yaml

from experiments.dino_runner import run_variant_once


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and sanity-check FP16/INT8 variants.")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    variants_root = Path(cfg["paths"]["variants_root"]).resolve()
    sanity = cfg["sanity"]

    for variant_name in cfg["variants"].keys():
        variant_root = variants_root / variant_name
        run_dir = variant_root / "sanity"
        variant_root.mkdir(parents=True, exist_ok=True)

        metrics, trace = run_variant_once(
            config_path=args.config,
            variant_name=variant_name,
            run_dir=str(run_dir),
            seed=int(sanity["seed"]),
            opt_steps=int(sanity["opt_steps"]),
            n_evals=int(sanity["n_evals"]),
        )

        variant_spec = {
            "variant_name": variant_name,
            "quant_backend_requested": cfg.get("quantization", {}).get("backend", "bitsandbytes"),
            "quant_backend_effective": trace.get("quant_backend_effective", "fp16"),
            "quantized_layer_paths": trace.get("quantized_layer_paths", []),
            "excluded_or_skipped_layer_paths": trace.get("excluded_or_skipped_layer_paths", []),
            "dtype_encoder": "fp16" if variant_name != "uniform_int8" else "int8_weight_only",
            "dtype_predictor": "fp16" if variant_name == "fp16" else "int8_weight_only",
            "base_checkpoint": trace.get("base_checkpoint"),
            "sanity_run": {
                "ran": True,
                "status": "ok",
                "n_evals": int(sanity["n_evals"]),
                "seed": int(sanity["seed"]),
                "opt_steps": int(sanity["opt_steps"]),
                "success_rate": metrics["success_rate"],
                "run_dir": str(run_dir),
            },
        }

        (variant_root / "variant_spec.json").write_text(
            json.dumps(variant_spec, indent=2), encoding="utf-8"
        )
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        print(f"[ok] {variant_name}: success_rate={metrics['success_rate']:.4f} run_dir={run_dir}")


if __name__ == "__main__":
    main()
