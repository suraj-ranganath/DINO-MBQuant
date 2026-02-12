from __future__ import annotations

import argparse
import json
from typing import Dict, Any

import yaml

from experiments.dino_runner import run_variant_once
from experiments.run_paths import resolve_path


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and sanity-check FP16/INT8 variants.")
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
        help="Allow writing into an existing run folder (for resume/continuation).",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    variants_root = resolve_path(cfg, key="variants_root", run_name=args.run_name)
    if args.run_name and variants_root.exists() and not args.allow_existing:
        raise SystemExit(
            f"Run folder already exists: {variants_root}\n"
            "Choose a new --run-name, or pass --allow-existing to continue."
        )
    sanity = cfg["sanity"]
    eval_cfg = cfg.get("evaluation", {})

    # Sanity runs should not depend on paired target manifests. If a config uses
    # goal_source=file without a concrete static goal_file_path, force random_state.
    goal_source_override = None
    goal_file_path_override = None
    goal_h_override = None
    planner_max_iter_override = None
    configured_goal_source = str(eval_cfg.get("goal_source", "random_state"))
    configured_goal_file_path = eval_cfg.get("goal_file_path")
    configured_goal_template = eval_cfg.get("goal_file_path_template")
    if configured_goal_source == "file" and not configured_goal_file_path and configured_goal_template:
        goal_source_override = "random_state"
        goal_h_override = int(eval_cfg.get("goal_H", 5))
    if eval_cfg.get("planner_max_iter") is not None:
        planner_max_iter_override = int(eval_cfg["planner_max_iter"])

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
            goal_source_override=goal_source_override,
            goal_file_path_override=goal_file_path_override,
            goal_H_override=goal_h_override,
            planner_max_iter_override=planner_max_iter_override,
            budget_id="sanity",
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
