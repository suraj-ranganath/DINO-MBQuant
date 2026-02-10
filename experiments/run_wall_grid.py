from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from experiments.dino_runner import run_variant_once


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wall evaluation grid for all variants.")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--fail-fast", action="store_true", default=False)
    parser.add_argument(
        "--interleave-variants",
        action="store_true",
        default=False,
        help="Run variants per (opt_steps, seed) condition instead of full variant blocks.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional cap on number of runs to execute (0 means all runs).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip runs whose metrics.json already exists.",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    eval_cfg = cfg["evaluation"]
    wall_root = Path(cfg["paths"]["wall_root"]).resolve()
    wall_root.mkdir(parents=True, exist_ok=True)

    failures = []

    variants = list(cfg["variants"].keys())
    opt_steps_list = list(eval_cfg["opt_steps"])
    seeds_list = list(eval_cfg["seeds"])
    run_budget = int(args.max_runs or 0)
    run_count = 0

    if args.interleave_variants:
        run_order = [
            (variant_name, opt_steps, seed)
            for opt_steps in opt_steps_list
            for seed in seeds_list
            for variant_name in variants
        ]
    else:
        run_order = [
            (variant_name, opt_steps, seed)
            for variant_name in variants
            for opt_steps in opt_steps_list
            for seed in seeds_list
        ]

    for variant_name, opt_steps, seed in run_order:
        try:
            run_dir = wall_root / variant_name / f"opt_steps_{opt_steps}" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"

            if args.skip_existing and metrics_path.exists():
                print(
                    f"[skip] variant={variant_name} opt_steps={opt_steps} seed={seed} "
                    f"(existing metrics)"
                )
                continue
            if run_budget > 0 and run_count >= run_budget:
                print(f"Reached run budget: {run_budget}")
                break
            run_count += 1

            metrics, _ = run_variant_once(
                config_path=args.config,
                variant_name=variant_name,
                run_dir=str(run_dir),
                seed=int(seed),
                opt_steps=int(opt_steps),
                n_evals=int(eval_cfg["n_evals"]),
            )
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(
                f"[ok] variant={variant_name} opt_steps={opt_steps} seed={seed} "
                f"success={metrics['success_rate']:.4f}"
            )
        except Exception as exc:
            failures.append(
                {
                    "variant": variant_name,
                    "opt_steps": int(opt_steps),
                    "seed": int(seed),
                    "error": str(exc),
                    "run_dir": str(run_dir),
                }
            )
            print(
                f"[fail] variant={variant_name} opt_steps={opt_steps} seed={seed} error={exc}"
            )
            if args.fail_fast:
                raise

    if failures:
        fail_path = wall_root / "failures.json"
        fail_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise SystemExit(f"Grid finished with {len(failures)} failure(s). See {fail_path}")

    print("Grid run completed successfully.")


if __name__ == "__main__":
    main()
