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


def _parse_csv_arg(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paired plan_targets.pkl files for fixed-goal evaluation."
    )
    parser.add_argument("--config", default="configs/experiment_config.mac_transition_study.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--budget-ids",
        default=None,
        help="Optional comma-separated subset of budget ids to generate.",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        default=False,
        help="Allow reusing existing paired-target run folder.",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    eval_cfg = cfg["evaluation"]
    paired_cfg = cfg.get("paired_targets", {})
    generator_variant = str(paired_cfg.get("generator_variant", "fp16"))
    generator_opt_steps = int(paired_cfg.get("generator_opt_steps", 1))
    requested_budget_ids = set(_parse_csv_arg(args.budget_ids))

    root = Path(str(paired_cfg.get("root", "results/paired_targets"))).resolve() / args.run_name
    if root.exists() and not args.allow_existing:
        raise SystemExit(
            f"Paired target folder already exists: {root}\n"
            "Choose a new --run-name or pass --allow-existing."
        )
    root.mkdir(parents=True, exist_ok=True)

    budget_grid = list(eval_cfg.get("budget_grid", []))
    if budget_grid:
        budgets = []
        for idx, b in enumerate(budget_grid):
            budget_id = str(b.get("id", b.get("name", f"b{idx+1}")))
            if requested_budget_ids and budget_id not in requested_budget_ids:
                continue
            budgets.append(
                {
                    "budget_id": budget_id,
                    "goal_H": int(b.get("goal_H", eval_cfg.get("goal_H", 5))),
                    "seeds": [int(s) for s in b.get("seeds", eval_cfg["seeds"])],
                }
            )
    else:
        budgets = [
            {
                "budget_id": "default",
                "goal_H": int(eval_cfg.get("goal_H", 5)),
                "seeds": [int(s) for s in eval_cfg["seeds"]],
            }
        ]

    manifest: Dict[str, Any] = {
        "run_name": args.run_name,
        "generator_variant": generator_variant,
        "generator_opt_steps": generator_opt_steps,
        "n_evals": int(eval_cfg["n_evals"]),
        "budgets": {},
    }

    for budget in budgets:
        budget_id = budget["budget_id"]
        manifest["budgets"][budget_id] = {}
        for seed in budget["seeds"]:
            run_dir = root / f"budget_{budget_id}" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            target_path = run_dir / "plan_targets.pkl"
            if target_path.exists() and not args.allow_existing:
                raise SystemExit(
                    f"Target file already exists: {target_path}\n"
                    "Use a new run-name or pass --allow-existing."
                )

            metrics, trace = run_variant_once(
                config_path=args.config,
                variant_name=generator_variant,
                run_dir=str(run_dir),
                seed=int(seed),
                opt_steps=generator_opt_steps,
                n_evals=int(eval_cfg["n_evals"]),
                goal_source_override="random_state",
                goal_H_override=int(budget["goal_H"]),
                planner_max_iter_override=1,
                budget_id=f"paired_{budget_id}",
            )
            if not target_path.exists():
                raise RuntimeError(f"Expected paired targets at {target_path}, but file is missing.")

            manifest["budgets"][budget_id][str(seed)] = {
                "goal_H": int(budget["goal_H"]),
                "plan_targets_path": str(target_path),
                "source_run_dir": str(run_dir),
                "source_metrics_path": str(run_dir / "metrics.json"),
                "generator_success_rate": float(metrics.get("success_rate", 0.0)),
                "trace_quant_backend": trace.get("quant_backend_effective"),
            }
            print(f"[ok] budget={budget_id} seed={seed} targets={target_path}")

    manifest_path = root / "paired_targets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

