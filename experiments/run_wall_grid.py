from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from experiments.dino_runner import run_variant_once
from experiments.run_paths import resolve_path


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_csv_arg(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _format_goal_file_path(
    template: str | None,
    *,
    run_name: str | None,
    budget_id: str,
    seed: int,
    opt_steps: int,
    goal_h: int,
) -> str | None:
    if not template:
        return None
    return template.format(
        run_name=run_name or "",
        budget_id=budget_id,
        seed=int(seed),
        opt_steps=int(opt_steps),
        goal_H=int(goal_h),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wall evaluation grid for all variants.")
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
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated subset of variants to run (default: all in config).",
    )
    parser.add_argument(
        "--budget-ids",
        default=None,
        help="Comma-separated subset of budget ids to run when evaluation.budget_grid is set.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated seed override applied to selected budgets.",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    eval_cfg = cfg["evaluation"]
    wall_root = resolve_path(cfg, key="wall_root", run_name=args.run_name)
    if args.run_name and wall_root.exists() and not args.allow_existing:
        raise SystemExit(
            f"Run folder already exists: {wall_root}\n"
            "Choose a new --run-name, or pass --allow-existing to continue."
        )
    wall_root.mkdir(parents=True, exist_ok=True)

    failures = []

    requested_variants = _parse_csv_arg(args.variants)
    if requested_variants:
        missing_variants = [v for v in requested_variants if v not in cfg["variants"]]
        if missing_variants:
            raise SystemExit(f"Unknown variants requested: {missing_variants}")
        variants = requested_variants
    else:
        variants = list(cfg["variants"].keys())

    requested_budget_ids = set(_parse_csv_arg(args.budget_ids))
    requested_seeds = [int(s) for s in _parse_csv_arg(args.seeds)]
    budget_grid = list(eval_cfg.get("budget_grid", []))
    has_budget_grid = bool(budget_grid)
    if budget_grid:
        budgets = []
        for idx, item in enumerate(budget_grid):
            budget_id = str(item.get("id", item.get("name", f"b{idx+1}")))
            if requested_budget_ids and budget_id not in requested_budget_ids:
                continue
            budgets.append(
                {
                    "budget_id": budget_id,
                    "opt_steps": int(item["opt_steps"]),
                    "goal_H": int(item.get("goal_H", eval_cfg.get("goal_H", 5))),
                    "planner_max_iter": (
                        int(item["planner_max_iter"])
                        if item.get("planner_max_iter") is not None
                        else (
                            int(eval_cfg["planner_max_iter"])
                            if eval_cfg.get("planner_max_iter") is not None
                            else None
                        )
                    ),
                    "goal_source": str(item.get("goal_source", eval_cfg.get("goal_source", "random_state"))),
                    "goal_file_path": item.get("goal_file_path"),
                    "goal_file_path_template": item.get(
                        "goal_file_path_template", eval_cfg.get("goal_file_path_template")
                    ),
                    "seeds": [int(s) for s in item.get("seeds", eval_cfg["seeds"])],
                }
            )
    else:
        opt_steps_list = list(eval_cfg["opt_steps"])
        budgets = [
            {
                "budget_id": f"opt{int(opt_steps)}",
                "opt_steps": int(opt_steps),
                "goal_H": int(eval_cfg.get("goal_H", 5)),
                "planner_max_iter": (
                    int(eval_cfg["planner_max_iter"])
                    if eval_cfg.get("planner_max_iter") is not None
                    else None
                ),
                "goal_source": str(eval_cfg.get("goal_source", "random_state")),
                "goal_file_path": eval_cfg.get("goal_file_path"),
                "goal_file_path_template": eval_cfg.get("goal_file_path_template"),
                "seeds": [int(s) for s in eval_cfg["seeds"]],
            }
            for opt_steps in opt_steps_list
        ]
    if requested_seeds:
        for b in budgets:
            b["seeds"] = requested_seeds
    if not budgets:
        raise SystemExit("No budgets selected to run.")

    run_budget = int(args.max_runs or 0)
    run_count = 0

    if args.interleave_variants:
        run_order = [
            (variant_name, b, seed)
            for b in budgets
            for seed in b["seeds"]
            for variant_name in variants
        ]
    else:
        run_order = [
            (variant_name, b, seed)
            for variant_name in variants
            for b in budgets
            for seed in b["seeds"]
        ]

    for variant_name, budget_cfg, seed in run_order:
        opt_steps = int(budget_cfg["opt_steps"])
        budget_id = str(budget_cfg["budget_id"])
        try:
            if has_budget_grid:
                run_dir = wall_root / variant_name / f"budget_{budget_id}" / f"seed_{seed}"
            else:
                run_dir = wall_root / variant_name / f"opt_steps_{opt_steps}" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"

            if args.skip_existing and metrics_path.exists():
                print(
                    f"[skip] variant={variant_name} budget={budget_id} opt_steps={opt_steps} seed={seed} "
                    f"(existing metrics)"
                )
                continue
            if run_budget > 0 and run_count >= run_budget:
                print(f"Reached run budget: {run_budget}")
                break
            run_count += 1

            goal_file_path = budget_cfg.get("goal_file_path")
            goal_file_template = budget_cfg.get("goal_file_path_template")
            if goal_file_template:
                goal_file_path = _format_goal_file_path(
                    str(goal_file_template),
                    run_name=args.run_name,
                    budget_id=budget_id,
                    seed=int(seed),
                    opt_steps=opt_steps,
                    goal_h=int(budget_cfg["goal_H"]),
                )
            if budget_cfg["goal_source"] == "file" and not goal_file_path:
                raise RuntimeError(
                    f"goal_source=file but no goal file path configured for budget={budget_id}, seed={seed}"
                )

            metrics, _ = run_variant_once(
                config_path=args.config,
                variant_name=variant_name,
                run_dir=str(run_dir),
                seed=int(seed),
                opt_steps=int(opt_steps),
                n_evals=int(eval_cfg["n_evals"]),
                goal_source_override=str(budget_cfg["goal_source"]),
                goal_file_path_override=goal_file_path,
                goal_H_override=int(budget_cfg["goal_H"]),
                planner_max_iter_override=budget_cfg["planner_max_iter"],
                budget_id=budget_id,
            )
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(
                f"[ok] variant={variant_name} budget={budget_id} opt_steps={opt_steps} seed={seed} "
                f"success={metrics['success_rate']:.4f}"
            )
        except Exception as exc:
            failures.append(
                {
                    "variant": variant_name,
                    "budget_id": budget_id,
                    "opt_steps": int(opt_steps),
                    "seed": int(seed),
                    "error": str(exc),
                    "run_dir": str(run_dir),
                }
            )
            print(
                f"[fail] variant={variant_name} budget={budget_id} opt_steps={opt_steps} seed={seed} error={exc}"
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
