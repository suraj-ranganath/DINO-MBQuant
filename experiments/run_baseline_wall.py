from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.dino_runner import run_variant_once


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FP16 Wall baseline and emit baseline metrics JSON.")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--opt-steps", type=int, default=None)
    parser.add_argument("--n-evals", type=int, default=None)
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    eval_cfg = cfg["evaluation"]
    seed = int(eval_cfg["seeds"][0] if args.seed is None else args.seed)
    opt_steps = int(eval_cfg["opt_steps"][0] if args.opt_steps is None else args.opt_steps)
    n_evals = int(eval_cfg["n_evals"] if args.n_evals is None else args.n_evals)

    run_dir = Path("results/baseline/fp16").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics, trace = run_variant_once(
        config_path=args.config,
        variant_name="fp16",
        run_dir=str(run_dir),
        seed=seed,
        opt_steps=opt_steps,
        n_evals=n_evals,
    )

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "variant_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
