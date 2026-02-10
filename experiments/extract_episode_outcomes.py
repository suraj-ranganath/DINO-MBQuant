from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from experiments.run_paths import resolve_path, run_scoped_file


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_video_outcomes(run_dir: Path) -> Dict[int, bool]:
    out: Dict[int, bool] = {}
    pattern = re.compile(r"output_final_(\d+)_(success|failure)\.mp4$")
    for p in sorted(run_dir.glob("output_final_*_*.mp4")):
        m = pattern.match(p.name)
        if not m:
            continue
        out[int(m.group(1))] = m.group(2) == "success"
    return out


def _load_goal_difficulty(run_dir: Path) -> Dict[int, float]:
    target_path = run_dir / "plan_targets.pkl"
    if not target_path.exists():
        return {}
    with target_path.open("rb") as f:
        data = pickle.load(f)
    s0 = np.array(data.get("state_0"))
    sg = np.array(data.get("state_g"))
    if s0.ndim != 2 or sg.ndim != 2 or s0.shape != sg.shape:
        return {}
    d = np.linalg.norm(s0[:, :2] - sg[:, :2], axis=1)
    return {int(i): float(v) for i, v in enumerate(d)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract episode-level outcomes for paired analysis.")
    parser.add_argument("--config", default="configs/experiment_config.mac_transition_study.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out", default="results/episode_outcomes.csv")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    wall_root = resolve_path(cfg, key="wall_root", run_name=args.run_name)
    if not wall_root.exists():
        raise SystemExit(f"Missing wall root: {wall_root}")

    rows: List[Dict[str, Any]] = []
    for metrics_path in sorted(wall_root.rglob("metrics.json")):
        run_dir = metrics_path.parent
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        outcomes_path = metrics.get("episode_successes_path")
        difficulty = _load_goal_difficulty(run_dir)

        episode_success_map: Dict[int, bool] = {}
        if outcomes_path and Path(outcomes_path).exists():
            entries = json.loads(Path(outcomes_path).read_text(encoding="utf-8"))
            for row in entries:
                episode_success_map[int(row["episode_id"])] = bool(row["success"])
        else:
            episode_success_map = _parse_video_outcomes(run_dir)

        if not episode_success_map:
            continue

        for episode_id, success in sorted(episode_success_map.items()):
            rows.append(
                {
                    "run_name": args.run_name,
                    "variant": metrics.get("variant"),
                    "budget_id": metrics.get("budget_id", "default"),
                    "seed": int(metrics.get("seed", -1)),
                    "opt_steps": int(metrics.get("opt_steps", -1)),
                    "episode_id": int(episode_id),
                    "success": int(bool(success)),
                    "goal_distance_init": difficulty.get(int(episode_id)),
                    "pair_id": f"{metrics.get('budget_id', 'default')}::seed{int(metrics.get('seed', -1))}::ep{int(episode_id)}",
                    "run_dir": str(run_dir),
                }
            )

    if not rows:
        raise SystemExit(
            "No episode outcomes found. Ensure runs were created with episode_outcomes.json "
            "or with output_final_<id>_(success|failure).mp4 files."
        )

    df = pd.DataFrame(rows).sort_values(
        by=["budget_id", "seed", "episode_id", "variant"]
    ).reset_index(drop=True)
    out_path = run_scoped_file(args.out, run_name=args.run_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

