from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_episode_videos(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    pattern = re.compile(r"output_final_(\d+)_(success|failure)\.mp4$")
    out: Dict[int, Dict[str, Any]] = {}
    for video in sorted(run_dir.glob("output_final_*_*.mp4")):
        m = pattern.match(video.name)
        if not m:
            continue
        idx = int(m.group(1))
        status = m.group(2)
        out[idx] = {
            "path": str(video.resolve()),
            "success": status == "success",
            "status": status,
        }
    return out


def score_episode(idx: int, per_variant: Dict[str, Dict[str, Any]]) -> Tuple[int, int]:
    # Prioritize episodes where mixed succeeds and uniform fails.
    mixed = per_variant.get("mixed_int8", {}).get("success", False)
    uniform = per_variant.get("uniform_int8", {}).get("success", False)
    score = 2 if (mixed and not uniform) else 1 if mixed else 0
    return (score, -idx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export replay artifacts for Streamlit demo.")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    demo_cfg = cfg["demo"]
    wall_root = Path(cfg["paths"]["wall_root"]).resolve()
    artifacts_root = Path(cfg["paths"]["demo_artifacts_root"]).resolve()
    artifacts_root.mkdir(parents=True, exist_ok=True)

    ref_opt = int(demo_cfg["reference_opt_steps"])
    ref_seed = int(demo_cfg["reference_seed"])
    candidate_pool = int(demo_cfg.get("candidate_pool", 30))
    episode_count = int(demo_cfg.get("episode_count", 10))

    variants = list(cfg["variants"].keys())
    episodes_by_variant: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for variant in variants:
        run_dir = wall_root / variant / f"opt_steps_{ref_opt}" / f"seed_{ref_seed}"
        if not run_dir.exists():
            raise SystemExit(f"Missing run dir for demo export: {run_dir}")
        episodes = parse_episode_videos(run_dir)
        if not episodes:
            raise SystemExit(f"No output_final mp4 files found in {run_dir}")
        episodes_by_variant[variant] = episodes

    common_idxs = set.intersection(*(set(v.keys()) for v in episodes_by_variant.values()))
    if not common_idxs:
        raise SystemExit("No common episode IDs across variants for demo export.")

    common_sorted = sorted(common_idxs)[:candidate_pool]

    candidates: List[Tuple[Tuple[int, int], int, Dict[str, Dict[str, Any]]]] = []
    for idx in common_sorted:
        per_variant = {variant: episodes_by_variant[variant][idx] for variant in variants}
        candidates.append((score_episode(idx, per_variant), idx, per_variant))

    candidates.sort(reverse=True)
    selected = candidates[:episode_count]

    manifest = {
        "reference_opt_steps": ref_opt,
        "reference_seed": ref_seed,
        "episodes": [],
        "variants": variants,
    }

    for new_idx, (_, original_idx, per_variant) in enumerate(selected):
        episode_entry = {
            "episode_id": new_idx,
            "source_episode_idx": original_idx,
            "variants": {},
        }
        for variant, info in per_variant.items():
            variant_dir = artifacts_root / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            dst = variant_dir / f"episode_{new_idx}.mp4"
            shutil.copy2(info["path"], dst)
            episode_entry["variants"][variant] = {
                "video_path": str(dst.relative_to(artifacts_root)),
                "success": bool(info["success"]),
                "status": info["status"],
            }
        manifest["episodes"].append(episode_entry)

    manifest_path = artifacts_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
