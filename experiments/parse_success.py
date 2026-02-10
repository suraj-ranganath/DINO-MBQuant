from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional


def parse_logs_success(logs_path: Path) -> Optional[float]:
    if not logs_path.exists():
        return None
    lines = logs_path.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "final_eval/success_rate" in row:
            return float(row["final_eval/success_rate"])
    return None


def parse_videos_success(run_dir: Path) -> Dict[str, int]:
    counts = {"success": 0, "failure": 0}
    pattern = re.compile(r"output_final_(\d+)_(success|failure)\.mp4$")
    for path in run_dir.glob("output_final_*_*.mp4"):
        m = pattern.match(path.name)
        if not m:
            continue
        counts[m.group(2)] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse DINO-WM planning success from run outputs.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n-evals", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    logs_path = run_dir / "logs.json"

    success_rate = parse_logs_success(logs_path)
    video_counts = parse_videos_success(run_dir)

    n_evals = args.n_evals
    if n_evals is None:
        n_evals = video_counts["success"] + video_counts["failure"]

    success_count = None
    if success_rate is not None and n_evals:
        success_count = int(round(success_rate * n_evals))

    payload = {
        "run_dir": str(run_dir),
        "logs_path": str(logs_path),
        "n_evals": n_evals,
        "success_rate": success_rate,
        "success_count": success_count,
        "video_success_count": video_counts["success"],
        "video_failure_count": video_counts["failure"],
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
