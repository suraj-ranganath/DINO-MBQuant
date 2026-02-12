from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from experiments.quantize_utils import apply_int8_to_linears, estimate_module_storage_mb


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(project_root: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


@contextmanager
def working_directory(path: Path):
    prev = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _merge_plan_cfg(base_cfg_path: Path, overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = OmegaConf.load(str(base_cfg_path))
    cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    # Keep unresolved Hydra interpolations (e.g., ${now:...}) as plain strings.
    # They are not needed by planning_main and resolving them here requires Hydra resolvers.
    return OmegaConf.to_container(cfg, resolve=False)


def _extract_success_rate(metrics: Dict[str, Any], logs_path: Path) -> Optional[float]:
    if "final_eval/success_rate" in metrics:
        try:
            return float(metrics["final_eval/success_rate"])
        except Exception:
            pass

    if logs_path.exists():
        with logs_path.open("r", encoding="utf-8") as f:
            for line in reversed(f.readlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = row.get("final_eval/success_rate")
                if value is not None:
                    return float(value)
    return None


def _build_variant_overrides(config: Dict[str, Any], seed: int, opt_steps: int, n_evals: int) -> Dict[str, Any]:
    eval_cfg = config["evaluation"]
    dino_cfg = config["dino"]

    overrides = {
        "ckpt_base_path": dino_cfg["ckpt_base_path"],
        "model_name": dino_cfg["model_name"],
        "model_epoch": dino_cfg.get("model_epoch", "final"),
        "seed": int(seed),
        "n_evals": int(n_evals),
        "goal_source": eval_cfg.get("goal_source", "random_state"),
        "goal_H": int(eval_cfg.get("goal_H", 5)),
        "n_plot_samples": int(min(n_evals, 10)),
        "planner": {
            "sub_planner": {
                "opt_steps": int(opt_steps),
            }
        },
    }
    if eval_cfg.get("planner_max_iter") is not None:
        overrides["planner"]["max_iter"] = int(eval_cfg["planner_max_iter"])
    return overrides


def _parse_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if not name:
        return None
    norm = str(name).strip().lower()
    if norm in {"float16", "fp16", "half"}:
        return torch.float16
    if norm in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if norm in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {name}")


def _batch_len(value: Any) -> Optional[int]:
    if isinstance(value, torch.Tensor):
        return int(value.shape[0])
    if isinstance(value, np.ndarray):
        return int(value.shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _slice_batch(value: Any, n: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value[:n]
    if isinstance(value, np.ndarray):
        return value[:n]
    if isinstance(value, list):
        return value[:n]
    if isinstance(value, tuple):
        return value[:n]
    return value


def _align_goal_file_to_n_evals(goal_file_path: str, n_evals: int, run_path: Path) -> str:
    path = Path(goal_file_path).resolve()
    with path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        return str(path)

    current_n = _batch_len(data.get("state_0"))
    if current_n is None:
        obs_0 = data.get("obs_0")
        if isinstance(obs_0, dict) and obs_0:
            first_val = next(iter(obs_0.values()))
            current_n = _batch_len(first_val)
    if current_n is None:
        return str(path)

    if int(current_n) == int(n_evals):
        return str(path)
    if int(current_n) < int(n_evals):
        raise RuntimeError(
            f"Paired goal file has too few episodes for n_evals={n_evals}: {path} (contains {current_n})."
        )

    aligned = dict(data)
    if isinstance(aligned.get("obs_0"), dict):
        aligned["obs_0"] = {k: _slice_batch(v, n_evals) for k, v in aligned["obs_0"].items()}
    if isinstance(aligned.get("obs_g"), dict):
        aligned["obs_g"] = {k: _slice_batch(v, n_evals) for k, v in aligned["obs_g"].items()}
    aligned["state_0"] = _slice_batch(aligned.get("state_0"), n_evals)
    aligned["state_g"] = _slice_batch(aligned.get("state_g"), n_evals)
    if aligned.get("gt_actions") is not None:
        aligned["gt_actions"] = _slice_batch(aligned.get("gt_actions"), n_evals)

    out_path = run_path / f"plan_targets_aligned_n{int(n_evals)}.pkl"
    with out_path.open("wb") as f:
        pickle.dump(aligned, f)
    return str(out_path.resolve())


def run_variant_once(
    config_path: str,
    variant_name: str,
    run_dir: str,
    seed: int,
    opt_steps: int,
    n_evals: int,
    goal_source_override: Optional[str] = None,
    goal_file_path_override: Optional[str] = None,
    goal_H_override: Optional[int] = None,
    planner_max_iter_override: Optional[int] = None,
    budget_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    project_root = Path(config_path).resolve().parent.parent
    config = load_yaml(config_path)

    dino_repo = resolve_path(project_root, config["dino"]["repo_path"])
    plan_cfg_name = config["dino"].get("plan_config_name", "plan_wall.yaml")
    plan_cfg_path = dino_repo / "conf" / plan_cfg_name

    if str(dino_repo) not in sys.path:
        sys.path.insert(0, str(dino_repo))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    plan_module = importlib.import_module("plan")

    variant_cfg = config["variants"][variant_name]
    eval_cfg = config.get("evaluation", {})
    quant_cfg = config.get("quantization", {})
    quant_backend = quant_cfg.get("backend", "bitsandbytes")
    fallback_backend = quant_cfg.get("fallback_backend", "fake_int8")
    bnb_4bit_quant_type = str(quant_cfg.get("bnb_4bit_quant_type", "nf4"))
    bnb_4bit_compute_dtype = _parse_dtype(quant_cfg.get("bnb_4bit_compute_dtype"))
    bnb_4bit_use_double_quant = bool(quant_cfg.get("bnb_4bit_use_double_quant", True))
    requested_quant_bits = int(variant_cfg.get("quant_bits", quant_cfg.get("quant_bits", 8)))
    target_bits_overrides = variant_cfg.get("quant_bits_by_target", {}) or {}
    render_videos = bool(eval_cfg.get("render_videos", False))
    episode_successes: Optional[list[bool]] = None

    quantized_paths = []
    skipped_paths = []
    applied_target_bits: Dict[str, int] = {}
    effective_backend = "fp16"
    model_size_mb = None

    original_load_model = plan_module.load_model
    original_eval_actions = plan_module.PlanEvaluator.eval_actions

    def patched_load_model(model_ckpt: Path, train_cfg: Any, num_action_repeat: int, device: torch.device):
        nonlocal quantized_paths, skipped_paths, effective_backend, model_size_mb
        model = original_load_model(model_ckpt, train_cfg, num_action_repeat, device)

        targets = variant_cfg.get("quantize_targets", [])
        for target_name in targets:
            module = getattr(model, target_name, None)
            if module is None:
                skipped_paths.append(f"{target_name}.__missing__")
                continue
            target_bits = int(
                variant_cfg.get(f"{target_name}_quant_bits", target_bits_overrides.get(target_name, requested_quant_bits))
            )
            applied_target_bits[target_name] = target_bits

            report = apply_int8_to_linears(
                module,
                include_paths=variant_cfg.get(f"{target_name}_include_paths"),
                exclude_paths=variant_cfg.get(f"{target_name}_exclude_paths"),
                backend=quant_backend,
                fallback_backend=fallback_backend,
                quant_bits=target_bits,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            )
            quantized_paths.extend([f"{target_name}.{p}" for p in report.quantized_layer_paths])
            skipped_paths.extend([f"{target_name}.{p}" for p in report.skipped_layer_paths])
            if report.quantized_layer_paths:
                effective_backend = report.backend

        # Measure full assembled model size, not just checkpoint payload.
        model_size_mb = estimate_module_storage_mb(model)
        return model

    def patched_eval_actions(self, actions, action_len=None, filename="output", save_video=False):
        nonlocal episode_successes
        effective_save_video = bool(save_video) and render_videos
        logs, successes, e_obses, e_states = original_eval_actions(
            self,
            actions,
            action_len,
            filename=filename,
            save_video=effective_save_video,
        )
        if filename.startswith("output_final"):
            try:
                episode_successes = [bool(x) for x in successes.tolist()]
            except Exception:
                episode_successes = [bool(x) for x in successes]
        return logs, successes, e_obses, e_states

    plan_module.load_model = patched_load_model
    plan_module.PlanEvaluator.eval_actions = patched_eval_actions

    run_path = Path(run_dir).resolve()
    run_path.mkdir(parents=True, exist_ok=True)

    overrides = _build_variant_overrides(
        config=config,
        seed=seed,
        opt_steps=opt_steps,
        n_evals=n_evals,
    )
    if goal_source_override is not None:
        overrides["goal_source"] = str(goal_source_override)
    if goal_H_override is not None:
        overrides["goal_H"] = int(goal_H_override)
    if planner_max_iter_override is not None:
        overrides.setdefault("planner", {})
        overrides["planner"]["max_iter"] = int(planner_max_iter_override)
    if goal_file_path_override is not None:
        overrides["goal_file_path"] = str(Path(goal_file_path_override).resolve())
    cfg_dict = _merge_plan_cfg(plan_cfg_path, overrides)
    cfg_dict["saved_folder"] = str(run_path)
    cfg_dict["wandb_logging"] = False
    if str(overrides.get("goal_source", eval_cfg.get("goal_source", "random_state"))) == "file":
        goal_path = overrides.get("goal_file_path")
        if goal_path:
            cfg_dict["goal_file_path"] = _align_goal_file_to_n_evals(
                goal_file_path=str(goal_path),
                n_evals=int(n_evals),
                run_path=run_path,
            )
            overrides["goal_file_path"] = cfg_dict["goal_file_path"]

    metrics: Dict[str, Any] = {}
    trace: Dict[str, Any] = {}

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        with working_directory(run_path):
            logs = plan_module.planning_main(cfg_dict)
        elapsed = time.perf_counter() - t0

        peak_gpu_mem_mb = 0.0
        if torch.cuda.is_available():
            peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

        logs_path = run_path / "logs.json"
        success_rate = _extract_success_rate(logs, logs_path)
        if success_rate is None:
            raise RuntimeError(f"Could not extract success rate from {logs_path}")

        success_count = int(round(success_rate * n_evals))
        targets = variant_cfg.get("quantize_targets", [])
        if not targets:
            effective_quant_bits = 16
            quant_bits_desc = "fp16"
        else:
            unique_bits = sorted(set(applied_target_bits.values()))
            effective_quant_bits = unique_bits[0] if len(unique_bits) == 1 else -1
            quant_bits_desc = "_".join([f"{k}{applied_target_bits[k]}" for k in sorted(applied_target_bits.keys())])

        metrics = {
            "variant": variant_name,
            "budget_id": str(budget_id) if budget_id is not None else "default",
            "seed": int(seed),
            "opt_steps": int(opt_steps),
            "n_evals": int(n_evals),
            "goal_source": str(overrides.get("goal_source", eval_cfg.get("goal_source", "random_state"))),
            "goal_H": int(overrides.get("goal_H", eval_cfg.get("goal_H", 5))),
            "planner_max_iter": (
                int(overrides.get("planner", {}).get("max_iter"))
                if overrides.get("planner", {}).get("max_iter") is not None
                else None
            ),
            "goal_file_path": (
                str(overrides.get("goal_file_path"))
                if overrides.get("goal_file_path") is not None
                else None
            ),
            "success_count": success_count,
            "success_rate": float(success_rate),
            "avg_plan_time_seconds": float(elapsed / max(1, n_evals)),
            "peak_gpu_mem_mb": float(peak_gpu_mem_mb),
            "model_size_mb": float(model_size_mb or 0.0),
            "run_id": f"{variant_name}_opt{opt_steps}_seed{seed}",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_output_dir": str(run_path),
            "logs_path": str(logs_path),
            "quant_backend_effective": effective_backend,
            "quant_bits": int(effective_quant_bits),
            "quant_bits_desc": quant_bits_desc,
            "episode_success_count": (
                int(sum(1 for s in episode_successes if s))
                if episode_successes is not None
                else None
            ),
            "episode_successes_path": (
                str((run_path / "episode_outcomes.json").resolve())
                if episode_successes is not None
                else None
            ),
        }
        if episode_successes is not None:
            episode_rows = [
                {"episode_id": int(i), "success": bool(s)}
                for i, s in enumerate(episode_successes)
            ]
            (run_path / "episode_outcomes.json").write_text(
                json.dumps(episode_rows, indent=2), encoding="utf-8"
            )

        trace = {
            "variant_name": variant_name,
            "quant_backend_requested": quant_backend,
            "quant_backend_effective": effective_backend,
            "quant_bits_requested": int(requested_quant_bits),
            "quant_bits_effective": int(effective_quant_bits),
            "quant_bits_by_target": {k: int(v) for k, v in applied_target_bits.items()},
            "quant_bits_desc": quant_bits_desc,
            "budget_id": str(budget_id) if budget_id is not None else "default",
            "quantized_layer_paths": quantized_paths,
            "excluded_or_skipped_layer_paths": skipped_paths,
            "model_size_mb": float(model_size_mb or 0.0),
            "base_checkpoint": str(
                resolve_path(project_root, config["dino"]["ckpt_base_path"])
                / "outputs"
                / config["dino"]["model_name"]
                / "checkpoints"
                / f"model_{config['dino'].get('model_epoch', 'final')}.pth"
            ),
        }
    finally:
        plan_module.load_model = original_load_model
        plan_module.PlanEvaluator.eval_actions = original_eval_actions

    # Stabilize trace output for reproducibility and diffing.
    quantized_paths = sorted(set(quantized_paths))
    skipped_paths = sorted(set(skipped_paths))

    return metrics, trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one DINO-WM planning variant run.")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--opt-steps", type=int, required=True)
    parser.add_argument("--n-evals", type=int, required=True)
    parser.add_argument("--goal-source", default=None)
    parser.add_argument("--goal-file-path", default=None)
    parser.add_argument("--goal-H", type=int, default=None)
    parser.add_argument("--planner-max-iter", type=int, default=None)
    parser.add_argument("--budget-id", default=None)
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--trace-out", default=None)
    args = parser.parse_args()

    metrics, trace = run_variant_once(
        config_path=args.config,
        variant_name=args.variant,
        run_dir=args.run_dir,
        seed=args.seed,
        opt_steps=args.opt_steps,
        n_evals=args.n_evals,
        goal_source_override=args.goal_source,
        goal_file_path_override=args.goal_file_path,
        goal_H_override=args.goal_H,
        planner_max_iter_override=args.planner_max_iter,
        budget_id=args.budget_id,
    )

    if args.metrics_out:
        metrics_out = Path(args.metrics_out)
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with metrics_out.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    if args.trace_out:
        trace_out = Path(args.trace_out)
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        with trace_out.open("w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2)

    print(json.dumps({"metrics": metrics, "trace": trace}, indent=2))


if __name__ == "__main__":
    main()
