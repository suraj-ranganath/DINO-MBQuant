from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from experiments.quantize_utils import apply_int8_to_linears


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_predictor(config_path: str, device: torch.device) -> torch.nn.Module:
    cfg = load_yaml(config_path)
    project_root = Path(config_path).resolve().parent.parent
    dino_repo = (project_root / cfg["dino"]["repo_path"]).resolve()
    if str(dino_repo) not in sys.path:
        sys.path.insert(0, str(dino_repo))

    import plan as plan_module

    ckpt_base_path = cfg["dino"]["ckpt_base_path"]
    model_name = cfg["dino"]["model_name"]
    model_epoch = cfg["dino"].get("model_epoch", "latest")
    model_path = Path(ckpt_base_path) / "outputs" / model_name
    model_ckpt = model_path / "checkpoints" / f"model_{model_epoch}.pth"
    with (model_path / "hydra.yaml").open("r", encoding="utf-8") as f:
        model_cfg = OmegaConf.load(f)

    model = plan_module.load_model(
        model_ckpt=model_ckpt,
        train_cfg=model_cfg,
        num_action_repeat=model_cfg.num_action_repeat,
        device=device,
    )
    model.eval()
    return model.predictor.eval()


def bench_torch_module(
    module: torch.nn.Module,
    x: torch.Tensor,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = module(x)
        if x.device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = module(x)
        if x.device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - t0
    return {
        "total_seconds": float(elapsed),
        "per_iter_ms": float(1000.0 * elapsed / iters),
    }


def build_coreml_models(
    predictor_cpu: torch.nn.Module,
    example_cpu: torch.Tensor,
):
    import coremltools as ct
    from coremltools.optimize import coreml as cto

    traced = torch.jit.trace(predictor_cpu, example_cpu, strict=False)
    traced.eval()
    input_name = "x"
    ml_fp16 = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name=input_name, shape=example_cpu.shape)],
        compute_units=ct.ComputeUnit.ALL,
    )

    quant_cfg_int8 = cto.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int8",
        granularity="per_channel",
    )
    opt_cfg_int8 = cto.OptimizationConfig(global_config=quant_cfg_int8)
    ml_int8 = cto.linear_quantize_weights(ml_fp16, config=opt_cfg_int8)

    ml_int4 = None
    try:
        quant_cfg_int4 = cto.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int4",
            granularity="per_channel",
        )
        opt_cfg_int4 = cto.OptimizationConfig(global_config=quant_cfg_int4)
        ml_int4 = cto.linear_quantize_weights(ml_fp16, config=opt_cfg_int4)
    except Exception:
        ml_int4 = None

    return ml_fp16, ml_int8, ml_int4, input_name


def bench_coreml_model(
    mlmodel: Any,
    input_name: str,
    x_np: np.ndarray,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    out_name = mlmodel.get_spec().description.output[0].name
    for _ in range(warmup):
        _ = mlmodel.predict({input_name: x_np})[out_name]
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = mlmodel.predict({input_name: x_np})[out_name]
    elapsed = time.perf_counter() - t0
    return {
        "total_seconds": float(elapsed),
        "per_iter_ms": float(1000.0 * elapsed / iters),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DINO-WM predictor on Apple backend: MPS vs CoreML."
    )
    parser.add_argument("--config", default="configs/experiment_config.mac.yaml")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--out", default="results/mac_quant/benchmark_predictor_coreml.json")
    parser.add_argument("--save-models-dir", default="results/mac_quant/models")
    args = parser.parse_args()

    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; this benchmark targets Apple Silicon MPS.")

    os.environ.setdefault("DINO_WM_DEVICE", "mps")

    device_mps = torch.device("mps")
    predictor_mps = load_predictor(args.config, device=device_mps)
    token_dim = int(predictor_mps.pos_embedding.shape[1])
    emb_dim = int(predictor_mps.pos_embedding.shape[2])
    x_mps = torch.randn(
        args.batch_size,
        token_dim,
        emb_dim,
        device=device_mps,
        dtype=torch.float16,
    )

    fp16_stats = bench_torch_module(
        predictor_mps,
        x_mps,
        warmup=args.warmup,
        iters=args.iters,
    )

    predictor_fake_int8 = copy.deepcopy(predictor_mps).eval()
    _ = apply_int8_to_linears(
        predictor_fake_int8,
        backend="fake_int8",
        fallback_backend="fake_int8",
    )
    fake_stats = bench_torch_module(
        predictor_fake_int8,
        x_mps,
        warmup=args.warmup,
        iters=args.iters,
    )

    # CoreML path (true weight quantization on Apple runtime).
    predictor_cpu = copy.deepcopy(predictor_mps).to("cpu", dtype=torch.float32).eval()
    x_cpu = x_mps.detach().to("cpu", dtype=torch.float32)
    ml_fp16, ml_int8, ml_int4, input_name = build_coreml_models(predictor_cpu, x_cpu)
    x_np = x_cpu.numpy()
    coreml_fp16_stats = bench_coreml_model(
        ml_fp16,
        input_name=input_name,
        x_np=x_np,
        warmup=max(2, args.warmup // 2),
        iters=args.iters,
    )
    coreml_int8_stats = bench_coreml_model(
        ml_int8,
        input_name=input_name,
        x_np=x_np,
        warmup=max(2, args.warmup // 2),
        iters=args.iters,
    )
    coreml_int4_stats = None
    if ml_int4 is not None:
        coreml_int4_stats = bench_coreml_model(
            ml_int4,
            input_name=input_name,
            x_np=x_np,
            warmup=max(2, args.warmup // 2),
            iters=args.iters,
        )

    baseline = fp16_stats["per_iter_ms"]
    models_dir = Path(args.save_models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    fp16_pkg = models_dir / "predictor_fp16.mlpackage"
    int8_pkg = models_dir / "predictor_int8.mlpackage"
    if fp16_pkg.exists():
        import shutil
        shutil.rmtree(fp16_pkg)
    if int8_pkg.exists():
        import shutil
        shutil.rmtree(int8_pkg)
    ml_fp16.save(str(fp16_pkg))
    ml_int8.save(str(int8_pkg))
    int4_pkg = None
    if ml_int4 is not None:
        int4_pkg = models_dir / "predictor_int4.mlpackage"
        if int4_pkg.exists():
            import shutil
            shutil.rmtree(int4_pkg)
        ml_int4.save(str(int4_pkg))

    def _dir_size_mb(path: Path) -> float:
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total / (1024.0 * 1024.0)

    pkg_sizes = {
        "coreml_fp16_package_mb": _dir_size_mb(fp16_pkg),
        "coreml_int8_package_mb": _dir_size_mb(int8_pkg),
        "coreml_int4_package_mb": _dir_size_mb(int4_pkg) if int4_pkg is not None else None,
    }

    result = {
        "shape": {
            "batch_size": args.batch_size,
            "token_dim": token_dim,
            "emb_dim": emb_dim,
        },
        "runs": {
            "torch_mps_fp16": fp16_stats,
            "torch_mps_fake_int8": fake_stats,
            "coreml_fp16": coreml_fp16_stats,
            "coreml_int8": coreml_int8_stats,
            "coreml_int4": coreml_int4_stats,
        },
        "speedup_vs_torch_mps_fp16": {
            "torch_mps_fake_int8": baseline / fake_stats["per_iter_ms"],
            "coreml_fp16": baseline / coreml_fp16_stats["per_iter_ms"],
            "coreml_int8": baseline / coreml_int8_stats["per_iter_ms"],
            "coreml_int4": (
                baseline / coreml_int4_stats["per_iter_ms"]
                if coreml_int4_stats is not None
                else None
            ),
        },
        "coreml_model_package_sizes_mb": pkg_sizes,
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
