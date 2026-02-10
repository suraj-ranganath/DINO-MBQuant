from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantizeReport:
    backend: str
    quant_bits: int
    quantized_layer_paths: List[str]
    skipped_layer_paths: List[str]


class Int8WeightOnlyLinear(nn.Module):
    """Simple weight-only int8 linear fallback.

    This fallback is used when bitsandbytes is unavailable or conversion fails.
    It stores int8 weights with per-output-channel scales and dequantizes on forward.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_bits: int = 8,
    ):
        super().__init__()
        if quant_bits < 2 or quant_bits > 8:
            raise ValueError(f"quant_bits must be in [2, 8], got {quant_bits}")
        self.in_features = in_features
        self.out_features = out_features
        self.quant_bits = int(quant_bits)
        self.register_buffer("qweight", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scales", torch.empty(out_features, 1, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, layer: nn.Linear, quant_bits: int = 8) -> "Int8WeightOnlyLinear":
        out_features, in_features = layer.weight.shape
        new_layer = cls(
            in_features=in_features,
            out_features=out_features,
            bias=layer.bias is not None,
            quant_bits=quant_bits,
        )

        with torch.no_grad():
            weight = layer.weight.detach().float()
            max_q = float((1 << (int(quant_bits) - 1)) - 1)
            max_abs = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            scales = max_abs / max_q
            qweight = torch.round(weight / scales).clamp(-max_q, max_q).to(torch.int8)
            new_layer.qweight.copy_(qweight)
            new_layer.scales.copy_(scales)
            if layer.bias is not None:
                new_layer.bias.copy_(layer.bias.detach().float())

        device = layer.weight.device
        dtype = layer.weight.dtype
        new_layer = new_layer.to(device=device)
        if new_layer.bias is not None and dtype in (torch.float16, torch.bfloat16):
            new_layer.bias.data = new_layer.bias.data.to(dtype)
        return new_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.qweight.float() * self.scales
        weight = weight.to(device=x.device, dtype=x.dtype)
        bias = self.bias
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, bias)


def _path_matches(path: str, prefixes: Optional[Sequence[str]]) -> bool:
    if not prefixes:
        return True
    for prefix in prefixes:
        if path == prefix or path.startswith(prefix + "."):
            return True
    return False


def _should_quantize(
    path: str,
    include_paths: Optional[Sequence[str]],
    exclude_paths: Optional[Sequence[str]],
) -> bool:
    if include_paths and not _path_matches(path, include_paths):
        return False
    if exclude_paths and _path_matches(path, exclude_paths):
        return False
    return True


def _to_bnb_int8(linear: nn.Linear) -> nn.Module:
    import bitsandbytes as bnb  # type: ignore

    linear_cls = getattr(bnb.nn, "Linear8bitLt", None)
    if linear_cls is None:
        raise RuntimeError("bitsandbytes.nn.Linear8bitLt not found")

    kwargs = {}
    sig = inspect.signature(linear_cls)
    if "has_fp16_weights" in sig.parameters:
        kwargs["has_fp16_weights"] = False
    if "threshold" in sig.parameters:
        kwargs["threshold"] = 6.0

    quant = linear_cls(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        **kwargs,
    )
    quant = quant.to(device=linear.weight.device)
    quant.load_state_dict(linear.state_dict(), strict=False)
    return quant


def _quantize_linear(
    linear: nn.Linear,
    backend: str,
    fallback_backend: str,
    quant_bits: int = 8,
) -> Tuple[nn.Module, str]:
    if backend == "bitsandbytes" and int(quant_bits) == 8:
        try:
            return _to_bnb_int8(linear), "bitsandbytes"
        except Exception:
            if fallback_backend != "fake_int8":
                raise
    if fallback_backend == "fake_int8":
        return Int8WeightOnlyLinear.from_float(linear, quant_bits=int(quant_bits)), "fake_int8"
    raise RuntimeError(f"No supported fallback backend: {fallback_backend}")


def apply_int8_to_linears(
    module: nn.Module,
    include_paths: Optional[Sequence[str]] = None,
    exclude_paths: Optional[Sequence[str]] = None,
    backend: str = "bitsandbytes",
    fallback_backend: str = "fake_int8",
    quant_bits: int = 8,
) -> QuantizeReport:
    quantized: List[str] = []
    skipped: List[str] = []
    chosen_backend = backend

    def _recurse(parent: nn.Module, prefix: str = "") -> None:
        nonlocal chosen_backend
        for child_name, child in list(parent.named_children()):
            path = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                if _should_quantize(path, include_paths, exclude_paths):
                    quantized_child, used_backend = _quantize_linear(
                        child,
                        backend=backend,
                        fallback_backend=fallback_backend,
                        quant_bits=int(quant_bits),
                    )
                    chosen_backend = used_backend if used_backend != backend else chosen_backend
                    setattr(parent, child_name, quantized_child)
                    quantized.append(path)
                else:
                    skipped.append(path)
            else:
                _recurse(child, path)

    _recurse(module)
    return QuantizeReport(
        backend=chosen_backend,
        quant_bits=int(quant_bits),
        quantized_layer_paths=quantized,
        skipped_layer_paths=skipped,
    )


def estimate_module_storage_mb(module: nn.Module) -> float:
    seen = set()
    total_bytes = 0

    def _tensor_key(tensor: torch.Tensor) -> tuple[int, int, str, str]:
        return (tensor.data_ptr(), tensor.numel(), str(tensor.dtype), str(tensor.device))

    # Account low-bit layers with effective bitwidth, then mark their tensors as seen.
    for submodule in module.modules():
        if isinstance(submodule, Int8WeightOnlyLinear):
            q_bits = int(getattr(submodule, "quant_bits", 8))
            # Effective compressed weight size (bit-packed idealized accounting).
            total_bytes += int(round(submodule.qweight.numel() * q_bits / 8.0))
            total_bytes += submodule.scales.numel() * submodule.scales.element_size()
            seen.add(_tensor_key(submodule.qweight))
            seen.add(_tensor_key(submodule.scales))
            if submodule.bias is not None:
                total_bytes += submodule.bias.numel() * submodule.bias.element_size()
                seen.add(_tensor_key(submodule.bias))

    for tensor in list(module.parameters()) + list(module.buffers()):
        if tensor is None:
            continue
        key = _tensor_key(tensor)
        if key in seen:
            continue
        seen.add(key)
        total_bytes += tensor.numel() * tensor.element_size()

    return total_bytes / (1024.0 * 1024.0)


def estimate_payload_size_mb(payload: Dict[str, object]) -> float:
    total = 0.0
    for value in payload.values():
        if isinstance(value, nn.Module):
            total += estimate_module_storage_mb(value)
    return total
