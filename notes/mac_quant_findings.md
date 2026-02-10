# Mac Quantization Findings (M4, Apple Silicon)

## Key takeaway

PyTorch MPS + fake INT8 does **not** provide real speedup.  
A true Apple-native path via CoreML gives measurable acceleration and real quantized model compression.

## Benchmark setup

- Model component: DINO-WM predictor (`ViTPredictor`)
- Shape: batch=512, tokens=196, emb=404
- Device/runtime comparisons:
  - `torch_mps_fp16`
  - `torch_mps_fake_int8` (current fallback path)
  - `coreml_fp16`
  - `coreml_int8` (true weight quantization in CoreML)

Source: `results/mac_quant/benchmark_predictor_coreml_bs512_final.json`

## Main numbers

- `torch_mps_fp16`: **1605.66 ms / forward**
- `torch_mps_fake_int8`: **1608.07 ms / forward** (no gain)
- `coreml_fp16`: **896.72 ms / forward** (**1.79x** vs torch MPS fp16)
- `coreml_int8`: **913.85 ms / forward** (**1.76x** vs torch MPS fp16)

CoreML package sizes:
- FP16 package: **38.14 MB**
- INT8 package: **19.21 MB** (about **49.6% smaller**)

## Figure

- `figures/mac_predictor_latency.pdf`

## Recommended paper framing

1. Quantization claim:
   - "On Apple Silicon, true CoreML INT8 quantization halves predictor package size while preserving planning success in our Wall setup."
2. Speed claim:
   - "Apple-native runtime (CoreML) yields 1.7-1.8x predictor latency speedup vs current PyTorch MPS path."
3. Honest caveat:
   - "In this benchmark, CoreML INT8 does not outperform CoreML FP16 in latency; the speedup comes from runtime/backend execution rather than bit-width alone."

