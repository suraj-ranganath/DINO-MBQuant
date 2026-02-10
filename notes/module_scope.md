# DINO-WM Module Scope for Quantization

Source inspected: `third_party_dino_wm/models/visual_world_model.py`

Top-level world model modules relevant to this project:
- `wm.encoder`: feature encoder branch (preserved FP16 in mixed precision setting)
- `wm.predictor`: latent dynamics/predictive branch (INT8 in mixed and uniform settings)

Checkpoint loading path (from `third_party_dino_wm/plan.py`):
- keys in checkpoint payload include `encoder`, `predictor`, `decoder`, `proprio_encoder`, `action_encoder`
- planning model is instantiated from these keys in `load_model(...)`

Exact quantized linear layer paths are emitted automatically at run time to:
- `results/variants/<variant>/variant_spec.json`
