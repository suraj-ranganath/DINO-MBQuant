# Demo Viewer

## Launch

```bash
streamlit run demo/app.py
```

## Requirements

- `demo/demo_artifacts/manifest.json` must exist.
- `demo/demo_artifacts/<variant>/episode_<k>.mp4` files must be present.

Generate artifacts from completed Wall runs:

```bash
python -m experiments.export_demo_artifacts --config configs/experiment_config.yaml
```
