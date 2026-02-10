Release bundle contents:
- paper.pdf
- supplemental.zip (optional)
- this README

Mac demo:
1) Install Python dependencies from requirements.txt
2) Ensure demo/demo_artifacts is present
3) Run: streamlit run demo/app.py
4) Mac sprint pipeline: bash scripts/run_mac_pipeline.sh configs/experiment_config.mac.yaml

GPU reproducibility (A6000/A100):
1) Set ckpt_base_path in configs/experiment_config.yaml
2) Build variants: python -m experiments.build_variants --config configs/experiment_config.yaml
3) Run grid: python -m experiments.run_wall_grid --config configs/experiment_config.yaml
4) Aggregate: python -m experiments.aggregate --config configs/experiment_config.yaml
5) Figures: python -m experiments.make_figures --config configs/experiment_config.yaml
