# tmpl

High-performance shape-based template matching experiments.

## Quick Start

```powershell
git clone git@github.com:friku/tmplmt.git
cd tmplmt
git lfs install
git lfs pull
poetry install
poetry run python scripts/eval_shape_matching.py --dataset pet_cap
# or
poetry run python scripts/eval_shape_matching.py --dataset kiban
```

The evaluation script prints per-frame accuracy metrics (pixel and angle error) and latency statistics so you can iterate on matcher parameters quickly. It also saves annotated overlays for each frame under `artifacts/match_viz/` so you can visually inspect detections. Use `--dataset`, `--data-root`, and `--template-name` to switch between bundled datasets, and adjust `--angle-step` plus the Canny thresholds to trade off speed versus robustness. Optional knobs such as `--downscale-factor`, `--refine-angle-window`, and `--refine-roi-scale` let you balance coarse localization cost versus accuracy. If the dataset CSV provides a `msec` column, the report will also show latency slack (measured – target). Large image assets are tracked with Git LFS, so run `git lfs install && git lfs pull` after cloning to retrieve them.

Angles are normalized to match the dataset convention (clockwise rotations are negative), so `err_deg` corresponds directly to the `theta` values stored in the CSV.
