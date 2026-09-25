# YOLO11 + Slim-BiFPN — Fabric Defect Detection

Custom Ultralytics YOLO11 fork for woven-fabric defect detection, with a lightweight **Slim-BiFPN neck** that detects on **4 scales (P2–P5)** for small defects.

Based on `ultralytics 8.4.163` (AGPL-3.0 — see `LICENSE`).

## What was customized

| Change | File | Notes |
|---|---|---|
| `DepthwiseSeparableConv` module | `ultralytics/nn/modules/slim_bifpn.py` | Depthwise + pointwise conv, BN + SiLU. ~8× fewer params than a standard conv. |
| Slim-BiFPN model config | `ultralytics/cfg/models/11/yolo11_slim_bifpn.yaml` | YOLO11n backbone (unchanged) + 4-scale BiFPN head. 208 layers, **1.9M params**, 7.6 GFLOPs |
| Module export | `ultralytics/nn/modules/__init__.py` | Import + `__all__` entry |
| `parse_model` registration | `ultralytics/nn/tasks.py` | Import + `base_modules` entry, so yaml args are channel-scaled correctly |

Head structure: 1×1 lateral convs → top-down path (upsample + concat + reduce + DSC) → bottom-up path (downsample + concat + reduce + DSC) → `Detect` on P2/P3/P4/P5 (strides 4/8/16/32).

## Usage

```python
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/11/yolo11_slim_bifpn.yaml')
model.train(data='fabric.yaml', epochs=100, imgsz=640)
```

> Note: `nc: 3` is hardcoded in the yaml — change it to your class count, or pass `nc=` at build time.

## Kaggle

```bash
%%capture
!git clone --depth 1 https://github.com/fahadcn/ultralytics-fabric.git
%cd ultralytics-fabric
!pip install -e .
```

Install **after** any other `pip install ultralytics` in the notebook — otherwise the PyPI package overwrites this fork and `yolo11_slim_bifpn.yaml` won't exist.

## Adding new custom layers (checklist)

1. Define the module **once** in `ultralytics/nn/modules/<file>.py` (with its own imports).
2. Export it from `ultralytics/nn/modules/__init__.py` (import + `__all__`).
3. Register it in `ultralytics/nn/tasks.py`: add to the `ultralytics.nn.modules` import list, and to `base_modules` in `parse_model` (or `repeat_modules` if it takes a repeat count). Skipping this causes raw yaml args to be passed → channel mismatch errors.
4. Use the exact class name in the yaml. First yaml arg = output channels (auto-scaled by width multiple).

Verify in a **fresh** Python process (notebook kernels keep stale classes loaded):

```bash
python -c "from ultralytics import YOLO; YOLO('your.yaml').info()"
```

Check the `(in_ch, out_ch)` columns in the summary — wrong channel counts there mean step 3 is misconfigured.

## Repo layout notes

- `origin` = this fork. `upstream` = `ultralytics/ultralytics` (pull to sync with upstream releases).
- Upstream's docs/tests/examples/CI files were removed — only the package, build config, and license remain.
