# Training Workflow

## 1) Prepare Configuration

Training is driven by YAML config files under `config/`.

Typical workflow:

1. Generate template configs with `geomoka-template`
2. Update dataset paths and split CSV paths
3. Select model architecture and backbone
4. Tune training hyperparameters

## 2) Train Model

Run training with a config file:

```bash
geomoka-train --config config/isprs_postdam/train_dpt.yaml --save_dir output
```

`--save_dir` stores checkpoints and logs.

## 3) Validate and Evaluate

Validation/test behavior depends on the trainer/evaluation settings inside your config.

Evaluation utilities are implemented in:

- `src/geomoka/eval/evaluate.py`
- `src/geomoka/eval/metrics.py`

## 4) Confidence Threshold Calibration

Use the calibration CLI to tune prediction confidence threshold on validation data:

```bash
python -m geomoka.cli.calibrate \
  --config config/isprs_postdam/train_dpt.yaml \
  --checkpoint output/<run>/best_model.pth \
  --mode sliding_window \
  --output output/<run>/confidence_calibration.png
```

This command saves:

- Calibration curve image
- JSON summary with best threshold and mIoU curve
