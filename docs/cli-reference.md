# CLI Reference

## `geomoka-train`

Train a segmentation model from a YAML config.

```bash
geomoka-train --config <path/to/config.yaml> [--save_dir output]
```

Arguments:

- `--config` (required): config file path
- `--save_dir` (optional): directory for checkpoints/logs (default: `output`)

## `geomoka-split`

Generate train/val/test split CSVs from image and label folders.

```bash
geomoka-split --root <dataset_root> [--out_dir <splits_dir>] [--train_val_test 0.7 0.15 0.15]
```

Expected under `--root`:

- `images/`
- `labels/`

Outputs:

- `train.csv`
- `val.csv`
- `test.csv`
- `unlabeled.csv` (only when unmatched images exist)

## `geomoka-scaffold`

Create minimal project directories and copy starter script/test files.

```bash
geomoka-scaffold [--root <project_root>]
```

## `geomoka-template`

Copy template config files into your project.

```bash
geomoka-template [--root <project_root>] [--config generic|isprs_postdam] [--force]
```

Arguments:

- `--config`: template set name (`generic` by default)
- `--force`: overwrite existing files

## `geomoka.cli.calibrate` (module entry)

Calibrate confidence threshold to maximize mIoU on validation set.

```bash
python -m geomoka.cli.calibrate --config <config.yaml> --checkpoint <model.pth>
```
