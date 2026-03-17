# Getting Started

## Requirements

- Python 3.9+
- Linux/macOS/Windows environment with PyTorch support

## Installation

From the project root:

```bash
pip install -r requirements.txt
pip install -e .
```

## Project Bootstrap

Create base folders for a new project workspace:

```bash
geomoka-scaffold --root .
```

Create starter configuration files:

```bash
geomoka-template --root . --config generic
```

Available config templates:

- `generic`
- `isprs_postdam`

## Dataset Layout

Expected segmentation dataset structure:

```text
dataset/
  <dataset-name>/
    images/
    labels/
    splits/
      train.csv
      val.csv
      test.csv
```

Use the split CLI to generate CSV splits:

```bash
geomoka-split --root dataset/<dataset-name>
```

## Build and Serve Documentation

Install docs dependencies:

```bash
pip install -e .[docs]
```

Run local docs server:

```bash
mkdocs serve
```

Build static docs:

```bash
mkdocs build
```
