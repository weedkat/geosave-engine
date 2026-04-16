# TorchGeo Notes

This note explains two things:
1. how to split geospatial datasets
2. how to sample patches for DataLoader

Main references:
- https://torchgeo.readthedocs.io/en/latest/api/datasets.html
- https://torchgeo.readthedocs.io/en/latest/api/samplers.html

## Why splitting is different in geospatial data

In normal computer vision, each image is an item, so split is easy.

In geospatial data, one file can be a huge area and nearby pixels are highly correlated.
If train and val are spatially too close, metrics can look unrealistically good.

So TorchGeo split helpers split the geospatial index (bounding boxes), not simple row IDs.

## Dataset Split or Subset

Common arguments:
- dataset: GeoDataset to split
- fractions or lengths: how much goes to train val test
- generator: random seed control for reproducibility

### What is an "original bounding box"?

In TorchGeo, the dataset index contains many geospatial entries. Each entry has its own spatial extent (a bounding box).
That per-entry extent is what this document calls the "original bounding box."

Practical mental model:
- Dataset item in index: one geospatial record (often one raster file/tile footprint)
- Original bounding box: the full extent of that one indexed record
- Split helpers operate on these indexed extents, not on random pixels

So if your dataset has N indexed items, split helpers start from those N item-level bounding boxes.

### random_bbox_assignment

Mental model:
- For each indexed item bounding box, keep it whole
- Randomly assign each whole box to one split

Example:
- Original boxes: A B C D E
- Fractions: 0.6 0.2 0.2
- Result could be:
  - train: A C E
  - val: B
  - test: D

Good:
- very simple
- no cutting of original boxes

Limitation:
- if you have only a few large boxes, split can be coarse and imbalanced

Use when:
- each box is already a meaningful independent unit
- you want simple random assignment with minimal geometry operations

### random_bbox_splitting

Mental model:
- For each indexed item bounding box
- Randomly cut it (horizontal or vertical), potentially multiple times
- Assign resulting pieces to different splits

Example for one box A:
- A is split into A1 and A2
- A1 goes train, A2 goes val

This means one original area can contribute to multiple splits.

Good:
- can produce better fraction matching when boxes are large
- useful when dataset has a few giant extents

Limitation:
- higher leakage risk than assignment, because adjacent regions from the same original box can land in different splits

Use when:
- your index is very coarse and you must subdivide large extents
- you accept some adjacency across splits

### random_grid_cell_assignment

Mental model:
- Overlay a grid on each bounding box
- Treat each grid cell as a small spatial unit
- Randomly assign cells to splits

Good:
- finer-grained than bbox assignment
- usually better spatial coverage balance

Limitation:
- more knobs, especially grid_size

Use when:
- you want practical spatial stratification with controllable granularity
- this is often a strong default for geospatial train val test split

### roi_split

Mental model:

- You manually define region polygons
- each region becomes a split

Use when:

- split boundaries are known by geography or policy
- example: city A train, city B val, city C test

### time_series_split

Mental model:

- Split by time intervals, not by space

Use when:

- forecasting or temporal generalization matters
- example: train on 2018 to 2021, validate on 2022, test on 2023

## Which split should I pick

Quick rule of thumb:

- default: random_grid_cell_assignment
- simplest random baseline: random_bbox_assignment
- coarse boxes that need subdivision: random_bbox_splitting
- explicit geography: roi_split
- explicit temporal holdout: time_series_split

## Samplers for DataLoader

A GeoDataset is indexed by spatial query windows, so sampler choice controls which patches are read.

### RandomGeoSampler

What it does:

- randomly samples patch windows
- windows may overlap

Why use it:

- maximizes patch diversity for training

Tradeoff:

- not guaranteed to cover every pixel in one epoch

Best for:

- training

Important parameters:

- dataset
- size: patch size
- length: number of sampled patches per epoch
- roi and toi optional filters
- generator for reproducible randomness

### GridGeoSampler

What it does:

- scans patches in grid order over ROI
- deterministic full coverage

Why use it:

- stable and complete evaluation or prediction

Tradeoff:

- slower than random sampling

Best for:

- validation test prediction

Important parameters:

- dataset
- size: patch size
- stride: step between patches, set smaller than size for overlap
- roi toi units as needed

### PreChippedGeoSampler

What it does:

- samples already tiled files as units

Best for:

- datasets already prechipped on disk

## Practical recipe used in this project

- Split: random_grid_cell_assignment with fixed seed
- Train loader: RandomGeoSampler
- Val and test and predict loaders: GridGeoSampler

This gives:

- random patch diversity for optimization
- deterministic full-area coverage for evaluation and inference

