# Model and training interfaces

> **Status:** The ML layer is under redesign. Existing classes and configuration shapes may change.

## Scope

The ML package connects geospatial samples to PyTorch and PyTorch Lightning. It owns model contracts, encoders, decoders, heads, tasks, transforms, and training-time callbacks.

## Design areas still open

- `GeoStack` and `GeoSample` tensor conversion.
- Dataset and datastore responsibilities.
- Model context and stage composition.
- Task and DataModule interfaces.
- Prediction stitching and output writing.
- Stable configuration and registry shapes.

Training examples and configuration reference will be written after the Spatial and ML interfaces settle.
