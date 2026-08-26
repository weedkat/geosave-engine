# GeoPipeline

> **Status:** Pipeline, dataset, and datastore interfaces are under redesign. Treat current source as provisional.

## Intended responsibility

`GeoPipeline` coordinates geospatial sources, preprocessing, model context, and output for requested spatial anchors. It should reuse the same transformations for dataset creation and live inference without owning unrelated training or storage policy.

## Design areas still open

- Source and STAC ingestion contracts.
- Relationship between one anchor and multiple output samples.
- Persistence versus streaming interfaces.
- Error recovery and provenance ownership.
- Integration with strict `GeoStack` and `GeoSample`.

Concrete examples and method references will be restored after these interfaces settle.
