src/
├── geosave_engine
│   ├── __about__.py
│   ├── __init__.py
│   ├── cli
│   │   ├── commands
│   │   │   ├── __init__.py
│   │   │   ├── create.py
│   │   │   ├── infra.py
│   │   │   └── upload.py
│   │   ├── workspace
│   │   │   ├── __init__.py
│   │   │   ├── artifact.py
│   │   │   ├── model.py
│   │   │   ├── scaffold.py
│   │   │   └── templates.py
│   │   ├── errors.py
│   │   └── main.py
│   ├── geodata
│   │   ├── datasets
│   │   │   ├── __init__.py
│   │   │   ├── geo_dataset.py
│   │   │   ├── non_geo_dataset.py
│   │   │   └── samplers.py
│   │   ├── errors
│   │   │   ├── __init__.py
│   │   │   └── errors.py
│   │   ├── features
│   │   │   ├── __init__.py
│   │   │   ├── cloud_mask.py
│   │   │   ├── ndvi.py
│   │   │   └── shadow_mask.py
│   │   ├── pipeline
│   │   │   ├── __init__.py
│   │   │   ├── anchor_sources.py
│   │   │   └── geo_pipeline.py
│   │   ├── sensors
│   │   │   ├── __init__.py
│   │   │   ├── sensors.py
│   │   │   └── sensors.yaml
│   │   ├── stac
│   │   │   ├── __init__.py
│   │   │   ├── client.py
│   │   │   ├── query.py
│   │   │   └── source.py
│   │   ├── tile
│   │   │   ├── __init__.py
│   │   │   ├── geoanchor.py
│   │   │   ├── geostack.py
│   │   │   └── geotile.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── archives.py
│   │       ├── crs.py
│   │       ├── datetime.py
│   │       ├── geodata.py
│   │       ├── geolocator.py
│   │       ├── geovis.py
│   │       └── stac_query.py
│   ├── infra
│   │   ├── docker-compose.yml
│   │   └── .env.example
│   ├── ml
│   │   ├── callbacks
│   │   │   ├── __init__.py
│   │   │   ├── prediction_logger.py
│   │   │   ├── prediction_writer.py
│   │   │   └── threshold_calibrator.py
│   │   ├── cli
│   │   │   ├── __init__.py
│   │   │   └── cli.py
│   │   ├── inference
│   │   │   ├── protocol.py
│   │   │   ├── sliding_window.py
│   │   │   └── thresholding.py
│   │   ├── loss
│   │   │   ├── __init__.py
│   │   │   └── ohem.py
│   │   ├── metrics
│   │   │   ├── __init__.py
│   │   │   └── semantic_segmentation.py
│   │   ├── models
│   │   │   ├── __init__.py
│   │   │   ├── contract
│   │   │   │   ├── __init__.py
│   │   │   │   ├── chain.py
│   │   │   │   ├── context.py
│   │   │   │   └── normalization.py
│   │   │   ├── decoder
│   │   │   │   ├── __init__.py
│   │   │   │   ├── dpt.py
│   │   │   │   └── unet.py
│   │   │   ├── encoder
│   │   │   │   ├── __init__.py
│   │   │   │   ├── clay.py
│   │   │   │   ├── dinov3.py
│   │   │   │   └── prithvi.py
│   │   │   ├── head
│   │   │   │   ├── __init__.py
│   │   │   │   └── dense.py
│   │   │   └── monolith
│   │   │       ├── __init__.py
│   │   │       └── ibm_granite_biomass.py
│   │   ├── optimizer
│   │   │   ├── __init__.py
│   │   │   ├── adagrad.py
│   │   │   ├── adam.py
│   │   │   ├── adamw.py
│   │   │   ├── rmsprop.py
│   │   │   └── sgd.py
│   │   ├── registry
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── loss.py
│   │   │   ├── model.py
│   │   │   ├── optimizer.py
│   │   │   └── scheduler.py
│   │   ├── tasks
│   │   │   ├── __init__.py
│   │   │   └── semantic_segmentation.py
│   │   ├── transforms
│   │   │   ├── __init__.py
│   │   │   ├── augmenter.py
│   │   │   └── processor.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── torch_params.py
│   │       └── weights.py
│   ├── templates
│   │   ├── common
│   │   │   ├── .env
│   │   │   └── main.py
│   │   └── semantic_segmentation
│   │       └── supervised
│   │           ├── configs
│   │           │   ├── augmentation.yaml
│   │           │   ├── metadata.yaml
│   │           │   └── model.yaml
│   │           └── modules
│   │               └── data_pipeline.py
│   └── utils
│       ├── __init__.py
│       ├── colorize.py
│       ├── file_ops.py
│       └── fn.py
