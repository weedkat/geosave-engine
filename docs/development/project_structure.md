src/
├── geosave_engine
│   ├── api
│   │   ├── __init__.py
│   │   └── upload.py
│   ├── cli
│   │   ├── commands
│   │   │   ├── __init__.py
│   │   │   ├── artifact.py
│   │   │   ├── create.py
│   │   │   └── infra.py
│   │   ├── workspace
│   │   │   ├── __init__.py
│   │   │   ├── artifacts.py
│   │   │   ├── model.py
│   │   │   ├── scaffold.py
│   │   │   └── templates.py
│   │   ├── errors.py
│   │   └── main.py
│   ├── core
│   │   ├── __init__.py
│   │   └── resolver.py
│   ├── geodata
│   │   ├── ingestion
│   │   │   ├── base.py
│   │   │   ├── __init__.py
│   │   │   ├── manifest.py
│   │   │   └── sentinel2.py
│   │   ├── processing
│   │   │   ├── composting.py
│   │   │   └── masking.py
│   │   ├── stac_client
│   │   │   ├── base_client.py
│   │   │   ├── cdse_client.py
│   │   │   ├── element84_client.py
│   │   │   ├── __init__.py
│   │   │   └── planetary_client.py
│   │   └── stac_query
│   │       ├── cdse
│   │       │   ├── __init__.py
│   │       │   └── sentinel2.py
│   │       ├── element84
│   │       │   └── sentinel2.py
│   │       ├── planetary
│   │       │   └── sentinel2.py
│   │       └── base_query.py
│   ├── ml
│   │   ├── callbacks
│   │   │   ├── calibration.py
│   │   │   ├── __init__.py
│   │   │   ├── prediction_writer.py
│   │   │   └── training_monitor.py
│   │   ├── cli
│   │   │   ├── cli.py
│   │   │   └── __init__.py
│   │   ├── core
│   │   │   ├── base.py
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py
│   │   │   └── transform.py
│   │   ├── inference
│   │   │   ├── geo_predict.py
│   │   │   ├── __init__.py
│   │   │   └── sliding_window.py
│   │   ├── losses
│   │   │   ├── cross_entropy.py
│   │   │   ├── __init__.py
│   │   │   └── ohem.py
│   │   ├── models
│   │   │   ├── dpt
│   │   │   │   ├── backbone
│   │   │   │   │   ├── dinov2_layers
│   │   │   │   │   │   ├── attention.py
│   │   │   │   │   │   ├── block.py
│   │   │   │   │   │   ├── drop_path.py
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── layer_scale.py
│   │   │   │   │   │   ├── mlp.py
│   │   │   │   │   │   ├── patch_embed.py
│   │   │   │   │   │   └── swiglu_ffn.py
│   │   │   │   │   └── dinov2.py
│   │   │   │   ├── semseg
│   │   │   │   │   └── dpt.py
│   │   │   │   ├── util
│   │   │   │   │   └── blocks.py
│   │   │   │   ├── build.py
│   │   │   │   └── __init__.py
│   │   │   ├── smp
│   │   │   │   ├── build.py
│   │   │   │   └── __init__.py
│   │   │   └── __init__.py
│   │   └── optimizers
│   │       ├── adamw.py
│   │       └── __init__.py
│   ├── templates
│   │   ├── common
│   │   │   ├── .env
│   │   │   └── main.py
│   │   ├── object_detection
│   │   ├── pixelwise_regression
│   │   └── semantic_segmentation
│   ├── utils
│   │   ├── archives.py
│   │   ├── cql2.py
│   │   ├── datetime.py
│   │   ├── fs_ops.py
│   │   ├── geom.py
│   │   ├── __init__.py
│   │   ├── pretrained.py
│   │   ├── strings.py
│   │   ├── tiff.py
│   │   ├── torch_params.py
│   │   └── yaml_config.py
│   ├── __about__.py
│   └── __init__.py
