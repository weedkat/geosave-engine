src/
├── geosave_engine
│   ├── api
│   │   ├── __init__.py
│   │   └── upload.py
│   ├── cli
│   │   ├── docs
│   │   │   ├── generator.py
│   │   │   ├── __init__.py
│   │   │   ├── parser.py
│   │   │   └── rendering.py
│   │   ├── generate
│   │   │   ├── generator.py
│   │   │   ├── __init__.py
│   │   │   ├── request.py
│   │   │   └── scaffold.py
│   │   ├── io
│   │   │   ├── console.py
│   │   │   ├── __init__.py
│   │   │   └── prompter.py
│   │   ├── runtime
│   │   │   ├── arguments.py
│   │   │   ├── environment.py
│   │   │   ├── executor.py
│   │   │   ├── __init__.py
│   │   │   ├── runner.py
│   │   │   └── workspace.py
│   │   ├── search
│   │   │   ├── __init__.py
│   │   │   ├── library.py
│   │   │   └── project.py
│   │   ├── errors.py
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── paths.py
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
└── templates
    ├── plugins
    │   ├── notebook
    │   │   └── exploratory_data_analysis.ipynb
    │   └── scripts
    │       ├── dynamic_world_ingest
    │       └── sentinel2_l1c_ingest
    └── workspace
        ├── common
        │   ├── artifacts
        │   ├── data
        │   ├── main.py
        │   └── scripts
        ├── object_detection
        ├── pixelwise_regression
        └── semantic_segmentation
            ├── supervised
            └── unimatch_v2
