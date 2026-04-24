# Commands

## geosave build

`geosave build` creates a workspace scaffold by selecting project metadata, task, and method.

It does not select or inject model choices into config files. Model selection is done by editing
`configs/default.yaml` (for example `model.model_config.dense_model.class_path`) and other model
settings directly.
