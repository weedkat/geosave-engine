# Create a workspace

```bash
geosave create demo --workspace segmentation --description "Land cover experiment"
cd demo
python main.py fit --config configs/train.yaml
```

Choose `segmentation`, `custom_lightning`, or `blank`. Without `--workspace`, the
CLI prompts for a starter. `-w` is the short option; `-d` sets the description.
The old task/method selectors are replaced by workspace selection.

Segmentation imports the library's supervised Lightning Module and DataModule.
Custom Lightning supplies editable `modules/task.py` and `modules/data.py` with
a small deterministic example. Blank supplies only shared startup and empty
working directories. The starter is an initial choice: configs select the
training classes, so a workspace can evolve without regeneration.

```text
demo/
├── artifacts/
├── configs/
├── data/
├── logs/
├── modules/
├── notebooks/
├── predictions/
├── scripts/
├── .env
├── geosave.toml
└── main.py
```

`geosave.toml` records project information and `workspace.template`, rather than
binding the project to one training task and method. Runtime processing remains
in the library; generated files compose its public APIs.

## Add an optional scaffold

```bash
geosave make scripts prepare_dense_data.py
python scripts/prepare_dense_data.py --help
```

Scaffolds are optional complete files copied into an existing workspace. The
first argument names the destination category, the second its file. Existing
collision prompts let you overwrite or skip files. The prepare script loads
environment settings, configures GDAL, and calls the public preparation flow.
The corresponding `geosave workflow prepare-dense-data` command is also available.

## Maintain the bundled starters

```text
src/geosave_engine/templates/
├── common/                         # main.py and .env for every workspace
├── workspaces/
│   ├── segmentation/               # configs, README, example script
│   └── custom_lightning/           # configs and editable Lightning modules
└── scaffolds/
    └── scripts/prepare_dense_data.py
```

Generation copies common first, then the chosen workspace starter. Optional
scaffolds are installed only through make. Paths within a starter or scaffold
category match their destination paths. Selection descriptions and Python caches
are excluded from generated workspaces.

Edit this source tree when improving future generated projects. A user's
existing `workspace/` remains their editable consumer project.
