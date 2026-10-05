# Library structure

Status: first iteration implemented and verified, 2026-10-04.
The user requested implementation after the design discussion. Training/prediction
fundamentals and raster EDA below remain future iterations.

Resolved naming: common/ keeps shared startup; workspaces/ contains segmentation
and custom_lightning starters; scaffolds/ contains optional files installed by
make. Workspace creation selects --workspace/-w rather than task/method flags.

Development verification uses the editable source installation. Local wheel
builds are not required for this refactor; the existing GitHub release workflow
owns packaging and PyPI publication for v-prefixed tags. An extra wheel smoke
was performed before this preference was clarified, and is recorded below only
as completed history.

## Caller view

```python
from geosave_engine import geodata as gs
from geosave_engine.geodata.benchmarks import dynamic_world
from geosave_engine.ml.segmentation import supervised
from geosave_engine.model.spec import ModelSpec
from geosave_engine.workflow.flows import prepare_dense_data

# Original publisher files, extracted locally; returns pathlib.Path.
labels = dynamic_world.download("data/dynamic_world", subset="experts")

# Existing interfaces remain available.
spec = ModelSpec.load("configs/model_spec.yaml")
scene = gs.read_raster("scene.tif")
# supervised.Module and supervised.DataModule remain native Lightning classes.
```

Downloaded labels are not a ready-to-train paired imagery dataset. The first
benchmark module downloads original files and metadata only. Imagery acquisition,
catalog conversion, splitting, and training are separate operations.

## Intent and agreed constraints

Make the package layout express ownership before adding raster EDA or more
training methods. Geodata owns native geospatial data and operations; model owns
what a release needs; ml owns training. Keep CLI and workflow responsibilities
visible, and document the resulting imports and behavior.

User decisions carried into this draft:

- Benchmark access initially means downloading and extracting original files.
- Training datasets follow their task and method, including future methods that
  consume unlabeled data. They do not move wholesale into model.
- Kornia already supplies random crop. No new RandomSampler or separate public
  RandomDataset/EvaluationDataset is required just to select a spatial crop.
- Validation and test should reconstruct scenes, as real tiled inference does.
- Output interpretation belongs to the model's head; its concrete interface
  remains to be designed.
- A future task-based fit API is a direction, not part of this refactor.
- Preserve native objects, lazy raster operations, and unrelated working changes.
  Alpha import changes need no compatibility aliases.

## Maintainer view

```text
src/geosave_engine/
├── cli/                          # arguments and workspace generation
│   ├── commands/
│   └── core/copy.py              # workspace copying, moved from root utils
├── geodata/                      # no torch imports
│   ├── core/                     # native-object factories and accessors
│   ├── attrs/
│   │   └── palette.py            # palette type and RGB parsing
│   ├── io/                       # move from utils/io
│   ├── benchmarks/
│   │   └── dynamic_world.py      # original-file download and extraction
│   ├── transform/                # includes Tiles and TileMerger
│   ├── features/  sensors/  stac/
│   ├── viz/                      # current plotting; explorer added later
│   └── utils/                    # internal geodata helpers only
├── model/
│   ├── chain/  encoder/  decoder/  head/  monolith/
│   ├── factory.py  registry.py
│   └── spec/  release/
├── ml/
│   ├── cli.py                    # native LightningCLI entry point
│   ├── builders/                 # reusable construction helpers
│   ├── datasets/                 # existing tile reader; ownership review later
│   ├── criterion/  transforms/   # retain until task-locality review
│   └── segmentation/
│       ├── callbacks.py          # segmentation prediction logger
│       ├── metrics.py  calibrate.py  transforms.py
│       └── supervised/
│           └── module.py  data.py
├── workflow/
│   └── configs/  flows/  tasks/
└── templates/                    # source for generated consumers
```

There is no root utils package in the implemented first iteration. Internal helpers may stay
under a domain's utils package; public format I/O deserves its own named module.
Do not create empty regression, detection, semi-supervised, or explorer packages
to fill out this tree.

Dependency direction stays `geodata <- model <- ml`. Workflow composes geodata
and torch-free model.spec. CLI invokes library operations and complete workflows;
it does not own geospatial processing or training policy. A benchmark download
can be called directly from Python without a Prefect flow or a new CLI command.

## First iteration: cleanup, benchmarks, and matching docs

| Previous implementation | New home | Reason |
| --- | --- | --- |
| utils/file_ops.py | cli/core/copy.py | Only workspace generation uses safe_copy. |
| utils/sentinel.py | Private definition in geodata/utils/gdal_env.py | Only GDAL configuration uses the sentinel. |
| utils/colorize.py Palette and parse_color | geodata/attrs/palette.py | Legend, raster accessor, I/O, and plotting share palette semantics. |
| utils/colorize.py tensor rendering | Private helper in ml/segmentation/callbacks.py | The current consumer renders segmentation labels and argmax logits. |
| ml/callbacks/prediction_logger.py | ml/segmentation/callbacks.py | Its label and class contract is segmentation-specific. |
| geodata/utils/io | geodata/io | Public format readers and writers are discoverable domain operations. |
| workspace/scripts/download.py download behavior | geodata/benchmarks/dynamic_world.py | Reusable data access belongs in the library. |

Preserve safe_copy collision handling, palette behavior, callback behavior, and
I/O encoding. Move mirrored tests and update imports and template class paths.
The existing DensePredictionLogger class name can remain during this move;
renaming it is not needed to establish ownership.

Keep the existing geodata read_raster/read_stack/read_vector imports. Moving the
implementation of geodata.io changes deep imports, not those entry points.
The generated workspace remains a consumer; update template source rather than
quietly rewriting the user's workspace script.

### Benchmark interface and behavior

```python
def download(
    root: str | Path,
    *,
    subset: Literal["experts", "non_expert", "validation", "test"] = "experts",
) -> Path:
    """Download and extract the original Dynamic World files for one subset."""
```

Use a small explicit file mapping for this benchmark, not a general registry or
plugin framework. Use publisher sources rather than the workspace's Drive
mirrors. Preserve archive contents, label values, and publisher metadata.

Return a subset-specific directory under root. Each directory contains the
original archive layout and applicable accompanying metadata; do not rename its
members or reinterpret validation labels and test predictions as one target.
Document those distinctions in the benchmark guide.

Download to a temporary file, validate the archive, and extract to a temporary
directory before marking the subset complete. Repeated calls reuse only a
completed result. Partial files or extraction failures must not return success.
Prevent archive members from escaping the extraction directory. Errors propagate
to callers instead of being reduced to a printed message and bool.

Use existing dependencies and native archive capabilities. Keep download details
private in this module until another benchmark demonstrates actual repetition.
Checksum validation should use publisher checksums where supplied. A completion
record describes successful extraction; it does not claim to detect arbitrary
later user edits or deletions.

No automatic imagery download, GeoDataFrame catalog conversion, class remapping,
training split generation, or remote streaming in the first version.

Publisher references to verify during implementation:

- Training and validation labels: https://doi.pangaea.de/10.1594/PANGAEA.933475
- Test release: https://zenodo.org/records/4766508

### Documentation in this iteration

Add a package ownership/import guide and a benchmark download guide. Update the
model README so its model/release explanation and Lightning examples have clear
ownership. Update affected workflow examples and template imports. Expand
Zensical navigation and its segmentation-only site description to match current
library scope. Document import breaks without describing proposed future APIs as
available. API examples must run against the completed implementation.

### Template structure: workspace starters

User clarification: templates are for workspaces, and the current tasks category
should become workspaces. The organizing unit is a workspace starter. The prior
rename-only proposal and the task-neutral-scaffold recommendation are withdrawn
as recommendations. On reviewing whether common should be removed, retain the
existing shared startup: independent workspace presets do not require duplicate
entry points. The earlier recommendation to remove common was premature.

Implemented template layout:

```text
templates/
├── common/
│   ├── main.py
│   └── .env
└── workspaces/
    ├── segmentation/
    │   ├── README.md
    │   ├── configs/
    │   │   ├── train.yaml
    │   │   └── model_spec.yaml
    │   └── scripts/prepare_example.py
    └── custom_lightning/
        ├── README.md
        ├── configs/train.yaml
        └── modules/
            ├── task.py
            └── data.py
```

Each workspaces leaf contains the files specific to one starter, with paths
relative to the generated workspace root. Its README explains that project.
Common supplies startup shared by the current starters. A starter supplies an
initial configuration, not a permanent restriction on task or training method. Additional
training configs can be authored in the same workspace. Do not create empty
semi-supervised, regression, or detection starters in advance.

Maintainer composition is shared startup plus an optional selected workspace:

```python
# Shared startup plus a selected workspace starter.
def create_workspace(root: Path, workspace: str | None = None) -> None:
    # Make the usual empty output directories.
    safe_copy(COMMON_DIR, root, exclude=_EXCLUDE)
    if workspace is not None:
        safe_copy(WORKSPACES_DIR / workspace, root, exclude=_EXCLUDE)
```

The existing common directory contains only main.py and .env. Both current
starters use the same native LightningCLI startup. Retaining common keeps CLI
settings and environment defaults in one place, including for a bare workspace.
Two direct copies need no inheritance engine, manifest, or merging system.

Removing common would be reasonable if independent starters needed different
startup behavior or had to be distributed as self-contained directories. Neither
requirement is established here. A smaller source tree alone does not justify
duplicating shared startup. If actual divergence appears later, move only the
diverging file into its workspace preset rather than adding conditional rules to
the shared entry point.

The current semantic_segmentation/supervised starter becomes segmentation; its
initial config still imports ml.segmentation.supervised. custom/lightning becomes
custom_lightning because it is an editable Lightning project rather than a
learning method. The implementation uses these starter names and the --workspace selector. Model and DataModule selection still lives in training config class paths.

Optional complete files live in templates/scaffolds, grouped by destination
category. The existing prepare_dense_data.py script is retained because it also
loads environment settings and configures GDAL. make preserves its category/file
selection and collision handling. A benchmark download example lives in the
guide; no new generated download wrapper is needed.

create now selects --workspace/-w from segmentation, custom_lightning, or blank.
Blank copies common only. geosave.toml records workspace.template; training
selection remains in the chosen config's class paths. Old task/method CLI options
and old import paths have no compatibility aliases.

Fresh smoke on 2026-10-04: one bare workspace with an unmodified main.py parsed
both library segmentation and custom Lightning configs through --print_config,
without geosave.toml task metadata. This confirms that a generated workspace need
not remain tied to its initial training choice. An earlier smoke verified current
create and optional-file make behavior. Neither smoke implements the proposed
starter moves. Exercise generated blank, segmentation, and custom starters and
editable-source resource access before settling implementation.

Installed-wheel generation verifies shared startup, both starters, and optional
scaffold resources independently of the source checkout.

## Second iteration: training and prediction fundamentals

Retain method-owned Dataset, Module, and DataModule. Review the existing
ml/datasets/read_inputs helper with actual training and development-prediction
callers before moving or splitting it. No model/datasets package is proposed.

Separate three decisions currently coupled through spec.tiles:

1. Model input requirements and deterministic preparation.
2. Training source-window size and Kornia crop/augmentation.
3. Exhaustive evaluation/prediction tiling and scene assembly.

For example, reading a 256-pixel training window allows Kornia to choose a
224-pixel crop. Reading a 224-pixel tile and cropping to 224 cannot change its
spatial position. The new DataModule configuration for source-window size is
unsettled; do not introduce it in a file-move refactor. Spatially varying model
inputs such as location coordinates must remain consistent with the cropped
image, and temporal/context fields must stay paired with their source frame.

Scene evaluation follows this order:

```text
scene -> deterministic preparation -> exhaustive tiles -> tensor transforms
      -> raw model output -> scene assembly -> head interpretation -> metrics
```

For segmentation, merge logits before deriving labels; whole-scene loss uses
merged logits. Decide explicitly whether production-calibrated labels also feed
reported metrics. Preserve Tiles/TileMerger for reusable raster geometry.
Regression assembles continuous raster values. Detection needs spatial box
mapping and duplicate suppression, not raster averaging. Do not make all heads
use one merger or assume every ModelChain has an attribute named head.

Calibrated thresholds needed by production interpretation must survive native
model save/load. The current Lightning-module-owned thresholds do not establish
that release contract. Define the head interface and ownership using a native
release round-trip smoke before implementing new training head types.

## Third iteration: raster EDA, then additional training capabilities

Build the HoloViz/Panel explorer in geodata.viz around native xarray objects.
It should expose attributes, variable selection, spatial display, and time
scrubbing. Keep metadata display lazy; select a frame before requesting pixels;
compute statistics on demand. Resolve CRS handling and Bokeh initialization in
the explorer design. Preserve existing static plotting where useful.

Add regression and detection training only after their output assembly and
release contracts are exercised. Add method packages when implemented, with
paired Lightning Module/DataModule and task-local helpers. Defer a fit facade.

## Verification and review points

Fresh baseline on 2026-10-04: legend, GDAL configuration, workspace generation,
prediction callback, and plotting suites: 38 passed, 2 deselected, 4 warnings.
No source code was changed for this baseline.

First-iteration acceptance checks:

- Focused affected suites preserve behavior; tests mirror new source paths.
- Fresh-process layering checks preserve torch-free geodata/model.spec imports
  and the absence of model-to-ml dependencies, including new benchmark imports.
- If I/O moves, persistence round trips and lazy reader tests pass after the move.
- Local controlled download tests cover first use, reuse, transfer failure,
  corrupt archive, interrupted extraction, and an escaping archive member.
- A small real publisher download smoke verifies URLs and metadata delivery;
  the large test archive is not required in the regular test suite.
- Template generation, new documentation examples, Zensical build, scoped lint,
  and diff whitespace checks pass.
- Editable-source workspace generation and entry-point checks verify the
  development behavior. The existing release workflow handles wheel builds.

The implemented first iteration combines public I/O relocation, root utils
cleanup, benchmark access, workspace/scaffold naming, and matching documentation.
Fresh affected suites: 445 passed, 22 deselected. Native publisher validation
smoke: 5,794 original files extracted, both metadata spreadsheets present,
README downloaded, and a second call reused the completion record. Additional
model/training suites: 246 passed, 1 deselected. Slow layering/custom-workspace
checks: 6 passed, 16 deselected. Wheel-only generation/config parsing/scaffolds,
Ruff, scoped BasedPyright, Zensical build, and whitespace checks passed.

The earlier package restructure draft is useful history for the model/ml split.
The head-owned prediction draft is not an execution instruction for this work:
its TileMerger removal, runtime-owned tiling, and fixed encoder/head assumptions
need revision before prediction work starts.
