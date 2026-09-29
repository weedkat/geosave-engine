# Native Model Release Design

## Summary

GeoSave trains through Lightning and deploys native PyTorch inference graphs.
A Lightning checkpoint remains a training artifact. A model release contains the
selected native `ModelChain`, its generated construction configuration, its
safetensors state, and `model_spec.yaml`.

`ModelChain` is an ordinary `torch.nn.Module`. It is the initial publication unit
because, unlike an isolated child module, it retains the resolved construction
recipe needed to rebuild itself in another process.

Users publish and load native models through GeoSave functions. The
Transformers-specific `GeoSaveModel` wrapper remains an installed internal
persistence adapter and does not appear in the normal training or deployment
workflow. Releases do not contain executable Python source.

```python
from geosave_engine.release import load_model, load_spec, publish_model


# Training
task = SemanticSegmentationTask(...)
trainer.fit(task)

# Select the checkpoint and publish its native inference graph.
task = SemanticSegmentationTask.load_from_checkpoint("best.ckpt")
publish_model(
    task.model,
    spec="configs/model_spec.yaml",
    repo_id="geosave/my-model",
)

# Raster discovery, validation, and preparation load only the spec.
spec = load_spec("geosave/my-model", revision=commit)

# Tensor deployment loads only the native model.
model = load_model("geosave/my-model", revision=commit)
model = model.to(device).eval()
with torch.inference_mode():
    output = model(**model_inputs)
```

## Goals

- Keep Lightning responsible for training, validation, optimization, callbacks,
  and checkpoint recovery.
- Make the native PyTorch inference graph the unit that is versioned and served.
- Support one inference graph containing one or many cooperating PyTorch models.
- Hide the Transformers adapter behind concise GeoSave publication and loading
  functions.
- Publish one immutable artifact containing weights, construction, and the
  geospatial processing contract.
- Preserve `ModelChain` and `chain_step` as the structured model-development
  interface.
- Load the released model without constructing a Lightning task or Trainer.

## Non-goals

- Publishing an arbitrary workspace source tree as executable Hub code.
- Inferring which child of an arbitrary custom LightningModule should be
  deployed.
- Turning `model_spec.yaml` into LightningCLI configuration.
- Putting STAC acquisition, scene tiling, spatial aggregation, or georeferenced
  persistence inside the native model.
- Adding ONNX, TorchScript, TensorRT, or a generic serving protocol in this
  change.
- Making a Lightning checkpoint the production inference artifact.

## Ownership

```text
configs/train.yaml
    Human-authored LightningCLI configuration.
    Owns Trainer, LightningModule, DataModule, loss, optimizer, and scheduler.

LightningModule
    One training system.
    May contain several native models used for training.

task.model
    Complete deployable inference graph for built-in GeoSave tasks.
    Contains every learned module and buffer needed during tensor inference.

ModelChain
    Reproducibly constructed native PyTorch inference graph.
    May contain one model, a sequential cascade, branches, or multiple heads.

model_spec.yaml
    Required raster inputs, preprocessing, tiling, tensor conversion,
    aggregation, postprocessing, and exported geospatial values.

Hugging Face release
    Generated model construction, safetensors state, model_spec.yaml, and model
    card. It contains no executable Python source.

LitServe
    Runtime transport, device placement, request batching, and invocation of the
    loaded native model.
```

`configs/model.yaml` is renamed to `configs/train.yaml` because the document
configures the complete Lightning training run rather than only the native
model. Alpha status permits the direct rename without an alias.

## Lightning Contract

A LightningModule is a training system, not the publication unit. It may own
multiple PyTorch models, for example a student and teacher or a generator and
discriminator. Custom LightningModules retain ordinary Lightning freedom.

Built-in GeoSave tasks follow one additional convention:

```python
class SemanticSegmentationTask(LightningModule):
    model: ModelChain

    def forward(self, **model_inputs: object) -> object:
        return self.model(**model_inputs)
```

`task.model` is the complete deployable tensor-inference graph. `forward()`
delegates without applying task-only interpretation. Training, validation, and
test steps select the tensor outputs required by their losses and metrics.

The task may retain training-only models under other attributes. Publication
never inspects a LightningModule and guesses which child to deploy. A custom
task explicitly passes its chosen configured `ModelChain` to `publish_model()`.

```python
# The task chooses a configured ModelChain; the publisher does not inspect it.
publish_model(task.inference_model, spec=spec, repo_id=repo_id)
```

For built-in semantic segmentation, the native graph returns raw tile logits.
Losses, metrics, and validation callbacks consume those logits. Spatial
aggregation happens before nonlinear scene interpretation.

## Multiple-model Inference Graphs

Models that always execute, version, and deploy together form one native
inference graph:

```python
class Segmenter(nn.Module):
    @chain_step(outputs=("logits",))
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)


class Refiner(nn.Module):
    @chain_step(head=True)
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.network(logits)


model = ModelChain(
    segmenter=Segmenter(),
    refiner=Refiner(),
)
```

Both modules appear in one `state_dict()` and one safetensors artifact. A
complex composite `nn.Module` may instead be registered as one ModelChain stage.

Models with independent release lifecycles use separate artifacts and are
composed by a workflow. A model that consumes stitched, georeferenced scene
output is a later workflow inference step, not a per-tile ModelChain stage.

### Backbone-only releases

Pretraining may need more modules than downstream inference. For example, an MAE
task keeps the deployable GFM backbone in `task.model` and its reconstruction
decoder as a separate training-only child:

```python
class MAETask(LightningModule):
    def __init__(self, ...):
        super().__init__()
        self.model = build_model(
            {"encoder": {"name": "gfm", "init_args": {...}}}
        )
        self.decoder = MAEDecoder(...)

    def training_step(self, batch, batch_idx):
        features = self.model(image=batch["image"])["features"]
        reconstruction = self.decoder(features)
        return reconstruction_loss(reconstruction, batch)
```

Publishing `task.model` writes only the encoder parameters. After loading,
`model.encoder` is the native GFM module. Publishing `task.model.encoder`
directly is not supported initially because that child does not independently
own the resolved constructor recipe held by its parent chain.

A chain with no terminal head returns only the named values produced by its
selected `chain_step` methods. It does not return external inputs or other
internal context. Therefore a backbone-only chain can return
`{"features": features}`. One terminal head still returns a tensor, and multiple
terminal heads still return a mapping keyed by stage name.

## Construction Contract

Only a `ModelChain` built through `build_model()` is publishable initially. Its
`stage_specs` property is the construction recipe and contains the ordered stage
selectors plus resolved constructor arguments.

```python
model = build_model(
    {
        "encoder": {"name": "dinov3"},
        "decoder": {"name": "dpt"},
        "head": {"name": "dense"},
    }
)
```

An arbitrary native `nn.Module` remains usable as a registered ModelChain stage.
GeoSave does not add a second construction interface for arbitrary modules.

Published stage recipes must be JSON-serializable, use registered `name`
selectors, and resolve from the installed GeoSave library in a fresh process.
`class_path` remains available for workspace training but is rejected by
publication. This keeps executable implementation code in the versioned library
rather than copying workspace or third-party source into a model repository.

Promotion from a workspace path to a library name is explicit:

```python
trained = task.model
released = build_model(stable_stage_specs)
released.load_state_dict(trained.state_dict(), strict=True)
publish_model(released, spec=spec, repo_id=repo_id)
```

Strict state transfer proves that the promoted implementation has the same
parameter and buffer structure. Publication does not silently rewrite workspace
class paths.

## Publication Interface

`geosave_engine.release` owns transport for the two independently loadable parts
of a release:

```python
from geosave_engine.release import load_model, load_spec, publish_model, save_model


def save_model(
    model: ModelChain,
    path: str | Path,
    *,
    spec: ModelSpec | str | Path,
) -> Path:
    """Write one complete local inference release."""


def publish_model(
    model: ModelChain,
    repo_id: str,
    *,
    spec: ModelSpec | str | Path,
    revision: str | None = None,
    token: str | bool | None = None,
) -> str:
    """Upload one complete release and return its immutable commit hash."""


def load_model(
    path_or_repo_id: str | Path,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> ModelChain:
    """Load one native inference graph."""


def load_spec(
    path_or_repo_id: str | Path,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> ModelSpec:
    """Load one model-owned geospatial processing contract."""
```

`save_model()` is the primitive. The destination must not exist. It stages the
complete release in a sibling temporary directory and renames that directory to
the destination only after every file is valid. `publish_model()` builds the same
local release in a temporary directory, uploads that directory, and returns the
Hub commit hash. Its `revision` argument selects the target Hub branch or tag;
deployment uses the returned commit hash.

`load_model()` passes the local release directory or Hub repository directly to
the installed `GeoSaveModel.from_pretrained()` implementation, honoring
`revision`, `token`, and `local_files_only`. Transformers downloads only the
configuration and weights it needs. The function does not execute remote code
and returns only the adapter's native chain.

`load_spec()` passes local files and directories to `ModelSpec.load()`. For a Hub
repository it downloads only `model_spec.yaml` with `hf_hub_download()` and then
loads that local file. It does not import Transformers or download weights.
`ModelSpec` therefore remains an independent workflow contract for raster
acquisition, validation, and processing.

The top-level module uses function-local imports for the optional Transformers
adapter, so spec-only consumers do not require the Hub extra. A process that
needs both halves may download one pinned snapshot and load them independently:

```python
snapshot = snapshot_download(repo_id, revision=commit)
spec = load_spec(snapshot)
model = load_model(snapshot)
```

This keeps the two consumers separate while proving both files came from the
same immutable release. `load_model()` never performs raster acquisition or
returns workflow configuration as a tuple.

`GeoSaveConfig` and `GeoSaveModel` remain importable installed classes because
Transformers persistence requires concrete classes. They are adapter
implementation, not the documented user workflow. `ModelChain` remains free of
`save_pretrained()`, `push_to_hub()`, and Transformers inheritance.

## Release Layout

```text
release/
├── config.json
├── model.safetensors
├── model_spec.yaml
└── README.md
```

`config.json` is generated; users do not edit it. `save_model()` translates
`ModelChain.stage_specs` into `GeoSaveConfig`, and inherited `save_pretrained()`
writes the JSON and safetensors files. `save_model()` also copies
`model_spec.yaml` and generates the minimal model card. A simplified generated
configuration is:

```json
{
  "model_type": "geosave_chain",
  "format_version": 1,
  "geosave_version": "0.2.0",
  "architectures": ["GeoSaveModel"],
  "stages": [
    {
      "stage": "encoder",
      "spec": {
        "name": "gfm",
        "init_args": {"embed_dim": 768, "patch_size": 16, "pretrained": false}
      }
    }
  ]
}
```

The public functions hide the adapter call. On load, the generated stage recipe
constructs installed library modules and safetensors supplies their trained
parameters.

The existing `huggingface.py` export is a copy of the Transformers adapter source
created by `register_for_auto_class()`. It exists only so generic
`AutoModel.from_pretrained(..., trust_remote_code=True)` can execute repository
code without first importing GeoSave. It is not the native model architecture or
the user's model implementation. The release API removes this remote-code mode:
deployment installs `geosave-engine[hub]`, and the installed `GeoSaveModel`
interprets `config.json`. The exporter therefore does not call
`register_for_auto_class()` and does not write `huggingface.py` or `auto_map`.

The configuration records `format_version: 1` for the GeoSave construction
contract and the exact installed GeoSave package version. Because implementation
code remains in the installed Alpha library, `load_model()` rejects a different
GeoSave version instead of silently executing changed code. This strict check can
be relaxed after the model construction API has a compatibility policy. Hub
deployments pin the immutable commit hash returned by `publish_model()`; a branch
name is acceptable for interactive loading but is not reproducible deployment.

The release does not contain `train.yaml`, the Lightning checkpoint, optimizer
state, scheduler state, callbacks, metrics, or the DataModule.

## Inference State

Every value required by native tensor inference lives in the published
ModelChain as a parameter, persistent buffer, or JSON-serializable constructor
argument.

Values applied after spatial aggregation live explicitly in `model_spec.yaml`.
Learned class thresholds cannot remain only on the Lightning task. Threshold
calibration must produce explicit release input for postprocessing before a
thresholded release can be published. The existing argmax template does not
require calibrated thresholds.

The release path does not attempt to serialize task attributes implicitly.

## Model Spec Validation

Publication loads and revalidates `model_spec.yaml` before writing any artifact.
It also requires:

- every required external `ModelChain.inputs` name to be a named output of the
  model-input stage;
- JSON-serializable construction specifications.

The publisher does not guess whether a runtime model result will be a tensor or a
mapping, so result-to-aggregation validation remains in `predict_dense()`, where
the actual result exists. Publication validation is structural and does not
execute STAC acquisition or compute a raster. Tests exercise a bounded synthetic
sample through preprocessing, model-input conversion, native inference,
aggregation, and postprocessing.

## LitServe

LitServe loads the release once during setup:

```python
class GeoSaveAPI(ls.LitAPI):
    def setup(self, device):
        self.model = load_model(
            self.model_id,
            revision=self.revision,
        )
        self.model.to(device).eval()

    def predict(self, model_inputs):
        with torch.inference_mode():
            return self.model(**model_inputs)
```

The initial server adapter accepts prepared named tensor inputs and returns raw
tensor or named-tensor outputs. Prefect workflows retain raster acquisition,
preprocessing, tiling, spatial aggregation, postprocessing, and persistence.

Offline dense prediction keeps DataLoader batching as the single batch owner.
LitServe dynamic batching is deferred until a per-sample request interface and
concurrent remote runner exist; nested batch dimensions are not introduced.

## Errors

Publication fails before upload when:

- the chain was not built from stage specifications;
- a stage uses `class_path` instead of a registered `name`;
- the construction recipe is not JSON-serializable;
- `model_spec.yaml` is absent or structurally incompatible;
- the target local release directory already exists.

Loading fails when:

- the artifact revision is missing required files;
- a stage selector cannot be resolved or model construction fails;
- safetensors contain missing, unexpected, or mismatched state;
- the GeoSave artifact format version is unsupported;
- the installed GeoSave version differs from the generated version.

Loading `ModelSpec` fails independently when `model_spec.yaml` is absent or
invalid. This does not prevent tensor-only serving from loading the model.

No compatibility alias preserves the direct `GeoSaveModel.from_chain()` user
workflow in documentation. The adapter remains available for Transformers, but
the supported GeoSave workflow uses `save_model()`, `publish_model()`,
`load_model()`, and `load_spec()`.

## Testing

Focused tests cover:

- a local release containing all required files;
- generated `config.json` preserving ordered resolved stage recipes;
- generated `config.json` recording artifact and exact GeoSave versions;
- `load_model()` returning only a native `ModelChain`;
- `load_spec()` independently reading a local or Hub processing contract without
  importing Transformers or downloading weights;
- exact weights and outputs after an installed-adapter fresh-process round trip;
- no `huggingface.py`, `auto_map`, or remote-code execution path;
- a multi-stage segmenter/refiner graph in one safetensors artifact;
- strict failure for missing, unexpected, and mismatched weights;
- rejection of non-serializable constructor arguments;
- rejection of `class_path` stage selectors during publication;
- rejection of a directly composed chain without construction specs;
- promotion by strict state transfer from a workspace-built chain to a
  library-built chain;
- built-in task `forward()` matching `task.model()`;
- a backbone-only MAE task publishing `task.model` without its training decoder;
- a headless chain returning only named stage outputs, not its external inputs;
- custom Lightning tasks explicitly publishing a selected configured chain;
- the renamed `configs/train.yaml` working through LightningCLI;
- no Lightning checkpoint, optimizer, scheduler, criterion, metric, or
  DataModule state in the release;
- a mocked Hub upload containing the complete release;
- bounded synthetic raster execution through `model_spec.yaml`.

Run focused publication, model-chain, Lightning-task, template, and workflow
prediction tests, followed by scoped Ruff, BasedPyright, and `git diff --check`.

## Breaking Changes

- `configs/model.yaml` becomes `configs/train.yaml` in generated workspaces.
- Documentation stops presenting `GeoSaveModel.from_chain()` as the publication
  workflow.
- Built-in task `forward()` returns the native model output without silently
  selecting `result["logits"]`.
- Publishing supports configured `ModelChain` instances only in the initial
  version.
- Publication requires registered `name` selectors; workspace `class_path`
  implementations must be promoted into the library first.
- Headless `ModelChain` results no longer include external inputs in their output
  mapping.
- Loading requires the exact GeoSave version recorded by the release during
  Alpha development.
- Generic loading through uninstalled Hub remote code is no longer supported;
  deployments install `geosave-engine[hub]` and use `load_model()`.
- Deployments install the GeoSave Hub extra and load the immutable Hub commit
  returned by publication.
