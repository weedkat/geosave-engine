"""Prefect orchestration for native acquisition and model prediction."""

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from pydoc import locate

from prefect import flow, task
from prefect.task_runners import ThreadPoolTaskRunner
from pydantic import JsonValue
from torch import nn
import xarray as xr

from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.utils.io import zarr

from .inference import infer
from .ingestion import acquire
from .io import open_rasters, write_stack
from .postprocessing import postprocess
from .preprocessing import preprocess
from .runtime import open_anchor, open_sources
from .spec import ModelSpec


# Native lazy graphs and loaded models remain in the local process. Durable
# checkpoints are explicit writes, never automatic serialization of task results.
_acquire = task(acquire, cache_policy=None, persist_result=False)
_preprocess = task(preprocess, cache_policy=None, persist_result=False)
_infer = task(infer, cache_policy=None, persist_result=False)
_postprocess = task(postprocess, cache_policy=None, persist_result=False)
_write_stack = task(write_stack, cache_policy=None, persist_result=False)


def _new_local_path(path: str | Path, *, zarr_store: bool = False) -> Path:
    if "://" in str(path):
        raise ValueError("Workflow output requires a local path")
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    if zarr_store and destination.suffix.lower() != ".zarr":
        raise ValueError("Raster stack output must end in .zarr")
    return destination


@task(cache_policy=None, persist_result=False)
def _write_predictions(
    outputs: tuple[xr.Dataset, ...], destination: Path
) -> tuple[str, ...]:
    destination = _new_local_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Only expose the result directory once every temporal window is complete.
    with TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / "predictions"
        staged.mkdir()
        names = tuple(f"window-{index:04d}.zarr" for index in range(len(outputs)))
        for result, name in zip(outputs, names, strict=True):
            zarr.write(result, staged / name, compute=True, overwrite=False)
        staged.rename(destination)
    return tuple(str(destination / name) for name in names)


@flow(
    name="ingest",
    task_runner=ThreadPoolTaskRunner(max_workers=4),
    persist_result=False,
)
def ingest(
    sources: dict[str, dict[str, JsonValue]],
    anchor: dict[str, JsonValue],
    *,
    output: str,
    spec: str | None = None,
) -> str:
    """Acquire STAC rasters from primitive settings into a completed local stack.

    Args:
        sources: Names mapped to STAC url, collection, optional query and load.
        anchor: Explicit coordinates or GeoJSON path with native grid settings.
        output: New local .zarr store.
        spec: Optional model YAML or artifact directory to select required sources.

    Returns:
        Path of the completed raw raster stack.

    Raises:
        ValueError: Source, anchor or output settings are invalid.
        FileExistsError: The destination already exists.
    """
    destination = _new_local_path(output, zarr_store=True)
    requirements = None if spec is None else ModelSpec.load(spec).sources
    if not sources:
        raise ValueError("At least one source is required")
    if requirements is not None and (missing := requirements.keys() - sources.keys()):
        raise ValueError(f"Source bindings are missing: {sorted(missing)}")
    selected = (
        sources
        if requirements is None
        else {name: sources[name] for name in requirements}
    )
    region = open_anchor(anchor)
    bound = open_sources(selected)
    pending = {
        name: _acquire.submit(
            {name: source},
            region,
            requirements=None if requirements is None else {name: requirements[name]},
        )
        for name, source in bound.items()
    }
    raw = stack(
        {name: result.result().gs.rasters[name] for name, result in pending.items()}
    )
    return _write_stack(raw, destination)


@task(cache_policy=None, persist_result=False)
def _load_model(path: str, loader: str, options: dict[str, JsonValue]) -> nn.Module:
    factory = locate(loader)
    if not callable(factory):
        raise ValueError(f"Model loader {loader!r} is not an importable callable")
    model = factory(path, **options)
    if not isinstance(model, nn.Module):
        raise TypeError("The model loader must return a torch.nn.Module")
    return model


@flow(name="predict", persist_result=False)
def predict(
    inputs: str | dict[str, str],
    *,
    model: str,
    output: str,
    spec: str | None = None,
    context: dict[str, JsonValue] | None = None,
    model_context: str | None = None,
    model_loader: str = "geosave_engine.ml.models.contract.ModelChain.from_pretrained",
    model_options: dict[str, JsonValue] | None = None,
    batch_size: int = 1,
    device: str | None = None,
    prepared_output: str | None = None,
) -> tuple[str, ...]:
    """Predict from raster paths and a saved model using serializable inputs.

    Args:
        inputs: Saved raw stack path, or raster names mapped to file paths.
        model: Saved model path supplied to the loader inside the worker.
        output: New local directory for completed per-window Zarr predictions.
        spec: Model YAML/artifact path; None loads model_spec.yaml beside the model.
        context: Primitive additional model arguments shared across batches.
        model_context: Optional import path to a per-tile DataTree context callable.
        model_loader: Import path to the model's native loading function.
        model_options: Primitive keyword arguments for that loader.
        batch_size: Maximum tiles per model invocation.
        device: Explicit model device; None uses its current device or CPU.
        prepared_output: Optional new .zarr checkpoint, reopened for inference.

    Returns:
        Completed prediction paths in temporal-window order.

    Raises:
        ValueError: Input paths, model settings or destinations are invalid.
        FileExistsError: A requested destination already exists.
    """
    settings = ModelSpec.load(model if spec is None else spec)
    prepared_path = (
        _new_local_path(prepared_output, zarr_store=True)
        if prepared_output is not None
        else None
    )
    destination = _new_local_path(output)
    if prepared_path is not None:
        prepared_resolved, output_resolved = (
            prepared_path.resolve(),
            destination.resolve(),
        )
        if (
            prepared_resolved == output_resolved
            or prepared_resolved in output_resolved.parents
            or output_resolved in prepared_resolved.parents
        ):
            raise ValueError(
                "Prepared and prediction outputs must use separate destinations"
            )
    extractor = None if model_context is None else locate(model_context)
    if model_context is not None and not callable(extractor):
        raise ValueError(
            f"Model context {model_context!r} is not an importable callable"
        )
    with ExitStack() as resources:
        opened = resources.enter_context(open_rasters(inputs))
        prepared = _preprocess(opened, spec=settings)
        if prepared_path is not None:
            saved = _write_stack(prepared, prepared_path)
            prepared = resources.enter_context(open_rasters(saved))
        loaded = _load_model(model, model_loader, model_options or {})
        logits = _infer(
            prepared,
            model=loaded,
            settings=settings.inference,
            context=context,
            model_context=extractor,
            batch_size=batch_size,
            device=device,
        )
        outputs = _postprocess(logits, settings=settings.postprocessing)
        return _write_predictions(outputs, destination)
