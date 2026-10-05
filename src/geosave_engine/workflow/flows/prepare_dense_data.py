"""Prepare bounded, label-aligned dense training datasets."""

from pathlib import Path
from typing import Literal

from prefect import flow
from prefect.futures import PrefectFuture, as_completed
from pydantic import JsonValue, PositiveInt

from geosave_engine.geodata import GeoDataFrame, GeoVector, read_vector
from geosave_engine.geodata.stac.item import ITEM_COLUMNS
from geosave_engine.model.spec import ModelSpec
from geosave_engine.workflow.tasks import prepare_dense_sample
from geosave_engine.workflow.tasks.dense import validate_row
from geosave_engine.workflow.tasks.labels import read_labels


def _read_manifest(path: Path) -> GeoDataFrame | None:
    """Read the samples a previous run recorded, if it left any."""
    if not path.exists():
        return None
    manifest = read_vector(path)
    if "id" not in manifest or "assets" not in manifest:
        raise ValueError(
            f"{path} is not a sample manifest; move it away before preparing "
            f"samples into its directory"
        )
    return manifest


@flow(name="prepare-dense-data", persist_result=False)
def prepare_dense_data(
    labels: str,
    *,
    output: str,
    spec: str,
    pattern: str = "**/*.tif",
    max_concurrency: PositiveInt = 1,
    format: Literal["geotiff", "zarr"] = "geotiff",
    write_options: dict[str, JsonValue] | None = None,
) -> str:
    """Prepare dense training samples and record them in a spatial manifest.

    The manifest is written as each sample finishes, so a run that fails
    leaves the samples it finished recorded, and the next run continues from
    them.

    Args:
        labels: GeoParquet STAC table with one row per label, each naming its
            raster as the asset `label`, or a local directory of label
            rasters. A table's other columns are carried onto the manifest.
        output: Directory for prepared samples, each a folder holding one
            raster per layer, and the GeoParquet manifest listing them as
            STAC items.
        spec: Model specification defining required rasters and STAC recipes.
        pattern: Recursive label glob, for a directory of labels.
        max_concurrency: Maximum number of active sample ingestions. Defaults
            to one to protect raster reads and writes.
        format: Persisted representation of each sample raster.
        write_options: Serializable options for the native raster writer.

    Returns:
        Path to the completed GeoParquet manifest.

    Raises:
        ValueError: If label discovery, the existing manifest, or a sample is
            invalid.
    """
    model_spec = ModelSpec.load(spec)
    if "label" in model_spec.rasters:
        raise ValueError("Model raster name 'label' is reserved")

    destination = Path(output)
    manifest_path = destination / "manifest.parquet"
    label_table = read_labels(labels, pattern)
    manifest = _read_manifest(manifest_path)

    positions = {} if manifest is None else dict(zip(manifest["id"], manifest.index))
    columns = [name for name in label_table if name not in ITEM_COLUMNS]
    futures: set[PrefectFuture[GeoDataFrame]] = set()
    for _, label in label_table.iterrows():
        sample_id = str(label["id"])
        if manifest is not None and sample_id in positions:
            validate_row(manifest.loc[positions[sample_id]], model_spec)
            continue
        if len(futures) == max_concurrency:
            manifest = _record(manifest, futures, manifest_path, every=False)
        futures.add(
            prepare_dense_sample.submit(
                str(label["assets"]["label"]["href"]),
                model_spec,
                destination / sample_id,
                sample_id=sample_id,
                properties={name: label[name] for name in columns},
                format=format,
                write_options=write_options,
            )
        )
    manifest = _record(manifest, futures, manifest_path, every=True)

    if manifest is None:
        raise ValueError(f"No labels to prepare under {labels}")
    # Rows follow label order, and caller columns the label table as it is now.
    rows = GeoVector.concat(
        [manifest.set_index("id", drop=False).loc[label_table["id"]]]
    )
    for name in columns:
        rows[name] = label_table[name].to_numpy()
    owned = [name for name in rows if name in ITEM_COLUMNS and name != "geometry"]
    rows[[*owned, *columns, "geometry"]].gs.to_geoparquet(manifest_path, overwrite=True)
    return str(manifest_path)


def _record(
    manifest: GeoDataFrame | None,
    futures: set[PrefectFuture[GeoDataFrame]],
    path: Path,
    *,
    every: bool,
) -> GeoDataFrame | None:
    """Add finished samples' rows to the manifest and write it.

    Args:
        manifest: Rows recorded so far, or None before the first.
        futures: Running sample tasks, which lose the ones collected here.
        path: Manifest path, rewritten after each row.
        every: Wait for every task. False returns after the first to finish.

    Returns:
        The manifest holding each collected row.

    Raises:
        Exception: A sample task failed. Every other running task is still
            collected and recorded before its error is raised.
    """
    failure: Exception | None = None
    for future in as_completed(list(futures)):
        futures.discard(future)
        try:
            row = future.result()
        except Exception as error:
            # The samples finishing beside a failed one are recorded first.
            failure = failure or error
            continue
        manifest = row if manifest is None else manifest.gs.upsert(row, on="id")
        manifest.gs.to_geoparquet(path, overwrite=True)
        if failure is None and not every:
            break
    if failure is not None:
        raise failure
    return manifest
