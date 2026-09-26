"""Publish completed prepared samples as a spatial catalog."""

from pathlib import Path

from prefect import task
from prefect.cache_policies import NO_CACHE

from geosave_engine.geodata import GeoVector
from geosave_engine.geodata.utils import io


@task(cache_policy=NO_CACHE, persist_result=False)
def write_manifest(
    samples: dict[str, str], destination: str | Path
) -> str:
    """Register completed sample stores and atomically publish GeoParquet."""
    records = []
    for sample_id, path in samples.items():
        sample_path = Path(path).resolve()
        with io.read_stack(sample_path, chunks="auto") as sample:
            records.append(
                GeoVector.from_xarray(
                    sample,
                    crs="EPSG:4326",
                    fields=("time", "grid", "variables"),
                    path=sample_path,
                    sample_id=sample_id,
                )
            )

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    GeoVector.concat(records).to_geoparquet(output, overwrite=True)
    return str(output)
