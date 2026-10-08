"""A STAC GeoParquet table answers the queries a STAC API does."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from geosave_engine.geodata import stac
from geosave_engine.geodata.errors import CollectionNotFoundError
from geosave_engine.geodata.stac import StacClient, StacQuery, StacTableClient, table


@pytest.fixture
def catalog(scene, tmp_path: Path):
    """Write two dated scenes of one Collection as one table."""
    forest = stac.create_collection(
        "forest", description="Forest samples", license="CC-BY-4.0"
    )
    paths = scene.gs.to_cog(tmp_path / "forest", split_bands=True)
    first, second = stac.create_items(paths, collection=forest)
    first.properties["eo:cloud_cover"] = 5.0
    second.properties["eo:cloud_cover"] = 40.0
    path = table.write(
        [first, second], tmp_path / "catalog" / "items.parquet", collections=[forest]
    )
    return path, (first, second)


def test_a_table_answers_a_stac_query(catalog) -> None:
    folder, (first, second) = catalog
    client = StacTableClient(folder)
    forest = StacQuery(collections=["forest"])

    assert [item.id for item in client.search(forest)] == [first.id, second.id]
    assert client.search(StacQuery(collections=["absent"])) == []
    assert [
        item.id for item in client.search(forest.set_filter("eo:cloud_cover <= 10"))
    ] == [first.id]
    assert [item.id for item in client.search(forest.sort_by("-eo:cloud_cover"))] == [
        second.id,
        first.id,
    ]
    assert [
        item.id
        for item in client.search(StacQuery(collections=["forest"], ids=[second.id]))
    ] == [second.id]


def test_a_table_is_searched_by_place_and_time(catalog) -> None:
    folder, (first, second) = catalog
    client = StacTableClient(folder)

    inside = StacQuery(collections=["forest"], bbox=tuple(first.bbox))
    outside = StacQuery(collections=["forest"], bbox=(0.0, 0.0, 1.0, 1.0))
    early = StacQuery(
        collections=["forest"],
        datetime=(datetime(2025, 5, 30), datetime(2025, 6, 5)),
    )

    assert len(client.search(inside)) == 2
    assert client.search(outside) == []
    assert [item.id for item in client.search(early)] == [first.id]


def test_found_items_point_at_files_that_open(catalog) -> None:
    folder, (first, _) = catalog

    (found, _) = StacTableClient(folder).search(StacQuery(collections=["forest"]))

    assert found.assets["red"].href == first.assets["red"].href
    assert found.assets["nir"].href == first.assets["nir"].href
    assert Path(found.assets["red"].href).is_file()


def test_a_table_states_its_collections(catalog) -> None:
    folder, _ = catalog
    client = StacTableClient(folder)

    assert client.collections() == {"forest"}
    assert client.collection("forest").license == "CC-BY-4.0"
    with pytest.raises(CollectionNotFoundError, match="missing"):
        client.collection("missing")


def test_a_table_without_stored_collections_derives_them(scene, tmp_path) -> None:
    forest = stac.create_collection("forest", description="Forest samples")
    items = stac.create_items(scene.gs.to_cog(tmp_path / "forest"), collection=forest)
    path = table.write(items, tmp_path / "items.parquet")
    client = StacTableClient(path)

    assert table.read_collections(path) == {}
    assert client.collections() == {"forest"}
    assert client.collection("forest").id == "forest"


def test_opening_a_table_path_gives_a_table_client(catalog) -> None:
    folder, _ = catalog

    assert isinstance(StacClient.open(str(folder)), StacTableClient)


def test_a_source_loads_from_a_table(catalog, scene) -> None:
    folder, _ = catalog
    source = StacClient.open(str(folder)).source("forest")
    source = source.set_config(bands=["red", "nir"], groupby="time")

    loaded = source.load(scene.gs.anchor)

    assert dict(loaded.sizes) == {"time": 2, "y": 64, "x": 64}
    # odc-stac masks the file's fill value; every stored reading comes back.
    stored = scene.red.values != 0
    np.testing.assert_array_equal(loaded.red.values[stored], scene.red.values[stored])
    assert loaded.attrs["license"] == "CC-BY-4.0"


def test_search_uses_the_rustac_session_configuration(catalog) -> None:
    import rustac

    folder, _ = catalog
    duckdb = rustac.DuckdbClient()
    duckdb.execute("SET enable_external_access = false")
    client = StacClient.open(str(folder), duckdb=duckdb)

    with pytest.raises(rustac.RustacError, match="disabled by configuration"):
        client.search(StacQuery(collections=["forest"]))


def test_table_client_reads_stored_collections_through_rustac(
    catalog, monkeypatch
) -> None:
    folder, _ = catalog

    def refuse(*args, **kwargs):
        raise AssertionError("table client opened a second filesystem")

    # Search and Collection metadata must use the same configured DuckDB session.
    monkeypatch.setattr(table.storage, "filesystem_path", refuse)

    assert StacTableClient(folder).collection("forest").license == "CC-BY-4.0"


def test_a_custom_s3_endpoint_reads_items_and_stored_collections(catalog, tmp_path):
    import subprocess
    import sys

    import rustac

    _, (first, _) = catalog
    # rustac holds the GIL during I/O, so the endpoint needs its own process.
    code = """
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        requested = self.headers.get('Range')
        if requested is None:
            return super().do_GET()
        path = Path(self.translate_path(self.path))
        start, end = requested.removeprefix('bytes=').split('-')
        start, end = int(start), int(end) if end else path.stat().st_size - 1
        with path.open('rb') as source:
            source.seek(start)
            data = source.read(end - start + 1)
        self.send_response(206)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Content-Range', f'bytes {start}-{end}/{path.stat().st_size}')
        self.end_headers()
        self.wfile.write(data)

server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
print(server.server_port, flush=True)
server.serve_forever()
"""
    server = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert server.stdout is not None
    port = int(server.stdout.readline())
    try:
        duckdb = rustac.DuckdbClient()
        duckdb.execute(
            "CREATE SECRET (TYPE s3, KEY_ID 'test-key', SECRET 'test-secret', "
            f"ENDPOINT '127.0.0.1:{port}', URL_STYLE 'path', "
            "USE_SSL false)"
        )
        # The HTTP server represents an S3 endpoint with tmp_path as its buckets.
        location = "s3://catalog/items.parquet"
        client = StacClient.open(location, duckdb=duckdb)

        (found, _) = client.search(StacQuery(collections=["forest"]))

        assert found.id == first.id
        assert client.collection("forest").license == "CC-BY-4.0"
        assert found.assets["red"].href.startswith("s3://")
    finally:
        server.terminate()
        server.wait(timeout=5)
        server.stdout.close()
