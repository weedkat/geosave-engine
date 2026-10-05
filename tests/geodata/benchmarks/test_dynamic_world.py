"""Original-file downloads publish only completed benchmark subsets."""

import hashlib
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pytest

from geosave_engine.geodata.benchmarks import dynamic_world


@pytest.fixture
def publisher(tmp_path, monkeypatch):
    source = tmp_path / "publisher"
    source.mkdir()
    archive = source / "experts.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("Experts/EH/1/tile.tif", b"original pixels")
    (source / "README.txt").write_text("Publisher description")
    (source / "metadata.xlsx").write_bytes(b"original spreadsheet")
    monkeypatch.setattr(
        dynamic_world, "_ARCHIVES", {"experts": (archive.as_uri(), None)}
    )
    metadata = {
        "README.txt": (source / "README.txt").as_uri(),
        "metadata.xlsx": (source / "metadata.xlsx").as_uri(),
    }
    monkeypatch.setattr(dynamic_world, "_METADATA", {"experts": metadata})
    return source, tmp_path / "data"


def test_download_preserves_archive_layout_and_metadata(publisher):
    _, root = publisher
    result = dynamic_world.download(root)
    assert result == root / "experts"
    assert (result / "Experts/EH/1/tile.tif").read_bytes() == b"original pixels"
    assert (result / "metadata.xlsx").read_bytes() == b"original spreadsheet"
    assert (result / "README.txt").read_text() == "Publisher description"
    assert sorted(path.name for path in root.iterdir()) == ["experts"]


def test_completed_download_is_reused_without_network(publisher, monkeypatch):
    _, root = publisher
    result = dynamic_world.download(root)

    def unexpected_download(*args):
        pytest.fail("Completed subset downloaded again")

    monkeypatch.setattr(dynamic_world, "_download_file", unexpected_download)
    assert dynamic_world.download(root) == result


def test_incomplete_destination_is_not_overwritten(publisher):
    _, root = publisher
    (root / "experts").mkdir(parents=True)
    keep = root / "experts/user.txt"
    keep.write_text("Keep this")
    with pytest.raises(FileExistsError, match="incomplete"):
        dynamic_world.download(root)
    assert keep.read_text() == "Keep this"


@pytest.mark.parametrize("failed_file", ["experts.zip", "metadata.xlsx"])
def test_transfer_failure_is_not_published(publisher, monkeypatch, failed_file):
    _, root = publisher
    transfer = dynamic_world._download_file

    def fail(url, target):
        if url.endswith(failed_file):
            target.write_bytes(b"partial")
            raise OSError("transfer interrupted")
        transfer(url, target)

    monkeypatch.setattr(dynamic_world, "_download_file", fail)
    with pytest.raises(OSError, match="transfer interrupted"):
        dynamic_world.download(root)
    assert list(root.iterdir()) == []
    monkeypatch.setattr(dynamic_world, "_download_file", transfer)
    assert dynamic_world.download(root).is_dir()


def test_corrupt_archive_is_not_published(publisher):
    source, root = publisher
    (source / "experts.zip").write_bytes(b"not a zip")
    with pytest.raises(BadZipFile):
        dynamic_world.download(root)
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("valid", [False, True])
def test_publisher_checksum_is_verified(publisher, monkeypatch, valid):
    source, root = publisher
    archive = source / "experts.zip"
    checksum = hashlib.md5(archive.read_bytes()).hexdigest() if valid else "0" * 32
    monkeypatch.setattr(
        dynamic_world, "_ARCHIVES", {"experts": (archive.as_uri(), checksum)}
    )
    if valid:
        assert dynamic_world.download(root).is_dir()
    else:
        with pytest.raises(ValueError, match="checksum"):
            dynamic_world.download(root)
        assert list(root.iterdir()) == []


@pytest.mark.parametrize(
    "member", ["../escape.txt", "/escape.txt", "..\\escape.txt", "C:\\escape.txt"]
)
def test_archive_paths_cannot_escape(publisher, member):
    source, root = publisher
    with ZipFile(source / "experts.zip", "w") as output:
        output.writestr(member, "escape")
    with pytest.raises(ValueError, match="archive member"):
        dynamic_world.download(root)
    assert list(root.iterdir()) == []
    assert not (root / "escape.txt").exists()


def test_interrupted_extraction_is_not_published(publisher, monkeypatch):
    _, root = publisher

    def interrupt(self, path):
        Path(path, "partial.tif").write_bytes(b"partial")
        raise OSError("extraction interrupted")

    monkeypatch.setattr(ZipFile, "extractall", interrupt)
    with pytest.raises(OSError, match="extraction interrupted"):
        dynamic_world.download(root)
    assert list(root.iterdir()) == []


def test_unknown_subset_does_not_create_destination(tmp_path):
    root = tmp_path / "data"
    with pytest.raises(ValueError, match="subset"):
        dynamic_world.download(root, subset="missing")
    assert not root.exists()
