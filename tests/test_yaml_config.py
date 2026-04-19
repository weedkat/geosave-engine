from __future__ import annotations

from pathlib import Path

import pytest

from geosave_engine.utils.yaml_config import inject_into_file, inject_value, load_yaml


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_load_yaml_requires_mapping_root(tmp_path: Path) -> None:
    target = tmp_path / "list.yaml"
    _write_yaml(target, "- a\n- b\n")

    with pytest.raises(TypeError):
        load_yaml(target)


def test_load_yaml_enforces_required_sections(tmp_path: Path) -> None:
    target = tmp_path / "partial.yaml"
    _write_yaml(target, "trainer:\n  max_epochs: 10\n")

    with pytest.raises(KeyError):
        load_yaml(target, required_mapping_keys=("model",))


def test_inject_value_walks_dotted_path() -> None:
    data = {"model": {"name": "old"}}
    inject_value(data, "model.name", "new")
    assert data == {"model": {"name": "new"}}


def test_inject_value_raises_when_parent_missing() -> None:
    with pytest.raises(KeyError):
        inject_value({}, "model.name", "new")


def test_inject_value_creates_missing_parents_when_allowed() -> None:
    data: dict = {}
    inject_value(data, "a.b.c", 1, create_missing_parents=True)
    assert data == {"a": {"b": {"c": 1}}}


def test_inject_into_file_roundtrips(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    _write_yaml(target, "model:\n  name: old\ntrainer:\n  max_epochs: 1\n")

    inject_into_file(target, "model.name", "new", required_mapping_keys=("model",))

    loaded = load_yaml(target)
    assert loaded["model"]["name"] == "new"
    assert loaded["trainer"]["max_epochs"] == 1
