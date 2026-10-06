"""Package dependency direction and removed import paths."""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def _run(code: str) -> str:
    """Return the last line a fresh interpreter prints for `code`."""
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().splitlines()[-1]


def _loads(imports: str, module: str) -> bool:
    """Report whether importing `imports` also loads `module`."""
    return (
        _run(f"import sys\nimport {imports}\nprint({module!r} in sys.modules)")
        == "True"
    )


@pytest.mark.parametrize(
    "path",
    [
        "geosave_engine.utils",
        "geosave_engine.geodata.utils.io",
        "geosave_engine.ml.callbacks",
        "geosave_engine.ml.datasets",
        "geosave_engine.templates.tasks",
        "geosave_engine.templates.boilerplate",
        "geosave_engine.geodata.datasets",
        "geosave_engine.ml.model_chain",
        "geosave_engine.ml.models",
        "geosave_engine.ml.encoding",
        "geosave_engine.model_spec",
        "geosave_engine.release",
        "geosave_engine.ml.registry",
        "geosave_engine.ml.lightning",
        "geosave_engine.ml.metrics",
        "geosave_engine.ml.transforms.semantic_segmentation",
        "geosave_engine.geodata.core.profile",
        "geosave_engine.geodata.utils.gdal_env",
        "geosave_engine.geodata.transform.window",
    ],
)
def test_old_path_is_gone(path):
    assert importlib.util.find_spec(path) is None


def test_geodata_has_no_torch_or_ml_imports():
    root = Path(__file__).parents[1] / "src" / "geosave_engine" / "geodata"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            assert not any(
                module == "torch"
                or module.startswith("torch.")
                or module == "geosave_engine.ml"
                or module.startswith("geosave_engine.ml.")
                for module in modules
            ), path


@pytest.mark.slow
def test_geodata_loads_no_torch():
    assert not _loads(
        "geosave_engine.geodata, geosave_engine.geodata.io, geosave_engine.geodata.benchmarks",
        "torch",
    )


@pytest.mark.slow
def test_model_core_loads_no_lightning():
    imports = "geosave_engine.model.chain, geosave_engine.model.registry, geosave_engine.model.head"
    assert not _loads(imports, "lightning")


@pytest.mark.slow
def test_every_stage_package_registers_its_models():
    listed = _run(
        "from geosave_engine.model.registry import list_models\nprint(list_models())"
    )
    assert ast.literal_eval(listed) == {
        "decoder": ["DPT", "UNET"],
        "encoder": ["CLAY", "DINOV3", "PRITHVI", "PRITHVI_TL"],
        "head": ["CLASSIFICATION", "DETECTION", "REGRESSION", "SEGMENTATION"],
        "model": ["IBM_GRANITE_BIOMASS"],
    }


@pytest.mark.slow
def test_spec_and_workflow_load_no_torch():
    assert not _loads(
        "geosave_engine.model.spec, geosave_engine.workflow.flows", "torch"
    )


@pytest.mark.slow
def test_release_loads_no_lightning():
    assert not _loads("geosave_engine.model.release", "lightning")


def test_supervised_segmentation_pairs_its_module_and_data():
    from lightning import LightningDataModule, LightningModule

    from geosave_engine.ml.segmentation import supervised

    assert issubclass(supervised.Module, LightningModule)
    assert issubclass(supervised.DataModule, LightningDataModule)
