"""Only workspace starters and scaffold files are selectable."""


def test_template_discovery_ignores_caches(tmp_path, monkeypatch):
    from geosave_engine.cli.core import templates

    workspaces = tmp_path / "workspaces"
    scaffolds = tmp_path / "scaffolds"
    for name in [
        "segmentation",
        "custom_lightning",
        "__pycache__",
        ".ipynb_checkpoints",
    ]:
        (workspaces / name).mkdir(parents=True)
    for name in ["prepare.py", "__pycache__", ".ipynb_checkpoints"]:
        (scaffolds / "scripts" / name).mkdir(parents=True)
    (scaffolds / "scripts" / "prepare.py").rmdir()
    (scaffolds / "scripts" / "prepare.py").write_text("print('prepare')\n")
    monkeypatch.setattr(templates, "WORKSPACES_DIR", workspaces)
    monkeypatch.setattr(templates, "SCAFFOLDS_DIR", scaffolds)
    assert templates.get_workspaces() == ["custom_lightning", "segmentation"]
    assert templates.get_scaffolds() == {"scripts": ["prepare.py"]}
