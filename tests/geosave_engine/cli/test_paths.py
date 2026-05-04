from pathlib import Path

import pytest

from geosave_engine.cli.paths import (
    get_plugin_templates,
    get_task_templates,
    plugins_template_dir,
    tasks_template_dir,
)


class TestGetTaskTemplates:
    """Test get_task_templates function."""

    def test_returns_dict(self):
        """Test that function returns a dict."""
        result = get_task_templates()
        assert isinstance(result, dict)

    def test_keys_are_task_names(self):
        """Test that dict keys are task directory names."""
        result = get_task_templates()
        tasks_dir = tasks_template_dir()
        expected_keys = {d.name for d in tasks_dir.iterdir() if d.is_dir()}
        assert set(result.keys()) == expected_keys

    def test_values_are_lists_of_paths(self):
        """Test that dict values are lists of Path objects."""
        result = get_task_templates()
        for task_name, templates in result.items():
            assert isinstance(templates, list)
            assert all(isinstance(p, Path) for p in templates)

    def test_templates_are_sorted(self):
        """Test that template paths are sorted."""
        result = get_task_templates()
        for task_name, templates in result.items():
            assert templates == sorted(templates)

    def test_excludes_file_exceptions(self):
        """Test that file exceptions are excluded."""
        file_exceptions = ["__pycache__"]
        result = get_task_templates(file_exceptions=file_exceptions)
        for task_name, templates in result.items():
            for template_path in templates:
                assert template_path.name not in file_exceptions

    def test_default_excludes_pycache(self):
        """Test that __pycache__ is excluded by default."""
        result = get_task_templates()
        for task_name, templates in result.items():
            for template_path in templates:
                assert template_path.name != "__pycache__"

    def test_semantic_segmentation_task_exists(self):
        """Test that semantic_segmentation task is present."""
        result = get_task_templates()
        assert "semantic_segmentation" in result
        assert len(result["semantic_segmentation"]) > 0


class TestGetPluginTemplates:
    """Test get_plugin_templates function."""

    def test_returns_dict(self):
        """Test that function returns a dict."""
        result = get_plugin_templates()
        assert isinstance(result, dict)

    def test_keys_are_plugin_names(self):
        """Test that dict keys are plugin directory names."""
        result = get_plugin_templates()
        plugins_dir = plugins_template_dir()
        expected_keys = {d.name for d in plugins_dir.iterdir() if d.is_dir()}
        assert set(result.keys()) == expected_keys

    def test_values_are_lists_of_paths(self):
        """Test that dict values are lists of Path objects."""
        result = get_plugin_templates()
        for plugin_name, templates in result.items():
            assert isinstance(templates, list)
            assert all(isinstance(p, Path) for p in templates)

    def test_templates_are_sorted(self):
        """Test that template paths are sorted."""
        result = get_plugin_templates()
        for plugin_name, templates in result.items():
            assert templates == sorted(templates)

    def test_excludes_file_exceptions(self):
        """Test that file exceptions are excluded."""
        file_exceptions = ["__pycache__"]
        result = get_plugin_templates(file_exceptions=file_exceptions)
        for plugin_name, templates in result.items():
            for template_path in templates:
                assert template_path.name not in file_exceptions

    def test_default_excludes_pycache(self):
        """Test that __pycache__ is excluded by default."""
        result = get_plugin_templates()
        for plugin_name, templates in result.items():
            for template_path in templates:
                assert template_path.name != "__pycache__"

    def test_semantic_segmentation_plugin_exists(self):
        """Test that semantic_segmentation plugin is present."""
        result = get_plugin_templates()
        assert "semantic_segmentation" in result
        assert len(result["semantic_segmentation"]) > 0
