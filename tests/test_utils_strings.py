from __future__ import annotations

import pytest

from geosave_engine.utils.strings import (
    normalize_slug,
    parse_shell_args,
    resolve_from_choices,
)


def test_normalize_slug_collapses_separators_and_case():
    assert normalize_slug("Semantic_Segmentation") == "semantic segmentation"
    assert normalize_slug("object-detection") == "object detection"
    assert normalize_slug("  Foo   Bar  ") == "foo bar"


def test_resolve_from_choices_matches_normalized_forms():
    choices = ["semantic segmentation", "object detection"]
    assert resolve_from_choices("Semantic_Segmentation", choices) == "semantic segmentation"
    assert resolve_from_choices("object-detection", choices) == "object detection"


def test_resolve_from_choices_returns_none_when_no_match():
    assert resolve_from_choices("unknown-task", ["a", "b"]) is None
    assert resolve_from_choices("", ["a", "b"]) is None


def test_parse_shell_args_handles_quoting_and_empty_input():
    assert parse_shell_args("") == []
    assert parse_shell_args("   ") == []
    assert parse_shell_args("--foo bar --baz='hello world'") == [
        "--foo",
        "bar",
        "--baz=hello world",
    ]


def test_parse_shell_args_bubbles_shlex_errors():
    with pytest.raises(ValueError):
        parse_shell_args("unterminated \"quote")
