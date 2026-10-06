import pytest

from geosave_engine.geodata.utils.color import parse_color


@pytest.mark.parametrize(
    ("color", "rgb"),
    [
        ("#ff8000", (255, 128, 0)),
        ("FF8000", (255, 128, 0)),
        ((255, 128, 0), (255, 128, 0)),
        ([255, 128, 0], (255, 128, 0)),
    ],
)
def test_parse_color_reads_hex_and_triples_as_an_rgb_tuple(color, rgb) -> None:
    assert parse_color(color) == rgb


@pytest.mark.parametrize(
    "color",
    ["#fff", "#ff80000", "#gg0000", "", (1, 2), [1, 2, 3, 4], (0, 0, 256), (-1, 0, 0)],
)
def test_parse_color_refuses_what_is_not_a_colour(color) -> None:
    with pytest.raises(ValueError, match="colour"):
        parse_color(color)
