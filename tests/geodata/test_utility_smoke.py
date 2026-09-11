import os

from geosave_engine.geodata import configure_gdal
from geosave_engine.geodata.utils.geo import geolocator


def test_gdal_configuration_sets_supplied_values(monkeypatch) -> None:
    monkeypatch.delenv("GDAL_HTTP_MAX_RETRY", raising=False)
    monkeypatch.delenv("AWS_NO_SIGN_REQUEST", raising=False)

    configure_gdal(aws_no_sign_request=True, gdal_http_max_retry=3)

    assert os.environ["AWS_NO_SIGN_REQUEST"] == "TRUE"
    assert os.environ["GDAL_HTTP_MAX_RETRY"] == "3"


def test_geolocator_uses_rounded_coordinates(monkeypatch) -> None:
    seen: list[tuple[float, float]] = []

    def request_address(latitude: float, longitude: float) -> dict[str, str]:
        seen.append((latitude, longitude))
        return {"city": "Malang"}

    monkeypatch.setattr(geolocator, "_request_nominatim_address", request_address)

    assert geolocator.reverse_geocode(-8.051, 112.149) == {"city": "Malang"}
    assert seen == [(-8.05, 112.15)]
