from geosave_engine.geodata.utils import geolocator


def test_geolocator_uses_rounded_coordinates(monkeypatch) -> None:
    seen: list[tuple[float, float]] = []

    def request_address(latitude: float, longitude: float) -> dict[str, str]:
        seen.append((latitude, longitude))
        return {"city": "Malang"}

    monkeypatch.setattr(geolocator, "_request_nominatim_address", request_address)

    assert geolocator.reverse_geocode(-8.051, 112.149) == {"city": "Malang"}
    assert seen == [(-8.05, 112.15)]
