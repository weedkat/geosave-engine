import numpy as np
from rasterio.io import MemoryFile

from geosave_engine.geodata.attrs import CFVariable, GDALVariable
from geosave_engine.geodata.attrs.headers.gdal import create_header


def test_gdal_factory_combines_band_slots_and_tags() -> None:
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff", width=1, height=1, count=1, dtype="uint8"
        ) as dst:
            dst.write(np.array([[[1]]], dtype="uint8"))
            dst.set_band_description(1, "Red reflectance")
            dst.update_tags(1, variable_name="red", units="1")

        with memory.open() as src:
            header = create_header(src)

    namespace = header.data_vars["red"]
    assert namespace.get(GDALVariable).variable_name == "red"
    assert namespace.get(CFVariable) == CFVariable(
        long_name="Red reflectance", units="1"
    )
