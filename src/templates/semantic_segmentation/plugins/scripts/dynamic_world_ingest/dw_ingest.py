import warnings
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from geosave_engine.geodata.algorithms import (
    build_shadow_mask,
    compute_b10_mask,
    compute_cdi_mask,
    compute_s2c_mask,
)
from geosave_engine.geodata.pipeline import (
    BandMeta,
    ClassMeta,
    Derived,
    ManifestWriter,
    MaskMeta,
    Pipeline,
    Source,
    Anchor,
)
from geosave_engine.geodata.stac import StacClient
from geosave_engine.utils.geodata import spatial_da

# --- Internal Registry / Config ---
L1C_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", 
    "B8A", "B09", "B10", "B11", "B12"
]

sentinel_client = StacClient.cdse()
load_dotenv()

def cloud_compute_fn(cache):
    """
    Consolidated compute function to minimize redundant .median() calls
    and improve performance during the ingest pipeline.
    """
    source = cache["sentinel_2_l1c"]
    ds = source.ds.median("time")
    sun_az = source.items[0].properties["view:sun_azimuth"]

    s2c = compute_s2c_mask(
        b01=ds["B01"].values, b02=ds["B02"].values, b04=ds["B04"].values,
        b05=ds["B05"].values, b08=ds["B08"].values, b8a=ds["B8A"].values,
        b09=ds["B09"].values, b10=ds["B10"].values, b11=ds["B11"].values,
        b12=ds["B12"].values, cloud_threshold=0.65
    )
    cdi = compute_cdi_mask(b07=ds["B07"].values, b08=ds["B08"].values, b8a=ds["B8A"].values)
    cirrus = compute_b10_mask(b10=ds["B10"].values, b10_threshold=0.0012)

    cloud_combined = s2c & cdi & cirrus
    shadow = build_shadow_mask(cloud_combined, sun_azimuth_deg=sun_az)
    final_mask = cloud_combined | shadow

    return spatial_da(final_mask, ds)

def create_pipeline(tiff_path):
    anchor = Anchor.from_tiff(tiff_path)
    
    source = Source.sentinel_2_l1c(
        name="sentinel_2_l1c",
        client=sentinel_client,
        time_range=timedelta(days=1),
        bands=L1C_BANDS,
    )

    derived_layers = [
        Derived.label_from_anchor(name="dynamicworld", remap={
            0: 255,  # nodata
            1: 0,    # water
            2: 1,    # trees
            3: 2,    # grass
            4: 3,    # flooded_vegetation
            5: 4,    # crops
            6: 5,    # shrub_and_scrub
            7: 6,    # built
            8: 7,    # bare
            9: 255,  # snow_and_ice -> ignore_index
            10: 255, # cloud -> ignore_index
        }),

        # All bands except for B1, B8A, B9, and B10 were kept, as they are not used in the original Dynamic World model and can be noisy. --- IGNORE ---
        Derived.image_from_source(name="sentinel_2_l1c", source="sentinel_2_l1c", bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]),
        Derived.from_cache(
            name="cloud_mask", 
            compute_fn=cloud_compute_fn, 
            sources=["sentinel_2_l1c"]
        ),
    ]

    return Pipeline(anchor=anchor, sources=[source], deriveds=derived_layers)

def run_ingest(raw_data_root: Path, output_dir: Path):
    manifest = ManifestWriter(output_dir)
    
    # --- Metadata Initialization ---
    manifest.create_layer(
        name="sentinel_2_l1c", 
        role="image", 
        meta=[BandMeta(name=b) for b in L1C_BANDS]
    )
    
    cloud_meta = [
        MaskMeta(id=0, name="clear", color="#ffffff"),
        MaskMeta(id=1, name="cloud_or_shadow", color="#000000"),
    ]
    manifest.create_layer(name="cloud_mask", role="mask", meta=cloud_meta)

    dw_meta = [
        ClassMeta(id=0, name="water", color="#419BDF"),
        ClassMeta(id=1, name="trees", color="#397D49"),
        ClassMeta(id=2, name="grass", color="#88B053"),
        ClassMeta(id=3, name="flooded_vegetation", color="#7A87C6"),
        ClassMeta(id=4, name="crops", color="#E49635"),
        ClassMeta(id=5, name="shrub_and_scrub", color="#DFC35A"),
        ClassMeta(id=6, name="built", color="#C4281B"),
        ClassMeta(id=7, name="bare", color="#A59B8F"),
        ClassMeta(id=255, name="ignore_index", color="#000000"),
    ]
    manifest.create_layer(name="dynamicworld", role="label", meta=dw_meta)

    # --- Processing Loop ---
    # Iterate through splits: train, test, val
    for split in ["train", "test", "val"]:
        split_path = raw_data_root / split
        if not split_path.exists():
            raise FileNotFoundError(f"Split directory not found: {split_path}")
        
        if split == "test":
            tifs = list(split_path.glob("label_*.tif"))
        else:
            tifs = list(split_path.rglob("*.tif"))

        for tiff_path in tqdm(tifs, desc=f"{split} ingest", unit="tile"):
            source_id = tiff_path.stem
            if manifest.is_written(source_id):
                tqdm.write(f"Skipping {source_id} (already ingested)")
                continue

            try:
                pipeline = create_pipeline(tiff_path)
                result = pipeline.run()
                manifest.write_tile(result, split=split, source_id=source_id)
            except Exception as e:
                warnings.warn(f"Failed to process {source_id}: {e}")

if __name__ == "__main__":
    DATA_ROOT = Path("data/dynamicworld_raw/").resolve()
    OUT_DIR = Path("data/").resolve()
    run_ingest(DATA_ROOT, OUT_DIR)