"""Disposable offline prediction experiment; not a GeoSave public API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from affine import Affine
import geopandas as gpd
import numpy as np
from odc.geo.geobox import GeoBox
import pandas as pd
from prefect import flow, task
from scipy.signal.windows import hann
from shapely.geometry import Polygon
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.ops import MultiScaleRoIAlign, batched_nms

from geosave_engine.geodata import GeoVector, io, raster
from geosave_engine.ml.inputs import model_inputs
from geosave_engine.ml.segmentation.transforms import softmax_argmax
from geosave_engine.model.chain import ModelChain, chain_step
from geosave_engine.model.head import (
    ClassificationHead,
    DetectionHead,
    RegressionHead,
    SegmentationHead,
)
from geosave_engine.model.spec import ModelSpec
from geosave_engine.model.spec.call import Ref


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    @chain_step(outputs=("feature_map", "pyramid"))
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = self.conv(image)
        return features, [features]


class TinyBackbone(nn.Module):
    out_channels = 4

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, image):
        return {"0": self.conv(image)}


def native_detector():
    return FasterRCNN(
        TinyBackbone(),
        min_size=16,
        max_size=16,
        rpn_anchor_generator=AnchorGenerator(
            sizes=((4, 8),), aspect_ratios=((0.5, 1.0, 2.0),)
        ),
        box_roi_pool=MultiScaleRoIAlign(["0"], output_size=3, sampling_ratio=1),
        box_head=TwoMLPHead(4 * 3 * 3, 16),
        box_predictor=FastRCNNPredictor(16, 3),
        rpn_pre_nms_top_n_test=30,
        rpn_post_nms_top_n_test=15,
        box_detections_per_img=8,
        box_score_thresh=0.0,
        image_mean=[0.0] * 3,
        image_std=[1.0] * 3,
    )


class StructuredDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = native_detector()

    @chain_step(head=True)
    def forward(self, image: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        return self.model(list(image))


def models():
    torch.manual_seed(41)
    dense = ModelChain(
        features=TinyEncoder(),
        cover=SegmentationHead(classes=["water", "land"], feature_channels=4),
        amount=RegressionHead(variables=["quantity"], feature_channels=4, units="m"),
    ).eval()
    classification = ModelChain(
        features=TinyEncoder(),
        category=ClassificationHead(classes=["water", "land"], pyramid_channels=[4]),
    ).eval()
    detection = ModelChain(objects=StructuredDetector()).eval()
    return dense, classification, detection


def fixture(root):
    root = Path(root)
    grid = GeoBox((12, 16), Affine(10, 0, 300000, 0, -10, 5000000), "EPSG:32633")
    values = np.random.default_rng(9).random((3, 12, 16), dtype=np.float32)
    values[:, 3, 5] = np.nan
    paths = {}
    for name, geobox, pixels in [
        ("scene_a", grid, values),
        ("scene_b", grid, values * 0.5),
        ("pixel_only", None, values * 0.75),
    ]:
        source = raster(
            dict(zip(["red", "green", "blue"], pixels, strict=True)),
            geobox,
            nodata=np.nan,
        )
        path = root / f"{name}{'.nc' if geobox is None else '.tif'}"
        if geobox is None:
            io.netcdf.write(source, path)
        else:
            source.gs.to_cog(path)
        paths[name] = str(path)
    (root / "parents.json").write_text(json.dumps(paths))
    spec = ModelSpec(
        schema_version=2,
        rasters={
            "image": {"variables": ["red", "green", "blue"], "require_crs": False}
        },
        tiles={"size": 8, "overlap": 4, "window": "hann"},
        inputs={"image": Ref("image")},
    )
    spec.save(root / "model_spec.yaml")


@task(name="probe-prepare-reference", persist_result=False)
def prepare_reference(root: str) -> str:
    folder = Path(root)
    paths = json.loads((folder / "parents.json").read_text())
    spec = ModelSpec.load(folder / "model_spec.yaml")
    rows = []
    for parent_id, path in paths.items():
        with io.read_raster(path) as source:
            prepared = spec.preprocess({"image": source})["image"]
            shape = tuple(prepared.sizes[d] for d in prepared.gs.grid_dims)
            layout = spec.tiles.layout(shape)
            padded_shape, padding = layout.calculate_padding()
            layout.recalculate(data_shape=padded_shape)
            reference = GeoVector.from_layouts({parent_id: prepared}, {parent_id: layout}, padding={parent_id: padding})
            rows.extend(reference.to_dict("records"))
    path = folder / "reference.parquet"
    io.geoparquet.write(gpd.GeoDataFrame(rows, crs="EPSG:4326"), path, index=False)
    return str(path)


def read_record_inputs(parent, row, spec):
    reference = gpd.GeoDataFrame([row], geometry="geometry", crs="EPSG:4326" if row.geometry is not None else None)
    pixels = row.gs.crop(parent)
    inputs = model_inputs(spec, {"image": pixels}, row)
    image = inputs["image"]
    valid = torch.isfinite(image).all(dim=0)
    return {"image": torch.nan_to_num(image)}, valid


@task(name="probe-predict-native-models", persist_result=False)
def infer(root: str, reference_path: str) -> dict:
    folder = Path(root)
    paths = json.loads((folder / "parents.json").read_text())
    spec = ModelSpec.load(folder / "model_spec.yaml")
    reference = io.geoparquet.read(reference_path).set_index(
        "id", drop=False, verify_integrity=True
    )
    parents = {name: io.read_raster(path) for name, path in paths.items()}
    samples = []
    for row in reference.itertuples(index=False):
        inputs, valid = read_record_inputs(parents[row.parent_id], row, spec)
        samples.append((inputs, valid, row.id))
    dense, classification, detection = models()
    loader = DataLoader(
        samples, batch_size=3, shuffle=True, generator=torch.Generator().manual_seed(8)
    )
    dense_rows, class_rows, detections, completed_tiles = [], [], [], []
    expected = {}
    with torch.inference_mode():
        for parent_id, parent in parents.items():
            image = torch.nan_to_num(parent.gs.to_tensor()).unsqueeze(0)
            expected[parent_id] = {
                name: value[0].numpy() for name, value in dense(image=image).items()
            }
        for inputs, validity, ids in loader:
            outputs = dense(**inputs)
            categories = classification(**inputs)
            objects = detection(**inputs)
            assert isinstance(outputs, dict) and set(outputs) == {"cover", "amount"}
            assert len(objects) == len(ids)
            for position, key in enumerate(ids):
                row = reference.loc[key]
                transform = row["proj:transform"]
                grid = (
                    None
                    if transform is None
                    else GeoBox(
                        tuple(row["proj:shape"]), Affine(*transform), row["proj:code"]
                    )
                )
                for name, channels in [
                    ("cover", ["water", "land"]),
                    ("amount", ["quantity"]),
                ]:
                    values = outputs[name][position].numpy().copy()
                    values[:, ~validity[position].numpy()] = np.nan
                    result = raster(
                        dict(zip(channels, values, strict=True)), grid, nodata=np.nan
                    )
                    path = (
                        folder
                        / f"prediction-{len(dense_rows)}_{name}{'.nc' if grid is None else '.tif'}"
                    )
                    if grid is None:
                        io.netcdf.write(result, path)
                    else:
                        result.gs.to_cog(path)
                    dense_rows.append(
                        {"tile_id": key, "output": name, "asset": path.name}
                    )
                probabilities = categories[position].softmax(0)
                class_rows.append(
                    {
                        "tile_id": key,
                        "class_id": int(probabilities.argmax()),
                        "score": float(probabilities.max()),
                        "geometry": row.geometry,
                    }
                )
                prediction = objects[position]
                completed_tiles.append(
                    {"tile_id": key, "detection_count": len(prediction["boxes"])}
                )
                assert prediction["boxes"].shape[-1] == 4
                assert (prediction["boxes"] >= 0).all() and (
                    prediction["boxes"] <= 8
                ).all()
                for box, label, score in zip(
                    prediction["boxes"],
                    prediction["labels"],
                    prediction["scores"],
                    strict=True,
                ):
                    detections.append(
                        {
                            "tile_id": key,
                            "parent_id": row.parent_id,
                            "box": box.tolist(),
                            "label": int(label),
                            "score": float(score),
                        }
                    )
    # No live cut is returned or retained by the assembler.
    for parent in parents.values():
        parent.close()
    pd.DataFrame(dense_rows).to_parquet(folder / "raw_rasters.parquet", index=False)
    io.geoparquet.write(
        gpd.GeoDataFrame(class_rows, crs="EPSG:4326"),
        folder / "classification.parquet",
        index=False,
    )
    pd.DataFrame(detections).to_parquet(folder / "raw_detections.parquet", index=False)
    pd.DataFrame(completed_tiles).to_parquet(
        folder / "completed_tiles.parquet", index=False
    )
    return {
        "expected": expected,
        "tiles": len(reference),
        "raw_detections": len(detections),
        "detector_parameters": sum(p.numel() for p in detection.parameters()),
    }


def accumulate(rows, arrays, shape, window):
    total = np.zeros((arrays[0].shape[0], *shape), dtype="float64")
    weight = np.zeros_like(total)
    for row, values in zip(rows, arrays, strict=True):
        yy, xx = row.row_off, row.col_off
        y0, y1 = max(0, yy), min(shape[0], yy + row.height)
        x0, x1 = max(0, xx), min(shape[1], xx + row.width)
        src = (slice(y0 - yy, y1 - yy), slice(x0 - xx, x1 - xx))
        dst = (slice(y0, y1), slice(x0, x1))
        taper = (
            np.ones((row.height, row.width))
            if window == "boxcar"
            else np.outer(hann(row.height, sym=False), hann(row.width, sym=False))
        )
        valid = np.isfinite(values[:, src[0], src[1]])
        effective = taper[src][None] * valid
        total[:, dst[0], dst[1]] += np.nan_to_num(values[:, src[0], src[1]]) * effective
        weight[:, dst[0], dst[1]] += effective
    result = np.full(total.shape, np.nan, dtype="float32")
    np.divide(total, weight, out=result, where=weight > 0, casting="unsafe")
    return result, weight


def check_assets(assets, reference):
    expected = {(key, name) for key in reference.index for name in ("cover", "amount")}
    actual = set(zip(assets.tile_id, assets.output, strict=True))
    if assets.duplicated(["tile_id", "output"]).any() or actual != expected:
        raise ValueError("Missing or repeated prediction records")


@task(name="probe-assemble-dense", persist_result=False)
def assemble_dense(root: str, reference_path: str, inference: dict) -> dict:
    folder = Path(root)
    reference = io.geoparquet.read(reference_path).set_index("id", drop=False)
    assets = pd.read_parquet(folder / "raw_rasters.parquet").sample(
        frac=1, random_state=5
    )
    check_assets(assets, reference)
    for invalid in (
        pd.concat([assets, assets.iloc[:1]], ignore_index=True),
        assets.loc[
            ~assets.tile_id.isin(reference.loc[reference.parent_id == "scene_a", "id"])
        ],
    ):
        try:
            check_assets(invalid, reference)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid completion records accepted")
    parents = json.loads((folder / "parents.json").read_text())
    outputs = {}
    for parent_id, path in parents.items():
        with io.read_raster(path) as parent:
            shape = tuple(parent.sizes[dim] for dim in parent.gs.grid_dims)
            mask = np.isfinite(parent.gs.to_numpy()).all(axis=0)
            products = {}
            for name in ("cover", "amount"):
                selected = assets.loc[
                    (assets.output == name)
                    & assets.tile_id.isin(
                        reference.loc[reference.parent_id == parent_id, "id"]
                    )
                ]
                rows = reference.loc[selected.tile_id.tolist()]
                assert selected.tile_id.is_unique
                assert set(selected.tile_id) == set(
                    reference.loc[reference.parent_id == parent_id, "id"]
                )
                arrays = []
                for asset in selected.asset:
                    with io.read_raster(folder / asset) as tile:
                        arrays.append(tile.gs.to_numpy())
                for window in ("boxcar", "hann"):
                    combined, weights = accumulate(
                        list(rows.itertuples()), arrays, shape, window
                    )
                    expected = inference["expected"][parent_id][name]
                    np.testing.assert_allclose(
                        combined[:, mask], expected[:, mask], atol=1e-6
                    )
                    assert np.isnan(combined[:, ~mask]).all()
                    assert (weights[:, mask] > 0).all()
                products[name] = combined
            labels, confidence = softmax_argmax(
                torch.from_numpy(products["cover"]).unsqueeze(0)
            )
            labels = labels[0].numpy().astype("uint8")
            labels[~mask] = 255
            result = raster({"label": labels}, parent.gs.geobox, nodata=255)
            target = (
                folder
                / f"{parent_id}_labels{'.nc' if parent.gs.geobox is None else '.tif'}"
            )
            if parent.gs.geobox is None:
                io.netcdf.write(result, target)
            else:
                result.gs.to_cog(target)
            with io.read_raster(target) as restored:
                assert (restored.gs.geobox is None) == (parent.gs.geobox is None)
                if parent.gs.geobox is not None:
                    assert restored.gs.geobox == parent.gs.geobox
                np.testing.assert_array_equal(restored.label.values, labels)
            reg = raster(
                {"quantity": products["amount"][0]}, parent.gs.geobox, nodata=np.nan
            )
            reg.quantity.attrs["units"] = "m"
            reg_path = folder / f"{parent_id}_regression.nc"
            io.netcdf.write(reg, reg_path)
            with io.read_raster(reg_path) as restored:
                assert restored.quantity.attrs["units"] == "m"
                assert (restored.gs.geobox is None) == (parent.gs.geobox is None)
                np.testing.assert_allclose(
                    restored.quantity.values, products["amount"][0], equal_nan=True
                )
            outputs[parent_id] = {
                "labels": str(target),
                "regression": str(reg_path),
                "valid_pixels": int(mask.sum()),
            }
    classification = io.geoparquet.read(folder / "classification.parquet")
    assert classification.tile_id.is_unique and set(classification.tile_id) == set(
        reference.index
    )
    assert (
        classification.geometry.isna().sum()
        == (reference.parent_id == "pixel_only").sum()
    )
    return outputs


def parent_boxes(records, reference):
    converted = []
    for row in records.itertuples(index=False):
        tile = reference.loc[row.tile_id]
        x1, y1, x2, y2 = row.box
        converted.append(
            [x1 + tile.col_off, y1 + tile.row_off, x2 + tile.col_off, y2 + tile.row_off]
        )
    return torch.tensor(converted, dtype=torch.float32).reshape(-1, 4)


@task(name="probe-assemble-detections", persist_result=False)
def assemble_detection(root: str, reference_path: str) -> dict:
    folder = Path(root)
    reference = io.geoparquet.read(reference_path).set_index("id", drop=False)
    raw = pd.read_parquet(folder / "raw_detections.parquet")
    completed = pd.read_parquet(folder / "completed_tiles.parquet")
    assert completed.tile_id.is_unique and set(completed.tile_id) == set(
        reference.index
    )
    assert completed.detection_count.sum() == len(raw)
    paths = json.loads((folder / "parents.json").read_text())
    result_rows = []
    for parent_id, path in paths.items():
        records = raw.loc[raw.parent_id == parent_id]
        boxes = parent_boxes(records, reference)
        scores = torch.tensor(records.score.to_numpy(), dtype=torch.float32)
        labels = torch.tensor(records.label.to_numpy(), dtype=torch.int64)
        with io.read_raster(path) as parent:
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, parent.sizes["x"])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, parent.sizes["y"])
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
            keep = batched_nms(boxes, scores, labels, 0.5)
            for index in keep.tolist():
                x1, y1, x2, y2 = boxes[index].tolist()
                geometry = None
                if parent.gs.geobox is not None:
                    transform = parent.gs.geobox.transform
                    polygon = Polygon(
                        [
                            transform * point
                            for point in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                        ]
                    )
                    geometry = (
                        gpd.GeoSeries([polygon], crs=parent.gs.geobox.crs.to_wkt())
                        .to_crs("EPSG:4326")
                        .iloc[0]
                    )
                result_rows.append(
                    {
                        "parent_id": parent_id,
                        "box": [x1, y1, x2, y2],
                        "label": int(labels[index]),
                        "score": float(scores[index]),
                        "geometry": geometry,
                    }
                )
    result = gpd.GeoDataFrame(
        result_rows,
        columns=["parent_id", "box", "label", "score", "geometry"],
        geometry="geometry",
        crs="EPSG:4326",
    )
    path = folder / "detections.parquet"
    io.geoparquet.write(result, path, index=False)
    restored = io.geoparquet.read(path)
    assert restored.loc[restored.parent_id == "pixel_only"].geometry.isna().all()
    assert restored.loc[restored.parent_id != "pixel_only"].geometry.notna().all()
    # Independent controlled cross-tile duplicate test, not a model-quality claim.
    selected = reference.loc[
        (reference.parent_id == "scene_a")
        & (reference.row_off == -2)
        & reference.col_off.isin([-2, 2])
    ].sort_values("col_off")
    first, second = selected.index.tolist()
    controlled = pd.DataFrame(
        [
            {"tile_id": first, "box": [4, 4, 6, 6]},
            {"tile_id": second, "box": [0, 4, 2, 6]},
            {"tile_id": first, "box": [4, 4, 6, 6]},
        ]
    )
    boxes = parent_boxes(controlled, reference)
    assert torch.equal(boxes, torch.tensor([[2, 2, 4, 4]] * 3, dtype=torch.float32))
    keep = batched_nms(
        boxes, torch.tensor([0.9, 0.8, 0.7]), torch.tensor([1, 1, 2]), 0.5
    )
    assert keep.tolist() == [0, 2]
    assert (
        batched_nms(
            torch.empty((0, 4)), torch.empty(0), torch.empty(0, dtype=torch.int64), 0.5
        ).numel()
        == 0
    )
    assert (
        len(
            gpd.GeoDataFrame(
                [], columns=["box", "geometry"], geometry="geometry", crs="EPSG:4326"
            )
        )
        == 0
    )
    return {
        "path": str(path),
        "raw_records": len(raw),
        "assembled_records": len(restored),
        "class_aware_duplicate_and_empty_tests": True,
    }


def assess_model_contracts():
    dense, classifier, detector = models()
    image = torch.rand(2, 3, 8, 8)
    with torch.inference_mode():
        assert set(dense(image=image)) == {"cover", "amount"}
        assert classifier(image=image).shape == (2, 2)
        assert len(detector(image=image)) == 2
        detector.objects.model.roi_heads.score_thresh = 1.0
        empty = detector(image=image)
        assert all(value["boxes"].shape == (0, 4) for value in empty)
        raw = ModelChain(
            features=TinyEncoder(),
            objects=DetectionHead(
                classes=["tree"],
                pyramid_channels=[4],
                pyramid_strides=[1],
                hidden_channels=4,
            ),
        ).eval()(image=image)
        assert raw.shape == (2, 64, 5)
        mixed = ModelChain(
            features=TinyEncoder(),
            cover=SegmentationHead(classes=["water", "land"], feature_channels=4),
            objects=StructuredDetector(),
        ).eval()
        result = mixed(image=image)
        assert set(result) == {"cover", "objects"}
        assert isinstance(result["cover"], torch.Tensor)
        assert len(result["objects"]) == 2
    return {
        "tensor_multihead": True,
        "native_structured_terminal": True,
        "raw_geosave_detection_shape": list(raw.shape),
        "mixed_structured_outputs": True,
    }


@flow(name="prediction-design-probe", persist_result=False)
def predict_probe(root: str) -> dict:
    reference = prepare_reference(root)
    inference = infer(root, reference)
    dense = assemble_dense(root, reference, inference)
    detection = assemble_detection(root, reference)
    contracts = assess_model_contracts()
    return {
        "root": root,
        "reference": reference,
        "tiles": inference["tiles"],
        "detector_parameters": inference["detector_parameters"],
        "dense": dense,
        "classification": str(Path(root) / "classification.parquet"),
        "detection": detection,
        "model_contracts": contracts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    args = parser.parse_args()
    root = (
        Path(args.root)
        if args.root
        else Path(tempfile.mkdtemp(prefix="geosave-full-predict-", dir="/tmp"))
    )
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    fixture(root)
    report = predict_probe(str(root))
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
