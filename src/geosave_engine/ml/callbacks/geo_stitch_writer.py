"""Stitching predicted tiles back onto the rasters they were cut from."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lightning.pytorch.callbacks import Callback

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike

    import xarray as xr
    from lightning.pytorch import LightningModule, Trainer

    from geosave_engine.geodata.core.stitcher import StitchWindow


class GeoStitchWriter(Callback):
    """Rebuild predicted tiles into rasters during `Trainer.predict` and write them.

    Each sample states its own row in `tiles` under `"index"`, naming the tile
    whose grid and tiling record place its prediction. A group is written once
    it holds every tile. Predict on one device, because ranks shard tiles.

    Args:
        tiles: Tiles the predict dataset reads, indexed by the row each sample
            states. Tiles of several rasters may share one list.
        destination: Directory each finished raster is written into, named
            after its own `Tiling.group_id`.
        key: Prediction to stitch, among what `predict_step` returns.
        variable: Data variable name the written raster carries.
        nodata: Value marking pixels no tile covered.
        window: Weighting applied across each tile before overlaps are
            averaged. None weighs every tile alike. A class map is stitched
            from tiles cut without overlap, because averaging class codes
            invents codes no tile predicted.

    Examples:
        >>> tiles = scene.gs.tile((256, 256), group_id="scene-001")
        >>> writer = GeoStitchWriter(tiles, "predictions/")
        >>> Trainer(callbacks=[writer], devices=1).predict(
        ...     task, loader, return_predictions=False
        ... )
    """

    def __init__(
        self,
        tiles: Sequence[xr.Dataset],
        destination: str | PathLike[str],
        *,
        key: str = "pred",
        variable: str = "class",
        nodata: float | int | None = None,
        window: StitchWindow | None = None,
    ) -> None:
        """Start a writer holding no accumulator.

        Args:
            tiles: Tiles the predict dataset reads, indexed by the row each
                sample states.
            destination: Directory each finished raster is written into.
            key: Prediction to stitch, among what `predict_step` returns.
            variable: Data variable name the written raster carries.
            nodata: Value marking pixels no tile covered.
            window: Weighting applied across each tile before overlaps are
                averaged.

        Raises:
            ValueError: `window` names no window `tiler` supports.
        """
        raise NotImplementedError

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Place this batch's predictions and write every group they complete.

        Raises:
            KeyError: `batch` states no `"index"`, or `outputs` carries no
                `key`.
            ValueError: A row names no tile, the tile states no tiling, or the
                prediction disagrees with its group's grid, dtype, or fill
                value.
        """
        raise NotImplementedError

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Write every group still held, complete or not.

        Raises:
            ValueError: A group is missing tiles and states no fill value to
                mark the holes they leave.
        """
        raise NotImplementedError
