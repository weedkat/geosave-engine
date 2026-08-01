from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from geosave_engine.geodata.datasets.base_dataset import BaseDataset, filter_by_split

TABLE_EXTENSIONS = (".csv", ".xlsx", ".parquet")


def to_plain(value: Any) -> Any:
    """Numpy/pandas scalar to plain Python value."""
    return value.item() if hasattr(value, "item") else value


def read_table(path: Path) -> pd.DataFrame:
    """Read a table file, dispatched by extension.

    Args:
        path: One of `TABLE_EXTENSIONS`.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".xlsx":
        return pd.read_excel(path)
    return pd.read_parquet(path)


class TableDataset(BaseDataset):
    """Table-backed dataset (csv/xlsx/parquet), one key per distinct `id_col` value.

    A key with one row gives scalar column values; several rows sharing an
    id give list column values instead. Numeric columns (int/float dtype)
    render as tensors; everything else stays a plain Python value/list.

    Args:
        path: One of `TABLE_EXTENSIONS`.
        id_col: Column identifying which rows belong to the same sample.
        sel_col: Columns to forward, besides `id_col`. None forwards every
            other column.
        split: Text file of `id_col` values to keep, one per line. None
            keeps every row in `path`.

    Raises:
        ValueError: `path`'s extension isn't in `TABLE_EXTENSIONS`, `id_col`
            isn't a column in the table, or `sel_col` names one that isn't.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        id_col: str,
        sel_col: list[str] | None = None,
        split: str | Path | None = None,
    ) -> None:
        self.split = split
        self.path = Path(path)

        if self.path.suffix.lower() not in TABLE_EXTENSIONS:
            raise ValueError(f"path must be one of {TABLE_EXTENSIONS}, got {self.path.suffix!r}")
        table = read_table(self.path)

        if id_col not in table.columns:
            raise ValueError(f"id_col '{id_col}' not found in columns: {list(table.columns)}")
        value_cols = [col for col in table.columns if col != id_col]
        if sel_col is not None:
            missing = set(sel_col) - set(value_cols)
            if missing:
                raise ValueError(f"sel_col names columns not in {self.path.name}: {sorted(missing)}")
            value_cols = sel_col

        self.numeric_cols = {col for col in value_cols if pd.api.types.is_numeric_dtype(table[col])}

        samples: dict[str, dict[str, Any]] = {}
        for key, group in table.groupby(id_col, sort=False):
            row: dict[str, Any] = {}
            for col in value_cols:
                row[col] = to_plain(group[col].iloc[0]) if len(group) == 1 else group[col].tolist()
            samples[str(key)] = row
        self.samples = filter_by_split(samples, split)
        self.reindex(self.samples)

    def render(self, key: str) -> dict[str, Any]:
        """Render one sample.

        Args:
            key: `id_col` value for this sample — one of `self.keys`.

        Returns:
            Dict of column values — numeric columns as `Tensor`, everything
            else as the plain Python value/list stored in `self.samples`.
        """
        sample: dict[str, Any] = {}
        for col, value in self.samples[key].items():
            sample[col] = torch.tensor(value) if col in self.numeric_cols else value
        return sample

    def to_row(self, key: str) -> dict[str, Any]:
        """Manifest row for `key` — the table row itself.

        Args:
            key: `id_col` value for this sample — one of `self.keys`.
        """
        return self.samples[key]
