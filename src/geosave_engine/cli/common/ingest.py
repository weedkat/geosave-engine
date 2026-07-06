from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from geosave_engine.geodata.core import source_from_dict
from geosave_engine.geodata.pipeline import PipelineRunner


def main(config_path: str, splits: list[str] | None = None) -> None:
    """Populate a workspace's raw cache + derived layers from an ingest config.

    Args:
        config_path: Path to an ``ingest.yaml``-shaped config (``cache_root``,
            ``requires``, ``pipeline``, ``labels``, ``data.root``, ``data.sources``).
        splits: Limit to these split names. ``None`` runs every split in
            ``data.sources``.
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    cache_root = Path(cfg["cache_root"])
    workspace_root = Path(cfg["data"]["root"])

    for split, src_dict in cfg["data"]["sources"].items():
        if splits and split not in splits:
            continue
        source = source_from_dict(src_dict)
        split_root = workspace_root / split
        # No ground truth for predict splits — skip labels: entirely.
        labels = cfg.get("labels", []) if split != "predict" else []
        PipelineRunner(cache_root, split_root, cfg.get("requires", []), cfg["pipeline"], labels).ingest_from(source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to ingest.yaml")
    parser.add_argument("--splits", nargs="*", default=None, help="Limit to these splits")
    args = parser.parse_args()
    main(args.config, args.splits)
