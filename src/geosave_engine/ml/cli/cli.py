from __future__ import annotations

from lightning.pytorch.cli import LightningCLI, LightningArgumentParser


class GeosaveCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # parser.link_arguments("data.patch_size", "model.patch_size")
        # parser.link_arguments("data.batch_size", "model.batch_size")
        pass
