from lightning.pytorch.cli import LightningCLI

from src.data_module import GeosaveDataModule
from src.lightning_module import GeosaveLightningModule


class GeosaveCLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        subcommand = self.config.subcommand
        conf = self.config[subcommand]

        if hasattr(conf.model, "name"):
            conf.trainer.logger.init_args.name = conf.model.name


if __name__ == "__main__":
    GeosaveCLI(model_class=GeosaveLightningModule, datamodule_class=GeosaveDataModule)
