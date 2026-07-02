from geosave_engine.ml.core.cli import GeosaveCLI
from src.data_module import GeosaveDataModule # noqa: F401
from src.lightning_module import GeosaveLightningModule # noqa: F401


if __name__ == "__main__":
    GeosaveCLI(
        model_class=GeosaveLightningModule,
        datamodule_class=GeosaveDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
    )
