from lightning.pytorch.cli import LightningCLI
from src.data_module import GeosaveDataModule
from src.lightning_module import GeosaveLightningModule


def cli_main():
    LightningCLI(
        GeosaveLightningModule,
        GeosaveDataModule
    )

if __name__ == "__main__":
    cli_main()

