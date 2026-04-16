from lightning.pytorch.cli import LightningCLI
from src.data_module import GeosaveDataModule
from src.lightning_module import GeosaveLightningModule

class GeosaveCLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        # Access the config passed via CLI or YAML
        subcommand = self.config.subcommand
        conf = self.config[subcommand]
        
        # If the user changed the model name in the config, 
        # ensure the logger uses that same name.
        if hasattr(conf.model, "name"):
            conf.trainer.logger.init_args.name = conf.model.name


cli = GeosaveCLI(model_class=GeosaveLightningModule)

def cli_main():
    LightningCLI(
        GeosaveLightningModule,
        GeosaveDataModule
    )

if __name__ == "__main__":
    cli_main()

