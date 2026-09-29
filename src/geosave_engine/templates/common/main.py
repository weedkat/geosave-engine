from geosave_engine.ml.lightning.cli import GeosaveCLI


if __name__ == "__main__":
    GeosaveCLI(
        auto_configure_optimizers=False,
        subclass_mode_model=True,
        subclass_mode_data=True,
    )
