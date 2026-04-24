from __future__ import annotations

from lightning.pytorch.cli import LightningCLI, LightningArgumentParser


DEFAULT_EXPERIMENT_NAME = "default"
TENSORBOARD_LOGGER_CLASS_PATH = "lightning.pytorch.loggers.TensorBoardLogger"


class GeosaveCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # parser.link_arguments("data.patch_size", "model.patch_size")
        # parser.link_arguments("data.batch_size", "model.batch_size")
        pass

    def before_instantiate_classes(self) -> None:
        exp_name = getattr(self.config, "model_name", DEFAULT_EXPERIMENT_NAME)

        tensorboard_logger = {
            "class_path": TENSORBOARD_LOGGER_CLASS_PATH,
            "init_args": {
                "name": exp_name,
                "save_dir": "artifacts",
                "log_graph": True,
            },
        }

        trainer_cfg = self.config.trainer
        current_logger = getattr(trainer_cfg, "logger", None)

        if current_logger is False:
            return  # respect explicit disable

        if current_logger in (None, True):
            trainer_cfg.logger = [tensorboard_logger]
            return

        if isinstance(current_logger, list):
            already_present = any(
                isinstance(logger, dict)
                and logger.get("class_path") == TENSORBOARD_LOGGER_CLASS_PATH
                for logger in current_logger
            )
            if not already_present:
                current_logger.append(tensorboard_logger)
            return

        if isinstance(current_logger, dict) and current_logger.get("class_path") == TENSORBOARD_LOGGER_CLASS_PATH:
            return

        trainer_cfg.logger = [current_logger, tensorboard_logger]