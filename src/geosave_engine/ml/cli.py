from lightning.pytorch.cli import LightningCLI, LightningArgumentParser

DEFAULT_EXPERIMENT_NAME = "default"
TENSORBOARD_LOGGER_CLASS_PATH = "lightning.pytorch.loggers.TensorBoardLogger"


class GeosaveCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument(
            "--model_name",
            type=str,
            default=DEFAULT_EXPERIMENT_NAME,
            help="Experiment name used by the auto-injected TensorBoardLogger.",
        )

    def before_instantiate_classes(self) -> None:
        cfg = self._subcommand_config()
        exp_name = self._resolve_model_name() or DEFAULT_EXPERIMENT_NAME

        tensorboard_logger = {
            "class_path": TENSORBOARD_LOGGER_CLASS_PATH,
            "init_args": {
                "name": exp_name,
                "save_dir": "artifacts",
                "log_graph": True,
            },
        }

        trainer_cfg = cfg.trainer
        current_logger = getattr(trainer_cfg, "logger", None)

        if current_logger is False:
            return  # respect explicit disable

        if current_logger in (None, True):
            trainer_cfg.logger = [tensorboard_logger]
            return

        # Single Namespace/dict logger: replace if it's the TB default,
        # otherwise pair it with our injected one.
        if not isinstance(current_logger, list):
            class_path = self._logger_class_path(current_logger)
            if class_path == TENSORBOARD_LOGGER_CLASS_PATH:
                # Mutate in place so any other parser defaults stay intact.
                init_args = getattr(current_logger, "init_args", current_logger)
                if hasattr(init_args, "name"):
                    init_args.name = exp_name
                if hasattr(init_args, "save_dir"):
                    init_args.save_dir = "artifacts"
                return
            trainer_cfg.logger = [current_logger, tensorboard_logger]
            return

        # List of loggers: ensure exactly one TB instance with our exp_name.
        tb_existing = [
            lg for lg in current_logger
            if self._logger_class_path(lg) == TENSORBOARD_LOGGER_CLASS_PATH
        ]
        if not tb_existing:
            current_logger.append(tensorboard_logger)
            return
        for lg in tb_existing:
            init_args = getattr(lg, "init_args", lg)
            if hasattr(init_args, "name") or (isinstance(init_args, dict) and "name" in init_args):
                if isinstance(init_args, dict):
                    init_args["name"] = exp_name
                    init_args["save_dir"] = "artifacts"
                else:
                    init_args.name = exp_name
                    init_args.save_dir = "artifacts"

    @staticmethod
    def _logger_class_path(logger) -> str | None:
        if isinstance(logger, dict):
            return logger.get("class_path")
        return getattr(logger, "class_path", None)

    def _resolve_model_name(self) -> str | None:
        """Find ``model_name`` regardless of whether jsonargparse parked it on
        the root namespace, the subcommand namespace, or as a dict key."""
        candidates = [self.config, self._subcommand_config()]
        for ns in candidates:
            if ns is None:
                continue
            value = getattr(ns, "model_name", None)
            if value:
                return str(value)
            if hasattr(ns, "get"):
                value = ns.get("model_name")
                if value:
                    return str(value)
        return None

    def _subcommand_config(self):
        """Return the active subcommand's config namespace.

        LightningCLI nests parsed config under ``self.config[subcommand]`` when
        invoked with subcommands (``fit``, ``test``, ...). Fall back to
        ``self.config`` for the no-subcommand case.
        """
        subcommand = getattr(self, "subcommand", None)
        if subcommand and subcommand in self.config:
            return self.config[subcommand]
        return self.config