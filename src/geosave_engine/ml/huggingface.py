"""Optional Transformers AutoModel integration for publishable model chains."""

from __future__ import annotations

from typing import Any, TypedDict

import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.initialization import no_init_weights

from geosave_engine.ml.models.contract import ModelChain
from geosave_engine.ml.registry.base import BuildSpec


class StageConfig(TypedDict):
    """One named stage in an ordered Transformers configuration."""

    stage: str
    spec: BuildSpec


class GeoSaveConfig(PretrainedConfig):
    """Describe a chain in a Transformers model artifact.

    Args:
        stages: Ordered stage names and construction specifications.
        **kwargs: Standard Transformers configuration fields.

    Raises:
        ValueError: Stage names are repeated.
    """

    model_type = "geosave_chain"

    def __init__(self, stages: list[StageConfig] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stages = stages if stages is not None else []
        names = [entry["stage"] for entry in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("Each stage must have a unique name")


class GeoSaveModel(PreTrainedModel):
    """Publish a model chain through the Transformers model contract.

    Args:
        config: Ordered construction specifications.
        chain: Existing chain to wrap without rebuilding or copying weights.
            Its specifications must match config. Omit when loading an artifact.

    Raises:
        ValueError: The supplied chain does not match the configuration.

    Examples:
        >>> published = GeoSaveModel.from_chain(task.model)
        >>> published.push_to_hub("your-account/your-model")
    """

    config_class = GeoSaveConfig
    base_model_prefix = "chain"

    def __init__(
        self, config: GeoSaveConfig, *, chain: ModelChain | None = None
    ) -> None:
        super().__init__(config)
        specs = {entry["stage"]: entry["spec"] for entry in config.stages}
        if chain is not None and list(chain.stage_specs.items()) != list(specs.items()):
            raise ValueError("Chain construction specifications do not match config")
        self.chain = chain if chain is not None else ModelChain(stages=specs)
        # Stage constructors own initialization; wrapping must preserve trained weights.
        with no_init_weights():
            self.post_init()

    @classmethod
    def from_chain(cls, chain: ModelChain) -> GeoSaveModel:
        """Wrap a trained chain, sharing its modules and weights.

        Args:
            chain: Model built from serializable stage specifications.

        Returns:
            Transformers model sharing the chain and its training mode.

        Raises:
            ValueError: The chain has no construction specifications.
        """
        config = GeoSaveConfig(
            stages=[
                StageConfig(stage=name, spec=spec)
                for name, spec in chain.stage_specs.items()
            ]
        )
        model = cls(config, chain=chain)
        model.training = chain.training
        return model

    def forward(self, **inputs: object) -> dict[str, object] | torch.Tensor:
        """Evaluate the chain using named tensor and geospatial context inputs.

        Args:
            **inputs: Values required by the chain's selected methods.

        Returns:
            The chain's predictions or feature context.
        """
        return self.chain(**inputs)


AutoConfig.register(GeoSaveConfig.model_type, GeoSaveConfig)
AutoModel.register(GeoSaveConfig, GeoSaveModel)
GeoSaveConfig.register_for_auto_class()
GeoSaveModel.register_for_auto_class("AutoModel")
