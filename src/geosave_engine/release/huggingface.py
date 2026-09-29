"""Installed Transformers persistence adapter for native model releases."""

from __future__ import annotations

from typing import Any, TypedDict

import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.initialization import no_init_weights

from geosave_engine.__about__ import __version__
from geosave_engine.ml.model_chain import ModelChain
from geosave_engine.ml.registry import build_model

ARTIFACT_FORMAT_VERSION = 1


class StageConfig(TypedDict):
    """One named stage in an ordered Transformers configuration."""

    stage: str
    spec: dict[str, Any]


class GeoSaveConfig(PretrainedConfig):
    """Describe a native chain in a generated model artifact.

    Args:
        stages: Ordered stage names and construction specifications.
        format_version: GeoSave release format version.
        geosave_version: Exact GeoSave version required to load the artifact.
        **kwargs: Standard Transformers configuration fields.

    Raises:
        ValueError: The format version is unsupported or stage names repeat.
        RuntimeError: The artifact requires a different GeoSave version.
    """

    model_type = "geosave_chain"

    def __init__(
        self,
        stages: list[StageConfig] | None = None,
        format_version: int = ARTIFACT_FORMAT_VERSION,
        geosave_version: str = __version__,
        **kwargs: Any,
    ) -> None:
        if format_version != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported GeoSave artifact format version {format_version}; "
                f"expected {ARTIFACT_FORMAT_VERSION}"
            )
        if geosave_version != __version__:
            raise RuntimeError(
                f"Artifact requires GeoSave version {geosave_version}; "
                f"installed version is {__version__}"
            )
        super().__init__(**kwargs)
        self.stages = stages if stages is not None else []
        self.format_version = format_version
        self.geosave_version = geosave_version
        names = [entry["stage"] for entry in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("Each stage must have a unique name")


class GeoSaveModel(PreTrainedModel):
    """Persist a native model chain through the Transformers file contract.

    Args:
        config: Ordered construction specifications.
        chain: Existing chain to wrap without rebuilding or copying weights.
            Its specifications must match config. Omit when loading an artifact.

    Raises:
        ValueError: The supplied chain does not match the configuration.

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
        self.chain = chain if chain is not None else build_model(specs)
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
        specs = chain.stage_specs
        for stage, spec in specs.items():
            if not spec.get("name") or "class_path" in spec:
                raise ValueError(
                    f"Stage {stage!r} must use a registered name for publication"
                )
        config = GeoSaveConfig(
            stages=[
                StageConfig(stage=name, spec=spec)
                for name, spec in specs.items()
            ]
        )
        model = cls(config, chain=chain)
        model.training = chain.training
        return model

    @staticmethod
    def _finalize_model_loading(
        model: PreTrainedModel,
        load_config: Any,
        loading_info: Any,
    ) -> Any:
        """Reject artifacts that do not exactly match their configured chain."""
        problems = []
        if loading_info.missing_keys:
            problems.append(f"Missing keys: {sorted(loading_info.missing_keys)}")
        if loading_info.unexpected_keys:
            problems.append(
                f"Unexpected keys: {sorted(loading_info.unexpected_keys)}"
            )
        if loading_info.mismatched_keys:
            problems.append(
                f"Mismatched keys: {sorted(loading_info.mismatched_keys)}"
            )
        if problems:
            raise RuntimeError(
                "GeoSaveModel requires an exact checkpoint match. "
                + "; ".join(problems)
            )
        loading_info = PreTrainedModel._finalize_model_loading(
            model, load_config, loading_info
        )
        problems = []
        if loading_info.error_msgs:
            problems.append(f"Loading errors: {loading_info.error_msgs}")
        if loading_info.conversion_errors:
            problems.append(f"Conversion errors: {loading_info.conversion_errors}")
        if problems:
            raise RuntimeError(
                "GeoSaveModel requires an exact checkpoint match. "
                + "; ".join(problems)
            )
        return loading_info

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
