"""Compose model stages and save their construction recipe with their weights."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Self

from huggingface_hub import PyTorchModelHubMixin
import torch
from torch import nn

from geosave_engine.ml.registry.model import StageSpec
from .graph import external_inputs, resolve_steps


class ModelChain(nn.Module, PyTorchModelHubMixin, library_name="geosave-engine"):
    """Run one declared method per stage in dependency order.

    Args:
        *args: Modules named stage_0, stage_1, and so on.
        stages: Ordered construction specifications for a saveable model.
            Cannot be combined with module instances.
        **modules: Module instances under explicit stage names.

    Raises:
        ValueError: Construction forms are mixed or stage/output names collide.
        TypeError: A stage has no step or ambiguous alternatives.
        graphlib.CycleError: Required inputs form a dependency cycle.

    Examples:
        >>> model = ModelChain(stages={
        ...     "encoder": {"name": "dinov3", "init_args": {"pretrained": False}},
        ... })
        >>> model.save_pretrained("artifacts/model")
        >>> restored = ModelChain.from_pretrained("artifacts/model")
    """

    def __init__(
        self,
        *args: nn.Module,
        stages: dict[str, StageSpec] | None = None,
        **modules: nn.Module,
    ) -> None:
        super().__init__()
        self._hub_mixin_config = None
        if stages is not None:
            from geosave_engine.ml.registry.model import build_stages

            if args or modules:
                raise ValueError("Supply stages or module instances, not both")
            built = build_stages(stages)
            modules = built.modules
            self._hub_mixin_config = {"stages": built.config}
        positional = {f"stage_{index}": module for index, module in enumerate(args)}
        if positional.keys() & modules.keys():
            raise ValueError("Positional stage names collide with named modules")
        modules = positional | modules
        for name, module in modules.items():
            self.add_module(name, module)
        self._steps = resolve_steps(modules)
        self._inputs = external_inputs(self._steps, list(modules))

    @property
    def stage_specs(self) -> dict[str, StageSpec]:
        """Return an independent copy of the ordered construction specifications.

        Returns:
            Stage selectors and resolved constructor arguments.

        Raises:
            ValueError: The chain was constructed directly from module instances.
        """
        if not isinstance(self._hub_mixin_config, dict):
            raise ValueError("Build with stage specifications before saving")
        return deepcopy(self._hub_mixin_config["stages"])

    @property
    def inputs(self) -> dict[str, type]:
        """Return required external names and types, in positional argument order."""
        return dict(self._inputs)

    def forward(
        self, *args: object, **kwargs: object
    ) -> dict[str, object] | torch.Tensor:
        """Route external inputs through the selected stage methods.

        Args:
            *args: Required external values in inputs order.
            **kwargs: External values by name, including optional model context.

        Returns:
            A single head's Tensor, multiple heads keyed by stage, or the merged
            context when the chain has no head.

        Raises:
            TypeError: Positional arguments exceed inputs or repeat keyword values;
                a step receives or returns a value of the wrong type.
            KeyError: A required input is missing.
        """
        if len(args) > len(self._inputs):
            raise TypeError(
                f"ModelChain takes at most {len(self._inputs)} positional arguments"
            )
        positional = dict(zip(self._inputs, args))
        repeated = positional.keys() & kwargs.keys()
        if repeated:
            raise TypeError(
                f"Values supplied both a positional and keyword way: {sorted(repeated)}"
            )
        context = positional | kwargs
        heads: dict[str, object] = {}
        for step in self._steps:
            result = step.method.invoke(step.module, context)
            if isinstance(result, torch.Tensor):
                heads[step.stage] = result
            else:
                context.update(result)
        if len(heads) == 1:
            result = next(iter(heads.values()))
            assert isinstance(result, torch.Tensor)
            return result
        return heads if heads else context

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and the ordered construction recipe.

        Args:
            save_directory: Directory created by Hugging Face.

        Raises:
            ValueError: The model was built without construction specifications.
            TypeError: Constructor arguments are not JSON serializable.
        """
        if self._hub_mixin_config is None:
            raise ValueError("Build with stage specifications before saving")
        config = json.dumps(self._hub_mixin_config, indent=2, allow_nan=False)
        super()._save_pretrained(save_directory)
        # Published constructor arguments depend on stage order; the mixin sorts keys.
        (save_directory / "config.json").write_text(config, encoding="utf-8")

    @classmethod
    def _from_pretrained(cls, *, strict: bool = True, **kwargs: Any) -> Self:
        """Rebuild a model through Hugging Face and strictly load its weights.

        Args:
            strict: Reject missing or unexpected state keys when True.
            **kwargs: Loading arguments supplied by the Hugging Face mixin.

        Returns:
            Restored model in evaluation mode.
        """
        return super()._from_pretrained(strict=strict, **kwargs)

    def __repr__(self) -> str:
        """Return external inputs, resolved method order, and registered modules."""

        def describe(values: dict[str, type]) -> str:
            return ", ".join(
                f"{name}: {getattr(kind, '__name__', kind)}"
                for name, kind in values.items()
            )

        lines = [f"inputs: {describe(self._inputs)}", "", "data flow:"]
        for step in self._steps:
            method = step.method
            output = "Tensor" if method.head else f"{{{describe(method.outputs)}}}"
            lines.append(
                f"  {step.stage}: {step.name}({describe(method.inputs)}) -> {output}"
            )
        return "\n".join(lines) + f"\n\n{super().__repr__()}"
