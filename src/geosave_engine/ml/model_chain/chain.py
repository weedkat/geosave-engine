"""Compose model modules through their declared named data flow."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from .routing import ordered_steps, required_inputs


class ModelChain(nn.Module):
    """Run one declared method per module in dependency order.

    Args:
        *args: Modules named stage_0, stage_1, and so on.
        **modules: Module instances under explicit stage names.

    Raises:
        ValueError: Positional and explicit stage names collide.
        TypeError: A stage has no step or ambiguous alternatives.
        graphlib.CycleError: Required inputs form a dependency cycle.
    """

    def __init__(self, *args: nn.Module, **modules: nn.Module) -> None:
        super().__init__()
        positional = {f"stage_{index}": module for index, module in enumerate(args)}
        if positional.keys() & modules.keys():
            raise ValueError("Positional stage names collide with named modules")
        modules = positional | modules
        for name, module in modules.items():
            self.add_module(name, module)
        self._steps = ordered_steps(modules)
        self._inputs = required_inputs(self._steps, list(modules))
        self._stage_specs: dict[str, dict[str, object]] | None = None

    @property
    def stage_specs(self) -> dict[str, dict[str, object]]:
        """Return an independent configured-construction recipe.

        Returns:
            Stage selectors and resolved constructor arguments.

        Raises:
            ValueError: The chain was composed directly from module instances.
        """
        if self._stage_specs is None:
            raise ValueError("Build with stage specifications before export")
        return deepcopy(self._stage_specs)

    @property
    def inputs(self) -> dict[str, type]:
        """Return required external names and types in positional order."""
        return dict(self._inputs)

    def forward(
        self, *args: object, **kwargs: object
    ) -> dict[str, object] | torch.Tensor:
        """Route external inputs through the selected module methods."""
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

    def __repr__(self) -> str:
        """Return external inputs, execution order, and registered modules."""

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
