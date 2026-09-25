"""Resolve one declared method per stage and order their dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter

from torch import nn

from .step import Step


@dataclass(frozen=True)
class BoundStep:
    """Associate a step declaration with its stage and module instance.

    Args:
        stage: Name under which the chain registers the module.
        module: Module containing the method.
        method: Method's step declaration.
    """

    stage: str
    module: nn.Module = field(compare=False)
    method: Step

    @property
    def name(self) -> str:
        """Return the class and method name used in diagnostics."""
        return f"{type(self.module).__name__}.{self.method.name}"


def _dependencies(steps: list[BoundStep]) -> dict[BoundStep, set[BoundStep]]:
    """Connect matching output and input names with identical declared types.

    Args:
        steps: Candidate or selected steps.

    Returns:
        Each step mapped to its producers.
    """
    producers: dict[tuple[str, type], set[BoundStep]] = {}
    for step in steps:
        for output in step.method.outputs.items():
            producers.setdefault(output, set()).add(step)
    dependencies: dict[BoundStep, set[BoundStep]] = {}
    for step in steps:
        dependencies[step] = set()
        for required in step.method.inputs.items():
            dependencies[step].update(producers.get(required, ()))
    return dependencies


def resolve_steps(modules: Mapping[str, nn.Module]) -> list[BoundStep]:
    """Choose each stage's earliest available method and sort the resulting chain.

    Args:
        modules: Registered stage names and instances, in construction order.

    Returns:
        One method per stage, in dependency order.

    Raises:
        TypeError: A stage has no declared step or equally early alternatives.
        CycleError: Step inputs and outputs form a cycle.
        ValueError: Selected steps publish the same output name.
    """
    candidates: list[BoundStep] = []
    for stage, module in modules.items():
        attributes: dict[str, object] = {}
        for cls in reversed(type(module).__mro__):
            attributes.update(vars(cls))
        methods = [getattr(value, "_chain_step", None) for value in attributes.values()]
        declared = [value for value in methods if isinstance(value, Step)]
        if not declared:
            raise TypeError(f"{type(module).__name__}: no @chain_step method found")
        candidates.extend(BoundStep(stage, module, method) for method in declared)

    graph = TopologicalSorter(_dependencies(candidates))
    try:
        graph.prepare()
    except CycleError as error:
        cycle: list[BoundStep] = error.args[1]
        message = cycle[0].name
        for before, after in zip(cycle, cycle[1:]):
            shared = before.method.outputs.keys() & after.method.inputs.keys()
            message += f" -({', '.join(sorted(shared))})-> {after.name}"
        raise CycleError(f"Steps are in a cycle: {message}", cycle) from None

    selected: dict[str, BoundStep] = {}
    while graph.is_active():
        ready = graph.get_ready()
        choices: dict[str, list[BoundStep]] = {}
        for step in ready:
            if step.stage not in selected:
                choices.setdefault(step.stage, []).append(step)
        for stage, methods in choices.items():
            if len(methods) != 1:
                raise TypeError(
                    f"Stage {stage!r} has ambiguous methods: {[step.name for step in methods]}"
                )
            selected[stage] = methods[0]
        graph.done(*ready)

    steps = [step for step in candidates if selected[step.stage] == step]
    outputs: dict[str, str] = {}
    for step in steps:
        for name in step.method.outputs:
            if name in outputs:
                raise ValueError(
                    f"Output {name!r} is produced by both {outputs[name]} and {step.name}"
                )
            outputs[name] = step.name
    return list(TopologicalSorter(_dependencies(steps)).static_order())


def external_inputs(steps: list[BoundStep], stages: list[str]) -> dict[str, type]:
    """Find required inputs that the selected chain does not produce.

    Args:
        steps: Selected methods.
        stages: Construction order, which defines positional input order.

    Returns:
        Required external names and types in stage declaration order.

    Raises:
        TypeError: One external name has incompatible declarations.
    """
    produced = {
        name: kind for step in steps for name, kind in step.method.outputs.items()
    }
    inputs: dict[str, type] = {}
    for step in sorted(steps, key=lambda step: stages.index(step.stage)):
        for name, declared in step.method.inputs.items():
            if name in produced:
                if produced[name] != declared:
                    raise TypeError(
                        f"Input {name!r} expects {declared}, but its producer declares {produced[name]}"
                    )
                continue
            if name in inputs and inputs[name] != declared:
                raise TypeError(f"External input {name!r} has conflicting types")
            inputs[name] = declared
    return inputs
