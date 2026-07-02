from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from typing import Any


@dataclass(frozen=True)
class ChannelSpec:
    """Declared reads and writes for one ModelContext channel.

    Attributes:
        reads: Keys this method reads from the channel before running.
        writes: Keys this method produces in the channel (inputs only; replaces prior keys).
    """

    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelContextSpec:
    """Full contract for one @model_context-decorated forward method.

    `inputs` uses replace semantics: after the method runs, only declared writes
    survive for subsequent modules. `sample_meta` and `metadata` always pass
    through unchanged; their `writes` field is unused.

    Attributes:
        inputs: Tensor keys consumed and produced.
        sample_meta: Per-sample metadata keys required (reads only).
        metadata: Shared batch-level keys required (reads only).
    """

    inputs: ChannelSpec = field(default_factory=ChannelSpec)
    sample_meta: ChannelSpec = field(default_factory=ChannelSpec)
    metadata: ChannelSpec = field(default_factory=ChannelSpec)


@dataclass
class ModelContext:
    """Universal context container flowing through the model pipeline.

    Args:
        inputs: Named tensors or intermediate features. Each module explicitly
            declares what it reads and produces; only declared writes survive.
        sample_meta: Per-sample metadata lists, always preserved.
            e.g. {"crs": ["EPSG:4326", ...], "datetime": [...]}.
        metadata: Shared batch-level config, always preserved.
            e.g. {"modalities": ["rgb", "sar"]}.
    """

    inputs: dict[str, Any]
    sample_meta: dict[str, list[Any] | None]
    metadata: dict[str, Any] = field(default_factory=dict)


def _parse_channel(spec: list | tuple | None) -> ChannelSpec:
    if spec is None:
        return ChannelSpec()
    if isinstance(spec, list):
        return ChannelSpec(reads=list(spec))
    if isinstance(spec, tuple) and len(spec) == 2:
        reads, writes = spec
        return ChannelSpec(reads=list(reads), writes=list(writes))
    raise ValueError(
        f"Invalid channel spec {spec!r}. "
        "Use list[str] for reads-only or (reads: list[str], writes: list[str]) tuple."
    )


def model_context(
    inputs: list | tuple | None = None,
    sample_meta: list | tuple | None = None,
    metadata: list | tuple | None = None,
):
    """Declare and validate ModelContext reads/writes for a forward method.

    Validates declared reads exist in the incoming ModelContext before the
    method body runs. Attaches a ModelContextSpec to the function for use
    by _discover_chain.

    Args:
        inputs: ctx.inputs keys. List = reads-only. Tuple = (reads, writes).
        sample_meta: ctx.sample_meta keys required. List = reads-only.
        metadata: ctx.metadata keys required. List = reads-only.

    Examples:
        >>> @model_context(inputs=(['image'], ['pyramid', 'prefix_tokens']))
        ... def forward_pyramid(self, ctx: ModelContext) -> ModelContext: ...
    """
    spec = ModelContextSpec(
        inputs=_parse_channel(inputs),
        sample_meta=_parse_channel(sample_meta),
        metadata=_parse_channel(metadata),
    )
    reads_inputs = spec.inputs.reads
    reads_sm = spec.sample_meta.reads
    reads_md = spec.metadata.reads

    def decorator(fn):
        @wraps(fn)
        def wrapper(self, ctx: ModelContext) -> ModelContext:
            for key in reads_inputs:
                if key not in ctx.inputs or ctx.inputs[key] is None:
                    raise KeyError(
                        f"{type(self).__name__}.{fn.__name__}: missing ctx.inputs['{key}']"
                    )
            for key in reads_sm:
                if key not in ctx.sample_meta or ctx.sample_meta[key] is None:
                    raise KeyError(
                        f"{type(self).__name__}.{fn.__name__}: missing ctx.sample_meta['{key}']"
                    )
            for key in reads_md:
                if key not in ctx.metadata or ctx.metadata[key] is None:
                    raise KeyError(
                        f"{type(self).__name__}.{fn.__name__}: missing ctx.metadata['{key}']"
                    )
            return fn(self, ctx)

        setattr(wrapper, '_model_context_spec', spec)
        return wrapper

    return decorator
