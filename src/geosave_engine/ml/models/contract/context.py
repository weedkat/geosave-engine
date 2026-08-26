from __future__ import annotations

import ast
import inspect
import textwrap
import types
import typing
from functools import wraps
from typing import Any

import torch

_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _strip_optional(hint: type) -> type:
    """Unwrap `X | None` down to `X`; any other hint passes through unchanged.

    Args:
        hint: A resolved type hint (from `typing.get_type_hints`).

    Returns:
        `X` for an `X | None` union of exactly one non-`None` member, else `hint` as-is.
    """
    if typing.get_origin(hint) is types.UnionType:
        members = [arg for arg in typing.get_args(hint) if arg is not type(None)]
        if len(members) == 1:
            return members[0]
    return hint


def _direct_returns(node: ast.AST) -> list[ast.Return]:
    """`Return` nodes belonging to `node`'s own body -- not any nested def/lambda/class.

    Args:
        node: an `ast` node (typically a `FunctionDef`) to search inside.

    Returns:
        Every `ast.Return` reachable without crossing into a nested scope.
    """
    returns: list[ast.Return] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Return):
            returns.append(child)
        elif isinstance(child, _SCOPE_NODES):
            continue
        else:
            returns.extend(_direct_returns(child))
    return returns


def _return_names(fn) -> tuple[str, ...]:
    """Variable name(s) in `fn`'s own `return name` or `return name1, name2, ...` statement.

    Requires exactly one `return` in `fn`'s own body (not a nested closure's),
    whose value is either a single bare local variable or a tuple of them --
    not an arbitrary expression, not zero/multiple return points. Any other
    shape is a decoration-time `TypeError`, not a silently-skipped case.

    Args:
        fn: the undecorated method.

    Returns:
        Variable name(s), in return order, e.g. `('feature_map',)` for a
        single return, or `('pyramid', 'prefix_tokens')` for a tuple one.

    Raises:
        TypeError: not exactly one `return`, or its value isn't a bare name
            or tuple of bare names.
    """
    source = textwrap.dedent(inspect.getsource(fn))
    func_def = ast.parse(source).body[0]
    returns = _direct_returns(func_def)
    if len(returns) != 1:
        raise TypeError(
            f"{fn.__qualname__}: must have exactly one return statement, found {len(returns)}"
        )

    value = returns[0].value
    if isinstance(value, ast.Name):
        return (value.id,)
    if isinstance(value, ast.Tuple):
        names: list[str] = []
        for elt in value.elts:
            if not isinstance(elt, ast.Name):
                raise TypeError(
                    f"{fn.__qualname__}: return statement must be `return name` or "
                    "`return name1, name2, ...` of bare local variables"
                )
            names.append(elt.id)
        return tuple(names)
    raise TypeError(
        f"{fn.__qualname__}: return statement must be `return name` or "
        "`return name1, name2, ...` of bare local variables"
    )


def _provides_from_return_type(fn) -> tuple[dict[str, type], bool]:
    """Derive `provides` from `fn`'s `-> T` or `-> tuple[T1, T2, ...]` annotation + its return statement.

    Strict on purpose, one code path for every rejection -- no fallback for a
    bare unparametrized `tuple`, no variadic `tuple[T, ...]`, no arity
    mismatch between the return statement and the annotation. All of these
    are the same defect (the annotation doesn't give one concrete type per
    returned value) and raise the same way.

    Args:
        fn: the undecorated method; must have a `-> T` or `-> tuple[T1, T2, ...]` hint.

    Returns:
        (`{name: type}`, is_single) -- the dict is insertion-ordered to match
        the return statement's own value order (the wrapper zips a real
        call's returned tuple against its keys, no separate ordered-names
        value needed); `is_single` tells the wrapper whether to expect a bare
        value at call time instead of a tuple.

    Raises:
        TypeError: return annotation isn't a concrete type or a fixed-arity
            `tuple[T1, T2, ...]` of concrete types, or its arity doesn't
            match the return statement's own value count.
    """
    hints = typing.get_type_hints(fn)
    return_hint = hints.get('return')
    is_single = typing.get_origin(return_hint) is not tuple
    if is_single:
        if return_hint is None or return_hint is type(None):
            raise TypeError(
                f"{fn.__qualname__}: must return `-> T` or `-> tuple[T1, T2, ...]` of "
                f"concrete, fixed-arity types, got {return_hint!r}"
            )
        arg_types: tuple[type, ...] = (return_hint,)
    else:
        arg_types = typing.get_args(return_hint)
        if not arg_types or Ellipsis in arg_types:
            raise TypeError(
                f"{fn.__qualname__}: must return `-> T` or `-> tuple[T1, T2, ...]` of "
                f"concrete, fixed-arity types, got {return_hint!r}"
            )

    names = _return_names(fn)
    if len(names) != len(arg_types):
        raise TypeError(
            f"{fn.__qualname__}: return statement has {len(names)} value(s) but "
            f"return type {return_hint!r} has {len(arg_types)} -- must match"
        )
    return dict(zip(names, arg_types)), is_single


def chain_step(head: bool = False):
    """Mark a module method as a context-chain step; validate keys and types.

    The decorated method takes typed tensor/list params (its real inputs),
    not a raw ``ctx`` dict — ``requires`` is derived from the method's own
    signature (param name -> resolved type hint), so there's one source of
    truth for what a step needs instead of a hand-typed dict that can drift
    from the body. ``provides`` is derived the same way, from the *output*
    side: the method's own ``-> T`` or ``-> tuple[T1, T2, ...]`` return
    annotation gives the type(s), its own ``return name`` or
    ``return name1, name2, ...`` statement gives the name(s) — one source of
    truth there too, no separately hand-typed dict to drift from either the
    signature or the body. ``ContextChain`` still passes a shared
    ``dict[str, Any]`` between steps; the wrapper this decorator builds
    unpacks it into the typed call, re-packs the method's plain (or
    single-value) return into that dict for ``ContextChain`` to merge in.

    A param with a real default (``x: T | None = None``) is optional —
    dropped from ``requires``, so ``ContextChain`` never demands it from the
    caller or the graph. The wrapper still forwards it from ctx when
    present (type-checked against ``T``, the non-``None`` half of the
    hint); when absent or ``None`` in ctx, the call omits it and the
    method's own default runs instead, so the method body decides what to
    do without it.

    ``head=True`` methods are terminal — the chain stops and returns the
    value directly instead of merging it into ctx. Must return
    ``-> torch.Tensor`` (checked both at decoration time against the
    annotation, and at call time against the actual returned value).

    Whether a module is the chain's entry point is a separate concern, not
    this decorator's — see ``ContextChain`` in ``chain.py``. Entry is decided
    by whoever wires up a specific pipeline, not declared by the class author
    here.

    Args:
        head: True for a terminal method (a task head) — returns
            ``torch.Tensor`` directly, ends the chain, adds nothing to ctx.

    Raises:
        TypeError: A required param has no type hint; ``head=False`` and the
            return annotation isn't a concrete type or a fixed-arity
            ``tuple[T1, T2, ...]`` of concrete types, or its arity doesn't
            match the return statement's own value count, or the return
            statement isn't exactly one ``return name`` or
            ``return name1, name2, ...`` of bare local variables, or a
            ``requires``/``provides`` pair shares the same ``(name, type)``
            key (reads as a self-cycle in ``ContextChain``'s key graph —
            return a differently-named local instead); ``head=True`` and the
            return annotation isn't ``torch.Tensor``; (at call time) a
            declared key is missing/mismatched in ctx, or the actual
            returned value doesn't match what was declared.

    Examples:
        >>> @chain_step()
        ... def encode(self, image: torch.Tensor) -> torch.Tensor:
        ...     feature_map = self.encoder(image)
        ...     return feature_map

        >>> @chain_step()
        ... def forward_pyramid(self, image: torch.Tensor) -> tuple[list, list]:
        ...     features, prefix_tokens = self.backbone(image)
        ...     return features, prefix_tokens

        >>> @chain_step(head=True)
        ... def forward_logits(self, feature_map: torch.Tensor) -> torch.Tensor:
        ...     return self.head(feature_map)
    """

    def decorator(fn):
        hints = typing.get_type_hints(fn)
        sig = inspect.signature(fn)
        param_names = [name for name in sig.parameters if name != 'self']
        missing_hints = [name for name in param_names if name not in hints]
        if missing_hints:
            raise TypeError(f"{fn.__qualname__}: missing type hint(s) for {missing_hints}")
        # A param with a real default is optional — excluded from requires below.
        _optional = {name for name in param_names if sig.parameters[name].default is not inspect.Parameter.empty}
        _requires: dict[str, type] = {name: hints[name] for name in param_names if name not in _optional}

        if head:
            if hints.get('return') is not torch.Tensor:
                raise TypeError(
                    f"{fn.__qualname__}: head=True must return `-> torch.Tensor`, "
                    f"got {hints.get('return')!r}"
                )
            _provides: dict[str, type] = {}
            _is_single = False
        else:
            _provides, _is_single = _provides_from_return_type(fn)
            self_referencing = {
                name for name, expected in _requires.items() if _provides.get(name) is expected
            }
            if self_referencing:
                raise TypeError(
                    f"{fn.__qualname__}: param(s) {sorted(self_referencing)} are both required "
                    "and provided under the same name and type — ContextChain's key graph "
                    "can't tell 'value in' from 'value out' for the same key, so this reads as "
                    "a self-cycle, not a real transform step. Return a differently-named local "
                    "instead, e.g. `decoded = self.decoder(feature_map); return decoded` — the "
                    "parameter can keep its name, only the return statement's variable needs "
                    "to differ."
                )

        @wraps(fn)
        def wrapper(self, ctx: dict[str, Any]) -> dict[str, Any] | torch.Tensor:
            for key, expected in _requires.items():
                if key not in ctx or ctx[key] is None:
                    raise KeyError(
                        f"{type(self).__name__}.{fn.__name__}: missing ctx['{key}']"
                    )
                if not isinstance(ctx[key], expected):
                    raise TypeError(
                        f"{type(self).__name__}.{fn.__name__}: ctx['{key}'] expected "
                        f"{expected.__name__}, got {type(ctx[key]).__name__}"
                    )

            call_kwargs: dict[str, Any] = {name: ctx[name] for name in _requires}
            for name in _optional:
                if name in ctx and ctx[name] is not None:
                    expected = _strip_optional(hints[name])
                    if not isinstance(ctx[name], expected):
                        raise TypeError(
                            f"{type(self).__name__}.{fn.__name__}: ctx['{name}'] expected "
                            f"{expected.__name__}, got {type(ctx[name]).__name__}"
                        )
                    call_kwargs[name] = ctx[name]

            result = fn(self, **call_kwargs)

            if head:
                if not isinstance(result, torch.Tensor):
                    raise TypeError(
                        f"{type(self).__name__}.{fn.__name__}: head=True must return "
                        f"torch.Tensor, got {type(result).__name__}"
                    )
                return result

            if _is_single:
                result = (result,)
            elif not isinstance(result, tuple) or len(result) != len(_provides):
                raise TypeError(
                    f"{type(self).__name__}.{fn.__name__}: expected a {len(_provides)}"
                    f"-tuple ({', '.join(_provides)}), got {result!r}"
                )
            result_dict = dict(zip(_provides, result))
            for key, expected in _provides.items():
                if not isinstance(result_dict[key], expected):
                    raise TypeError(
                        f"{type(self).__name__}.{fn.__name__}: provides['{key}'] expected "
                        f"{expected.__name__}, got {type(result_dict[key]).__name__}"
                    )
            return result_dict

        setattr(wrapper, '_is_chain_step', True)
        setattr(wrapper, '_requires', _requires)
        setattr(wrapper, '_provides', _provides)
        return wrapper

    return decorator
