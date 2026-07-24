from __future__ import annotations

from typing import Any

import networkx as nx
import torch
import torch.nn as nn

def _model_context_methods(module: nn.Module) -> list[str]:
    """Every @model_context method name declared on one module (MRO-collapsed).

    A module may declare more than one — e.g. several alternate accepted
    shapes from upstream. Which one actually gets used is resolved by
    `_solve_dag`, not here.

    Args:
        module: One chain-stage instance.

    Returns:
        Method names, in MRO order (base class first).
    """
    methods: list[str] = []
    for klass in reversed(type(module).__mro__):
        for name, val in vars(klass).items():
            if callable(val) and getattr(val, '_is_model_context', False) and name not in methods:
                methods.append(name)
    return methods


def _build_graph(modules: list[nn.Module]) -> nx.DiGraph:
    """Build the bipartite key/method graph for `modules` — pure wiring, no validation.

    Two node kinds: key nodes (``(name, type)`` tuples, e.g.
    ``('image', torch.Tensor)``) and method nodes (``(module, method_name)``
    tuples). Every module's every ``@model_context`` method becomes one
    method node, wired ``key -> method`` for each of its ``requires`` and
    ``method -> key`` for each of its ``provides``. Nothing here checks
    whether a `requires` key is ever actually produced by anything — that's
    resolution's job, done separately by walking this graph outward from
    whatever keys the caller supplies. A terminal method (``provides``
    empty, returns a raw ``torch.Tensor``) ends up with no outgoing edges —
    a plain graph sink, not a separately-flagged concept.

    Key identity is ``(name, type)``, not just ``name`` — two methods that
    use the same key name but declare different types end up as two
    distinct, unconnected key nodes rather than a caught collision. If
    that split leaves a method's ``('image', list)`` requirement with no
    producer, it just surfaces as an extra entry in `ContextChain.required_keys`
    — same as any other unproduced key — not an error.

    Args:
        modules: nn.Module instances to include. Order doesn't matter for
            building the graph — only resolution (walking it afterward)
            cares about order.

    Returns:
        Bipartite ``nx.DiGraph``. Method nodes carry ``kind='method'``,
        ``requires``, ``provides`` (the method's own dicts, for resolution
        to read without re-deriving them); key nodes carry ``kind='key'``,
        ``name``, ``type``.

    Raises:
        TypeError: A module has no `@model_context` method at all.
    """
    graph = nx.DiGraph()

    for module in modules:
        names = _model_context_methods(module)
        if not names:
            raise TypeError(f"{type(module).__name__}: no @model_context method found.")

        for name in names:
            method = getattr(module, name)
            requires: dict[str, type] = getattr(method, '_requires', {})
            provides: dict[str, type] = getattr(method, '_provides', {})
            node = (module, name)
            graph.add_node(node, kind='method', requires=requires, provides=provides)

            for key, expected in requires.items():
                graph.add_node((key, expected), kind='key', name=key, type=expected)
                graph.add_edge((key, expected), node)
            for key, expected in provides.items():
                graph.add_node((key, expected), kind='key', name=key, type=expected)
                graph.add_edge(node, (key, expected))
    return graph


def _solve_dag(graph: nx.DiGraph) -> nx.DiGraph:
    """Prune `graph` down to exactly one @model_context method per module.

    `graph` may hold several candidate methods per module; walk it by
    `nx.topological_generations` (networkx's Kahn's algorithm) and keep
    only whichever candidate surfaces first for its module — cut the rest.
    Two candidates surfacing in the same generation can't be cut down to
    one — that's a genuine ambiguity, not decided here.

    Args:
        graph: Full bipartite graph from `_build_graph` — may contain
            multiple candidate method nodes per module, not yet resolved.

    Returns:
        The induced subgraph on exactly the chosen method nodes plus the
        key nodes actually connecting them — one method node per module.
        `nx.topological_sort` on this gives the execution order.

    Raises:
        nx.NetworkXUnfeasible: `graph` has a cycle — no valid order exists.
        TypeError: A module has more than one candidate method surfacing
            in the same generation.
    """
    chosen: dict[nn.Module, tuple[nn.Module, str]] = {}

    for generation in nx.topological_generations(graph):
        methods_by_module: dict[nn.Module, list[tuple[nn.Module, str]]] = {}
        for node in generation:
            if graph.nodes[node]['kind'] != 'method':
                continue
            module, _ = node
            if module in chosen:
                continue
            methods_by_module.setdefault(module, []).append(node)

        for module, methods in methods_by_module.items():
            if len(methods) > 1:
                raise TypeError(
                    f"{type(module).__name__}: {len(methods)} methods surfaced in the "
                    f"same generation — ambiguous: {[m[1] for m in methods]}"
                )
            chosen[module] = methods[0]

    chosen_nodes = set(chosen.values())
    key_nodes: set[tuple[str, type]] = set()
    for node in chosen_nodes:
        for mapping in (graph.nodes[node]['requires'], graph.nodes[node]['provides']):
            for name, expected in mapping.items():
                key_nodes.add((name, expected))

    return graph.subgraph(chosen_nodes | key_nodes).copy()


def _graph_to_chain(graph: nx.DiGraph) -> list[tuple[nn.Module, str]]:
    """Linearize a resolved graph (from `_solve_dag`) into execution order.

    `nx.topological_sort` gives one valid order over every node — key nodes
    and method nodes both, since the graph is bipartite. Only method nodes
    are what `ContextChain.forward` actually calls; key nodes are filtered
    out here, their position is already implied by the method nodes around
    them.

    Args:
        graph: A resolved graph (exactly one method node per module) from
            `_solve_dag` — not the full graph from `_build_graph`.

    Returns:
        (module, method_name) pairs, in the order `ContextChain.forward`
        should call them.
    """
    return [node for node in nx.topological_sort(graph) if graph.nodes[node]['kind'] == 'method']


class ContextChain(nn.Module):
    """nn.Module that wires submodules together via their @model_context methods.

    Takes named (or auto-named) modules, registers each as a named submodule,
    then resolves call order from the modules themselves — not from argument
    order. `_build_graph` builds a bipartite key/method dependency graph from
    every module's `@model_context` method(s) (``requires -> method``,
    ``method -> provides``); `_solve_dag` walks it by
    `nx.topological_generations` (Kahn's algorithm) to pick exactly one method
    per module — raising if more than one of a module's candidate methods
    becomes ready in the same generation (genuine ambiguity, not decidable
    from the graph alone); `_graph_to_chain` linearizes the resolved graph
    into the order `forward` calls modules in.

    A module offering several `@model_context` methods (alternate accepted
    input shapes) is fine — whichever one's `requires` the graph can satisfy
    gets picked automatically. Branching (independent modules each consuming
    caller-supplied input directly) and merging (a later module requiring
    outputs from more than one earlier module) both fall out of the same
    graph walk, no special-casing needed for either.

    Each module receives the shared ``dict[str, Any]`` context and returns a
    dict of its outputs. These are merged immutably into the context
    (``ctx = {**ctx, **result}``) before the next module runs — prior keys are
    preserved without mutation, so branching and intermediate inspection are safe.

    A terminal module (typically a head) may return a ``torch.Tensor`` directly
    to end the chain early.

    Args:
        *args: Positional modules, auto-named ``stage_0``, ``stage_1``, ...
            Mix freely with ``**modules`` as long as names don't collide.
            Argument order doesn't affect resolution — the graph decides call
            order, not how modules were passed in.
        **modules: Name → module, e.g. ``encoder=enc, decoder=dec, head=hd``.

    Raises:
        ValueError: A positional arg's auto-generated name (``stage_N``)
            collides with an explicit keyword name.
        TypeError: A module has no `@model_context` method at all, or more
            than one of a module's candidate methods becomes ready in the
            same DAG generation (see `_solve_dag`).
        nx.NetworkXUnfeasible: The requires/provides graph has a cycle — no
            valid execution order exists.

    Examples:
        >>> chain = ContextChain(encoder=enc, decoder=dec, head=hd)
        >>> chain.required_keys  # {'image': torch.Tensor} -- what forward() needs
        >>> logits = chain({'image': x})  # enc → dec → hd; head returns Tensor
    """

    def __init__(self, *args: nn.Module, **modules: nn.Module) -> None:
        super().__init__()
        positional = {f"stage_{i}": module for i, module in enumerate(args)}
        collision = set(positional) & set(modules)
        if collision:
            raise ValueError(
                f"positional arg auto-name(s) {collision} collide with explicit "
                "keyword name(s) — rename the keyword or don't mix"
            )
        named = {**positional, **modules}

        for name, module in named.items():
            self.add_module(name, module)

        graph = _build_graph(list(named.values()))
        self._dag = _solve_dag(graph)
        self._chain = _graph_to_chain(self._dag)

    def __repr__(self) -> str:
        def sig(types: dict[str, type]) -> str:
            return ", ".join(f"{key}: {getattr(t, '__name__', t)}" for key, t in types.items())

        name_by_module = {module: name for name, module in self.named_children()}
        lines = [f"{type(self).__name__}("]
        for module, method_name in self._chain:
            method = getattr(module, method_name)
            requires = sig(getattr(method, '_requires', {}))
            provides = getattr(method, '_provides', {})
            out = f"{{{sig(provides)}}}" if provides else "Tensor"
            lines.append(f"  {name_by_module[module]}: {type(module).__name__}.{method_name}({requires}) -> {out}")
        lines.append(")")
        return "\n".join(lines)

    @property
    def required_keys(self) -> dict[str, type]:
        """Keys ``forward()``'s ctx must already contain before calling it.

        Generation 0 of the resolved DAG — key nodes with no producer
        inside this chain, read straight off the graph (see `_solve_dag`),
        not re-derived after the fact by diffing the flattened chain.
        Includes every such key, not just whichever entry method
        topological order happened to place first — branching means more
        than one method can independently need something from the caller.
        A method with no `requires` at all lands in generation 0 too;
        filtered out here since it's not a key.
        """
        first_generation = next(nx.topological_generations(self._dag))
        return {
            self._dag.nodes[node]['name']: self._dag.nodes[node]['type']
            for node in first_generation
            if self._dag.nodes[node]['kind'] == 'key'
        }

    def forward(self, ctx: dict[str, Any]) -> dict[str, Any] | torch.Tensor:
        for module, method_name in self._chain:
            result = getattr(module, method_name)(ctx)
            if isinstance(result, torch.Tensor):
                return result
            ctx = {**ctx, **result}
        return ctx
