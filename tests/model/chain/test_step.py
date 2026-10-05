"""@chain_step derives a step's keys from the method's own signature and return."""

from __future__ import annotations

import pytest
import torch
from typing import Any, Literal, Protocol, TypedDict, TypeVar
import torch.nn as nn

from geosave_engine.model.chain import chain_step
from geosave_engine.model.chain.step import Step


def contract(method) -> Step:
    declaration = getattr(method, "_chain_step")
    assert isinstance(declaration, Step)
    return declaration


def invoke(module, method, context):
    return contract(getattr(module, method)).invoke(module, context)


class Encode(nn.Module):
    @chain_step()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        feature_map = image * 2
        return feature_map


class SplitTwo(nn.Module):
    @chain_step()
    def split(self, image: torch.Tensor) -> tuple[torch.Tensor, list]:
        pyramid, prefix_tokens = image, [image]
        return pyramid, prefix_tokens


class Optional(nn.Module):
    @chain_step()
    def run(
        self, image: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        masked = image if mask is None else image * mask
        return masked


@pytest.fixture
def x() -> torch.Tensor:
    return torch.ones(3)


def test_inputs_come_from_parameters(x):
    assert contract(Encode().encode).inputs == {"image": torch.Tensor}


def test_outputs_pair_return_names_with_types(x):
    assert contract(Encode().encode).outputs == {"feature_map": torch.Tensor}


def test_a_tuple_return_names_each_value_in_order(x):
    assert contract(SplitTwo().split).outputs == {
        "pyramid": torch.Tensor,
        "prefix_tokens": list,
    }

    result = invoke(SplitTwo(), "split", {"image": x})
    assert list(result) == ["pyramid", "prefix_tokens"]


def test_a_step_is_called_with_the_context_and_returns_its_outputs(x):
    result = invoke(Encode(), "encode", {"image": x})

    assert isinstance(result, dict)
    assert isinstance(result["feature_map"], torch.Tensor)
    assert result["feature_map"].tolist() == [2.0, 2.0, 2.0]


def test_a_parameter_with_a_default_is_not_required(x):
    assert contract(Optional().run).inputs == {"image": torch.Tensor}


def test_an_absent_optional_falls_back_to_the_method_default(x):
    result = invoke(Optional(), "run", {"image": x})

    assert isinstance(result, dict)
    assert isinstance(result["masked"], torch.Tensor)
    assert result["masked"].tolist() == [1.0, 1.0, 1.0]


def test_a_supplied_optional_is_forwarded(x):
    result = invoke(Optional(), "run", {"image": x, "mask": torch.zeros(3)})

    assert isinstance(result, dict)
    assert isinstance(result["masked"], torch.Tensor)
    assert result["masked"].tolist() == [0.0, 0.0, 0.0]


def test_a_head_returns_its_tensor_rather_than_a_dict(x):
    class Head(nn.Module):
        @chain_step(head=True)
        def logits(self, image: torch.Tensor) -> torch.Tensor:
            return image * 3

    result = invoke(Head(), "logits", {"image": x})

    assert isinstance(result, torch.Tensor)
    assert result.tolist() == [3.0, 3.0, 3.0]


def test_a_head_has_no_context_outputs():
    class Head(nn.Module):
        @chain_step(head=True)
        def logits(self, image: torch.Tensor) -> torch.Tensor:
            return image

    assert contract(Head().logits).outputs == {}


def test_a_missing_context_key_names_the_key(x):
    with pytest.raises(KeyError, match="missing input 'image'"):
        invoke(Encode(), "encode", {})


def test_a_context_key_of_the_wrong_type_is_refused():
    with pytest.raises(TypeError, match="expected Tensor, got list"):
        invoke(Encode(), "encode", {"image": [1, 2, 3]})


def test_a_none_context_value_counts_as_missing():
    with pytest.raises(KeyError, match="missing input 'image'"):
        invoke(Encode(), "encode", {"image": None})


def test_an_optional_of_the_wrong_type_is_refused(x):
    with pytest.raises(TypeError, match=r"expected torch.Tensor \| None, got list"):
        invoke(Optional(), "run", {"image": x, "mask": [1, 2, 3]})


def test_an_unannotated_parameter_is_refused_at_decoration():
    with pytest.raises(TypeError, match=r"missing type hint\(s\)"):

        class Bad(nn.Module):
            @chain_step()
            def run(self, image) -> torch.Tensor:
                out = image
                return out


class DetectionRecord(TypedDict):
    boxes: torch.Tensor


class ResultProtocol(Protocol):
    def score(self) -> torch.Tensor: ...


@pytest.mark.parametrize(
    "kind",
    [
        None,
        type(None),
        Any,
        Literal["box"],
        TypeVar("T"),
        DetectionRecord,
        ResultProtocol,
        type(None) | DetectionRecord,
        type(None) | ResultProtocol,
    ],
)
def test_terminal_requires_runtime_return_annotation(kind):
    def forward(self, image: torch.Tensor):
        return image

    forward.__annotations__["return"] = kind
    with pytest.raises(TypeError, match="terminal result"):
        chain_step(head=True)(forward)


def test_terminal_requires_return_annotation():
    def forward(self, image: torch.Tensor):
        return image

    with pytest.raises(TypeError, match="terminal result"):
        chain_step(head=True)(forward)


@pytest.mark.parametrize("empty", [False, True])
def test_terminal_native_list(x, empty):
    class Detector(nn.Module):
        @chain_step(head=True)
        def forward(self, image: torch.Tensor) -> list[dict[str, torch.Tensor]]:
            return [] if empty else [{"boxes": image}, {"boxes": image + 1}]

    result = invoke(Detector(), "forward", {"image": x})
    assert len(result) == (0 if empty else 2)
    if not empty:
        assert result[0]["boxes"] is x
        torch.testing.assert_close(result[1]["boxes"], x + 1)


def test_terminal_native_dict(x):
    class Classifier(nn.Module):
        @chain_step(head=True)
        def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"scores": image}

    result = invoke(Classifier(), "forward", {"image": x})
    assert result["scores"] is x


def test_terminal_tuple_is_not_unpacked(x):
    class Reconstruction(nn.Module):
        @chain_step(head=True)
        def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return image, image + 1

    result = invoke(Reconstruction(), "forward", {"image": x})
    assert isinstance(result, tuple)
    assert result[0] is x


def test_terminal_outer_type_mismatch(x):
    class Wrong(nn.Module):
        @chain_step(head=True)
        def forward(self, image: torch.Tensor) -> list[dict[str, torch.Tensor]]:
            return image  # type: ignore[return-value]

    with pytest.raises(TypeError, match="result expected list"):
        invoke(Wrong(), "forward", {"image": x})


@pytest.mark.parametrize("as_list", [False, True])
def test_terminal_union_validates_outer_types(x, as_list):
    class Choice(nn.Module):
        @chain_step(head=True)
        def forward(self, image: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
            return [image] if as_list else image

    result = invoke(Choice(), "forward", {"image": x})
    assert (result[0] if as_list else result) is x


def test_two_return_statements_are_refused():
    with pytest.raises(TypeError, match="exactly one return statement"):

        class Bad(nn.Module):
            @chain_step()
            def run(self, image: torch.Tensor) -> torch.Tensor:
                if image.sum() > 0:
                    early = image
                    return early
                late = image
                return late


def test_returning_an_expression_instead_of_a_name_is_refused():
    with pytest.raises(TypeError, match="must be `return name`"):

        class Bad(nn.Module):
            @chain_step()
            def run(self, image: torch.Tensor) -> torch.Tensor:
                return image * 2


def test_a_return_arity_that_disagrees_with_the_annotation_is_refused():
    with pytest.raises(TypeError, match="must match"):

        class Bad(nn.Module):
            @chain_step()
            def run(self, image: torch.Tensor) -> tuple[torch.Tensor, list]:
                only = image
                return only  # type: ignore[return-value]


def test_a_variadic_tuple_return_is_refused():
    with pytest.raises(TypeError, match="fixed-arity"):

        class Bad(nn.Module):
            @chain_step()
            def run(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
                items = image
                return items  # type: ignore[return-value]


def test_consuming_and_producing_one_key_reads_as_a_self_cycle():
    with pytest.raises(TypeError, match="self-cycle"):

        class Bad(nn.Module):
            @chain_step()
            def run(self, image: torch.Tensor) -> torch.Tensor:
                image = image * 2
                return image


def test_a_head_returning_a_non_tensor_is_caught_at_call_time(x):
    class Sneaky(nn.Module):
        @chain_step(head=True)
        def logits(self, image: torch.Tensor) -> torch.Tensor:
            return [image]  # type: ignore[return-value]

    with pytest.raises(TypeError, match="result expected Tensor"):
        invoke(Sneaky(), "logits", {"image": x})


def test_a_declared_output_type_is_checked_against_the_real_value(x):
    class Sneaky(nn.Module):
        @chain_step()
        def run(self, image: torch.Tensor) -> torch.Tensor:
            wrong = [image]
            return wrong  # type: ignore[return-value]

    with pytest.raises(TypeError, match="output 'wrong' expected Tensor"):
        invoke(Sneaky(), "run", {"image": x})


def test_a_class_with_no_readable_source_is_refused_with_a_clear_message():
    generated = (
        "import torch\n"
        "import torch.nn as nn\n"
        "from geosave_engine.model.chain import chain_step\n"
        "class Generated(nn.Module):\n"
        "    @chain_step()\n"
        "    def run(self, image: torch.Tensor) -> torch.Tensor:\n"
        "        out = image\n"
        "        return out\n"
    )

    with pytest.raises(TypeError, match="no source is available"):
        exec(compile(generated, "<generated>", "exec"), {})


def test_a_subclass_inherits_its_base_step():
    class Child(Encode):
        pass

    assert contract(Child().encode).inputs == {"image": torch.Tensor}


class ExplicitOutputs(nn.Module):
    @chain_step(outputs=("features", "tokens"))
    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, list]:
        if image.sum() > 0:
            return image * 2, [image]
        return image - 1, []


def test_explicit_outputs_name_expressions_and_branches(x):
    method = ExplicitOutputs().encode
    assert contract(method).outputs == {"features": torch.Tensor, "tokens": list}
    positive = contract(method).invoke(ExplicitOutputs(), {"image": x})
    negative = contract(method).invoke(ExplicitOutputs(), {"image": -x})
    assert isinstance(positive, dict)
    assert isinstance(negative, dict)
    torch.testing.assert_close(positive["features"], x * 2)
    torch.testing.assert_close(negative["features"], -x - 1)
    assert isinstance(positive["tokens"], list)
    assert len(positive["tokens"]) == 1
    assert negative["tokens"] == []


def test_explicit_outputs_do_not_require_source(x):
    namespace = {}
    exec(
        "import torch\n"
        "from geosave_engine.model.chain import chain_step\n"
        "@chain_step(outputs=('encoded',))\n"
        "def encode(self, image: torch.Tensor) -> torch.Tensor:\n"
        "    return image * 3\n",
        namespace,
    )
    result = namespace["encode"](None, x)
    torch.testing.assert_close(result, x * 3)


@pytest.mark.parametrize("outputs", [(), ("a", "a"), ("",), "features", (1,)])
def test_invalid_explicit_output_names_are_rejected(outputs):
    with pytest.raises(TypeError, match="non-empty tuple of unique names"):

        @chain_step(outputs=outputs)
        def encode(self, image: torch.Tensor) -> torch.Tensor:
            return image


def test_explicit_output_count_must_match_annotation():
    with pytest.raises(TypeError, match="must match"):

        @chain_step(outputs=("features",))
        def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, list]:
            return image, []


def test_explicit_outputs_still_validate_runtime_values(x):
    class Wrong(nn.Module):
        @chain_step(outputs=("features",))
        def encode(self, image: torch.Tensor) -> torch.Tensor:
            return [image]  # type: ignore[return-value]

    with pytest.raises(TypeError, match="output 'features' expected Tensor"):
        invoke(Wrong(), "encode", {"image": x})


def test_explicit_outputs_still_reject_self_cycles():
    with pytest.raises(TypeError, match="self-cycle"):

        @chain_step(outputs=("image",))
        def encode(self, image: torch.Tensor) -> torch.Tensor:
            return image * 2


def test_terminal_head_cannot_declare_context_outputs():
    with pytest.raises(TypeError, match="cannot declare outputs"):
        chain_step(head=True, outputs=("logits",))


def test_decorated_methods_keep_their_ordinary_signature(x):
    torch.testing.assert_close(Encode().encode(x), x * 2)
    torch.testing.assert_close(
        Optional().run(x, mask=torch.zeros_like(x)), torch.zeros_like(x)
    )
