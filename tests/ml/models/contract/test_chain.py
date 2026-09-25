"""ModelChain resolves call order from the modules themselves."""

from __future__ import annotations

from graphlib import CycleError

import pytest
import torch
import torch.nn as nn

from geosave_engine.ml.models.contract import ModelChain, chain_step


class Encode(nn.Module):
    @chain_step()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        feature_map = image * 2
        return feature_map


class Decode(nn.Module):
    @chain_step()
    def decode(self, feature_map: torch.Tensor) -> torch.Tensor:
        decoded = feature_map + 1
        return decoded


class Head(nn.Module):
    @chain_step(head=True)
    def logits(self, decoded: torch.Tensor) -> torch.Tensor:
        return decoded.sum(0, keepdim=True)


class OtherHead(nn.Module):
    @chain_step(head=True)
    def other(self, decoded: torch.Tensor) -> torch.Tensor:
        return decoded.mean(0, keepdim=True)


class Split(nn.Module):
    @chain_step()
    def split(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low, high = image - 1, image + 1
        return low, high


class Merge(nn.Module):
    @chain_step()
    def merge(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        merged = low + high
        return merged


@pytest.fixture
def x() -> torch.Tensor:
    return torch.arange(4, dtype=torch.float32)


def test_a_linear_chain_runs_encoder_then_decoder_then_head(x):
    chain = ModelChain(encoder=Encode(), decoder=Decode(), head=Head())

    assert chain.inputs == {"image": torch.Tensor}
    assert chain(x).tolist() == [16.0]


def test_declaration_order_does_not_decide_call_order(x):
    backwards = ModelChain(head=Head(), decoder=Decode(), encoder=Encode())

    assert [step.method.name for step in backwards._steps] == [
        "encode",
        "decode",
        "logits",
    ]
    assert backwards(x).tolist() == [16.0]


def test_a_step_may_produce_several_keys_at_once(x):
    chain = ModelChain(Split(), Merge())

    ctx = chain(x)
    assert ctx["low"].tolist() == [-1.0, 0.0, 1.0, 2.0]
    assert ctx["high"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert ctx["merged"].tolist() == [0.0, 2.0, 4.0, 6.0]


def test_a_step_merging_two_upstream_keys_runs_once(x):
    calls: list[str] = []

    class CountingMerge(nn.Module):
        @chain_step()
        def merge(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
            calls.append("merge")
            merged = low + high
            return merged

    ModelChain(Split(), CountingMerge())(x)

    assert calls == ["merge"]


def test_no_head_returns_the_merged_context(x):
    ctx = ModelChain(Split(), Merge())(x)

    assert set(ctx) == {"image", "low", "high", "merged"}


def test_one_head_returns_its_bare_tensor(x):
    result = ModelChain(encoder=Encode(), decoder=Decode(), head=Head())(x)

    assert isinstance(result, torch.Tensor)


def test_two_heads_return_one_tensor_each_under_their_stage_names(x):
    chain = ModelChain(encoder=Encode(), decoder=Decode(), a=Head(), b=OtherHead())

    result = chain(x)
    assert sorted(result) == ["a", "b"]
    assert result["a"].tolist() == [16.0]
    assert result["b"].tolist() == [4.0]


def test_inputs_follow_declaration_order_not_call_order(x):
    class Late(nn.Module):
        @chain_step()
        def use(self, seed: torch.Tensor, early: torch.Tensor) -> torch.Tensor:
            used = seed + early
            return used

    class Early(nn.Module):
        @chain_step()
        def make(self, origin: torch.Tensor) -> torch.Tensor:
            early = origin * 3
            return early

    chain = ModelChain(late=Late(), early=Early())

    assert list(chain.inputs) == ["seed", "origin"]
    assert [step.method.name for step in chain._steps] == ["make", "use"]

    ctx = chain(torch.ones(2), torch.ones(2) * 5)
    assert ctx["used"].tolist() == [16.0, 16.0]


def test_positional_and_keyword_arguments_agree(x):
    chain = ModelChain(encoder=Encode(), decoder=Decode(), head=Head())

    assert chain(x).tolist() == chain(image=x).tolist()


def test_a_module_offering_two_steps_uses_whichever_resolves_first(x):
    class TwoWays(nn.Module):
        @chain_step()
        def from_image(self, image: torch.Tensor) -> torch.Tensor:
            picked = image * 10
            return picked

        @chain_step()
        def from_merged(self, merged: torch.Tensor) -> torch.Tensor:
            picked = merged * 10
            return picked

    chain = ModelChain(Split(), Merge(), TwoWays())

    assert [step.method.name for step in chain._steps][1] == "from_image"


def test_a_module_with_no_chain_step_is_refused():
    with pytest.raises(TypeError, match="no @chain_step method found"):
        ModelChain(nn.Identity())


def test_two_steps_ready_at_once_are_refused_as_ambiguous():
    class Ambiguous(nn.Module):
        @chain_step()
        def one(self, image: torch.Tensor) -> torch.Tensor:
            first = image
            return first

        @chain_step()
        def two(self, image: torch.Tensor) -> torch.Tensor:
            second = image
            return second

    with pytest.raises(TypeError, match="ambiguous"):
        ModelChain(Ambiguous())


def test_mutually_dependent_modules_are_refused_as_a_cycle():
    class Forward(nn.Module):
        @chain_step()
        def step(self, left: torch.Tensor) -> torch.Tensor:
            right = left
            return right

    class Backward(nn.Module):
        @chain_step()
        def step(self, right: torch.Tensor) -> torch.Tensor:
            left = right
            return left

    with pytest.raises(CycleError) as raised:
        ModelChain(Forward(), Backward())

    named = {
        f"{type(step.module).__name__}.{step.method.name}"
        for step in raised.value.args[1]
    }
    assert named == {"Forward.step", "Backward.step"}


def test_a_cycle_names_the_key_joining_each_pair_of_steps():
    class Forward(nn.Module):
        @chain_step()
        def step(self, left: torch.Tensor) -> torch.Tensor:
            right = left
            return right

    class Backward(nn.Module):
        @chain_step()
        def step(self, right: torch.Tensor) -> torch.Tensor:
            left = right
            return left

    with pytest.raises(CycleError) as raised:
        ModelChain(Forward(), Backward())

    message = raised.value.args[0]
    assert "-(right)-> Backward.step" in message
    assert "-(left)-> Forward.step" in message


def test_a_positional_name_colliding_with_a_keyword_name_is_refused():
    with pytest.raises(ValueError, match="collide"):
        ModelChain(Encode(), stage_0=Decode())


def test_more_positional_arguments_than_inputs_are_refused(x):
    chain = ModelChain(encoder=Encode(), decoder=Decode(), head=Head())

    with pytest.raises(TypeError, match="at most 1 positional"):
        chain(x, x)


def test_a_key_given_twice_is_refused_rather_than_silently_picked(x):
    chain = ModelChain(encoder=Encode(), decoder=Decode(), head=Head())

    with pytest.raises(TypeError, match="both a positional and keyword"):
        chain(x, image=x)


def test_a_missing_required_key_names_the_key_it_wanted():
    chain = ModelChain(encoder=Encode(), decoder=Decode(), head=Head())

    with pytest.raises(KeyError, match="image"):
        chain()


def test_submodules_are_registered_under_their_stage_names():
    chain = ModelChain(encoder=Encode(), decoder=Decode(), head=Head())

    assert [name for name, _ in chain.named_children()] == [
        "encoder",
        "decoder",
        "head",
    ]


def test_positional_modules_are_named_by_position():
    chain = ModelChain(Split(), Merge())

    assert [name for name, _ in chain.named_children()] == ["stage_0", "stage_1"]


def test_repr_reports_inputs_and_the_resolved_flow():
    chain = ModelChain(encoder=Encode(), decoder=Decode(), head=Head())

    flow = repr(chain).split("\n\n")[0:2]
    assert flow[0] == "inputs: image: Tensor"
    assert "encoder: Encode.encode(image: Tensor) -> {feature_map: Tensor}" in flow[1]
    assert "head: Head.logits(decoded: Tensor) -> Tensor" in flow[1]


def test_a_forward_step_runs_module_hooks(x):
    class Forward(nn.Module):
        @chain_step(head=True)
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return image * 2

    module = Forward()
    captured = []
    module.register_forward_hook(lambda module, args, output: captured.append(output))
    torch.testing.assert_close(ModelChain(module)(x), x * 2)
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], x * 2)


def test_an_undecorated_override_removes_the_inherited_step():
    class Plain(Encode):
        def encode(self, image: torch.Tensor) -> torch.Tensor:
            return image

    with pytest.raises(TypeError, match="no @chain_step"):
        ModelChain(Plain())


def test_two_selected_producers_cannot_overwrite_one_output():
    with pytest.raises(ValueError, match="feature_map.*both"):
        ModelChain(first=Encode(), second=Encode())


def test_a_producer_and_consumer_must_agree_on_the_value_type():
    class Consumer(nn.Module):
        @chain_step(head=True)
        def forward(self, feature_map: list) -> torch.Tensor:
            return torch.tensor(len(feature_map))

    with pytest.raises(TypeError, match="feature_map.*producer declares"):
        ModelChain(Encode(), Consumer())
