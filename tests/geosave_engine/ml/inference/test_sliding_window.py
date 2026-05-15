import pytest
import torch
import torch.nn as nn

from geosave_engine.ml.inference.sliding_window import infer_sliding_window


# ---------------------------------------------------------------------------
# Dummy models
# ---------------------------------------------------------------------------

class _CountingIdentity(nn.Module):
    """Identity that counts forward calls and records input shapes."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0
        self.seen_shapes: list[torch.Size] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        self.seen_shapes.append(x.shape)
        return x.clone()


class _ConstantAdder(nn.Module):
    def __init__(self, constant: float) -> None:
        super().__init__()
        self.constant = constant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.constant


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def identity() -> nn.Module:
    return nn.Identity()


@pytest.fixture
def img() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(1, 3, 128, 128)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_int_grid(self, identity, img):
        assert infer_sliding_window(identity, img, 64, "cpu").shape == img.shape

    def test_tuple_grid(self, identity, img):
        assert infer_sliding_window(identity, img, (64, 48), "cpu").shape == img.shape

    def test_non_square_image(self, identity):
        img = torch.rand(1, 2, 96, 160)
        assert infer_sliding_window(identity, img, 64, "cpu").shape == img.shape

    def test_batch_gt_one(self, identity):
        img = torch.rand(4, 3, 128, 128)
        assert infer_sliding_window(identity, img, 64, "cpu").shape == img.shape

    def test_single_channel(self, identity):
        img = torch.rand(1, 1, 128, 128)
        assert infer_sliding_window(identity, img, 64, "cpu").shape == img.shape

    def test_grid_equals_image_size(self, identity):
        img = torch.rand(1, 3, 64, 64)
        out = infer_sliding_window(identity, img, 64, "cpu", overlap_ratio=0.0, pad_size=0)
        assert out.shape == img.shape

    def test_output_dtype_float32(self):
        img = torch.rand(1, 3, 128, 128)  # float32 input
        out = infer_sliding_window(nn.Identity(), img, 64, "cpu")
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Value correctness
# ---------------------------------------------------------------------------

class TestValueCorrectness:
    def test_identity_preserves_values(self, identity, img):
        out = infer_sliding_window(identity, img, 64, "cpu", overlap_ratio=0.5)
        assert torch.allclose(out, img, atol=1e-5), "Identity model must preserve pixel values"

    def test_all_zeros_input(self, identity):
        img = torch.zeros(1, 3, 128, 128)
        out = infer_sliding_window(identity, img, 64, "cpu")
        assert torch.allclose(out, img, atol=1e-6)

    def test_constant_offset_model(self, img):
        K = 7.5
        model = _ConstantAdder(K)
        out = infer_sliding_window(model, img, 64, "cpu", overlap_ratio=0.5)
        expected = img + K
        # Hann-weighted average of (patch + K) = input + K everywhere
        assert torch.allclose(out, expected.float(), atol=1e-4), \
            "Constant-offset model: output must equal input + constant"

    def test_identity_no_overlap_shape(self, identity):
        # Hann window endpoints = 0, so patch boundary pixels get weight 0
        # with zero overlap (covered by one patch only). Values not preserved.
        # Test shape only — value preservation requires overlap > 0.
        img = torch.rand(1, 2, 128, 128)
        out = infer_sliding_window(identity, img, 32, "cpu", overlap_ratio=0.0)
        assert out.shape == img.shape

    def test_identity_high_overlap(self, identity):
        # Overlap ensures every pixel gets non-zero Hann weight from multiple patches
        img = torch.rand(1, 2, 128, 128)
        out = infer_sliding_window(identity, img, 32, "cpu", overlap_ratio=0.75)
        assert torch.allclose(out, img, atol=1e-4)


# ---------------------------------------------------------------------------
# Model interaction
# ---------------------------------------------------------------------------

class TestModelInteraction:
    def test_patch_shape_matches_grid_int(self):
        model = _CountingIdentity()
        img = torch.rand(1, 3, 128, 128)
        infer_sliding_window(model, img, 64, "cpu")
        assert all(s[-2:] == torch.Size([64, 64]) for s in model.seen_shapes), \
            "Every patch must have spatial dims equal to grid_size"

    def test_patch_shape_matches_grid_tuple(self):
        model = _CountingIdentity()
        img = torch.rand(1, 3, 128, 128)
        infer_sliding_window(model, img, (48, 32), "cpu")
        assert all(s[-2:] == torch.Size([48, 32]) for s in model.seen_shapes)

    def test_batch_dim_propagated_to_model(self):
        model = _CountingIdentity()
        B = 3
        img = torch.rand(B, 2, 128, 128)  # must be > 2*pad_size for reflect pad
        infer_sliding_window(model, img, 32, "cpu")
        assert all(s[0] == B for s in model.seen_shapes)

    def test_single_patch_call_count(self):
        """pad_size=0 + grid=image → exactly one model call."""
        model = _CountingIdentity()
        img = torch.rand(1, 1, 64, 64)
        infer_sliding_window(model, img, 64, "cpu", overlap_ratio=0.0, pad_size=0)
        assert model.call_count == 1

    def test_no_overlap_four_patch_call_count(self):
        """pad_size=0, grid=32, img=64×64, overlap=0 → 2×2=4 calls."""
        model = _CountingIdentity()
        img = torch.rand(1, 1, 64, 64)
        infer_sliding_window(model, img, 32, "cpu", overlap_ratio=0.0, pad_size=0)
        assert model.call_count == 4

    def test_model_called_at_least_once(self):
        model = _CountingIdentity()
        img = torch.rand(1, 3, 128, 128)
        infer_sliding_window(model, img, 64, "cpu")
        assert model.call_count >= 1


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

class TestPadding:
    def test_pad_size_zero_shape(self, identity):
        img = torch.rand(1, 3, 128, 128)
        out = infer_sliding_window(identity, img, 64, "cpu", pad_size=0)
        assert out.shape == img.shape

    def test_pad_size_zero_interior_values_preserved(self, identity):
        # Global image edges get Hann weight=0 from their only covering patch
        # (no reflection padding to push them into interior of another patch).
        # Interior pixels covered by multiple patches → weighted avg = original value.
        img = torch.rand(1, 3, 128, 128)
        out = infer_sliding_window(identity, img, 64, "cpu", overlap_ratio=0.5, pad_size=0)
        assert torch.allclose(out[:, :, 1:-1, 1:-1], img[:, :, 1:-1, 1:-1], atol=1e-4)

    def test_large_pad_size(self, identity):
        # F.pad reflect requires pad_size < image_dim; use 256x256 so 128 < 256
        img = torch.rand(1, 3, 256, 256)
        out = infer_sliding_window(identity, img, 64, "cpu", pad_size=128)
        assert out.shape == img.shape

    def test_grid_too_large_for_padded_image_raises(self):
        model = nn.Identity()
        tiny = torch.rand(1, 1, 16, 16)
        with pytest.raises(ValueError, match="too large"):
            infer_sliding_window(model, tiny, 512, "cpu", pad_size=8)


# ---------------------------------------------------------------------------
# Overlap validation
# ---------------------------------------------------------------------------

class TestOverlapValidation:
    def test_overlap_exactly_zero(self, identity, img):
        out = infer_sliding_window(identity, img, 64, "cpu", overlap_ratio=0.0)
        assert out.shape == img.shape

    def test_overlap_0_75(self, identity, img):
        out = infer_sliding_window(identity, img, 64, "cpu", overlap_ratio=0.75)
        assert out.shape == img.shape

    def test_overlap_one_raises(self, identity, img):
        with pytest.raises(ValueError, match="overlap_ratio"):
            infer_sliding_window(identity, img, 64, "cpu", overlap_ratio=1.0)

    def test_overlap_above_one_raises(self, identity, img):
        with pytest.raises(ValueError, match="overlap_ratio"):
            infer_sliding_window(identity, img, 64, "cpu", overlap_ratio=1.5)

    def test_overlap_negative_raises(self, identity, img):
        with pytest.raises(ValueError, match="overlap_ratio"):
            infer_sliding_window(identity, img, 64, "cpu", overlap_ratio=-0.1)

    def test_stride_zero_raises(self, identity, img):
        # grid=1, overlap=0.5 → stride = int(1 * 0.5) = 0
        with pytest.raises(ValueError, match="overlap_ratio too high"):
            infer_sliding_window(identity, img, 1, "cpu", overlap_ratio=0.5)


# ---------------------------------------------------------------------------
# Grid size validation
# ---------------------------------------------------------------------------

class TestGridValidation:
    def test_grid_zero_raises(self, identity, img):
        with pytest.raises(ValueError, match="grid_size must be > 0"):
            infer_sliding_window(identity, img, 0, "cpu")

    def test_grid_negative_raises(self, identity, img):
        with pytest.raises(ValueError, match="grid_size must be > 0"):
            infer_sliding_window(identity, img, -8, "cpu")

    def test_tuple_with_zero_dim_raises(self, identity, img):
        with pytest.raises(ValueError, match="grid_size must be > 0"):
            infer_sliding_window(identity, img, (0, 64), "cpu")


# ---------------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------------

class TestDevice:
    def test_string_device(self, identity, img):
        out = infer_sliding_window(identity, img, 64, "cpu")
        assert out.device.type == "cpu"

    def test_device_object(self, identity, img):
        out = infer_sliding_window(identity, img, 64, torch.device("cpu"))
        assert out.device.type == "cpu"
