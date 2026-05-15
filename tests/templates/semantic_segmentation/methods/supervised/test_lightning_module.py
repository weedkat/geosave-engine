import pytest
import torch

from templates.semantic_segmentation.methods.supervised.src.lightning_module import (
    GeosaveLightningModule,
)

NUM_CLASSES = 3
BATCH = 2
H, W = 8, 8

_BASE_KWARGS = dict(
    model_config={"name": "unet", "init_args": {}},
    optim_config={"name": "AdamW.split", "init_args": {"encoder_lr": 1e-4, "decoder_lr": 1e-3}},
    loss_config={"name": "CELoss", "init_args": {}},
    input_size=(H, W),
    num_classes=NUM_CLASSES,
    in_channels=2,
)


def _make_module(ignore_index: int = 255, **override) -> GeosaveLightningModule:
    kwargs = {**_BASE_KWARGS, "ignore_index": ignore_index, **override}
    m = GeosaveLightningModule(**kwargs)
    m.register_buffer("class_thresholds", torch.full((kwargs["num_classes"],), 0.5))
    return m


@pytest.fixture
def module() -> GeosaveLightningModule:
    return _make_module()


@pytest.fixture
def logits() -> torch.Tensor:
    return torch.randn(BATCH, NUM_CLASSES, H, W)


class TestGeosaveLightningModuleInit:
    def test_stores_num_classes(self):
        assert _make_module().num_classes == NUM_CLASSES

    def test_stores_ignore_index_default(self):
        assert _make_module().ignore_index == 255

    def test_stores_ignore_index_custom(self):
        assert _make_module(ignore_index=0).ignore_index == 0

    def test_palette_none_when_no_class_map(self):
        assert _make_module().palette is None

    def test_palette_built_from_class_map(self):
        class_map = {0: {"name": "bg", "color": "#000000"}, 1: {"name": "fg", "color": "#FFFFFF"}}
        m = GeosaveLightningModule(**{**_BASE_KWARGS, "class_map": class_map})
        assert m.palette == {0: "#000000", 1: "#FFFFFF"}

    def test_loss_built_on_init(self):
        import torch.nn as nn
        assert isinstance(_make_module().loss, nn.Module)


class TestPostprocessOutputShape:
    def test_pred_shape(self, module, logits):
        preds, _ = module.postprocess(logits)
        assert preds.shape == (BATCH, H, W)

    def test_prob_shape(self, module, logits):
        _, probs = module.postprocess(logits)
        assert probs.shape == (BATCH, H, W)

    def test_single_pixel(self, module):
        out_preds, out_probs = module.postprocess(torch.randn(1, NUM_CLASSES, 1, 1))
        assert out_preds.shape == (1, 1, 1)
        assert out_probs.shape == (1, 1, 1)

    def test_batch_size_one(self, module):
        preds, _ = module.postprocess(torch.randn(1, NUM_CLASSES, H, W))
        assert preds.shape == (1, H, W)


class TestPostprocessProbs:
    def test_probs_in_unit_range(self, module, logits):
        _, probs = module.postprocess(logits)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_probs_are_softmax_max(self, module):
        logits = torch.randn(1, NUM_CLASSES, 4, 4)
        _, probs = module.postprocess(logits)
        expected = logits.softmax(dim=1).max(dim=1).values
        assert torch.allclose(probs, expected)


class TestPostprocessPredValues:
    def test_preds_only_valid_classes_or_ignore_index(self, module, logits):
        preds, _ = module.postprocess(logits)
        valid = torch.isin(preds, torch.tensor([*range(NUM_CLASSES), module.ignore_index]))
        assert valid.all()

    def test_argmax_is_argmax_of_logits(self, module):
        # threshold=0 means no pixel gets suppressed; preds == argmax(logits)
        module.class_thresholds = torch.zeros(NUM_CLASSES)
        logits = torch.randn(1, NUM_CLASSES, 4, 4)
        preds, _ = module.postprocess(logits)
        assert torch.equal(preds, logits.argmax(dim=1))


class TestPostprocessThreshold:
    def test_threshold_one_all_become_ignore_index(self, module, logits):
        module.class_thresholds = torch.ones(NUM_CLASSES)
        preds, _ = module.postprocess(logits)
        assert (preds == module.ignore_index).all()

    def test_threshold_zero_no_ignore_index(self, module, logits):
        module.class_thresholds = torch.zeros(NUM_CLASSES)
        preds, _ = module.postprocess(logits)
        assert (preds != module.ignore_index).all()

    def test_per_class_threshold_applied_independently(self, module):
        # Use logit=10 (not 1e6): softmax([10,0,0]) ≈ 0.9999 — strictly < 1.0
        # so threshold=1.0 for class 1 reliably suppresses those pixels.
        logits = torch.zeros((1, NUM_CLASSES, 1, 4))
        logits[0, 0, 0, :2] = 10.0   # class 0 wins, max_prob ≈ 0.9999
        logits[0, 1, 0, 2:] = 10.0   # class 1 wins, max_prob ≈ 0.9999

        # class 0 threshold=0 (always pass), class 1 threshold=1 (never pass)
        module.class_thresholds = torch.tensor([0.0, 1.0, 0.5])
        preds, _ = module.postprocess(logits)

        assert (preds[0, 0, :2] == 0).all(), "class-0 pixels should pass"
        assert (preds[0, 0, 2:] == module.ignore_index).all(), "class-1 pixels below threshold"

    def test_high_confidence_pixel_not_suppressed(self, module):
        logits = torch.full((1, NUM_CLASSES, 1, 1), -1e6)
        logits[0, 2, 0, 0] = 1e6  # class 2 with near-prob 1.0
        module.class_thresholds = torch.tensor([0.9, 0.9, 0.9])
        preds, _ = module.postprocess(logits)
        assert preds[0, 0, 0] == 2


class TestPostprocessMask:
    def test_partial_mask_excluded_pixels_become_ignore_index(self, module, logits):
        module.class_thresholds = torch.zeros(NUM_CLASSES)  # threshold off
        mask = torch.ones(BATCH, H, W, dtype=torch.bool)
        mask[:, 0, :] = False  # exclude first row
        preds, _ = module.postprocess(logits, mask=mask)
        assert (preds[:, 0, :] == module.ignore_index).all()

    def test_partial_mask_included_pixels_unaffected(self, module, logits):
        module.class_thresholds = torch.zeros(NUM_CLASSES)  # threshold off
        mask = torch.ones(BATCH, H, W, dtype=torch.bool)
        mask[:, 0, :] = False
        preds, _ = module.postprocess(logits, mask=mask)
        assert (preds[:, 1:, :] != module.ignore_index).all()

    def test_all_false_mask_all_become_ignore_index(self, module, logits):
        module.class_thresholds = torch.zeros(NUM_CLASSES)  # threshold off
        mask = torch.zeros(BATCH, H, W, dtype=torch.bool)
        preds, _ = module.postprocess(logits, mask=mask)
        assert (preds == module.ignore_index).all()

    def test_all_true_mask_same_as_no_mask(self, module, logits):
        module.class_thresholds = torch.zeros(NUM_CLASSES)
        mask = torch.ones(BATCH, H, W, dtype=torch.bool)
        preds_masked, _ = module.postprocess(logits, mask=mask)
        preds_none, _ = module.postprocess(logits, mask=None)
        assert torch.equal(preds_masked, preds_none)

    def test_mask_and_threshold_both_apply(self, module):
        module.class_thresholds = torch.zeros(NUM_CLASSES)  # threshold: all pass
        logits = torch.randn(1, NUM_CLASSES, 4, 4)

        mask = torch.ones(1, 4, 4, dtype=torch.bool)
        mask[0, :2, :] = False  # mask first 2 rows

        preds, _ = module.postprocess(logits, mask=mask)
        assert (preds[0, :2, :] == module.ignore_index).all()  # masked rows
        assert (preds[0, 2:, :] != module.ignore_index).all()  # unmasked, threshold=0

    def test_none_mask_does_not_crash(self, module, logits):
        preds, _ = module.postprocess(logits, mask=None)
        assert preds.shape == (BATCH, H, W)

    def test_mask_does_not_alter_probs(self, module, logits):
        module.class_thresholds = torch.zeros(NUM_CLASSES)
        mask = torch.ones(BATCH, H, W, dtype=torch.bool)
        mask[:, 0, :] = False
        _, probs_masked = module.postprocess(logits, mask=mask)
        _, probs_none = module.postprocess(logits, mask=None)
        # max_probs come from softmax before masking; should be identical
        assert torch.allclose(probs_masked, probs_none)
