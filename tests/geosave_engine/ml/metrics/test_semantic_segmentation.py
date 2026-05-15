"""
Thorough tests for SemanticSegmentationMetrics.

Value assertions use known inputs (perfect preds, all-wrong, ignore masks)
so failures point to a real regression, not a flaky random input.
"""
import pytest
import torch

from geosave_engine.ml.metrics.semantic_segmentation import SemanticSegmentationMetrics

NUM_CLASSES = 4
LABELS = ["background", "water", "vegetation", "urban"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perfect(num_classes: int, h: int = 8, w: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Balanced targets + identical preds (100% correct)."""
    targets = torch.arange(num_classes, dtype=torch.long).repeat(h * w // num_classes + 1)[: h * w]
    targets = targets.reshape(1, h, w)
    return targets.clone(), targets.clone()


def _all_wrong_binary(n: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """2-class, perfectly wrong: pred=1-target, balanced targets."""
    half = n // 2
    targets = torch.cat([torch.zeros(half, dtype=torch.long), torch.ones(half, dtype=torch.long)])
    targets = targets.reshape(1, 8, 8)
    preds = 1 - targets
    return preds, targets


# ---------------------------------------------------------------------------
# Structure — default collection keys
# ---------------------------------------------------------------------------

class TestDefaultStructure:
    def test_all_aggregate_keys_present(self):
        preds, targets = _perfect(NUM_CLASSES)
        m = SemanticSegmentationMetrics(num_classes=NUM_CLASSES)
        result = m(preds, targets)
        expected = {"accuracy_macro", "f1_macro", "iou_macro", "precision_macro", "recall_macro", "mcc", "kappa"}
        missing = expected - result.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_per_class_keys_use_label_names(self):
        preds, targets = _perfect(NUM_CLASSES)
        m = SemanticSegmentationMetrics(num_classes=NUM_CLASSES, labels=LABELS)
        result = m(preds, targets)
        for prefix in ("multiclassaccuracy", "multiclassf1score", "multiclassjaccardindex"):
            for label in LABELS:
                key = f"{prefix}_{label}"
                assert key in result, f"Missing per-class key {key!r}"

    def test_per_class_keys_numeric_when_no_labels(self):
        preds, targets = _perfect(NUM_CLASSES)
        m = SemanticSegmentationMetrics(num_classes=NUM_CLASSES)
        result = m(preds, targets)
        for i in range(NUM_CLASSES):
            assert f"multiclassaccuracy_{i}" in result, f"Missing numeric per-class key for class {i}"

    def test_num_classes_2_works(self):
        preds, targets = _all_wrong_binary()
        m = SemanticSegmentationMetrics(num_classes=2)
        result = m(preds, targets)
        assert "accuracy_macro" in result

    def test_all_values_are_tensors(self):
        preds, targets = _perfect(NUM_CLASSES)
        m = SemanticSegmentationMetrics(num_classes=NUM_CLASSES)
        result = m(preds, targets)
        for k, v in result.items():
            assert isinstance(v, torch.Tensor), f"{k} is not a Tensor"

    def test_scalar_metrics_are_0dim(self):
        preds, targets = _perfect(NUM_CLASSES)
        m = SemanticSegmentationMetrics(num_classes=NUM_CLASSES)
        result = m(preds, targets)
        for key in ("mcc", "kappa"):
            assert result[key].ndim == 0, f"{key} should be 0-dim scalar"

    def test_per_class_values_are_scalar_per_label(self):
        preds, targets = _perfect(NUM_CLASSES)
        m = SemanticSegmentationMetrics(num_classes=NUM_CLASSES, labels=LABELS)
        result = m(preds, targets)
        for label in LABELS:
            v = result[f"multiclassaccuracy_{label}"]
            assert v.numel() == 1, f"Per-class value for {label!r} should have 1 element"


# ---------------------------------------------------------------------------
# Override behaviour
# ---------------------------------------------------------------------------

class TestOverrides:
    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            SemanticSegmentationMetrics(num_classes=NUM_CLASSES, metrics=["bad_metric"])

    def test_empty_list_is_noop(self):
        preds, targets = _perfect(NUM_CLASSES)
        m_default = SemanticSegmentationMetrics(num_classes=NUM_CLASSES)
        m_empty = SemanticSegmentationMetrics(num_classes=NUM_CLASSES, metrics=[])
        assert set(m_default(preds, targets).keys()) == set(m_empty(preds, targets).keys())

    def test_bare_name_no_mode_tokens_is_noop(self):
        """metrics=['f1'] (no dot tokens) must not change f1 config."""
        preds, targets = _perfect(NUM_CLASSES)
        m_default = SemanticSegmentationMetrics(num_classes=NUM_CLASSES)
        m_bare = SemanticSegmentationMetrics(num_classes=NUM_CLASSES, metrics=["f1"])
        assert set(m_default(preds, targets).keys()) == set(m_bare(preds, targets).keys())

    def test_f1_macro_only(self):
        preds, targets = _perfect(NUM_CLASSES, 16, 16)
        m = SemanticSegmentationMetrics(num_classes=NUM_CLASSES, labels=LABELS, metrics=["f1.macro"])
        result = m(preds, targets)
        assert "f1_macro" in result
        assert all(f"multiclassf1score_{lbl}" not in result for lbl in LABELS)

    def test_f1_per_class_only(self):
        preds, targets = _perfect(NUM_CLASSES, 16, 16)
        m = SemanticSegmentationMetrics(num_classes=NUM_CLASSES, labels=LABELS, metrics=["f1.per_class"])
        result = m(preds, targets)
        assert all(f"multiclassf1score_{lbl}" in result for lbl in LABELS)
        assert "f1_macro" not in result

    def test_f1_macro_and_per_class_both_present(self):
        preds, targets = _perfect(NUM_CLASSES, 16, 16)
        m = SemanticSegmentationMetrics(num_classes=NUM_CLASSES, labels=LABELS, metrics=["f1.macro.per_class"])
        result = m(preds, targets)
        assert "f1_macro" in result
        assert all(f"multiclassf1score_{lbl}" in result for lbl in LABELS)

    def test_accuracy_exclude(self):
        preds, targets = _perfect(NUM_CLASSES, 16, 16)
        m = SemanticSegmentationMetrics(num_classes=NUM_CLASSES, labels=LABELS, metrics=["accuracy.exclude"])
        result = m(preds, targets)
        assert "accuracy_macro" not in result
        assert all(f"multiclassaccuracy_{lbl}" not in result for lbl in LABELS)
        assert "f1_macro" in result  # other metrics untouched

    def test_precision_per_class_removes_macro(self):
        preds, targets = _perfect(NUM_CLASSES, 16, 16)
        m = SemanticSegmentationMetrics(num_classes=NUM_CLASSES, labels=LABELS, metrics=["precision.per_class"])
        result = m(preds, targets)
        assert "precision_macro" not in result
        assert all(f"multiclassprecision_{lbl}" in result for lbl in LABELS)

    def test_mcc_exclude(self):
        preds, targets = _perfect(NUM_CLASSES)
        m = SemanticSegmentationMetrics(num_classes=NUM_CLASSES, metrics=["mcc.exclude"])
        result = m(preds, targets)
        assert "mcc" not in result
        assert "kappa" in result

    def test_kappa_exclude(self):
        preds, targets = _perfect(NUM_CLASSES)
        m = SemanticSegmentationMetrics(num_classes=NUM_CLASSES, metrics=["kappa.exclude"])
        result = m(preds, targets)
        assert "kappa" not in result
        assert "mcc" in result

    def test_scalar_metric_ignores_per_class_token(self):
        """mcc.per_class — mode silently ignored, mcc still scalar."""
        preds, targets = _perfect(NUM_CLASSES)
        m = SemanticSegmentationMetrics(num_classes=NUM_CLASSES, metrics=["mcc.per_class"])
        result = m(preds, targets)
        assert "mcc" in result
        assert result["mcc"].ndim == 0

    def test_multiple_overrides_combined(self):
        preds, targets = _perfect(NUM_CLASSES, 16, 16)
        m = SemanticSegmentationMetrics(
            num_classes=NUM_CLASSES,
            labels=LABELS,
            metrics=["f1.macro", "accuracy.exclude", "mcc.exclude"],
        )
        result = m(preds, targets)
        assert "f1_macro" in result
        assert "accuracy_macro" not in result
        assert "mcc" not in result
        assert "kappa" in result


# ---------------------------------------------------------------------------
# Value correctness
# ---------------------------------------------------------------------------

class TestValueCorrectness:
    def test_perfect_preds_aggregate_all_one(self):
        preds, targets = _perfect(NUM_CLASSES, 16, 16)
        m = SemanticSegmentationMetrics(num_classes=NUM_CLASSES)
        result = m(preds, targets)
        for key in ("accuracy_macro", "f1_macro", "iou_macro", "precision_macro", "recall_macro"):
            assert result[key].item() == pytest.approx(1.0, abs=1e-5), \
                f"{key} should be 1.0 for perfect predictions"

    def test_perfect_preds_per_class_all_one(self):
        preds, targets = _perfect(NUM_CLASSES, 16, 16)
        m = SemanticSegmentationMetrics(num_classes=NUM_CLASSES, labels=LABELS)
        result = m(preds, targets)
        for label in LABELS:
            v = result[f"multiclassaccuracy_{label}"].item()
            assert v == pytest.approx(1.0, abs=1e-5), f"Per-class accuracy for {label!r} should be 1.0"

    def test_all_wrong_binary_accuracy_zero(self):
        preds, targets = _all_wrong_binary()
        m = SemanticSegmentationMetrics(num_classes=2)
        result = m(preds, targets)
        assert result["accuracy_macro"].item() == pytest.approx(0.0, abs=1e-5)

    def test_all_wrong_binary_iou_zero(self):
        preds, targets = _all_wrong_binary()
        m = SemanticSegmentationMetrics(num_classes=2)
        result = m(preds, targets)
        assert result["iou_macro"].item() == pytest.approx(0.0, abs=1e-5)

    def test_mcc_range(self):
        preds = torch.randint(0, NUM_CLASSES, (1, 32, 32))
        targets = torch.randint(0, NUM_CLASSES, (1, 32, 32))
        m = SemanticSegmentationMetrics(num_classes=NUM_CLASSES)
        result = m(preds, targets)
        v = result["mcc"].item()
        assert -1.0 <= v <= 1.0, f"MCC {v} outside [-1, 1]"

    def test_mcc_perfect_is_one(self):
        preds, targets = _perfect(NUM_CLASSES, 16, 16)
        m = SemanticSegmentationMetrics(num_classes=NUM_CLASSES)
        result = m(preds, targets)
        assert result["mcc"].item() == pytest.approx(1.0, abs=1e-5)

    def test_ignore_index_excludes_corrupted_pixel(self):
        """Pixel with ignore_index=255 in target must not affect result."""
        # Build perfect preds for 2 classes
        targets = torch.zeros(1, 8, 8, dtype=torch.long)
        targets[0, 4:, :] = 1
        preds = targets.clone()

        # Corrupt one pixel's prediction, then mark that target as ignore
        preds[0, 0, 0] = 1  # wrong
        targets[0, 0, 0] = 255  # but ignored

        m = SemanticSegmentationMetrics(num_classes=2, ignore_index=255)
        result = m(preds, targets)
        assert result["accuracy_macro"].item() == pytest.approx(1.0, abs=1e-5), \
            "Ignored pixel must not penalise accuracy"

    def test_ignore_index_affects_result_when_not_ignored(self):
        """Same setup without ignore_index must differ (proves ignore actually does something)."""
        targets = torch.zeros(1, 8, 8, dtype=torch.long)
        targets[0, 4:, :] = 1
        preds = targets.clone()
        preds[0, 0, 0] = 1  # intentionally wrong, not ignored here

        m = SemanticSegmentationMetrics(num_classes=2)
        result = m(preds, targets)
        # With the wrong pixel counted accuracy should be < 1.0
        assert result["accuracy_macro"].item() < 1.0


# ---------------------------------------------------------------------------
# Stateful behaviour — accumulate and reset
# ---------------------------------------------------------------------------

class TestStateful:
    def test_two_updates_equal_one_cat(self):
        torch.manual_seed(42)
        p1 = torch.randint(0, NUM_CLASSES, (1, 8, 8))
        t1 = torch.randint(0, NUM_CLASSES, (1, 8, 8))
        p2 = torch.randint(0, NUM_CLASSES, (1, 8, 8))
        t2 = torch.randint(0, NUM_CLASSES, (1, 8, 8))

        m_accum = SemanticSegmentationMetrics(num_classes=NUM_CLASSES)
        m_accum.update(p1, t1)
        m_accum.update(p2, t2)
        r_accum = m_accum.compute()

        m_single = SemanticSegmentationMetrics(num_classes=NUM_CLASSES)
        r_single = m_single(torch.cat([p1, p2]), torch.cat([t1, t2]))

        for key in r_single:
            assert torch.allclose(r_accum[key], r_single[key], atol=1e-5), \
                f"Accumulation mismatch at {key!r}"

    def test_reset_restores_fresh_state(self):
        preds, targets = _perfect(NUM_CLASSES, 16, 16)

        m = SemanticSegmentationMetrics(num_classes=NUM_CLASSES)
        result_before = m(preds, targets)
        m.reset()
        m.update(preds, targets)
        result_after = m.compute()

        for key in result_before:
            assert torch.allclose(result_before[key], result_after[key], atol=1e-6), \
                f"reset() broke state for key {key!r}"

    def test_reset_clears_accumulator(self):
        """After reset, computing with different data must not include earlier data."""
        torch.manual_seed(7)
        p_dirty = torch.randint(0, NUM_CLASSES, (1, 8, 8))
        t_dirty = torch.randint(0, NUM_CLASSES, (1, 8, 8))
        p_clean, t_clean = _perfect(NUM_CLASSES, 16, 16)

        m = SemanticSegmentationMetrics(num_classes=NUM_CLASSES)
        m.update(p_dirty, t_dirty)
        m.reset()
        m.update(p_clean, t_clean)
        result_reset = m.compute()

        m2 = SemanticSegmentationMetrics(num_classes=NUM_CLASSES)
        result_fresh = m2(p_clean, t_clean)

        for key in result_fresh:
            assert torch.allclose(result_reset[key], result_fresh[key], atol=1e-6), \
                f"Stale data leaked through reset at {key!r}"
