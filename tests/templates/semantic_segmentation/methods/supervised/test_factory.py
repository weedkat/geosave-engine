import pytest
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR

from templates.semantic_segmentation.methods.supervised.src.factory import (
    build_loss,
    build_model,
    build_optimizer,
    build_scheduler,
)


class TestBuildLoss:
    def test_ce_loss_type(self):
        assert isinstance(build_loss({"name": "CELoss", "init_args": {}}), CrossEntropyLoss)

    def test_ce_loss_ignore_index_forwarded(self):
        loss = build_loss({"name": "CELoss", "init_args": {"ignore_index": 255}})
        assert loss.ignore_index == 255

    def test_ohem_loss_type(self):
        from geosave_engine.ml.loss import ProbOhemCrossEntropy2d

        loss = build_loss({"name": "OHEMLoss", "init_args": {"ignore_index": 255}})
        assert isinstance(loss, ProbOhemCrossEntropy2d)

    def test_missing_init_args_uses_defaults(self):
        # init_args key absent — should not raise
        build_loss({"name": "CELoss"})

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="FakeLoss"):
            build_loss({"name": "FakeLoss", "init_args": {}})

    def test_name_case_insensitive(self):
        assert isinstance(build_loss({"name": "celoss", "init_args": {}}), CrossEntropyLoss)


class TestBuildModel:
    def test_unet_type(self):
        import segmentation_models_pytorch as smp

        model = build_model({
            "name": "unet",
            "init_args": {"num_classes": 2, "in_channels": 3, "encoder_weights": None},
        })
        assert isinstance(model, smp.Unet)

    def test_model_is_nn_module(self):
        model = build_model({
            "name": "unet",
            "init_args": {"num_classes": 4, "in_channels": 1, "encoder_weights": None},
        })
        assert isinstance(model, nn.Module)

    def test_model_forward_shape(self):
        model = build_model({
            "name": "unet",
            "init_args": {"num_classes": 3, "in_channels": 2, "encoder_weights": None},
        })
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 2, 64, 64))
        assert out.shape == (1, 3, 64, 64)

    def test_missing_init_args_key_uses_defaults(self):
        # init_args absent — builder should not crash for models with all defaults
        build_model({"name": "unet", "init_args": {"num_classes": 2, "in_channels": 3, "encoder_weights": None}})

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            build_model({"name": "ghost_net", "init_args": {}})


class TestBuildScheduler:
    @pytest.fixture
    def optimizer(self):
        return torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.01)

    def test_none_config_returns_none(self, optimizer):
        assert build_scheduler(None, optimizer) is None

    def test_cosine_annealing_type(self, optimizer):
        sched = build_scheduler(
            {"name": "CosineAnnealingLR", "init_args": {"T_max": 10}},
            optimizer,
        )
        assert isinstance(sched, CosineAnnealingLR)

    def test_cosine_annealing_t_max_set(self, optimizer):
        sched = build_scheduler(
            {"name": "CosineAnnealingLR", "init_args": {"T_max": 42}},
            optimizer,
        )
        assert sched.T_max == 42

    def test_unknown_scheduler_raises(self, optimizer):
        with pytest.raises(ValueError):
            build_scheduler({"name": "WarmupScheduler", "init_args": {}}, optimizer)


class TestBuildOptimizer:
    @pytest.fixture
    def model(self):
        return nn.Linear(4, 2)

    def test_adamw_split_type(self, model):
        opt = build_optimizer(
            {"name": "AdamW.split", "init_args": {"encoder_lr": 1e-4, "decoder_lr": 1e-3}},
            model,
        )
        assert isinstance(opt, torch.optim.AdamW)

    def test_adamw_split_two_param_groups(self, model):
        opt = build_optimizer(
            {"name": "AdamW.split", "init_args": {"encoder_lr": 1e-4, "decoder_lr": 1e-3}},
            model,
        )
        assert len(opt.param_groups) == 2

    def test_adamw_split_lr_values(self, model):
        opt = build_optimizer(
            {"name": "AdamW.split", "init_args": {"encoder_lr": 1e-4, "decoder_lr": 5e-3}},
            model,
        )
        lrs = {pg["lr"] for pg in opt.param_groups}
        assert lrs == {1e-4, 5e-3}

    def test_adamw_default(self, model):
        opt = build_optimizer(
            {"name": "AdamW.default", "init_args": {"lr": 1e-3}},
            model,
        )
        assert isinstance(opt, torch.optim.AdamW)

    def test_unknown_optimizer_raises(self, model):
        with pytest.raises(ValueError):
            build_optimizer({"name": "LAMB.default", "init_args": {}}, model)

    def test_unknown_method_raises(self, model):
        with pytest.raises(ValueError):
            build_optimizer({"name": "AdamW.cyclical", "init_args": {}}, model)
