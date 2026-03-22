import lightning as L
import torch

from geosave_engine.ai_tasks.semseg.core.inference import Inference

from .registry import optim_registry, loss_registry, model_registry
from ..core.metric import get_metrics
from ..core.utils import extract_prefixed

from ..core.metadata import MetadataInterpreter


class SemSegModel(L.LightningModule):
    def __init__(self, arch, optim, loss, metadata_dict, transform_dict, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.transform_dict = transform_dict
        self.metadata_dict = metadata_dict
        self.metadata_interpreter = MetadataInterpreter(metadata_dict)
        self.ignore_index = self.metadata_interpreter.ignore_index

        self.arch_cfg = extract_prefixed(kwargs, 'arch')
        self.optim_cfg = extract_prefixed(kwargs, 'optim')
        self.loss_cfg = extract_prefixed(kwargs, 'loss')
        
        self.arch_cfg['nclass'] = self.metadata_interpreter.nclass
        self.arch_cfg['in_channels'] = self.metadata_interpreter.in_channels
        self.model = model_registry.build(arch, **self.arch_cfg)
        
        self.loss_cfg['ignore_index'] = self.ignore_index
        self.loss_fn = loss_registry.build(loss, **self.loss_cfg)

        metrics = get_metrics(
            num_classes=self.metadata_interpreter.nclass, 
            class_names=self.metadata_interpreter.class_names, 
            ignore_index=self.ignore_index
        )
        test_metrics = get_metrics(
            num_classes=self.metadata_interpreter.nclass, 
            class_names=self.metadata_interpreter.class_names, 
            ignore_index=None
        ) 

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = test_metrics.clone(prefix="test/")

        self.calibrating = False
        self.register_buffer("class_thresholds", torch.zeros(self.metadata_interpreter.nclass), persistent=True)

    def setup(self, stage=None):
        if stage in ("validate", "test", "predict"):
            self.inferencer = Inference(self)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optim_cfg = {
            'lr': 5e-5,
            'lr_multi': 40.0,
        }
        optim_cfg.update(self.optim_cfg)

        enc, dec = self.get_encoder_decoder_params()
        encoder_lr = optim_cfg['lr']
        decoder_lr = encoder_lr * optim_cfg['lr_multi']

        parameter = [
            {'params': enc, 'lr': encoder_lr},
            {'params': dec, 'lr': decoder_lr}
        ]
        optimizer = optim_registry.build(self.optim, parameter, **optim_cfg)
        
        return optimizer

    # ================== Training ==================

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.model(imgs)
        loss = self.loss_fn(logits, labels)
     
        self.train_metrics.update(logits, labels)

        self.log("train_loss", loss)
        
        return loss

    def on_train_epoch_end(self):
        output = self.train_metrics.compute()
        self.log_dict(output, prog_bar=True)
        self.train_metrics.reset()
    
    # ================== Validation ==================

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.inferencer(imgs)
        loss = self.loss_fn(logits, labels)
        
        self.val_metrics.update(logits, labels)

        self.log("val_loss", loss)
    
    def on_validation_epoch_end(self):
        output = self.val_metrics.compute()
        self.log_dict(output, prog_bar=True)
        self.val_metrics.reset()

    # ================== Testing ==================

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.inferencer(imgs)

        preds, max_probs = self.postprocess(logits)
        self.test_metrics.update(preds, labels)

        return preds, max_probs, labels

    def on_test_epoch_end(self):
        output = self.test_metrics.compute()
        self.log_dict(output, prog_bar=True)
        self.test_metrics.reset()

    # ================== Prediction ==================

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        imgs = batch
        logits = self.inferencer(imgs)
        
        return self.postprocess(logits)

    # ================== Utility Methods ==================

    def __repr__(self):
        # Now your model can describe exactly what it predicts
        classes = self.metadata.get("classes", "Unknown")
        return f"SegmentationModel(predicting_classes={classes})"
    
    def lock_encoder(self):
        enc, _ = self.get_encoder_decoder_params()
        for p in enc:
            p.requires_grad = False

    def get_encoder_decoder_params(self):
        """Split parameters into encoder / decoder groups."""
        if hasattr(self.model, 'backbone'):
            enc = [p for p in self.model.backbone.parameters() if p.requires_grad]
            dec = [p for n, p in self.model.named_parameters() if 'backbone' not in n]
        elif hasattr(self.model, 'encoder'):
            enc = list(self.model.encoder.parameters())
            dec = [p for n, p in self.model.named_parameters() if not n.startswith('encoder')]
        else:
            raise ValueError("Model must have either 'backbone' or 'encoder' attribute to split parameters.")
        return enc, dec
    
    def postprocess(self, logits: torch.Tensor):
        """
        Convert model output tensor to predicted labels with vectorized per-class thresholding.
        
        Args:
            logits: (B, C, H, W) logits output from model
            
        Returns:
            preds: (B, H, W) predicted class indices with rejections applied
            max_probs: (B, H, W) confidence scores
        """
        probs = logits.softmax(dim=1)
        max_probs, preds = probs.max(dim=1) # (B, H, W)
        
        if not self.calibrating:
            # VECTORIZED LOOKUP: Map every pixel to its class-specific threshold
            pixel_thresholds = self.class_thresholds[preds] # (B, H, W)
            
            reject_mask = max_probs < pixel_thresholds
            preds[reject_mask] = self.ignore_index

        return preds, max_probs

