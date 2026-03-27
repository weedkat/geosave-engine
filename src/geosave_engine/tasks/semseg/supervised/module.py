import lightning as L
import torch

from ..common.inference import infer_sliding_window
from torch.optim.lr_scheduler import CosineAnnealingLR

from .registry import optim_registry, loss_registry, model_registry
from ..common.metric import get_metrics
from ..common.utils import extract_prefixed

from ..common.metadata import MetadataInterpreter

class SemSegModel(L.LightningModule):
    def __init__(self, arch_cfg, optim_cfg, loss_cfg, metadata_dict, transform_dict, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.arch = arch_cfg['name']
        self.optim = optim_cfg['name']
        self.loss = loss_cfg['name']

        self.transform_dict = transform_dict
        self.metadata_dict = metadata_dict
        self.metadata_interpreter = MetadataInterpreter(metadata_dict)

        self.ignore_index = self.metadata_interpreter.ignore_index
        self.input_size = self.metadata_interpreter.input_size
        self.nclass = self.metadata_interpreter.nclass
        self.in_channels = self.metadata_interpreter.in_channels
        self.class_names = self.metadata_interpreter.class_names

        self.arch_cfg = arch_cfg
        self.optim_cfg = optim_cfg
        self.loss_cfg = loss_cfg
        
        self.arch_cfg['nclass'] = self.nclass
        self.arch_cfg['in_channels'] = self.in_channels
        self.model = model_registry.build(self.arch, **self.arch_cfg)
        
        self.loss_cfg['ignore_index'] = self.ignore_index
        self.loss_fn = loss_registry.build(self.loss, **self.loss_cfg)

        metrics = get_metrics(
            num_classes=self.nclass, 
            class_names=self.class_names, 
            ignore_index=self.ignore_index
        )
        test_metrics = get_metrics(
            num_classes=self.nclass, 
            class_names=self.class_names, 
            ignore_index=None
        ) 

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = test_metrics.clone(prefix="test/")

        self.calibrating = False
        self.register_buffer("class_thresholds", torch.zeros(self.nclass), persistent=True)
        
        # For logging validation images
        self.val_sample_img = None
        self.val_sample_pred = None
        self.val_sample_label = None

    def forward(self, x):
        return infer_sliding_window(self, x)

    def configure_optimizers(self):
        optim_cfg = self.optim_cfg.copy()
        
        enc, dec = self.get_encoder_decoder_params()
        encoder_lr = optim_cfg.pop('lr')
        decoder_lr = encoder_lr * optim_cfg.pop('lr_multi')

        parameter = [ 
            {'params': enc, 'lr': encoder_lr},
            {'params': dec, 'lr': decoder_lr}
        ]
        
        optimizer = optim_registry.build(self.optim, parameter, **optim_cfg)
        
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # 'epoch' or 'step'
                "frequency": 1,       # How often to step the scheduler
                # "monitor": "val_loss", # ONLY required if using ReduceLROnPlateau
            },
        }

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

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, labels = batch
        logits = infer_sliding_window(self, imgs)
        loss = self.loss_fn(logits, labels)
        
        self.val_metrics.update(logits, labels)

        # Log first image from first batch of first dataloader
        if batch_idx == 0 and dataloader_idx == 0:
            self.val_sample_img = imgs[0].detach().cpu()
            self.val_sample_pred = logits[0].detach().cpu().argmax(dim=0)
            self.val_sample_label = labels[0].detach().cpu()

        self.log("val_loss", loss)
    
    def on_validation_epoch_end(self):
        output = self.val_metrics.compute()
        self.log_dict(output, prog_bar=True)
        self.val_metrics.reset()
        
        # Log validation image
        if self.val_sample_img is not None:
            pred_rgb = torch.tensor(self.metadata_interpreter.class_to_rgb(self.val_sample_pred.numpy()), dtype=torch.uint8).permute(2, 0, 1)
            label_rgb = torch.tensor(self.metadata_interpreter.class_to_rgb(self.val_sample_label.numpy()), dtype=torch.uint8).permute(2, 0, 1)
            
            img_vis = self.val_sample_img / 255.0 if self.val_sample_img.max() > 1 else self.val_sample_img
            
            self.logger.log_image("val/sample", [img_vis, pred_rgb / 255.0, label_rgb / 255.0])

    # ================== Testing ==================

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = infer_sliding_window(self, imgs)

        preds, max_probs = self.postprocess(logits)
        self.test_metrics.update(preds, labels)

        return preds, max_probs, labels

    def on_test_epoch_end(self):
        output = self.test_metrics.compute()
        self.log_dict(output, prog_bar=True)
        self.test_metrics.reset()

    # ================== Prediction ==================

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, img_ids, meta_profiles = batch
        logits = infer_sliding_window(self, imgs)
        preds, max_probs = self.postprocess(logits)
        
        return preds, max_probs, img_ids, meta_profiles

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

