import lightning as L
import torch

from geosave_engine.ai_tasks.semseg.core.inference import Inference

from .registry import optim_registry, loss_registry, model_registry
from ..core.metric import get_metrics
from ..core.utils import extract_prefixed

from .metadata import MetadataInterpreter


class SemSegModel(L.LightningModule):
    def __init__(self, arch, optim, loss, metadata_dict, transform_dict, calibrate=False, thresholds=None, **kwargs):
        metadata_interpreter = MetadataInterpreter(metadata_dict)
        if thresholds is None:
            """
            {'class_name': threshold_value, ...}
            """
            thresholds = {key: 0.0 for key in metadata_interpreter.class_names}

        super().__init__()
        self.save_hyperparameters()

        self.arch = arch
        self.optim = optim
        self.loss = loss
        self.metadata_interpreter = metadata_interpreter
        self.transform_dict = transform_dict
        self.calibrate = calibrate
        self.metadata_dict = metadata_dict
        self.thresholds = thresholds

        for attr in ['arch', 'optim', 'loss']:
            # arch_cfg, optim_cfg, loss_cfg will be extracted from kwargs if available
            setattr(self, f"{attr}_cfg", extract_prefixed(kwargs, attr))
        
        self.arch_cfg['nclass'] = self.metadata_interpreter.nclass
        self.arch_cfg['in_channels'] = self.metadata_interpreter.in_channels
        self.model = model_registry.build(arch, **self.arch_cfg)
        
        self.loss_cfg['ignore_index'] = self.metadata_interpreter.ignore_index
        self.loss_fn = loss_registry.build(loss, **self.loss_cfg)

        metrics = get_metrics(num_classes=self.metadata_interpreter.nclass, 
                              class_names=self.metadata_interpreter.class_names, 
                              ignore_index=self.metadata_interpreter.ignore_index)
        
        # Create separate instances for each stage
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")


    def setup(self, stage=None):
        self.transform_trn = self.transform_dict.get("train")
        self.transform_infer = self.transform_dict.get("infer")
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
        logits = self.inferencer(imgs, logits=True)
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
        probs, preds = self.inferencer(imgs, probs=True)

        if self.calibrate:
            max_conf = probs[preds]

            self.cal_maxconf.append(max_conf.detach().cpu())
            self.cal_preds.append(preds.detach().cpu())
            self.cal_labels.append(labels.detach().cpu())

        preds = (probs > self.threshold).long()
        self.test_metrics.update(preds, labels)

    def on_test_epoch_end(self):
        if self.calibrate and self.cal_preds:
            all_probs = torch.cat(self.cal_preds)
            all_labels = torch.cat(self.cal_labels)

            new_val = self.find_best_threshold(all_probs, all_labels)
            
            self.threshold = torch.tensor(new_val, device=self.device)
            
            self.cal_preds.clear()
            self.cal_labels.clear()
            
            print(f"Calibration complete. New threshold: {self.threshold:.4f}")
            self.log("calibrated_threshold", self.threshold, on_epoch=True, prog_bar=True)

        output = self.test_metrics.compute()
        self.log_dict(output, prog_bar=True)
        self.test_metrics.reset()

    # ================== Prediction ==================

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        imgs = batch
        preds = self.inferencer(imgs, logits=False)
        
        return preds
    
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
    
    def find_best_threshold(self, probs, labels):
        best_threshold = 0.0
        best_metric = -float('inf')

        for class_idx, _ in self.thresholds.items():
            for threshold in torch.linspace(0, 1, steps=100):
                preds = (probs > threshold).long()
                metric = self.test_metrics.compute(preds, labels)['test/iou']
                if metric > best_metric:
                    best_metric = metric
                    best_threshold = threshold.item()

        return best_threshold

