from __future__ import annotations
import segmentation_models_pytorch as smp
from geosave_engine.core.base import BaseModel


class Unet(BaseModel):
    """U-Net model wrapper based on segmentation_models_pytorch."""
    tasks = {
        "semantic segmentation": ["supervised"],
        "pixelwise regression": []
        }
    doc_links = ["https://docs.geosave.dev/models/unet"]
    model = smp.Unet

class UnetPlusPlus(BaseModel):
    """U-Net++ model wrapper based on segmentation_models_pytorch."""
    tasks = {
        "semantic segmentation": [],
        "pixelwise regression": []
        }
    doc_links = ["https://docs.geosave.dev/models/unetplusplus"]
    model = smp.UnetPlusPlus

class DeepLabV3(BaseModel):
    """DeepLabV3 model wrapper based on segmentation_models_pytorch."""
    tasks = {
        "semantic segmentation": []
        }
    doc_links = ["https://docs.geosave.dev/models/deeplabv3"]
    model = smp.DeepLabV3

class DeepLabV3Plus(BaseModel):
    """DeepLabV3+ model wrapper based on segmentation_models_pytorch."""
    tasks = {
        "semantic segmentation": []
        }
    doc_links = ["https://docs.geosave.dev/models/deeplabv3plus"]
    model = smp.DeepLabV3Plus

class FPN(BaseModel):
    """FPN model wrapper based on segmentation_models_pytorch."""
    tasks = {
        "semantic segmentation": []
        }
    doc_links = ["https://docs.geosave.dev/models/fpn"]
    model = smp.FPN

class PSPNet(BaseModel):
    """PSPNet model wrapper based on segmentation_models_pytorch."""
    tasks = {
        "semantic segmentation": []
        }
    doc_links = ["https://docs.geosave.dev/models/pspnet"]
    model = smp.PSPNet

class PAN(BaseModel):
    """PAN model wrapper based on segmentation_models_pytorch."""
    tasks = {
        "semantic segmentation": []
        }
    doc_links = ["https://docs.geosave.dev/models/pan"]
    model = smp.PAN

class Linknet(BaseModel):
    """LinkNet model wrapper based on segmentation_models_pytorch."""
    tasks = {
        "semantic segmentation": []
        }
    doc_links = ["https://docs.geosave.dev/models/linknet"]
    model = smp.Linknet

class MAnet(BaseModel):
    """MA-Net model wrapper based on segmentation_models_pytorch."""
    tasks = {
        "semantic segmentation": []
        }
    doc_links = ["https://docs.geosave.dev/models/manet"]
    model = smp.MAnet
