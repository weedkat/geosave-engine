from __future__ import annotations
import segmentation_models_pytorch as smp
from geosave_engine.core.factory import BaseModelFactory


class Unet(BaseModelFactory):
    """U-Net model wrapper based on segmentation_models_pytorch."""
    tasks = {
        "semantic segmentation": ["supervised"],
        "pixelwise regression": []
        }
    task = tasks
    doc_links = ["https://docs.geosave.dev/models/unet"]
    model = smp.Unet

class UnetPlusPlus(BaseModelFactory):
    """U-Net++ model wrapper based on segmentation_models_pytorch."""
    tasks = {
        "semantic segmentation": [],
        "pixelwise regression": []
        }
    task = tasks
    doc_links = ["https://docs.geosave.dev/models/unetplusplus"]
    model = smp.UnetPlusPlus

class DeepLabV3(BaseModelFactory):
    """DeepLabV3 model wrapper based on segmentation_models_pytorch."""
    tasks = {
        "semantic segmentation": []
        }
    task = tasks
    doc_links = ["https://docs.geosave.dev/models/deeplabv3"]
    model = smp.DeepLabV3

class DeepLabV3Plus(BaseModelFactory):
    """DeepLabV3+ model wrapper based on segmentation_models_pytorch."""
    tasks = {
        "semantic segmentation": []
        }
    task = tasks
    doc_links = ["https://docs.geosave.dev/models/deeplabv3plus"]
    model = smp.DeepLabV3Plus

class FPN(BaseModelFactory):
    """FPN model wrapper based on segmentation_models_pytorch."""
    tasks = {
        "semantic segmentation": []
        }
    task = tasks
    doc_links = ["https://docs.geosave.dev/models/fpn"]
    model = smp.FPN

class PSPNet(BaseModelFactory):
    """PSPNet model wrapper based on segmentation_models_pytorch."""
    tasks = {
        "semantic segmentation": []
        }
    task = tasks
    doc_links = ["https://docs.geosave.dev/models/pspnet"]
    model = smp.PSPNet

class PAN(BaseModelFactory):
    """PAN model wrapper based on segmentation_models_pytorch."""
    tasks = {
        "semantic segmentation": []
        }
    task = tasks
    doc_links = ["https://docs.geosave.dev/models/pan"]
    model = smp.PAN

class Linknet(BaseModelFactory):
    """LinkNet model wrapper based on segmentation_models_pytorch."""
    tasks = {
        "semantic segmentation": []
        }
    task = tasks
    doc_links = ["https://docs.geosave.dev/models/linknet"]
    model = smp.Linknet

class MAnet(BaseModelFactory):
    """MA-Net model wrapper based on segmentation_models_pytorch."""
    tasks = {
        "semantic segmentation": []
        }
    task = tasks
    doc_links = ["https://docs.geosave.dev/models/manet"]
    model = smp.MAnet