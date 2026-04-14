import segmentation_models_pytorch as smp
from geosave_engine.core.model import BaseGeosaveModel


class Unet(BaseGeosaveModel):
    task = {
        "semantic segmentation": ["supervised"],
        "pixelwise regression": []
        }
    model = smp.Unet

class UnetPlusPlus(BaseGeosaveModel):
    task = {
        "semantic segmentation": [],
        "pixelwise regression": []
        }
    model = smp.UnetPlusPlus

class DeepLabV3(BaseGeosaveModel):
    task = {
        "semantic segmentation": []
        }
    model = smp.DeepLabV3

class DeepLabV3Plus(BaseGeosaveModel):
    task = {
        "semantic segmentation": []
        }
    model = smp.DeepLabV3Plus

class FPN(BaseGeosaveModel):
    task = {
        "semantic segmentation": []
        }
    model = smp.FPN

class PSPNet(BaseGeosaveModel):
    task = {
        "semantic segmentation": []
        }
    model = smp.PSPNet

class PAN(BaseGeosaveModel):
    task = {
        "semantic segmentation": []
        }
    model = smp.PAN

class Linknet(BaseGeosaveModel):
    task = {
        "semantic segmentation": []
        }
    model = smp.Linknet

class MAnet(BaseGeosaveModel):
    task = {
        "semantic segmentation": []
        }
    model = smp.MAnet