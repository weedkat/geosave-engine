from __future__ import annotations
import segmentation_models_pytorch as smp


def unet(num_classes, in_channels, **kwargs) -> smp.Unet:
    return smp.Unet(classes=num_classes, in_channels=in_channels, **kwargs)

def unetplusplus(num_classes, in_channels, **kwargs) -> smp.UnetPlusPlus:
    return smp.UnetPlusPlus(classes=num_classes, in_channels=in_channels, **kwargs)

def deeplabv3(num_classes, in_channels, **kwargs) -> smp.DeepLabV3:
    return smp.DeepLabV3(classes=num_classes, in_channels=in_channels, **kwargs)

def deeplabv3plus(num_classes, in_channels, **kwargs) -> smp.DeepLabV3Plus:
    return smp.DeepLabV3Plus(classes=num_classes, in_channels=in_channels, **kwargs)

def segformer(num_classes, in_channels, **kwargs) -> smp.Segformer:
    return smp.Segformer(classes=num_classes, in_channels=in_channels, **kwargs)

def dpt(num_classes, in_channels, **kwargs) -> smp.DPT:
    return smp.DPT(classes=num_classes, in_channels=in_channels, **kwargs)

def fpn(num_classes, in_channels, **kwargs) -> smp.FPN:
    return smp.FPN(classes=num_classes, in_channels=in_channels, **kwargs)

def linknet(num_classes, in_channels, **kwargs) -> smp.Linknet:
    return smp.Linknet(classes=num_classes, in_channels=in_channels, **kwargs)

def manet(num_classes, in_channels, **kwargs) -> smp.MAnet:
    return smp.MAnet(classes=num_classes, in_channels=in_channels, **kwargs)

def pan(num_classes, in_channels, **kwargs) -> smp.PAN:
    return smp.PAN(classes=num_classes, in_channels=in_channels, **kwargs)

def pspnet(num_classes, in_channels, **kwargs) -> smp.PSPNet:
    return smp.PSPNet(classes=num_classes, in_channels=in_channels, **kwargs)

def upernet(num_classes, in_channels, **kwargs) -> smp.UPerNet:
    return smp.UPerNet(classes=num_classes, in_channels=in_channels, **kwargs)


