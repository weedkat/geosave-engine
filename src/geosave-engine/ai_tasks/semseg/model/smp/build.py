import segmentation_models_pytorch as smp
from .dpt.build import build as build_dpt
from geomoka._core.registry import Registry

@Registry.register_model('unet')
def unet(in_channels, nclass, **kwargs):
    return smp.Unet(in_channels=in_channels, classes=nclass, **kwargs)

@Registry.register_model('unet++')
def unetplusplus(in_channels, nclass, **kwargs):
    return smp.UnetPlusPlus(in_channels=in_channels, classes=nclass, **kwargs)

@Registry.register_model('deeplabv3')
def deeplabv3(in_channels, nclass, **kwargs):
    return smp.DeepLabV3(in_channels=in_channels, classes=nclass, **kwargs)

@Registry.register_model('deeplabv3+')
def deeplabv3plus(in_channels, nclass, **kwargs):
    return smp.DeepLabV3Plus(in_channels=in_channels, classes=nclass, **kwargs)

@Registry.register_model('fpn')
def fpn(in_channels, nclass, **kwargs):
    return smp.FPN(in_channels=in_channels, classes=nclass, **kwargs)

@Registry.register_model('pspnet')
def pspnet(in_channels, nclass, **kwargs):
    return smp.PSPNet(in_channels=in_channels, classes=nclass, **kwargs)

@Registry.register_model('pan')
def pan(in_channels, nclass, **kwargs):
    return smp.PAN(in_channels=in_channels, classes=nclass, **kwargs)

@Registry.register_model('linknet')
def linknet(in_channels, nclass, **kwargs):
    return smp.Linknet(in_channels=in_channels, classes=nclass, **kwargs)

@Registry.register_model('manet')
def manet(in_channels, nclass, **kwargs):
    return smp.MAnet(in_channels=in_channels, classes=nclass, **kwargs)