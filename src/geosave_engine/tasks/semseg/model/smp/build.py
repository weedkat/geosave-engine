import segmentation_models_pytorch as smp
from geosave_engine._core.registry import Registry

registry = Registry()

@registry.register('unet')
def unet(in_channels, nclass, **kwargs):
    return smp.Unet(in_channels=in_channels, classes=nclass, **kwargs)

@registry.register('unet++')
def unetplusplus(in_channels, nclass, **kwargs):
    return smp.UnetPlusPlus(in_channels=in_channels, classes=nclass, **kwargs)

@registry.register('deeplabv3')
def deeplabv3(in_channels, nclass, **kwargs):
    return smp.DeepLabV3(in_channels=in_channels, classes=nclass, **kwargs)

@registry.register('deeplabv3+')
def deeplabv3plus(in_channels, nclass, **kwargs):
    return smp.DeepLabV3Plus(in_channels=in_channels, classes=nclass, **kwargs)

@registry.register('fpn')
def fpn(in_channels, nclass, **kwargs):
    return smp.FPN(in_channels=in_channels, classes=nclass, **kwargs)

@registry.register('pspnet')
def pspnet(in_channels, nclass, **kwargs):
    return smp.PSPNet(in_channels=in_channels, classes=nclass, **kwargs)

@registry.register('pan')
def pan(in_channels, nclass, **kwargs):
    return smp.PAN(in_channels=in_channels, classes=nclass, **kwargs)

@registry.register('linknet')
def linknet(in_channels, nclass, **kwargs):
    return smp.Linknet(in_channels=in_channels, classes=nclass, **kwargs)

@registry.register('manet')
def manet(in_channels, nclass, **kwargs):
    return smp.MAnet(in_channels=in_channels, classes=nclass, **kwargs)