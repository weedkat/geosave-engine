import random
from statistics import mean

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms
import albumentations as A


class TransformsCompose:
    """
    Compose a list of transformations specified in the config.
    """
    def __init__(self, cfg=None, input_size=None):
        
        if not isinstance(cfg, list):
            raise ValueError("Expected a list of transform specifications")
        
        self.input_size = input_size

        transforms = [self.build_transforms(spec) for spec in cfg]
        
        self.transform = A.Compose(transforms)
     
    def __call__(self, **kwargs):
        return self.transform(**kwargs)

    def build_transforms(self, spec):
        name = spec['name']
        args = spec.get('kwargs', {}).copy()
        cls = getattr(A, name)
        
        if name in ("OneOf", "SomeOf", "Compose"):
            nested_spec = args.pop('transforms', [])
            transforms = [self.build_transforms(t) for t in nested_spec]
            return cls(transforms, **args)

        if name in ("RandomResizedCrop", "RandomCrop", "CenterCrop", "Resize"):
            if self.input_size is not None:
                args['size'] = [self.input_size, self.input_size]
            else:
                raise ValueError(f"Input size must be specified for {name} transform")
            
        return cls(**args)

    def __add__(self, other):
        if not isinstance(other, TransformsCompose):
            raise ValueError("Can only add another TransformsCompose instance")
        
        new = TransformsCompose()
        new.transform = A.Compose(self.transform.transforms + other.transform.transforms)
        return new
    

def crop(img, mask, size, ignore_value=255):
    """
    Randomly crop the image and mask to the given size. Add padding if size is larger than image size.

    Args:
        img: PIL Image
        mask: PIL Image
        size (int): crop size
        ignore_value (int): value to fill for padded region in mask

    Returns:
        cropped img and mask
    """
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    # Ensure we can actually crop - if w == size, randint(0, 0) is valid
    x = random.randint(0, max(0, w - size))
    y = random.randint(0, max(0, h - size))
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


def hflip(img, mask, p=0.5):
    """
    Randomly horizontally flip the image and mask with probability p.
    
    Args:
        img: PIL Image
        mask: PIL Image
        p (float): probability of flipping
    
    Returns:
        flipped img and mask
    """
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    """
    Normalize the image using ImageNet mean and std. Convert image and mask to tensor.
    
    Args:
        img: PIL Image
        mask: PIL Image or None
    
    Returns:
        normalized img and (mask if provided)
    """
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def resize(img, mask, ratio_range):
    """
    Resize the image and mask by a random scale factor within the given ratio range.
    
    Args:
        img: PIL Image
        mask: PIL Image
        ratio_range (tuple): tuple of (min_ratio, max_ratio) for scaling
    
    Returns:
        resized img and mask
    """
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    """
    Apply Gaussian blur to the image with probability p.
    
    Args:
        img: PIL Image
        p (float): probability of applying blur
    
    Returns:     
        blurred img
    """ 
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    """
    Obtain a cutmix box mask for the given image size.
    
    Args:
        img_size (int): size of the image (assumed square)
        p (float): probability of applying cutmix
        size_min (float): minimum area ratio of the cutmix box
        size_max (float): maximum area ratio of the cutmix box
        ratio_1 (float): minimum aspect ratio of the cutmix box
        ratio_2 (float): maximum aspect ratio of the cutmix box
    
    Returns:
        cutmix box mask (2D tensor)
    """
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask