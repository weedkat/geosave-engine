# Use cache directory for pretrained weights
# Works for both dev and pip-installed versions
from pathlib import Path
import os
import torch
from .semseg.dpt import DPT
from geomoka._core.registry import Registry

PRETRAINED_DIR = Path.home() / '.cache' / 'geomoka' / 'pretrained'

# Map model names to smp classes
dpt_encoder_map = {
    'dinov2_small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'dinov2_base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'dinov2_large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'dinov2_giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

def download_pretrained_dinov2(encoder, weights, pretrain_dir=PRETRAINED_DIR):
    dpt_urls ={
    'imagenet':{
        'dinov2_small': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
        'dinov2_base': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
        'dinov2_large': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
        'dinov2_giant': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth'
        }
    }

    extract_dir = pretrain_dir
    os.makedirs(extract_dir, exist_ok=True)

    if weights not in dpt_urls:
        raise ValueError(f"{weights} not recognized. Valid options are: {list(dpt_urls.keys())}")

    urls = dpt_urls[weights]

    if encoder not in urls:
        raise ValueError(f"{encoder} not recognized. Valid options are: {list(urls.keys())}")

    url = urls[encoder]
    pth_path = os.path.join(extract_dir, f"{encoder}.pth")

    if not os.path.exists(pth_path):
        print(f"Downloading pretrained DINOv2 {encoder} weights...")
        torch.hub.download_url_to_file(url, pth_path)
        print("Download complete.")
    else:
        print(f"Pretrained DINOv2 {encoder} weights already exists at {pth_path}. Skipping download.")
    
    return pth_path

@Registry.register_model('dpt')
def build(in_channels, nclass, encoder='dinov2_base', weights='imagenet', pretrain_dir=PRETRAINED_DIR, **kwargs):
    model_cfg = dpt_encoder_map[encoder]
    
    model = DPT(**{**model_cfg, 'nclass': nclass, 'in_chans': in_channels, **kwargs})
    
    if weights == 'imagenet':
        pth_path = download_pretrained_dinov2(encoder, weights, pretrain_dir)
        state_dict = torch.load(pth_path, map_location='cpu')
        model.backbone.load_state_dict(state_dict)
        print(f'Pretrained DINOv2 weights loaded from {pth_path}')
    else:
        print(f'No pretrained weights, training from scratch')
    
    return model