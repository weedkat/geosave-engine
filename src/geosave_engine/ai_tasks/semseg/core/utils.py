from pathlib import Path
import re
import torchmetrics

def extract_id(filename):
    stem = Path(filename).stem
    match = re.findall(r'\d+', stem)
    
    if match:
        return match[-1]  # Return last numeric sequence
    
    return stem  # Fallback to full stem if no numbers found

def extract_prefixed(kwargs: dict, prefix: str) -> dict:
    """Extracts and removes keys from kwargs that start with the given prefix.
    Example:
    kwargs = {
        'loader_batch_size': 32,
        'loader_num_workers': 4,
        'model_lr': 0.001
    }
    prefix = 'loader'
    Returns:
    {
        'batch_size': 32,
        'num_workers': 4
    }
    And kwargs will be modified to:
    {
        'model_lr': 0.001
    }
    """
    key_prefix = f"{prefix}_"
    extracted = {
        key[len(key_prefix):]: value
        for key, value in kwargs.items()
        if key.startswith(key_prefix)
    }
    for key in list(extracted.keys()):
        kwargs.pop(f"{prefix}_{key}", None)
        
    return extracted
