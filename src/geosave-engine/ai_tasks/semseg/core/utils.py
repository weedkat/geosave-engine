from pathlib import Path
import re

def extract_id(filename):
    stem = Path(filename).stem
    match = re.findall(r'\d+', stem)
    
    if match:
        return match[-1]  # Return last numeric sequence
    
    return stem  # Fallback to full stem if no numbers found
