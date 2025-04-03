from typing import Union, List, Dict
from pathlib import Path

def handle_binarized_npz(root_path: Union[str, Path]):
    path = Path(root_path)
    npz_path = next(path.glob("*.npz"), path)
    .....