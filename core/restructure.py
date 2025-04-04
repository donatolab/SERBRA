from typing import Union, List, Dict
from pathlib import Path

rec_output = {
    "inscopix": "001P-I",
    "femtonics": "002P-F",
    "thorlabs": "002P-T",
}

photon_type = {
    "1p": "001P",
    "2p": "002P",
}


def file_in_path(dir: Union[str, Path], fname: str = None, regex: str = None) -> Path:
    dir = Path(dir)
    if fname:
        path = dir / fname
    elif regex:
        path = next(dir.glob(regex), dir)
    else:
        raise ValueError("Either fname or regex must be provided.")


def move_file_to_folder(fname: Union[str, Path], folder: Union[str, Path]) -> None:
    fname = Path(fname)
    folder = Path(folder)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    new_path = folder / fname.name
    fname.rename(new_path)
    print(f"Moved {fname} to {new_path}")


def restruct_nathalie_animals_dir() -> None:
    # This function restructures the directory containing animal data files.
    # It moves the files to a new folder based on the provided folder name.
    # The function is not fully implemented in the original code.
    upphase_source = "binarized_traces_V3_curated.npz"
    move_file_to_folder()
    npz_path = next(path.glob("*.npz"), path)
    raise NotImplementedError("This function is not fully implemented.")
