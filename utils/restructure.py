from typing import Union, List, Dict, Tuple, Optional
from pathlib import Path
import re
import numpy as np
import datetime as dt
from tqdm import tqdm
from Helper import (
    init_path_checks,
    extract_date_from_filename,
    num_to_date,
    global_logger,
    search_filedir,
    regex_search,
)
import shutil

naming_structure = {
    "animal": r"^DON-\d{6,8}$",
    "date": r"^\d{8}$",
    "photon_rec": r"^00[123]P-[IFT]$",
    "rec_output": r"^-[IFT]$",
}

behavior_naming_structure = {
    "wheel": r"^TRD-2P$",
}

cam_naming_structure = {
    "vr": "0000VR",
    "cam": "0000CM",
    "top": "0000BSM",
    "openfield": "TR-BSL",
}

rec_outputs = {
    "inscopix": "-I",
    "femtonics": "-F",
    "thorlabs": "-T",
}

photon_types = {
    "1p": "001P",
    "2p": "002P",
}

analysis_outputs = ["Bayesian_decoder", "Opexebo_cell_analysis", "models", "figures"]

neural_output_folder_names = [
    photon_type + rec_output
    for photon_type in photon_types.values()
    for rec_output in rec_outputs.values()
]
behavior_output_folder_names = [
    behavior_naming_structure[behavior] for behavior in behavior_naming_structure
]
cam_output_folder_names = [cam_naming_structure[cam] for cam in cam_naming_structure]
forbidden_names = (
    neural_output_folder_names + behavior_output_folder_names + cam_output_folder_names
)


def file_in_path(dir: Union[str, Path], fname: str = None, regex: str = None) -> Path:
    dir = Path(dir)
    if fname:
        path = dir / fname
    elif regex:
        path = next(dir.glob(regex), dir)
    else:
        raise ValueError("Either fname or regex must be provided.")
    return init_path_checks(path, check="file")


def move_file_to_folder(fname: Union[str, Path], folder: Union[str, Path]) -> None:
    fname = Path(fname)
    folder = Path(folder)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    new_path = folder / fname.name
    fname.rename(new_path)
    global_logger.info(f"Moved {fname} to {new_path}")


def move_task_into_date_folder(task_dir: Union[str, Path], date: str = None) -> None:
    task_dir = init_path_checks(task_dir, check="dir")
    # search for date inside file names
    file_names = [f.name for f in task_dir.iterdir() if f.is_file()]
    if date is None:
        dates = np.unique(
            [
                extract_date_from_filename(fname)
                for fname in file_names
                if extract_date_from_filename(fname)
            ]
        )
        if len(dates) == 0:
            raise ValueError(f"No date found in the task directory {task_dir}.")
        elif len(dates) > 1:
            raise ValueError(
                f"Multiple dates found in the task directory {task_dir}: {dates}."
            )
        else:
            date = dates[0]

    # create a date folder in parent directory and move task folder and files into it
    date_folder = task_dir.parent / date
    date_folder.mkdir(parents=True, exist_ok=True)
    shutil.move(str(task_dir), str(date_folder))


def restructure_task_dir(
    task_dir: Union[str, Path],
    task_data_locations: Dict[str, str],
) -> None:
    """Restructure the task directory to match the CEBRA format.

    Parameters
    ----------
        task_dir (Union[str, Path]): Path to the task directory.
        task_data_locations (Dict[str, str]): Dictionary containing the task data locations and their corresponding regex patterns.
    """
    task_dir = init_path_checks(task_dir, check="dir")
    # get the list of files in the task directory
    files = search_filedir(
        path=task_dir,
        #exclude_regex=forbidden_names,
        type="file",
    )
    if len(files) == 0:
        raise ValueError(f"No files found in the task directory {task_dir}.")

    task_name = task_dir.name
    date = task_dir.parent.name
    animal_id = task_dir.parent.parent.name
    
    # move neural recording files to the neural output folder
    for folder_name, category in task_data_locations.items():
        folder_path = task_dir / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        formated_category = category.format(animal_id=animal_id, 
                                            date=date, 
                                            behavior_setup=folder_name,
                                            task=task_name)
        files_to_move = regex_search(files, formated_category)
        for file in files_to_move:
            move_file_to_folder(file, folder_path)


def restructure_date_dir(
    date_dir: Union[str, Path],
    task_data_locations: Dict[str, str],
) -> None:
    """Restructure the date directory to match the CEBRA format.

    Parameters
    ----------
        date_dir (Union[str, Path]): Path to the date directory.
        task_data_locations (Dict[str, str]): Dictionary containing the task data locations and their corresponding regex patterns.
    """
    date_dir = init_path_checks(date_dir, check="dir")
    # get the list of task folders in the date directory
    folders = search_filedir(
        path=date_dir,
        exclude_regex=forbidden_names,
        type="dir",
    )
    for task_folder in folders:
        restructure_task_dir(
            task_dir=task_folder,
            task_data_locations=task_data_locations,
        )


def restructure_animal_dir(
    animal_dir: Union[str, Path],
    task_data_locations: Dict[str, str],
) -> None:
    """Restructure the animal directory to match the CEBRA format.

    Parameters
    ----------
        animal_dir (Union[str, Path]): Path to the animal directory.
        task_data_locations (Dict[str, str]): Dictionary containing the task data locations and their corresponding regex patterns.

    """
    animal_dir = init_path_checks(animal_dir, check="dir")

    # get the list of folders in the directory based on animal naming structure
    folders = search_filedir(
        path=animal_dir,
        include_regex=naming_structure["date"],
        exclude_regex=None,
        type="dir",
    )

    filtered_folders = []
    if len(folders) > 0:
        # checks if foldernames are real dates
        for date_folder in folders:
            date = num_to_date(date_folder.name)
            if date is None:
                global_logger.warning(
                    f"Folder {date_folder} in {animal_dir} is not a valid date. Skipping."
                )
            else:
                filtered_folders.append(date_folder)
        folders = filtered_folders

    # if no folders are found, check for task folders
    if len(folders) == 0:
        folders = search_filedir(path=animal_dir, type="dir")
        for task_folder in folders:
            move_task_into_date_folder(task_folder)
        folders = search_filedir(
            path=animal_dir,
            include_regex=naming_structure["date"],
            exclude_regex=None,
            type="dir",
        )

    if len(folders) == 0:
        raise ValueError(f"No date folders found in the animal directory {animal_dir}.")

    for date_folder in folders:
        restructure_date_dir(
            date_dir=date_folder,
            task_data_locations=task_data_locations,
        )


def restructure_animal_dirs(
    path: Union[str, Path],
    photon_type: str,
    rec_output: str,
    behavior_rec_type: str,
    neural_related: str,
    location_related: str,
) -> None:
    neural_rec_output_folder = photon_types[photon_type] + rec_outputs[rec_output]
    behavior_rec_output_folder = cam_naming_structure[behavior_rec_type]
    # get list of folders in the directory based on animal naming structure
    folders = search_filedir(
        path=path,
        include_regex=naming_structure["animal"],
        exclude_regex=None,
        type="dir",
    )
    task_data_locations = {
        neural_rec_output_folder: neural_related,
        behavior_rec_output_folder: location_related,
    }
    # iterate through each animal directory
    for animal_dir in tqdm(folders, desc="Restructuring animal directories"):
        restructure_animal_dir(animal_dir, task_data_locations)