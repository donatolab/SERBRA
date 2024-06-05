import os
from pathlib import Path


# type hints
from typing import List, Union, Dict, Any, Tuple, Optional


# calculations
import numpy as np
from scipy.signal import butter, filtfilt  # , lfilter, freqz
import sklearn

# import cupy as cp  # numpy using CUDA for faster computation
import yaml
import re

from datetime import datetime

# debugging
import logging
from time import time


class GlobalLogger:
    def __init__(self, save_dir=""):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.configure_logger(save_dir=save_dir)

    def configure_logger(self, save_dir=""):
        self.logger.setLevel(logging.DEBUG)  # Set the desired level here

        # Create a file handler which logs even debug messages.
        log_file_path = os.path.join(save_dir, "Global_log.log")
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.DEBUG)

        # Create a console handler with a higher log level.
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        # Create a formatter and add it to the handlers.
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to the logger.
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def set_save_dir(self, save_dir):
        # Get the old handler
        old_handler = self.logger.handlers[0]

        # Create a new handler with the updated filename
        new_handler = logging.FileHandler(save_dir + "/Global_log.log")
        new_handler.setLevel(old_handler.level)
        new_handler.setFormatter(old_handler.formatter)

        # Remove the old handler and add the new one
        for old_handler in self.logger.handlers:
            self.logger.removeHandler(old_handler)
        self.logger.addHandler(new_handler)


global_logger_object = GlobalLogger()
global_logger = global_logger_object.logger

# global_logger_object.set_save_dir(os.path.join(os.getcwd(), "logs"))
# global_logger = global_logger_object.logger
# global_logger.info('check if log file is created in new working directory')
# global_logger
# print(global_logger_object.logger.handlers)
# global_logger_object.logger.handlers[0].baseFilename


def yield_animal(animals_dict):
    for animal_id, animal in animals_dict.items():
        yield animal


def yield_animal_session(animals_dict):
    for animal in yield_animal(animals_dict):
        for session_date, session in animal.sessions.items():
            yield animal, session


def yield_animal_session_task(animals_dict):
    for animal, session in yield_animal_session(animals_dict):
        for task_id, task in session.tasks.items():
            yield animal, session, task


#### directory, file search
def dir_exist_create(directory):
    """
    Checks if a directory exists and creates it if it doesn't.

    Parameters:
    dir (str): Path of the directory to check and create.

    Returns:
    None
    """
    # Check if the directory exists
    directory = Path(directory)
    if not directory.exists():
        # Create the directory
        directory.mkdir()


def create_dirs(dirs):
    """
    Create a new directory hierarchy.

    Args:
        dirs (list): A list of strings representing the path to the new directory.

    Returns:
        str: The path to the newly created directory.
    """
    new_path = dirs[0]
    for path_part in dirs[1:]:
        new_path = os.path.join(new_path, path_part)
        dir_exist_create(new_path)
    return new_path


def get_directories(directory, regex_search=""):
    """
    This function returns a list of directories from the specified directory that match the regular expression search pattern.

    Parameters:
    directory (str): The directory path where to look for directories.
    regex_search (str, optional): The regular expression pattern to match. Default is an empty string, which means all directories are included.

    Returns:
    list: A list of directory names that match the regular expression search pattern.
    """
    directory = Path(directory)
    directories = None
    if directory.exists():
        directories = [
            name
            for name in os.listdir(directory)
            if directory.joinpath(name).is_dir()
            and len(re.findall(regex_search, name)) > 0
        ]
    else:
        global_logger.warning(f"Directory does not exist: {directory}")
        print(f"Directory does not exist: {directory}")
    return directories


def get_files(directory, ending="", regex_search=""):
    """
    This function returns a list of files from the specified directory that match the regular expression search pattern and have the specified file ending.

    Parameters:
    directory (str): The directory path where to look for files.
    ending (str, optional): The file ending to match. Default is '', which means all file endings are included.
    regex_search (str, optional): The regular expression pattern to match. Default is an empty string, which means all files are included.

    Returns:
    list: A list of file names that match the regular expression search pattern and have the specified file ending.
    """
    directory = Path(directory)
    files_list = None
    if directory.exists():
        files_list = []
        for name in os.listdir(directory):
            if directory.joinpath(name).is_file():
                if len(re.findall(regex_search, name)) > 0 and name.endswith(ending):
                    files_list.append(name)
    else:
        global_logger.warning(f"Directory does not exist: {directory}")
        print(f"Directory does not exist: {directory}")
    return files_list


def search_file(directory, filename):
    """
    This function searches for a file with a given filename within a specified directory and its subdirectories.

    :param directory: The directory in which to search for the file.
    :type directory: str
    :param filename: The name of the file to search for.
    :type filename: str
    :return: The full path of the file if found, otherwise returns the string "Not found".
    :rtype: str
    """
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return Path(root).joinpath(filename)
    return None


def make_list_ifnot(string_or_list):
    if isinstance(string_or_list, np.ndarray):
        return string_or_list.tolist()
    return [string_or_list] if type(string_or_list) != list else string_or_list


def save_file_present(file_path, show_print=False):
    file_path = Path(file_path)
    file_present = False
    if file_path.exists():
        if show_print:
            global_logger.warning(f"File already present {file_path}")
            print(f"File already present {file_path}")
        file_present = True
    else:
        if show_print:
            global_logger.info(f"Saving {file_path.name} to {file_path}")
            print(f"Saving {file_path.name} to {file_path}")
    return file_present


# Math
def calc_cumsum_distances(positions, length, distance_threshold=0.30):
    """
    Calculates the cumulative sum of distances between positions along a track.

    Args:
    - positions (array-like): List or array of positions along the track.
    - length (numeric): Length of the track.
    - distance_threshold (numeric, optional): Threshold for considering a frame's distance change.
                                            Defaults to 30.

    Returns:
    - cumsum_distance (numpy.ndarray): Cumulative sum of distances between positions.
    """
    cumsum_distances = []
    cumsum_distance = 0
    old_position = positions[0]
    for position in positions[1:]:
        frame_distance = position - old_position
        if abs(frame_distance) < distance_threshold:
            cumsum_distance += frame_distance
        else:
            # check if mouse moves from end to start of the track
            if frame_distance < 0:
                cumsum_distance += frame_distance + length
            # check if mouse moves from start to end of the track
            else:
                cumsum_distance += frame_distance - length
        cumsum_distances.append(cumsum_distance)
        old_position = position
    return np.array(cumsum_distances)


def calculate_derivative(data):
    """
    Calculates the derivative of the given data.

    Parameters:
    - data (array-like): The input data (position or velocity) for which the derivative needs to be computed.

    Returns:
    - derivative (numpy.ndarray): The calculated derivative, representing either velocity based on position or acceleration based on velocity.
    """
    timestamps = np.array(range(len(data)))
    # Calculate differences in data
    diff = np.diff(data)
    # Calculate differences in time
    time_diff = np.diff(timestamps)

    derivative = diff / time_diff

    return derivative


def moving_average(data, window_size=30):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, "valid")


def smooth_array(data, window_size=5, axis=0):
    """
    Smooth a NumPy array along the specified axis using convolution.

    Args:
        data (np.ndarray): The input 2D array.
        window_size (int, optional): Size of the smoothing window (default is 5).
        axis (int, optional): Axis along which to apply smoothing (default is 0).

    Returns:
        np.ndarray: The smoothed array.
    """
    weights = np.ones(window_size) / window_size
    smoothed_data = np.apply_along_axis(
        lambda x: np.convolve(x, weights, "same"), axis, data
    )
    return smoothed_data


def continuouse_to_discrete(continuouse_array, lengths: list):
    """
    Converts continuouse data into discrete values based on track lengths using NumPy.

    Parameters:
    - continuouse_array: array of continuouse data.
    - lengths:  containing lengths of track parts.

    Returns:
    - discrete_values: NumPy array of discrete values corresponding to continuouse data.
    """
    # Calculate cumulative track lengths
    cumulative_lengths = np.cumsum(lengths)

    # Broadcast and compare continuouses against track boundaries
    discrete_values = np.sum(
        continuouse_array[:, np.newaxis] >= cumulative_lengths, axis=1
    )

    return discrete_values


def butter_lowpass(cutoff, fs, order=5):
    """
    Design a lowpass Butterworth filter.

    Parameters:
        cutoff (float): The cutoff frequency in Hertz.
        fs (float): The sampling frequency in Hertz.
        order (int, optional): The filter order. Defaults to 5.

    Returns:
        b (array-like): Numerator (zeros) coefficients of the filter.
        a (array-like): Denominator (poles) coefficients of the filter.
    """

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Apply a lowpass Butterworth filter to the input data.

    Parameters:
        data (array-like): The input data to filter.
        cutoff (float): The cutoff frequency in Hertz.
        fs (float): The sampling frequency in Hertz.
        order (int, optional): The filter order. Defaults to 5.

    Returns:
        y (array-like): The filtered output data.
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def cosine_similarity(v1, v2):
    """
    A cosine similarity can be seen as the correlation between two vectors.
    Returns:
        float: The cosine similarity between the two vectors.
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def correlate_vectors(vectors: np.ndarray):
    """
    Matrix Multiplikation is used to calculate the correlation matrix.

    Equivalent to cosine measure for normalized vectors. Example code:
        similarity_matrix = np.zeros((vectors.shape[0], vectors.shape[0]))
        for i, v1 in enumerate(vectors):
            for j, v2 in enumerate(vectors):
                similarity_matrix[i, j] = cosine_similarity(v1, v2)

    Returns:
        np.ndarray: A matrix of correlations/cosine similarities between each pair of vectors.

    Example:
        vectors = np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]])
        correlation_matrix = correlation_matrix(vectors)
        print(correlation_matrix)
    """
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    correlation_matrix = normalized_vectors @ normalized_vectors.T

    return correlation_matrix


## normalization
def normalize_01(vector, axis=1):
    axis = 0 if axis == 1 and len(vector.shape) == 1 else axis
    min_val = np.min(vector, axis=axis, keepdims=True)
    max_val = np.max(vector, axis=axis, keepdims=True)
    normalized_vector = (vector - min_val) / (max_val - min_val)
    return normalized_vector


# strings
def filter_strings_by_properties(
    strings, include_properties=None, exclude_properties=None
):
    """
    Filters a list of strings based on given properties.

    Args:
    - strings (list): List of strings to filter.
    - include_properties (list or str, optional): List of properties to include in filtered strings.
        Each property can be a string or a list of strings.
    - exclude_properties (list or str, optional): List of properties to exclude from filtered strings.
        Each property can be a string or a list of strings.

    Returns:
    - filtered_strings (list): List of strings filtered based on the given properties.
    """
    filtered_strings = []

    if include_properties:
        if not isinstance(include_properties, list):
            include_properties = [include_properties]
        elif isinstance(include_properties, list) and all(
            isinstance(prop, str) for prop in include_properties
        ):
            include_properties = [include_properties]

    if exclude_properties:
        if isinstance(exclude_properties, str):
            exclude_properties = [exclude_properties]
        elif isinstance(exclude_properties, list) and all(
            isinstance(prop, str) for prop in exclude_properties
        ):
            exclude_properties = [exclude_properties]

    for string in strings:
        if include_properties:
            # Check if any of the include properties lists are present in the string
            include_check = any(
                all(prop.lower() in string.lower() for prop in props)
                for props in include_properties
            )
        else:
            include_check = True

        if exclude_properties:
            # Check if any of the exclude properties lists are present in the string
            exclude_check = any(
                all(prop.lower() in string.lower() for prop in props)
                for props in exclude_properties
            )
        else:
            exclude_check = False

        # Only include the string if it matches include properties and does not match exclude properties
        if include_check and not exclude_check:
            filtered_strings.append(string)

    return filtered_strings


def check_correct_metadata(string_or_list, name_parts):
    name_parts = make_list_ifnot(name_parts)
    success = True
    if isinstance(string_or_list, Path):
        if not string_or_list.exists():
            success = False
            global_logger.error(f"No matching file found")
            print(f"No matching file found")
        else:
            string_or_list = string_or_list.name

    if success:
        for name_part in name_parts:
            if not name_part in string_or_list:
                success = False
            if not success:
                global_logger.error(
                    f"Metadata naming does not match Object Metadata: {string_or_list} != {name_parts}"
                )
                print(
                    f"Metadata naming does not match Object Metadata: {string_or_list} != {name_parts}"
                )
    return success


# array
def sort_arr_by(arr, axis=1, sorting_indices=None):
    """
    if no sorting indices are given array is sorted by maximum value of 2d array
    """
    if sorting_indices is not None:
        indices = sorting_indices
    else:
        maxes = np.argmax(arr, axis=axis)
        indices = np.argsort(maxes)
    sorted_arr = arr[indices]
    return sorted_arr, indices


def split_array_by_zscore(array, zscore, threshold=2.5):
    above_threshold = np.where(zscore >= threshold)[0]
    below_threshold = np.where(zscore < threshold)[0]
    return array[above_threshold], array[below_threshold]


def bin_array(arr, bin_size, min_bin=None, max_bin=None):
    """
    Bin an array of floats based on a given bin size.

    Parameters:
    arr (numpy.ndarray): Input array of floats.
    bin_size (float): Size of each bin.

    Returns:
    numpy.ndarray: Binned array. starting at bin 0 to n-1
    """
    min_bin = min_bin or arr.min()
    max_bin = max_bin or arr.max()
    # Calculate the number of bins
    num_bins = int(np.ceil((max_bin - min_bin) / bin_size))

    # Calculate the edges of the bins
    bins_edges = np.linspace(min_bin, min_bin + num_bins * bin_size, num_bins + 1)

    # Bin the array
    binned_array = np.digitize(arr, bins_edges) - 1
    return binned_array


def fill_continuous_array(data_array, fps, time_gap):
    frame_gap = fps * time_gap
    # Find indices where values change
    value_changes = np.where(np.abs(np.diff(data_array)) > np.finfo(float).eps)[0] + 1
    filled_array = data_array.copy()
    # Fill gaps after continuous 2 seconds of the same value
    for i in range(len(value_changes) - 1):
        start = value_changes[i]
        end = value_changes[i + 1]
        if (
            end - start <= frame_gap
            and filled_array[start - 1] == filled_array[end + 1]
        ):
            filled_array[start:end] = [filled_array[start - 1]] * (end - start)
    return filled_array


def convert_values_to_binary(vec: np.ndarray, threshold=2.5):
    smaller_idx = np.where(vec < threshold)
    bigger_idx = np.where(vec > threshold)
    vec[smaller_idx] = 0
    vec[bigger_idx] = 1
    vec = vec.astype(int)
    return vec


def fill_vector(vector, indices, fill_value=np.nan):
    filled_vector = vector.copy()
    filled_vector[indices] = fill_value
    return filled_vector


def fill_matrix(matrix, indices, fill_value=np.nan, axis=0):
    filled_matrix = matrix.copy()
    if axis == 0:
        filled_matrix[indices, :] = fill_value
    elif axis == 1:
        filled_matrix[:, indices] = fill_value
    else:
        raise ValueError("Axis must be 0 or 1. 3D not supported.")
    return filled_matrix


def fill(matrix, indices, fill_value=np.nan):
    if len(matrix.shape) == 1:
        return fill_vector(matrix, indices, fill_value)
    else:
        return fill_matrix(matrix, indices, fill_value)


def fill_inputs(inputs: dict, indices: np.ndarray, fill_value=np.nan):
    filtered_inputs = inputs.copy()
    for key, value in filtered_inputs.items():
        if len(value.shape) == 1:
            filtered_inputs[key] = fill_vector(value, indices, fill_value=fill_value)
        else:
            filtered_inputs[key] = fill_matrix(value, indices, fill_value=fill_value)
    return filtered_inputs


def get_top_percentile_indices(vector, percentile=5, indices_smaller=True):
    cutoff = np.percentile(vector, 100 - percentile)
    if indices_smaller:
        indices = np.where(vector < cutoff)[0]
    else:
        indices = np.where(vector > cutoff)[0]
    return indices


# dict
def filter_dict_by_properties(
    dictionary,
    include_properties: List[List[str]] = None,  # or [str] or str
    exclude_properties: List[List[str]] = None,  # or [str] or str):
):
    dict_keys = dictionary.keys()
    if include_properties or exclude_properties:
        dict_keys = filter_strings_by_properties(
            dict_keys,
            include_properties=include_properties,
            exclude_properties=exclude_properties,
        )
    filtered_dict = {key: dictionary[key] for key in dict_keys}
    return filtered_dict


def sort_dict(dictionary):
    return {key: value for key, value in sorted(dictionary.items())}


def load_yaml(path, mode="r"):
    with open(path, mode) as file:
        dictionary = yaml.safe_load(file)
    return dictionary


def get_str_from_dict(dictionary, keys):
    present_keys = dictionary.keys()
    string = ""
    for variable in keys:
        if variable in present_keys:
            string += f" {dictionary[variable]}"
    return string


def keys_present(dictionary, keys):
    present_keys = dictionary.keys()
    for key in keys:
        if key not in present_keys:
            return key
    return True


def check_needed_keys(metadata, needed_attributes):
    present = keys_present(metadata, needed_attributes)
    if present != True:
        raise NameError(f"Missing metadata for: {present} not defined")


# class
def define_cls_attributes(cls_object, attributes_dict, override=False):
    """
    Defines attributes for a class object based on a dictionary.

    Args:
        cls_object (object): The class object to define attributes for.
        attributes_dict (dict): A dictionary containing attribute names as keys and their corresponding values.
        override (bool, optional): If True, existing attributes will be overridden. Defaults to False.

    Returns:
        object: The modified class object with the defined attributes.
    """
    for key, value in attributes_dict.items():
        if key not in cls_object.__dict__.keys() or override:
            setattr(cls_object, key, value)
    return cls_object


def copy_attributes_to_object(
    propertie_name_list,
    set_object,
    get_object=None,
    propertie_values=None,
    override=True,
):
    """
    Set attributes of a target object based on a list of property names and values.

    This function allows you to set attributes on a target object (the 'set_object') based on a list of
    property names provided in 'propertie_name_list' and corresponding values. You can specify these
    values directly through 'propertie_values' or retrieve them from another object ('get_object').
    If 'propertie_values' is not provided, this function will attempt to fetch the values from the
    'get_object' using the specified property names.

    Args:
        propertie_name_list (list): A list of property names to set on the 'set_object.'
        set_object (object): The target object for attribute assignment.
        get_object (object, optional): The source object to retrieve property values from. Default is None.
        propertie_values (list, optional): A list of values corresponding to the property names.
            Default is None.

    Returns:
        None

    Raises:
        ValueError: If the number of properties in 'propertie_name_list' does not match the number of values
            provided in 'propertie_values' (if 'propertie_values' is specified).

    Example Usage:
        # Example 1: Set attributes directly with values
        copy_attributes_to_object(["attr1", "attr2"], my_object, propertie_values=[value1, value2])

        # Example 2: Retrieve attribute values from another object
        copy_attributes_to_object(["attr1", "attr2"], my_object, get_object=source_object)
    """
    propertie_name_list = list(propertie_name_list)
    if propertie_values:
        propertie_values = list(propertie_values)
        if len(propertie_values) != len(propertie_name_list):
            raise ValueError(
                f"Number of properties does not match given propertie values: {len(propertie_name_list)} != {len(propertie_values)}"
            )
    elif get_object:
        propertie_values = []
        for propertie in propertie_name_list:
            if propertie in get_object.__dict__.keys():
                propertie_values.append(getattr(get_object, propertie))
            else:
                propertie_values.append(None)

    propertie_values = (
        propertie_values if propertie_values else [None] * len(propertie_name_list)
    )
    for propertie, value in zip(propertie_name_list, propertie_values):
        if propertie in set_object.__dict__.keys() and not override:
            continue
        setattr(set_object, propertie, value)


def attributes_present(attribute_names, object):
    for attribute_name in attribute_names:
        defined_variable = getattr(object, attribute_name)
        if defined_variable == None:
            return attribute_name
    return True


def set_attributes_check_presents(
    propertie_name_list,
    set_object,
    get_object=None,
    propertie_values=None,
    needed_attributes=None,
):
    copy_attributes_to_object(
        propertie_name_list=propertie_name_list,
        set_object=set_object,
        get_object=get_object,
        propertie_values=propertie_values,
    )

    if needed_attributes:
        present = attributes_present(needed_attributes, set_object)
        if present != True:
            raise NameError(
                f"Variable {present} is not defined in yaml file for {set_object.id}"
            )


# matlab
def load_matlab_metadata(mat_fpath):
    import scipy.io as sio

    mat = sio.loadmat(mat_fpath)
    # get the metadata
    dtypes = mat["metadata"][0][0].dtype.fields
    values = mat["metadata"][0][0]
    for attribute, value in zip(dtypes, values):
        value = value[0] if len(value) > 0 else " "
        print(f"{attribute:>12}: {str(value)}")


# date time
def num_to_date(date_string):
    if type(date_string) != str:
        date_string = str(date_string)
    date = datetime.strptime(date_string, "%Y%m%d")
    return date


def range_to_seconds(end: int, fps: float, start: int = 0):
    return np.arange(start, end) / fps


def second_to_time(second: float, striptime_format: str = "%M:%S"):
    return datetime.fromtimestamp(second).strftime("%M:%S")


def seconds_to_times(seconds: List[float]):
    if seconds[-1] > 60 * 60:  # if more than 1 hour
        times = np.full(len(seconds), "00:00:00")
        striptime_format = "%H:%M:%S"
    else:
        times = np.full(len(seconds), "00:00")
        striptime_format = "%M:%S"

    for i, sec in enumerate(seconds):
        times[i] = second_to_time(sec, striptime_format)
    return times


def range_to_times(end: int, fps: float, start: int = 0):
    seconds = range_to_seconds(end, fps, start)
    times = seconds_to_times(seconds)
    return times


def range_to_times_xlables_xpos(
    end: int, fps: float, start: int = 0, seconds_per_label: float = 30
):
    seconds = range_to_seconds(end, fps, start)
    if seconds[-1] > 60 * 60:  # if more than 1 hour
        striptime_format = "%H:%M:%S"
    else:
        striptime_format = "%M:%S"
    minutes = seconds_to_times(seconds)
    xticks = [minutes[0]]
    xpos = [0]
    for i, minute in enumerate(minutes[1:]):
        time_before = datetime.strptime(xticks[-1], striptime_format)
        time_current = datetime.strptime(minute, striptime_format)
        if (time_current - time_before).seconds > seconds_per_label:
            xticks.append(minute)
            xpos.append(i + 1)
    return xticks, xpos


# decorator functions
def timer(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func
