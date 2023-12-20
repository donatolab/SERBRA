# import
import os
from pathlib import Path

# calculations
import numpy as np
from scipy.signal import butter, filtfilt  # , lfilter, freqz
import sklearn

# import cupy as cp  # numpy using CUDA for faster computation
import yaml
import re

# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator


from cebra import CEBRA
import cebra
import cebra.integrations.sklearn.utils as sklearn_utils
from datetime import datetime
import torch

plt.style.use("dark_background")

# set seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def decode(
    embedding_train,
    embedding_test,
    label_train,
    label_test,
    n_neighbors=36,
    metric="cosine",
):
    # TODO: Improve decoder
    # Define decoding function with kNN decoder. For a simple demo, we will use the fixed number of neighbors 36.
    decoder = cebra.KNNDecoder(n_neighbors=n_neighbors, metric=metric)
    label_train = force_2d(label_train)
    label_train = force_1_dim_larger(label_train)

    label_test = force_2d(label_test)
    label_test = force_1_dim_larger(label_test)

    embedding_train, label_train = force_equal_dimensions(embedding_train, label_train)

    decoder.fit(embedding_train, label_train[:, 0])

    pred = decoder.predict(embedding_test)

    prediction = np.stack([pred], axis=1)

    test_score = 0  # sklearn.metrics.r2_score(label_test, prediction)
    # test_score = sklearn.metrics.r2_score(label_test[:,:2], prediction)
    pos_test_err = np.median(abs(prediction - label_test))
    pos_test_score = sklearn.metrics.r2_score(label_test[:, 0], prediction[:, 0])

    return test_score, pos_test_err, pos_test_score


def load_all_animals(
    root_dir,
    wanted_animal_ids=["all"],
    wanted_dates=["all"],
    model_settings=None,
    behavior_datas=None,
    regenerate_plots=False,
    **kwargs,
):
    """
    Loads animal data from the specified root directory for the given animal IDs.

    Parameters:
    - root_dir (string): The root directory path where the animal data is stored.
    - animal_ids (list, optional): A list of animal IDs to load. Default is ["all"].
    - generate (bool, optional): If True, generates new session data. Default is False.
    - regenerate (bool, optional): If True, regenerates existing session data. Default is False.
    - sessions (string, optional): Specifies the sessions. Default is "single".
    - delete (bool, optional): If True, deletes session data. Default is False.

    Returns:
    - animals_dict (dict): A dictionary containing animal IDs as keys and corresponding Animal objects as values.
    """
    root_dir = Path(root_dir)
    present_animal_ids = get_directories(root_dir, regex_search="DON-")
    animals_dict = {}
    if not model_settings:
        model_settings = kwargs

    # Search for animal_ids
    for animal_id in present_animal_ids:
        if animal_id in wanted_animal_ids or "all" in wanted_animal_ids:
            animal = Animal(
                animal_id=animal_id, root_dir=root_dir, model_settings=model_settings
            )
            animal.add_all_sessions(
                wanted_dates=wanted_dates,
                behavior_datas=behavior_datas,
                regenerate_plots=regenerate_plots,
            )
        animals_dict[animal_id] = animal
    animals_dict = sort_dict(animals_dict)
    return animals_dict


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
        files_list = [
            name
            for name in os.listdir(directory)
            if directory.joinpath(name).is_file()
            and len(re.findall(regex_search, name)) > 0
            and name.endswith(ending)
        ]
    else:
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
    return [string_or_list] if type(string_or_list) != list else string_or_list


def save_file_present(file_path, show_print=False):
    file_path = Path(file_path)
    file_present = False
    if file_path.exists():
        if show_print:
            print(f"File already present {file_path}")
        file_present = True
    else:
        if show_print:
            print(f"Saving {file_path.name} to {file_path}")
    return file_present


# Math
def calc_cumsum_distances(positions, track_length, distance_threshold=30):
    """
    Calculates the cumulative sum of distances between positions along a track.

    Args:
    - positions (array-like): List or array of positions along the track.
    - track_length (numeric): Length of the track.
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
                cumsum_distance += frame_distance + track_length
            # check if mouse moves from start to end of the track
            else:
                cumsum_distance += frame_distance - track_length
        old_position = position
        cumsum_distances.append(cumsum_distance)
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
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, "valid")


def shuffle_data(data: np.ndarray):
    shuffled_data = []
    data = force_1_dim_larger(data)
    for data_type in data.transpose():
        shuffled_data.append(np.random.permutation(data_type))
    return np.array(shuffled_data).transpose()


def continuouse_to_discrete(continuouse_array, lengths: list):
    """
    Converts continuouse data into discrete values based on track lengths using NumPy.

    Parameters:
    - continuouse_array: array of continuouse data.
    - track_lengths:  containing lengths of track parts.

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


def force_1_dim_larger(arr: np.ndarray):
    if len(arr.shape) == 1 or arr.shape[0] < arr.shape[1]:
        print("Data is probably transposed. Needed Shape [Time, cells] Transposing...")
        return arr.T  # Transpose the array if the condition is met
    else:
        return arr  # Return the original array if the condition is not met


def force_2d(arr: np.ndarray):
    arr_2d = arr
    if len(arr.shape) == 1:
        arr_2d = np.array([arr])
    return arr_2d


def force_equal_dimensions(array1: np.ndarray, array2: np.ndarray):
    shape_0_diff = array1.shape[0] - array2.shape[0]
    if shape_0_diff > 0:
        array1 = array1[:-shape_0_diff]
    elif shape_0_diff < 0:
        array2 = array2[:shape_0_diff]
    return array1, array2


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
        if isinstance(include_properties, str):
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
            print(f"No matching file found")
        else:
            string_or_list = string_or_list.name

    if success:
        for name_part in name_parts:
            if not name_part in string_or_list:
                success = False
            if not success:
                print(
                    f"Metadata naming does not match Object Metadata: {string_or_list} != {name_parts}"
                )
    return success


# date
def num_to_date(date_string):
    if type(date_string) != str:
        date_string = str(date_string)
    date = datetime.strptime(date_string, "%Y%m%d")
    return date


# array
def fill_continuous_array(data_array, fps, time_gap):
    frame_gap = fps * time_gap
    # Find indices where values change
    value_changes = np.where(np.abs(np.diff(data_array)) > np.finfo(float).eps)[0] + 1

    # Fill gaps after continuous 2 seconds of the same value
    for i in range(len(value_changes) - 1):
        start = value_changes[i]
        end = value_changes[i + 1] - 1
        if end - start <= frame_gap and data_array[start - 1] == data_array[end + 1]:
            data_array[start:end] = [data_array[start - 1]] * (end - start)

    return data_array


# dict
def filter_dict_by_properties(
    dictionary,
    include_properties: [[str]] = None,  # or [str] or str
    exclude_properties: [[str]] = None,  # or [str] or str):
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


# class
def define_cls_attributes(cls_object, attributes_dict, override=False):
    for key, value in attributes_dict.items():
        if key not in cls_object.__dict__.keys() or override:
            setattr(cls_object, key, value)
    return cls_object


def copy_attributes_to_object(
    propertie_name_list, set_object, get_object=None, propertie_values=None
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
                f"Variable {present} is not defined in yaml file {set_object.yaml_path}"
            )


class Dataset:
    def __init__(self, key, path=None, data=None, raw_data_object=None, metadata=None):
        self.path: Path = path
        self.data: np.ndarray = data
        self.key = key
        self.raw_data_object = raw_data_object
        self.metadata = metadata
        self.fps = metadata["fps"] if "fps" in metadata.keys() else None
        self.title = None
        self.ylable = None
        self.xlable = None
        self.xlimits = None
        self.num_ticks = None
        self.figsize = None
        self.save_path = None

    # TODO: use __subclass__?
    # def __init_subclass__(cls, key: str, **kwargs):
    #    cls.key = key
    #    super().__init_subclass__(**kwargs)

    def load(
        self,
        path=None,
        save=True,
        regenerate=True,
        plot=True,
        regenerate_plot=False,
    ):
        if not type(self.data) == np.ndarray:
            self.path = path if path else self.path
            if not self.raw_data_object and regenerate:
                print(
                    f"No raw data given. Regeneration not possible. Loading old data."
                )
                regenerate = False
            if path.exists() and not regenerate:
                print(f"Loading {self.path}")
                # ... and similarly load the .h5 file, providing the columns to keep
                # continuous_label = cebra.load_data(file="auxiliary_behavior_data.h5", key="auxiliary_variables", columns=["continuous1", "continuous2", "continuous3"])
                # discrete_label = cebra.load_data(file="auxiliary_behavior_data.h5", key="auxiliary_variables", columns=["discrete"]).flatten()
                self.data = cebra.load_data(file=self.path)
                data_dimensions = self.data.shape
                if len(data_dimensions) == 2:
                    num_time_points, num_cells = self.data.shape
                    if num_cells > num_time_points:
                        print(
                            "Data is probably transposed. Needed Shape [Time, cells] Transposing..."
                        )
                        self.data = self.data.transpose()
            else:
                self.create_dataset(self.raw_data_object)
                if save:
                    np.save(self.path, self.data)
        self.data = self.correct_data(self.data)
        if plot:
            self.plot(regenerate_plot=regenerate_plot)
        return self.data

    def create_dataset(self, raw_data_object=None):
        print(f"No {self.key} data found at {self.path}. Creation not possible.")
        raw_data_object = raw_data_object or self.raw_data_object
        if self.raw_data_object:
            print(f"Creating {self.key} dataset...")
            data = self.process_raw_data()
            return data
        else:
            print(f"No raw data given. Creation not possible. Skipping.")

    def process_raw_data(self):
        raise NotImplementedError(
            f"ERROR: Function for creating {self.key} dataset from raw data is not defined."
        )

    def correct_data(self, data):
        return data

    def split(self, split_ratio=0.8):
        data = self.data
        split_index = int(len(data) * split_ratio)
        data_train = data[:split_index]
        data_test = data[split_index:]
        return data_train, data_test

    def shuffle(self):
        # TODO: test this
        """
        def shuffle_data(data: np.ndarray):
            shuffled_data = []
            data = force_1_dim_larger(data)
            for data_type in data.transpose():
                shuffled_data.append(np.random.permutation(data_type))
            return np.array(shuffled_data).transpose()
        """
        return sklearn.utils.shuffle(self.data)

    def set_plot_parameter(
        self,
        fps=None,
        title=None,
        ylable=None,
        xlable=None,
        num_ticks=None,
        xlimits=None,
        figsize=None,
        regenerate_plot=None,
        save_path=None,
    ):
        self.fps = self.fps or fps or 30
        self.title = self.title or title or f"{self.path.stem} data"
        self.title = self.title if self.title[-4:] == "data" else self.title + " data"
        descriptive_metadata_keys = [
            "area",
            "stimulus_type",
            "method",
            "processing_software",
        ]
        self.title += get_str_from_dict(
            dictionary=self.metadata, keys=descriptive_metadata_keys
        )
        self.num_ticks = self.num_ticks or num_ticks or 50
        self.ylable = self.ylable or ylable or self.key
        self.xlable = self.xlable or xlable or "seconds"
        self.xlimits = self.xlimits or xlimits or (0, len(self.data))
        self.figsize = self.figsize or figsize or (20, 3)
        self.regenerate_plot = regenerate_plot or False
        self.save_path = (
            self.save_path
            or save_path
            or self.path.parent.parent.parent.joinpath(
                "figures", self.path.stem + ".png"
            )
        )

    def set_data_plot(self):
        plt.plot(self.data)

    def set_xticks_plot(
        self,
        seconds_interval=5,
        written_label_steps=2,
    ):
        num_frames = self.data.shape[0]
        frame_interval = self.fps * seconds_interval
        time = [
            int(frame / frame_interval) * seconds_interval
            for frame in range(num_frames)
            if frame % frame_interval == 0
        ]
        steps = round(len(time) / (2 * self.num_ticks))
        time_shortened = time[::steps]
        pos = np.arange(0, num_frames, frame_interval)[::steps]
        labels = [
            time if num % written_label_steps == 0 else ""
            for num, time in enumerate(time_shortened)
        ]
        plt.xticks(pos, labels, rotation=40)

    def plot(
        self,
        figsize=None,
        title=None,
        xlable=None,
        xlimits=None,
        ylable=None,
        seconds_interval=5,
        fps=None,
        num_ticks=50,
        written_label_steps=2,
        save_path=None,
        regenerate_plot=None,
        show=False,
        dpi=300,
    ):
        self.set_plot_parameter(
            fps=fps,
            xlable=xlable,
            xlimits=xlimits,
            num_ticks=num_ticks,
            ylable=ylable,
            title=title,
            figsize=figsize,
            regenerate_plot=regenerate_plot,
            save_path=save_path,
        )

        plt.figure(figsize=self.figsize)
        if regenerate_plot or not save_file_present(self.save_path):
            plt.title(self.title)
            plt.ylabel(self.ylable)
            plt.xlabel(self.xlable)
            plt.xlim(self.xlimits)
            plt.tight_layout()

            self.set_data_plot()
            self.set_xticks_plot(
                seconds_interval=seconds_interval,
                written_label_steps=written_label_steps,
            )
            plt.savefig(self.save_path, dpi=dpi)
        else:
            # Load the image
            image = plt.imread(self.save_path)
            plt.imshow(image)
            plt.axis("off")

        if show:
            plt.show()


class Data_Position(Dataset):
    def __init__(self, raw_data_object=None, metadata=None):
        super().__init__(
            key="position", raw_data_object=raw_data_object, metadata=metadata
        )
        self.track_length = self.metadata["track_length"]

    # FIXME: why is this data 1 datapoint smaller than neural data?


class Data_Stimulus(Dataset):
    def __init__(self, raw_data_object=None, metadata=None):
        super().__init__(
            key="stimulus", raw_data_object=raw_data_object, metadata=metadata
        )
        # TODO: only working for steffens treadmil introduce into yaml
        # file no handling other types of stimuly besides track stimulus
        self.stimulus_sequence = self.metadata["stimulus_sequence"]
        self.simulus_length = self.metadata["stimulus_length"]
        self.stimulus_type = self.metadata["stimulus_type"]

    def process_raw_data(self):
        """ "
        Returns:
            - data: Numpy array composed of stimulus type at frames.
        """
        stimulus_raw_data = self.raw_data_object.data  # e.g. Position on a track/time
        stimulus_type_indexes = continuouse_to_discrete(
            stimulus_raw_data, self.simulus_length
        )
        stimulus_type_at_frame = np.array(self.stimulus_sequence)[
            stimulus_type_indexes % len(self.stimulus_sequence)
        ]
        self.data = stimulus_type_at_frame
        return self.data


class Data_Distance(Dataset):
    def __init__(self, raw_data_object=None, metadata=None):
        super().__init__(
            key="distance", raw_data_object=raw_data_object, metadata=metadata
        )
        self.track_length = self.metadata["track_length"]

    def process_raw_data(self):
        track_positions = self.raw_data_object.data
        self.data = calc_cumsum_distances(track_positions, self.track_length)
        return self.data


class Data_Velocity(Dataset):
    def __init__(self, raw_data_object=None, metadata=None):
        super().__init__(
            key="velocity", raw_data_object=raw_data_object, metadata=metadata
        )
        self.raw_velocitys = None
        self.ylable = "velocity cm/s"

    def process_raw_data(self):
        """
        calculating velocity based on velocity data in raw_data_object
        """
        raw_data_type = self.raw_data_object.key
        if raw_data_type == "position":
            track_positions = self.raw_data_object.data
            track_length = self.raw_data_object.track_length
            walked_distances = calc_cumsum_distances(track_positions, track_length)
        elif raw_data_type == "distance":
            walked_distances = self.raw_data_object.data
        else:
            raise ValueError(f"Raw data type {raw_data_type} not supported.")
        self.raw_velocitys = calculate_derivative(walked_distances)
        smoothed_velocity = butter_lowpass_filter(
            self.raw_velocitys, cutoff=2, fs=self.fps, order=2
        )
        # smoothed_velocity = moving_average(self.raw_velocitys)
        print(
            f"Calculating smooth velocity based on butter_lowpass_filter 2Hz, {self.fps}fps, 2nd order."
        )
        # change value/frame to value/second
        self.data = smoothed_velocity * self.fps
        return self.data


class Data_Acceleration(Dataset):
    def __init__(self, raw_data_object=None, metadata=None):
        super().__init__(
            key="acceleration", raw_data_object=raw_data_object, metadata=metadata
        )
        self.raw_acceleration = None
        self.ylable = "acceleration cm/s^2"

    def process_raw_data(self):
        """
        calculating acceleration based on velocity data in raw_data_object
        """
        velocity = self.raw_data_object.data
        self.raw_acceleration = calculate_derivative(velocity)
        smoothed_acceleration = butter_lowpass_filter(
            self.raw_acceleration, cutoff=2, fs=self.fps, order=2
        )
        print(
            f"Calculating smooth acceleration based on butter_lowpass_filter 2Hz, {self.fps}fps, 2nd order."
        )
        self.data = smoothed_acceleration
        return self.data


class Data_Moving(Dataset):
    def __init__(self, raw_data_object=None, metadata=None):
        super().__init__(
            key="moving", raw_data_object=raw_data_object, metadata=metadata
        )
        self.ylable = "moving"
        self.velocity_threshold = 2  # cm/s
        self.brain_processing_delay = {
            "CA1": 2,  # seconds
            "CA3": 2,
            "M1": None,
            "S1": None,
            "V1": None,
        }

    def process_raw_data(self):
        velocities = self.raw_data_object.data
        moving_frames = velocities > self.velocity_threshold
        self.data = self.fit_moving_to_brainarea(moving_frames, self.metadata["area"])
        return self.data

    def fit_moving_to_brainarea(self, data, area):
        processing_movement_lag = self.brain_processing_delay[area]  # seconds
        if not processing_movement_lag:
            raise NotImplementedError(f"{area} processing lag not implemented.")
        processing_movement_frames = fill_continuous_array(
            data, fps=self.fps, time_gap=processing_movement_lag
        )
        return processing_movement_frames

    def correct_data(self, data):
        return self.fit_moving_to_brainarea(data, self.metadata["area"])


class Data_CA(Dataset):
    def __init__(self, raw_data_object=None, metadata=None):
        super().__init__(
            key="neural", raw_data_object=raw_data_object, metadata=metadata
        )
        self.title = f"Raster Plot of Binarized Neural Data: : {self.metadata}"
        self.ylable = "Neuron ID"
        self.figsize = (20, 10)

    def set_data_plot(self):
        binarized_data = self.data
        num_time_steps, num_neurons = binarized_data.shape
        # Find spike indices for each neuron
        spike_indices = np.nonzero(binarized_data)
        # Creating an empty image grid
        image = np.zeros((num_neurons, num_time_steps))
        # Marking spikes as pixels in the image grid
        image[spike_indices[1], spike_indices[0]] = 1
        # Plotting the raster plot using pixels
        plt.imshow(image, cmap="gray", aspect="auto", interpolation="none")
        plt.gca().invert_yaxis()  # Invert y-axis for better visualization of trials/neurons


class Data_CAM(Dataset):
    def __init__(self, raw_data_object=None, metadata=None):
        super().__init__(key="cam", raw_data_object=raw_data_object, metadata=metadata)
        # TODO: implement cam data loading


class Datasets:
    def __init__(self, data_dir, metadata={}):
        self.data_dir = data_dir
        self.metadata = metadata
        self.data_sources = {}

    def load(self, taskial_fname, data_source, regenerate_plot=False):
        fname_ending = self.data_sources[data_source]
        fname = f"{taskial_fname}_{fname_ending}.npy"
        fpath = self.data_dir.joinpath(fname)
        data_object = getattr(self, data_source)
        data = data_object.load(fpath, regenerate_plot=regenerate_plot)
        return data

    def get_multi_data(self, sources, shuffle=False, idx_to_keep=None, split_ratio=1):
        # FIXME: is this correct?
        sources = make_list_ifnot(sources)
        concatenated_data = None
        for source in sources:
            dataset_object = getattr(self, source)
            data = dataset_object.data
            data = np.array([data]).transpose() if len(data.shape) == 1 else data
            if type(concatenated_data) != np.ndarray:
                concatenated_data = data
            else:
                concatenated_data, data = force_equal_dimensions(
                    concatenated_data, data
                )
                concatenated_data = np.concatenate([concatenated_data, data], axis=1)
        concatenated_data_filtered = self.filter_by_idx(idx_to_keep=idx_to_keep)
        concatenated_data_shuffled = (
            self.shuffle(concatenated_data_filtered)
            if shuffle
            else concatenated_data_filtered
        )
        concatenated_data_tain, concatenated_data_test = self.split(
            concatenated_data_shuffled, split_ratio
        )
        return concatenated_data_tain, concatenated_data_test

    def split(self, data, split_ratio=0.8):
        split_index = int(len(data) * split_ratio)
        data_train = data[:split_index]
        data_test = data[split_index:]
        return data_train, data_test

    def shuffle(self, data):
        """
        def shuffle_data(data: np.ndarray):
            shuffled_data = []
            data = force_1_dim_larger(data)
            for data_type in data.transpose():
                shuffled_data.append(np.random.permutation(data_type))
            return np.array(shuffled_data).transpose()
        """
        return sklearn.utils.shuffle(data)  # TODO: Test this

    def filter_by_idx(self, data, idx_to_keep=None):
        if idx_to_keep:
            data = data[idx_to_keep]
        return data


class Datasets_neural(Datasets):
    def __init__(self, data_dir, metadata={}):
        super().__init__(data_dir, metadata=metadata)
        self.suite2p = Data_CA(metadata=self.metadata)
        self.inscopix = Data_CA(metadata=self.metadata)
        self.data_sources = {"suite2p": "ca_data", "inscopix": "???"}


class Datasets_behavior(Datasets):
    def __init__(self, data_dir, metadata={}):
        super().__init__(data_dir, metadata=metadata)
        self.position = Data_Position(metadata=self.metadata)
        self.stimulus = Data_Stimulus(
            raw_data_object=self.position, metadata=self.metadata
        )
        self.distance = Data_Distance(
            raw_data_object=self.position, metadata=self.metadata
        )
        self.velocity = Data_Velocity(
            raw_data_object=self.distance, metadata=self.metadata
        )
        self.acceleration = Data_Acceleration(
            raw_data_object=self.velocity, metadata=self.metadata
        )
        self.moving = Data_Moving(raw_data_object=self.velocity, metadata=self.metadata)
        self.cam = Data_CAM(metadata=self.metadata)
        self.data_sources = {
            "position": "pos_track",
            "distance": "distance_track",
            "velocity": "velocity_track",
            "acceleration": "acceleration_track",
            "stimulus": "stimulus_track",
            "moving": "moving_track",
            "cam": "???",
        }


class Animal:
    def __init__(
        self, animal_id, root_dir, animal_dir=None, model_settings=None, **kwargs
    ):
        self.id = animal_id
        self.cohort_year = None
        self.dob = None
        self.sex = None
        self.root_dir = Path(root_dir)
        self.dir = animal_dir or self.root_dir.joinpath(animal_id)
        self.yaml_path = self.dir.joinpath(f"{animal_id}.yaml")
        self.sessions = {}
        self.model_settings = model_settings or kwargs
        success = self.load_metadata()
        if not success:
            return None

    def load_metadata(self, yaml_path=None, name_parts=None):
        yaml_path = yaml_path or self.yaml_path
        name_parts = name_parts or self.id
        needed_attributes = None
        success = check_correct_metadata(
            string_or_list=yaml_path, name_parts=name_parts
        )

        if success:
            animal_metadata_dict = load_yaml(yaml_path)
            # Load any additional metadata into session object
            set_attributes_check_presents(
                propertie_name_list=animal_metadata_dict.keys(),
                set_object=self,
                propertie_values=animal_metadata_dict.values(),
                needed_attributes=needed_attributes,
            )
        return success

    def add_session(
        self,
        date,
        model_settings=None,
        behavior_datas=None,
        regenerate_plots=False,
        **kwargs,
    ):
        if not model_settings:
            model_settings = kwargs if len(kwargs) > 0 else self.model_settings

        session = Session(
            self.id,
            date=date,
            animal_dir=self.dir,
            behavior_datas=behavior_datas,
            model_settings=model_settings,
            regenerate_plots=regenerate_plots,
        )

        if session:
            session.pday = (num_to_date(session.date) - num_to_date(self.dob)).days
            self.sessions[date] = session
            self.sessions = sort_dict(self.sessions)
            return self.sessions[date]
        else:
            print(f"Skipping {self.animal_id} {date}")
        return

    def add_all_sessions(
        self,
        wanted_dates=["all"],
        behavior_datas=None,
        model_settings=None,
        regenerate_plots=False,
        **kwargs,
    ):
        if not model_settings:
            model_settings = kwargs if len(kwargs) > 0 else self.model_settings
        # Search for Sessions
        sessions_root_path = self.root_dir.joinpath(self.animal_id)
        present_session_dates = get_directories(sessions_root_path)
        for date in present_session_dates:
            if date in wanted_dates or "all" in wanted_dates:
                self.add_session(
                    date,
                    model_settings=model_settings,
                    behavior_datas=behavior_datas,
                    regenerate_plots=regenerate_plots,
                )

    def plot_consistency_scores(
        self,
        wanted_stimulus_types,
        wanted_embeddings,
        exclude_properties=None,
        figsize=(7, 7),
    ):
        # TODO: change to only inter Session?
        viz = Vizualizer(root_dir=self.root_dir)
        labels = {}  # {wanted_embedding 1: {animal_session_task_id: embedding}, ...}
        for wanted_embedding in wanted_embeddings:
            labels[wanted_embedding] = {"embeddings": {}, "labels": {}}
            for wanted_stimulus_type in wanted_stimulus_types:
                for session_date, session in self.sessions.items():
                    for task_name, task in session.tasks.items():
                        if (
                            task.behavior_metadata["stimulus_type"]
                            == wanted_stimulus_type
                        ):
                            wanted_embeddings_dict = filter_dict_by_properties(
                                task.embeddings,
                                include_properties=wanted_embedding,
                                exclude_properties=exclude_properties,
                            )
                            for (
                                embedding_key,
                                embedding,
                            ) in wanted_embeddings_dict.items():
                                labels_id = f"{session_date[-3:]}_{task.task} {wanted_stimulus_type}"
                                position_lables = task.behavior.position.data
                                position_lables, embedding = force_equal_dimensions(
                                    position_lables, embedding
                                )
                                labels[wanted_embedding]["embeddings"][
                                    labels_id
                                ] = embedding
                                labels[wanted_embedding]["labels"][
                                    labels_id
                                ] = position_lables

            dataset_ids = list(labels[wanted_embedding]["embeddings"].keys())
            embeddings = list(labels[wanted_embedding]["embeddings"].values())
            labeling = list(labels[wanted_embedding]["labels"].values())

            title = f"CEBRA-{wanted_embedding} embedding consistency"
            fig = plt.figure(figsize=figsize)
            ax1 = plt.subplot(111)
            ax1 = viz.plot_consistency_scores(
                ax1, title, embeddings, labeling, dataset_ids
            )
            plt.show()


class Session:
    def __init__(
        self,
        animal_id,
        date,
        animal_dir,
        session_dir=None,
        data_dir=None,
        model_dir=None,
        behavior_datas=None,
        regenerate_plots=False,
        model_settings={},
        **kwargs,
    ):
        self.animal_id = animal_id
        self.date = date
        self.id = f"{self.animal_id}_{self.date}"
        self.animal_dir = animal_dir
        self.dir = Path(session_dir or self.animal_dir.joinpath(date))
        self.data_dir = data_dir or self.dir
        self.model_dir = Path(model_dir or self.dir.joinpath("models"))
        self.model_settings = model_settings
        self.yaml_path = self.dir.joinpath(f"{self.date}.yaml")
        self.tasks_infos = None
        self.tasks = {}
        success = self.load_metadata()
        if not success:
            return
        if behavior_datas:
            self.add_all_tasks(model_settings=self.model_settings, **kwargs)
            self.load_all_data(
                behavior_datas=behavior_datas, regenerate_plots=regenerate_plots
            )

    def load_metadata(self, yaml_path=None, name_parts=None):
        name_parts = name_parts or self.date
        yaml_path = yaml_path or self.yaml_path
        success = check_correct_metadata(
            string_or_list=yaml_path, name_parts=name_parts
        )
        if success:
            # Load any additional metadata into session object
            session_metadata_dict = load_yaml(yaml_path)
            needed_attributes = ["tasks_infos"]
            set_attributes_check_presents(
                propertie_name_list=session_metadata_dict.keys(),
                set_object=self,
                propertie_values=session_metadata_dict.values(),
                needed_attributes=needed_attributes,
            )
        return success

    def add_task(self, task, metadata=None, model_settings=None, **kwargs):
        success = check_correct_metadata(
            string_or_list=self.tasks_infos.keys(), name_parts=task
        )
        if success:
            task_object = Task(
                self.id,
                task,
                self.dir,
                self.data_dir,
                self.model_dir,
                metadata=metadata,
            )
            if not model_settings:
                model_settings = kwargs if len(kwargs) > 0 else self.model_settings
            task_object.init_models(model_settings)
            self.tasks[task] = task_object
        else:
            print(f"Skipping Task {task}")

    def add_all_tasks(self, model_settings=None, **kwargs):
        for task, metadata in self.tasks_infos.items():
            if not model_settings:
                model_settings = kwargs if len(kwargs) > 0 else self.model_settings
            self.add_task(task, metadata)
        return self.tasks[task]

    def load_all_data(self, behavior_datas=["position"], regenerate_plots=False):
        data = {}
        for task_name, task in self.tasks.items():
            data[task_name] = task.load_all_data(
                behavior_datas=behavior_datas, regenerate_plots=regenerate_plots
            )
        return data

    def plot_multiple_consistency_scores(
        self,
        animals,
        wanted_stimulus_types=["time", "behavior"],
        wanted_embeddings=["A", "B", "A'"],
        exclude_properties=None,
        figsize=(7, 7),
    ):
        # TODO: implement consistency session plot
        pass


class Task:
    def __init__(
        self,
        session_id,
        task,
        session_dir,
        data_dir=None,
        model_dir=None,
        metadata: dict = {},
    ):
        self.session_id = session_id
        self.id = f"{session_id}-{task}"
        self.task = task
        self.task_num = int(task[1:]) - 1

        self.load_metadata(metadata)
        self.data_dir = data_dir or session_dir
        self.neural = Datasets_neural(self.data_dir, self.neural_metadata)
        self.behavior_metadata["area"] = self.neural_metadata[
            "area"
        ]  # TODO: ok like this?
        self.behavior = Datasets_behavior(self.data_dir, self.behavior_metadata)

        self.model_dir = model_dir or self.data_dir.joinpath("models")
        self.models = None
        self.embeddings = {}

    def load_metadata(self, metadata: dict = {}):
        needed_attributes = ["neural_metadata", "behavior_metadata"]
        set_attributes_check_presents(
            propertie_name_list=metadata.keys(),
            set_object=self,
            propertie_values=metadata.values(),
            needed_attributes=needed_attributes,
        )

    def load_data(self, data_source, data="neural", regenerate_plot=False):
        source = "TRD-2P"
        task_source_id = (
            f"{self.session_id}_{source}_{self.task}-ACQ_eval_session_{self.task_num}"
        )
        # loads neural or behaviour data
        data_object = getattr(self, data)
        data = data_object.load(
            task_source_id, data_source=data_source, regenerate_plot=regenerate_plot
        )
        return data

    def load_all_data(self, behavior_datas=["position"], regenerate_plots=False):
        """
        neural_Data = ["suite2p", "inscopix"]
        behavior_datas = ["position", "cam"]
        """
        data = {"neural": {}, "behavior": {}}
        for mouse_data in ["neural", "behavior"]:
            if mouse_data == "neural":
                # TODO: not sure if this is enough, maybe path set in yaml
                # TODO: currently only suite2p loading is done, add inscopix
                data_to_load = self.neural_metadata["preprocessing_software"]
            else:
                data_to_load = behavior_datas
            data_to_load = make_list_ifnot(data_to_load)
            for data_source in data_to_load:
                data[mouse_data][data_source] = self.load_data(
                    data_source, data=mouse_data, regenerate_plot=regenerate_plots
                )
        return data

    def init_models(self, model_settings=None, **kwargs):
        if not model_settings:
            model_settings = kwargs
        self.models = Models(
            self.model_dir, model_id=self.task, model_settings=model_settings
        )

    def set_model_name(self, model_type, name, shuffled):
        model_name = model_type
        model_name = model_name if name == model_type else f"{model_name}_{name}"
        model_name = f"{model_name}_shuffled" if shuffled else model_name
        return model_name

    def train_model(
        self,
        model_type,
        regenerate: bool = False,
        shuffled: bool = False,
        only_moving: bool = False,
        split_ratio: float = None,
        model_name: str = None,
        neural_data: np.ndarray = None,
        behavior_data: np.ndarray = None,
        behavior_data_types: [str] = ["position"],
        manifolds_pipeline: str = "cebra",
    ):
        """
        available model_types are: time, behavior, hybrid
        """
        model_name_id = f"{model_type}_{model_name}" if model_name else model_type
        model_name = model_name or model_type
        model_name_id_shuffled = (
            f"{model_name_id}_shuffled" if shuffled else model_name_id
        )
        if manifolds_pipeline == "cebra":
            cebras_class = self.models.cebras
            if model_name_id_shuffled in cebras_class.models.keys():
                model = cebras_class.models[model_name_id_shuffled]
            else:
                model_creation_function = getattr(cebras_class, model_type)

                model = model_creation_function(shuffled=shuffled, name=model_name)

            if not model.fitted or regenerate:
                idx_to_keep = self.behavior.moving.data if only_moving else None
                if not isinstance(neural_data, np.ndarray):
                    neural_data, _ = self.neural.get_multi_data(
                        self.neural_metadata["preprocessing_software"],
                        idx_to_keep=idx_to_keep,
                    )
                    # shuffle=shuffled,
                    # split_ratio=split_ratio)
                if not isinstance(behavior_data, np.ndarray):
                    behavior_data, _ = self.behavior.get_multi_data(
                        behavior_data_types,
                        idx_to_keep=idx_to_keep,
                    )
                    # shuffle=shuffled,
                    # split_ratio=split_ratio)

                neural_data_filtered = self.neural.filter(
                    neural_data, idx_to_keep=idx_to_keep
                )
                neural_data_shuffled = (
                    self.neural.shuffle(neural_data_filtered)
                    if shuffled
                    else neural_data_filtered
                )
                neural_data_train, neural_data_test = self.neural.split(
                    neural_data_shuffled, split_ratio
                )

                behavior_data_filtered = self.behavior.filter(
                    behavior_data, idx_to_keep=idx_to_keep
                )
                behavior_data_shuffled = (
                    self.behavior.shuffle(behavior_data_filtered)
                    if shuffled
                    else behavior_data_filtered
                )
                behavior_data_train, behavior_data_test = self.behavior.split(
                    behavior_data_shuffled, split_ratio
                )

                neural_data = force_2d(neural_data)
                behavior_data = force_2d(behavior_data)
                neural_data = force_1_dim_larger(neural_data)
                behavior_data = force_1_dim_larger(behavior_data)

                #####################################################
                # delete
                print(asdf)
                behavior_data = (
                    shuffle_data(behavior_data) if shuffled else behavior_data
                )
                #####################################################

                print(f"{self.id}: Training  {model.name} model.")
                if model_type == "time":
                    model.fit(neural_data)
                else:
                    neural_data, behavior_data = force_equal_dimensions(
                        neural_data, behavior_data
                    )
                    model.fit(neural_data, behavior_data)
                model.fitted = cebras_class.fitted(model)
                model.save(model.save_path)
            else:
                print(f"{self.id}: {model.name} model already trained. Skipping.")
        return model

    def create_embeddings(
        self,
        models=None,
        to_transform_data=None,
        model_naming_filter_include: [[str]] = None,  # or [str] or str
        model_naming_filter_exclude: [[str]] = None,  # or [str] or str
        manifolds_pipeline="cebra",
    ):
        # TODO: create function to integrate train and test embeddings into models
        if not type(to_transform_data) == np.ndarray:
            print(f"No neural data given. Using default labels. suite2p")
            to_transform_data = self.neural.suite2p.data

        filtered_models = models or self.get_pipeline_models(
            manifolds_pipeline, model_naming_filter_include, model_naming_filter_exclude
        )

        for model_name, model in filtered_models.items():
            embedding_title = f"{model_name} - {model.max_iterations}"
            embedding = model.transform(to_transform_data)
            self.embeddings[embedding_title] = embedding
        return self.embeddings

    def plot_model_embeddings(
        self,
        model_naming_filter_include: [[str]] = None,  # or [str] or str
        model_naming_filter_exclude: [[str]] = None,  # or [str] or str
        to_transform_data=None,
        embedding_labels: dict = None,
        behavior_data_types=["position"],
        manifolds_pipeline="cebra",
        set_title=None,
    ):
        embeddings = self.create_embeddings(
            to_transform_data=to_transform_data,
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
            manifolds_pipeline="cebra",
        )

        # get embedding lables
        if not isinstance(embedding_labels, np.ndarray) and not isinstance(
            embedding_labels, dict
        ):
            print(
                f"No embedding labels given. Using behavior_data_types: {behavior_data_types}"
            )
            behavior_datas = self.behavior.get_multi_data(behavior_data_types)
            embedding_labels
            embedding_labels_dict = {}
            for behavior_data_type, behavior_data in zip(
                behavior_data_types, behavior_datas.transpose()
            ):
                embedding_labels_dict[behavior_data_type] = behavior_data
        else:
            if isinstance(embedding_labels, np.ndarray):
                embedding_labels_dict = {"Provided_labels": embedding_labels}
            else:
                embedding_labels_dict = embedding_labels

        viz = Vizualizer(self.data_dir.parent.parent)

        for embedding_title, embedding_labels in embedding_labels_dict.items():
            if set_title:
                title = set_title
            else:
                title = f"{manifolds_pipeline.upper()} embeddings {self.id}"
                descriptive_metadata_keys = [
                    "stimulus_type",
                    "method",
                    "processing_software",
                ]
                title += (
                    get_str_from_dict(
                        dictionary=self.behavior_metadata,
                        keys=descriptive_metadata_keys,
                    )
                    + f" - {embedding_title}"
                )
            labels_dict = {"name": embedding_title, "labels": embedding_labels}
            viz.plot_multiple_embeddings(embeddings, labels=labels_dict, title=title)
        return embeddings

    def get_pipeline_models(
        self,
        manifolds_pipeline="cebra",
        model_naming_filter_include: [[str]] = None,  # or [str] or str
        model_naming_filter_exclude: [[str]] = None,  # or [str] or str
    ):
        if manifolds_pipeline == "cebra":
            models = self.models.cebras.get_models(
                model_naming_filter_include=model_naming_filter_include,
                model_naming_filter_exclude=model_naming_filter_exclude,
            )
        else:
            raise ValueError(
                f"Manifolds Pipeline {manifolds_pipeline} is not implemented. Use 'cebra'."
            )
        return models

    def get_models_splitted_original_shuffled(
        self,
        models=None,
        manifolds_pipeline="cebra",
        model_naming_filter_include: [[str]] = None,  # or [str] or str
        model_naming_filter_exclude: [[str]] = None,  # or [str] or str
    ):
        models_original = []
        models_shuffled = []
        models = models or self.get_pipeline_models(
            manifolds_pipeline, model_naming_filter_include, model_naming_filter_exclude
        )

        # convert list of model objects to dictionary[model_name] = model
        if isinstance(models, list):
            models_dict = {}
            for model in models:
                models_dict[model.name] = model
            models = models_dict

        for model_name, model in models.items():
            if "shuffled" in model.name:
                models_shuffled.append(model)
            else:
                models_original.append(model)
        return models_original, models_shuffled

    def plot_model_losses(
        self,
        models=None,
        title=None,
        manifolds_pipeline="cebra",
        coloring_type="rainbow",
        plot_original=True,
        plot_shuffled=True,
        plot_model_iterations=False,
        model_naming_filter_include: [[str]] = None,  # or [str] or str
        model_naming_filter_exclude: [[str]] = None,  # or [str] or str
        alpha=0.8,
        figsize=(10, 10),
    ):
        models_original, models_shuffled = self.get_models_splitted_original_shuffled(
            models=models,
            manifolds_pipeline=manifolds_pipeline,
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
        )
        stimulus_type = self.behavior_metadata["stimulus_type"]

        title = title or f"Losses {self.session_id} {stimulus_type}"
        title += (
            f" - {models_original[0].max_iterations} Iterartions"
            if not plot_model_iterations
            else ""
        )

        viz = Vizualizer(self.data_dir.parent.parent)
        viz.plot_losses(
            models=models_original,
            models_shuffled=models_shuffled,
            title=title,
            coloring_type=coloring_type,
            plot_original=plot_original,
            plot_shuffled=plot_shuffled,
            alpha=alpha,
            figsize=figsize,
            plot_model_iterations=plot_model_iterations,
        )

    def plot_multiple_consistency_scores(
        self,
        animals,
        wanted_stimulus_types,
        wanted_embeddings,
        exclude_properties=None,
        figsize=(7, 7),
    ):
        # TODO: implement consistency task plot
        pass


class Models:
    def __init__(self, model_dir, model_id, model_settings=None, **kwargs):
        if not model_settings:
            model_settings = kwargs
        self.cebras = Cebras(model_dir, model_id, model_settings)

    def train(self):
        # TODO: move train_model function from task class
        pass

    def set_model_name(self):
        # TODO: move set_model_name function from task class
        pass


class Cebras:
    def __init__(self, model_dir, model_id, model_settings=None, **kwargs):
        self.model_id = model_id
        self.model_dir = Path(model_dir)
        if not model_settings:
            model_settings = kwargs
        self.model_settings = model_settings
        self.models = {}
        self.time_model = self.time()
        self.behavior_model = self.behavior()
        self.hybrid_model = self.hybrid()

    def get_models(
        self,
        model_naming_filter_include: [[str]] = None,  # or [str] or str
        model_naming_filter_exclude: [[str]] = None,  # or [str] or str):
    ):
        filtered_models = filter_dict_by_properties(
            self.models, model_naming_filter_include, model_naming_filter_exclude
        )
        return filtered_models

    def embedd_into_model(self, model, data):
        # TODO: embedd data into model, may create train and test embeddings, integrate this function into model?
        pass

    def define_parameter_save_path(self, model):
        dir_exist_create(self.model_dir)
        save_path = self.model_dir.joinpath(
            f"cebra_{model.name}_iter-{model.max_iterations}_dim-{model.output_dimension}_model-{self.model_id}.pt"
        )
        return save_path

    def fitted(self, model):
        return sklearn_utils.check_fitted(model)

    def create_defaul_model(self):
        default_model = CEBRA(
            model_architecture="offset10-model",
            batch_size=512,
            learning_rate=3e-4,
            temperature=1,
            output_dimension=3,
            max_iterations=5000,
            distance="cosine",
            conditional="time_delta",
            device="cuda_if_available",
            verbose=True,
            time_offsets=10,
        )
        return default_model

    def init_model(self, settings_dict):
        default_model = self.create_defaul_model()
        if len(settings_dict) == 0:
            settings_dict = self.model_settings
        initial_model = define_cls_attributes(
            default_model, settings_dict, override=True
        )
        initial_model.fitted = False
        return initial_model

    def load_fitted_model(self, model):
        fitted_model_path = model.save_path
        if fitted_model_path.exists():
            fitted_model = CEBRA.load(fitted_model_path)
            if fitted_model.get_params() == model.get_params():
                fitted_full_model = define_cls_attributes(fitted_model, model.__dict__)
                model = fitted_full_model
                print(f"Loaded matching model {fitted_model_path}")
            else:
                print(
                    f"Loaded model parameters do not match to initialized model. Not loading {fitted_model_path}"
                )
        model.fitted = self.fitted(model)
        return model

    def time(self, name="time", **kwargs):
        model = self.init_model(kwargs)
        model.temperature = (
            1.12 if kwargs.get("temperature") is None else model.temperature
        )
        model.conditional = "time" if kwargs.get("time") is None else model.conditional
        model.name = name

        model.save_path = self.define_parameter_save_path(model)
        model = self.load_fitted_model(model)
        self.models[model.name] = model
        return model

    def behavior(self, name="behavior", **kwargs):
        model = self.init_model(kwargs)
        model.name = name
        model.save_path = self.define_parameter_save_path(model)
        model = self.load_fitted_model(model)
        self.models[model.name] = model
        return model

    def hybrid(self, name="hybrid", **kwargs):
        model = self.init_model(kwargs)
        model.name = name
        model.hybrid = True if kwargs.get("hybrid") is None else model.hybrid
        model.save_path = self.define_parameter_save_path(model)
        model = self.load_fitted_model(model)
        self.models[model.name] = model
        return model


class Vizualizer:
    def __init__(self, root_dir) -> None:
        self.save_dir = Path(root_dir).joinpath("figures")
        dir_exist_create(self.save_dir)

    def plot_dataplot_summary(self, plot_dir, title="Data Summary"):
        # TODO: combine plots from top to bottom aligned by time
        # distance, velocity, acceleration, position, raster
        pass

    def plot_embedding(
        self,
        ax,
        embedding,
        embedding_labels: dict,
        title="Embedding",
        cmap="rainbow",
        plot_legend=True,
        colorbar_ticks=None,
    ):
        embedding, labels = force_equal_dimensions(
            embedding, embedding_labels["labels"]
        )

        ax = cebra.plot_embedding(
            ax=ax,
            embedding=embedding,
            embedding_labels=labels,
            title=title,
            figsize=(10, 10),
            cmap=cmap,
        )
        if plot_legend:
            # Create a ScalarMappable object using the specified colormap
            sm = plt.cm.ScalarMappable(cmap=cmap)
            unique_labels = np.unique(labels)
            unique_labels.sort()
            sm.set_array(unique_labels)  # Set the range of values for the colorbar

            # Manually create colorbar
            cbar = plt.colorbar(sm, ax=ax)
            # Adjust colorbar ticks if specified
            cbar.set_label(embedding_labels["name"])  # Set the label for the colorbar

            if colorbar_ticks:
                cbar.ax.yaxis.set_major_locator(
                    MaxNLocator(integer=True)
                )  # Adjust ticks to integers
                cbar.set_ticks(colorbar_ticks)  # Set custom ticks

        return ax

    def plot_multiple_embeddings(
        self,
        embeddings: dict,
        labels: dict,
        title="Embeddings",
        cmap="rainbow",
        projection="3d",
        figsize=(20, 4),
        plot_legend=True,
    ):
        figsize_x, figsize_y = figsize
        fig = plt.figure(figsize=figsize)

        # Compute the number of subplots
        num_subplots = len(embeddings)
        rows = 1
        cols = num_subplots
        if num_subplots > 4:
            rows = int(num_subplots**0.5)
            cols = (num_subplots + rows - 1) // rows
            figsize = (figsize_x, figsize_y * rows)

        fig, axes = plt.subplots(
            rows, cols, figsize=figsize, subplot_kw={"projection": projection}
        )

        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, (subplot_title, embedding) in enumerate(embeddings.items()):
            ax = axes[i]
            ax = self.plot_embedding(
                ax, embedding, labels, subplot_title, cmap, plot_legend=False
            )

        for ax in axes[num_subplots:]:
            ax.remove()  # Remove any excess subplot axes

        if plot_legend:
            # Create a ScalarMappable object using the specified colormap
            sm = plt.cm.ScalarMappable(cmap=cmap)
            unique_labels = np.unique(labels["labels"])
            unique_labels.sort()
            sm.set_array(unique_labels)  # Set the range of values for the colorbar

            # Manually create colorbar
            cbar = plt.colorbar(sm, ax=ax)
            # Adjust colorbar ticks if specified
            cbar.set_label(labels["name"])  # Set the label for the colorbar

        self.plot_ending(title)

    def plot_losses(
        self,
        models,
        models_shuffled=[],
        title="Losses",
        coloring_type="rainbow",
        plot_original=True,
        plot_shuffled=True,
        plot_model_iterations=False,
        alpha=0.8,
        figsize=(10, 10),
    ):
        plt.figure(figsize=figsize)
        ax = plt.subplot(111)

        if coloring_type == "rainbow":
            num_colors = len(models) + len(models_shuffled)
            rainbow_colors = [
                mcolors.to_rgba(c, alpha=alpha)
                for c in plt.cm.rainbow(np.linspace(0, 1, num_colors))
            ]
            colors = (
                rainbow_colors[: len(models)],
                rainbow_colors[len(models) :],
            )

        elif coloring_type == "distinct":
            # Generate distinct colors for models and models_shuffled
            colors_original = [
                mcolors.to_rgba(c, alpha=alpha)
                for c in plt.cm.tab10(np.linspace(0, 1, len(models)))
            ]
            colors_shuffled = [
                mcolors.to_rgba(c, alpha=alpha)
                for c in plt.cm.Set3(np.linspace(0, 1, len(models_shuffled)))
            ]
            colors = colors_original, colors_shuffled

        elif coloring_type == "mono":  # Blues and Reds
            blue_colors = [
                mcolors.to_rgba(c, alpha=alpha)
                for c in plt.cm.Blues(np.linspace(0.3, 1, len(models)))
            ]
            reds_colors = [
                mcolors.to_rgba(c, alpha=alpha)
                for c in plt.cm.Reds(np.linspace(0.3, 1, len(models_shuffled)))
            ]
            colors = (
                blue_colors,  # colors_original
                reds_colors,  # colors_shuffled
            )

        else:
            raise ValueError("Invalid coloring type. Choose 'rainbow' or 'distinct'.")

        # Plotting
        if plot_original and plot_shuffled:
            models_to_plot = models + models_shuffled
            colors_to_use = colors[0] + colors[1]
        elif plot_original:
            models_to_plot = models
            colors_to_use = colors[0]
            title += f"{title} not shuffled"
        else:
            models_to_plot = models_shuffled
            title += f"{title} shuffled"
            colors_to_use = colors[1]

        for color, model in zip(colors_to_use, models_to_plot):
            label = model.name.split("behavior_")[-1]
            label += f" - {model.max_iterations} Iter" if plot_model_iterations else ""
            ax = cebra.plot_loss(model, color=color, alpha=alpha, label=label, ax=ax)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("InfoNCE Loss")
        plt.legend(bbox_to_anchor=(0.5, 0.3), frameon=False)
        self.plot_ending(title)

    def plot_ending(self, title):
        plt.suptitle(title)
        plt.tight_layout()  # Ensure subplots fit within figure area
        plot_save_path = self.save_dir.joinpath(title + ".png")
        plt.savefig(plot_save_path, dpi=300)
        plt.show()

    def plot_consistency_scores(self, ax1, title, embeddings, labels, dataset_ids):
        (
            time_scores,
            time_pairs,
            time_subjects,
        ) = cebra.sklearn.metrics.consistency_score(
            embeddings=embeddings,
            labels=labels,
            dataset_ids=dataset_ids,
            between="datasets",
        )
        ax1 = cebra.plot_consistency(
            time_scores,
            pairs=time_pairs,
            datasets=time_subjects,
            ax=ax1,
            title=title,
            colorbar_label="consistency score",
        )
        return ax1

    def plot_multiple_consistency_scores(
        self,
        animals,
        wanted_stimulus_types,
        wanted_embeddings,
        exclude_properties=None,
        figsize=(7, 7),
    ):
        # TODO: change this to a more modular funciton, integrate into classes
        # labels to align the subjects is the position of the mouse in the arena
        # labels = {}  # {wanted_embedding 1: {animal_session_task_id: embedding}, ...}
        # for wanted_embedding in wanted_embeddings:
        #    labels[wanted_embedding] = {"embeddings": {}, "labels": {}}
        #    for wanted_stimulus_type in wanted_stimulus_types:
        #        for animal, session, task in yield_animal_session_task(animals):
        #            if task.behavior_metadata["stimulus_type"] == wanted_stimulus_type:
        #                wanted_embeddings_dict = filter_dict_by_properties(
        #                    task.embeddings,
        #                    include_properties=wanted_embedding,
        #                    exclude_properties=exclude_properties,
        #                )
        #                for embedding_key, embedding in wanted_embeddings_dict.items():
        #                    labels_id = f"{session.date[-3:]}_{task.task} {wanted_stimulus_type}"
        #                    position_lables = task.behavior.position.data
        #                    position_lables, embedding = force_equal_dimensions(
        #                        position_lables, embedding
        #                    )
        #                    labels[wanted_embedding]["embeddings"][
        #                        labels_id
        #                    ] = embedding
        #                    labels[wanted_embedding]["labels"][
        #                        labels_id
        #                    ] = position_lables
        #
        #    dataset_ids = list(labels[wanted_embedding]["embeddings"].keys())
        #    embeddings = list(labels[wanted_embedding]["embeddings"].values())
        #    labeling = list(labels[wanted_embedding]["labels"].values())
        #
        #    title = f"CEBRA-{wanted_embedding} embedding consistency"
        #    fig = plt.figure(figsize=figsize)
        #    ax1 = plt.subplot(111)
        #    ax1 = self.plot_consistency_score(
        #        ax1, title, embeddings, labeling, dataset_ids
        #    )
        #    plt.show()
        pass
        # self.plot_ending(title)

    def plot_decoding_score(
        self,
        decoded_model_lists,
        labels,
        title="Behavioral Decoding of Position",
        colors=["deepskyblue", "gray"],
        figsize=(13, 5),
    ):
        # TODO: improve this function, modularity, flexibility
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=16)
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)

        overall_num = 0
        for color, docoded_model_list in zip(colors, decoded_model_lists):
            for num, decoded_model in enumerate(docoded_model_list):
                alpha = 1 - ((1 / len(docoded_model_list)) * num / 1.3)
                x_pos = overall_num + num
                width = 0.4  # Width of the bars
                ax1.bar(
                    x_pos, decoded_model.decoded[1], width=0.4, color=color, alpha=alpha
                )
                label = "".join(decoded_model.name.split("_train")).split("behavior_")[
                    -1
                ]
                ax2.scatter(
                    decoded_model.state_dict_["loss"][-1],
                    decoded_model.decoded[1],
                    s=50,
                    c=color,
                    alpha=alpha,
                    label=label,
                )
            overall_num += x_pos + 1

        x_label = "InfoNCE Loss (contrastive learning)"
        ylabel = "Median position error in cm"

        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.set_ylabel(ylabel)
        labels = labels
        label_pos = np.arange(len(labels))
        ax1.set_xticks(label_pos)
        ax1.set_xticklabels(labels, rotation=45, ha="right")

        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.set_xlabel(x_label)
        ax2.set_ylabel(ylabel)
        plt.legend(bbox_to_anchor=(1, 1), frameon=False)
        plt.show()
