# import
import os
from pathlib import Path

# setups and preprocessing software
from Setups import *
from Helper import *


# type hints
from typing import List, Union, Dict, Any, Tuple, Optional

# calculations
import numpy as np
from scipy.signal import butter, filtfilt  # , lfilter, freqz
import sklearn
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import seaborn as sns


from cebra import CEBRA
import cebra
import cebra.integrations.sklearn.utils as sklearn_utils
import torch

# pip install binarize2pcalcium
# from binarize2pcalcium import binarize2pcalcium as binca

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
    label_train = Dataset.force_2d(label_train)
    # label_train = Dataset.force_1_dim_larger(label_train)

    label_test = Dataset.force_2d(label_test)
    # label_test = force_1_dim_larger(label_test)

    embedding_train, label_train = Datasets.force_equal_dimensions(
        embedding_train, label_train
    )

    decoder.fit(embedding_train, label_train[:, 0])

    pred = decoder.predict(embedding_test)

    prediction = np.stack([pred], axis=1)

    test_score = 0  # sklearn.metrics.r2_score(label_test, prediction)
    # test_score = sklearn.metrics.r2_score(label_test[:,:2], prediction)

    # pos_test_err = np.median(abs(prediction - label_test))
    ########################################################
    pos_test_err = np.mean(abs(prediction - label_test))
    ########################################################
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


class DataPlotterInterface:
    def __init__(self):
        self.data = None
        self.metadata = None
        self.plot_attributes = {
            "title": None,
            "ylable": None,
            "ylimits": None,
            "yticks": None,
            "xlable": None,
            "xlimits": None,
            "xticks": None,
            "num_ticks": None,
            "figsize": None,
            "save_path": None,
        }

    def set_plot_parameter(
        self,
        fps=None,
        title=None,
        ylable=None,
        ylimits=None,
        yticks=None,
        xlable=None,
        xticks=None,
        num_ticks=None,
        xlimits=None,
        figsize=None,
        regenerate_plot=None,
        save_path=None,
    ):
        self.fps = self.fps or fps or 30
        self.plot_attributes["title"] = (
            self.plot_attributes["title"] or title or f"{self.path.stem} data"
        )
        self.plot_attributes["title"] = (
            self.plot_attributes["title"]
            if self.plot_attributes["title"][-4:] == "data"
            else self.plot_attributes["title"] + " data"
        )
        descriptive_metadata_keys = [
            "area",
            "stimulus_type",
            "method",
            "processing_software",
        ]
        self.plot_attributes["title"] += get_str_from_dict(
            dictionary=self.metadata, keys=descriptive_metadata_keys
        )
        self.plot_attributes["ylable"] = (
            self.plot_attributes["ylable"] or ylable or self.key
        )
        self.plot_attributes["ylimits"] = (
            self.plot_attributes["ylimits"] or ylimits or None
        )
        self.plot_attributes["yticks"] = (
            self.plot_attributes["yticks"] or yticks or None
        )
        self.plot_attributes["xlable"] = (
            self.plot_attributes["xlable"] or xlable or "seconds"
        )
        self.plot_attributes["xlimits"] = (
            self.plot_attributes["xlimits"] or xlimits or (0, len(self.data))
        )
        self.plot_attributes["xticks"] = (
            self.plot_attributes["xticks"] or xticks or None
        )
        self.plot_attributes["num_ticks"] = (
            self.plot_attributes["num_ticks"] or num_ticks or 50
        )
        self.plot_attributes["figsize"] = (
            self.plot_attributes["figsize"] or figsize or (20, 3)
        )
        self.regenerate_plot = regenerate_plot or False
        self.plot_attributes["save_path"] = (
            self.plot_attributes["save_path"]
            or save_path
            or self.path.parent.parent.parent.joinpath(
                "figures", self.path.stem + ".png"
            )
        )
        # create plot dir if missing
        if not self.plot_attributes["save_path"].parent.exists():
            self.plot_attributes["save_path"].parent.mkdir(parents=True, exist_ok=True)

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
        steps = round(len(time) / (2 * self.plot_attributes["num_ticks"]))
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
        xticks=None,
        ylable=None,
        ylimits=None,
        yticks=None,
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
            ylimits=ylimits,
            yticks=yticks,
            ylable=ylable,
            xlable=xlable,
            xlimits=xlimits,
            xticks=xticks,
            num_ticks=num_ticks,
            title=title,
            figsize=figsize,
            regenerate_plot=regenerate_plot,
            save_path=save_path,
        )

        plt.figure(figsize=self.plot_attributes["figsize"])
        if regenerate_plot or not save_file_present(self.plot_attributes["save_path"]):
            plt.title(self.plot_attributes["title"])
            plt.ylabel(self.plot_attributes["ylable"])
            if self.plot_attributes["ylimits"]:
                plt.ylim(self.plot_attributes["ylimits"])
            if self.plot_attributes["yticks"]:
                plt.yticks(
                    self.plot_attributes["yticks"][0], self.plot_attributes["yticks"][1]
                )
            plt.xlabel(self.plot_attributes["xlable"])
            plt.tight_layout()
            plt.xlim(self.plot_attributes["xlimits"])

            self.set_data_plot()

            if self.plot_attributes["xticks"]:
                plt.xticks(self.plot_attributes["xticks"])
            else:
                self.set_xticks_plot(
                    seconds_interval=seconds_interval,
                    written_label_steps=written_label_steps,
                )
            plt.savefig(self.plot_attributes["save_path"], dpi=dpi)
        else:
            # Load the image
            image = plt.imread(self.plot_attributes["save_path"])
            plt.imshow(image)
            plt.axis("off")

        if show:
            plt.show()
            plt.close()


class Dataset(DataPlotterInterface):
    def __init__(
        self,
        key,
        path=None,
        data=None,
        raw_data_object=None,
        metadata=None,
        root_dir=None,
        task_id=None,
    ):
        # initialize plotting parameters
        super().__init__()
        self.root_dir: Path = root_dir
        self.path: Path = path
        self.data_dir = None
        self.data: np.ndarray = data
        self.key = key
        self.task_id = task_id
        self.raw_data_object = raw_data_object
        self.metadata = metadata
        self.fps = None if "fps" not in self.metadata.keys() else self.metadata["fps"]
        self.setup = self.get_setup(self.metadata["setup"])

    def get_setup(self, setup_name, preprocessing_name, method_name):
        raise NotImplementedError(
            f"ERROR: Function get_setup is not defined for {self.__class__}."
        )

    def load(
        self,
        path=None,
        save=True,
        regenerate=False,
        plot=True,
        regenerate_plot=False,
    ):
        if not type(self.data) == np.ndarray:
            self.path = path if path else self.path
            self.data_dir = self.path.parent
            # if no raw data exists, regeneration is not possible
            if not self.raw_data_object and regenerate:
                global_logger.warning(f"No raw data given. Regeneration not possible.")
                global_logger.info(f"Loading old data.")
                print(
                    f"No raw data given. Regeneration not possible. Loading old data."
                )
                regenerate = False

            # Check if the file exists, and if it does, load the data, else create the dataset
            if path.exists() and not regenerate:
                global_logger.info(f"Loading {self.path}")
                print(f"Loading {self.path}")
                # ... and similarly load the .h5 file, providing the columns to keep
                # continuous_label = cebra.load_data(file="auxiliary_behavior_data.h5", key="auxiliary_variables", columns=["continuous1", "continuous2", "continuous3"])
                # discrete_label = cebra.load_data(file="auxiliary_behavior_data.h5", key="auxiliary_variables", columns=["discrete"]).flatten()
                self.data = cebra.load_data(file=self.path)
                data_dimensions = self.data.shape
                if len(data_dimensions) == 2:
                    # format of data should be [num_time_points, num_cells]
                    self.data = self.force_1_dim_larger(data=self.data)
            else:
                global_logger.warning(f"No {self.key} data found at {self.path}.")
                print(f"No {self.key} data found at {self.path}.")
                self.create_dataset(self.raw_data_object)
                if save:
                    np.save(self.path, self.data)
        self.data = self.correct_data(self.data)
        if plot:
            self.plot(regenerate_plot=regenerate_plot)
        return self.data

    def create_dataset(self, raw_data_object=None):
        raw_data_object = raw_data_object or self.raw_data_object
        if self.raw_data_object:
            global_logger.info(
                f"Creating {self.key} dataset based on raw data from {raw_data_object.key}."
            )
            print(
                f"Creating {self.key} dataset based on raw data from {raw_data_object.key}."
            )
            data = self.process_raw_data()
            return data
        else:
            global_logger.warning(
                f"No raw data given. Creation not possible. Skipping."
            )
            print(f"No raw data given. Creation not possible. Skipping.")

    def process_raw_data(self):
        global_logger.critical(
            f"Function for creating {self.key} dataset from raw data is not defined."
        )
        raise NotImplementedError(
            f"ERROR: Function for creating {self.key} dataset from raw data is not defined."
        )

    def correct_data(self, data):
        return data

    @staticmethod
    def split(data, split_ratio=0.8):
        split_index = int(len(data) * split_ratio)
        data_train = data[:split_index]
        data_test = data[split_index:]
        return data_train, data_test

    @staticmethod
    def shuffle(data):
        return sklearn.utils.shuffle(data)

    @staticmethod
    def filter_by_idx(data, idx_to_keep=None):
        if isinstance(idx_to_keep, np.ndarray) or isinstance(idx_to_keep, list):
            data_unfiltered, idx_to_keep = Datasets.force_equal_dimensions(
                data, idx_to_keep
            )
            data_filtered = data_unfiltered[idx_to_keep]
            return data_filtered
        else:
            global_logger
            print(f"No idx_to_keep given. Returning unfiltered data.")
            return data

    @staticmethod
    def force_1_dim_larger(data: np.ndarray):
        if len(data.shape) == 1 or data.shape[0] < data.shape[1]:
            global_logger.warning(
                f"Data is probably transposed. Needed Shape [Time, cells] Transposing..."
            )
            print(
                "Data is probably transposed. Needed Shape [Time, cells] Transposing..."
            )
            return data.T  # Transpose the array if the condition is met
        else:
            return data  # Return the original array if the condition is not met

    @staticmethod
    def force_2d(data: np.ndarray):
        data_2d = data
        dimensions = len(data.shape)
        if dimensions == 1:
            data_2d = np.array([data])
            data_2d = data_2d.T
        elif dimensions > 2:
            raise ValueError("Data has more than 2 dimensions.")
        return data_2d


class BehaviorDataset(Dataset):
    def __init__(
        self,
        key,
        path=None,
        data=None,
        raw_data_object=None,
        metadata=None,
        root_dir=None,
        task_id=None,
    ):
        super().__init__(
            key, path, data, raw_data_object, metadata, root_dir, task_id=task_id
        )
        needed_attributes = ["setup"]
        check_needed_keys(metadata, needed_attributes)

    def get_setup(self, setup_name):
        self.setup_name = self.metadata["setup"]
        if setup_name == "active_avoidance":
            setup = Active_Avoidance(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        elif setup_name == "box":
            setup = Box(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        elif setup_name == "treadmill":
            setup = Treadmill_Setup(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        elif setup_name == "trackball":
            setup = Trackball_Setup(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        elif setup_name == "openfield":
            setup = Openfield_Setup(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        elif setup_name == "vr_treadmill":
            setup = VR_Wheel(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        elif setup_name == "wheel":
            setup = Rotary_Wheel(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        else:
            raise ValueError(f"Behavior setup {setup_name} not supported.")
        return setup


class NeuralDataset(Dataset):
    def __init__(
        self,
        key,
        path=None,
        data=None,
        raw_data_object=None,
        metadata=None,
        root_dir=None,
        task_id=None,
    ):
        super().__init__(
            key, path, data, raw_data_object, metadata, root_dir, task_id=task_id
        )
        needed_attributes = ["method", "preprocessing_software", "setup"]
        check_needed_keys(metadata, needed_attributes)

    def process_raw_data(self):
        self.setup.preprocess.process_data()


class Data_Position(BehaviorDataset):
    def __init__(
        self, raw_data_object=None, metadata=None, root_dir=None, task_id=None
    ):
        super().__init__(
            key="position",
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        self.plot_attributes["ylable"] = "position cm"
        self.environment_dimensions = self.metadata["environment_dimensions"]

    def create_dataset(self, raw_data_object=None):
        data = self.process_raw_data()
        return data

    def process_raw_data(self):
        self.data = self.setup.process_data(self.task_id)
        return self.data

    # FIXME: why is this data 1 datapoint smaller than neural data?


class Data_Stimulus(BehaviorDataset):
    def __init__(
        self, raw_data_object=None, metadata=None, root_dir=None, task_id=None
    ):
        super().__init__(
            key="stimulus",
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        # TODO: only working for steffens treadmil introduce into yaml
        # file no handling other types of stimuly besides track stimulus
        self.stimulus_sequence = self.metadata["stimulus_sequence"]
        self.simulus_length = self.metadata["stimulus_length"]
        self.stimulus_type = self.metadata["stimulus_type"]
        self.stimulus_by = self.metadata["stimulus_by"]
        self.fps = self.metadata["fps"] if "fps" in self.metadata.keys() else None

    def process_raw_data(self):
        """ "
        Returns:
            - data: Numpy array composed of stimulus type at frames.
        """
        stimulus_raw_data = self.raw_data_object.data  # e.g. Position on a track/time
        if self.stimulus_by == "location":
            stimulus_type_at_frame = self.stimulus_by_location(stimulus_raw_data)
        elif self.stimulus_by == "frames":
            stimulus_type_at_frame = self.stimulus_by_time(stimulus_raw_data)
        elif self.stimulus_by == "seconds":
            stimulus_type_at_frame = self.stimulus_by_time(
                stimulus_raw_data, time_to_frame_multiplier=self.fps
            )
        elif self.stimulus_by == "minutes":
            stimulus_type_at_frame = self.stimulus_by_time(
                stimulus_raw_data, time_to_frame_multiplier=60 * self.fps
            )
        self.data = stimulus_type_at_frame
        return self.data

    def stimulus_by_location(self, stimulus_raw_data):
        """
        location on a treadmill
        """
        # TODO: implement stimulus by location in 2D and 3D
        stimulus_type_indexes = continuouse_to_discrete(
            stimulus_raw_data, self.simulus_length
        )
        stimulus_type_at_frame = np.array(self.stimulus_sequence)[
            stimulus_type_indexes % len(self.stimulus_sequence)
        ]
        return stimulus_type_at_frame

    def stimulus_by_time(self, stimulus_raw_data, time_to_frame_multiplier=1):
        """
        time in frames in an experiment
        """
        stimulus_type_at_frame = []
        for stimulus, duration in zip(self.stimulus_sequence, self.simulus_length):
            stimulus_type_at_frame += [stimulus] * int(
                duration * time_to_frame_multiplier
            )
        return np.array(stimulus_type_at_frame)


class Data_Distance(BehaviorDataset):
    def __init__(
        self, raw_data_object=None, metadata=None, root_dir=None, task_id=None
    ):
        super().__init__(
            key="distance",
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        self.environment_dimensions = make_list_ifnot(
            self.metadata["environment_dimensions"]
        )

    def process_raw_data(self):
        track_positions = self.raw_data_object.data
        if len(self.environment_dimensions) == 1:
            self.data = self.process_1D_data(track_positions)
        elif len(self.environment_dimensions) == 2:
            self.data = self.process_2D_data(track_positions)
        elif len(self.environment_dimensions) == 3:
            self.data = self.process_3D_data(track_positions)
        return self.data

    def process_1D_data(self, track_positions):
        """
        calculating velocity based on velocity data in raw_data_object
        """
        return calc_cumsum_distances(track_positions, self.environment_dimensions)

    def process_2D_data(self, track_positions):
        """
        calculating velocity based on velocity data in raw_data_object
        """
        # TODO: implement 2D distance calculation
        raise NotImplementedError("2D distance calculation not implemented.")
        return calc_cumsum_distances(track_positions, self.environment_dimensions)

    def process_3D_data(self, track_positions):
        """
        calculating velocity based on velocity data in raw_data_object
        """
        # TODO: implement 3D distance calculation
        raise NotImplementedError("3D distance calculation not implemented.")
        return calc_cumsum_distances(track_positions, self.environment_dimensions)


class Data_Velocity(BehaviorDataset):
    def __init__(
        self, raw_data_object=None, metadata=None, root_dir=None, task_id=None
    ):
        super().__init__(
            key="velocity",
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        self.raw_velocitys = None
        self.plot_attributes["ylable"] = "velocity cm/s"

    def process_raw_data(self):
        """
        calculating velocity based on velocity data in raw_data_object
        """
        raw_data_type = self.raw_data_object.key
        if raw_data_type == "distance":
            walked_distances = self.raw_data_object.data
        else:
            raise ValueError(f"Raw data type {raw_data_type} not supported.")
        self.raw_velocitys = calculate_derivative(walked_distances)
        smoothed_velocity = butter_lowpass_filter(
            self.raw_velocitys, cutoff=2, fs=self.fps, order=2
        )
        # smoothed_velocity = moving_average(self.raw_velocitys)
        global_logger
        print(
            f"Calculating smooth velocity based on butter_lowpass_filter 2Hz, {self.fps}fps, 2nd order."
        )
        # change value/frame to value/second
        self.data = smoothed_velocity * self.fps
        return self.data


class Data_Acceleration(BehaviorDataset):
    def __init__(
        self, raw_data_object=None, metadata=None, root_dir=None, task_id=None
    ):
        super().__init__(
            key="acceleration",
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        self.raw_acceleration = None
        self.plot_attributes["ylable"] = "acceleration cm/s^2"

    def process_raw_data(self):
        """
        calculating acceleration based on velocity data in raw_data_object
        """
        velocity = self.raw_data_object.data
        self.raw_acceleration = calculate_derivative(velocity)
        smoothed_acceleration = butter_lowpass_filter(
            self.raw_acceleration, cutoff=2, fs=self.fps, order=2
        )
        global_logger
        print(
            f"Calculating smooth acceleration based on butter_lowpass_filter 2Hz, {self.fps}fps, 2nd order."
        )
        self.data = smoothed_acceleration
        return self.data


class Data_Moving(BehaviorDataset):
    def __init__(
        self, raw_data_object=None, metadata=None, root_dir=None, task_id=None
    ):
        super().__init__(
            key="moving",
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        self.plot_attributes["ylable"] = "Movement State"
        self.plot_attributes["ylimits"] = (-0.1, 1.1)
        self.plot_attributes["yticks"] = [[0, 1], ["Stationary", "Moving"]]
        self.velocity_threshold = 2  # cm/s
        self.brain_processing_delay = {
            "CA1": 2,  # seconds
            "CA3": 2,
            "M1": None,
            "S1": None,
            "V1": None,
        }
        # FIXME: move processing to setup
        needed_attributes = ["setup"]
        check_needed_keys(metadata, needed_attributes)

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


class Data_Photon(NeuralDataset):
    """
    Dataset class for managing photon data.
    Attributes:
        - data (np.ndarray): The binariced traces.
        - method (str): The method used to record the data.
        - preprocessing_software (str): The software used to preprocess the data.
            - raw fluoresence traces can be found inside preprocessing object #TODO: implement example
        - setup (str): The setup used to record the data.
        - setup.data = raw_data_object
    """

    def __init__(
        self,
        path=None,
        raw_data_object=None,
        metadata=None,
        root_dir=None,
        task_id=None,
    ):
        super().__init__(
            key="photon",
            path=path,
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        self.plot_attributes["title"] = (
            f"Raster Plot of Binarized {self.key} Data: : {self.metadata}"
        )
        self.plot_attributes["ylable"] = "Neuron ID"
        self.plot_attributes["figsize"] = (20, 10)

    def process_raw_data(self):
        self.data = self.setup.process_raw_data()
        return self.data

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

    def get_setup(self, setup_name):
        if setup_name == "femtonics":
            setup = Femtonics(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        elif setup_name == "thorlabs":
            setup = Thorlabs(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        elif setup_name == "inscopix":
            setup = Inscopix(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        else:
            raise ValueError(f"Imaging setup {setup_name} not supported.")
        return setup


class Data_Probe(Dataset):
    def __init__(self, path=None, raw_data_object=None, metadata=None):
        super().__init__(
            key="probe", path=path, raw_data_object=raw_data_object, metadata=metadata
        )
        # TODO: implement probe data loading
        raise NotImplementedError("Probe data loading not implemented.")


class Datasets:
    """
    Dataset class for managing multiple datasets. Getting correct data paths and loading data.
    """

    def __init__(self, root_dir, metadata={}, task_id=None):
        if "data_dir" in metadata.keys():
            self.data_dir = Path(root_dir).joinpath(metadata["data_dir"])
        else:
            self.data_dir = Path(root_dir)
        self.metadata = metadata
        self.task_id = task_id

    def get_object(self, data_source):
        raise NotImplementedError(
            f"ERROR: Function get_object is not defined for {self.__class__}."
        )

    def load(self, task_id, data_source, regenerate_plot=False):
        """
        Load data from a specific data source.
            data_object is a object inhereting the Dataset attributes and functions.
        """
        # build path to data
        data_object = self.get_object(data_source)
        fpath = data_object.setup.get_data_path(task_id)
        data = data_object.load(fpath, regenerate_plot=regenerate_plot)
        return data

    def get_multi_data(self, sources, shuffle=False, idx_to_keep=None, split_ratio=1):
        sources = make_list_ifnot(sources)
        concatenated_data = None
        for source in sources:
            dataset_object = getattr(self, source)
            data = dataset_object.data
            data = np.array([data]).transpose() if len(data.shape) == 1 else data
            if type(concatenated_data) != np.ndarray:
                concatenated_data = data
            else:
                concatenated_data, data = self.force_equal_dimensions(
                    concatenated_data, data
                )
                concatenated_data = np.concatenate([concatenated_data, data], axis=1)
        concatenated_data_filtered = Dataset.filter_by_idx(
            concatenated_data, idx_to_keep=idx_to_keep
        )
        concatenated_data_shuffled = (
            Dataset.shuffle(concatenated_data_filtered)
            if shuffle
            else concatenated_data_filtered
        )
        concatenated_data_tain, concatenated_data_test = Dataset.split(
            concatenated_data_shuffled, split_ratio
        )
        return concatenated_data_tain, concatenated_data_test

    @staticmethod
    def force_equal_dimensions(array1: np.ndarray, array2: np.ndarray):
        """
        Force two arrays to have the same dimensions.
        By cropping the larger array to the size of the smaller array.
        """
        shape_0_diff = array1.shape[0] - array2.shape[0]
        if shape_0_diff > 0:
            array1 = array1[:-shape_0_diff]
        elif shape_0_diff < 0:
            array2 = array2[:shape_0_diff]
        return array1, array2


class Datasets_Neural(Datasets):
    def __init__(self, root_dir, metadata={}, task_id=None):
        super().__init__(root_dir=root_dir, metadata=metadata, task_id=task_id)
        # TODO: currently not able to manage individual analyzed sessions, needed?
        self.photon = Data_Photon(
            root_dir=root_dir, metadata=self.metadata, task_id=self.task_id
        )
        # TODO: implement probe data loading
        # self.probe = Data_Probe(root_dir=root_dir, metadata=self.metadata, task_id=self.task_id)
        # TODO: split into different datasets if needed
        self.photon_imaging_methods = ["femtonics", "thorlabs", "inscopix"]
        self.probe_imaging_methods = ["neuropixels", "tetrode"]
        if "fps" not in self.metadata.keys():
            # TODO: this is not implemented for probe data
            self.metadata["fps"] = self.photon.setup.get_fps()

    def get_object(self, data_source):
        if data_source in self.photon_imaging_methods:
            imaging_type = "photon"
        elif data_source in self.probe_imaging_methods:
            imaging_type = "probe"
        else:
            raise ValueError(f"Imaging type {data_source} not supported.")

        data_object = getattr(self, imaging_type)
        return data_object


class Datasets_Behavior(Datasets):
    def __init__(self, root_dir, metadata={}, task_id=None):
        super().__init__(root_dir=root_dir, metadata=metadata, task_id=task_id)
        self.data_dir = (
            self.data_dir
            if not self.data_dir == root_dir
            else self.data_dir.joinpath("TRD-2P")
        )
        self.position = Data_Position(
            root_dir=root_dir, metadata=self.metadata, task_id=self.task_id
        )
        self.stimulus = Data_Stimulus(
            raw_data_object=self.position,
            root_dir=root_dir,
            metadata=self.metadata,
            task_id=self.task_id,
        )
        self.distance = Data_Distance(
            raw_data_object=self.position,
            root_dir=root_dir,
            metadata=self.metadata,
            task_id=self.task_id,
        )
        self.velocity = Data_Velocity(
            raw_data_object=self.distance,
            root_dir=root_dir,
            metadata=self.metadata,
            task_id=self.task_id,
        )
        self.acceleration = Data_Acceleration(
            raw_data_object=self.velocity,
            root_dir=root_dir,
            metadata=self.metadata,
            task_id=self.task_id,
        )
        self.moving = Data_Moving(
            raw_data_object=self.velocity,
            root_dir=root_dir,
            metadata=self.metadata,
            task_id=self.task_id,
        )

    def get_object(self, data_source):
        data_object = getattr(self, data_source)
        return data_object


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
            global_logger
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
                                position_lables, embedding = (
                                    Datasets.force_equal_dimensions(
                                        position_lables, embedding
                                    )
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
        self.tasks_infos = None  # loaded from yaml
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
            global_logger
            print(f"Skipping Task {task}")

    def add_all_tasks(self, model_settings=None, **kwargs):
        for task, metadata in self.tasks_infos.items():
            if not model_settings:
                model_settings = kwargs if len(kwargs) > 0 else self.model_settings
            self.add_task(task, metadata=metadata, model_settings=model_settings)
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
        self.id = f"{session_id}_{task}"
        self.name = task

        self.neural_metadata, self.behavior_metadata = self.load_metadata(metadata)
        self.data_dir = data_dir or session_dir
        self.neural = Datasets_Neural(
            root_dir=self.data_dir, metadata=self.neural_metadata, task_id=self.id
        )
        self.behavior_metadata = self.fit_behavior_metadata(self.neural)
        self.behavior = Datasets_Behavior(
            root_dir=self.data_dir, metadata=self.behavior_metadata, task_id=self.id
        )
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
        self.neural_metadata["task_id"] = self.id
        self.behavior_metadata["task_id"] = self.id
        return self.neural_metadata, self.behavior_metadata

    def fit_behavior_metadata(self, neural):
        self.behavior_metadata["method"] = neural.metadata["method"]
        self.behavior_metadata["imaging_fps"] = neural.metadata["fps"]
        if "area" in neural.metadata.keys():
            self.behavior_metadata["area"] = neural.metadata["area"]
        return self.behavior_metadata

    def load_data(self, data_source, data_type="neural", regenerate_plot=False):
        # loads neural or behaviour data
        datasets_object = getattr(self, data_type)
        data = datasets_object.load(
            self.id, data_source=data_source, regenerate_plot=regenerate_plot
        )
        return data

    def load_all_data(self, behavior_datas=["position"], regenerate_plots=False):
        """
        neural_Data = ["femtonics", "inscopix"]
        behavior_datas = ["position", "cam"]
        """
        data = {"neural": {}, "behavior": {}}
        for mouse_data in ["neural", "behavior"]:
            if mouse_data == "neural":
                data_to_load = self.neural_metadata["setup"]
            else:
                data_to_load = behavior_datas
            data_to_load = make_list_ifnot(data_to_load)
            for data_source in data_to_load:
                data[mouse_data][data_source] = self.load_data(
                    data_source=data_source,
                    data_type=mouse_data,
                    regenerate_plot=regenerate_plots,
                )
        return data

    def init_models(self, model_settings=None, **kwargs):
        if not model_settings:
            model_settings = kwargs
        self.models = Models(
            self.model_dir, model_id=self.name, model_settings=model_settings
        )

    def get_training_data(
        self,
        datasets_object: Datasets,
        data_types,
        data=None,
        movement_state="all",
        shuffled=False,
        split_ratio=1,
    ):
        # get movement state data
        if movement_state == "all":
            idx_to_keep = None
        elif movement_state == "moving":
            idx_to_keep = self.behavior.moving.data
        elif movement_state == "stationary":
            idx_to_keep = self.behavior.moving.data == False
        else:
            raise ValueError(f"Movement state {movement_state} not supported.")

        # get data
        if not isinstance(data, np.ndarray):
            data, _ = datasets_object.get_multi_data(
                data_types,
                idx_to_keep=idx_to_keep,
                # shuffle=shuffled,
                # split_ratio=split_ratio
            )
        data_filtered = Dataset.filter_by_idx(data, idx_to_keep=idx_to_keep)
        data_shuffled = Dataset.shuffle(data_filtered) if shuffled else data_filtered
        data_train, data_test = Dataset.split(data_shuffled, split_ratio)
        data_train = Dataset.force_2d(data_train)
        # data_train = force_1_dim_larger(data_train)
        return data_train, data_test

    def set_model_name(
        self,
        model_type,
        model_name=None,
        shuffled=False,
        movement_state="all",
        split_ratio=1,
    ):
        if model_name:
            if model_type not in model_name:
                model_name = f"{model_type}_{model_name}"
            if shuffled and "shuffled" not in model_name:
                model_name = f"{model_name}_shuffled"
        else:
            model_name = model_type
            model_name = f"{model_name}_shuffled" if shuffled else model_name

        if movement_state != "all":
            model_name = f"{model_name}_{movement_state}"

        if split_ratio != 1:
            model_name = f"{model_name}_{split_ratio}"
        return model_name

    def get_model(self, models_class, model_name, model_type, model_settings=None):
        # check if model with given model_settings is available
        model_available = False
        if model_name in models_class.models.keys():
            model_available = True
            model = models_class.models[model_name]
            model_parameter = model.get_params()
            if model_settings:
                for (
                    model_setting_name,
                    model_setting_value,
                ) in model_settings.items():
                    if model_parameter[model_setting_name] != model_setting_value:
                        model_available = False
                        break

        if not model_available:
            model_creation_function = getattr(models_class, model_type)
            model = model_creation_function(
                name=model_name, model_settings=model_settings
            )
        return model

    def train_model(
        self,
        model_type: str,  # types: time, behavior, hybrid
        regenerate: bool = False,
        shuffled: bool = False,
        movement_state: str = "all",
        split_ratio: float = 1,
        model_name: str = None,
        neural_data: np.ndarray = None,
        behavior_data: np.ndarray = None,
        neural_data_types: List[str] = None,  # ["femtonics"],
        behavior_data_types: List[str] = None,  # ["position"],
        manifolds_pipeline: str = "cebra",
        model_settings: dict = None,
    ):
        """
        available model_types are: time, behavior, hybrid
        """
        model_name = self.set_model_name(
            model_type, model_name, shuffled, movement_state, split_ratio
        )

        # TODO: add other manifolds pipelines
        if manifolds_pipeline == "cebra":
            models_class = self.models.cebras
        else:
            raise ValueError(f"Manifolds pipeline {manifolds_pipeline} not supported.")

        model = self.get_model(
            models_class=models_class,
            model_name=model_name,
            model_type=model_type,
            model_settings=model_settings,
        )

        # get neural data
        neural_data_types = (
            neural_data_types or self.neural_metadata["preprocessing_software"]
        )
        neural_data_train, neural_data_test = self.get_training_data(
            datasets_object=self.neural,
            data_types=self.neural_metadata[
                "preprocessing_software"
            ],  # e.g. ["femtonics"], iscopix
            data=neural_data,
            movement_state=movement_state,
            shuffled=shuffled,
            split_ratio=split_ratio,
        )

        # get behavior data
        if behavior_data_types:
            behavior_data_train, behavior_data_test = self.get_training_data(
                datasets_object=self.behavior,
                data_types=behavior_data_types,  # e.g. ["position", "cam"]
                data=behavior_data,
                movement_state=movement_state,
                shuffled=shuffled,
                split_ratio=split_ratio,
            )

        if not model.fitted or regenerate:
            # skip if no neural data available
            if neural_data_train.shape[0] == 0:
                global_logger.error(f"No neural data to use. Skipping {self.id}")
                print(f"Skipping {self.id}: No neural data given.")
            else:
                # train model
                global_logger.info(f"{self.id}: Training  {model.name} model.")
                print(f"{self.id}: Training  {model.name} model.")
                if model_type == "time":
                    model.fit(neural_data_train)
                else:
                    if not behavior_data_types:
                        raise ValueError(
                            f"No behavior data types given for {model_type} model."
                        )
                    neural_data_train, behavior_data_train = (
                        Datasets.force_equal_dimensions(
                            neural_data_train, behavior_data_train
                        )
                    )
                    model.fit(neural_data_train, behavior_data_train)
                model.fitted = models_class.fitted(model)
                model.save(model.save_path)
        else:

            global_logger.info(
                f"{self.id}: {model.name} model already trained. Skipping."
            )
            print(f"{self.id}: {model.name} model already trained. Skipping.")

        # TODO: add saving test dataset somewhere and maybe provide decoded results or decode directly.....

        if split_ratio != 1:
            return (
                model,
                neural_data_train,
                neural_data_test,
                behavior_data_train,
                behavior_data_test,
            )
        return model

    def create_embeddings(
        self,
        models=None,
        to_transform_data=None,
        model_naming_filter_include: List[List[str]] = None,  # or [str] or str
        model_naming_filter_exclude: List[List[str]] = None,  # or [str] or str
        manifolds_pipeline="cebra",
    ):
        # TODO: create function to integrate train and test embeddings into models
        if not type(to_transform_data) == np.ndarray:
            global_logger.warning(f"No data to transform given. Using default labels.")
            print(f"No neural data given. Using default labels. photon")
            to_transform_data = self.neural.photon.data
            if not isinstance(to_transform_data, np.ndarray):
                raise ValueError("No data to neural transform given.")

        filtered_models = models or self.get_pipeline_models(
            manifolds_pipeline, model_naming_filter_include, model_naming_filter_exclude
        )
        embeddings = {}
        for model_name, model in filtered_models.items():
            embedding_title = f"{model_name} - {model.max_iterations}"
            if model.fitted:
                embedding = model.transform(to_transform_data)
                self.embeddings[embedding_title] = embedding
                embeddings[embedding_title] = embedding
            else:
                global_logger.error(f"{model_name} model. Not fitted.")
                global_logger.warning(
                    f"Skipping {model_name} model. May because of no statioary frames"
                )
                print(
                    f"Skipping {model_name} model. May because of no statioary frames"
                )
        return embeddings

    def plot_model_embeddings(
        self,
        model_naming_filter_include: List[List[str]] = None,  # or [str] or str
        model_naming_filter_exclude: List[List[str]] = None,  # or [str] or str
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        to_transform_data: Optional[np.ndarray] = None,
        embedding_labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        behavior_data_types: List[str] = ["position"],
        manifolds_pipeline: str = "cebra",
        set_title: Optional[str] = None,
        title_comment: Optional[str] = None,
        markersize: float = 0.05,
        alpha: float = 0.4,
        dpi: int = 300,
    ):
        if not embeddings:
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
            global_logger.error("No embedding labels given.")
            global_logger.warning(f"Using behavior_data_types: {behavior_data_types}")
            print(f"Using behavior_data_types: {behavior_data_types}")
            behavior_datas, _ = self.behavior.get_multi_data(behavior_data_types)
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
                    + f" - {embedding_title}{' '+str(title_comment) if title_comment else ''}"
                )
            labels_dict = {"name": embedding_title, "labels": embedding_labels}
            viz.plot_multiple_embeddings(
                embeddings,
                labels=labels_dict,
                title=title,
                markersize=markersize,
                alpha=alpha,
                dpi=dpi,
            )
        return embeddings

    def get_pipeline_models(
        self,
        manifolds_pipeline="cebra",
        model_naming_filter_include: List[List[str]] = None,  # or [str] or str
        model_naming_filter_exclude: List[List[str]] = None,  # or [str] or str
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
        model_naming_filter_include: List[List[str]] = None,  # or [str] or str
        model_naming_filter_exclude: List[List[str]] = None,  # or [str] or str
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
        num_iterations=None,
        plot_model_iterations=False,
        model_naming_filter_include: List[List[str]] = None,  # or [str] or str
        model_naming_filter_exclude: List[List[str]] = None,  # or [str] or str
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
        num_iterations = (
            models_original[0].max_iterations if not num_iterations else num_iterations
        )
        title = title or f"Losses {self.session_id} {stimulus_type}"
        title += f" - {num_iterations} Iterartions" if not plot_model_iterations else ""

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

    def get_model(self):
        # TODO: move get_model function from task class
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
        model_naming_filter_include: List[List[str]] = None,  # or [str] or str
        model_naming_filter_exclude: List[List[str]] = None,  # or [str] or str):
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

    def init_model(self, model_settings_dict):
        default_model = self.create_defaul_model()
        if len(model_settings_dict) == 0:
            model_settings_dict = self.model_settings
        initial_model = define_cls_attributes(
            default_model, model_settings_dict, override=True
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
                global_logger.info(f"Loaded matching model {fitted_model_path}")
                print(f"Loaded matching model {fitted_model_path}")
            else:
                global_logger.error(
                    f"Loaded model parameters do not match to initialized model. Not loading {fitted_model_path}"
                )
        model.fitted = self.fitted(model)
        return model

    def model_settings_start(self, name, model_settings_dict):
        model = self.init_model(model_settings_dict)
        model.name = name
        return model

    def model_settings_end(self, model):
        model.save_path = self.define_parameter_save_path(model)
        model = self.load_fitted_model(model)
        self.models[model.name] = model
        return model

    def time(self, name="time", model_settings=None, **kwargs):
        model_settings = model_settings or kwargs
        model = self.model_settings_start(name, model_settings)
        model.temperature = (
            1.12 if kwargs.get("temperature") is None else model.temperature
        )
        model.conditional = "time" if kwargs.get("time") is None else model.conditional
        model = self.model_settings_end(model)
        return model

    def behavior(self, name="behavior", model_settings=None, **kwargs):
        model_settings = model_settings or kwargs
        model = self.model_settings_start(name, model_settings)
        model = self.model_settings_end(model)
        return model

    def hybrid(self, name="hybrid", model_settings=None, **kwargs):
        model_settings = model_settings or kwargs
        model = self.model_settings_start(name, model_settings)
        model.hybrid = True if kwargs.get("hybrid") is None else model.hybrid
        model = self.model_settings_end(model)
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
        markersize=0.05,
        alpha=0.4,
        figsize=(10, 10),
        dpi=300,
    ):
        embedding, labels = Datasets.force_equal_dimensions(
            embedding, embedding_labels["labels"]
        )

        ax = cebra.plot_embedding(
            ax=ax,
            embedding=embedding,
            embedding_labels=labels,
            markersize=markersize,
            alpha=alpha,
            dpi=dpi,
            title=title,
            figsize=figsize,
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
        markersize=0.05,
        alpha=0.4,
        dpi=300,
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
                ax,
                embedding,
                labels,
                subplot_title,
                cmap,
                plot_legend=False,
                markersize=markersize,
                alpha=alpha,
                dpi=dpi,
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
            if model.fitted:
                ax = cebra.plot_loss(
                    model, color=color, alpha=alpha, label=label, ax=ax
                )
            else:
                global_logger.error(f"{label} Not fitted.")
                global_logger.warning(f"Skipping model {label}.")
                print(f"Skipping model {label}.")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("InfoNCE Loss")
        plt.legend(bbox_to_anchor=(0.5, 0.3), frameon=False)
        self.plot_ending(title)

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

    def plot_histogram(self, data, title, bins=100, figsize=(10, 5)):
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.hist(data, bins=bins)
        plt.show()
        plt.close()

    def histogam_subplot(
        self,
        data: np.ndarray,
        title: str,
        ax,
        bins=100,
        xlim=[0, 1],
        xlabel="",
        ylabel="Frequency",
        xticklabels=None,
        color=None,
    ):
        ax.set_title(title)
        ax.hist(data.flatten(), bins=bins, color=color)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xticklabels == "empty":
            ax.set_xticklabels("")

    def heatmap_subplot(
        self,
        matrix,
        title,
        ax,
        sort=False,
        xlabel="Cell ID",
        ylabel="Cell ID",
        cmap="YlGnBu",
    ):
        if sort:
            # Assuming correlations is your correlation matrix as a NumPy array
            # Convert it to a Pandas DataFrame
            correlations_df = pd.DataFrame(matrix)
            # sort the correlation matrix
            matrix = correlations_df.sort_values(by=0, axis=1, ascending=False)

        # Creating a heatmap with sort correlations
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        sns.heatmap(matrix, annot=False, cmap=cmap, ax=ax)

    def plot_corr_hist_heat_salience(
        self,
        correlation: np.ndarray,
        saliences,
        title: str,
        bins: int = 100,
        sort=False,
        figsize=(17, 5),
    ):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        self.histogam_subplot(
            correlation,
            "Correlation",
            ax1,
            bins=bins,
            xlim=[-1, 1],
            xlabel="Correlation Value",
            ylabel="Frequency",
        )
        self.heatmap_subplot(correlation, "Correlation Heatmap", ax2, sort=sort)
        self.histogam_subplot(
            saliences,
            "Saliences",
            ax3,
            xlim=[0, 2],
            bins=bins,
            xlabel="n",
            ylabel="Frequency",
        )
        self.plot_ending(title, save=True)

    def plot_dist_sal_dims(
        self, distances, saliences, normalized_saliences, title, bins=100
    ):
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(17, 10))
        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]
        title = title + " Histograms"

        self.histogam_subplot(
            distances,
            "Distance from Origin",
            ax1,
            bins=bins,
            color=colors[0],
            xlim=[0, 2],
            xticklabels="empty",
        )
        self.histogam_subplot(
            saliences,
            "Normalized Distances",
            ax2,
            bins=bins,
            color=colors[1],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences[:, 0],
            "normalized X",
            ax3,
            bins=bins,
            color=colors[2],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences[:, 1],
            "normalized Y",
            ax4,
            bins=bins,
            color=colors[3],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences[:, 2], "normalized Z", ax5, bins=bins, color=colors[4]
        )
        self.plot_ending(title, save=True)

    def plot_dist_sal_dims_2(
        self,
        distances,
        saliences,
        normalized_saliences,
        distances2,
        saliences2,
        normalized_saliences2,
        title,
        bins=100,
    ):
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 2, figsize=(17, 10))
        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]
        title = title + " Histograms"

        self.histogam_subplot(
            distances,
            "Distance from Origin",
            ax1[0],
            bins=bins,
            color=colors[0],
            xlim=[0, 2],
        )
        self.histogam_subplot(
            saliences,
            "Normalized Distances",
            ax2[0],
            bins=bins,
            color=colors[1],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences[:, 0],
            "normalized X",
            ax3[0],
            bins=bins,
            color=colors[2],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences[:, 1],
            "normalized Y",
            ax4[0],
            bins=bins,
            color=colors[3],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences[:, 2],
            "normalized Z",
            ax5[0],
            bins=bins,
            color=colors[4],
        )

        self.histogam_subplot(
            distances2,
            "Distance from Origin",
            ax1[1],
            bins=bins,
            color=colors[0],
            xlim=[0, 2],
        )
        self.histogam_subplot(
            saliences2,
            "Normalized Distances",
            ax2[1],
            bins=bins,
            color=colors[1],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences2[:, 0],
            "normalized X",
            ax3[1],
            bins=bins,
            color=colors[2],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences2[:, 1],
            "normalized Y",
            ax4[1],
            bins=bins,
            color=colors[3],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences2[:, 2],
            "normalized Z",
            ax5[1],
            bins=bins,
            color=colors[4],
        )
        self.plot_ending(title, save=True)

    def plot_corr_heat_corr_heat(
        self, correlation1, correlation2, title1, title2, sort=False, figsize=(17, 5)
    ):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
        title = title1 + " vs " + title2
        self.histogam_subplot(
            correlation1,
            title1 + " Correlation",
            ax1,
            bins=100,
            xlim=[-1, 1],
            xlabel="Correlation Value",
            ylabel="Frequency",
            color="tab:blue",
        )
        self.heatmap_subplot(correlation1, title1, ax2, sort=sort)
        self.histogam_subplot(
            correlation2,
            title2 + " Correlation",
            ax3,
            bins=100,
            xlim=[-1, 1],
            xlabel="Correlation Value",
            ylabel="Frequency",
            color="tab:orange",
        )
        self.heatmap_subplot(correlation2, title2, ax4, sort=sort)
        self.plot_ending(title, save=True)

    def plot_ending(self, title, save=True):
        plt.suptitle(title)
        plt.tight_layout()  # Ensure subplots fit within figure area
        plot_save_path = (
            str(self.save_dir.joinpath(title + ".png"))
            .replace(">", "bigger")
            .replace("<", "smaller")
        )
        if save:
            plt.savefig(plot_save_path, dpi=300)
        plt.show()
        plt.close()
