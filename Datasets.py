import os
from pathlib import Path

# type hints
from typing import List, Union, Dict, Any, Tuple, Optional

# plotting
import matplotlib.pyplot as plt
from Visualizer import Vizualizer

# calculations
import numpy as np
from scipy.signal import butter, filtfilt  # , lfilter, freqz

# load data
import cebra

# own
from Helper import *

## Imaging Setups
from Setups import Femtonics, Thorlabs, Inscopix, Environment

## Experimental Setups
from Setups import (
    Active_Avoidance,
    Treadmill_Setup,
    Trackball_Setup,
    Openfield_Setup,
    Wheel_Setup,
)


class Dataset:
    def __init__(
        self,
        key,
        metadata,
        path=None,
        data=None,
        raw_data_object=None,
        root_dir=None,
        task_id=None,
    ):
        # initialize plotting parameters
        super().__init__()
        self.root_dir: Path = root_dir
        self.path: Path = path
        self.data_dir = None
        self.data: np.ndarray = data
        self.binned_data: np.ndarray = None
        self.key = key
        self.task_id = task_id
        self.raw_data_object = raw_data_object
        self.metadata = metadata
        self.fps = None if "fps" not in self.metadata.keys() else self.metadata["fps"]
        check_needed_keys(metadata, ["setup"])
        self.setup = self.get_setup(self.metadata["setup"])
        self.plot_attributes = Vizualizer.default_plot_attributes()

    def get_setup(self, setup_name, preprocessing_name, method_name):
        raise NotImplementedError(
            f"ERROR: Function get_setup is not defined for {self.__class__}."
        )

    def load(
        self,
        path=None,
        save=True,
        plot=True,
        regenerate=False,
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
                if self.data.ndim == 2:
                    # format of data should be [num_time_points, num_cells]
                    self.data = force_1_dim_larger(data=self.data)
            else:
                global_logger.warning(
                    f"Generating data based on raw_data_object for {self.key} in {self.path}."
                )
                print(
                    f"Generating data based on raw_data_object for {self.key} in {self.path}."
                )
                self.create_dataset(self.raw_data_object, save=save)
                if save and not self.path.exists():
                    np.save(self.path, self.data)
        self.data = self.correct_data(self.data)
        if plot:
            self.plot(regenerate_plot=regenerate_plot)
        self.binned_data = self.bin_data(self.data)
        return self.data

    def create_dataset(self, raw_data_object=None, save=True):
        raw_data_object = raw_data_object or self.raw_data_object
        if self.raw_data_object:
            global_logger.info(
                f"Creating {self.key} dataset based on raw data from {raw_data_object.key}."
            )
            print(
                f"Creating {self.key} dataset based on raw data from {raw_data_object.key}."
            )
            data = self.process_raw_data(save=save)
            return data
        else:
            global_logger.warning(
                f"No raw data given. Creation not possible. Skipping."
            )
            print(f"No raw data given. Creation not possible. Skipping.")

    def process_raw_data(self, save=True):
        global_logger.critical(
            f"Function for creating {self.key} dataset from raw data is not defined."
        )
        raise NotImplementedError(
            f"ERROR: Function for creating {self.key} dataset from raw data is not defined."
        )

    def correct_data(self, data):
        return data

    def bin_data(self, data, bin_size=None):
        """
        binning is not applied to this type of data
        """
        return data

    def refine_plot_attributes(
        self,
        title=None,
        ylable=None,
        xlimits=None,
        save_path=None,
    ):
        self.plot_attributes["title"] = (
            self.plot_attributes["title"] or title or f"{self.path.stem} data"
        )
        self.plot_attributes["title"] = (
            self.plot_attributes["title"]
            if self.plot_attributes["title"][-4:] == "data"
            else self.plot_attributes["title"] + " data"
        )

        self.plot_attributes["ylable"] = (
            self.plot_attributes["ylable"] or ylable or self.key
        )

        self.plot_attributes["xlimits"] = (
            self.plot_attributes["xlimits"] or xlimits or (0, len(self.data))
        )

        descriptive_metadata_keys = [
            "area",
            "stimulus_type",
            "method",
            "preprocessing_software",
            "setup",
        ]

        descriptive_metadata_txt = get_str_from_dict(
            dictionary=self.metadata, keys=descriptive_metadata_keys
        )
        if descriptive_metadata_txt not in self.plot_attributes["title"]:
            self.plot_attributes["title"] += f" {descriptive_metadata_txt}"

        self.plot_attributes["save_path"] = (
            self.plot_attributes["save_path"]
            or save_path
            or self.path.parent.parent.parent.joinpath(
                "figures", self.path.stem + ".png"
            )
        )

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
        save_path=None,
        regenerate_plot=None,
        show=False,
        dpi=300,
    ):
        self.refine_plot_attributes(
            title=title, ylable=ylable, xlimits=xlimits, save_path=save_path
        )
        if regenerate_plot or not save_file_present(
            fpath=self.plot_attributes["save_path"],
        ):
            self.plot_attributes = Vizualizer.default_plot_start(
                plot_attributes=self.plot_attributes,
                figsize=figsize,
                xlable=xlable,
                xlimits=xlimits,
                xticks=xticks,
                ylimits=ylimits,
                yticks=yticks,
                num_ticks=num_ticks,
                fps=fps,
            )
            self.plot_data()
            Vizualizer.default_plot_ending(
                plot_attributes=self.plot_attributes,
                regenerate_plot=regenerate_plot,
                show=show,
                dpi=dpi,
            )
        else:
            Vizualizer.plot_image(
                plot_attributes=self.plot_attributes,
                show=show,
            )

    def plot_data(self):
        if self.data.ndim == 1:
            Vizualizer.data_plot_1D(
                data=self.data,
                plot_attributes=self.plot_attributes,
            )
        elif self.data.ndim == 2:
            if self.key != "position":
                raw_data_object = self.raw_data_object
                while raw_data_object is not None:
                    if raw_data_object.key == "position":
                        position_data = raw_data_object.data
                        break
                    else:
                        raw_data_object = raw_data_object.raw_data_object
            else:
                position_data = self.data
            Vizualizer.data_plot_2D(
                data=self.data,
                position_data=position_data,
                border_limits=self.metadata["environment_dimensions"],
                plot_attributes=self.plot_attributes,
            )

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
            data_unfiltered, idx_to_keep = force_equal_dimensions(data, idx_to_keep)
            data_filtered = data_unfiltered[idx_to_keep]
            return data_filtered
        else:
            global_logger
            print(f"No idx_to_keep given. Returning unfiltered data.")
            return data

    @staticmethod
    def force_2d(data: np.ndarray, transepose=True):
        data_2d = data
        if data_2d.ndim == 1:
            data_2d = data_2d.reshape(1, -1)
            if transepose:
                data_2d = data_2d.T
        elif data_2d.ndim > 2:
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
            key=key,
            path=path,
            data=data,
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        needed_attributes = ["setup"]
        check_needed_keys(metadata, needed_attributes)
        self.plot_attributes["fps"] = (
            self.metadata["imaging_fps"]
            if "imaging_fps" in self.metadata.keys()
            else self.metadata["fps"]
        )
        # default binning size is 1cm
        self.binning_size = None

    def get_setup(self, setup_name):
        self.setup_name = self.metadata["setup"]
        if setup_name == "active_avoidance":
            setup = Active_Avoidance(
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
        elif setup_name == "wheel":
            setup = Wheel_Setup(
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
            key=key,
            path=path,
            data=data,
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        needed_attributes = ["method", "preprocessing", "processing", "setup"]
        check_needed_keys(metadata, needed_attributes)

    def create_dataset(self, raw_data_object=None, save=True):
        self.data = self.process_raw_data(save=save)
        return


class Data_Position(BehaviorDataset):
    """
    Dataset class for managing position data.
    Attributes:
        - data (np.ndarray): The position data.
        - binned_data (np.ndarray): The binned position data into default: 1cm bins.
        - raw_position (np.ndarray): The raw position data.
        - lap_starts (np.ndarray): The indices of the lap starts.
        - environment_dimensions (List): The dimensions of the environment.
    """

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
        self.metadata["environment_dimensions"] = make_list_ifnot(
            self.metadata["environment_dimensions"]
        )
        if not "binning_size" in self.metadata.keys():
            self.binning_size = (
                0.01  # meters in 1D
                if len(self.metadata["environment_dimensions"]) == 1
                else 0.05  # meters in 2D
            )
        else:
            self.binning_size = self.metadata["binning_size"]
        self.max_bin = None
        self.raw_position = None
        self.lap_starts = None
        self.define_plot_attributes()

    def define_plot_attributes(self):
        if len(self.metadata["environment_dimensions"]) == 1:
            self.plot_attributes["ylable"] = "position cm"
        elif len(self.metadata["environment_dimensions"]) == 2:
            self.plot_attributes["figsize"] = (11, 10)

    def plot_data(self):
        if self.data.ndim == 1:
            marker = "^" if self.lap_starts is not None else None
            Vizualizer.data_plot_1D(
                data=self.data,
                plot_attributes=self.plot_attributes,
                marker_pos=self.lap_starts,
                marker=marker,
            )
        elif self.data.ndim == 2:
            Vizualizer.data_plot_2D(
                data=self.data,
                position_data=self.data,
                # border_limits=self.metadata["environment_dimensions"],
                plot_attributes=self.plot_attributes,
                color_by="time",
            )

    def plot_local_env_shape_types(self):
        # TODO: move to Stimulus class
        stimulus = Environment.get_env_shapes_from_pos(self.data)
        Vizualizer.data_plot_2D(
            data=stimulus,
            position_data=self.data,
            # border_limits=self.metadata["environment_dimensions"],
            plot_attributes=self.plot_attributes,
        )

    def bin_data(self, data=None, bin_size=None):
        """
        Bin the position data into 1cm bins in 1D and 5cm^2 bins for 2D environments.
        Args:
            - data (np.ndarray): The position data.
            - bin_size (float): The size of the bins in meters.
        return: binned_data (np.ndarray): The binned position data into cm bins (default 1cm).
        """
        if data is None:
            data = self.data
        if bin_size is None:
            bin_size = self.binning_size
        if self.binned_data is not None and bin_size == self.binning_size:
            return self.binned_data
        dimensions = self.metadata["environment_dimensions"]
        binned_data = bin_array(data, bin_size=bin_size, min_bin=0, max_bin=dimensions)
        self.max_bin = np.array(dimensions) / np.array(bin_size)
        return binned_data

    def create_dataset(self, raw_data_object=None, save=True):
        data = self.process_raw_data(save=save)
        return data

    def process_raw_data(self, save=True):
        self.data = self.setup.process_data(save=save)
        return self.data

    def get_lap_starts(self, fps=None, sec_thr=5):
        # TODO: This is only working for 1D (laps)
        if self.lap_starts and sec_thr == 5:
            return self.lap_starts
        fps = fps or self.metadata["imaging_fps"]
        # get index of beeing at lap start
        at_start_indices = np.where(self.bin_data() == 0)[0]
        # take first indices in case multiple frames are at start
        num_frames_threshold = fps * sec_thr
        start_indices = [at_start_indices[0]]
        old_idx = at_start_indices[0]
        for index in at_start_indices[1:]:
            if index - old_idx > num_frames_threshold:
                start_indices.append(index)
                old_idx = index
        return np.array(start_indices)


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
        optional_attributes = [
            "stimulus_dimensions",
            "stimulus_sequence",
            "stimulus_type",
            "stimulus_by",
        ]
        add_missing_keys(metadata, optional_attributes, fill_value=None)

        self.stimulus_sequence = self.metadata["stimulus_sequence"]
        self.stimulus_dimensions = self.metadata["stimulus_dimensions"]
        self.stimulus_type = self.metadata["stimulus_type"]
        self.stimulus_by = self.metadata["stimulus_by"]
        self.fps = self.metadata["fps"] if "fps" in self.metadata.keys() else None

    def process_raw_data(self, save=True):
        """ "
        Returns:
            - data: Numpy array composed of stimulus type at frames.
        """
        stimulus_raw_data = self.raw_data_object.data  # e.g. Position on a track/time
        if self.stimulus_by == "location":
            stimulus_type_at_frame = Environment.get_stimulus_at_position(
                positions=stimulus_raw_data,
                stimulus_dimensions=self.stimulus_dimensions,
                stimulus_sequence=self.stimulus_sequence,
                max_position=self.stimulus_dimensions,
            )
        elif self.stimulus_by == "frames":
            stimulus_type_at_frame = self.stimulus_by_time(stimulus_raw_data)
        elif self.stimulus_by == "seconds":
            stimulus_type_at_frame = self.stimulus_by_time(
                stimulus_raw_data,
                time_to_frame_multiplier=self.self.metadata["imaging_fps"],
            )
        elif self.stimulus_by == "minutes":
            stimulus_type_at_frame = self.stimulus_by_time(
                stimulus_raw_data,
                time_to_frame_multiplier=60 * self.self.metadata["imaging_fps"],
            )
        self.data = stimulus_type_at_frame
        return self.data

    def stimulus_by_time(self, stimulus_raw_data, time_to_frame_multiplier=1):
        """
        time in frames in an experiment
        """
        stimulus_type_at_frame = []
        for stimulus, duration in zip(self.stimulus_sequence, self.stimulus_dimensions):
            stimulus_type_at_frame += [stimulus] * int(
                duration * time_to_frame_multiplier
            )
        raise NotImplementedError("Stimulus by time may not implemented. Check results")
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
        self.metadata["environment_dimensions"] = make_list_ifnot(
            self.metadata["environment_dimensions"]
        )
        self.binning_size = (
            0.01
            if not "binning_size" in self.metadata.keys()
            else self.metadata["binning_size"]
        )
        self.define_plot_attributes()

    def define_plot_attributes(self):
        self.plot_attributes["ylable"] = "distance in m"

    def plot_data(self):
        absolute_distance = np.linalg.norm(self.data, axis=1)
        Vizualizer.data_plot_1D(
            data=absolute_distance,
            plot_attributes=self.plot_attributes,
        )

    def bin_data(self, data, bin_size=None):
        bin_size = bin_size or self.binning_size
        binned_data = bin_array(data, bin_size=bin_size, min_bin=0)
        return binned_data

    def process_raw_data(self, save=True):
        track_positions = self.raw_data_object.data
        self.data = Environment.get_cumdist_from_position(track_positions)
        return self.data


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
        self.raw_velocity = None
        self.binning_size = 0.005  # 0.005m/s
        self.define_plot_attributes()

    def define_plot_attributes(self):
        self.plot_attributes["ylable"] = "velocity m/s"
        if len(self.metadata["environment_dimensions"]) == 2:
            self.plot_attributes["figsize"] = (10, 10)

    def bin_data(self, data, bin_size=None):
        bin_size = bin_size or self.binning_size
        binned_data = bin_array(data, bin_size=bin_size, min_bin=0)
        return binned_data

    def process_raw_data(self, save=True):
        """
        calculating velocity based on velocity data in raw_data_object
        """
        raw_data_type = self.raw_data_object.key
        data = self.raw_data_object.data
        if raw_data_type == "distance":
            walked_distances = (
                self.raw_data_object.process_raw_data(save=save)
                if data is None
                else data
            )
        elif raw_data_type == "position":
            walked_distances = Environment.get_distance_from_position(data)
        else:
            raise ValueError(f"Raw data type {raw_data_type} not supported.")
        self.raw_velocity = Environment.get_velocity_from_cumdist(
            walked_distances, imaging_fps=self.metadata["imaging_fps"], smooth=False
        )
        velocity_smoothed = may_butter_lowpass_filter(
            self.raw_velocity,
            smooth=True,
            cutoff=2,
            fps=self.metadata["imaging_fps"],
            order=2,
        )
        # velocity_smoothed = moving_average(self.raw_velocity)
        global_logger
        print(
            f"Calculating smooth velocity based on butter_lowpass_filter 2Hz, {self.metadata['imaging_fps']}fps, 2nd order."
        )
        # change value/frame to value/second
        self.data = velocity_smoothed
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
        self.binning_size = 0.0005  # 0.0005m/s^2
        self.define_plot_attributes()

    def define_plot_attributes(self):
        self.plot_attributes["ylable"] = "acceleration m/s^2"
        if len(self.metadata["environment_dimensions"]) == 2:
            self.plot_attributes["figsize"] = (10, 10)

    def bin_data(self, data, bin_size=None):
        bin_size = bin_size or self.binning_size
        binned_data = bin_array(data, bin_size=bin_size, min_bin=0)
        return binned_data

    def process_raw_data(self, save=True):
        """
        calculating acceleration based on velocity data in raw_data_object
        """
        velocity = self.raw_data_object.data
        self.raw_acceleration = Environment.get_acceleration_from_velocity(
            velocity, smooth=False
        )
        smoothed_acceleration = may_butter_lowpass_filter(
            self.raw_acceleration,
            cutoff=2,
            fps=self.metadata["imaging_fps"],
            order=2,
        )
        global_logger
        print(
            f"Calculating smooth acceleration based on butter_lowpass_filter 2Hz, {self.metadata['imaging_fps']}fps, 2nd order."
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
        self.velocity_threshold = 0.02  # m/s
        self.brain_processing_delay = {
            "CA1": 2,  # seconds
            "CA3": 2,
            "M1": None,
            "S1": None,
            "V1": None,
        }
        needed_attributes = ["setup"]
        check_needed_keys(metadata, needed_attributes)
        self.define_plot_attributes()

    def define_plot_attributes(self):
        self.plot_attributes["ylable"] = "Movement State"
        self.plot_attributes["ylimits"] = (-0.1, 1.1)
        self.plot_attributes["yticks"] = [[0, 1], ["Stationary", "Moving"]]

    def process_raw_data(self, save=True):
        velocities = self.raw_data_object.data
        velocities_abs = np.linalg.norm(np.abs(velocities), axis=1)
        moving_frames = velocities_abs > self.velocity_threshold
        self.data = self.fit_moving_to_brainarea(moving_frames, self.metadata["area"])
        return self.data

    def fit_moving_to_brainarea(self, data, area):
        processing_movement_lag = self.brain_processing_delay[area]  # seconds
        if not processing_movement_lag:
            raise NotImplementedError(f"{area} processing lag not implemented.")
        processing_movement_frames = fill_continuous_array(
            data, fps=self.metadata["imaging_fps"], time_gap=processing_movement_lag
        )
        return processing_movement_frames

    def get_idx_to_keep(self, movement_state):
        if movement_state == "all":
            idx_to_keep = None
        elif movement_state == "moving":
            idx_to_keep = self.data
        elif movement_state == "stationary":
            idx_to_keep = self.data == False
        else:
            raise ValueError(f"Movement state {movement_state} not supported.")
        return idx_to_keep

    def correct_data(self, data):
        return self.fit_moving_to_brainarea(data, self.metadata["area"])


class Data_Photon(NeuralDataset):
    """
    Dataset class for managing photon data.
    Attributes:
        - data (np.ndarray): The binariced traces.
        - method (str): The method used to record the data.
        - preprocessing_software (str): The software used to preprocess the data.
            - raw fluoresence traces can be found inside preprocessing object
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

        self.define_plot_attributes()

    def define_plot_attributes(self):
        self.plot_attributes["ylable"] = "Neuron ID"
        self.plot_attributes["figsize"] = (20, 10)
        if "fps" not in self.metadata.keys():
            self.metadata["fps"] = self.setup.get_fps()
            self.plot_attributes["fps"] = self.metadata["fps"]

    def process_raw_data(self, save=True):
        self.data = self.setup.process_data(save=save)
        return self.data

    def plot_data(self):
        Vizualizer.plot_neural_activity_raster(
            self.data,
            fps=self.fps,
            num_ticks=self.plot_attributes["num_ticks"],
        )

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

    def load(self, data_source, regenerate=False, regenerate_plot=False):
        """
        Load data from a specific data source.
            data_object is a object inhereting the Dataset attributes and functions.
        """
        # build path to data
        data_object = self.get_object(data_source)
        fpath = data_object.setup.get_data_path()
        data = data_object.load(
            fpath, regenerate=regenerate, regenerate_plot=regenerate_plot
        )
        return data

    def get_multi_data(
        self, sources, shuffle=False, idx_to_keep=None, split_ratio=1, binned=True
    ):
        sources = make_list_ifnot(sources)
        concatenated_data = None
        for source in sources:
            dataset_object = getattr(self, source)
            data = dataset_object.data
            # data = dataset_object.binned_data if binned else dataset_object.data
            data = np.array([data]).transpose() if len(data.shape) == 1 else data
            if type(concatenated_data) != np.ndarray:
                concatenated_data = data
            else:
                concatenated_data, data = force_equal_dimensions(
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


class Datasets_Neural(Datasets):
    def __init__(self, root_dir, metadata={}, task_id=None):
        super().__init__(root_dir=root_dir, metadata=metadata, task_id=task_id)
        self.photon_imaging_methods = ["femtonics", "thorlabs", "inscopix"]
        self.probe_imaging_methods = ["neuropixels", "tetrode"]
        # TODO: split into different datasets if needed
        self.imaging_type = self.define_imaging_type(self.metadata["setup"])
        if self.imaging_type == "photon":
            self.photon = Data_Photon(
                root_dir=root_dir, metadata=self.metadata, task_id=self.task_id
            )
        else:
            # TODO: implement probe data loading
            self.probe = Data_Probe(
                root_dir=root_dir, metadata=self.metadata, task_id=self.task_id
            )

        if "fps" not in self.metadata.keys():
            # TODO: this is not implemented for probe data
            data_object = self.get_object(self.metadata["setup"])  # photon or probe
            self.metadata["fps"] = data_object.metadata["fps"]

    def define_imaging_type(self, data_source):
        if data_source in self.photon_imaging_methods:
            imaging_type = "photon"
        elif data_source in self.probe_imaging_methods:
            imaging_type = "probe"
        else:
            raise ValueError(f"Imaging type {data_source} not supported.")
        return imaging_type

    def get_object(self, data_source):
        imaging_type = self.define_imaging_type(data_source)
        data_object = getattr(self, imaging_type)
        return data_object


class Datasets_Behavior(Datasets):
    def __init__(self, root_dir, metadata={}, task_id=None):
        super().__init__(root_dir=root_dir, metadata=metadata, task_id=task_id)
        self.data_dir = (
            self.data_dir
            if not self.data_dir == root_dir
            else self.data_dir.joinpath(f"TRD-{self.metadata['method']}")
        )
        self.position = Data_Position(
            root_dir=root_dir, metadata=self.metadata, task_id=self.task_id
        )
        self.distance = Data_Distance(
            raw_data_object=self.position,
            root_dir=root_dir,
            metadata=self.metadata,
            task_id=self.task_id,
        )
        self.stimulus = Data_Stimulus(
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

    def split_by_laps(self, data=None, lap_starts=None, fps=None, sec_thr=5):
        """
        Splits data into individual laps based on lap start indices or calculates lap start indices using the provided parameters.

        Parameters:
            - data (optional): The data to be split into laps. This can be either 1D or 2D data.
            - lap_starts (optional): Indices indicating the start of each lap in the data.
            - fps (optional): Frames per second. Required if lap_starts is not provided and data is in video format.
            - sec_thr (optional): Threshold in seconds to consider as the minimum lap duration when calculating lap starts.

        Returns:
            - laps: A list containing each individual lap of the data.

        If lap_starts is not provided, lap start indices are calculated using the given fps and sec_thr parameters.
        If data is not provided, it will be expected to be accessed from the instance attribute 'position'.

        Note:
        - 1D data: If the data is 1-dimensional (e.g., time series data), lap_starts must be provided.
        - 2D data: If the data is 2-dimensional (e.g., time series data for multiple neurons).
        ```
        """
        lap_starts = lap_starts or self.position.get_lap_starts(
            fps=fps, sec_thr=sec_thr
        )
        laps = []
        for start, end in zip(lap_starts[:-1], lap_starts[1:]):
            laps.append(data[start:end])
        return laps

    def get_object(self, data_source):
        data_object = getattr(self, data_source)
        return data_object
