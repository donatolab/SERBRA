from typing import List, Union, Dict, Any, Tuple, Optional

from pathlib import Path

from scipy.signal import butter, filtfilt  # , lfilter, freqz
import numpy as np

# parallel processing
from numba import jit, njit, prange


# Meta Meta Class
class Output:
    """
    Class to manage the output of the setups and preprocessing steps
    Attribute:
        - static_outputs: dictionary of static output files structured as {Patch(): [file_name1, file_name2]}
        - variable_outpus: dictionary of variable output files structured as {Path(): [regex_search1, regex_search2]}
    """

    def __init__(self, key, root_dir=None, metadata={}):
        self.key = key
        self.root_dir_name = None
        self.root_dir = Path(root_dir) if root_dir else Path()
        self.method = metadata["method"] if "method" in metadata.keys() else None
        self.preprocess_name = (
            metadata["preprocessing_software"]
            if "preprocessing_software" in metadata.keys()
            else None
        )
        self.static_outputs: dict = None
        self.variable_outputs: dict = None
        self.data_naming_scheme = None
        self.identifier: dict = self.extract_identifier(metadata["task_id"])
        self.raw_data_path = None
        self.metadata = metadata

    def define_full_root_dir(self, full_root_dir: Path = None):
        return full_root_dir if full_root_dir else self.root_dir

    def define_raw_data_path(self, fname: str = None):
        if not fname:
            fname = self.data_naming_scheme.format(**self.identifier)
        raw_data_path = self.get_file_path(file_name=fname)
        return raw_data_path

    def get_static_output_paths(self, full_root_dir: Path = None):
        full_root_dir = self.define_full_root_dir(full_root_dir)
        full_output_paths = {}
        for relative_output_dir, fnames in self.static_outputs.items():
            full_folder_path = full_root_dir.joinpath(relative_output_dir)
            full_output_paths[full_folder_path] = fnames
        return full_output_paths

    def get_variable_output_paths(self, full_root_dir):
        full_root_dir = self.define_full_root_dir(full_root_dir)
        full_output_paths = {}
        for relative_output_dir, naming_templates in self.variable_outputs.items():
            full_folder_path = full_root_dir.joinpath(relative_output_dir)
            files_list = []
            for template in naming_templates:
                fname = template.format(**self.identifier)
                files_list.append(fname)
            full_output_paths[full_folder_path] = files_list
        return full_output_paths

    def get_output_paths(self, full_root_dir: Path = None):
        full_root_dir = self.define_full_root_dir(full_root_dir)
        if not full_root_dir and self.variable_outputs:
            raise ValueError(
                "No output file paths can be defined. Because outputs are variable."
            )
        elif not full_root_dir:
            output_paths = self.get_static_output_paths()
        else:
            static_output_paths = self.get_static_output_paths(full_root_dir)
            variable_output_paths = self.get_variable_output_paths(full_root_dir)
            # Merge dictionaries of static and variable output paths by unpacking
            # combine values of the same key
            output_paths = static_output_paths
            for path, file_names in variable_output_paths.items():
                if path in output_paths:
                    if file_names:
                        for file_name in file_names:
                            if file_name not in output_paths[path]:
                                output_paths[path].append(file_name)
                else:
                    output_paths[path] = file_names
        return output_paths

    @staticmethod
    def extract_identifier(task_id: str):
        animal_id, date, name = task_id.split("_")
        identifier = {"animal_id": animal_id, "date": date, "task_name": name}
        return identifier

    @staticmethod
    def transpose_output_paths(output_paths: dict):
        transposed_output_paths = {}
        for path, file_names in output_paths.items():
            for file_name in file_names:
                transposed_output_paths[file_name] = path
        return transposed_output_paths

    def get_file_path(self, file_name=None, full_root_dir: Path = None):
        if not file_name:
            raise ValueError("file_name must be provided to get file path.")

        full_root_dir = self.define_full_root_dir(full_root_dir)
        output_paths = self.get_output_paths(full_root_dir)
        transposed_output_paths = self.transpose_output_paths(output_paths)

        fpath = transposed_output_paths[file_name].joinpath(file_name)
        """else:
            if len(transposed_output_paths.keys()) > 1:
                raise ValueError(
                    "Multiple files found. Please provide file_name to select one."
                )
            elif len(transposed_output_paths.keys()) == 0:
                raise ValueError("No files found.")
            else:
                fpath = list(transposed_output_paths.values())[0]"""
        return fpath

    def load_data(
        self,
        file_name=None,
        full_root_dir: Path = None,
    ):
        """
        Load data from file_name in the root_dir.
        The following file types are supported:
            - Numpy files: npy, npz;
        """
        fpath = self.get_file_path(
            file_name=file_name,
            full_root_dir=full_root_dir,
        )

        if fpath.exists():
            data = np.load(fpath)
        else:
            print(f"File {file_name} not found in {fpath}")
        return data

    def get_data_path(self, task_id: str = None, fname: str = None):
        identifier = Output.extract_identifier(task_id)
        task_name = identifier["task_name"]
        fname = f"{task_name}_{self.key}.npy" if not fname else fname
        fpath = self.get_file_path(fname)
        return fpath

    def save_data(self, data, overwrite=False):
        for data_name, data_type in data.items():
            fname = f"{self.identifier['task_name']}_{data_name}"
            fpath = self.raw_data_path.parent.joinpath(fname)
            if fpath.exists() and not overwrite:
                print(f"File {fpath} already exists. Skipping.")
            else:
                np.save(fpath, data_type)

    def process_data(self, task_id=None):
        raise NotImplementedError(
            f"Data processing not implemented for {self.__class__}"
        )
        return data


# Setup Classes
## Meta Class
class Setup(Output):
    """
    All Paths are relative to the root folder of the setup
    """

    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)


class Wheel_Setup(Setup):
    """
    Class managing the wheel setup of Steffen.
    Rotation data is stored in .mat files.
    The wheel is connected to a rotary encoder.
    """

    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.root_dir_name = f"TRD-{self.method}"
        self.root_dir = self.root_dir.joinpath(self.root_dir_name)
        self.static_outputs = {self.root_dir: ["results.zip"]}
        self.data_naming_scheme = (
            "{animal_id}_{session_date}_" + self.root_dir_name + "_{task_names}.mat"
        )
        self.variable_outputs = {}  # {self.root_dir: [self.data_naming_scheme]}
        needed_attributes = ["radius", "clicks_per_rotation", "fps"]
        check_needed_keys(metadata, needed_attributes)

        self.wheel = Wheel(
            radius=metadata["radius"],
        )

        optional_attributes = ["stimulus_length", "stimulus_sequence", "stimulus_type"]
        add_missing_keys(metadata, optional_attributes, fill_value=None)

        self.track = self.wheel.get_track(
            segment_len=metadata["stimulus_length"],
            segment_seq=metadata["stimulus_sequence"],
            type=metadata["stimulus_type"],
        )

    def extract_rotary_data(self, max_speed: float = 0.4, smooth=False):
        """
        Extracts rotary encoder data from the raw data file. The rotary encoder data is used to calculate the distance moved by the wheel.
        The distance moved is then used to calculate the velocity and acceleration of the wheel.
        Optionally, the velocity can be smoothed using a low-pass filter.

        Output is provided in meters per rotary encoder sampling rate.

        Parameters:
            - smooth: bool, optional (default=False)
                A flag indicating whether to smooth the velocity data.
            - max_speed: float, optional (default=0.4)
                The maximum speed of movement in meters per second. This parameter currently not used.

        Returns:
            - data: dict
                A dictionary containing the following data:
                    - distance: np.ndarray
                        An array representing the cumulative distance moved by the wheel.
                    - velocity: np.ndarray
                        An array representing the velocity of the wheel.
                    - acceleration: np.ndarray
                        An array representing the acceleration of the wheel.
        """
        # load data
        rotary_ch_a, rotary_ch_b = AndresDataLoader.fetch_rotary_data(
            self.raw_data_path
        )

        # convert rotary data
        rotary_encoder = RotaryEncoder(
            sample_rate=self.metadata["fps"],
            clicks_per_rotation=self.metadata["clicks_per_rotation"],
        )
        rotary_distances = rotary_encoder.rotary_to_distances(
            wheel_radius=self.wheel.radius,
            ch_a=rotary_ch_a,
            ch_b=rotary_ch_b,
        )

        cumulative_distance = np.cumsum(rotary_distances)
        velocity_from_distances = np.diff(cumulative_distance)

        if smooth:
            velocity_smoothed = butter_lowpass_filter(
                velocity_from_distances, cutoff=2, fs=self.metadata["fps"], order=2
            )
        else:
            velocity_smoothed = velocity_from_distances

        acceleration = np.diff(velocity_smoothed)

        data = {
            "distance": cumulative_distance,
            "velocity": velocity_smoothed,
            "acceleration": acceleration,
        }
        return data

    def process_data(self, save: bool = True, overwrite: bool = False):
        rotary_data = self.extract_rotary_data()

        wheel_data = self.wheel.track.extract_data(
            rotary_data["distance"],
            lap_start_frame=None,
        )
        data = {**rotary_data, **wheel_data}

        if save:
            self.save_data(data, overwrite)

        return data[self.key]


class AndresDataLoader:
    def __init__(self):
        pass

    @staticmethod
    def fetch_rotary_data(fpath):
        """
        columns: roatry encoder state1 | roatry encoder state 2 |        lab sync      | frames
        values          0 or   1       |       0 or   1         |0 or 1 reflecor strip | galvosync

        lab sync is not alway detected
        """
        data = np.load(fpath)

        # get wheel electricity contact values
        rotary_ch_a = data["rotary_encoder1_abstime"]
        rotary_ch_b = data["rotary_encoder2_abstime"]

        # get reference times for alignment to imaging frames
        return rotary_ch_a, rotary_ch_b


class Track:
    def __init__(
        self,
        length: float = None,  # in m
        segment_len: List[float] = None,  # [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        segment_seq: List[int] = None,  # [1, 2, 3, 4, 5, 6],
        type: str = None,  # "A",
        circular: bool = True,
    ):
        self.length = length
        self.segment_len = segment_len
        self.segment_seq = segment_seq
        self.type = type
        self.circular = circular

    @staticmethod
    def get_position_from_cumdist(
        cumulative_distance: np.ndarray,
        length: List[int],
        lap_start_frame: float = None,
    ):
        """
        Get position of the animal on the belt.
        output:
            position: position of the animal on the belt
        """
        # shift distance up, so lap_start is at 0
        lap_start_frame = 0 if lap_start_frame is None else lap_start_frame

        shifted_distance = cumulative_distance + (
            180 - cumulative_distance[lap_start_frame]
        )

        positions = np.zeros(len(shifted_distance))
        started_lap = 1
        ended_lap = 0
        undmapped_positions = True
        while undmapped_positions:
            distance_min = ended_lap * length
            distance_max = started_lap * length
            distances_in_range = np.where(
                (shifted_distance >= distance_min) & (shifted_distance < distance_max)
            )[0]

            positions[distances_in_range] = (
                shifted_distance[distances_in_range] - distance_min
            )

            if max(shifted_distance) > distance_max:
                ended_lap += 1
                started_lap += 1
            else:
                undmapped_positions = False
        return positions

    @staticmethod
    def get_stimulus_at_position(
        positions: np.ndarray, segment_lenghts: List[float], segment_seq: List[int]
    ):
        """
        Get stimulus type at the position.
        output:
            stimulus: stimulus type at the position
        """
        # get segement/stimulus type at each frame
        stimulus_type_indexes = continuouse_to_discrete(positions, segment_lenghts)
        stimulus_type_at_frame = np.array(segment_seq)[
            stimulus_type_indexes % len(segment_seq)
        ]
        return stimulus_type_at_frame

    def extract_data(
        self,
        cumulative_distance: np.ndarray,
        lap_start_frame: float = None,
    ):
        """
        Extract data from the cumulative distance of the belt.
        output:
            data: dictionary with
                - position: position of the animal on the belt
                - stimulus: stimulus type at the position if track segments are provided
        """
        # Normalize cumulative distance to treadmill length
        positions = self.get_position_from_cumdist(
            cumulative_distance,
            length=self.length,
            lap_start_frame=lap_start_frame,
        )
        data = {"position": positions}

        if self.segment_len is None or self.segment_seq is None:
            print("No segment lengths or sequence provided. Not extracting stimulus.")
        else:
            stimulus = self.get_stimulus_at_position(
                positions,
                segment_lenghts=self.segment_len,
                segment_seq=self.segment_seq,
            )
            data["stimulus"] = stimulus

        return data


## Hardware
class Wheel:
    def __init__(self, radius):
        self.radius = radius  # in meters

    @property
    def circumfrence(self):
        return 2 * np.pi * self.radius

    def get_track(self, segment_len=None, segment_seq=None, type=None):
        return Track(
            length=self.circumfrence,
            segment_len=segment_len,
            segment_seq=segment_seq,
            type=type,
            circular=True,
        )


class RotaryEncoder:
    """
    The rotary encoder is used to measure the amount of rotation.
    This can be used to calculate the distance moved by the wheel.
    The rotary encoder is connected to a wheel.
    """

    def __init__(self, sample_rate=10000, clicks_per_rotation=500):
        self.sample_rate = sample_rate  # 10kHz
        self.clicks_per_rotation = clicks_per_rotation

    @staticmethod
    @njit(parallel=True)
    def quadrature_rotation_encoder(ch_a, ch_b):
        """
        Calculates the rotation direction of a quadrature rotary encoder.

        <ch_a, ch_b> two encoder channels with High (1) and Low (0) states
        <out> rotation vector with fwd (1) and rev (-1) motion, same size as input

        This code was originally written in MATLAB
        20050528 Steffen Kandler
        """
        # encode-state look-up table
        # create a state map for fast lookup
        state_map = {
            (0, 0): 0,  # state 1
            (1, 0): 1,  # state 2
            (1, 1): 2,  # state 3
            (0, 1): 3,  # state 4
        }

        # create rotation vector
        rot_dirs = np.zeros(len(ch_a))
        old_state = (ch_a[0], ch_b[0])

        for i in prange(1, len(ch_a)):
            state = (ch_a[i], ch_b[i])
            state_diff = state_map[state] - state_map[old_state]

            if state_diff == -3:
                rot_dirs[i] = 1  # fwd
            elif state_diff == 3:
                rot_dirs[i] = -1  # rev

            old_state = state

        return rot_dirs

    @staticmethod
    def convert_rotary_data(ch_a, ch_b):
        # filter out noise
        ch_a_bin = convert_values_to_binary(ch_a)
        ch_b_bin = convert_values_to_binary(ch_b)

        # calulates rotation direction of quadrature rotary encoder
        rotary_binarized = -RotaryEncoder.quadrature_rotation_encoder(
            ch_a_bin, ch_b_bin
        )
        return rotary_binarized

    def sync_to_lap(
        self,
        moved_distances_in_frame,
        imaging_fps: np.ndarray,
        reference_times: np.ndarray,
        lap_sync: np.ndarray,
        track_length: float,  # in meters
        max_speed=0.6,  # in m/s
    ):
        """
        Synchronizes the moved distances in frames to lap starts using reference times and lap sync signals.

        The reference times show times when the 2p obtained a frame.
        |        lab sync      | frames
        |0 or 1 reflecor strip | galvosync

        Parameters:
            - moved_distances_in_frame: np.ndarray
                Array of distances moved in each frame.
            - start_frame_time: np.ndarray
                The start time of the frames.
            - reference_times: np.ndarray
                Array of reference times corresponding to each frame.
            - lap_sync: np.ndarray
                Array of lap sync signals indicating the start of each lap.
            - track_length: float
                The length of the track in meters.
            - max_speed: float, optional (default=0.5)
                The maximum speed of movement in meters per second.

        Returns:
            - fited_moved_distance_in_frame: np.ndarray
                Array of moved distances in frames, adjusted based on lap sync signals.
            - lap_start_frame: int
                The index of the first frame where a lap start signal is detected.

        This function performs the following steps:
        1. Determines the time window (`mute_lap_detection_time`) in which lap detection is muted, based on `track_length` and `max_speed`.
        2. Identifies unique lap start indices by muting subsequent lap start signals within the determined time window.
        3. Maps lap start signals to imaging frames using `reference_times`.
        4. Finds the first lap start signal that occurs after the `start_frame_time`.
        5. Adjusts the moved distances between lap start signals to account for possible discrepancies due to the wheel continuing to move after the subject stops.
        6. Applies the mean adjustment ratio to distances before the first and after the last detected lap start signals to create a more realistic data set.
        """

        mute_lap_detection_time = track_length / max_speed
        start_frame_time = self.get_first_usefull_recording(
            reference_times, imaging_fps
        )

        # get moved distance until start of track
        ## get first lab sync signal in frame
        ## get unique lap start indices
        lap_start_boolean = lap_sync > lap_sync.max() * 0.9  # 80% of max value
        mute_detection_sampling_frames = self.sample_rate * mute_lap_detection_time
        unique_lap_start_boolean = lap_start_boolean.copy()

        # get unique lap start signals
        muted_sampling_frames = 0
        for index, lap_start_bool in enumerate(lap_start_boolean):
            if muted_sampling_frames == 0:
                if lap_start_bool:
                    muted_sampling_frames = mute_detection_sampling_frames
            else:
                muted_sampling_frames = (
                    muted_sampling_frames - 1 if muted_sampling_frames > 0 else 0
                )
                unique_lap_start_boolean[index] = False
        lap_start_indices = np.where(unique_lap_start_boolean)[0]

        # get lap start signals in imaging frames
        lap_sync_in_frame = np.full(len(reference_times), False)
        old_idx = start_frame_time
        for lap_start_index in lap_start_indices:
            for frame, idx in enumerate(reference_times):
                if old_idx < lap_start_index and lap_start_index < idx:
                    lap_sync_in_frame[frame] = True
                    old_idx = idx
                    break
                old_idx = idx

        # get first lap start signal in imaging frame
        first_lap_start_time = 0
        for lap_start_time in lap_start_indices:
            if lap_start_time > start_frame_time:
                first_lap_start_time = lap_start_time
                break
        lap_start_frame = len(np.where(reference_times < first_lap_start_time)[0])

        # squeze moved distance based on lap sync to compensate for the fact that the wheel continues moving if mouse stops fast after running
        lap_sync_frame_indices = np.where(lap_sync_in_frame == True)[0]
        old_lap_sync_frame_index = lap_sync_frame_indices[0]
        fited_moved_distance_in_frame = moved_distances_in_frame.copy()
        fit_ratios = np.zeros(len(lap_sync_frame_indices) - 1)
        for i, lap_sync_frame_index in enumerate(lap_sync_frame_indices[1:]):
            moved_distances_between_laps = np.sum(
                moved_distances_in_frame[old_lap_sync_frame_index:lap_sync_frame_index]
            )
            # check if lap sync was not detected by comparing with moved distance
            # assuming maximal additional tracked distance distance is < 5%
            max_moved_distance_difference = 0.1
            probable_moved_laps = round(track_length / moved_distances_between_laps)
            real_moved_distance = track_length * probable_moved_laps

            # fit moved distances between laps to create more realistic data
            fit_ratio = real_moved_distance / moved_distances_between_laps
            fit_ratios[i] = fit_ratio
            if fit_ratio > 5 or fit_ratio < 0.2:
                raise ValueError(
                    "Fit ratio is too different. Check data. Maybe wrong lap sync signal detection. Could be because of wrong maximum speed setting."
                )
            fited_moved_distance_in_frame[
                old_lap_sync_frame_index:lap_sync_frame_index
            ] *= fit_ratio
            old_lap_sync_frame_index = lap_sync_frame_index

        # fit moved distances before and after first and last lap sync
        mean_fit_ratio = np.mean(fit_ratios)
        fited_moved_distance_in_frame[0 : lap_sync_frame_indices[0]] *= mean_fit_ratio
        fited_moved_distance_in_frame[lap_sync_frame_indices[-1] : -1] *= mean_fit_ratio

        return fited_moved_distance_in_frame, lap_start_frame

    def rotary_to_distances(
        self, wheel_radius: int, ch_a: np.ndarray, ch_b: np.ndarray
    ):
        """
        Converts rotary encoder data into distances moved.

        Parameters:
            - wheel_radius: int
                The radius of the wheel in meters.
            - ch_a: np.ndarray
                The signal data from channel A of the rotary encoder.
            - ch_b: np.ndarray
                The signal data from channel B of the rotary encoder.

        Returns:
            - distances: np.ndarray
                An array representing the distance moved in meters.

        This function performs the following steps:
        1. Calculates the distance per click of the rotary encoder based on the wheel radius and the number of clicks per rotation.
        2. Binarizes the rotary encoder data using the `convert_rotary_data` method.
        3. Converts the binarized data into distances by multiplying with the distance per click.
        """
        click_distance = (
            2 * np.pi * wheel_radius
        ) / self.clicks_per_rotation  # in meters

        binarized = self.convert_rotary_data(ch_a, ch_b)

        distances = binarized * click_distance
        return distances

    def convert_data_fps_to_imaging_fps(
        self, data: np.ndarray, imaging_fps: float, reference_times: np.ndarray
    ):
        """
        Converts data sampled at the original sampling rate to data corresponding to imaging frame times.

        Parameters:
            - data: np.ndarray
                Array of data sampled at the original sampling rate.
            - imaging_sample_rate: float
                The sample rate of the imaging data.
            - reference_times: np.ndarray
                Array of reference times for each frame.

        Returns:
            - moved_distances_in_frame: np.ndarray
                An array of data resampled to match the imaging frame times.

        This function performs the following steps:
        1. Determines the start index for the first useful recording based on the reference times and imaging sample rate.
        2. Initializes an array to hold the resampled data for each frame.
        3. Iterates through the reference times to sum the data between each frame.
        """
        moved_distances_in_frame = np.zeros(len(reference_times))
        old_idx = self.get_first_usefull_recording(reference_times, imaging_fps)
        for frame, idx in enumerate(reference_times):
            moved_distances_in_frame[frame] = np.sum(data[old_idx:idx])
            old_idx = idx
        return moved_distances_in_frame

    def get_first_usefull_recording(self, reference_times, imaging_sample_rate):
        """
        Determines the start time of the first useful recording based on reference times and the imaging sample rate.

        Parameters:
            - reference_times: np.ndarray
                Array of reference times for each frame.
            - imaging_sample_rate: float
                The sample rate of the imaging data.

        Returns:
            - start_frame_time: int
                The start time of the first useful recording.

        This function performs the following steps:
        1. Calculates the ratio of the original sample rate to the imaging sample rate.
        2. Determines the start time of the first useful recording by subtracting the ratio from the first reference time.
        """
        # reduce the distances to the galvo trigger times to reduce the amount of data
        recording_ratio = int(self.sample_rate / imaging_sample_rate)
        start_frame_time = reference_times[0] - recording_ratio
        return start_frame_time


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


def keys_missing(dictionary, keys):
    present_keys = dictionary.keys()
    missing = []
    for key in keys:
        if key not in present_keys:
            missing.append(key)
    if len(missing) > 0:
        return missing
    return False


def check_needed_keys(metadata, needed_attributes):
    missing = keys_missing(metadata, needed_attributes)
    if missing:
        raise NameError(f"Missing metadata for: {missing} not defined")


def add_missing_keys(metadata, needed_attributes, fill_value=None):
    missing = keys_missing(metadata, needed_attributes)
    if missing:
        for key in missing:
            metadata[key] = fill_value
    return metadata


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


def convert_values_to_binary(vec: np.ndarray, threshold=2.5):
    smaller_idx = np.where(vec < threshold)
    bigger_idx = np.where(vec > threshold)
    vec[smaller_idx] = 0
    vec[bigger_idx] = 1
    vec = vec.astype(int)
    return vec
