from pathlib import Path
from Helper import *

# parallel processing
from numba import jit, njit

# loading mat files
import scipy.io as sio

# math
from scipy.interpolate import interp1d
import cebra

# debugging
import inspect


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
        self.metadata = metadata

    def define_full_root_dir(self, full_root_dir: Path = None):
        return full_root_dir if full_root_dir else self.root_dir

    def get_static_output_paths(self, full_root_dir: Path = None):
        full_root_dir = self.define_full_root_dir(full_root_dir)
        full_output_paths = {}
        for relative_output_dir, fnames in self.static_outputs.items():
            full_folder_path = full_root_dir.joinpath(relative_output_dir)
            full_output_paths[full_folder_path] = fnames
        return full_output_paths

    def get_variable_output_paths(self, full_root_dir, identifier: dict = None):
        full_output_paths = {}
        for output_dir, naming_templates in self.variable_outputs.items():
            full_output_dir = full_root_dir.joinpath(output_dir)
            files_list = []
            for template in naming_templates:
                fname = template.format(**identifier)
                files_list.append(fname)
            full_output_paths[output_dir] = files_list
        return full_output_paths

    def get_output_paths(self, full_root_dir: Path = None, identifier: dict = None):
        full_root_dir = self.define_full_root_dir(full_root_dir)
        if not full_root_dir and self.variable_outputs:
            raise ValueError(
                "No output file paths can be defined. Because outputs are variable."
            )
        elif not full_root_dir:
            output_paths = self.get_static_output_paths()
        else:
            static_output_paths = self.get_static_output_paths(full_root_dir)
            variable_output_paths = self.get_variable_output_paths(
                full_root_dir, identifier
            )
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

    def get_file_path(
        self, file_name=None, full_root_dir: Path = None, identifier: dict = None
    ):
        if not file_name and not identifier:
            raise ValueError("Either file_name or identifier must be provided.")

        full_root_dir = self.define_full_root_dir(full_root_dir)
        output_paths = self.get_output_paths(full_root_dir, identifier)
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
        task_name=None,
        identifier=None,
        full_root_dir: Path = None,
    ):
        """
        Load data from file_name in the root_dir.
        The following file types are supported:
            - Numpy files: npy, npz;
            - HDF5 files: h5, hdf, hdf5, including h5 generated through DLC;
            - PyTorch files: pt, p;
            - csv files;
            - Excel files: xls, xlsx, xlsm;
            - Joblib files: jl;
            - Pickle files: p, pkl;
            - MAT-files: mat.
        """
        if file_name or identifier:
            fpath = self.get_file_path(
                file_name=file_name,
                identifier=identifier,
                full_root_dir=full_root_dir,
            )
        elif task_name:
            fpath = self.get_data_path(task_name)
        else:
            raise ValueError("Either file_name or task_name must be provided.")

        if fpath.exists():
            data = cebra.load_data(fpath)
        else:
            print(f"File {file_name} not found in {fpath}")
        return data

    def get_data_path(self, task_id: str = None, fname: str = None):
        identifier = Output.extract_identifier(task_id)
        task_name = identifier["task_name"]
        fname = f"{task_name}_{self.key}.npy" if not fname else fname
        fpath = self.get_file_path(fname, identifier=identifier)
        return fpath

    def save_data(self, data, raw_data_path, identifier, overwrite=False):
        for data_name, data_type in data.items():
            fname = f"{identifier['task_name']}_{data_name}.npy"
            fpath = raw_data_path.parent.joinpath(data_name)
            if fpath.exists() and not overwrite:
                print(f"File {fpath} already exists. Skipping.")
            else:
                np.save(fpath, data_type)


# Setup Classes
## Meta Class
class Setup(Output):
    """
    All Paths are relative to the root folder of the setup
    """

    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.data_naming_scheme = None
        self.data_paths = None
        self.preproces = None

    def process_data(self, task_id=None):
        raise NotImplementedError(
            f"Data processing not implemented for {self.__class__}"
        )
        # raw_data = self.get_data_to_process(task_id=task_id)
        # data = self.preprocess.process_data(raw_data=raw_data, task_id=task_id)
        return data


class NeuralSetup(Setup):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)

    def get_preprocess(self):
        preprocess_name = self.preprocess_name
        if preprocess_name == "suite2p":
            preprocess = Suite2p(
                key=self.key, root_dir=self.root_dir, metadata=self.metadata
            )
        elif preprocess_name == "inscopix":
            preprocess = Inscopix_Processing(
                key=self.key, root_dir=self.root_dir, metadata=self.metadata
            )
        else:
            raise ValueError(f"Preprocessing software {preprocess_name} not supported.")
        return preprocess

    def get_data_path(self, task_id):
        fpath = self.preprocess.get_data_path(task_id)
        return fpath

    def process_data(self, task_id=None):
        animal_id, date, task_name = Output.extract_identifier(task_id)
        raw_data = self.load_data(identifier=task_name)
        data = self.preprocess.process_data(raw_data=raw_data, task_name=task_name)
        return data


## Behavior
class Wheel:
    def __init__(self, radius=0.1, clicks_per_rotation=500):
        self.radius = radius  # in meters
        self.clicks_per_rotation = clicks_per_rotation
        self.click_distance = (
            2 * np.pi * self.radius
        ) / self.clicks_per_rotation  # in meters


class Treadmill:
    def __init__(
        self,
        belt_len: int = 180,
        belt_segment_len: List[int] = [30, 30, 30, 30, 30, 30],
        belt_segment_seq: List[int] = [1, 2, 3, 4, 5, 6],
        belt_type: str = "A",
        wheel_radius=0.1,
        wheel_clicks_per_rotation=500,
    ):
        self.belt_len = belt_len
        self.belt_segment_len = belt_segment_len
        self.belt_segment_seq = belt_segment_seq
        self.belt_type = belt_type
        self.wheel = Wheel(wheel_radius, wheel_clicks_per_rotation)

    def extract_data(self, cumulative_distance):
        """
        Extract data from the cumulative distance of the belt.
        output:
            data: dictionary with
                - position: position of the animal on the belt
                - stimulus: stimulus type at the position
        """
        # Normalize cumulative distance to treadmill length
        ...... finish this function
        positions = (cumulative_distance / cumulative_distance[-1]) * 180

        segment_indices = (positions // segment_length).astype(int)

        stimulus_type_indexes = continuouse_to_discrete(
            positions, self.belt_segment_len
        )
        stimulus_type_at_frame = np.array(self.belt_segment_seq)[
            stimulus_type_indexes % len(self.belt_segment_seq)
        ]

        data = {"position": positions, "stimulus": stimulus_type_at_frame}
        return data


class RotaryEncoder:
    def __init__(self, sample_rate=10000, imaging_sample_rate=30):
        # TODO: imaging_sample_rate should be read from the data or metadata
        # TODO: imaging_sample_rate should be read from the data or metadata
        self.sample_rate = sample_rate  # 10kHz
        self.imaging_sample_rate = imaging_sample_rate  # 30Hz

    def extract_data_from_matlab_file(self, fpath):
        mat = sio.loadmat(fpath)
        data = mat["trainingdata"]
        return data

    def convert_wheel_data(self, data):
        """
        columns: roatry encoder state1 | roatry encoder state 2 |        lab sync      | frames
        values          0 or   1       |       0 or   1         |0 or 1 reflecor strip | galvosync

        lab sync is not alway detected
        """
        # get wheel electricity contact values
        wheel_ch_a = data[:, 0]
        wheel_ch_b = data[:, 1]

        # filter out noise
        wheel_ch_a_bin = convert_values_to_binary(wheel_ch_a)
        wheel_ch_b_bin = convert_values_to_binary(wheel_ch_b)

        # calulates rotation direction of quadrature rotary encoder
        rotary_binarized = -self.quadrature_rotation_encoder(
            wheel_ch_a_bin, wheel_ch_b_bin
        )
        lab_sync = np.array(data[:, 1]).reshape(-1, 1)
        galvo_sync = np.array(data[:, 3]).reshape(-1, 1)
        return rotary_binarized, lab_sync, galvo_sync

    @staticmethod
    def quadrature_rotation_encoder(ch_a, ch_b):
        """
        calulates rotation direction of quadrature rotary encoder.

        <ch_a,ch_b> two encoder channels with High (1) and Low (0) states
        <out> rotation vector with fwd (1) and rev (-1) motion, same size as input

        This code was originally written in matlab
        20050528 Steffen Kandler
        """
        # encode-state look-up table
        lut = [
            [0, 0],  # state 1
            [1, 0],  # state 2
            [1, 1],  # state 3
            [0, 1],  # state 4
        ]

        # create state vector
        statevec = np.stack([ch_a, ch_b]).transpose()

        # create rotation vector
        rot_dirs = np.zeros([len(statevec), 1])
        old_state = statevec[0]
        for i, state in enumerate(statevec[1:]):
            state_diff = lut.index(list(state)) - lut.index(list(old_state))
            if state_diff == -3:
                rot_dirs[i] = 1  # fwd
            elif state_diff == 3:
                rot_dirs[i] = -1  # rev

            old_state = state
        return rot_dirs

    @staticmethod
    def galvo_trigger_times(galvo_sync):
        """
        value > 0 for when 2P frame is obtained;
        value = 0 in between

        find rising edge of trigger
        """
        indices = np.where(galvo_sync >= 1)[0]
        galvo_trigger = np.zeros(len(galvo_sync))
        galvo_trigger[indices] = 1

        triggers = []
        for idx in indices:
            if galvo_trigger[idx - 1] == 0:
                triggers.append(idx)
        return triggers

    def rotary_to_distances(
        self,
        rotary_binarized: np.ndarray,
        click_distance: float,
        galvo_triggers_times: np.ndarray,
    ):
        """
        Convert rotary encoder data to distance
        """
        # extract distance from rotary encoder
        # calculate distance
        at_time_moved = rotary_binarized * click_distance

        # reduce the distance to the galvo trigger times
        moved_distances_in_frame = np.zeros(len(galvo_triggers_times))
        old_idx = 0
        for frame, idx in enumerate(galvo_triggers_times[1:]):
            moved_distances_in_frame[frame] = np.sum(at_time_moved[old_idx:idx])
            old_idx = idx
        return moved_distances_in_frame

    def rotary_to_velocity(
        self,
        rotary_binarized: np.ndarray,
        click_distance: float,
        galvo_triggers_times: np.ndarray,
    ):
        """
        This METHOD is DEPRECATED and should not be used anymore.
        Convert rotary encoder data to velocity.
        """

        def remove_stationary_times(self, array, rotary_binarized):
            # remove stationary times
            stationary_times = np.where(array == 0)[0]
            movement_values = np.delete(array, stationary_times)

            if len(movement_values) == 0:
                # create dummy array
                movement_values = np.array([0])
                movement_times = np.array([0])
            else:
                movement_times = np.arange(rotary_binarized.shape[0], dtype=np.int32)
                movement_times = np.delete(movement_times, stationary_times)
            return movement_values, movement_times

        def calc_velocities(self, rotary_binarized, click_distance):
            # get times of rotary encoder
            rotation_clicks_at_times = np.where(rotary_binarized != 0)[0]

            # calculate velocity
            temp_velocities = np.zeros(len(rotary_binarized))
            last_click_time = 0
            for click_time in rotation_clicks_at_times:
                #
                forward_or_backward = rotary_binarized[click_time]  # can be +/- 1
                # delta distance / delta time
                temp_velocities[click_time] = (forward_or_backward * click_distance) / (
                    (click_time - last_click_time) / self.sample_rate
                )
                last_click_time = click_time

            velocities, vel_times = self.remove_stationary_times(
                temp_velocities, rotary_binarized
            )
            return velocities, vel_times

        def fill_gaps(self, array, times, time_threshold=0.5):
            """
            Fill in the rotary encoder when it's in an undefined state for > 0.5 sec. This is done for more accurate extrapolation
            """
            # refill in the rotary encoder when it's in an undefined state for > 0.5 sec
            max_time = self.sample_rate * time_threshold

            full = [array[0]]
            times_full = [times[0]]

            prev_time = times[0]
            for k, time in enumerate(times[1:]):  # , desc="refilling stationary times"
                # if time between two rotary encoder values is larger than threshold
                # create array of zeros for the time between the two rotary encoder values
                if (time - prev_time) >= max_time:
                    zero_array = np.arange(prev_time, time, max_time)[1:]
                    times_full.append(zero_array)
                    full.append(zero_array * 0)
                else:
                    # append the velocity and time
                    times_full.append(time)
                    full.append(array[k])
                prev_time = time
            # convert lists to numpy arrays
            times_full = np.hstack(times_full)
            array_full = np.hstack(full)
            return array_full, times_full

        def extrapolate_fit(self, times, values, galvo_triggers_times):
            if values.shape[0] != 0:
                # create an extrapolation function
                F = interp1d(x=times, y=values, fill_value="extrapolate")

                # extrapolate the velocity to fill gaps of no velocity
                # resample the fit function at the correct galvo_tirgger times
                values_extra = F(galvo_triggers_times)
            else:
                values_extra = np.zeros(len(galvo_triggers_times))
            return values_extra

        # extract velocity from rotary encoder
        velocities, vel_times = self.calc_velocities(rotary_binarized, click_distance)

        # fill in the gaps in the velocity data
        velocity_full, vel_times_full = self.fill_gaps(velocities, vel_times)

        velocity_extra = self.extrapolate_fit(
            times=vel_times_full,
            values=velocity_full,
            galvo_triggers_times=galvo_triggers_times,
        )

        return velocity_extra

    def extract_data(self, fpath: np.ndarray, wheel: Wheel):
        data = self.extract_data_from_matlab_file(fpath)

        rotary_binarized, lab_sync, galvo_sync = self.convert_wheel_data(data)

        # get galvo trigger times for 2P frame
        galvo_triggers_times = self.galvo_trigger_times(galvo_sync)

        ## get distance of the wheel surface
        distances = self.rotary_to_distances(
            rotary_binarized, wheel.click_distance, galvo_triggers_times
        )
        cumulative_distance = np.cumsum(distances)
        velocity_from_distances = np.diff(cumulative_distance)

        ## get velocity m/s of the wheel surface
        # velocity = self.rotary_to_velocity(
        #    rotary_binarized, wheel.click_distance, galvo_triggers_times
        # )

        velocity_smoothed = butter_lowpass_filter(
            velocity_from_distances, cutoff=2, fs=self.imaging_sample_rate, order=2
        )
        acceleration = np.diff(velocity_smoothed)

        data = {
            "distance": cumulative_distance,
            "velocity": velocity_smoothed,
            "acceleration": acceleration,
        }
        return data


class Treadmill_Setup(Setup):
    """
    Class managing the treadmill setup of Steffen.
    Rotation data is stored in .mat files.
    """

    # TODO: change naming of class to better describing treadmill setup
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.root_dir_name = f"TRD-{self.method}"
        self.root_dir = self.root_dir.joinpath(self.root_dir_name)
        # TODO: add posibility to provide data_naming_scheme in yaml file
        # DON-017115_20231120_TRD-2P_S5-ACQ.mat
        self.static_outputs = {}
        self.data_naming_scheme = (
            "{animal_id}_{date}_" + self.root_dir_name + "_{task_name}-ACQ.mat"
        )
        self.variable_outputs = {
            self.root_dir: [
                self.data_naming_scheme,
                "{task_name}_velocity.npy",
                "{task_name}_position.npy",
                "{task_name}_distance.npy",
                "{task_name}_velocity.npy",
                "{task_name}_acceleration.npy",
                "{task_name}_stimulus.npy",
                "{task_name}_moving.npy",
            ]
        }
        needed_attributes = [
            "radius",
            "clicks_per_rotation",
            "stimulus_type",
            "stimulus_sequence",
            "stimulus_length",
            "environment_dimensions",
            "fps",
            "imaging_fps",
        ]
        check_needed_keys(metadata, needed_attributes)
        self.treadmill = Treadmill(
            belt_len=metadata["environment_dimensions"],
            belt_segment_len=metadata["stimulus_length"],
            belt_segment_seq=metadata["stimulus_sequence"],
            belt_type=metadata["stimulus_type"],
            wheel_radius=metadata["radius"],
            wheel_clicks_per_rotation=metadata["clicks_per_rotation"],
        )
        self.rotary_encoder = RotaryEncoder(
            sample_rate=metadata["fps"], imaging_sample_rate=metadata["imaging_fps"]
        )

    def process_data(self, task_id=None, save: bool = True, overwrite: bool = False):
        identifier = Output.extract_identifier(task_id)
        fname = self.data_naming_scheme.format(**identifier)
        raw_data_path = self.get_file_path(file_name=fname, identifier=identifier)
        rotary_data = self.rotary_encoder.extract_data(
            raw_data_path, wheel=self.treadmill.wheel
        )
        treadmil_data = self.treadmill.extract_data(rotary_data["distance"])
        data = {**rotary_data, **treadmil_data}

        if save:
            self.save_data(data, raw_data_path, identifier, overwrite)

        raise NotImplementedError(
            f"remove preprocessing metadata from yaml file for {self.__class__}"
        )
        return data[key]

    # get_file_path is inherited from Output class
    # get_output_paths is inherited from Output class


class Wheel_Setup(Setup):
    """
    Class managing the wheel setup of Steffen.
    Rotation data is stored in .mat files.
    """

    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.root_dir_name = f"TRD-{self.method}"
        self.root_dir = self.root_dir.joinpath(self.root_dir_name)
        self.static_outputs = {}
        self.data_naming_scheme = (
            "{animal_id}_{session_date}_" + self.root_dir_name + "_{task_names}.mat"
        )
        self.variable_outputs = {self.root_dir: [self.data_naming_scheme]}
        needed_attributes = [
            "radius",
            "clicks_per_rotation",
        ]
        check_needed_keys(metadata, needed_attributes)
        self.wheel = Wheel(
            radius=metadata["radius"],
            clicks_per_rotation=metadata["clicks_per_rotation"],
        )
        self.rotary_encoder = RotaryEncoder(
            sample_rate=metadata["fps"], imaging_sample_rate=metadata["fps"]
        )
        raise NotImplementedError("Wheel setup not implemented yet")


class Trackball_Setup(Setup):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        root_folder = f"???-{self.method}"
        raise NotImplementedError("Treadmill setup not implemented yet")


class Rotary_Wheel(Wheel_Setup):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.rotary_encoder = RotaryEncoder()

    def process_data(self, task_name=None):
        raise NotImplementedError(
            f"Rotary wheel setup not implemented yet {self.__class__}"
        )
        rotary_binarized, lab_sync, galvo_sync = self.rotary_encoder.convert_wheel_data(
            raw_data
        )
        data = self.preprocess.process_data(raw_data=raw_data, task_name=task_name)
        return data


class VR_Wheel(Wheel_Setup):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.static_outputs = {}
        self.variable_outputs = {self.root_dir: ["*"]}

    def process_data(self, task_name=None):
        raise NotImplementedError(
            f"VR wheel setup not implemented yet {self.__class__}"
        )


class Openfield_Setup(Setup):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)

        # TODO: Typically data is gathered through a camera
        #
        raise NotImplementedError("Openfield setup not implemented yet")


class Box(Setup):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)

        # TODO: Typically data is gathered through a camera
        #
        raise NotImplementedError("Box setup not implemented yet")


class Active_Avoidance(Setup):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        # TODO: Typically data is gathered through a camera
        raise NotImplementedError("Active Avoidance setup not implemented yet")


class Cam(Setup):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        vr_root_folder = "0000VR"  # VR
        cam_root_folder = "0000MC"  # Cam Data
        cam_top_root_folder = "000BSM"  # Top View Mousecam
        cam_top_root_folder = "TR-BSL"  # Top View Mousecam Inscopix
        # TODO: implement cam data loading
        root_folder = f"???-{self.method}"
        raise NotImplementedError("Treadmill setup not implemented yet")


## Imaging
class Femtonics(NeuralSetup):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.root_dir_name = f"00{self.method}-F"
        self.root_dir = self.root_dir.joinpath(self.root_dir_name)
        self.static_outputs = {}
        self.data_naming_scheme = (
            "{animal_id}_{session_date}_" + self.root_dir_name + "_{task_names}.mesc"
        )
        self.variable_outputs = {self.root_dir: [self.data_naming_scheme]}
        self.preprocess = self.get_preprocess()

    # get_output_paths is inherited from Output class


class Thorlabs(NeuralSetup):
    def __init__(self):
        self.root_dir_name = Path(f"00{self.method}-T")
        self.data_dir = self.root_dir_name.joinpath("data")
        self.static_outputs = {self.data_dir: ["Image_001_001.raw"]}
        # TODO: output files not defined
        raise NotImplementedError("Thorlabs setup not implemented yet")


class Inscopix(NeuralSetup):
    def __init__(self, method):
        root_folder = f"00{method}-I"
        output_fname = None
        # TODO: output files not defined
        raise NotImplementedError("Inscopix setup not implemented yet")


# Preprocessing Classes
## Meta Class
class Preprocessing(Output):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.root_dir = root_dir

    def process_data(self, raw_data, task_name=None, save=True):
        raise NotImplementedError(
            f"Raw data processing not implemented for {self.__class__}"
        )
        return data


## Neural
class Inscopix_Processing(Preprocessing):
    def __init__(self, method):
        # TODO: implement inscopix manager
        # TODO: implement inscopix attributes
        raise NotImplementedError("Inscopix preprocessing not implemented yet")


class Suite2p(Preprocessing):
    def __init__(self, key, root_dir=None, metadata={}):
        # TODO: implement suite2p manager
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.data_dir = self.root_dir.joinpath("tif", "suite2p", "plane0")
        self.static_outputs = {
            self.data_dir: [
                "F.npy",
                "F_neu.npy",
                "iscell.npy",
                "ops.npy",
                "spks.npy",
                "stat.npy",
                "binarized_traces.npz",
                "cell_drying.npy",
                "data.bin",
            ]
        }
        self.variable_outputs = {self.data_dir: ["{task_name}_photon.npy"]}
        self.output_fnames = {
            "f_raw": "F.npy",
            "f_neuropil": "F_neu.npy",
            "f_upphase": "F_neu.npy",
            "iscell": "iscell.npy",
            "ops": "ops.npy",
            "spks": "spks.npy",
            "stat": "stat.npy",
            "cabincorr": "binarized_traces.npz",
            "cell_geldrying": "cell_drying.npy",
            "binary": "data.bin",
        }

    def process_data(self, raw_data, task_name=None, save=True):
        raise NotImplementedError(
            f"Suite2p data processing not implemented for {self.__class__}"
        )

    # get_data_path is inherited from Output class
    # process_raw_data(self, task_name) is inherited from Preprocessing class
    # load_data is inherited from Output class
    # self.load_data(file_name, full_root_dir=self.root_dir)
    # get_file_path is inherited from Output class
