from pathlib import Path
from Helper import *

# math
from scipy.interpolate import interp1d

import cebra


# Meta Meta Class
class Output:
    """
    Class to manage the output of the setups and preprocessing steps
    Attribute:
        - static_outputs: dictionary of static output files structured as {Patch(): [file_name1, file_name2]}
        - variable_outpus: dictionary of variable output files structured as {Path(): [regex_search1, regex_search2]}
    """

    def __init__(self, method, key, root_dir=None):
        self.key = key
        self.root_dir_name = None
        self.root_dir = Path(root_dir) if root_dir else Path()
        self.method = method
        self.static_outputs = None
        self.variable_outputs = None
        self.preprocess_name = None

    def define_full_root_dir(self, full_root_dir: Path = None):
        return full_root_dir if full_root_dir else self.root_dir

    def get_static_output_paths(self, full_root_dir: Path = None):
        full_root_dir = self.define_full_root_dir(full_root_dir)
        full_output_paths = {}
        for relative_output_dir, fnames in self.static_outputs.items():
            full_folder_path = full_root_dir.joinpath(relative_output_dir)
            full_output_paths[full_folder_path] = fnames
        return full_output_paths

    def get_variable_output_paths(self, full_root_dir, defined_task_name=None):
        full_output_paths = {}
        for output_dir, regex_searches in self.variable_outputs.items():
            full_output_dir = full_root_dir.joinpath(output_dir)
            found_files_list = []
            for regex_search in regex_searches:
                if defined_task_name:
                    regex_search = regex_search.replace("*", defined_task_name)
                found_files = get_files(full_output_dir, regex_search=regex_search)
                if found_files:
                    found_files_list += found_files
            full_output_paths[output_dir] = found_files_list
        return full_output_paths

    def get_output_paths(self, full_root_dir: Path = None, defined_task_name=None):
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
                full_root_dir, defined_task_name
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
    def transpose_output_paths(output_paths):
        transposed_output_paths = {}
        for path, file_names in output_paths.items():
            for file_name in file_names:
                transposed_output_paths[file_name] = path
        return transposed_output_paths

    def get_file_path(
        self, file_name=None, full_root_dir: Path = None, defined_task_name: str = None
    ):
        if not file_name and not defined_task_name:
            raise ValueError("Either file_name or defined_task_name must be provided.")

        full_root_dir = self.define_full_root_dir(full_root_dir)
        output_paths = self.get_output_paths(full_root_dir, defined_task_name)
        transposed_output_paths = self.transpose_output_paths(output_paths)

        if file_name:
            fpath = transposed_output_paths[file_name].joinpath(file_name)
        else:
            if len(transposed_output_paths.keys()) > 1:
                raise ValueError(
                    "Multiple files found. Please provide file_name to select one."
                )
            elif len(transposed_output_paths.keys()) == 0:
                raise ValueError("No files found.")
            else:
                fpath = list(transposed_output_paths.values())[0]
        return fpath

    def load_data(
        self,
        file_name=None,
        task_name=None,
        defined_task_name=None,
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
        if file_name or defined_task_name:
            fpath = self.get_file_path(
                file_name=file_name,
                defined_task_name=defined_task_name,
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

    def get_data_path(self, task_name):
        raise NotImplementedError(f"Data path not implemented for {self.__class__}")
        return fpath


# Setup Classes
## Meta Class
class Setup(Output):
    """
    All Paths are relative to the root folder of the setup
    """

    def __init__(self, preprocess_name, method, key, root_dir=None):
        super().__init__(method, key, root_dir)
        self.preprocess_name = preprocess_name
        self.data_naming_scheme = None
        self.data_paths = None
        self.preproces = None

    def get_data_path(self, task_name):
        fpath = self.preprocess.get_data_path(task_name)
        return fpath

    def process_data(self, task_name=None):
        raise NotImplementedError(
            f"Data processing not implemented for {self.__class__}"
        )
        # raw_data = self.get_data_to_process(task_name=task_name)
        # data = self.preprocess.process_data(raw_data=raw_data, task_name=task_name)
        return data


class NeuralSetup(Setup):
    def __init__(self, preprocess_name, method, key, root_dir=None):
        super().__init__(preprocess_name, method, key, root_dir)

    def get_preprocess(self, preprocess_name):
        if preprocess_name == "suite2p":
            preprocess = Suite2p(self.method, self.key, self.root_dir)
        elif preprocess_name == "inscopix":
            preprocess = Inscopix_Processing(self.method, self.key, self.root_dir)
        else:
            raise ValueError(f"Preprocessing software {preprocess_name} not supported.")
        return preprocess

    def process_data(self, task_name=None):
        raw_data = self.load_data(defined_task_name=task_name)
        data = self.preprocess.process_data(raw_data=raw_data, task_name=task_name)
        return data


## Behavior
class Wheel:
    def __init__(self, radius=0.1, clicks_per_rotation=500):
        self.radius = radius  # in meters
        self.clicks_per_rotation = clicks_per_rotation
        self.click_distance = (2 * np.pi * self.radius) / self.clicks_per_rotation


class Treadmill:
    def __init__(
        self,
        belt_len: int = 180,
        belt_segment_len: int = 30,
        belt_segment_num: int = 6,
        belt_type: dict = {
            "A": [1, 2, 3, 4, 5, 6],
            "A'": [1, 6, 4, 2, 3, 5],
            "B": [1, 1, 1, 1, 1, 1],
        },
    ):
        self.belt_len = belt_len
        self.belt_segment_len = belt_segment_len
        self.belt_segment_num = belt_segment_num
        self.belt_type = belt_type


class RotaryEncoder:
    def __init__(self, sample_rate=10000, imaging_sample_rate=30):
        # TODO: imaging_sample_rate should be read from the data or metadata
        # TODO: imaging_sample_rate should be read from the data or metadata
        # TODO: imaging_sample_rate should be read from the data or metadata
        # TODO: imaging_sample_rate should be read from the data or metadata
        # TODO: imaging_sample_rate should be read from the data or metadata
        # TODO: imaging_sample_rate should be read from the data or metadata
        # TODO: imaging_sample_rate should be read from the data or metadata
        # TODO: imaging_sample_rate should be read from the data or metadata
        # TODO: imaging_sample_rate should be read from the data or metadata
        self.sample_rate = sample_rate  # 10kHz
        self.imaging_sample_rate = imaging_sample_rate  # 30Hz

    @staticmethod
    def convert_wheel_data(self, data):
        """
        columns: roatry encoder state1 | roatry encoder state 2 |        lab sync      | frames
        values          0 or   1       |       0 or   1         |0 or 1 reflecor strip | galvosync

        lab sync is not alway detected
        """
        print("Processing matlab files")
        # get wheel electricity contact values
        wheel_ch_a = data[:, 0]
        wheel_ch_b = data[:, 1]

        # filter out noise
        wheel_ch_a_bin = convert_values_to_binary(wheel_ch_a)
        wheel_ch_b_bin = convert_values_to_binary(wheel_ch_b)

        # calulates rotation direction of quadrature rotary encoder
        rotary_binarized = self.quadrature_rotation_encoder(
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
        galvo_trigger[:] = 0
        galvo_trigger[indices] = 1

        triggers = []
        for idx in indices:
            if galvo_trigger[idx - 1] == 0:
                triggers.append(idx)
        return triggers

    def calc_distances(self, rotary_binarized, click_distance):
        # get times of rotary encoder
        rotation_clicks_at_times = np.where(rotary_binarized != 0)[0]

        # calculate distance
        distances = np.zeros(len(rotation_clicks_at_times), dtype=np.float32)
        for click_time in rotation_clicks_at_times:
            forward_or_backward = rotary_binarized[click_time]  # can be +/- 1
            distances[click_time] = forward_or_backward * click_distance

        # remove 0 distances
        no_movement = np.where(distances == 0)[0]
        distances = np.delete(distances, no_movement)

        if len(distances) == 0:
            # create dummy array
            distances = np.array([0])
            dist_times = np.array([0])
        else:
            # set distances times based on non-zero rotatry encoder vals
            dist_times = np.arange(rotary_binarized.shape[0], dtype=np.int32)
            dist_times = np.delete(dist_times, no_movement)
        return distances, dist_times

    def fill_distance_gaps(self, distance, times, time_threshold=0.5):
        """
        Fill in the rotary encoder when it's in an undefined state for > 0.5 sec
        """
        # refill in the rotary encoder when it's in an undefined state for > 0.5 sec
        max_time = self.sample_rate * time_threshold

        dist_full = [distance[0]]
        times_full = [times[0]]

        prev_dist_time = times[0]
        for k, dist_time in enumerate(times[1:]):
            # if time between two rotary encoder values is larger than threshold
            if (dist_time - prev_dist_time) >= max_time:
                # create array of zeros for the time between the two rotary encoder values
                temp = np.arange(prev_dist_time, dist_time, max_time)[1:]
                times_full.append(temp)
                dist_full.append(temp * 0)
            else:
                # append the distance and time
                times_full.append(dist_time)
                dist_full.append(distance[k])
            prev_dist_time = dist_time
        # convert lists to numpy arrays
        dist_times_full = np.hstack(times_full)
        distance_full = np.hstack(dist_full)
        return distance_full, dist_times_full

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
        distances, dist_times = self.calc_distances(rotary_binarized, click_distance)

        # fill in the gaps in the distance data
        distance_full, dist_times_full = self.fill_distance_gaps(distances, dist_times)

        # Fit the detected distance times and values
        if distance_full.shape[0] != 0:
            # create an extrapolation function
            F = interp1d(x=dist_times_full, y=distance_full, fill_value="extrapolate")

            # extrapolate the distance to fill gaps of no distance
            # resample the fit function at the correct galvo_tirgger times
            distance_extra = F(galvo_triggers_times)
        else:
            distance_extra = np.zeros(len(rotary_binarized))

        # np.save(fname_out, galvo_dist)
        return distance_extra

    def calc_velocities(self, rotary_binarized, click_distance):
        # get times of rotary encoder
        rotation_clicks_at_times = np.where(rotary_binarized != 0)[0]

        # calculate velocity
        velocities = np.zeros(len(rotation_clicks_at_times), dtype=np.float32)
        last_click_time = 0
        for click_time in rotation_clicks_at_times:
            #
            forward_or_backward = rotary_binarized[click_time]  # can be +/- 1
            # delta distance / delta time
            velocities[click_time] = (forward_or_backward * click_distance) / (
                (click_time - last_click_time) / self.sample_rate
            )
            last_click_time = click_time

        # remove 0 velocities
        no_velocities = np.where(velocities == 0)[0]
        velocities = np.delete(velocities, no_velocities)

        if len(velocities) == 0:
            # create dummy array
            velocities = np.array([0])
            vel_times = np.array([0])
        else:
            # set velocities times based on non-zero rotatry encoder vals
            vel_times = np.arange(rotary_binarized.shape[0], dtype=np.int32)
            vel_times = np.delete(vel_times, no_velocities)
        return velocities, vel_times

    def fill_velocity_gaps(self, velocity, times, time_threshold=0.5):
        """
        Fill in the rotary encoder when it's in an undefined state for > 0.5 sec
        """
        # refill in the rotary encoder when it's in an undefined state for > 0.5 sec
        max_time = self.sample_rate * time_threshold

        vel_full = [velocity[0]]
        times_full = [times[0]]

        prev_vel_time = times[0]
        for k, vel_time in enumerate(times[1:]):  # , desc="refilling stationary times"
            # if time between two rotary encoder values is larger than threshold
            if (vel_time - prev_vel_time) >= max_time:
                # create array of zeros for the time between the two rotary encoder values
                temp = np.arange(prev_vel_time, vel_time, max_time)[1:]
                times_full.append(temp)
                vel_full.append(temp * 0)
            else:
                # append the velocity and time
                times_full.append(vel_time)
                vel_full.append(velocity[k])
            prev_vel_time = vel_time
        # convert lists to numpy arrays
        vel_times_full = np.hstack(times_full)
        velocity_full = np.hstack(vel_full)
        return velocity_full, vel_times_full

    def rotary_to_velocity(
        self,
        rotary_binarized: np.ndarray,
        click_distance: float,
        galvo_triggers_times: np.ndarray,
    ):
        """
        Convert rotary encoder data to velocity
        """
        # extract velocity from rotary encoder
        velocities, vel_times = self.calc_velocities(rotary_binarized, click_distance)

        # fill in the gaps in the velocity data
        velocity_full, vel_times_full = self.fill_velocity_gaps(velocities, vel_times)

        # Fit the detected velocity times and values
        if velocity_full.shape[0] != 0:
            # create an extrapolation function
            F = interp1d(x=vel_times_full, y=velocity_full, fill_value="extrapolate")

            # extrapolate the velocity to fill gaps of no velocity
            # resample the fit function at the correct galvo_tirgger times
            velocity_extra = F(galvo_triggers_times)
        else:
            velocity_extra = np.zeros(len(rotary_binarized))

        # np.save(fname_out, galvo_vel)
        return velocity_extra

    def extract_data(self, data: np.ndarray, wheel: Wheel):
        rotary_binarized, lab_sync, galvo_sync = self.convert_wheel_data(data)

        # get rotary encoder times
        rotary_times = np.arange(rotary_binarized.shape[0]) / self.sample_rate

        # get turned distances of the wheel surface
        rotary_distances = np.cumsum(rotary_binarized) * wheel.click_distance

        # get galvo trigger times for 2P frame
        galvo_triggers_times = self.galvo_trigger_times(galvo_sync)

        # get distance of the wheel surface
        distances = self.rotary_to_distances(
            rotary_binarized, wheel.click_distance, galvo_triggers_times
        )
        distance_over_time = np.cumsum(distances)

        velocity_from_distances = np.diff(distances)

        # get velocity of the wheel surface
        velocity = self.rotary_to_velocity(
            rotary_binarized, wheel.click_distance, galvo_triggers_times
        )

        print(asdff)
        # ........................
        data = {
            "distance": distance_over_time,
            "velocity": velocity,
            "acceleration": None,
            "moving": None,
        }
        return data


class Treadmill_Setup(Setup):
    """
    Class managing the treadmill setup of Steffen.
    Rotation data is stored in .mat files.
    """

    # TODO: change naming of class to better describing treadmill setup
    def __init__(self, preprocess_name, method, key, root_dir=None):
        super().__init__(preprocess_name, method, key, root_dir)
        self.root_dir_name = f"TRD-{method}"
        self.root_dir = self.root_dir.joinpath(self.root_dir_name)
        # TODO: define correct output files
        # DON-017115_20231120_TRD-2P_S5-ACQ.mat
        self.static_outputs = {}
        self.variable_outputs = {self.root_dir: ["*.mat"]}
        self.data_naming_scheme = (
            "{animal_id}_{session_date}_" + self.root_dir_name + "_{task_names}"
        )
        # TODO: integrate metadata, so wheel and Treadmil can be set differently?
        self.wheel = Wheel()
        self.treadmill = Treadmill()
        self.rotary_encoder = RotaryEncoder()
        if preprocess_name != "mat_to_position":
            raise NameError(
                f"WARNING: preprocess_name should be 'mat_to_position'! Got {preprocess_name}, which can lead to errors for this Behavior Setup."
            )

    def get_data_path(self, task_name):
        fname = f"{task_name}_{self.key}.npy"
        fpath = self.get_file_path(fname, defined_task_name=task_name)
        fpath = fpath if fpath.exists() else None
        # TODO: is this correct?......
        return fpath

    def process_data(self, task_name=None):
        raw_data = self.load_data(defined_task_name=task_name)
        rotary_data = self.rotary_encoder.extract_data(raw_data)
        treadmil_data = self.treadmill.get_positions(rotary_data["distance"])

        data = {**rotary_data, **treadmil_data}
        # .....................
        print(asdf)
        raise NotImplementedError(
            f"remove preprocessing metadata from yaml file for {self.__class__}"
        )
        self.static_outputs = {}
        self.variable_outputs = {
            self.root_dir: [
                "*_2p_galvo_trigger.npy",
                "*_triggers.npy",
                "*_velocity.npy",
                "*_wheel.npy",
                "*_position.npy",
                "*_distance.npy",
                "*_velocity.npy",
                "*_acceleration.npy",
                "*_stimulus.npy",
                "*_moving.npy",
            ]
        }
        return data

    # get_file_path is inherited from Output class
    # get_output_paths is inherited from Output class


class Wheel_Setup(Setup):
    """
    Class managing the wheel setup of Steffen.
    Rotation data is stored in .mat files.
    """

    def __init__(self, preprocess_name, method, key, root_dir=None):
        super().__init__(preprocess_name, method, key, root_dir)
        self.root_dir_name = f"TRD-{method}"
        self.root_dir = self.root_dir.joinpath(self.root_dir_name)
        self.static_outputs = {}
        self.variable_outputs = {self.root_dir: ["*.mat"]}
        self.data_naming_scheme = (
            "{animal_id}_{session_date}_" + self.root_dir_name + "_{task_names}"
        )
        self.wheel = Wheel()
        if preprocess_name != "mat_to_velocity":
            raise NameError(
                f"WARNING: preprocess_name should be 'mat_to_velocity'! Got {preprocess_name}, which can lead to errors for this Behavior Setup."
            )
        self.preprocess = self.get_preprocess(preprocess_name)
        raise NotImplementedError("Wheel setup not implemented yet")


class Trackball_Setup(Setup):
    def __init__(self, preprocess_name, method, key, root_dir=None):
        super().__init__(preprocess_name, method, key, root_dir)
        root_folder = f"???-{method}"
        raise NotImplementedError("Treadmill setup not implemented yet")


class Rotary_Wheel(Wheel_Setup):
    def __init__(self, preprocess_name, method, key, root_dir=None):
        super().__init__(preprocess_name, method, key, root_dir)
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
    def __init__(self, preprocess_name, method, key, root_dir=None):
        super().__init__(preprocess_name, method, key, root_dir)
        self.static_outputs = {}
        self.variable_outputs = {self.root_dir: ["*"]}

    def process_data(self, task_name=None):
        raise NotImplementedError(
            f"VR wheel setup not implemented yet {self.__class__}"
        )


class Openfield_Setup(Setup):
    def __init__(self, preprocess_name, method, key, root_dir=None):
        super().__init__(preprocess_name, method, key, root_dir)

        # TODO: Typically data is gathered through a camera
        #
        raise NotImplementedError("Openfield setup not implemented yet")
        self.preprocess = self.get_preprocess(preprocess_name)


class Box(Setup):
    def __init__(self, preprocess_name, method, key, root_dir=None):
        super().__init__(preprocess_name, method, key, root_dir)

        # TODO: Typically data is gathered through a camera
        #
        raise NotImplementedError("Box setup not implemented yet")
        self.preprocess = self.get_preprocess(preprocess_name)


class Cam(Setup):
    def __init__(self, preprocess_name, method, key, root_dir=None):
        super().__init__(preprocess_name, method, key, root_dir)
        vr_root_folder = "0000VR"  # VR
        cam_root_folder = "0000MC"  # Cam Data
        cam_top_root_folder = "000BSM"  # Top View Mousecam
        cam_top_root_folder = "TR-BSL"  # Top View Mousecam Inscopix
        # TODO: implement cam data loading
        root_folder = f"???-{method}"
        raise NotImplementedError("Treadmill setup not implemented yet")


## Imaging
class Femtonics(NeuralSetup):
    def __init__(self, preprocess_name, method, key, root_dir=None):
        super().__init__(preprocess_name, method, key, root_dir)
        self.root_dir_name = f"00{method}-F"
        self.root_dir = self.root_dir.joinpath(self.root_dir_name)
        self.static_outputs = {}
        self.variable_outputs = {self.root_dir: ["*.mesc"]}
        self.data_naming_scheme = (
            "{animal_id}_{session_date}_" + self.root_dir_name + "_{task_names}"
        )
        self.preprocess = self.get_preprocess(preprocess_name)

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
    def __init__(self, method, key, root_dir=None):
        super().__init__(method, key, root_dir)
        self.root_dir = root_dir

    def get_data_path(self, task_name):
        fname = f"{task_name}_{self.key}.npy"
        fpath = self.get_file_path(fname, defined_task_name=task_name)
        fpath = fpath if fpath.exists() else None
        return fpath

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
    def __init__(self, method, key, root_dir=None):
        # TODO: implement suite2p manager
        super().__init__(method, key, root_dir)
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
        self.variable_outputs = {self.data_dir: ["*_photon.npy"]}
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
