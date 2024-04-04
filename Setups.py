from pathlib import Path
from Helper import *

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

    def get_preprocess(self, preprocess_name):
        raise NotImplementedError(
            "Preprocess method not implemented for {self.__class__}"
        )

    def get_data_to_process(self, task_name=None):
        data_paths = self.get_output_paths()

    def process_data(self, task_name=None):
        raw_data = self.get_data_to_process(task_name=task_name)
        data = self.preprocess.process_data(raw_data=raw_data, task_name=task_name)
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


class BehaviorSetup(Setup):
    def __init__(self, preprocess_name, method, key, root_dir=None):
        super().__init__(preprocess_name, method, key, root_dir)

    def get_preprocess(self, preprocess_name):
        if preprocess_name == "manual":
            preprocess = Manual(self.method, self.key, self.root_dir)
        elif preprocess_name == "mat_to_velocity":
            preprocess = Mat_to_Velocity(self.method, self.key, self.root_dir)
        elif preprocess_name == "mat_to_position":
            preprocess = Mat_to_Position(self.method, self.key, self.root_dir)
        elif preprocess_name == "csv_to_py":
            preprocess = CSV_to_py(self.method, self.key, self.root_dir)
        else:
            raise ValueError(f"Preprocessing software {preprocess_name} not supported.")
        return preprocess


## Behavior
class Treadmill(BehaviorSetup):
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
        self.static_outputs = None
        self.variable_outputs = {self.root_dir: ["*.mat"]}
        self.data_naming_scheme = (
            "{animal_id}_{session_date}_" + self.root_dir_name + "_{task_names}"
        )
        #TODO: should metadata be variable or static?
        self.belt_len = 180 # cm
        self.belt_segment_len = 30 # cm
        self.belt_segment_num = 6
        self.belt_type = {"A": [1, 2, 3, 4, 5, 6], "A'": [1, 6, 4, 2, 3, 5], "B": [1, 1, 1, 1, 1, 1]}
        ################################################################################################
        if preprocess_name != "mat_to_position":
            raise NameError(
                f"WARNING: preprocess_name should be 'mat_to_position'! Got {preprocess_name}, which can lead to errors for this Behavior Setup."
            )
        self.preprocess = self.get_preprocess(preprocess_name)

    # get_file_path is inherited from Output class
    # get_output_paths is inherited from Output class


class Wheel(BehaviorSetup):
    """
    Class managing the wheel setup of Steffen.
    Rotation data is stored in .mat files.
    """

    def __init__(self, preprocess_name, method, key, root_dir=None):
        super().__init__(preprocess_name, method, key, root_dir)
        self.root_dir_name = f"TRD-{method}"
        self.root_dir = self.root_dir.joinpath(self.root_dir_name)
        self.static_outputs = None
        self.variable_outputs = {self.root_dir: ["*.mat"]}
        self.data_naming_scheme = (
            "{animal_id}_{session_date}_" + self.root_dir_name + "_{task_names}"
        )
        if preprocess_name != "mat_to_velocity":
            raise NameError(
                f"WARNING: preprocess_name should be 'mat_to_velocity'! Got {preprocess_name}, which can lead to errors for this Behavior Setup."
            )
        self.preprocess = self.get_preprocess(preprocess_name)
        raise NotImplementedError("Wheel setup not implemented yet")


class Trackball(BehaviorSetup):
    def __init__(self, preprocess_name, method, key, root_dir=None):
        super().__init__(preprocess_name, method, key, root_dir)
        root_folder = f"???-{method}"
        raise NotImplementedError("Treadmill setup not implemented yet")


class VR_Treadmill(BehaviorSetup):
    def __init__(self, preprocess_name, method, key, root_dir=None):
        super().__init__(preprocess_name, method, key, root_dir)
        self.root_dir_name = f"TRD-{method}"
        self.root_dir = self.root_dir.joinpath(self.root_dir_name)
        # TODO: define correct output files
        # DON-017115_20231120_TRD-2P_S5-ACQ.mat
        self.static_outputs = None
        self.variable_outputs = {self.root_dir: ["*.csv"]}
        self.data_naming_scheme = (
            "{animal_id}_{session_date}_" + self.root_dir_name + "_{task_names}"
        )
        self.preprocess = self.get_preprocess(preprocess_name)


class Openfield(BehaviorSetup):
    def __init__(self, preprocess_name, method, key, root_dir=None):
        super().__init__(preprocess_name, method, key, root_dir)

        # TODO: Typically data is gathered through a camera
        #
        raise NotImplementedError("Openfield setup not implemented yet")
        self.preprocess = self.get_preprocess(preprocess_name)


class Box(BehaviorSetup):
    def __init__(self, preprocess_name, method, key, root_dir=None):
        super().__init__(preprocess_name, method, key, root_dir)

        # TODO: Typically data is gathered through a camera
        #
        raise NotImplementedError("Box setup not implemented yet")
        self.preprocess = self.get_preprocess(preprocess_name)


class Cam(BehaviorSetup):
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
        self.static_outputs = None
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


## Behavior
class Manual(Preprocessing):
    """
    Class to manage manual preprocessing
    """

    def __init__(self, method):
        # TODO: implement manual manager
        raise NotImplementedError("Manual preprocessing not implemented yet")


class CSV_to_py(Preprocessing):
    """
    Class to manage the conversion of .csv files to .npy files
    """

    def __init__(self, method, key, root_dir=None):
        super().__init__(method, key, root_dir)
        self.static_outputs = {}
        self.variable_outputs = {self.root_dir: ["*"]}
        # TODO: maybe first from csv to mat then to py. depends on time implementing
        # ranans code in python
        raise NotImplementedError("CSV to numpy preprocessing not implemented yet")


class Mat_to_Position(Preprocessing):
    """
    Class to manage the conversion of .mat files to .npy files
    """

    # TODO: implement mat_to_Velocity manager
    def __init__(self, method, key, root_dir=None):
        # TODO: implement suite2p manager
        super().__init__(method, key, root_dir)
        self.static_outputs = {}
        self.variable_outputs = {
            self.root_dir: [
                "*_2p_galvo_trigger.npy",
                "*_triggers.npy",
                "*_velocity.npy",
                "*_wheel.npy",
                "*_wheel.npy",
                "*_position.npy",
                "*_distance.npy",
                "*_velocity.npy",
                "*_acceleration.npy",
                "*_stimulus.npy",
                "*_moving.npy",
            ]
        }

    def process_data(self, raw_data, task_name=None, save=True):
        """ """
        rotary_binarized, lab_sync, galvo = self.get_wheel_data(raw_data)
        self.load_tracks2(fname_tracks, bin_width)
        data = self.load_data(task_name=task_name)
        return data
    
    def asdf(self):
        ....................
        pass

    def get_wheel_data(self, data):
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
        rotary_binarized = self.quadrature_rotation_encoder(wheel_ch_a_bin, wheel_ch_b_bin)
        lab_sync = np.array(data[:, 1]).reshape(-1, 1)
        galvo = np.array(data[:, 3]).reshape(-1, 1)
        return rotary_binarized, lab_sync, galvo

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

    def load_tracks2(fname_tracks, bin_width):
        #
        raise NotImplementedError("Mat to position preprocessing not implemented yet")
        pos_tracks = []
        idx_tracks = []
        idx_ctr = 0

        # loop over all 3 segments here while loading
        for fname_track in fname_tracks:
            with h5py.File(fname_track, "r") as file:
                #
                pos = file["trdEval"]["position_atframe"][()].squeeze()

                n_timesteps = pos.shape[0]
                print(fname_track, "# of time steps: ", pos.shape)

            # bin speed for some # of bins
            # bin_width = 1
            sum_flag = False
            pos_bin = run_binning(pos, bin_width, sum_flag)
            # print ("pos bin: ", pos_bin.shape)

            #
            pos_tracks.append(pos_bin)

            # add locations of the belt realtive
            temp = np.arange(idx_ctr, idx_ctr + n_timesteps, 1)
            temp = np.int32(run_binning(temp, bin_width, sum_flag))
            idx_tracks.append(temp)

            idx_ctr += n_timesteps

        return pos_tracks, idx_tracks

    def run_binning(data, bin_size=7, sum_flag=True):
        # split data into bins
        idx = np.arange(0, data.shape[0], bin_size)
        d2 = np.array_split(data, idx[1:])

        # sum on time axis; drop last value
        if sum_flag:
            d3 = np.array(d2[:-1]).sum(1)
        else:
            d3 = np.median(np.array(d2[:-1]), axis=1)

        print("   Data binned using ", bin_size, " frame bins, final size:  ", d3.shape)

        #
        return d3


class Mat_to_Velocity(Preprocessing):
    """
    Class to manage the conversion of .mat files to .npy files
    """

    # TODO: implement mat_to_Velocity manager
    def __init__(self, method, key, root_dir=None):
        # TODO: implement suite2p manager
        super().__init__(method, key, root_dir)
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

    def process_data(self, raw_data, task_name=None, save=True):
        self.convert_movement_data(self.root_dir)
        data = self.load_data(task_name=task_name)
        return data

    def convert_movement_data(
        self, directory=None, wheel_processing_fname="process_wheel.m"
    ):
        """
        Converts movement data from .mat files to velocity data and saves it as .npy files.

        Args:
            wheel_processing_fname (str, optional): Filename of the Octave/MATLAB function for processing wheel data.
                Defaults to "process_wheel.m".

        Returns:
            None: If the movement directory does not exist or if there is any error during the conversion process,
                None is returned. The velocity files are saved as .npy files in the specified movement directory.

        Raises:
            Any exceptions raised during the process are logged and handled, allowing the function to continue
            processing other files.

        Note:
            This function assumes the presence of specific directories and files within the movement directory
            for processing. It utilizes external resources like the Octave/MATLAB environment and the 'oct2py'
            package for MATLAB/Octave integration.

        """
        raise NotImplementedError(
            f"Movement data conversion not implemented for {self.__class__}"
        )

        w = wheel.Wheel()
        # w.root_dir = os.path.split("wheel.npy")[0]
        w.root_dir = movement_dir
        w.load_track(session_part=session_part)
        w.compute_velocity(session_part=session_part)

    # get_data_path is inherited from Output class
    # process_data(self, task_name) is inherited from Preprocessing class
    # load_data is inherited from Output class
    # self.load_data(file_name, full_root_dir=self.root_dir)
    # get_file_path is inherited from Output class


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

    # get_data_path is inherited from Output class
    # process_raw_data(self, task_name) is inherited from Preprocessing class
    # load_data is inherited from Output class
    # self.load_data(file_name, full_root_dir=self.root_dir)
    # get_file_path is inherited from Output class
