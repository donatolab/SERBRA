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

    def get_variable_output_paths(self, full_root_dir, defined_output_name=None):
        full_output_paths = {}
        for output_dir, regex_searches in self.variable_outputs.items():
            full_output_dir = full_root_dir.joinpath(output_dir)
            found_files_list = []
            for regex_search in regex_searches:
                if defined_output_name:
                    regex_search = regex_search.replace("*", defined_output_name)
                found_files = get_files(full_output_dir, regex_search=regex_search)
                if found_files:
                    found_files_list += found_files
            full_output_paths[output_dir] = found_files_list
        return full_output_paths

    def get_output_paths(self, full_root_dir: Path = None, defined_output_name=None):
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
                full_root_dir, defined_output_name
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
        self, file_name, full_root_dir: Path = None, defined_output_name=None
    ):
        full_root_dir = self.define_full_root_dir(full_root_dir)
        output_paths = self.get_output_paths(full_root_dir, defined_output_name)
        transposed_output_paths = self.transpose_output_paths(output_paths)
        fpath = transposed_output_paths[file_name].joinpath(file_name)
        return fpath

    def load_data(self, file_name=None, task_name=None, full_root_dir: Path = None):
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
        if file_name:
            fpath = self.get_file_path(file_name, full_root_dir)
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

    def get_preprocess(self, preprocess_name):
        raise NotImplementedError(
            "Preprocess method not implemented for {self.__class__}"
        )


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
        elif preprocess_name == "mat_to_py":
            preprocess = Mat_to_py(self.method, self.key, self.root_dir)
        else:
            raise ValueError(f"Preprocessing software {preprocess_name} not supported.")
        return preprocess


## Behavior
class Treadmill(BehaviorSetup):
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
        self.preprocess = self.get_preprocess(preprocess_name)

    # get_file_path is inherited from Output class

    # get_output_paths is inherited from Output class


class Trackball(BehaviorSetup):
    def __init__(self, preprocess_name, method, key, root_dir=None):
        super().__init__(preprocess_name, method, key, root_dir)
        root_folder = f"???-{method}"
        raise NotImplementedError("Treadmill setup not implemented yet")


class VR(BehaviorSetup):
    def __init__(self, preprocess_name, method, key, root_dir=None):
        super().__init__(preprocess_name, method, key, root_dir)
        root_folder = f"???-{method}"
        raise NotImplementedError("Treadmill setup not implemented yet")


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
        fpath = self.get_file_path(fname, defined_output_name=task_name)
        fpath = fpath if fpath.exists() else None
        return fpath

    def process_raw_data(self, task_name):
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


class Mat_to_py(Preprocessing):
    """
    Class to manage the conversion of .mat files to .npy files
    """

    # TODO: implement mat_to_py manager
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

    def process_raw_data(self, task_name):
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
        movement_dir = self.movement_dir
        if not os.path.exists(self.movement_dir):
            # print(f"No movement data directory found: {self.movement_dir}")
            global_logger.error(
                f"No movement data directory found: {self.movement_dir}"
            )
            return None
        from oct2py import octave

        root_function_path = "utils\\mat_movement_converter\\convert_trd"
        octave.addpath(root_function_path)
        src_dirs = ["npy-matlab", "calcium\\calcium"]
        for src_dir in src_dirs:
            octave.addpath(os.path.join(root_function_path, src_dir))

        w = wheel.Wheel()
        fnames = get_files(
            movement_dir,
            ending=".mat",
            regex_search=f"{self.animal_id}_{self.session_id}",
        )
        for fname in fnames:
            fpath = os.path.join(self.movement_dir, fname)
            session_part = re.search("S[0-9]", fname)[0]
            veloctiy_fname = f"{session_part}_velocity.npy"
            velocity_fpath = os.path.join(self.movement_dir, veloctiy_fname)
            if os.path.exists(velocity_fpath):
                print(
                    f"{self.animal_id} {self.session_id}: Velocity file already exists: {velocity_fpath} skipping"
                )
                global_logger.info(
                    f"{self.animal_id} {self.session_id}: Velocity file already exists: {velocity_fpath} skipping"
                )
                continue
            print(f"Converting {fname}")
            global_logger.info(f"Converting {fname}")
            # data.mat columns
            # 1,2 encoder
            # 3   lap detector
            # 4   galvos
            # nout=0 means that the function will not return anything.
            octave.feval(wheel_processing_fname, fpath, nout=0)
            # w.root_dir = os.path.split("wheel.npy")[0]
            w.root_dir = movement_dir
            w.load_track(session_part=session_part)
            w.compute_velocity(session_part=session_part)

    # get_data_path is inherited from Output class
    # process_raw_data(self, task_name) is inherited from Preprocessing class
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
