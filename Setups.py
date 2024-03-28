from pathlib import Path
from Helper import *


# Meta Meta Class
class Output:
    """
    Class to manage the output of the setups and preprocessing steps
    Attribute:
        - static_outputs: dictionary of static output files structured as {Patch(): [file_name1, file_name2]}
        - variable_outpus: dictionary of variable output files structured as {Path(): [regex_search1, regex_search2]}
    """

    def __init__(self, method, root_dir=None):
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
        output_paths = {
            key: full_root_dir.joinpath(value)
            for key, value in self.static_outputs.items()
        }
        return output_paths

    def get_variable_output_paths(self, full_root_dir):
        output_paths = {}
        for output_dir, regex_searches in self.variable_outputs.items():
            full_output_dir = full_root_dir.joinpath(output_dir)
            for regex_search in regex_searches:
                output_paths[output_dir] = get_files(full_output_dir, regex_search)
        return output_paths

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

    def get_file_path(self, file_name, full_root_dir: Path = None):
        full_root_dir = self.define_full_root_dir(full_root_dir)
        output_paths = self.get_output_paths(full_root_dir)
        transposed_output_paths = self.transpose_output_paths(output_paths)
        fpath = transposed_output_paths[file_name].joinpath(file_name)
        return fpath


# Setup Classes
## Meta Class
class Setup(Output):
    """
    All Paths are relative to the root folder of the setup
    """

    def __init__(self, preprocess_name, method, root_dir=None):
        super().__init__(method, root_dir)
        self.preprocess_name = preprocess_name
        self.data_naming_scheme = None
        self.data_paths = None
        self.preprocess = self.get_preprocess(preprocess_name)

    def get_preprocess(self, preprocess_name):
        raise NotImplementedError(
            "Preprocess method not implemented for {self.__class__}"
        )


class NeuralSetup(Setup):
    def __init__(self, preprocess_name, method, root_dir=None):
        super().__init__(preprocess_name, method, root_dir)

    def get_preprocess(self, preprocess_name):
        if preprocess_name == "suite2p":
            preprocess = Suite2p(self.method, self.root_dir)
        elif preprocess_name == "inscopix":
            preprocess = Inscopix_Processing(self.method, self.root_dir)
        else:
            raise ValueError(f"Preprocessing software {preprocess_name} not supported.")
        return preprocess


class BehaviorSetup(Setup):
    def __init__(self, preprocess_name, method, root_dir=None):
        super().__init__(preprocess_name, method, root_dir)

    def get_preprocess(self, preprocess_name):
        if preprocess_name == "manual":
            preprocess = Manual(self.method, self.root_dir)
        elif preprocess_name == "mat_to_py":
            preprocess = Mat_to_py(self.method, self.root_dir)
        else:
            raise ValueError(f"Preprocessing software {preprocess_name} not supported.")
        return preprocess


## Behavior
class Treadmill(BehaviorSetup):
    def __init__(self, preprocess_name, method, root_dir=None):
        super().__init__(preprocess_name, method, root_dir)
        self.root_dir_name = f"TRD-{method}"
        self.root_dir = self.root_dir.joinpath(self.root_dir_name)
        # TODO: define correct output files
        # DON-017115_20231120_TRD-2P_S5-ACQ.mat
        self.static_outputs = None
        self.variable_outputs = {self.root_dir: ["*track.py", "*.mat"]}

    # get_file_path is inherited from Output class

    # get_output_paths is inherited from Output class


class Trackball(BehaviorSetup):
    def __init__(self, preprocess_name, method, root_dir=None):
        super().__init__(preprocess_name, method, root_dir)
        root_folder = f"???-{method}"
        raise NotImplementedError("Treadmill setup not implemented yet")


class VR(BehaviorSetup):
    def __init__(self, preprocess_name, method, root_dir=None):
        super().__init__(preprocess_name, method, root_dir)
        root_folder = f"???-{method}"
        raise NotImplementedError("Treadmill setup not implemented yet")


class Cam(BehaviorSetup):
    def __init__(self, preprocess_name, method, root_dir=None):
        super().__init__(preprocess_name, method, root_dir)
        vr_root_folder = "0000VR"  # VR
        cam_root_folder = "0000MC"  # Cam Data
        cam_top_root_folder = "000BSM"  # Top View Mousecam
        cam_top_root_folder = "TR-BSL"  # Top View Mousecam Inscopix
        # TODO: implement cam data loading
        root_folder = f"???-{method}"
        raise NotImplementedError("Treadmill setup not implemented yet")


## Imaging
class Femtonics(NeuralSetup):
    def __init__(self, preprocess_name, method, root_dir=None):
        super().__init__(preprocess_name, method, root_dir)
        self.root_dir_name = f"00{method}-F"
        self.root_dir = self.root_dir.joinpath(self.root_dir_name)
        self.static_outputs = None
        self.variable_outputs = {self.root_dir: ["*.mesc"]}
        self.data_naming_scheme = (
            "{animal_id}_{session_date}_" + self.root_dir_name + "_{task_names}"
        )

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
    def __init__(self, method, root_dir=None):
        super().__init__(method, root_dir)
        self.root_dir = root_dir


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
    Class to manage the conversion of .mat files to .py files
    """

    # TODO: implement mat_to_py manager
    def __init__(self, method, root_dir=None):
        # TODO: implement suite2p manager
        super().__init__(method, root_dir)
        self.static_outputs = None
        self.variable_outputs = {
            self.root_dir: [
                "*_2p_galvo_trigger.npy",
                "*_triggers.npy",
                "*_velocity.npy",
                "*_wheel.npy",
            ]
        }
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

    @property
    def data(self):
        print(asdf)
        return self.output_fnames["binary"]


## Neural
class Inscopix_Processing(Preprocessing):
    def __init__(self, method):
        # TODO: implement inscopix manager
        # TODO: implement inscopix attributes
        raise NotImplementedError("Inscopix preprocessing not implemented yet")


class Suite2p(Preprocessing):
    def __init__(self, method, root_dir=None):
        # TODO: implement suite2p manager
        super().__init__(method, root_dir)
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

    @property
    def data(self):
        print(asdf)
        return self.output_fnames["binary"]
