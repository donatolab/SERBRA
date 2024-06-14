from typing import List, Union, Dict, Any, Tuple, Optional

from pathlib import Path
from Helper import *

# parallel processing
from numba import jit, njit, prange

# loading mesc files
import h5py

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
        needed_attributes = ["method", "preprocessing", "processing"]
        if key != "photon":
            metadata["processing"] = (
                "environment"  # TODO: change this if more behavioral processing methods are added
            )
        check_needed_keys(metadata, needed_attributes)
        self.method = metadata["method"]
        self.preprocess_name = metadata["preprocessing"]
        self.process_name = metadata["processing"]
        self.static_outputs: dict = {}
        self.variable_outputs: dict = {}
        self.own_outputs: List[str] = []
        self.data_naming_scheme = None
        self.identifier: dict = self.extract_identifier(metadata["task_id"])
        self.raw_data_path = None
        self.metadata = metadata

    def define_full_root_dir(self, full_root_dir: Path = None):
        return full_root_dir if full_root_dir else self.root_dir

    def define_raw_data_path(self, fname: str = None):
        if not fname:
            fname = self.data_naming_scheme.format(**self.identifier)
        raw_data_path = self.define_file_path(file_name=fname)
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
            global_logger.error(
                "No full_root_dir provided. Cannot define output paths for variable outputs."
            )
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

    def define_file_path(self, file_name=None, full_root_dir: Path = None):
        if not file_name:
            global_logger.error("No file_name provided.")
            raise ValueError("file_name must be provided to get file path.")

        full_root_dir = self.define_full_root_dir(full_root_dir)
        output_paths = self.get_output_paths(full_root_dir)
        transposed_output_paths = self.transpose_output_paths(output_paths)

        fpath = None
        if file_name in transposed_output_paths.keys():
            fpath = transposed_output_paths[file_name].joinpath(file_name)
        return fpath

    def load_data(
        self,
        fpath: Path = None,
        file_name=None,
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
        if not fpath:
            fpath = self.define_file_path(
                file_name=file_name,
                full_root_dir=full_root_dir,
            )

        if fpath.exists():
            data = cebra.load_data(fpath)
        else:
            print(f"File {file_name} not found in {fpath}")
        return data

    def define_data_path(self, task_id: str = None, fname: str = None):
        identifier = Output.extract_identifier(task_id)
        animal_id, date, task_name = identifier.values()
        fname = f"{animal_id}_{date}_{task_name}_{self.key}.npy" if not fname else fname
        fpath = self.define_file_path(fname)
        return fpath

    def get_data_path(self, task_id: str = None):
        raise NotImplementedError(f"get_data_path not implemented for {self.__class__}")

    def save_data(self, data, overwrite=False):
        for data_name, data_type in data.items():
            animal_id, date, task_name = self.identifier.values()
            fname = f"{animal_id}_{date}_{task_name}_{data_name}"
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

    def get_data_path(self, task_id):
        fpath = self.preprocess.get_data_path(task_id)
        if not fpath:
            fpath = self.define_data_path(task_id)
        return fpath

    def process_data(self, task_id=None):
        animal_id, date, task_name = Output.extract_identifier(task_id)
        # raw_data = self.load_data()
        raw_data_path = self.define_raw_data_path()
        data = self.preprocess.process_data(
            raw_data_path=raw_data_path, task_name=task_name
        )
        return data


class BehavioralSetup(Setup):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.root_dir_name = f"TRD-{self.method}"
        self.root_dir = self.root_dir.joinpath(self.root_dir_name)

    def get_preprocess(self):
        preprocess_name = self.preprocess_name
        if preprocess_name == "rotary_encoder":
            preprocess = RotaryEncoder(
                key=self.key, root_dir=self.data_dir, metadata=self.metadata
            )
        elif preprocess_name == "cam":
            preprocess = Cam(
                key=self.key, root_dir=self.data_dir, metadata=self.metadata
            )
        else:
            global_logger.error(
                f"Preprocessing software {preprocess_name} not supported for {self.__class__}."
            )
            raise ValueError(
                f"Preprocessing software {preprocess_name} not supported for {self.__class__}."
            )
        return preprocess


class NeuralSetup(Setup):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.root_dir_name = f"00{self.method}"
        self.own_outputs = []

    def get_preprocess(self):
        preprocess_name = self.preprocess_name
        if preprocess_name == "suite2p":
            preprocess = Suite2p(
                key=self.key, root_dir=self.root_dir, metadata=self.metadata
            )
        elif preprocess_name == "opexebo":
            preprocess = Opexebo(
                key=self.key, root_dir=self.root_dir, metadata=self.metadata
            )
        else:
            global_logger.error(
                f"Preprocessing software {preprocess_name} not supported for {self.__class__}."
            )
            raise ValueError(
                f"Preprocessing software {preprocess_name} not supported for {self.__class__}."
            )
        return preprocess

    def extract_fps(self):
        raise NotImplementedError(
            f"Extracting fps not implemented yet in {self.__class__}"
        )
        return fps

    def get_fps(self):
        fps = None
        if "fps" in self.metadata.keys():
            fps = self.metadata["fps"]
        else:
            fps = self.extract_fps()
        return fps

#######################           Hardware            ##############################################
class Wheel:
    def __init__(self, radius):
        self.radius = radius  # in meters

    @property
    def circumfrence(self):
        return 2 * np.pi * self.radius



## Behavior
class Treadmill_Setup(BehavioralSetup):
    """
    Class managing the treadmill setup of Steffen.
    Rotation data is stored in .mat files.
    The wheel is connected to a rotary encoder.
    The Belt is segmented in different segments with different stimuli.
    """

    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        # self.root_dir_name = f"TRD-{self.method}"
        # self.root_dir = self.root_dir.joinpath(self.root_dir_name)
        # TODO: add posibility to provide data_naming_scheme in yaml file
        self.data_dir = self.root_dir
        self.data_naming_scheme = (
            "{animal_id}_{date}_" + self.root_dir_name + "_{task_name}-ACQ.mat"
        )
        self.variable_outputs = {self.root_dir: []}
        for output in self.own_outputs:
            self.variable_outputs[self.root_dir].append(output)

        self.raw_data_path = self.define_raw_data_path()
        self.preprocess = self.get_preprocess()

    @staticmethod
    def fetch_rotary_lapsync_data(fpath):
        """
        columns: roatry encoder state1 | roatry encoder state 2 |        lab sync      | frames
        values          0 or   1       |       0 or   1         |0 or 1 reflecor strip | galvosync

        lab sync is not alway detected
        """
        mat = sio.loadmat(fpath)
        data = mat["trainingdata"]

        # get wheel electricity contact values
        rotary_ch_a = data[:, 0]
        rotary_ch_b = data[:, 1]

        lap_sync = np.array(data[:, 2]).reshape(-1, 1)
        galvo_sync = np.array(data[:, 3]).reshape(-1, 1)
        return rotary_ch_a, rotary_ch_b, lap_sync, galvo_sync

    def sync_wheel_to_belt........write this function umklammert mit **************************
    

    def process_data(self, save: bool = True, overwrite: bool = False):
        if self.preprocess_name == "rotary_encoder":
            rotary_ch_a, rotary_ch_b, lap_sync, galvo_sync = self.fetch_rotary_lapsync_data(self.raw_data_path)
        else:
            raise ValueError(f"data loading not supported for {self.preprocess_name} in {self.__class__}.")
        data = self.preprocess.process_data(rotary_ch_a, rotary_ch_b, lap_sync, galvo_sync, max_speed=0.3, save=save, overwrite=overwrite)

        return data[self.key]

    # define_file_path is inherited from Output class
    # get_output_paths is inherited from Output class


class Wheel_Setup(BehavioralSetup):
    """
    Class managing the wheel setup of Steffen.
    Rotation data is stored in .mat files.
    The wheel is connected to a rotary encoder.
    """

    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        # self.root_dir_name = f"TRD-{self.method}"
        # self.root_dir = self.root_dir.joinpath(self.root_dir_name)
        self.static_outputs = {self.root_dir: ["results.zip"]}
        self.data_naming_scheme = "results.npy"

        # TODO: this is not a good way to do it, preprocessing should be defined the dataset class to use rotary encoder or other methods
        self.variable_outputs = {self.root_dir: []}

        # TODO: this should be in preprocessing class, since it is not saved by setup
        for output in self.own_outputs:
            self.variable_outputs[self.root_dir].append(output)

        needed_attributes = ["radius", "clicks_per_rotation", "fps"]
        check_needed_keys(metadata, needed_attributes)

        self.wheel = Wheel(
            radius=metadata["radius"],
        )

        self.preprocess = self.get_preprocess()

    def fetch_rotary_data(fpath):
        """
        columns: roatry encoder state1 | roatry encoder state 2
        values          0 or   1       |       0 or   1        

        lab sync is not alway detected
        """
        data = np.load(fpath)

        # get wheel electricity contact values
        rotary_ch_a = data["rotary_encoder1_abstime"]
        rotary_ch_b = data["rotary_encoder2_abstime"]

        # get reference times for alignment to imaging frames
        return rotary_ch_a, rotary_ch_b, None, None

    def process_data(self, save: bool = True, overwrite: bool = False):
        if self.preprocess_name == "rotary_encoder":
            rotary_ch_a, rotary_ch_b, lap_sync, galvo_sync = self.fetch_rotary_lapsync_data(self.raw_data_path)
        else:
            raise ValueError(f"data loading not supported for {self.preprocess_name} in {self.__class__}.")

        data = self.preprocess.process_data(rotary_ch_a, rotary_ch_b, lap_sync, galvo_sync, max_speed=0.5, save=save, overwrite=overwrite)

        return data[self.key]


class Trackball_Setup(BehavioralSetup):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        root_folder = f"???-{self.method}"
        raise NotImplementedError("Treadmill setup not implemented yet")


class Openfield_Setup(BehavioralSetup):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)

        # TODO: Typically data is gathered through a camera
        self.data_naming_scheme = (
            "UNDEFINDE{animal_id}_{date}_"
            + self.root_dir_name
            + "_{task_name}-UNDEFINED.bla"
        )

        self.variable_outputs = {
            self.root_dir: [
                self.data_naming_scheme,
                "*_locs.npy",
            ]
        }
        for output in self.own_outputs:
            self.variable_outputs[self.root_dir].append(output)

        self.raw_data_path = self.define_raw_data_path()
        needed_attributes = ["environment_dimensions", "imaging_fps"]
        check_needed_keys(metadata, needed_attributes)

        optional_attributes = [
            "stimulus_dimensions",
            "stimulus_sequence",
            "stimulus_type",
        ]
        add_missing_keys(metadata, optional_attributes, fill_value=None)

        self.openfield = Environment(
            dimensions=metadata["environment_dimensions"],
            segment_len=metadata["stimulus_dimensions"],
            segment_seq=metadata["stimulus_sequence"],
            type=metadata["stimulus_type"],
        )

        self.preprocess = self.get_preprocess()

    def process_data(self, task_id=None):
        raise NotImplementedError("Openfield data extraction not implemented yet")


class Active_Avoidance(BehavioralSetup):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        # TODO: Typically data is gathered through a camera
        raise NotImplementedError("Active Avoidance setup not implemented yet")


########################        Imaging          ############################################################
class Femtonics(NeuralSetup):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.root_dir_name += "-F"
        self.root_dir = self.root_dir.joinpath(self.root_dir_name)
        self.data_dir = self.root_dir
        self.data_naming_scheme = (
            "{animal_id}_{date}_" + self.root_dir_name + "_{task_name}.mesc"
        )
        self.variable_outputs = {self.data_dir: [self.data_naming_scheme]}

        # TODO: this should be in preprocessing class, since it is not saved by setup
        for output in self.own_outputs:
            self.variable_outputs[self.data_dir].append(output)
        self.fps = self.get_fps()
        self.preprocess = self.get_preprocess()

    @staticmethod
    def convert_galvo_trigger_signal(galvo_sync):
        """
        value > 0 for when 2P frame is obtained;
        value = 0 in between

        find rising edge of trigger. This is the start of the 2P frame.
        output:
            triggers: indices of the rising edge of the trigger (start of 2P frame).
        """
        indices = np.where(galvo_sync >= 1)[0]
        galvo_trigger = np.zeros(len(galvo_sync))
        galvo_trigger[indices] = 1

        triggers = []
        for idx in indices:
            if galvo_trigger[idx - 1] == 0:
                triggers.append(idx)
        return triggers

    def define_raw_data_path(self, fname: str = None):
        """
        Define the path to the raw data file.
        If no unique file is found it is assumed that the data is stored in a multi file dataset.
        Dataset is choosen if all identifiers(animal_id, date and task_name) are found in the file name.
        """
        if not self.raw_data_path:
            if not fname:
                fname = self.data_naming_scheme.format(**self.identifier)
            raw_data_path = self.define_file_path(file_name=fname)

            # Search for multi file dataset if no unique file is found
            if not raw_data_path.exists():
                allowed_symbols = "\-A-Za-z_0-9"
                animal_id, date, task_name = self.identifier.values()
                regex_search = f"{animal_id}_{date}[{allowed_symbols}]*{task_name}[{allowed_symbols}]*"
                raw_data_paths = get_files(
                    raw_data_path.parent,
                    ending=raw_data_path.suffix,
                    regex_search=regex_search,
                )
                if len(raw_data_paths) == 0:
                    global_logger.error(
                        f"No MESC files found in {raw_data_path.parent} with {self.identifier} in name. Needed for imaging frame rate determination. Alternative could be to provide the fps rate in the neural metadata section in the yaml file."
                    )
                    raise FileNotFoundError(
                        f"No MESC files found in {raw_data_path.parent} with {self.identifier} in name. Needed for imaging frame rate determination. Alternative could be to provide the fps rate in the neural metadata section in the yaml file."
                    )
                else:
                    print(
                        f"WARNING: File {raw_data_path} not found. Choosing first matching multi task dataset found in {raw_data_path.parent}."
                    )
                    global_logger.warning(
                        f"File {raw_data_path} not found. Choosing first matching multi task dataset found in {raw_data_path.parent}. Needed for imaging frame rate determination. Alternative could be to provide the fps rate in the neural metadata section in the yaml file."
                    )
                    raw_data_path = raw_data_path.parent.joinpath(raw_data_paths[0])

            self.raw_data_path = raw_data_path
        return self.raw_data_path

    def extract_fps(self):
        """
        Extract the frame rate from the MESC file.
        """
        mesc_fpath = self.define_raw_data_path()
        fps = None
        with h5py.File(mesc_fpath, "r") as file:
            msessions = [
                msession_data
                for name, msession_data in file.items()
                if "MSession" in name
            ]
            msession = msessions[0]
            # msession_attribute_names = list(msession.attrs.keys())
            munits = (
                [munit_data for name, munit_data in msession.items() if "MUnit" in name]
                if len(msessions) > 0
                else []
            )
            # get frame times
            frTimes = (
                [
                    munit.attrs["ZAxisConversionConversionLinearScale"]
                    for munit in munits
                ]
                if len(munits) > 0
                else None
            )
            if frTimes:
                frTime = max(frTimes)  # in milliseconds
                fps = 1000 / frTime
        return fps

    def process_data(self, save: bool = True, overwrite: bool = False):
        raise NotImplementedError(
            f"Data processing not implemented for {self.__class__}"
        )
        tiff_data = self.raw_to_tiff(self.raw_data_path)
        suite2p_data = self.rotary_encoder.extract_data(tiff_data)
        cabincorr_data = self.treadmill.extract_data(suite2p_data)
        data = {**suite2p_data, **cabincorr_data}

        if save:
            self.save_data(data, overwrite)
        return data[self.key]


class Thorlabs(NeuralSetup):
    def __init__(self):
        self.root_dir_name += "-T"
        self.root_dir = self.root_dir.joinpath(self.root_dir_name)
        self.data_dir = self.root_dir_name.joinpath("data")
        self.static_outputs = {self.data_dir: ["Image_001_001.raw"]}
        # TODO: output files not defined
        raise NotImplementedError("Thorlabs setup not implemented yet")


class Inscopix(NeuralSetup):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.root_dir_name += "-I"
        self.root_dir = self.root_dir.joinpath(self.root_dir_name)
        self.data_dir = self.root_dir
        self.static_outputs = {}
        # TODO: output files not defined
        self.data_naming_scheme = (
            "UNDEFINED----{animal_id}_{date}_"
            + self.root_dir_name
            + "_{task_name}.UNDEFINED"
        )
        self.variable_outputs = {self.data_dir: [self.data_naming_scheme]}

        # TODO: this should be in preprocessing class, since it is not saved by setup
        for output in self.own_outputs:
            self.variable_outputs[self.data_dir].append(output)
        self.fps = self.get_fps()
        self.preprocess = self.get_preprocess()

    def define_raw_data_path(self, fname: str = None):
        raise NotImplementedError(
            f"Raw data path definition not implemented for {self.__class__}"
        )

    def extract_fps(self):
        raise NotImplementedError("{self.__class__} extract_fps not implemented yet")


# Preprocessing Classes


#######################       Preprocessing         ############################################################
class Preprocessing(Output):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.root_dir = root_dir

    def get_data_path(self, task_id):
        fpath = self.process.get_data_path(task_id)
        if not fpath:
            fpath = self.define_data_path(task_id)
        return fpath

    def process_data(self, task_id=None):
        animal_id, date, task_name = Output.extract_identifier(task_id)
        # raw_data = self.load_data(identifier=task_name)
        # TODO: implement intern processing data of data bevore sending to process
        raw_data_path = self.define_raw_data_path()
        data = self.process.process_data(
            raw_data_path=raw_data_path, task_name=task_name
        )
        return data


class Behavior_Preprocessing(Preprocessing):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.own_outputs = [
            "{animal_id}_{date}_{task_name}_position.npy",
            "{animal_id}_{date}_{task_name}_distance.npy",
        ]

    def get_process(self):
        process_name = self.process_name
        if process_name == "environment":
            process = Environment(
                key=self.key, root_dir=self.data_dir, metadata=self.metadata
            )
        else:
            raise ValueError(
                f"processing software {process_name} not supported for {self.__class__}."
            )
        return process


class Neural_Preprocessing(Preprocessing):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)

    def get_process(self):
        process_name = self.process_name
        if process_name == "cabincorr":
            process = CaBinCorr(
                key=self.key, root_dir=self.data_dir, metadata=self.metadata
            )
        else:
            raise ValueError(
                f"processing software {process_name} not supported for {self.__class__}."
            )
        return process


class RotaryEncoder(Behavior_Preprocessing):
    """
    The rotary encoder is used to measure the amount of rotation.
    This can be used to calculate the distance moved by the wheel.
    The rotary encoder is connected to a wheel.

    - needed_metadata: radius, clicks_per_rotation, fps
    """

    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.data_dir = self.root_dir
        self.variable_outputs = {self.root_dir: []}
        for output in self.own_outputs:
            self.variable_outputs[self.root_dir].append(output)

        needed_attributes = ["radius", "fps", "clicks_per_rotation"]
        check_needed_keys(metadata, needed_attributes)
        self.wheel = Wheel(
            radius=metadata["radius"],
        )

        self.sample_rate = self.metadata["fps"]  # 10kHz
        self.clicks_per_rotation = self.metadata["clicks_per_rotation"]

        self.process = self.get_process()

    def preprocess_data(self, 
                        rotary_ch_a: np.ndarray,
                        rotary_ch_b: np.ndarray,
                        lap_sync: np.ndarray = None,
                        galvo_sync: np.ndarray = None,
                        max_speed: float = 0.6,
                        save: bool = True,
                        overwrite: bool = False,
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

**************************This functions below
        if lap_sync is None or galvo_sync is None:
            lap_start_frame = 0
        else:
            # get galvo trigger times for 2P frame
            galvo_triggers_times = Femtonics.convert_galvo_trigger_signal(galvo_sync)

            needed_attributes = ["imaging_fps"]
            check_needed_keys(self.metadata, needed_attributes)
            if not imaging_fps:
                imaging_fps = self.metadata["imaging_fps"]

            ## get distance of the wheel surface
            moved_distances_in_frame = rotary_encoder.convert_data_fps_to_imaging_fps(
                data=rotary_distances,
                reference_times=galvo_triggers_times,
                imaging_fps=imaging_fps,
            )

            # syncronise moved distances, scale rotary encoder data to lap sync signal because of slipping wheel
            distances, lap_start_frame = rotary_encoder.sync_to_lap(
                moved_distances_in_frame,
                imaging_fps=imaging_fps,
                reference_times=galvo_triggers_times,
                lap_sync=lap_sync,
                track_dimensions=self.process.dimensions,
                max_speed=max_speed,  # in m/s,
            )
**************************This functions above
            cumulative_distance = np.cumsum(distances)

        data = {
            "distance": cumulative_distance,
        }
        return data, lap_start_frame

    def process_data(
        self,
        rotary_ch_a: np.ndarray,
        rotary_ch_b: np.ndarray,
        lap_sync: np.ndarray = None,
        galvo_sync: np.ndarray = None,
        max_speed: float = 0.6,  # in m/s
        save: bool = True,
        overwrite: bool = False,
    ):
        data, lap_start_frame = self.preprocess_data(
            rotary_ch_a=rotary_ch_a,
            rotary_ch_b=rotary_ch_b,
            lap_sync=lap_sync,
            galvo_sync=galvo_sync,
            max_speed=max_speed
            save=save,
            overwrite=overwrite,
            )

        processed_data = self.process.process_data(
            cumulative_distance=data["distance"],
            lap_start_frame=lap_start_frame,
            save=save,
            overwrite=overwrite,
        )

        data = {**data, **processed_data}
        
        return data

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
        track_dimensions: float,  # in meters
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
            - track_dimensions: float
                The dimensions of the track in meters.
            - max_speed: float, optional (default=0.5)
                The maximum speed of movement in meters per second.

        Returns:
            - fited_moved_distance_in_frame: np.ndarray
                Array of moved distances in frames, adjusted based on lap sync signals.
            - lap_start_frame: int
                The index of the first frame where a lap start signal is detected.

        This function performs the following steps:
        1. Determines the time window (`mute_lap_detection_time`) in which lap detection is muted, based on `track_dimensions` and `max_speed`.
        2. Identifies unique lap start indices by muting subsequent lap start signals within the determined time window.
        3. Maps lap start signals to imaging frames using `reference_times`.
        4. Finds the first lap start signal that occurs after the `start_frame_time`.
        5. Adjusts the moved distances between lap start signals to account for possible discrepancies due to the wheel continuing to move after the subject stops.
        6. Applies the mean adjustment ratio to distances before the first and after the last detected lap start signals to create a more realistic data set.
        """
        if isinstance(track_dimensions, list):
            track_dimensions = track_dimensions[0]

        mute_lap_detection_time = track_dimensions / max_speed
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
            probable_moved_laps = round(track_dimensions / moved_distances_between_laps)
            real_moved_distance = track_dimensions * probable_moved_laps

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


class Cam(Behavior_Preprocessing):
    """
    vr_root_folder = "0000VR"  # VR
    cam_root_folder = "0000MC"  # Cam Data
    cam_top_root_folder = "000BSM"  # Top View Mousecam
    cam_top_root_folder = "TR-BSL"  # Top View Mousecam Inscopix
    """

    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.data_dir = self.root_dir.joinpath("000BSM")
        self.static_outputs = {self.data_dir: ["NoClue.data"]}

        self.output_path = self.data_dir.parent.joinpath(f"TRD-{self.method}")
        self.variable_outputs = {self.self.output_path: []}

    def process_data(self, raw_data, task_name=None, save=True):
        raise NotImplementedError(
            f"cam data processing not implemented for {self.__class__}"
        )


class Suite2p(Neural_Preprocessing):
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
                "cell_drying.npy",
                "data.bin",
            ]
        }
        self.variable_outputs = {self.data_dir: []}

        self.ops_settings = {
            "tau": 1,  # 0.7 for GCaMP6f,   1.0 for GCaMP6m,    1.25-1.5 for GCaMP6s
            "max_overlap": 0.75,  # percentage of allowed overlap between cells
        }
        self.process = self.get_process()

    def process_data(self, raw_data, task_name=None, save=True):
        raise NotImplementedError(
            f"Suite2p data processing not implemented for {self.__class__}"
        )

    # get_data_path is inherited from Output class
    # process_raw_data(self, task_name) is inherited from Preprocessing class
    # load_data is inherited from Output class
    # self.load_data(file_name, full_root_dir=self.root_dir)
    # define_file_path is inherited from Output class


class Opexebo(Neural_Preprocessing):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.data_dir = self.root_dir
        self.static_outputs = {}

    def load_settings(self, settings_file):
        # TODO: implement opexebo settings loading
        raise NotImplementedError(
            f"Inscopix settings not implemented for {self.__class__}"
        )

    def process_data(self, raw_data, task_name=None, save=True):
        # TODO: Provide possibiltiy to load settings from file, because Nathalie has a lot of different settings
        raise NotImplementedError(
            f"Inscopix data processing not implemented for {self.__class__}"
        )


#######################       Processing         ############################################################
class Processing(Output):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key, root_dir, metadata)

    def get_data_path(self, task_id: str = None):
        return self.define_data_path(task_id)

    def process_data(self, raw_data_path: Path, task_name: str = None):
        raise NotImplementedError(
            f"Data processing not implemented for {self.__class__}"
        )


class Behavior_Processing(Processing):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.root_dir = root_dir
        self.own_outputs = [
            "{animal_id}_{date}_{task_name}_velocity.npy",
            "{animal_id}_{date}_{task_name}_position.npy",
            "{animal_id}_{date}_{task_name}_distance.npy",
            "{animal_id}_{date}_{task_name}_acceleration.npy",
            "{animal_id}_{date}_{task_name}_stimulus.npy",
            "{animal_id}_{date}_{task_name}_moving.npy",
        ]


class Neural_Processing(Processing):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.root_dir = root_dir
        self.own_outputs = ["{animal_id}_{date}_{task_name}_photon.npy"]


class Environment(Behavior_Processing):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.data_dir = self.root_dir

        self.variable_outputs = {self.data_dir: []}
        for output in self.own_outputs:
            self.variable_outputs[self.data_dir].append(output)

        needed_attributes = ["environment_dimensions", "imaging_fps"]
        check_needed_keys(metadata, needed_attributes)
        optional_attributes = [
            "stimulus_dimensions",
            "stimulus_sequence",
            "stimulus_type",
        ]
        add_missing_keys(metadata, optional_attributes, fill_value=None)

        self.dimensions = make_list_ifnot(metadata["environment_dimensions"])  # in m
        self.segment_len = (
            make_list_ifnot(metadata["stimulus_dimensions"])
            if metadata["stimulus_dimensions"] is not None
            else None
        )  # [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        self.segment_seq = (
            make_list_ifnot(metadata["stimulus_sequence"])
            if metadata["stimulus_sequence"] is not None
            else None
        )  # [1, 2, 3, 4, 5, 6],
        self.type = type if type is not None else None  # "A",

    @staticmethod
    def get_position_from_cumdist(
        cumulative_distance: np.ndarray,
        dimensions: List[int],
        lap_start_frame: float = None,
    ):
        """
        Get position of the animal on the 1D or 2D track.
        output:
            position: position of the animal on the track
        """
        # shift distance up, so lap_start is at 0
        lap_start_frame = 0 if lap_start_frame is None else lap_start_frame

        shifted_distance = cumulative_distance + (
            180 - cumulative_distance[lap_start_frame]
        )

        positions = np.zeros_like(shifted_distance)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        for dimension in range(len(dimensions)):
            environment_len = dimensions[dimension]
            started_lap = 1
            ended_lap = 0
            undmapped_positions = True
            while undmapped_positions:
                distance_min = ended_lap * environment_len
                distance_max = started_lap * environment_len
                distances_in_range = np.where(
                    (shifted_distance >= distance_min)
                    & (shifted_distance < distance_max)
                )[0]

                positions[dimension][distances_in_range] = (
                    shifted_distance[distances_in_range] - distance_min
                )

                if max(shifted_distance) > distance_max:
                    ended_lap += 1
                    started_lap += 1
                else:
                    undmapped_positions = False
        return positions

    @staticmethod
    def get_velocity_from_cumdist(
        cumulative_distance: np.ndarray, imaging_fps: float, smooth=True
    ):
        if cumulative_distance.ndim == 1:
            cumulative_distance = cumulative_distance.reshape(1, -1)
        velocity = np.diff(cumulative_distance, axis=1) * imaging_fps

        if smooth:
            velocity_smoothed = butter_lowpass_filter(
                velocity, cutoff=2, fs=imaging_fps, order=2
            )
        else:
            velocity_smoothed = velocity
        return velocity_smoothed

    @staticmethod
    def get_acceleration_from_velocity(
        velocity: np.ndarray, imaging_fps: float, smooth=False
    ):
        """
        Get acceleration from velocity. Be cautious with the smoothing, it can lead to wrong results. Smoothing parameters are not well defined

        """
        if velocity.ndim == 1:
            velocity = velocity.reshape(1, -1)

        acceleration = np.diff(velocity, axis=1) * imaging_fps
        if smooth:
            print(
                f"WARNING: Smoothing acceleration can lead to wrong results. Be cautious. Smoothing parameters are not well defined."
            )
            velocity_smoothed = butter_lowpass_filter(
                acceleration, cutoff=0.2, fs=imaging_fps, order=2
            )
        else:
            velocity_smoothed = acceleration

        return velocity_smoothed

    @staticmethod
    def create_stimulus_map(
        segment_dimensions: np.ndarray, segment_seq: List[List[int]], bin=0.01
    ):
        """
        Create a map of the environment with stimulus at position. ONLY FOR 1D ENVIRONMENTS
        Attributes:
            segment_dimensions: list of segment lenghts for each dimension
            segment_seq: list of stimulus sequence for each dimension

        Output:
            map: map of the environment with stimulus at position

        Examples:

            1D:
                segment_dimensions=array([[0.3],
                                    [0.3],
                                    [0.3],
                                    [0.3],
                                    [0.3],
                                    [0.3]])
                segment_seq=array([[1],
                                    [2],
                                    [3],
                                    [4],
                                    [5],
                                    [6]])
            2D:
                segment_dimensions = np.array([[0.3, 0.3],
                                                [0.3, 0.3],
                                                [0.3, 0.3]])
                segment_seq = [[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]]
        """
        # TODO: implement for 2D environments
        # Check if the length of segment_dimensions matches the length of segment_seq
        if len(segment_dimensions) != len(segment_seq):
            raise ValueError(
                "The length of segment_dimensions and segment_seq must match."
            )

        num_dimensions = segment_dimensions[0].ndim
        # Initialize an empty dictionary to hold the map
        max_positions = [None] * num_dimensions
        for dim in range(num_dimensions):
            max_positions[dim] = np.sum(segment_dimensions, axis=0)[0] / bin

        env_map = np.cumsum(segment_dimensions)
        return env_map

    @staticmethod
    def get_stimulus_at_position(
        positions: np.ndarray,
        segment_dimensions: List[List[float]],
        segment_seq: List[List[int]],
        max_position: List[float],
    ):
        """
        Get stimulus type at the position.
        output:
            stimulus: stimulus type at the position
        """
        segment_dimensions = np.array(segment_dimensions)
        segment_seq = np.array(segment_seq)

        if segment_dimensions.ndim == 1:
            segment_dimensions = segment_dimensions.reshape(-1, 1)
        if segment_seq.ndim == 1:
            segment_seq = segment_seq.reshape(-1, 1)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)

        if not positions.shape[0] == segment_dimensions[0].ndim == segment_seq[0].ndim:
            raise ValueError(
                f"Number of dimensions must be the same as the segment dimensions and sequence. Got {positions.ndim} and {segment_dimensions[0].ndim} and {segment_seq[0].ndim}"
            )

        stimulus_map = Environment.create_stimulus_map(segment_dimensions, segment_seq)
        if stimulus_map[-1] != max_position[0]:
            raise ValueError(
                f"Size of stimulus_map {stimulus_map[-1]} must be the same as the max position {max_position[0]}. Ensure that segment dimensions and sequence are correct."
            )

        # Broadcast and compare continuouses against track boundaries
        stimulus_type_indices = np.sum(positions.reshape(-1, 1) >= stimulus_map, axis=1)
        stimulus_type_at_frame = segment_seq[stimulus_type_indices % len(segment_seq)]
        stimulus_type_at_frame = stimulus_type_at_frame.reshape(1, -1)[0]
        return stimulus_type_at_frame

    def process_data(
        self,
        imaging_fps: float,
        cumulative_distance: np.ndarray = None,
        positions: np.ndarray = None,
        lap_start_frame: float = None,
        save: bool = True,
        overwrite: bool = False,
    ):
        """
        Extract data from the cumulative distance of the belt.
        output:
            data: dictionary with
                - position: position of the animal on the belt
                - stimulus: stimulus type at the position if Environment segments are provided
        """
        if positions is None and cumulative_distance is None:
            raise ValueError(
                f"No distance or position provided. Cannot extract data for {self.__class__}"
            )
        elif positions is None:
            positions = self.get_position_from_cumdist(
                cumulative_distance,
                dimensions=self.dimensions,
                lap_start_frame=lap_start_frame,
            )
        elif cumulative_distance is None:
            cumulative_distance = np.cumsum(positions)

        # TODO: check for multi dimensions!!!!!
        # TODO: check for multi dimensions!!!!!
        # TODO:.......................... check for multi dimensions!!!!!
        # TODO: check for multi dimensions!!!!!

        velocity_smoothed = self.get_velocity_from_cumdist(
            cumulative_distance, imaging_fps, smooth=True
        )

        acceleration = self.get_acceleration_from_velocity(
            velocity_smoothed, imaging_fps, smooth=False
        )

        data = {
            "distance": cumulative_distance,
            "position": positions,
            "velocity": velocity_smoothed,
            "acceleration": acceleration,
        }

        if self.segment_len is None or self.segment_seq is None:
            print(
                "No segment dimensionss or sequence provided. Not extracting stimulus."
            )
        else:
            stimulus = self.get_stimulus_at_position(
                positions,
                segment_dimensions=self.segment_len,
                segment_seq=self.segment_seq,
                max_position=self.dimensions,
            )
            data["stimulus"] = stimulus

        if save:
            self.save_data(data, overwrite)

        return data


class CaBinCorr(Neural_Processing):
    def __init__(self, key, root_dir=None, metadata={}):
        super().__init__(key=key, root_dir=root_dir, metadata=metadata)
        self.data_dir = self.root_dir
        self.static_outputs = {self.data_dir: ["binarized_traces.npz"]}
        self.data_naming_scheme = (
            "*binarized_traces*.npz"
            # "{animal_id}_{date}_" + self.root_dir_name + "_{task_name}.mesc"
        )
        self.variable_outputs = {self.data_dir: [self.data_naming_scheme]}

        self.output_fnames = {
            "F_filtered": "F_filtered.npy",
            "F_onphase": "F_onphase.npy",
            "F_upphase": "F_upphase.npy",
        }

        for output in self.own_outputs:
            self.variable_outputs[self.data_dir].append(output)

    def process_data(self, raw_data, task_name=None, save=True):
        raise NotImplementedError(
            f"CaBinCorr data processing not implemented for {self.__class__}"
        )
