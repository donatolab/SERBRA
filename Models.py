# type hints
from typing import List, Union, Dict, Any, Tuple, Optional

# show progress bar
from tqdm import tqdm, trange

# calculations
import numpy as np
import sklearn
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)

import scipy

# parralelize
from numba import jit, njit, prange
from numba import cuda  # @jit(target='cuda')

# manifolds
from cebra import CEBRA
import cebra
import cebra.integrations.sklearn.utils as sklearn_utils

# own
from Datasets import Datasets, Dataset
from Helper import *


class Models:
    def __init__(self, model_dir, model_id, model_settings=None, **kwargs):
        if not model_settings:
            model_settings = kwargs
        place_cell_settings = (
            model_settings["place_cell"] if "place_cell" in model_settings else None
        )
        cebra_cell_settings = (
            model_settings["cebra"] if "cebra" in model_settings else None
        )
        self.place_cell = PlaceCellDetectors(model_dir, model_id, place_cell_settings)
        self.cebras = Cebras(model_dir, model_id, cebra_cell_settings)

    def define_model_name(
        self,
        model_type,
        model_name=None,
        shuffled=False,
        movement_state="all",
        split_ratio=1,
        model_settings=None,
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

        if model_settings is not None:
            max_iterations = model_settings["max_iterations"]
            model_name = f"{model_name}_iter-{max_iterations}"
        return model_name

    def get_model(self, model_name, model_type, pipeline="cebra", model_settings=None):
        models_class = self.get_model_class(pipeline)

        model_settings = model_settings[pipeline]
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

    def get_pipeline_models(
        self,
        manifolds_pipeline="cebra",
        model_naming_filter_include: List[List[str]] = None,  # or [str] or str
        model_naming_filter_exclude: List[List[str]] = None,  # or [str] or str
    ):
        models_class = self.get_model_class(manifolds_pipeline)
        models = models_class.get_models(
            model_naming_filter_include, model_naming_filter_exclude
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

    def create_embeddings(
        self,
        models=None,
        to_transform_data=None,
        to_2d=False,
        model_naming_filter_include: List[List[str]] = None,  # or [str] or str
        model_naming_filter_exclude: List[List[str]] = None,  # or [str] or str
        manifolds_pipeline="cebra",
        save=False,
        return_labels=False,
    ):
        if not type(to_transform_data) == np.ndarray:
            global_logger.warning(f"No data to transform given. Using model training data.")
            print(f"No data to transform given. Using model training data.")

        model_class = self.get_model_class(manifolds_pipeline)

        models = model_class.get_models(
            model_naming_filter_include, model_naming_filter_exclude
        )

        embeddings = model_class.create_embeddings(
            models=models,
            to_transform_data=to_transform_data,
            to_2d=to_2d,
            save=save,
        )
        if return_labels:
            labels = []
            for model_name, model in models.items():
                labels.append(model_name)
            return embeddings, labels
        
        return embeddings

    def is_model_fitted(self, model, pipeline="cebra"):
        return self.get_model_class(pipeline).is_fitted(model)

    def train_model(self,
        neural_data,
        model=None,
        behavior_data=None,
        pipeline="cebra",
        model_type="time",
        model_name=None,
        idx_to_keep=None,
        shuffle=False,
        movement_state="moving",
        split_ratio=1,
        model_settings=None,
        create_embeddings=False,
        regenerate=False,
              ):
        if not is_dict_of_dicts(model_settings):
            model_settings = {pipeline: model_settings}
        model_name = self.define_model_name(
            model_type,
            model_name,
            shuffle,
            movement_state,
            split_ratio,
            model_settings[pipeline],
        )

        model = model or self.get_model(
            pipeline=pipeline,
            model_name=model_name,
            model_type=model_type,
            model_settings=model_settings,
        )

        neural_data_train, neural_data_test = Dataset.manipulate_data(neural_data, 
                                                                      idx_to_keep=idx_to_keep,
                                                                      shuffle=shuffle,
                                                                      split_ratio=split_ratio)
        
        if behavior_data is not None:
            behavior_data_train, behavior_data_test = Dataset.manipulate_data(behavior_data, 
                                                                              idx_to_keep=idx_to_keep,
                                                                              shuffle=shuffle,
                                                                              split_ratio=split_ratio)


        model_class = self.get_model_class(pipeline)
        
        model  = model_class.train(
            model=model,
            model_type=model_type,
            neural_data_train=neural_data_train,
            neural_data_test=neural_data_test,
            behavior_data_train=behavior_data_train,
            regenerate=regenerate,
        )

        model.data = {
            "train": {
                "neural": neural_data_train,
                "behavior": behavior_data_train,
                "embedding": None,
            },
            "test": {
                "neural": neural_data_test,
                "behavior": behavior_data_test,
                "embedding": None,
            },
        }
        
        if create_embeddings:
            train_embedding = model_class.create_embedding(model, to_transform_data=neural_data_train)
            test_embedding = model_class.create_embedding(model, to_transform_data=neural_data_test)
        
        model.data["train"]["embedding"] = train_embedding,
        model.data["test"]["embedding"] = test_embedding,

        return model


    def get_model_class(self, pipeline="cebra"):
        if pipeline == "cebra":
            models_class = self.cebras
        else:
            raise ValueError(f"Pipeline {pipeline} not supported. Choose 'cebra'.")
        return models_class

class ModelsWrapper:
    """
    Meta Class for Models wrapper
    """

    def __init__(self, model_dir, model_settings=None, **kwargs):
        self.model_dir = Path(model_dir)
        dir_exist_create(self.model_dir)
        self.model_settings = model_settings or kwargs
        self.models = {}

    def get_models(
        self,
        model_naming_filter_include: List[List[str]] = None,  # or [str] or str
        model_naming_filter_exclude: List[List[str]] = None,  # or [str] or str):
    ):
        filtered_models = filter_dict_by_properties(
            self.models, model_naming_filter_include, model_naming_filter_exclude
        )
        return filtered_models


class Model:
    def __init__(self, model_dir, model_id, model_settings=None, **kwargs):
        self.model_dir = model_dir
        self.model_id = model_id
        self.model_settings = model_settings or kwargs
        dir_exist_create(self.model_dir)

    def create_defaul_model(self):
        raise NotImplementedError(
            f"create_defaul_model not implemented for {self.__class__}"
        )

    def define_parameter_save_path(self, model):
        raise NotImplementedError(
            f"define_parameter_save_path not implemented for {self.__class__}"
        )

    def is_fitted(self, model):
        raise NotImplementedError(f"fitted not implemented for {self.__class__}")

    def load_fitted_model(self, model):
        raise NotImplementedError(
            f"load_fitted_model not implemented for {self.__class__}"
        )

    def init_model(self, model_settings_dict):
        default_model = self.create_defaul_model()
        if len(model_settings_dict) == 0:
            model_settings_dict = self.model_settings
        initial_model = define_cls_attributes(self, model_settings_dict, override=True)
        initial_model.fitted = False
        return initial_model

    def model_settings_start(self, name, model_settings_dict):
        model = self.init_model(model_settings_dict)
        model.name = name
        return model

    def model_settings_end(self, model):
        model.save_path = self.define_parameter_save_path(model)
        model = self.load_fitted_model(model)
        return model


        return model

class PlaceCellDetectors(ModelsWrapper):
    def __init__(self, model_dir, model_id, model_settings=None, **kwargs):
        super().__init__(model_dir, model_settings, **kwargs)
        self.si_model = self.spatial_information(name=model_id)
        self.rate_map = None
        self.time_map = None

    def spatial_information(self, name="si", model_settings=None, **kwargs):
        model_settings = model_settings or kwargs
        model_dir = self.model_dir.joinpath("spatial_information")

        model = SpatialInformation(
            model_dir, model_id=name, model_settings=model_settings
        )
        self.models[model.name] = model
        return model

    def get_maps(
        self,
        activity,
        binned_pos,
        smooth=True,
        window_size=2,
        max_bin=None,
    ):
        # #FIXME: ..............add uncommented line again
        # if self.rate_map is None or self.time_map is None:
        if True:
            self.rate_map, self.time_map = self.get_rate_time_map(
                activity,
                binned_pos,
                smooth=smooth,
                window_size=window_size,
                max_bin=max_bin,
            )
        return self.rate_map, self.time_map

    @staticmethod
    def get_time_map(binned_pos, range=None, bins=None, fps=None):
        """
        calculates time in binned_position counts per frame
        """
        # TODO: 2D occupancie map
        # hist, yedges, xedges = np.histogram2d(in_range_y, in_range_x, bins=bins, range=limits)
        #    edges = [xedges, yedges]
        time_map, bins_edges = np.histogram(binned_pos, bins=bins, range=range)
        return time_map, bins_edges

    @staticmethod
    @njit(parallel=True)
    def get_spike_map(activity, binned_pos, max_bin=None):
        """
        Computes the spike map for given neural activity and binned positions.

        Args:
            activity (np.ndarray): A 2D array where each row represents the neural activity at a specific time frame.
            binned_pos (np.ndarray): A 1D array where each element is the binned position corresponding to each time frame.
            max_bin (int, optional): The maximum bin value for the positions. If not provided, it will be computed as one more than the maximum value in `binned_pos`.

        Returns:
            np.ndarray: A 2D array where each row represents a cell, and each column represents a bin. The value at (i, j) represents the summed activity of cell `i` at bin `j`.
        """
        activity_2d = activity
        if activity.ndim == 1:
            activity_2d = activity_2d.reshape(1, -1)
        elif activity.ndim > 2:
            raise ValueError("Activity data has more than 2 dimensions.")

        num_cells = activity_2d.shape[1]
        # for every frame count the activity of each cell
        max_bin = max_bin or max(binned_pos) + 1
        spike_map = np.zeros((num_cells, max_bin))
        for frame in prange(len(activity_2d)):
            rate_map_vec = activity_2d[frame]
            pos_at_frame = binned_pos[frame]
            spike_map[:, pos_at_frame] += rate_map_vec
        return spike_map

    @staticmethod
    # @njit(parallel=True)
    def get_spike_map_per_laps(cell_neural_data_by_laps, binned_pos_by_lap, max_bin):
        """
        Computes the spike map for all laps.

        Args:
            cell_neural_data_by_laps (list of np.ndarray): A list where each element is a 2D array representing the neural activity for each lap.
            binned_pos_by_lap (list of np.ndarray): A list where each element is a 1D array representing the binned positions for each lap.
            max_bin (int): The maximum bin value for the positions.

        Returns:
            np.ndarray: A 3D array where the first dimension represents laps, the second dimension represents cells, and the third dimension represents bins. The value at (i, j, k) represents the summed activity of cell `j` at bin `k` during lap `i`.
        """
        # count spikes at position
        num_laps = len(cell_neural_data_by_laps)
        num_cells = cell_neural_data_by_laps[0].shape[1]
        cell_lap_activity = np.zeros((num_cells, num_laps, max_bin))
        for lap, (cell_neural_by_lap, lap_pos) in enumerate(
            zip(cell_neural_data_by_laps, binned_pos_by_lap)
        ):

            # this should be the following function, but was changed to the following for numba compatibility
            cells_spikes_at = PlaceCellDetectors.get_spike_map(
                cell_neural_by_lap, lap_pos, max_bin
            )
            ##################### numba compatibility #####################
            # activity_2d = cell_neural_by_lap
            # if activity_2d.ndim == 1:
            #    activity_2d = activity_2d.reshape(-1, 1)
            # elif activity_2d.ndim > 2:
            #    raise ValueError("Activity data has more than 2 dimensions.")
            #
            # num_cells = activity_2d.shape[1]
            ## for every frame count the activity of each cell
            # spike_map = np.zeros((num_cells, max_bin))
            # for frame in prange(len(activity_2d)):
            #    rate_map_vec = activity_2d[frame]
            #    pos_at_frame = lap_pos[frame]
            #    spike_map[:, pos_at_frame] += rate_map_vec
            # counts_at = = spike_map
            ##################### numba compatibility #####################
            cell_lap_activity[:, lap, :] = cells_spikes_at
        return cell_lap_activity

    def extract_all_spike_map_per_lap(
        self, cell_ids, neural_data_by_laps, binned_pos_by_laps, max_bin
    ):
        """
        Extracts the spike map for each cell across all laps.

        Args:
            cell_ids (list): A list of cell IDs.
            neural_data_by_laps (list of list of np.ndarray): A list where each element is a list of 2D arrays representing the neural activity for each lap for each cell.
            binned_pos_by_laps (list of list of np.ndarray): A list where each element is a list of 1D arrays representing the binned positions for each lap for each cell.
            max_bin (int): The maximum bin value for the positions.

        Returns:
            np.ndarray: A 4D array where the first dimension represents cells, the second dimension represents laps, the third dimension represents bins. The value at (i, j, k) represents the summed activity of cell `i` at bin `k` during lap `j`.
        """
        cell_lap_activities = np.zeros(
            (len(cell_ids), len(neural_data_by_laps), len(binned_pos_by_laps[0]))
        )
        for cell_id in range(len(cell_ids)):
            cell_neural_data_by_laps = neural_data_by_laps[cell_id]
            cell_lap_activity = self.get_spike_map_per_laps(
                cell_neural_data_by_laps, binned_pos_by_laps, max_bin
            )
            cell_lap_activities[cell_id] = cell_lap_activity

    @staticmethod
    def get_spike_maps_per_laps(cell_ids, neural_data, behavior: Datasets):
        """
        Computes the spike map for specified cells across all laps.

        Args:
            cell_ids (list): A list of cell IDs to compute the spike maps for.
            neural_data (np.ndarray): A 2D array representing the neural activity.
            behavior (Datasets): A dataset containing behavioral data including position information.

        Returns:
            dict: A dictionary where keys are cell IDs and values are dictionaries containing the spike maps for each lap.
        """
        cell_ids = make_list_ifnot(cell_ids)

        binned_pos = behavior.position.binned_data
        max_bin = behavior.position.max_bin or max(binned_pos) + 1

        # get neural_data for each cell
        wanted_neural_data = neural_data[:, cell_ids]
        cell_neural_data_by_laps = behavior.split_by_laps(wanted_neural_data)
        binned_pos_by_laps = behavior.split_by_laps(binned_pos)

        # get spike map for each lap
        cell_lap_activities = PlaceCellDetectors.get_spike_map_per_laps(
            cell_neural_data_by_laps, binned_pos_by_laps, max_bin=max_bin
        )

        cell_activity_dict = {}
        for cell_id, cell_lap_activity in zip(cell_ids, cell_lap_activities):
            cell_activity_dict[cell_id] = {}
            cell_activity_dict[cell_id]["lap_activity"] = cell_lap_activity
        return cell_activity_dict

    @staticmethod
    def get_rate_map(activity, binned_pos, max_bin=None):
        """
        outputs spike rate per position per time
        """
        spike_map = PlaceCellDetectors.get_spike_map(
            activity, binned_pos, max_bin=max_bin
        )
        # smooth and normalize
        # normalize by spatial occupancy
        time_map, _ = PlaceCellDetectors.get_time_map(
            binned_pos, bins=len(spike_map[0])
        )
        rate_map_occupancy = spike_map / time_map
        rate_map_occupancy = np.nan_to_num(rate_map_occupancy, nan=0.0)
        
        # normalize by activity
        rate_map = rate_map_occupancy / np.sum(rate_map_occupancy)

        return rate_map, time_map

    def get_rate_time_map(
        self,
        activity,
        binned_pos,
        max_bin=None,
        smooth=True,
        window_size=2,
    ):
        rate_map_org, time_map = self.get_rate_map(
            activity,
            binned_pos,
            max_bin=max_bin,
        )
        # smoof over 2 bins (typically 2cm)
        rate_map = (
            rate_map_org
            if not smooth
            else smooth_array(rate_map_org, window_size=window_size, axis=1)
        )
        return rate_map, time_map

    @staticmethod
    def get_rate_map_stats(rate_map, position_PDF=None):
        mean_rate = np.mean(rate_map)
        rmap_mean = np.nanmean(rate_map)
        rmap_peak = np.nanmax(rate_map)
        if position_PDF is not None:
            mean_rate_sq = np.ma.sum(np.power(rate_map, 2) * position_PDF)
        # check overall active
        # calculate sparsity of the rate map
        if mean_rate_sq != 0:
            sparsity = mean_rate * mean_rate / mean_rate_sq
            # get selectivity of the rate map
            # high selectivity would be tuning to one position
            max_rate = np.max(rate_map)
            selectivity = max_rate / mean_rate
        rate_map_stats = {}
        rate_map_stats["rmap_mean"] = rmap_mean
        rate_map_stats["rmap_peak"] = rmap_peak
        rate_map_stats["mean_rate_sq"] = mean_rate_sq
        rate_map_stats["sparsity"] = sparsity
        rate_map_stats["selectivity"] = selectivity
        return rate_map_stats


class SpatialInformation(Model):
    """
    A class used to model spatial information in a neuroscience context.

    Attributes
    ----------
    name : str
        The name identifier for the model.
    model_settings : dict
        The settings used for the model.

    Methods
    -------
    create_default_model()
        Creates a default model configuration.
    define_parameter_save_path(model)
        Defines the path where model parameters are saved.
    is_fitted(model)
        Checks if the model has been fitted.
    load_fitted_model(model)
        Loads a fitted model from a saved path.
    get_spatial_information(rate_map, time_map, spatial_information_method="opexebo")
        Computes the spatial information rate and content.
    compute_si_zscores(activity, binned_pos, rate_map, time_map, n_tests=500, spatial_information_method="skaggs", fps=None, max_bin=None)
        Computes spatial information and corresponding z-scores.
    """

    def __init__(self, model_dir, model_id, model_settings=None, **kwargs):
        """
        Initializes the SpatialInformation model.

        Parameters
        ----------
        model_dir : Path
            The directory where the model is stored.
        model_id : str
            The identifier for the model.
        model_settings : dict, optional
            The settings for the model (default is None).
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(model_dir, model_id, model_settings, **kwargs)
        self.name = "si"
        self.model_settings = model_settings
        self.model_settings_start(self.name, model_settings)
        self.model_settings_end(self)

    def create_defaul_model(self):
        """
        Creates a default model configuration.

        Returns
        -------
        model
            The instance of the SpatialInformation model with default settings.
        """
        model = self
        model.si_formula = "skaags"
        return model

    def define_parameter_save_path(self, model):
        """
        Defines the path where model parameters are saved.

        Parameters
        ----------
        model : Model
            The model for which the save path is defined.

        Returns
        -------
        Path
            The path where the model parameters are saved.
        """
        # TODO: implement the usage of this function
        save_path = self.model_dir.joinpath(
            f"place_cell_{model.name}_{self.model_id}.npz"
        )
        return save_path

    def is_fitted(self, model):
        """
        Checks if the model has been fitted to the data.

        Parameters
        ----------
        model : Model
            The model to check.

        Returns
        -------
        bool
            True if the model has been fitted, False otherwise.
        """
        # TODO: implement the usage of this function
        return model.save_path.exists()

    def load_fitted_model(self, model):
        """
        Loads a fitted model parameters from from a saved path.

        Parameters
        ----------
        model : Model
            The model to load.

        Returns
        -------
        Model
            The SpatialInformation loaded model.
        """
        fitted_model_path = model.save_path
        if fitted_model_path.exists():
            fitted_model = np.load(fitted_model_path)
            raise NotImplementedError(
                f"load_fitted_model not implemented for {self.__class__}"
            )
            # TODO: load data from model to prevent recalculation
            if fitted_model.get_params() == model.get_params():
                fitted_full_model = define_cls_attributes(fitted_model, model.__dict__)
                model = fitted_full_model
                global_logger.info(f"Loaded matching model {fitted_model_path}")
                print(f"Loaded matching model {fitted_model_path}")
            else:
                global_logger.error(
                    f"Loaded model parameters do not match to initialized model. Not loading {fitted_model_path}"
                )
        model.fitted = self.is_fitted(model)
        return model

    @staticmethod
    def get_spatial_information(
        rate_map, time_map=None, spatial_information_method="opexebo"
    ):
        """
        ... old documentation ...
        Computes the spatial information rate and content.

        Parameters
        ----------
        rate_map: np.ma.MaskedArray
            Smoothed rate map: n x m array where cell value is the firing rate,
            masked at locations with low occupancy

        time_map: np.ma.MaskedArray
            time map: n x m array where the cell value is the time the animal spent
            in each cell, masked at locations with low occupancy
            Already smoothed

        Returns
        -------
        spatial_information_rate: n x m array where cell value is the float information rate [bits/sec]
        spatial_information_content: n x m array where cell value is the float spatial information content [bits/spike]
        """
        # duration = np.sum(time_map)  # in frames
        ## spacing adds floating point precision to avoid DivideByZero errors
        # position_PDF = time_map / (duration + np.spacing(1))
        ## ................ use position pdf
        # p_spike = rate_map * position_PDF + np.spacing(1)

        p_spike = np.nansum(rate_map, axis=1)
        p_position = np.nansum(rate_map, axis=0)

        # mean_rate = np.sum(p_spike, axis=1) + np.spacing(1)

        # get statistics of the rate map
        # rate_map_stats = get_rate_map_stats(rate_map, position_PDF)

        # if np.sum(mean_rate) == 0:
        #    raise ValueError("Mean firering rate is 0, no brain activity")

        # new axis is added to ensure that the division is done along the right axis
        # log_argument = rate_map / mean_rate[:, np.newaxis]
        p_spike_at_pos = p_spike[:, None] * p_position
        log_argument = rate_map / p_spike_at_pos
        # ensure no number is below 1 before taking the log out of it
        log_argument[log_argument < 1] = 1

        if spatial_information_method == "skaggs":
            inf_rate = np.nansum(rate_map * np.log2(log_argument), axis=1)
            # FIXME: is this correct?
            # inf_rate = np.nansum(
            #    p_spike * np.log2(log_argument), axis=1
            # )
        elif spatial_information_method == "shanon":
            inf_rate = scipy.stats.entropy(pk=log_argument, axis=1)
        elif spatial_information_method == "kullback-leiber":
            inf_rate = scipy.stats.entropy(
                pk=rate_map, qk=p_spike_at_pos[:, np.newaxis], axis=1
            )
        else:
            raise ValueError("Spatial information method not recognized")

        # FIXME: is this correct?
        inf_content = inf_rate * p_spike

        return inf_rate, inf_content

    def compute_si_zscores(
        self,
        activity=None,
        binned_pos=None,
        rate_map=None,
        time_map=None,
        n_tests=500,
        spatial_information_method="skaggs",
        fps=None,
        max_bin=None,
    ):
        """
        Computes spatial information and corresponding z-scores.

        Parameters
        ----------
        activity : np.ndarray, optional
            The activity data (default is None).
        binned_pos : np.ndarray, optional
            The binned position data (default is None).
        rate_map : np.ndarray, optional
            The rate map data (default is None).
        time_map : np.ndarray, optional
            The time map data (default is None).
        n_tests : int, optional
            The number of tests for computing z-scores (default is 500).
        spatial_information_method : str, optional
            The method to compute spatial information (default is "skaggs").
        fps : float, optional
            Frames per second (default is None).
        max_bin : int, optional
            The maximum bin (default is None).

        Returns
        -------
        tuple
            A tuple containing:
            - zscore : np.ndarray
                The z-scores for spatial information.
            - si_rate : np.ndarray
                The spatial information rate.
            - si_content : np.ndarray
                The spatial information content.
        """

        rate_map, time_map = PlaceCellDetectors.get_rate_map(
            activity, binned_pos, max_bin=max_bin
        )

        inf_rate, inf_content = self.get_spatial_information(
            rate_map, time_map, spatial_information_method
        )
        si_rate = inf_rate
        si_content = inf_content

        num_cells, len_track = rate_map.shape

        # calculate the information rate for shuffled data:
        si_shuffle = np.zeros((n_tests, num_cells))
        for test_num in trange(n_tests):
            # shuffle the position data
            binned_pos_shuffled = np.roll(
                binned_pos, np.random.choice(np.arange(binned_pos.shape[0]), 1)
            )
            rate_map_shuffled, _ = PlaceCellDetectors.get_rate_map(
                activity, binned_pos_shuffled, max_bin=max_bin
            )

            inf_rate, _ = self.get_spatial_information(
                rate_map_shuffled, spatial_information_method
            )
            si_shuffle[test_num] = inf_rate

        # calculate z-scores for every cell
        stack = np.vstack([si_rate, si_shuffle])
        zscore = scipy.stats.zscore(stack)[0]

        si_rate *= fps
        si_content *= fps
        return zscore, si_rate, si_content


class Cebras(ModelsWrapper, Model):
    def __init__(self, model_dir, model_id, model_settings=None, **kwargs):
        super().__init__(model_dir, model_settings, **kwargs)
        Model.__init__(self, model_dir, model_id, model_settings, **kwargs)
        # self.time_model = self.time()
        # self.behavior_model = self.behavior()
        # self.hybrid_model = self.hybrid()

    def define_parameter_save_path(self, model):
        save_path = self.model_dir.joinpath(
            f"cebra_{model.name}_dim-{model.output_dimension}_model-{self.model_id}.pt"
        )
        return save_path

    def is_fitted(self, model):
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
        initial_model.data = None
        initial_model.decoding_statistics = None
        return initial_model

    def load_fitted_model(self, model):
        fitted_model_path = model.save_path
        if fitted_model_path.exists():
            fitted_model = CEBRA.load(fitted_model_path)
            if fitted_model.get_params() == model.get_params():
                fitted_full_model = define_cls_attributes(fitted_model, model.__dict__)
                fitted_full_model.name = model.name
                model = fitted_full_model
                global_logger.info(f"Loaded matching model {fitted_model_path}")
                print(f"Loaded matching model {fitted_model_path}")
            else:
                global_logger.error(
                    f"Loaded model parameters do not match to initialized model. Not loading {fitted_model_path}"
                )
        model.fitted = self.is_fitted(model)
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

    def train(
        self,
        model,
        model_type,
        neural_data_train,
        neural_data_test=None,
        behavior_data_train=None,
        regenerate=False,
    ):
        # remove list if neural data is a list and only one element
        if isinstance(neural_data_train, list) and len(neural_data_train) == 1:
            neural_data_test = neural_data_test[0]
            neural_data_train = neural_data_train[0]
            behavior_data_train = behavior_data_train[0]

        if not model.fitted or regenerate:
            # skip if no neural data available
            if isinstance(neural_data_train, np.ndarray) and neural_data_train.shape[0] < 10:
                global_logger.error(
                    f"Not enough frames to use for {model.name}. At least 10 are needed. Skipping"
                )
                print(
                    f"Skipping: Not enough frames to use for {model.name}. At least 10 are needed."
                )
            else:
                # train model
                global_logger.info(f"Training  {model.name} model.")
                print(f"Training  {model.name} model.")
                if model_type == "time":
                    model.fit(neural_data_train)
                else:
                    if behavior_data_train is None:
                        raise ValueError(
                            f"No behavior data types given for {model_type} model."
                        )
                    neural_data_train, behavior_data_train = force_equal_dimensions(
                        neural_data_train, behavior_data_train
                    )
                    model.fit(neural_data_train, behavior_data_train)
                model.fitted = self.is_fitted(model)
                model.save(model.save_path)
        else:
            global_logger.info(
                f"{model.name} model already trained. Skipping."
            )
            print(f"{model.name} model already trained. Skipping.")
        return model

    def create_embedding(
        self, model, session_id=None, to_transform_data=None, to_2d=False, save=False, return_labels=False
    ):
        embedding = None
        labels = None
        if model.fitted:
            if to_transform_data is None:
                to_transform_data = model.data["train"]["neural"]
                label = model.data["train"]["behavior"]
            
            if isinstance(to_transform_data, list) and len(to_transform_data) == 1:
                to_transform_data = to_transform_data[0]
            
            if isinstance(to_transform_data, np.ndarray):
                if session_id is not None:
                    # single session embedding from multi-session model
                    embedding_title = f"{model.name}_session_{session_id}"
                    embedding = model.transform(to_transform_data, session_id=session_id) if to_transform_data.shape[0] > 10 else None
                else:
                    embedding = model.transform(to_transform_data) if to_transform_data.shape[0] > 10 else None

                if to_2d:
                    if embedding.shape[1] > 2:
                        embedding = sphere_to_plane(embedding)
                    elif embedding.shape[1] == 2:
                        print(f"Embedding is already 2D.")
                if save:
                    raise NotImplementedError("Saving embeddings not implemented yet.")
                    import pickle
                    with open('multi_embeddings.pkl', 'wb') as f:
                        pickle.dump(embedding, f)
            else:
                    embedding = {}
                    # multi-session embedding
                    for i, data in enumerate(to_transform_data):
                        embedding_title = f"{model.name}_session_{i}"
                        if return_labels:
                            session_embedding, label = self.create_embedding(model, i, data, to_2d, save, return_labels)
                            if session_embedding is not None:
                                embedding[embedding_title] = session_embedding
                                labels[embedding_title] = label
                        else:
                            session_embedding = self.create_embedding(model, i, data, to_2d, save)
                            if session_embedding is not None:
                                embedding[embedding_title] = session_embedding

        else:
            global_logger.error(f"{model.name} model. Not fitted.")
            global_logger.warning(
                f"Skipping {model.name} model"
            )
            print(f"{model.name} model. Not fitted.")
            print(
                f"Skipping {model.name} model"
            )
        if return_labels:
            return embedding, labels
        return embedding

    def create_embeddings(self, 
                          models: Dict[str, Model],
                          to_transform_data: Union[np.ndarray, List[np.ndarray]]=None, 
                          to_2d=False,
                          save=False,
                          return_labels=False):

        embeddings = {}
        labels = {}
        for model_name, model in models.items():
            embedding_title = f"{model_name}"
            embedding = self.create_embedding(model, to_transform_data=to_transform_data, to_2d=to_2d, save=save)
            if return_labels:
                embedding, label = self.create_embedding(model, to_transform_data=to_transform_data, to_2d=to_2d, save=save, return_labels=return_labels)
            if embedding is not None:
                if isinstance(embedding, dict):
                    embeddings.update(embedding)
                    if return_labels:
                        labels.update(label)
                else:
                    embeddings[embedding_title] = embedding
                    if return_labels:
                        labels[embedding_title] = label
        
        if return_labels:
            return embeddings, labels
        return embeddings

def decode(
    model=None,
    neural_data_train_to_embedd=None,
    neural_data_test_to_embedd=None,
    embedding_train=None,
    embedding_test=None,
    labels_train=None,
    labels_test=None,
    n_neighbors=36,
    metric="cosine",
    detailed_accuracy=False,
):
    if model is None:
        if (
            neural_data_train_to_embedd is None
            or neural_data_test_to_embedd is None
            or embedding_train is None
            or embedding_test is None
            or labels_train is None
            or labels_test is None
        ):
            raise ValueError(
                "Not all data is provided. Please provide the model or the necessary data."
            )

    if (
        neural_data_train_to_embedd is None
        or neural_data_test_to_embedd is None
        or embedding_train is None
        or embedding_test is None
        or labels_train is None
        or labels_test is None
    ):
        print(
            "WARNING: Not all data is provided. Using model data. Make sure correct data is provided."
        )

    if neural_data_train_to_embedd is None:
        neural_data_train_to_embedd = model.data["train"]["neural"]

    if embedding_train is None:
        embedding_train = model.data["train"]["embedding"]
        if embedding_train is None:
            embedding_train = model.transform(neural_data_train_to_embedd)

    if labels_train is None:
        labels_train = model.data["train"]["behavior"]

    if neural_data_test_to_embedd is None:
        neural_data_test_to_embedd = model.data["test"]["neural"]

    if embedding_test is None:
        embedding_test = model.data["test"]["embedding"]
        if embedding_test is None:
            embedding_test = model.transform(neural_data_test_to_embedd)

    if labels_test is None:
        labels_test = model.data["test"]["behavior"]

    # Define decoding function with kNN decoder. For a simple demo, we will use the fixed number of neighbors 36.
    if is_floating(labels_train):
        knn = sklearn.neighbors.KNeighborsRegressor(
            n_neighbors=n_neighbors, metric=metric
        )
    elif is_integer(labels_train):
        knn = sklearn.neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors, metric=metric
        )
    else:
        raise NotImplementedError(
            f"Invalid tlabels_trainpe: targets must be either floats or integers, got labels_train:{labels_train.dtype}."
        )
    labels_train = Dataset.force_2d(labels_train)
    # labels_train = force_1_dim_larger(labels_train)
    labels_test = Dataset.force_2d(labels_test)
    # labels_test = force_1_dim_larger(labels_test)

    fit_labels_train = (
        labels_train.flatten() if labels_train.shape[1] == 1 else labels_train
    )
    knn.fit(embedding_train, fit_labels_train)

    # Predict the targets for data ``X``
    labels_pred = knn.predict(embedding_test)
    labels_pred = Dataset.force_2d(labels_pred)

    if is_floating(labels_test):
        # Use regression metrics
        abs_error = abs(labels_test - labels_pred)
        error_var = np.var(abs_error)
        rmse = np.mean(abs_error)
        r2 = r2_score(labels_test, labels_pred)
        # print(f"Root Mean Squared Error: {rmse}")
        # print(f"Variance of Absolute Error: {error_var}")
        # print(f"R²: {r2}")

        rmse_dict = {"mean": rmse, "variance": error_var}
        r2_dict = {"mean": r2}
        results = {"rmse": rmse_dict, "r2": r2_dict}

    elif is_integer(labels_test):
        accuracies = []
        precisions = []
        recalls = []
        f1s = []

        # handle multi-classification
        for i in range(labels_test.shape[1]):
            if labels_pred.ndim == 1:
                labels_pred = labels_pred.reshape(-1, 1)

            # Classification metrics for each output
            if detailed_accuracy:
                classes = np.unique(labels_test[:, i])
                accuracy = {}
                for class_ in classes:
                    label_test_class_idx = labels_test[:, i] == class_
                    labels_test_class = labels_test[label_test_class_idx, i]
                    labels_pred_class = labels_pred[label_test_class_idx, i]
                    accuracy[class_] = accuracy_score(labels_test_class, labels_pred_class)
            else:
                accuracy = accuracy_score(labels_test[:, i], labels_pred[:, i])
            
            precision = precision_score(
                labels_test[:, i],
                labels_pred[:, i],
                average="macro",
            )
            recall = recall_score(
                labels_test[:, i],
                labels_pred[:, i],
                average="macro",
            )
            f1 = f1_score(
                labels_test[:, i],
                labels_pred[:, i],
                average="macro",
            )

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            # ROC and AUC for each output
            # only for converting mutliclass to binary
            y_test_bin = label_binarize(
                labels_test[:, i],
                classes=np.unique(labels_test[:, i]),
            )
            y_pred_bin = label_binarize(
                labels_pred[:, i],
                classes=np.unique(labels_test[:, i]),
            )
            n_classes = y_test_bin.shape[1]

            class_roc_auc_scores = {}
            for j in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, j], y_pred_bin[:, j])
                auc = roc_auc_score(y_test_bin[:, j], y_pred_bin[:, j])
                class_roc_auc_scores[np.unique(labels_test)[j]] = {
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": auc,
                }

            # print(f"Accuracy for output {i}: {accuracy}")
            # print(f"Classification Report for output {i}:")
            # print(f"Average Accuracy: {np.mean(accuracies)}")
            # print(f"Average Precision: {np.mean(precisions)}")
            # print(f"Average Recall: {np.mean(recalls)}")
            # print(f"Average F1 Score: {np.mean(f1s)}")

            if detailed_accuracy:
                results = accuracy
            else:
                results = {
                    "accuracy": accuracies,
                    "precision": precisions,
                    "recall": recalls,
                    "f1-score": f1s,
                    "roc_auc": class_roc_auc_scores,
                }
    else:
        raise NotImplementedError(
            f"Invalid label_test type: targets must be either floats or integers, got label_test:{labels_test.dtype}."
        )

    return results
