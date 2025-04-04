from __future__ import annotations

# type hints
from typing import List, Union, Dict, Tuple, Optional, Literal

# show progress bar
from tqdm import tqdm, trange

# calculations
import numpy as np
import sklearn
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import KFold
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
from copy import deepcopy

# parralelize
from numba import jit, njit, prange
from numba import cuda  # @jit(target='cuda')

# manifolds
from cebra import CEBRA
import cebra.integrations.sklearn.utils as sklearn_utils
from structure_index import compute_structure_index

# own
from Datasets import Datasets, Dataset
from Helper import *
from Visualizer import Vizualizer


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
        model_type: str,
        name_comment: str = None,
        behavior_data_types: List[str] = None,
        shuffled: bool = False,
        transformation: str = None,
        movement_state: str = "all",
        split_ratio: float = 1,
        model_settings: dict = None,
    ):
        behavior_data_types = [] if behavior_data_types is None else behavior_data_types
        if name_comment:
            model_name = name_comment
            for behavior_data_type in behavior_data_types:
                if behavior_data_type not in model_name:
                    model_name = f"{model_name}_{behavior_data_type}"

            if model_type not in model_name:
                model_name = f"{model_type}_{model_name}"
            if shuffled and "shuffled" not in model_name:
                model_name = f"{model_name}_shuffled"
        else:
            model_name = model_type
            for behavior_data_type in behavior_data_types:
                model_name = f"{model_name}_{behavior_data_type}"
            model_name = f"{model_name}_shuffled" if shuffled else model_name

        if movement_state not in ["all", "moving", "stationary"]:
            global_logger.error(
                f"Movement state {movement_state} not supported. Choose 'all', 'moving', or 'stationary'."
            )
            raise ValueError(
                f"Movement state {movement_state} not supported. Choose 'all', 'moving', or 'stationary'."
            )
        elif movement_state != "all":
            model_name = f"{model_name}_{movement_state}"

        if split_ratio != 1:
            model_name = f"{model_name}_{split_ratio}"

        if transformation:
            if transformation not in ["relative", "binned"]:
                global_logger.error(
                    f"Transformation {transformation} not supported. Choose 'relative' or 'binned'."
                )
                raise ValueError(
                    f"Transformation {transformation} not supported. Choose 'relative' or 'binned'."
                )
            else:
                if transformation not in model_name:
                    model_name = f"{model_name}_{transformation}"

        if model_settings is not None:
            max_iterations = model_settings["max_iterations"]
            model_name = f"{model_name}_iter-{max_iterations}"
        return model_name

    def define_properties(self, **kwargs):
        """
        Define properties for the model.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def get_model(
        self,
        name_comment: str = None,
        model_type: str = "time",
        behavior_data_types: List[str] = None,
        shuffle: bool = False,
        movement_state: str = "all",
        split_ratio: float = 1,
        transformation: str = None,
        pipeline="cebra",
        model_settings=None,
        metadata: Dict[str, Dict[str, Any]] = None,
    ):
        models_class = self.get_model_class(pipeline)

        model_name = self.define_model_name(
            model_type=model_type,
            name_comment=name_comment,
            behavior_data_types=behavior_data_types,
            shuffled=shuffle,
            movement_state=movement_state,
            split_ratio=split_ratio,
            transformation=transformation,
            model_settings=model_settings[pipeline],
        )

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
                        global_logger.warning(
                            f"Model {model_name} with different model settings found. Creating new model."
                        )
                        break

        if not model_available:
            model_creation_function = getattr(models_class, model_type)
            model = model_creation_function(
                name=model_name,
                behavior_data_types=behavior_data_types,
                model_settings=model_settings,
            )

        model.define_metadata(
            data_transformation=transformation,
            data_split_ratio=split_ratio,
            data_shuffled=shuffle,
            data_movement_state=movement_state,
            behavior_data_types=behavior_data_types,
            metadata=metadata,
        )

        return model

    def get_pipeline_models(
        self,
        manifolds_pipeline="cebra",
        model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
        model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
    ) -> Tuple[Models, Dict[str, Model]]:
        """Get models from a specific model class.

        Parameters:
        -----------
            manifolds_pipeline : str
                The pipeline to use.
            model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
                Filter for model names to include. If None, all models will be included. 3 levels of filtering are possible.
                1. Include all models containing a specific string: "string"
                2. Include all models containing a specific combination of strings: ["string1", "string2"]
                3. Include all models containing one of the string combinations: [["string1", "string2"], ["string3", "string4"]]
            model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
                Same as model_naming_filter_include but for excluding models.

        Returns:
        --------
            model_class : class
                The class of the model.
            models : dict
                A dictionary containing the models.
        """
        models_class = self.get_model_class(manifolds_pipeline)
        models = models_class.get_models(
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
        )
        return models_class, models

    def get_models_splitted_original_shuffled(
        self,
        models=None,
        manifolds_pipeline="cebra",
        model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
        model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
    ):
        models_original = []
        models_shuffled = []
        if not models:
            models_class, models = self.get_pipeline_models(
                manifolds_pipeline,
                model_naming_filter_include,
                model_naming_filter_exclude,
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
        model_naming_filter_include: List[List[str]] = None,
        model_naming_filter_exclude: List[List[str]] = None,
        manifolds_pipeline="cebra",
        save=False,
        return_labels=False,
    ):
        """
        model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
        model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
        """
        if not type(to_transform_data) == np.ndarray:
            global_logger.warning(
                f"No data to transform given. Using model training data."
            )
            print(f"No data to transform given. Using model training data.")

        model_class, models = self.get_pipeline_models(
            manifolds_pipeline=manifolds_pipeline,
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
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

    def define_decoding_statistics(
        self,
        model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
        model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
        manifolds_pipeline: str = "cebra",
    ):
        models_class, models = self.get_pipeline_models(
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
            manifolds_pipeline=manifolds_pipeline,
        )
        to_delete_models_key_list = []
        for keys_list, model in traverse_dicts(models):
            decoding_statistics = model.define_decoding_statistics()
            if decoding_statistics is None:
                to_delete_models_key_list.append(keys_list)

            # model = dict_value_keylist(models, keys_list)

        for keys_list in to_delete_models_key_list:
            delete_nested_key(models, keys_list)
        return models

    def is_model_fitted(self, model, pipeline="cebra"):
        return self.get_model_class(pipeline).is_fitted(model)

    def train_model(
        self,
        neural_data,
        model=None,
        behavior_data=None,
        behavior_data_types=None,
        pipeline="cebra",
        model_type="time",
        name_comment=None,
        transformation=None,
        idx_to_keep=None,
        shuffle=False,
        movement_state="moving",
        split_ratio=1,
        model_settings=None,
        create_embeddings=False,
        regenerate=False,
        metadata: Dict[str, Dict[str, Any]] = None,
    ):
        if not is_dict_of_dicts(model_settings):
            model_settings = {pipeline: model_settings}

        model = model or self.get_model(
            pipeline=pipeline,
            model_type=model_type,
            name_comment=name_comment,
            behavior_data_types=behavior_data_types,
            shuffle=shuffle,
            movement_state=movement_state,
            split_ratio=split_ratio,
            transformation=transformation,
            model_settings=model_settings,
            metadata=metadata,
        )

        neural_data_train, neural_data_test = Dataset.manipulate_data(
            neural_data,
            idx_to_keep=idx_to_keep,
            shuffle=shuffle,
            split_ratio=split_ratio,
        )

        if behavior_data is not None:
            behavior_data_train, behavior_data_test = Dataset.manipulate_data(
                behavior_data,
                idx_to_keep=idx_to_keep,
                shuffle=shuffle,
                split_ratio=split_ratio,
            )
        else:
            behavior_data_test = None

        model = model.train(
            neural_data=neural_data_train,
            behavior_data=behavior_data_train,
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
            train_embedding = model.create_embedding(
                to_transform_data=neural_data_train
            )
            test_embedding = model.create_embedding(to_transform_data=neural_data_test)

        model.set_data(data=train_embedding, train_or_test="train", type="embedding")
        model.set_data(data=test_embedding, train_or_test="test", type="embedding")

        return model

    def get_model_class(self, pipeline="cebra"):
        if pipeline == "cebra":
            models_class = self.cebras
        else:
            raise ValueError(f"Pipeline {pipeline} not supported. Choose 'cebra'.")
        return models_class

    @staticmethod
    def cross_decoding(
        ref_models: Union[CebraOwn, List[CebraOwn], Dict[str, CebraOwn]],
        models: Union[CebraOwn, List[CebraOwn], Dict[str, CebraOwn]] = None,
        labels_describe_space=False,
        multiply_by: Optional[Union[int, float, Dict[str, Union[int, float]]]] = 1,
        title="Decoding perfomance between Models",
        additional_title: str = "",
        xticks: Optional[List[str]] = None,
        adapt: bool = True,
        n_neighbors: int = None,
        plot: bool = True,
        fig_size: tuple = (5, 7),
    ) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        Decodes the data using the one-to-one decoding method.

        It is important to mention, that the model name "random" is reserved for the random model.
        The behavior variables will be shuffled for decoding.

        Parameters
        ----------
        ref_models : Union[Model, List[Model]]
            The reference models to use for decoding.
        models : Union[Model, List[Model]], optional
            The models with data to adapt to the reference models (default is None).
            If not provided, the reference models will be used.
        labels_describe_space : bool, optional
            If labels are describing space (default is False).
            If space is described the Frobenius norm is used to calculate the distance between decoded positional values.
        multiply_by : Union[int, float, Dict[str, Union[int, float]]], optional
            The factor to multiply the decoding results by to normalize the results. (default is 1).
            The keys of the dictionary must match the keys of the models.
        xticks : Optional[List[str]], optional
            The labels for the x-axis (default is None). If not provided, the model names will be used.
            xticks must match the number of comparisons.
        adapt : bool, optional
            Whether to adapt the model (default is True). Currently only CEBRA models can be adapted.
        n_neighbors : int, optional
            The number of neighbors to use for the KNN algorithm (default is None).
            if None, the number of neighbors will be determined by k-fold cross-validation in the decoding function.
        plot : bool, optional
            Whether to plot the decoding results (default is True).

        Returns
        -------
        model_decoding_statistics : Dict[str, Dict[str, Union[float, np.ndarray]]]
            A dictionary containing the decoding statistics.
        """

        # convert reference models to dict
        if not isinstance(ref_models, dict):
            ref_models = make_list_ifnot(ref_models)
            # ensure that model_name is unique
            unique_ref_models = {}
            for model in ref_models:
                model_name = (
                    create_unique_dict_key(unique_ref_models, model.name)
                    if model.name in unique_ref_models.keys()
                    else model.name
                )
                unique_ref_models[model_name] = model
            ref_models = unique_ref_models

        if models is None:
            models = ref_models

        # convert models to dict
        if not isinstance(models, dict):
            models = make_list_ifnot(models)
            # ensure that model_name is unique
            unique_models = {}
            for model in models:
                model_name = (
                    create_unique_dict_key(unique_models, model.name)
                    if model.name in unique_models.keys()
                    else model.name
                )
                unique_models[model_name] = model
            models = unique_models

        cross_model_decoding_statistics = {}
        for ref_model_name, ref_model in iter_dict_with_progress(ref_models):
            global_logger.info(f"------------ Decoding statistics for {ref_model_name}")
            multiply_by_values = (
                multiply_by[ref_model_name]
                if isinstance(multiply_by, dict)
                else multiply_by
            )
            ref_model_dec_stats = ref_model.define_decoding_statistics(
                regenerate=True,
                n_neighbors=n_neighbors,
                labels_describe_space=labels_describe_space,
                multiply_by=multiply_by_values,
            )
            ref_model_decoding_statistics = {ref_model_name: ref_model_dec_stats}

            for model_name, model2 in models.items():
                global_logger.info(f"Decoding to {model_name}")
                neural_data_train_to_embedd = model2.get_data(
                    train_or_test="train", type="neural"
                )
                neural_data_test_to_embedd = model2.get_data(
                    train_or_test="test", type="neural"
                )
                labels_train = model2.get_data(train_or_test="train", type="behavior")
                labels_test = model2.get_data(train_or_test="test", type="behavior")
                adapted_model = (
                    ref_model.fit(neural_data_train_to_embedd, labels_train, adapt=True)
                    if adapt
                    else ref_model
                )

                multiply_by_values = (
                    multiply_by[model_name]
                    if isinstance(multiply_by, dict)
                    else multiply_by
                )
                ref_model_decoding_statistics["to " + model_name] = (
                    adapted_model.define_decoding_statistics(
                        neural_data_train_to_embedd=neural_data_train_to_embedd,
                        neural_data_test_to_embedd=neural_data_test_to_embedd,
                        labels_train=labels_train,
                        labels_test=labels_test,
                        n_neighbors=n_neighbors,
                        labels_describe_space=labels_describe_space,
                        multiply_by=multiply_by_values,
                    )
                )

            cross_model_decoding_statistics[ref_model_name] = (
                ref_model_decoding_statistics
            )

        if plot:
            if xticks is not None and len(xticks) != len(ref_models) * len(
                models
            ) + len(ref_models):
                global_logger.warning(
                    f"Plotting one to tone decoding xticks must be a list of length {len(ref_models) * len(models)}, got {len(xticks)}. Using model names instead."
                )
                xticks = None

            plot_dict = {}
            if len(cross_model_decoding_statistics) == 1:
                # format decoding ouput for plotting
                plot_cross_model_decoding_statistics = list(
                    cross_model_decoding_statistics.values()
                )[0]

                for key, value in plot_cross_model_decoding_statistics.items():
                    plot_dict[key] = {
                        "mean": value["rmse"]["mean"],
                        "variance": value["rmse"]["variance"],
                    }
                Vizualizer.barplot_from_dict(
                    plot_dict,
                    xticks=xticks,
                    title=title,
                    ylabel="RMSE (cm)",
                    xlabel="Models",
                    additional_title=additional_title,
                    figsize=fig_size,
                )

            elif len(cross_model_decoding_statistics) > 1:
                # format decoding ouput for plotting
                plot_cross_model_decoding_statistics = {}
                for i, (ref_model_name, ref_to_decodings) in enumerate(
                    cross_model_decoding_statistics.items()
                ):
                    plot_ref_model_decoding_statistics = {}
                    for j, (ref_to_name, ref_to_decoding) in enumerate(
                        ref_to_decodings.items()
                    ):
                        plot_ref_model_decoding_statistics[f"{ref_to_name}"] = {
                            "mean": ref_to_decoding["rmse"]["mean"],
                            "variance": ref_to_decoding["rmse"]["variance"],
                        }
                    plot_cross_model_decoding_statistics[ref_model_name] = (
                        plot_ref_model_decoding_statistics
                    )

                Vizualizer.barplot_from_dict_of_dicts(
                    plot_cross_model_decoding_statistics,
                    title=title,
                    xticks=xticks,
                    figsize=(20, 6),
                    additional_title=additional_title,
                )

        return cross_model_decoding_statistics


class ModelsWrapper:
    """
    Meta Class for Models wrapper
    """

    def __init__(self, model_dir, model_settings=None, **kwargs):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model_settings = model_settings or kwargs
        self.models = {}

    def get_models(
        self,
        model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
        model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
    ) -> Dict[str, Model]:
        filtered_models = filter_dict_by_properties(
            dictionary=self.models,
            include_properties=model_naming_filter_include,
            exclude_properties=model_naming_filter_exclude,
        )
        return filtered_models


class Model:
    def __init__(self, model_dir, model_id, model_settings=None, **kwargs):
        self.model_dir = model_dir
        self.model_id = model_id
        self.model_settings = model_settings or kwargs
        self.model_dir.mkdir(exist_ok=True)
        self.fitted = False
        self.decoding_statistics = None
        self.data = None
        self.data_transformation = None
        self.data_split_ratio = None
        self.data_shuffled = None
        self.data_movement_state = None
        self.behavior_data_types = None
        self.name = None
        self.save_path = None

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
        if self.rate_map is None or self.time_map is None:
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
            # TODO: load data from model to prevent recalculation
            raise NotImplementedError(
                f"load_fitted_model not implemented for {self.__class__}"
            )
        model.fitted = self.is_fitted(model)
        return model

    @staticmethod
    def get_spatial_information(
        rate_map, time_map=None, spatial_information_method="skaggs"
    ):
        """
        #FIXME: This is old documentation, change it to the new one
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
        ## use position pdf
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
    # TODO: restructure this and mode Model to CebraOwn class as well as other functions
    def __init__(self, model_dir, model_id, model_settings=None, **kwargs):
        super().__init__(model_dir, model_settings, **kwargs)
        self.model_id = model_id
        # self.time_model = self.time()
        # self.behavior_model = self.behavior()
        # self.hybrid_model = self.hybrid()

    def init_model(self, model_settings_dict):
        if len(model_settings_dict) == 0:
            model_settings_dict = self.model_settings
        initial_model = CebraOwn(**model_settings_dict)
        # initial_model = define_cls_attributes(
        #    default_model, model_settings_dict, override=True
        # )
        initial_model.fitted = False
        initial_model.data = None
        initial_model.decoding_statistics = None
        return initial_model

    def model_settings_start(self, name, model_settings_dict):
        model = self.init_model(model_settings_dict)
        model.name = name
        return model

    def model_settings_end(self, model):
        save_path = model.define_parameter_save_path(
            model_dir=self.model_dir, model_id=self.model_id
        )
        model = model.load_fitted_model()
        model.model_id = self.model_id
        self.models[model.name] = model
        return model

    def time(self, name="time", model_settings=None, **kwargs):
        model_settings = model_settings or kwargs
        model = self.model_settings_start(name, model_settings)
        model.temperature = (
            1.12 if kwargs.get("temperature") is None else model.temperature
        )
        model.type = name
        model.conditional = "time" if kwargs.get("time") is None else model.conditional
        model = self.model_settings_end(model)
        return model

    def behavior(self, name="behavior", model_settings=None, **kwargs):
        model_settings = model_settings or kwargs
        model = self.model_settings_start(name, model_settings)
        model.type = name
        model = self.model_settings_end(model)
        return model

    def hybrid(self, name="hybrid", model_settings=None, **kwargs):
        model_settings = model_settings or kwargs
        model = self.model_settings_start(name, model_settings)
        model.hybrid = True if kwargs.get("hybrid") is None else model.hybrid
        model.type = name
        model = self.model_settings_end(model)
        return model

    def create_embeddings(
        self,
        models: Dict[str, Model],
        to_transform_data: Union[np.ndarray, List[np.ndarray]] = None,
        to_2d=False,
        save=False,
        return_labels=False,
    ):
        embeddings = {}
        labels = {}
        for model_name, model in models.items():
            embedding_title = f"{model_name}"
            if return_labels:
                embedding, label = model.create_embedding(
                    model,
                    to_transform_data=to_transform_data,
                    to_2d=to_2d,
                    save=save,
                    return_labels=return_labels,
                )
            else:
                embedding = model.create_embedding(
                    to_transform_data=to_transform_data, to_2d=to_2d, save=save
                )
            if embedding is not None:
                embeddings[embedding_title] = embedding
                if return_labels:
                    labels[embedding_title] = label

        if return_labels:
            return embeddings, labels
        return embeddings


class CebraOwn(CEBRA):
    def __init__(
        self,
        model_architecture: str = "offset10-model",
        device: str = "cuda_if_available",
        criterion: str = "infonce",
        distance: str = "cosine",
        conditional: str = "time_delta",
        temperature: float = 1.0,
        temperature_mode: Literal["constant", "auto"] = "constant",
        min_temperature: Optional[float] = 0.1,
        time_offsets: int = 10,
        delta: float = None,
        max_iterations: int = 10000,
        max_adapt_iterations: int = 500,
        batch_size: int = 512,
        learning_rate: float = 3e-4,
        optimizer: str = "adam",
        output_dimension: int = 3,
        verbose: bool = True,
        num_hidden_units: int = 32,
        pad_before_transform: bool = True,
        hybrid: bool = False,
        optimizer_kwargs: Tuple[Tuple[str, object], ...] = (
            ("betas", (0.9, 0.999)),
            ("eps", 1e-08),
            ("weight_decay", 0),
            ("amsgrad", False),
        ),
    ):
        # TODO: change Cebras class and CebraOwn to make it work when CebraOwn also inherits from Model
        # Model.__init__(self, model_dir, model_id, model_settings=None, **kwargs)
        CEBRA.__init__(
            self,
            model_architecture=model_architecture,
            device=device,
            criterion=criterion,
            distance=distance,
            conditional=conditional,
            temperature=temperature,
            temperature_mode=temperature_mode,
            min_temperature=min_temperature,
            time_offsets=time_offsets,
            delta=delta,
            max_iterations=max_iterations,
            max_adapt_iterations=max_adapt_iterations,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            output_dimension=output_dimension,
            verbose=verbose,
            num_hidden_units=num_hidden_units,
            pad_before_transform=pad_before_transform,
            hybrid=hybrid,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.model_type = None
        self.decoding_statistics = None
        self.data = None
        self.data_transformation = None
        self.data_split_ratio = None
        self.data_shuffled = None
        self.data_movement_state = None
        self.behavior_data_types = None
        self.name = None
        self.save_path = None

    def define_metadata(self, **kwargs):
        """
        Define metadata for the model.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def get_metadata(self, type: str = None, key: str = None):
        """
        Get metadata for the model.

        Parameters
        ----------
        type : str, optional
            The type of metadata to get (default is None).
        key : str, optional
            The key of the metadata to get (default is None).

        Returns
        -------
        metadata_info : Any
            The metadata information
        """
        if type not in self.metadata:
            global_logger.error(
                f"Type {type} not found in metadata for {self.__class__}"
            )
            raise ValueError(f"Type {type} not found in metadata for {self.__class__}")

        if type is not None and key is not None:
            if key not in self.metadata[type]:
                global_logger.error(
                    f"Key {key} not found in metadata for {self.__class__}. Returning None."
                )
                metadata_info = None
            else:
                metadata_info = self.metadata[type][key]
        elif type is not None:
            metadata_info = self.metadata[type]
        elif key is not None:
            global_logger.error(
                f"Type must be provided to get metadata from {self.__class__}"
            )
            raise ValueError(
                f"Type must be provided to get metadata from {self.__class__}"
            )
        return metadata_info

    def define_parameter_save_path(
        self, model_dir: Union[str, Path] = None, model_id: str = None
    ):
        if model_dir is not None:
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = self.save_path.parent
        if model_id is None:
            model_id = self.model_id
        self.save_path = self.model_dir.joinpath(
            f"cebra_{self.name}_dim-{self.output_dimension}_model-{model_id}.pt"
        )
        return self.save_path

    def set_name(self, name):
        self.name = name
        self.define_parameter_save_path()

    def define_decoding_statistics(
        self,
        neural_data_train_to_embedd: np.ndarray = None,
        neural_data_test_to_embedd: np.ndarray = None,
        labels_train: np.ndarray = None,
        labels_test: np.ndarray = None,
        n_neighbors: int = None,
        regenerate: bool = False,
        labels_describe_space: bool = False,
        multiply_by: float = 1.0,
    ):
        """
        Decodes the data using the model.

        Parameters
        ----------
        neural_data_train_to_embedd : np.ndarray, optional
            The neural data to use for training the model (default is None). If not provided, the training data from the model will be used.
        neural_data_test_to_embedd : np.ndarray, optional
            The neural data to use for testing the model (default is None). If not provided, the testing data from the model will be used.
        labels_train : np.ndarray, optional
            The behavior data to use for training the model (default is None). If not provided, the training data from the model will be used.
        labels_test : np.ndarray, optional
            The behavior data to use for testing the model (default is None). If not provided, the testing data from the model will be used.
        n_neighbors : int, optional
            The number of neighbors to use for the KNN algorithm (default is None).
            if None, the number of neighbors will be determined by k-fold cross-validation in the decoding function.
        regenerate : bool, optional
            Whether to regenerate the decoding statistics (default is False).
        labels_describe_space : bool, optional
            If labels are describing space (default is False).
            If space is described the Frobenius norm is used to calculate the distance between decoded positional values.
        multiply_by : float, optional
            A multiplier to apply to the decoding statistics (default is 1.0). This is used to scale the decoding statistics.

        Returns
        -------
        decoding_statistics : Dict[Dict[str, Any]]
            A dictionary containing the decoding statistics.

        """
        if neural_data_train_to_embedd is not None:
            neural_data_train = self.create_embedding(
                to_transform_data=neural_data_train_to_embedd
            )
        else:
            if self.get_data() is None:
                self.set_data(
                    data=self.create_embedding(
                        to_transform_data=self.get_data(type="neural")
                    )
                )
            neural_data_train = self.get_data()

        if neural_data_test_to_embedd is not None:
            neural_data_test = self.create_embedding(
                to_transform_data=neural_data_test_to_embedd
            )
        else:
            if self.get_data(train_or_test="test") is None:
                self.set_data(
                    data=self.create_embedding(
                        to_transform_data=self.get_data(
                            train_or_test="test", type="neural"
                        )
                    ),
                    train_or_test="test",
                )
            neural_data_test = self.get_data(train_or_test="test")

        labels_train = (
            self.get_data(type="behavior") if labels_train is None else labels_train
        )
        labels_test = (
            self.get_data(train_or_test="test", type="behavior")
            if labels_test is None
            else labels_test
        )

        if neural_data_train is None or neural_data_train.shape[0] < 10:
            self.decoding_statistics = None
        else:
            if (
                self.decoding_statistics is None
                or regenerate
                or neural_data_test_to_embedd is not None
                or neural_data_train_to_embedd is not None
            ):
                self.decoding_statistics = decode(
                    embedding_train=neural_data_train,
                    embedding_test=neural_data_test,
                    labels_train=labels_train,
                    labels_test=labels_test,
                    n_neighbors=n_neighbors,
                    labels_describe_space=labels_describe_space,
                    multiply_by=multiply_by,
                )
        return self.decoding_statistics

    def load_fitted_model(self):
        fitted_model_path = self.save_path
        if fitted_model_path.exists():
            fitted_model = self.load(fitted_model_path)
            if equal_dicts(self.get_params(), fitted_model.get_params()):
                # load_cebra_with_sklearn_backend
                copy_attributes_to_object(
                    propertie_name_list=fitted_model.__dict__.keys(),
                    set_object=self,
                    get_object=fitted_model,
                )
                global_logger.info(f"Loaded matching model {fitted_model_path}")
                print(f"Loaded matching model {fitted_model_path}")
            else:
                global_logger.error(
                    f"Loaded model parameters do not match to initialized model. Not loading {fitted_model_path}"
                )
            self.fitted = self.is_fitted()
        else:
            self.fitted = False
        return self

    def is_fitted(self):
        return sklearn_utils.check_fitted(self)

    def remove_fitting(self):
        if "n_features" in self.__dict__:
            del self.n_features_

    def train(
        self,
        neural_data,
        behavior_data=None,
        regenerate=False,
    ):
        # remove list if neural data is a list and only one element
        if isinstance(neural_data, list) and len(neural_data) == 1:
            neural_data = neural_data[0]
            behavior_data = behavior_data[0]

        if not self.fitted or regenerate:
            # skip if no neural data available
            if isinstance(neural_data, np.ndarray) and neural_data.shape[0] < 10:
                global_logger.error(
                    f"Not enough frames to use for {self.name}. At least 10 are needed. Skipping"
                )
                print(
                    f"Skipping: Not enough frames to use for {self.name}. At least 10 are needed."
                )
            else:
                # train model
                global_logger.info(f"Training  {self.name} model.")
                print(f"Training  {self.name} model.")
                if self.type == "time":
                    self.fit(neural_data)
                else:
                    if behavior_data is None:
                        global_logger.error(
                            f"No behavior data given for {self.type} model."
                        )
                        raise ValueError(
                            f"No behavior data types given for {self.type} model."
                        )
                    neural_data, behavior_data = force_equal_dimensions(
                        neural_data, behavior_data
                    )
                    self.fit(neural_data, behavior_data)
                self.fitted = self.is_fitted()
                self.save(self.save_path)
        else:
            global_logger.info(f"{self.name} model already trained. Skipping.")
            print(f"{self.name} model already trained. Skipping.")
        return self

    def create_embedding(
        self,
        session_id=None,
        to_transform_data=None,
        transform_data_labels=None,
        train_or_test="train",
        to_2d=False,
        save=False,
        return_labels=False,
        plot=False,
        markersize=2,
        additional_title="",
        as_pdf=False,
        save_dir=None,
    ):
        embedding = None
        labels = None
        if self.fitted:
            if to_transform_data is None:
                to_transform_data = self.get_data(
                    train_or_test=train_or_test, type="neural"
                )
                embedding = self.get_data(train_or_test=train_or_test, type="embedding")
                label = self.get_data(train_or_test=train_or_test, type="behavior")
            else:
                label = transform_data_labels
                if label is None:
                    if plot:
                        global_logger.warning(
                            f"WARNING: Proper Plotting of transformed data only possible with provided labels."
                        )
                    label = np.zeros(to_transform_data.shape[0])

            if isinstance(to_transform_data, list) and len(to_transform_data) == 1:
                to_transform_data = to_transform_data[0]

            if isinstance(to_transform_data, np.ndarray):

                if embedding is None:
                    if session_id is not None:
                        # single session embedding from multi-session model
                        embedding = (
                            self.transform(to_transform_data, session_id=session_id)
                            if to_transform_data.shape[0] > 10
                            else None
                        )
                    else:
                        embedding = (
                            self.transform(to_transform_data)
                            if to_transform_data.shape[0] > 10
                            else None
                        )

                if to_2d:
                    if embedding.shape[1] > 2:
                        embedding = sphere_to_plane(embedding)
                    elif embedding.shape[1] == 2:
                        print(f"Embedding is already 2D.")
                if save:
                    raise NotImplementedError("Saving embeddings not implemented yet.")
                    import pickle

                    with open("multi_embeddings.pkl", "wb") as f:
                        pickle.dump(embedding, f)

                if plot:
                    plot_labels = {
                        "name": self.name,
                        "labels": label,
                    }
                    Vizualizer.plot_embedding(
                        embedding=embedding,
                        embedding_labels=plot_labels,
                        markersize=markersize,
                        additional_title=additional_title,
                        as_pdf=as_pdf,
                        save_dir=save_dir,
                        show=True,
                    )
            else:
                embedding = {}
                # multi-session embedding
                for i, data in enumerate(to_transform_data):
                    embedding_title = f"{self.name}_task_{i}"
                    if return_labels:
                        session_embedding, label = self.create_embedding(
                            self, i, data, to_2d, save, return_labels
                        )
                        if session_embedding is not None:
                            embedding[embedding_title] = session_embedding
                            labels[embedding_title] = label
                    else:
                        session_embedding = self.create_embedding(
                            self, i, data, to_2d, save
                        )
                        if session_embedding is not None:
                            embedding[embedding_title] = session_embedding

        else:
            global_logger.error(f"{self.name} model. Not fitted.")
            global_logger.warning(f"Skipping {self.name} model")
            print(f"{self.name} model. Not fitted.")
            print(f"Skipping {self.name} model")

        if return_labels:
            return embedding, labels
        return embedding

    def get_loss(self):
        return self.state_dict_["loss"]

    def get_data(
        self, train_or_test: str = "train", type: str = "embedding"
    ) -> np.ndarray:
        """
        Get the data for the model.

        Parameters
        ----------
        train_or_test : str
            The type of data to get (either "train" or "test").
        type : str
            The type of data to get (either "neural", "embedding" or "behavior").

        Returns
        -------
        np.ndarray
            The data for the model.
        """
        return self.data[train_or_test][type]

    def set_data(
        self, data: np.ndarray, train_or_test: str = "train", type: str = "embedding"
    ):
        """
        Set the data for the model.

        Parameters
        ----------
        data : np.ndarray
            The data to set.
        train_or_test : str
            The type of data to set (either "train" or "test").
        type : str
            The type of data to set (either "neural", "embedding" or "behavior").
        """
        self.data[train_or_test][type] = data

    def get_decoding_statistics(
        self, mean_or_variance: str = "mean", type: str = "rmse"
    ):
        """
        Get the decoding statistics for the model.

        Parameters
        ----------
        mean_or_variance : str
            The type of statistics to get (either "mean" or "variance").
        type : str
            The type of statistics to get (either "rmse" or "r2").

        Returns
        -------
        Union[float, np.ndarray]
            The decoding statistics for the model.
        """
        return self.decoding_statistics[type][mean_or_variance]

    def structure_index(
        self,
        params: Dict[str, Union[int, bool, List[int]]],
        use_raw: bool = False,
        labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        to_transform_data: Optional[np.ndarray] = None,
        regenerate: bool = False,
        to_2d: bool = False,
        plot: bool = True,
        as_pdf: bool = False,
        folder_name: Optional[str] = "structure_index",
        plot_save_dir: Optional[Path] = None,
    ) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:  # TODO: finish return types::
        """
        Computes the structural index for the model.
        """
        additional_title = f"{self.name}"
        ofname = f"structrual_indices_{self.name}"
        return_labels = True if labels is None else True
        if use_raw and not to_transform_data is None:
            raise ValueError(
                "Cannot use raw data and provide data to transform. Raw data is based on the model training data."
            )
        elif use_raw:
            data = self.get_data(train_or_test="train", type="neural")
            labels = self.get_data(train_or_test="train", type="behavior")
            additional_title += " - RAW"
            ofname += "_raw"
        else:
            if to_transform_data is None:
                data = self.create_embedding(to_2d=to_2d, return_labels=return_labels)
                additional_title += " - Embedding"
            else:
                data = self.create_embedding(
                    to_transform_data=to_transform_data,
                    to_2d=to_2d,
                    return_labels=return_labels,
                )
                additional_title += " - Custom Data Embedding"
                ofname += "_custom"
            if return_labels:
                data = data[0]
                labels = data[1]

        # check if parameter sweep is performed
        if isinstance(params["n_neighbors"], List) or isinstance(
            params["n_neighbors"], np.ndarray
        ):
            ofname += "_sweep"

        struc_ind = npy(
            ofname,
            task="load",
            backup_root_folder=self.save_path.parent,
            backup_folder_name=folder_name,
        )
        # TODO: remove after enough testing
        if list(struc_ind.keys())[0] == self.name:
            struc_ind = struc_ind[self.name]
            npy(
                ofname,
                task="save",
                data=struc_ind,
                backup_root_folder=self.save_path.parent,
                backup_folder_name=folder_name,
            )

        if struc_ind is None or regenerate:
            struc_ind = structure_index(
                data=data,
                labels=labels,
                params=params,
                additional_title=additional_title,
                plot=plot,
                as_pdf=as_pdf,
            )
            npy(
                ofname,
                task="save",
                data=struc_ind,
                backup_root_folder=self.save_path.parent,
                backup_folder_name=folder_name,
            )
        else:
            if plot:
                Vizualizer.plot_structure_index(
                    embedding=data,
                    feature=labels,
                    overlapMat=struc_ind["overlap_mat"],
                    SI=struc_ind["SI"],
                    binLabel=struc_ind["bin_label"],
                    additional_title=additional_title,
                    as_pdf=as_pdf,
                    save_dir=plot_save_dir
                    or self.save_path.parent.joinpath(folder_name),
                )

        return struc_ind

    def make_random(self, regenerate: bool = False, create_embedding: bool = True):
        random_self = copy.deepcopy(self)
        random_self.set_name(f"{self.name}_random")
        # randomize neural data
        for train_or_test in ["train", "test"]:
            for type in ["neural", "behavior", "embedding"]:
                data = random_self.get_data(train_or_test=train_or_test, type=type)
                random_self.set_data(
                    data=Dataset.shuffle(data), train_or_test=train_or_test, type=type
                )

        random_self.remove_fitting()
        random_self.load_fitted_model()

        random_self.train(
            neural_data=random_self.get_data(train_or_test="train"),
            behavior_data=random_self.get_data(train_or_test="train", type="behavior"),
            regenerate=regenerate,
        )

        if create_embedding:
            train_embedding = random_self.create_embedding(train_or_test="train")
            test_embedding = random_self.create_embedding(train_or_test="test")
            random_self.set_data(
                data=train_embedding, train_or_test="train", type="embedding"
            )
            random_self.set_data(
                data=test_embedding, train_or_test="test", type="embedding"
            )

        return random_self


def decode(
    embedding_train: np.ndarray,
    embedding_test: np.ndarray,
    labels_train: np.ndarray,
    labels_test: np.ndarray,
    labels_describe_space: bool = False,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
    n_folds: int = 5,
    detailed_metrics: bool = False,
    include_cv_stats: bool = False,
    multiply_by: float = 1.0,
    test_outlier_removal: bool = True,
    regression_outlier_removal_threshold: float = 0.004,
    min_train_class_samples: int = 200,
    min_test_class_samples: int = 30,
) -> Dict[str, Dict[str, Union[float, Dict]]]:
    """
    Decodes neural embeddings using k-Nearest Neighbors with automatic k selection.

    Before decoding, the function checks if the input data is valid and ensures that the training and test sets are compatible.
    Labels/Classes are checked to ensure that only those with sufficient training samples are used for testing.
    This function performs k-fold cross-validation to determine the optimal number of neighbors (k) for the kNN model.
    It then fits the kNN model with the optimal k and evaluates its performance on the test set.

    Parameters
    ----------
    embedding_train : np.ndarray
        Training embedding data
    embedding_test : np.ndarray
        Testing embedding data
    labels_train : np.ndarray
        Training target labels
    labels_test : np.ndarray
        Testing target labels
    labels_describe_space : bool, optional
        Whether to describe the label space (default: False).
        If True, the labels are assumed to be continuous and the prediction error is converted to a single number in euclidean space instead of multidimensional.
    n_neighbors : int, optional
        Number of neighbors for kNN (default: None, auto-determined via CV)
    metric : str, optional
        Distance metric for kNN (default: "cosine")
    n_folds : int, optional
        Number of folds for cross-validation (default: 5)
    detailed_metrics : bool, optional
        Whether to return detailed per-class metrics (default: False)
    include_cv_stats : bool, optional
        Whether to include cross-validation statistics (default: False)
    labels_describe_space : bool, optional
        If labels are describing space (default is False).
        If space is described the Frobenius norm is used to calculate the distance between decoded positional values.
    multiply_by : float, optional
        A multiplier to apply to the decoding statistics (default is 1.0). This is used to scale the decoding statistics.
    test_outlier_removal : bool, optional
        Whether to remove outliers from the test set (default: True)
    regression_outlier_removal_threshold : float, optional
        Threshold for outlier removal (default: 0.004). This is used to determine if a test sample is too far from the training samples.
        If the distance to the nearest training sample is greater than this threshold, the test sample is removed.
    min_train_class_samples : int, optional
        Minimum number of training samples per class (default: 200). This is used to determine if a class has enough training samples.
    min_test_class_samples : int, optional
        Minimum number of test samples per class (default: 30). This is used to determine if a class has enough test samples.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing decoding performance metrics
    """
    # Input validation
    if not all(
        isinstance(x, np.ndarray)
        for x in [embedding_train, embedding_test, labels_train, labels_test]
    ):
        raise ValueError("All input arrays must be numpy arrays")

    # Ensure 2D arrays
    labels_train = np.atleast_2d(labels_train)
    labels_test = np.atleast_2d(labels_test)

    # Determine if regression or classification
    is_regression = is_floating(labels_train) if not labels_describe_space else True
    knn_class = KNeighborsRegressor if is_regression else KNeighborsClassifier

    # Ensure only labels sufficiently available in training set are used for testing
    if test_outlier_removal:
        idx_remove = []
        if is_regression:
            # check if the test data is within range of the training data
            mins = np.min(labels_train, axis=0)
            maxs = np.max(labels_train, axis=0)
            ranges = maxs - mins
            if labels_describe_space:
                area = np.prod(ranges)
                min_acceptable_value = np.sqrt(
                    area * regression_outlier_removal_threshold
                )
            else:
                min_acceptable_value = ranges * regression_outlier_removal_threshold
            for loc in labels_test:
                if labels_describe_space:
                    dist = np.linalg.norm(loc - labels_train, axis=1)
                cl = np.min(dist)
                if cl > min_acceptable_value:
                    global_logger.info(
                        "Test sample removed from training set because too far from available training points",
                        cl,
                    )
                    idx_remove.append(k)
        else:
            # check if the test data class has sufficiently available training data samples
            for cl, num_test_samples in np.unique(labels_test, treturn_counts=True):
                num_train_samples = np.sum(labels_train == cl)
                if (
                    num_train_samples < min_train_class_samples
                    or num_test_samples < min_test_class_samples
                ):
                    idx_remove.extend(np.where(labels_test == cl)[0])
                    global_logger.info(
                        f"Test sample removed from training set because class {cl} has too few samples. Training samples: {num_train_samples}, Test samples: {num_test_samples}. At least {min_train_class_samples} training samples and {min_test_class_samples} test samples are needed."
                    )

        if len(idx_remove) > 0:
            global_logger.info(
                f"Removing {len(idx_remove)} test samples from training set"
            )
            embedding_test = np.delete(embedding_test, idx_remove, axis=0)
            labels_test = np.delete(labels_test, idx_remove, axis=0)

    # Define range of k values to test if n_neighbors is None
    if n_neighbors is None:
        max_k = min(embedding_train.shape[0] - 1, 50)  # Cap at 50 or n_samples-1
        k_range = np.unique(
            np.logspace(0, np.log10(max_k), num=10, base=10).astype(int)
        )
        # Internal CV to select best k
        global_logger.info(f"Performing internal CV to select best k")
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        k_scores = []

        for k in k_range:
            knn_model = knn_class(n_neighbors=k, metric=metric)
            fold_scores = []

            for train_idx, val_idx in kf.split(embedding_train):
                X_train_fold = embedding_train[train_idx]
                X_val_fold = embedding_train[val_idx]
                y_train_fold = labels_train[train_idx]
                y_val_fold = labels_train[val_idx]

                knn_model.fit(X_train_fold, y_train_fold)
                y_pred_fold = knn_model.predict(X_val_fold)

                score = (
                    r2_score(y_val_fold, y_pred_fold)
                    if is_regression
                    else accuracy_score(y_val_fold, y_pred_fold)
                )
                fold_scores.append(score)

            k_scores.append(np.mean(fold_scores))

        # Select best k
        best_k = k_range[np.argmax(k_scores)]
        global_logger.info(f"Best k: {best_k}")

    else:
        best_k = n_neighbors

    # Initialize final kNN model with best k
    knn_model = knn_class(n_neighbors=best_k, metric=metric)

    # Optional k-fold cross-validation for statistics (if requested)
    cv_results = []
    if include_cv_stats:
        for train_idx, val_idx in kf.split(embedding_train):
            X_train_fold = embedding_train[train_idx]
            X_val_fold = embedding_train[val_idx]
            y_train_fold = labels_train[train_idx]
            y_val_fold = labels_train[val_idx]

            knn_model.fit(X_train_fold, y_train_fold)
            y_pred_fold = knn_model.predict(X_val_fold)
            cv_results.append({"true": y_val_fold, "pred": np.atleast_2d(y_pred_fold)})

    # Final fit on full training data and predict on test
    knn_model.fit(embedding_train, labels_train)
    test_predictions = np.atleast_2d(knn_model.predict(embedding_test))

    # Calculate metrics based on label type
    results = {"best_k": int(best_k)}  # Include best k in output
    if is_regression:
        results.update(
            _compute_regression_metrics(
                labels_test,
                test_predictions,
                cv_results if include_cv_stats else None,
                labels_describe_space,
                multiply_by=multiply_by,
            )
        )
    else:
        results.update(
            _compute_classification_metrics(
                labels_test,
                test_predictions,
                cv_results if include_cv_stats else None,
                detailed_metrics,
            )
        )

    results["k_neighbors"] = best_k
    return results


def _compute_regression_metrics(
    labels_test: np.ndarray,
    test_predictions: np.ndarray,
    cv_results: Optional[list] = None,
    labels_describe_space: bool = False,
    multiply_by: float = 1.0,
) -> Dict[str, Union[float, Dict[str, Union[float, Dict]]]]:
    """Compute regression metrics with optional cross-validation results.

    Parameters
    ----------
    multiply_by : float, optional
        Factor to multiply the results by to normalize if original values have been transformed before (default: 1.0)
    """
    # Test set metrics
    err = np.abs(labels_test - test_predictions)
    err *= multiply_by
    if labels_describe_space:
        err = np.linalg.norm(err, axis=1)
    rmse = np.mean(err)
    error_variance = np.var(err)
    r2 = r2_score(labels_test, test_predictions)

    results = {
        "rmse": {"mean": float(rmse), "variance": float(error_variance)},
        "r2": float(r2),
    }

    # Include CV stats if requested
    if cv_results is not None:
        cv_rmse = [
            np.mean(np.abs(r["true"] - r["pred"])) * multiply_by for r in cv_results
        ]
        cv_r2 = [r2_score(r["true"], r["pred"]) for r in cv_results]
        results["cv_metrics"] = {
            "rmse": {"mean": float(np.mean(cv_rmse)), "std": float(np.std(cv_rmse))},
            "r2": {"mean": float(np.mean(cv_r2)), "std": float(np.std(cv_r2))},
        }

    return results


def _compute_classification_metrics(
    labels_test: np.ndarray,
    test_predictions: np.ndarray,
    cv_results: Optional[list] = None,
    detailed_metrics: bool = False,
) -> Dict[str, Union[float, Dict[str, Union[float, List[float]]]]]:
    """Compute classification metrics with optional cross-validation results."""
    n_outputs = labels_test.shape[1]
    test_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    roc_auc_data = {}

    for i in range(n_outputs):
        # Test metrics
        test_true = labels_test[:, i]
        test_pred = test_predictions[:, i]

        metric_funcs = {
            "accuracy": accuracy_score,
            "precision": lambda x, y: precision_score(
                x, y, average="macro", zero_division=0
            ),
            "recall": lambda x, y: recall_score(x, y, average="macro", zero_division=0),
            "f1": lambda x, y: f1_score(x, y, average="macro", zero_division=0),
        }

        for metric_name, func in metric_funcs.items():
            test_metrics[metric_name].append(func(test_true, test_pred))

        # ROC/AUC if requested
        if detailed_metrics:
            classes = np.unique(test_true)
            y_true_bin = label_binarize(test_true, classes=classes)
            y_pred_bin = label_binarize(test_pred, classes=classes)
            roc_auc_data[f"output_{i}"] = {}
            for j, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, j], y_pred_bin[:, j])
                auc = roc_auc_score(y_true_bin[:, j], y_pred_bin[:, j])
                roc_auc_data[f"output_{i}"][f"class_{cls}"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": float(auc),
                }

    results = {k: [float(x) for x in v] for k, v in test_metrics.items()}
    if detailed_metrics:
        results["roc_auc"] = roc_auc_data

    # Include CV stats if requested
    if cv_results is not None:
        cv_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        for metric_name, func in metric_funcs.items():
            cv_scores = [
                func(r["true"][:, i], r["pred"][:, i])
                for r in cv_results
                for i in range(n_outputs)
            ]
            cv_metrics[metric_name] = {
                "mean": float(np.mean(cv_scores)),
                "std": float(np.std(cv_scores)),
            }
        results["cv_metrics"] = cv_metrics

    return results


def structure_index(
    data: np.ndarray,
    labels,
    params: Dict[str, Union[int, bool, List[int]]],
    additional_title: str = "",
    plot: bool = False,
    plot_save_dir: Optional[Path] = None,
    as_pdf: bool = False,
) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:  # TODO: finish return types:

    if is_int_like(params["n_neighbors"]):
        parameter_sweep = False
    elif is_array_like(params["n_neighbors"]):
        parameter_sweep = True
        sweep_range = deepcopy(params["n_neighbors"])
    else:
        global_logger.error("n_neighbors must be an integer or a list of integers.")
        raise ValueError("n_neighbors must be an integer or a list of integers.")

    if parameter_sweep:
        structural_index = {}
        for n_neighbors in tqdm(sweep_range):
            params["n_neighbors"] = n_neighbors
            SI, binLabel, overlapMat, sSI = compute_structure_index(
                data, labels, **params
            )
            structural_index[n_neighbors] = {
                "SI": SI,
                "bin_label": binLabel,
                "overlap_mat": overlapMat,
                "shuf_SI": sSI,
            }
    else:
        SI, binLabel, overlapMat, sSI = compute_structure_index(data, labels, **params)
        structural_index = {
            "SI": SI,
            "bin_label": binLabel,
            "overlap_mat": overlapMat,
            "shuf_SI": sSI,
        }
        if plot:
            if not plot_save_dir:
                plot_save_dir = Path.cwd()
            Vizualizer.plot_structure_index(
                embedding=data,
                feature=labels,
                overlapMat=overlapMat,
                SI=SI,
                binLabel=binLabel,
                additional_title=additional_title,
                as_pdf=as_pdf,
                save_dir=plot_save_dir,
            )
    return structural_index
