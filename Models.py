# type hints
from typing import List, Union, Dict, Any, Tuple, Optional

# show progress bar
from tqdm import tqdm, trange

# calculations
import numpy as np
import sklearn
import scipy

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
        # TODO: catch possible error if model_settings is not a dict
        self.place_cell = PlaceCellDetectors(
            model_dir, model_id, model_settings["place_cell"]
        )
        self.cebras = Cebras(model_dir, model_id, model_settings["cebra"])

    def train(self):
        # TODO: move train_model function from task class
        pass

    def set_model_name(self):
        # TODO: move set_model_name function from task class
        pass

    def get_model(self):
        # TODO: move get_model function from task class
        pass


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
    ):
        # ..............add uncommented line again
        # if self.rate_map is None or self.time_map is None:
        if True:
            self.rate_map, self.time_map = self.get_rate_time_map(
                activity,
                binned_pos,
                smooth=smooth,
                window_size=window_size,
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
    def get_spike_map(activity, binned_pos, max_pos=None):
        num_cells = activity.shape[1]
        # for every frame count the activity of each cell
        max_pos = max_pos or max(binned_pos) + 1
        spike_map = np.zeros((num_cells, max_pos))
        for frame, rate_map_vec in enumerate(activity):
            pos_at_frame = binned_pos[frame]
            spike_map[:, pos_at_frame] += rate_map_vec
        return spike_map

    @staticmethod
    def get_rate_map(activity, binned_pos, max_pos=None):
        """
        outputs spike rate per position per time
        """
        spike_map = PlaceCellDetectors.get_spike_map(
            activity, binned_pos, max_pos=max_pos
        )
        # smooth and normalize
        # normalize by spatial occupancy
        time_map, _ = PlaceCellDetectors.get_time_map(
            binned_pos, bins=len(spike_map[0])
        )
        rate_map_occupancy = spike_map / time_map
        rate_map_occupancy = np.nan_to_num(rate_map_occupancy, nan=0.0)
        # .............is die summe richtig oder soll ich das anders machen?
        rate_map = rate_map_occupancy / np.sum(rate_map_occupancy)

        return rate_map, time_map

    def get_rate_time_map(
        self,
        activity,
        binned_pos,
        smooth=True,
        window_size=2,
    ):
        rate_map_org, time_map = self.get_rate_map(
            activity,
            binned_pos,
            max_pos=self.behavior_metadata["environment_dimensions"],
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
    def __init__(self, model_dir, model_id, model_settings=None, **kwargs):
        super().__init__(model_dir, model_id, model_settings, **kwargs)
        self.name = "si"
        self.model_settings = model_settings
        self.model_settings_start(self.name, model_settings)
        self.model_settings_end(self)

    def create_defaul_model(self):
        self.si_formula = "skaags"
        return self

    def define_parameter_save_path(self, model):
        save_path = self.model_dir.joinpath(
            f"place_cell_{model.name}_{self.model_id}.npz"
        )
        return save_path

    def is_fitted(self, model):
        return model.save_path.exists()

    def load_fitted_model(self, model):
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
        rate_map, time_map, spatial_information_method="opexebo"
    ):
        """
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

        # FIXME: this should be corrected or???................
        inf_content = inf_rate  # / mean_rate

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
        max_pos=None,
    ):
        """
        return spatial information and corresponding zscores
        """

        rate_map, time_map = PlaceCellDetectors.get_rate_map(
            activity,
            binned_pos,
            max_pos=self.behavior_metadata["environment_dimensions"],
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
            # shuffle the time map
            # time_map_shuffled = np.roll(
            #    time_map, np.random.choice(np.arange(time_map.shape[0]), 1)
            # )................
            binned_pos_shuffled = np.roll(
                binned_pos, np.random.choice(np.arange(binned_pos.shape[0]), 1)
            )
            rate_map_shuffled, time_map_shuffled = PlaceCellDetectors.get_rate_map(
                activity, binned_pos_shuffled, max_pos=max_pos
            )
            # inf_rate, _ = self.get_spatial_information(
            #    rate_map, time_map_shuffled, spatial_information_method
            # ).................
            inf_rate, _ = self.get_spatial_information(
                rate_map_shuffled, time_map_shuffled, spatial_information_method
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
        self.time_model = self.time()
        self.behavior_model = self.behavior()
        self.hybrid_model = self.hybrid()

    def define_parameter_save_path(self, model):
        save_path = self.model_dir.joinpath(
            f"cebra_{model.name}_iter-{model.max_iterations}_dim-{model.output_dimension}_model-{self.model_id}.pt"
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
        return initial_model

    def load_fitted_model(self, model):
        fitted_model_path = model.save_path
        if fitted_model_path.exists():
            fitted_model = CEBRA.load(fitted_model_path)
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


def decode(
    embedding_train,
    embedding_test,
    label_train,
    label_test,
    n_neighbors=36,
    metric="cosine",
):
    # TODO: Improve decoder
    # Define decoding function with kNN decoder. For a simple demo, we will use the fixed number of neighbors 36.
    decoder = cebra.KNNDecoder(n_neighbors=n_neighbors, metric=metric)
    label_train = Dataset.force_2d(label_train)
    # label_train = Dataset.force_1_dim_larger(label_train)

    label_test = Dataset.force_2d(label_test)
    # label_test = force_1_dim_larger(label_test)

    embedding_train, label_train = Datasets.force_equal_dimensions(
        embedding_train, label_train
    )

    decoder.fit(embedding_train, label_train[:, 0])

    pred = decoder.predict(embedding_test)

    prediction = np.stack([pred], axis=1)

    test_score = 0  # sklearn.metrics.r2_score(label_test, prediction)
    # test_score = sklearn.metrics.r2_score(label_test[:,:2], prediction)

    # pos_test_err = np.median(abs(prediction - label_test))
    ########################################################
    pos_test_err = np.mean(abs(prediction - label_test))
    ########################################################
    pos_test_score = sklearn.metrics.r2_score(label_test[:, 0], prediction[:, 0])

    return test_score, pos_test_err, pos_test_score
