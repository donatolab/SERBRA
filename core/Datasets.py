import os
from pathlib import Path

# type hints
from typing import List, Union, Dict, Any, Tuple, Optional

# plotting
import matplotlib.pyplot as plt

# calculations
import numpy as np
from sklearn.utils import shuffle

# load data
import cebra
from cebra import CEBRA

# own
from Helper import (
    check_needed_keys,
    get_str_from_dict,
    global_logger,
    force_1_dim_larger,
    force_equal_dimensions,
    encode_categorical,
    is_list_of_ndarrays,
    group_by_binned_data,
    pairwise_compare,
    compare_distribution_groups,
    calc_entropy,
    calc_kde_entropy,
    compute_density,
    bin_array,
    uncode_categories,
    save_file_present,
    make_list_ifnot,
    add_missing_keys,
    may_butter_lowpass_filter,
    fill_continuous_array,
    get_covariance_matrix,
    add_stream_lag,
)
from Visualizer import Vizualizer
from Setups import Femtonics, Thorlabs, Inscopix, Environment
from Setups import (
    Active_Avoidance,
    Treadmill_Setup,
    Trackball_Setup,
    Openfield_Setup,
    Wheel_Setup,
)


class Dataset:
    needed_keys = ["setup"]

    def __init__(
        self,
        key,
        metadata,
        path=None,
        data=None,
        raw_data_object=None,
        root_dir=None,
        task_id=None,
    ):
        # initialize plotting parameters
        super().__init__()
        self.root_dir: Path = root_dir
        self.path: Path = path
        self.data_dir = None
        self.data: np.ndarray = data
        self.binned_data: np.ndarray = None
        self.key = key
        self.task_id = task_id
        self.raw_data_object = raw_data_object
        self.metadata = metadata
        self.fps = None if "fps" not in self.metadata.keys() else self.metadata["fps"]
        check_needed_keys(metadata, Dataset.needed_keys)
        self.setup = self.get_setup(self.metadata["setup"])
        self.plot_attributes = Vizualizer.default_plot_attributes()
        self.category_map = None
        self.refine_metadata()

    def get_setup(self, setup_name, preprocessing_name, method_name):
        raise NotImplementedError(
            f"ERROR: Function get_setup is not defined for {self.__class__}."
        )

    def load(
        self,
        path=None,
        save=True,
        plot=True,
        regenerate=False,
        regenerate_plot=False,
    ):
        if not type(self.data) == np.ndarray:
            self.path = path if path else self.path
            self.data_dir = self.path.parent
            # if no raw data exists, regeneration is not possible
            if not self.raw_data_object and regenerate:
                global_logger.warning(f"No raw data given. Regeneration not possible.")
                global_logger.info(f"Loading old data.")
                print(
                    f"No raw data given. Regeneration not possible. Loading old data."
                )
                regenerate = False

            # Check if the file exists, and if it does, load the data, else create the dataset
            if path.exists() and not regenerate:
                global_logger.info(f"Loading {self.path}")
                print(f"Loading {self.path}")
                # ... and similarly load the .h5 file, providing the columns to keep
                # continuous_label = cebra.load_data(file="auxiliary_behavior_data.h5", key="auxiliary_variables", columns=["continuous1", "continuous2", "continuous3"])
                # discrete_label = cebra.load_data(file="auxiliary_behavior_data.h5", key="auxiliary_variables", columns=["discrete"]).flatten()
                self.data = cebra.load_data(file=self.path)
                if self.data.ndim == 2:
                    # format of data should be [num_time_points, num_cells]
                    self.data = force_1_dim_larger(data=self.data)
            else:
                global_logger.warning(
                    f"Generating data based on raw_data_object for {self.key} in {self.path}."
                )
                print(
                    f"Generating data based on raw_data_object for {self.key} in {self.path}."
                )
                self.create_dataset(self.raw_data_object, save=save)
                if save and not self.path.exists():
                    np.save(self.path, self.data)
        self.data = self.correct_data(self.data)
        self.refine_metadata()
        if plot:
            self.plot(regenerate_plot=regenerate_plot)
        self.binned_data = self.bin_data(self.data)
        return self.data

    def refine_metadata(self):
        self.define_plot_attributes()

    def define_plot_attributes(self):
        raise NotImplementedError(
            f"Function define_plot_attributes is not defined for {self.__class__}."
        )

    def create_dataset(self, raw_data_object=None, save=True):
        raw_data_object = raw_data_object or self.raw_data_object
        if self.raw_data_object:
            global_logger.info(
                f"Creating {self.key} dataset based on raw data from {raw_data_object.key}."
            )
            print(
                f"Creating {self.key} dataset based on raw data from {raw_data_object.key}."
            )
            data = self.process_raw_data(save=save)
            return data
        else:
            global_logger.warning(
                f"No raw data given. Creation not possible. Skipping."
            )
            print(f"No raw data given. Creation not possible. Skipping.")

    def process_raw_data(self, save=True):
        global_logger.critical(
            f"Function for creating {self.key} dataset from raw data is not defined."
        )
        raise NotImplementedError(
            f"ERROR: Function for creating {self.key} dataset from raw data is not defined."
        )

    def correct_data(self, data):
        return data

    def bin_data(self, data, bin_size=None):
        """
        binning is not applied to this type of data
        """
        return data

    def create_1d_discrete_labels(self, binned_data):
        # encode multi-dimensional binned_data (e.g individual x, y) to single vector combined X,Y Bin
        if len(self.metadata["environment_dimensions"]) != 1:
            encoded_data, self.category_map = encode_categorical(
                binned_data, return_category_map=True
            )
            return encoded_data
        else:
            return binned_data

    def refine_plot_attributes(
        self,
        title=None,
        ylable=None,
        xlimits=None,
        save_path=None,
    ):
        self.plot_attributes["title"] = (
            self.plot_attributes["title"] or title or f"{self.path.stem} data"
        )
        self.plot_attributes["title"] = (
            self.plot_attributes["title"]
            if self.plot_attributes["title"][-4:] == "data"
            else self.plot_attributes["title"] + " data"
        )

        self.plot_attributes["ylable"] = (
            self.plot_attributes["ylable"] or ylable or self.key
        )

        self.plot_attributes["xlimits"] = (
            self.plot_attributes["xlimits"] or xlimits or (0, len(self.data))
        )

        descriptive_metadata_keys = [
            "area",
            "stimulus_type",
            "method",
            "preprocessing_software",
            "setup",
        ]

        descriptive_metadata_txt = get_str_from_dict(
            dictionary=self.metadata, keys=descriptive_metadata_keys
        )
        if descriptive_metadata_txt not in self.plot_attributes["title"]:
            self.plot_attributes["title"] += f" {descriptive_metadata_txt}"

        self.plot_attributes["save_path"] = (
            self.plot_attributes["save_path"]
            or save_path
            or self.path.parent.parent.parent.joinpath(
                "figures", self.path.stem + ".png"
            )
        )

    def plot(
        self,
        figsize=None,
        title=None,
        xlable=None,
        xlimits=None,
        xticks=None,
        ylable=None,
        ylimits=None,
        yticks=None,
        seconds_interval=5,
        fps=None,
        num_ticks=50,
        save_path=None,
        regenerate_plot=None,
        show=False,
        dpi=300,
        as_pdf=False,
    ):
        self.refine_plot_attributes(
            title=title, ylable=ylable, xlimits=xlimits, save_path=save_path
        )
        plot_present = save_file_present(fpath=self.plot_attributes["save_path"])
        if regenerate_plot or not plot_present:
            self.plot_attributes = Vizualizer.default_plot_start(
                plot_attributes=self.plot_attributes,
                figsize=figsize,
                xlable=xlable,
                xlimits=xlimits,
                xticks=xticks,
                ylimits=ylimits,
                yticks=yticks,
                num_ticks=num_ticks,
                fps=fps,
            )
            self.plot_data()
            Vizualizer.default_plot_ending(
                plot_attributes=self.plot_attributes,
                regenerate_plot=True,
                show=show,
                dpi=dpi,
                as_pdf=as_pdf,
            )
        else:
            Vizualizer.plot_image(
                plot_attributes=self.plot_attributes,
                show=show,
            )

    def plot_data(self):
        # dimensions =  self.data.ndim
        dimensions = len(self.metadata["environment_dimensions"])
        if dimensions == 1:
            Vizualizer.data_plot_1D(
                data=self.data,
                plot_attributes=self.plot_attributes,
            )
        elif dimensions == 2:
            if self.key != "position":
                raw_data_object = self.raw_data_object
                while raw_data_object is not None:
                    if raw_data_object.key == "position":
                        position_data = raw_data_object.data
                        break
                    else:
                        raw_data_object = raw_data_object.raw_data_object
            else:
                position_data = self.data
            Vizualizer.data_plot_2D(
                data=self.data,
                position_data=position_data,
                border_limits=self.metadata["environment_dimensions"],
                plot_attributes=self.plot_attributes,
            )

    @staticmethod
    def split(data, split_ratio=0.8):
        global_logger.info(
            f"Splitting data into {split_ratio:.0%}% training and {1-split_ratio:.0%} testing."
        )
        split_index = int(len(data) * split_ratio)
        data_train = data[:split_index]
        data_test = data[split_index:]
        return data_train, data_test

    @staticmethod
    def shuffle(data):
        global_logger.info(f"Shuffling data.")
        return shuffle(data)

    @staticmethod
    def filter_by_idx(data, idx_to_keep=None):
        if isinstance(idx_to_keep, np.ndarray) or isinstance(idx_to_keep, list):
            global_logger.info(f"Filtering data by idx_to_keep.")
            data_unfiltered, idx_to_keep = force_equal_dimensions(data, idx_to_keep)
            data_filtered = data_unfiltered[idx_to_keep]
            return data_filtered
        else:
            global_logger.info(f"No idx_to_keep given. Returning unfiltered data.")
            return data

    @staticmethod
    def force_2d(data: np.ndarray, transepose=True):
        data_2d = data
        if data_2d.ndim == 1:
            data_2d = data_2d.reshape(1, -1)
            if transepose:
                data_2d = data_2d.T
        elif data_2d.ndim > 2:
            raise ValueError("Data has more than 2 dimensions.")
        return data_2d

    @staticmethod
    def manipulate_data(
        data, idx_to_keep=None, shuffle=False, split_ratio=None, as_list=False
    ):
        """
        Manipulates the input data by applying filtering, shuffling, splitting,
        and enforcing a 2D structure on the resulting subsets.

        Parameters:
        -----------
        data : list or any type
            The input data to be manipulated. If not a list, the data is first converted into a list.

        idx_to_keep : list or None, optional
            A list of indices to filter the data. Only data at these indices will be retained.
            If None, no filtering is applied. Default is None.

        shuffle : bool, optional
            If True, the filtered data will be shuffled before splitting.
            If False, the data remains in its original order. Default is False.

        split_ratio : float or None, optional
            The ratio used to split the data into training and testing subsets.
            If None, no splitting is applied and the entire dataset is considered training data. Default is None.

        Returns:
        --------
        data_train : list
            A list containing the training subsets of the input data.
            Each subset has been filtered, shuffled (if requested), split, and enforced to be 2D.

        data_test : list
            A list containing the testing subsets of the input data.
            Each subset has been filtered, shuffled (if requested), split, and enforced to be 2D.

        Notes:
        ------
        - The function assumes the existence of the `Dataset` class with the following methods:
            - `filter_by_idx(data, idx_to_keep)`: Filters data by keeping only the specified indices.
            - `shuffle(data)`: Shuffles the input data.
            - `split(data, split_ratio)`: Splits the data into training and testing subsets based on the `split_ratio`.
            - `force_2d(data)`: Ensures that the data has a 2D structure.
        """
        if as_list:
            if not is_list_of_ndarrays(data):
                data = [data]
            data_train = []
            data_test = []
            for i, data_i in enumerate(data):
                data_filtered = Dataset.filter_by_idx(data_i, idx_to_keep=idx_to_keep)
                data_shuffled = (
                    Dataset.shuffle(data_filtered) if shuffle else data_filtered
                )
                data_train_i, data_test_i = Dataset.split(data_shuffled, split_ratio)
                data_train_i = Dataset.force_2d(data_train_i)
                data_test_i = Dataset.force_2d(data_test_i)
                data_train.append(data_train_i)
                data_test.append(data_test_i)
        else:
            data_filtered = Dataset.filter_by_idx(data, idx_to_keep=idx_to_keep)
            data_shuffled = Dataset.shuffle(data_filtered) if shuffle else data_filtered
            data_train, data_test = Dataset.split(data_shuffled, split_ratio)
            data_train = Dataset.force_2d(data_train)
            data_test = Dataset.force_2d(data_test)
        return data_train, data_test


class BehaviorDataset(Dataset):
    def __init__(
        self,
        key,
        path=None,
        data=None,
        raw_data_object=None,
        metadata=None,
        root_dir=None,
        task_id=None,
    ):
        super().__init__(
            key=key,
            path=path,
            data=data,
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        self.plot_attributes["fps"] = (
            self.metadata["imaging_fps"]
            if "imaging_fps" in self.metadata.keys()
            else self.metadata["fps"]
        )
        # default binning size is 1cm
        self.binning_size = None
        self.max_bin = None

    def occupancy_by_binned_feature(
        self,
        data=None,
        group_values="count_norm",
        idx_to_keep=None,
        plot=False,
        additional_title="",
        figsize=(6, 5),
        xticks=None,
        xticks_pos=None,
        yticks=None,
        ylabel="",
        save_dir=None,
        save_plot=False,
    ):

        data = data if data is not None else self.binned_data
        filtered_data = self.filter_by_idx(data, idx_to_keep)
        if "max_bin" not in self.__dict__.keys():
            self.max_bin = None
        occupancy, _ = group_by_binned_data(
            data=filtered_data,
            binned_data=filtered_data,
            category_map=self.category_map,
            group_values=group_values,
            max_bin=self.max_bin,
            as_array=True,
        )
        occupancy[occupancy == 0] = np.nan
        if plot:
            title = f"Occupancy Map {self.metadata['task_id']} {additional_title}"
            if save_plot:
                save_dir = save_dir or self.plot_attributes["save_path"].parent
            else:
                save_dir = None

            if self.key == "stimulus":
                xticks = xticks or self.plot_attributes["yticks"][1]
                xticks_pos = xticks_pos or self.plot_attributes["yticks"][0]
            else:
                unique_bins = np.array(list(self.category_map.keys()))
                xticks = np.unique(unique_bins[:, 0])
                yticks = np.unique(unique_bins[:, 1])
                xticks_pos = xticks
                yticks_pos = yticks

            # Plot occupancy map
            Vizualizer.plot_heatmap(
                occupancy,
                figsize=figsize,
                title=title,
                xticks=xticks,
                xticks_pos=xticks_pos,
                yticks=yticks,
                yticks_pos=yticks_pos,
                xlabel=f"{self.key.capitalize()} bins",
                ylabel=ylabel,
                save_dir=save_dir,
            )

        return occupancy

    @property
    def relative_data(self):
        """
        Normalize the data to the environment dimensions from 0 to 1.

        Self.data is expected to be in the form of [x, y] or [x, y, z].

        Returns:
        --------
        relative : np.ndarray
            The relative data.
        """
        return Environment.to_relative(self.data)

    def get_setup(self, setup_name):
        self.setup_name = self.metadata["setup"]
        if setup_name == "active_avoidance":
            setup = Active_Avoidance(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        elif setup_name == "openfield":
            setup = Openfield_Setup(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        elif setup_name == "treadmill":
            setup = Treadmill_Setup(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        elif setup_name == "trackball":
            setup = Trackball_Setup(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        elif setup_name == "wheel":
            setup = Wheel_Setup(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        else:
            raise ValueError(f"Behavior setup {setup_name} not supported.")
        return setup


class NeuralDataset(Dataset):
    needed_keys = ["method", "preprocessing", "processing", "setup"]

    def __init__(
        self,
        key,
        path=None,
        data=None,
        raw_data_object=None,
        metadata=None,
        root_dir=None,
        task_id=None,
    ):
        super().__init__(
            key=key,
            path=path,
            data=data,
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        check_needed_keys(metadata, NeuralDataset.needed_keys)
        self.embedding = None

    def create_dataset(self, raw_data_object=None, save=True):
        self.data = self.process_raw_data(save=save)
        return

    def embedd_data(self, model):
        self.embedding = model.transform(self.data)
        return self.embedding

    def get_data(
        self,
        use_embedding: bool = False,
        model: CEBRA = None,
        idx_to_keep: List[bool] = None,
    ):
        """
        Retrieves and optionally processes data based on specified parameters.

        Parameters
        ----------
        use_embedding : bool, optional
            Whether to use the embedding representation of the data. Default is False.
        model : object, optional
            A model object to generate embeddings if `use_embedding` is True. If `None`,
            an existing embedding (if available) will be used. Default is None.
        idx_to_keep : list or array-like, optional
            Indices specifying which data points to retain in the output. If None, all data is kept. Default is None.

        Returns
        -------
        filtered_neural_data : array-like
            The filtered data, either in its raw form or as embeddings, based on the provided parameters.

        Raises
        ------
        ValueError
            If `use_embedding` is True and no embedding is available or model is provided.

        Notes
        -----
        - If `use_embedding` is True and `model` is provided, a new embedding is generated.
        - If `use_embedding` is False but a `model` is provided, the raw data is used and a warning message is printed.
        - Filters the resulting data using the `idx_to_keep` parameter if specified.
        """
        if use_embedding:
            if model is None:
                if self.embedding is None:
                    global_logger.error(
                        f"Embedding was not generated. Model needed for embedding."
                    )
                    raise (f"Embedding was not generated. Model needed for embedding.")
                else:
                    data = self.embedding
            else:
                if self.embedding is not None:
                    global_logger.warning(
                        f"Embedding already calculated. Overwriting with new model."
                    )
                data = self.embedd_data(model)
        else:
            if model is not None:
                global_logger.warning(
                    f"Model given but embedding not requested. Using raw data."
                )
            data = self.data
        filtered_neural_data = self.filter_by_idx(data, idx_to_keep=idx_to_keep)
        return filtered_neural_data

    def get_process_data(self, type:str="processed", fit_to_shape:Union[None, Tuple[int, int]]=None):
        """
        Get the data of the task.
        
        The data is processed and has 1 dimension larger to create the format of [time, features].
        
        Parameters
        ----------
        type : str, optional
            The type of data to get (default is "processed"). Other options could be available, depending on the function of setup.preprocess.process.data 
            Options are:
                - "processed": the processed data
                - "unprocessed": the unprocessed data
        
        fit_to_shape : tuple, optional
            The shape to fit the data to. Default is None, which does not change the shape of the data.
            
        Returns
        -------
            data : np.ndarray
                The data of the task with 1 dimension larger than the number of features. 
        """
        data = self.setup.preprocess.process.data(type=type)
        data = force_1_dim_larger(data)
        if fit_to_shape:
            empty_data = np.zeros(fit_to_shape)
            data, _ = force_equal_dimensions(data, empty_data)
        return data

    def similarity(
        self,
        similarities: np.ndarray = None,
        metric: str = "cosine",
        model=None,
        additional_title="",
        use_embedding: bool = False,
        binned_features: np.ndarray = None,
        category_map: Dict = None,
        inside_bin_similarity: bool = False,
        remove_outliers: bool = True,
        out_det_method: str = "density",
        parallel: bool = True,
        max_bin: List[int] = None,
        show_frames: int = None,
        idx_to_keep: np.ndarray = None,
        xticks=None,
        xticks_pos=None,
        xlabel=None,
        ylabel=None,
        title=None,
        figsize=(6, 5),
        plot: bool = False,
        save_plot: bool = False,
        save_dir=None,
    ):
        """
        metrics: euclidean, wasserstein, kolmogorov-smirnov, chi2, kullback-leibler, jensen-shannon, energy, mahalanobis, cosine
        the compare_distributions is also removing outliers on default

        Args:
            category map: maps discrete labels to multi dimensional position vectors

        """
        filtered_neural_data = self.get_data(use_embedding, model, idx_to_keep)

        if binned_features is not None:
            filtered_binned_features = self.filter_by_idx(
                binned_features, idx_to_keep=idx_to_keep
            )
            if inside_bin_similarity:
                # Calculate similarity between vectors
                if similarities is None:
                    similarities = pairwise_compare(filtered_neural_data, metric=metric)

                # Calculate similarity inside binned features
                binned_similarities, _ = group_by_binned_data(
                    data=similarities,
                    category_map=category_map,
                    binned_data=filtered_binned_features,
                    group_values="mean_symmetric_matrix",
                    max_bin=max_bin,
                    as_array=True,
                )
                title = "Similarity Inside Binned Features"
                title += f" Embedded" if use_embedding else ""
                unique_bins = np.array(list(category_map.keys()))
                xticks = np.unique(unique_bins[:, 0])
                yticks = np.unique(unique_bins[:, 1])
                xticks_pos = xticks
                yticks_pos = yticks
                to_show_similarities = binned_similarities
            else:
                group_vectors, bins = group_by_binned_data(
                    data=filtered_neural_data,
                    category_map=category_map,
                    binned_data=filtered_binned_features,
                    group_values="raw",
                    max_bin=max_bin,
                    as_array=False,
                )
                max_bin = np.max(bins, axis=0) + 1 if max_bin is None else max_bin
                max_bin = max_bin.astype(int)

                # This is a heuristic to determine the distance between two distributions, needed for density based outlier detection
                # based on the amount of space every bin has on the surface of a sphere
                neighbor_distance = np.sqrt(4 / len(group_vectors))

                if out_det_method == "density" and use_embedding == False:
                    global_logger.warning(
                        "WARNING: Density based outlier detection is not recommended for high dimensional data. Euclidean distance is used for samples distance calculation."
                    )

                if similarities is None:
                    similarities = compare_distribution_groups(
                        group_vectors=group_vectors,
                        max_bin=max_bin,
                        metric=metric,
                        neighbor_distance=neighbor_distance,
                        filter_outliers=remove_outliers,
                        parallel=parallel,
                        out_det_method=out_det_method,
                    )
                else:
                    if not isinstance(similarities, dict):
                        global_logger.error(
                            "Similarities should be dictionary. For plotting similarities in 2D"
                        )
                        raise ValueError(
                            "Similarities should be dictionary. For plotting similarities in 2D"
                        )
                    if not isinstance(list(similarities.values())[0], np.ndarray):
                        global_logger.error(
                            "Similarities inside dict should be numpy array."
                        )
                        raise ValueError(
                            "Similarities inside dict should be numpy array."
                        )

                bins = np.array(bins)
                plot_bins = xticks or bins
                if not xticks:
                    ticks = []
                    tick_steps = []
                    for dim in range(bins.ndim):
                        ticks.append(np.unique(bins[:, dim]))
                        tick_steps.append(int(len(ticks[-1]) / 3))
                    xticks = ticks[0]
                    yticks = ticks[1] if len(ticks) > 1 else None
                else:
                    xticks = bins
                additional_title += f" Embedded" if use_embedding else ""
                additional_title += f" from and to each Bin {self.metadata['task_id']}"
        else:
            # No binned features, calculate similarity directly
            if similarities is None:
                similarities = pairwise_compare(filtered_neural_data, metric=metric)
            else:
                if not isinstance(similarities, np.ndarray):
                    global_logger.error(
                        "Similarities should be numpy array. For inside bin similarity plotting."
                    )
                    raise ValueError(
                        "Similarities should be numpy array. For inside bin similarity plotting."
                    )
                if similarities.ndim != 2:
                    global_logger.error(
                        "Similarities should be 2D array. For plotting similarities in 2D"
                    )
                    raise ValueError(
                        "Similarities should be 2D array. For plotting similarities in 2D"
                    )

            inside_bin_similarity = True
            to_show_similarities = (
                similarities[:show_frames, :show_frames]
                if isinstance(show_frames, int)
                else similarities
            )
            title = title or f"Neural {metric} Similarity {self.metadata['task_id']}"
            yticks = None
            yticks_pos = None
            xlabel = xlabel or "Frames"
            ylabel = ylabel or "Frames"

        if plot:
            if save_plot:
                save_dir = save_dir or self.plot_attributes["save_path"].parent
            if inside_bin_similarity:
                title += f" {self.metadata['task_id']}"
                Vizualizer.plot_heatmap(
                    to_show_similarities,
                    additional_title=additional_title,
                    figsize=figsize,
                    title=title,
                    xticks=xticks,
                    yticks=yticks,
                    xticks_pos=xticks_pos,
                    yticks_pos=yticks_pos,
                    colorbar_label=metric,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    save_dir=save_dir,
                )
            else:
                Vizualizer.plot_group_distr_similarities(
                    {metric: similarities},
                    additional_title=additional_title,
                    bins=plot_bins,
                    colorbar=True,
                    xticks=xticks,
                    yticks=yticks,
                    tick_steps=tick_steps,
                    colorbar_label=metric,
                    save_dir=save_dir,
                )
        return similarities

    def information_content(
        self,
        method="kde",
        model=None,
        additional_title="",
        idx_to_keep: np.ndarray = None,
        use_embedding: bool = False,
        binned_features: np.ndarray = None,
        category_map: Dict = None,
        max_bin: List[int] = None,
        remove_outliers: bool = True,
        outlier_threshold: float = 0.2,
        plot=False,
        plot_density=False,
        figsize=(6, 5),
        save_plot: bool = False,
        save_dir=None,
        normalize=True,
        normalize_by_occupancy=False,
    ):
        """
        Evaluates the amount of information in the neural data. Based on the distribution of samples labeld by the binned features.
        Outlier are always removed based on the density of the samples in the feature space.

        Parameter:
            - method: str
                The method used to calculate the information content. Default is "entropy".
            - use_embedding: bool
                If True, the data is embedded before calculating the information content. Default is False.
            - model: cebra.model for embedding data
            - idx_to_keep: np.ndarray
                The indices of the samples to keep. Default is None.
            - binned_features: np.ndarray
                The binned features used to group the data. Default is None.
            - category_map: Dict
                The mapping of the discrete labels to the multi-dimensional position vectors. Default is None.
            - max_bin: List[int]
                The maximum bin for each dimension. Default is None and will be calculated based on the binned features (not perfect)
            - remove_outliers: bool
                If True, outliers are removed based on the density of the samples in the feature space. Default is True.
            - outlier_threshold: float
                The threshold used to determine outliers. Default is 0.2.
            - plot: bool
                If True, the information content is plotted. Default is False.
            - plot_density: bool
                If True, the density of the samples in the feature space is plotted. Default is False.
            - use_alpha: bool
                If True, the alpha value is used for plotting. Default is False.
            - plot_legend: bool
                If True, the legend is plotted. Default is False.
            - figsize: Tuple[int, int]
                The size of the figure. Default is (6, 5).
            - save_plot: bool
                If True, the plot is saved. Default is False.
            - save_dir: Path
                The directory where the plot is saved. Default is None.
            - normalize: bool
                If True, the entropy is normalized. Default is True.
        """
        if not use_embedding:
            print(
                f"WARNING: information content of raw data may be unusefull, since the number of features is high."
            )
            global_logger.warning(
                f"WARNING: information content of raw data may be unusefull, since the number of features is high."
            )

        # calculate density of samples in feature space
        densities = self.density(
            model=model,
            idx_to_keep=idx_to_keep,
            use_embedding=use_embedding,
            binned_features=binned_features,
            category_map=category_map,
            max_bin=max_bin,
            remove_outliers=remove_outliers,
            outlier_threshold=outlier_threshold,
            plot=plot_density,
            use_alpha=False,
        )

        inf_contents = {}
        max_inf_content = 0
        if method == "volume":
            print(
                f"WARNING: Volume of covariance matrix is not a measure of information content."
            )
        for group_name, data in densities.items():
            locations = data["locations"]
            group_densities = data["values"]
            # calculate entropy
            if method == "entropy":
                inf_content = calc_entropy(
                    group_densities, convert_to_probabilities=True, normalize=normalize
                )
            elif method == "kde":
                inf_content = calc_kde_entropy(
                    locations, bandwidth=None, normalize=normalize
                )
            else:
                raise ValueError(f"Method {method} not supported.")

            # normalize by occupancy
            if normalize_by_occupancy:
                occupancy = len(group_densities)
                inf_content /= occupancy
                max_inf_content = max(max_inf_content, inf_content)

            inf_contents[group_name] = inf_content

        if normalize_by_occupancy:
            for group_name, entropy in inf_contents.items():
                inf_contents[group_name] = entropy / max_inf_content

        if plot:
            # convert entropies dict to array for heatmap plotting
            unique_bins = np.unique(list(densities.keys()), axis=0)
            max_bins = np.max(unique_bins, axis=0) + 1
            heatmap_data = np.zeros(max_bins)
            for group_name, entropy in inf_contents.items():
                heatmap_data[group_name[0], group_name[1]] = entropy

            title = "KDE based Entropy" if method == "kde" else "Entropy"
            title += f" of density distributions {self.metadata['task_id']}"
            title += " embedded" if use_embedding else ""
            title += " filtered" if remove_outliers else ""
            colorbar_label = "Entropy" if not normalize else "% of Maximum Entropy"
            colorbar_label += " per Occupancy" if normalize_by_occupancy else ""

            if save_plot:
                save_dir = save_dir or self.plot_attributes["save_path"].parent
            else:
                save_dir = None
            xticks = np.unique(unique_bins[:, 0])
            yticks = np.unique(unique_bins[:, 1])
            xticks_pos = xticks
            yticks_pos = yticks
            Vizualizer.plot_heatmap(
                heatmap_data,
                title=title,
                additional_title=additional_title,
                figsize=figsize,
                xticks=xticks,
                yticks=yticks,
                xticks_pos=xticks_pos,
                yticks_pos=yticks_pos,
                xlabel="Position Bin X",
                ylabel="Position Bin Y",
                colorbar_label=colorbar_label,
                save_dir=save_dir,
            )
        return inf_contents

    def density(
        self,
        binned_features: Union[np.ndarray, List[Union[int, float]]] = None,
        model=None,
        additional_title="",
        idx_to_keep: np.ndarray = None,
        use_embedding: bool = False,
        category_map: Dict = None,
        max_bin: List[int] = None,
        remove_outliers: bool = True,
        outlier_threshold: float = 0.2,
        plot=False,
        plot_legend=False,
        use_alpha=False,
        save_plot: bool = False,
        save_dir=None,
    ):
        filtered_neural_data = self.get_data(use_embedding, model, idx_to_keep)
        filtered_binned_features = self.filter_by_idx(
            binned_features, idx_to_keep=idx_to_keep
        )
        group_vectors, bins = group_by_binned_data(
            data=filtered_neural_data,
            category_map=category_map,
            binned_data=filtered_binned_features,
            group_values="raw",
            max_bin=max_bin,
            as_array=False,
        )
        max_bin = np.max(bins, axis=0) + 1 if max_bin is None else max_bin

        # This is a heuristic to determine the distance between two distributions, needed for density based outlier detection
        # based on the amount of space every bin has on the surface of a sphere
        neighbor_distance = np.sqrt(4 / len(group_vectors))

        usefull_idx = []
        densities = {}
        for group_name, locations in group_vectors.items():
            group_densities = compute_density(locations, neighbor_distance)
            densities[group_name] = {"locations": locations, "values": group_densities}
            if remove_outliers:
                usefull_ids = np.where(group_densities > outlier_threshold)
                usefull_idx.append(usefull_ids)

        additional_title += f" Bin {self.metadata['task_id']}"

        if plot is not None:
            if save_plot:
                save_dir = save_dir or self.plot_attributes["save_path"].parent
            else:
                save_dir = None
        if plot == True or plot == "2d":
            Vizualizer.plot_2d_group_scatter(
                densities,
                additional_title=additional_title,
                plot_legend=plot_legend,
                use_alpha=use_alpha,
                filter_outlier=remove_outliers,
                outlier_threshold=outlier_threshold,
                save_dir=save_dir,
                same_subplot_range=True,
            )
        elif plot == "3d":
            Vizualizer.plot_3D_group_scatter(
                densities,
                additional_title=additional_title,
                plot_legend=plot_legend,
                use_alpha=use_alpha,
                filter_outlier=remove_outliers,
                outlier_threshold=outlier_threshold,
                save_dir=save_dir,
            )

        if remove_outliers:
            densities_filtered = {}
            for (group_name, data), usefull_ids in zip(densities.items(), usefull_idx):
                densities_filtered[group_name] = {
                    "locations": data["locations"][usefull_ids],
                    "values": data["values"][usefull_ids],
                }
            densities = densities_filtered
        return densities


class Data_Position(BehaviorDataset):
    """
    Dataset class for managing position data.
    Attributes:
        - data (np.ndarray): The position data.
        - binned_data (np.ndarray): The binned position data into default: 1cm bins.
        - raw_position (np.ndarray): The raw position data.
        - lap_starts (np.ndarray): The indices of the lap starts.
        - environment_dimensions (List): The dimensions of the environment.
    """

    def __init__(
        self, raw_data_object=None, metadata=None, root_dir=None, task_id=None
    ):
        super().__init__(
            key="position",
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        self.max_bin = None
        self.raw_position = None
        self.lap_starts = None

    def refine_metadata(self):
        if self.metadata["environment_dimensions"] is None:
            if self.data is not None:
                global_logger.warning(
                    f"No Environmental Data is provided. Defining environment dimensions based on position data."
                )
                self.metadata["environment_dimensions"] = (
                    Environment.define_border_by_pos(self.data, map=False)
                )
        elif self.metadata["environment_dimensions"] is not None:
            self.metadata["environment_dimensions"] = make_list_ifnot(
                self.metadata["environment_dimensions"]
            )

        if not "binning_size" in self.metadata.keys():
            if self.metadata["environment_dimensions"] is not None:
                self.binning_size = self.gues_binning_size()
        else:
            self.binning_size = self.metadata["binning_size"]

        self.define_plot_attributes()

    def gues_binning_size(self):
        environment_dimensions = len(self.metadata["environment_dimensions"])
        # 1D environment
        if environment_dimensions == 1:
            return 0.01
        # 2D environment
        elif environment_dimensions == 2:
            return 0.05
        else:
            return 0.01

    def define_plot_attributes(self):
        if self.metadata["environment_dimensions"] is not None:
            if len(self.metadata["environment_dimensions"]) == 1:
                self.plot_attributes["ylable"] = "position cm"
            elif len(self.metadata["environment_dimensions"]) == 2:
                self.plot_attributes["figsize"] = (12, 10)

    def plot_data(self):
        if self.data.ndim == 1:
            marker = "^" if self.lap_starts is not None else None
            Vizualizer.data_plot_1D(
                data=self.data,
                plot_attributes=self.plot_attributes,
                marker_pos=self.lap_starts,
                marker=marker,
            )
        elif self.data.ndim == 2:
            Vizualizer.data_plot_2D(
                data=self.data,
                position_data=self.data,
                # border_limits=self.metadata["environment_dimensions"],
                plot_attributes=self.plot_attributes,
                color_by="time",
            )

    def bin_data(self, data=None, bin_size=None, return_category_map=False):
        """
        Bin the position data into 1cm bins in 1D and 5cm^2 bins for 2D environments.
        Args:
            - data (np.ndarray): The position data.
            - bin_size (float): The size of the bins in meters.
        return: binned_data (np.ndarray): The binned position data into cm bins (default 1cm).
        """
        if data is None:
            data = self.data
        if bin_size is None:
            bin_size = self.binning_size
        dimensions = self.metadata["environment_dimensions"]
        if len(dimensions) == 1:
            min_bins = 0
            max_bins = dimensions
        elif len(dimensions) == 2:
            borders = Environment.define_border_by_pos(data)
            min_bins = borders[:, 0]
            max_bins = borders[:, 1]
        binned_data = bin_array(
            data, bin_size=bin_size, min_bin=min_bins, max_bin=max_bins
        )
        max_bin = np.ceil(np.array(dimensions) / np.array(bin_size)).astype(int) - 1
        # convert to 1d if 2d
        if len(max_bin.shape) != 1:
            max_bin = max_bin[:, 1]
            raise ValueError(
                "!!!! check why !!!!. Only 1D and 2D environments are supported."
            )
        self.max_bin = max_bin
        encoded_data = self.create_1d_discrete_labels(binned_data)
        if return_category_map:
            return encoded_data, self.category_map
        return encoded_data

    def create_dataset(self, raw_data_object=None, save=True):
        data = self.process_raw_data(save=save)
        return data

    def process_raw_data(self, save=True):
        self.data = self.setup.process_data(save=save)
        return self.data

    def get_lap_starts(self, fps=None, sec_thr=5):
        # TODO: This is only working for 1D (laps)
        if self.lap_starts and sec_thr == 5:
            return self.lap_starts
        fps = fps or self.metadata["imaging_fps"]
        # get index of beeing at lap start
        at_start_indices = np.where(self.bin_data() == 0)[0]
        # take first indices in case multiple frames are at start
        num_frames_threshold = fps * sec_thr
        start_indices = [at_start_indices[0]]
        old_idx = at_start_indices[0]
        for index in at_start_indices[1:]:
            if index - old_idx > num_frames_threshold:
                start_indices.append(index)
                old_idx = index
        return np.array(start_indices)

    def categorical_to_bin_position(
        self,
        values: Union[Dict[tuple, Any], np.ndarray],
        additional_title="",
        figsize=(7, 5),
        plot=True,
        save_dir=None,
        as_pdf=False,
    ):
        """
        parameters:
            values: dict or np.ndarray
                if dict: values are in dictionary with category as key
                if np.ndarray: values are already in 2D
        """
        # check if category map is 2D
        if self.category_map is None:
            raise ValueError("No category map found. Please bin data first.")

        values_array = np.full(self.max_bin, np.nan)

        if isinstance(values, dict):  # if values are in dictionary with category as key
            key_example = list(values.keys())[0]
            if (
                isinstance(key_example, tuple) and len(key_example) == 2
            ):  # if key is already 2D positional coordinates
                for bin_pos, value in values.items():
                    values_array[bin_pos] = value
            else:
                category_map_np = np.array(list(self.category_map.keys()))
                category_map_dimensions = category_map_np.shape[1]
                if category_map_dimensions != 2:
                    raise NotImplementedError(
                        "Only 2D category maps are supported. For plotting values in 2D"
                    )

                uncode_values = uncode_categories(values, self.category_map)
                for bin_pos, value in uncode_values.items():
                    values_array[bin_pos] = value

        elif isinstance(values, np.ndarray):  # if values are already in 2D
            if values.ndim != 2:
                raise ValueError("Values should be 2D array. For plotting values in 2D")
            if values.shape != values_array.shape:
                raise ValueError(
                    f"Values shape {values.shape} should be equal to category map shape {values_array.shape}"
                )
            values_array = values

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            xticks_pos = list(range(self.max_bin[0]))
            yticks_pos = list(range(self.max_bin[1]))
            xticks = [i if i % 2 == 0 else "" for i in xticks_pos]
            yticks = [i if i % 2 == 0 else "" for i in yticks]
            title = f"Accuracy for every class {additional_title} {self.task_id}"
            Vizualizer.heatmap_subplot(
                values_array,
                ax=ax,
                title=title,
                title_size=figsize[0] * 1.5,
                xticks=xticks,
                yticks=yticks,
                xticks_pos=xticks_pos,
                yticks_pos=yticks_pos,
                xlabel="X Bin",
                ylabel="Y Bin",
                colorbar=True,
            )
            # add colorbar
            plt.tight_layout()
            Vizualizer.save_plot(save_dir, title, "pdf" if as_pdf else "png")
            plt.show()

        return values_array


class Data_Stimulus(BehaviorDataset):
    optional_keys = [
        "stimulus_dimensions",
        "stimulus_sequence",
        "stimulus_type",
        "stimulus_by",
        "fps",
    ]

    def __init__(
        self, raw_data_object=None, metadata=None, root_dir=None, task_id=None
    ):
        super().__init__(
            key="stimulus",
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )

    def relative_data(self):
        global_logger.warning(
            f"Relative data is not implemented for {self.__class__}. Returning self.data"
        )
        return self.data

    def define_optional_attributes(self):
        self.metadata = add_missing_keys(
            self.metadata, Data_Stimulus.optional_keys, fill_value=None
        )
        self.stimulus_sequence = self.metadata["stimulus_sequence"]
        self.stimulus_dimensions = self.metadata["stimulus_dimensions"]
        self.stimulus_type = self.metadata["stimulus_type"]
        self.stimulus_by = self.metadata["stimulus_by"] or "location"
        self.fps = self.metadata["fps"] if "fps" in self.metadata.keys() else None

    def define_plot_attributes(self):
        self.define_optional_attributes()
        if self.metadata["environment_dimensions"] is not None:
            if len(self.metadata["environment_dimensions"]) == 1:
                self.plot_attributes["ylable"] = "position cm"
            elif len(self.metadata["environment_dimensions"]) == 2:
                self.plot_attributes["figsize"] = (12, 10)
        if self.stimulus_by == "location":
            shapes, stimulus_type_at_frame = Environment.get_env_shapes_from_pos()
            self.plot_attributes["yticks"] = [range(len(shapes)), shapes]

    def process_raw_data(self, save=True, stimulus_by="location"):
        """ "
        Returns:
            - data: Numpy array composed of stimulus type at frames.
        """
        stimulus_raw_data = self.raw_data_object.data  # e.g. Position on a track/time
        stimulus_by = self.stimulus_by or stimulus_by
        if stimulus_by == "location":
            if len(self.metadata["environment_dimensions"]) == 1:
                stimulus_type_at_frame = Environment.get_stimulus_at_position(
                    positions=stimulus_raw_data,
                    stimulus_dimensions=self.stimulus_dimensions,
                    stimulus_sequence=self.stimulus_sequence,
                    max_position=self.stimulus_dimensions,
                )
            elif len(self.metadata["environment_dimensions"]) == 2:
                shapes, stimulus_type_at_frame = Environment.get_env_shapes_from_pos(
                    stimulus_raw_data
                )
                self.plot_attributes["yticks"] = [range(len(shapes)), shapes]
        elif stimulus_by == "frames":
            stimulus_type_at_frame = self.stimulus_by_time(stimulus_raw_data)
        elif stimulus_by == "seconds":
            stimulus_type_at_frame = self.stimulus_by_time(
                stimulus_raw_data,
                time_to_frame_multiplier=self.self.metadata["imaging_fps"],
            )
        elif stimulus_by == "minutes":
            stimulus_type_at_frame = self.stimulus_by_time(
                stimulus_raw_data,
                time_to_frame_multiplier=60 * self.self.metadata["imaging_fps"],
            )
        self.data = stimulus_type_at_frame
        return self.data

    def stimulus_by_time(self, stimulus_raw_data, time_to_frame_multiplier=1):
        """
        time in frames in an experiment
        """
        stimulus_type_at_frame = []
        for stimulus, duration in zip(self.stimulus_sequence, self.stimulus_dimensions):
            stimulus_type_at_frame += [stimulus] * int(
                duration * time_to_frame_multiplier
            )
        raise NotImplementedError("Stimulus by time may not implemented. Check results")
        return np.array(stimulus_type_at_frame)


class Data_Distance(BehaviorDataset):
    def __init__(
        self, raw_data_object=None, metadata=None, root_dir=None, task_id=None
    ):
        super().__init__(
            key="distance",
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        self.metadata["environment_dimensions"] = make_list_ifnot(
            self.metadata["environment_dimensions"]
        )
        self.binning_size = (
            0.01
            if not "binning_size" in self.metadata.keys()
            else self.metadata["binning_size"]
        )

    def define_plot_attributes(self):
        self.plot_attributes["ylable"] = "distance in m"

    def plot_data(self):
        absolute_distance = np.linalg.norm(self.data, axis=1)
        Vizualizer.data_plot_1D(
            data=absolute_distance,
            plot_attributes=self.plot_attributes,
        )

    def bin_data(self, data, bin_size=None, return_category_map=False):
        bin_size = bin_size or self.binning_size
        binned_data = bin_array(data, bin_size=bin_size, min_bin=0)
        encoded_data = self.create_1d_discrete_labels(binned_data)
        if return_category_map:
            return encoded_data, self.category_map
        return encoded_data

    def process_raw_data(self, save=True):
        track_positions = self.raw_data_object.data
        self.data = Environment.get_cumdist_from_position(track_positions)
        return self.data


class Data_Velocity(BehaviorDataset):
    def __init__(
        self, raw_data_object=None, metadata=None, root_dir=None, task_id=None
    ):
        super().__init__(
            key="velocity",
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        self.raw_velocity = None
        self.binning_size = 0.005  # 0.005m/s

    def define_plot_attributes(self):
        self.plot_attributes["ylable"] = "velocity m/s"
        if self.metadata["environment_dimensions"] is not None:
            if len(self.metadata["environment_dimensions"]) == 2:
                self.plot_attributes["figsize"] = (10, 10)

    def bin_data(self, data, bin_size=None, return_category_map=False):
        bin_size = bin_size or self.binning_size
        binned_data = bin_array(data, bin_size=bin_size, min_bin=0)
        encoded_data = self.create_1d_discrete_labels(binned_data)
        if return_category_map:
            return encoded_data, self.category_map
        return encoded_data

    def process_raw_data(self, save=True, smooth=True):
        """
        calculating velocity based on velocity data in raw_data_object
        """
        raw_data_type = self.raw_data_object.key
        data = self.raw_data_object.data
        if raw_data_type == "distance":
            walked_distances = (
                self.raw_data_object.process_raw_data(save=save)
                if data is None
                else data
            )
        elif raw_data_type == "position":
            walked_distances = Environment.get_distance_from_position(data)
        else:
            raise ValueError(f"Raw data type {raw_data_type} not supported.")
        self.raw_velocity = Environment.get_velocity_from_cumdist(
            walked_distances, imaging_fps=self.metadata["imaging_fps"], smooth=False
        )
        if smooth:
            velocity_smoothed = may_butter_lowpass_filter(
                self.raw_velocity,
                smooth=True,
                cutoff=2,
                fps=self.metadata["imaging_fps"],
                order=2,
            )
            # velocity_smoothed = moving_average(self.raw_velocity)
            global_logger
            print(
                f"Calculating smooth velocity based on butter_lowpass_filter 2Hz, {self.metadata['imaging_fps']}fps, 2nd order."
            )
            # change value/frame to value/second
            self.data = velocity_smoothed
        else:
            self.data = self.raw_velocity
        return self.data


class Data_Acceleration(BehaviorDataset):
    def __init__(
        self, raw_data_object=None, metadata=None, root_dir=None, task_id=None
    ):
        super().__init__(
            key="acceleration",
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        self.raw_acceleration = None
        self.binning_size = 0.0005  # 0.0005m/s^2

    def define_plot_attributes(self):
        self.plot_attributes["ylable"] = "acceleration m/s^2"
        if self.metadata["environment_dimensions"] is not None:
            if len(self.metadata["environment_dimensions"]) == 2:
                self.plot_attributes["figsize"] = (10, 10)

    def bin_data(self, data, bin_size=None, return_category_map=False):
        bin_size = bin_size or self.binning_size
        binned_data = bin_array(data, bin_size=bin_size, min_bin=0)
        encoded_data = self.create_1d_discrete_labels(binned_data)
        if return_category_map:
            return encoded_data, self.category_map
        return encoded_data

    def process_raw_data(self, save=True):
        """
        calculating acceleration based on velocity data in raw_data_object
        """
        velocity = self.raw_data_object.data
        self.raw_acceleration = Environment.get_acceleration_from_velocity(
            velocity, smooth=False
        )
        smoothed_acceleration = may_butter_lowpass_filter(
            self.raw_acceleration,
            cutoff=2,
            fps=self.metadata["imaging_fps"],
            order=2,
        )
        global_logger
        print(
            f"Calculating smooth acceleration based on butter_lowpass_filter 2Hz, {self.metadata['imaging_fps']}fps, 2nd order."
        )
        self.data = smoothed_acceleration
        return self.data


class Data_Moving(BehaviorDataset):
    def __init__(
        self, raw_data_object=None, metadata=None, root_dir=None, task_id=None
    ):
        super().__init__(
            key="moving",
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        self.velocity_threshold = 0.02  # m/s
        self.brain_processing_delay = {
            "CA1": 2,  # seconds
            "CA3": 2,
            "M1": None,
            "S1": None,
            "V1": None,
        }

    def plot_data(self):
        Vizualizer.data_plot_1D(
            data=self.data,
            plot_attributes=self.plot_attributes,
        )

    def relative_data(self):
        global_logger.warning(
            f"Relative data is not implemented for {self.__class__}. Returning self.data"
        )
        return self.data

    def define_plot_attributes(self):
        self.plot_attributes["ylable"] = "Movement State"
        self.plot_attributes["ylimits"] = (-0.1, 1.1)
        self.plot_attributes["yticks"] = [[0, 1], ["Stationary", "Moving"]]

    def process_raw_data(self, save=True, fit_to_brainarea=True):
        velocities = self.raw_data_object.data
        if velocities is None:
            velocities = self.raw_data_object.process_raw_data(save=False)
        velocities_abs = np.linalg.norm(np.abs(velocities), axis=1)
        moving_frames = velocities_abs > self.velocity_threshold
        if fit_to_brainarea:
            self.data = self.fit_moving_to_brainarea(
                moving_frames, self.metadata["area"]
            )
        else:
            self.data = moving_frames
            global_logger.warning(
                f"Moving data not fitted to brain area {self.metadata['area']} lag"
            )
        return self.data

    def fit_moving_to_brainarea(self, data: np.ndarray, area: str):
        """
        Fit the movement data to the brain area reset time.

        1. Fill the gaps in the movement data based on the brain area reset time.

        Parameters:
            - data (np.ndarray): The movement data.
            - area (str): The brain area.
        """
        processing_movement_reset = self.brain_processing_delay[area]  # seconds
        if not processing_movement_reset:
            raise NotImplementedError(f"{area} reset time not provided.")

        # fill array with value if lag is biggern than gap
        processing_movement_frames = fill_continuous_array(
            data, fps=self.metadata["imaging_fps"], time_gap=processing_movement_reset
        )

        # No lag in the brain as far as we know
        """processing_movement_frames = add_stream_lag(processing_movement_frames,
                                                    min_stream_duration=self.brain_processing_delay[area],
                                                    fps=self.metadata["imaging_fps"],
                                                    lag=self.brain_processing_delay[area],)"""

        # add lag time to state before change
        for i, state in enumerate(processing_movement_frames):
            if state == 1:
                for j in range(1, processing_movement_reset):
                    if i + j < len(processing_movement_frames):
                        processing_movement_frames[i + j] = 1

        return processing_movement_frames

    def get_idx_to_keep(self, movement_state):
        if movement_state == "all":
            idx_to_keep = None
        elif movement_state == "moving":
            idx_to_keep = self.data
        elif movement_state == "stationary":
            idx_to_keep = self.data == False
        else:
            raise ValueError(f"Movement state {movement_state} not supported.")
        return idx_to_keep

    def correct_data(self, data):
        return self.fit_moving_to_brainarea(data, self.metadata["area"])


class Data_Photon(NeuralDataset):
    """
    Dataset class for managing photon data.
    Attributes:
        - data (np.ndarray): The binariced traces.
        - method (str): The method used to record the data.
        - preprocessing_software (str): The software used to preprocess the data.
            - raw fluoresence traces can be found inside preprocessing object
        - setup (str): The setup used to record the data.
        - setup.data = raw_data_object
    """

    def __init__(
        self,
        path=None,
        raw_data_object=None,
        metadata=None,
        root_dir=None,
        task_id=None,
    ):
        super().__init__(
            key="photon",
            path=path,
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )

    def define_plot_attributes(self):
        self.plot_attributes["ylable"] = "Neuron ID"
        self.plot_attributes["figsize"] = (20, 10)
        if "fps" not in self.metadata.keys():
            self.metadata["fps"] = self.setup.get_fps()
            self.plot_attributes["fps"] = self.metadata["fps"]

    def process_raw_data(self, save=True):
        self.data = self.setup.process_data(save=save)
        return self.data

    def plot_data(self):
        Vizualizer.plot_neural_activity_raster(
            self.data,
            fps=self.fps,
            num_ticks=self.plot_attributes["num_ticks"],
        )

    def get_setup(self, setup_name):
        if setup_name == "femtonics":
            setup = Femtonics(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        elif setup_name == "thorlabs":
            setup = Thorlabs(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        elif setup_name == "inscopix":
            setup = Inscopix(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        else:
            raise ValueError(f"Imaging setup {setup_name} not supported.")
        return setup


class Data_Probe(Dataset): #Maybe NeuralDataset
    def __init__(self, path=None, raw_data_object=None, metadata=None):
        super().__init__(
            key="probe", path=path, raw_data_object=raw_data_object, metadata=metadata
        )
        # TODO: implement probe data loading
        raise NotImplementedError("Probe data loading not implemented.")


class Datasets:
    """
    Dataset class for managing multiple datasets. Getting correct data paths and loading data.
    """

    def __init__(self, root_dir, metadata={}, task_id=None):
        if "data_dir" in metadata.keys():
            self.data_dir = Path(root_dir).joinpath(metadata["data_dir"])
        else:
            self.data_dir = Path(root_dir)
        self.metadata = metadata
        self.task_id = task_id

    def get_object(self, data_source):
        raise NotImplementedError(
            f"ERROR: Function get_object is not defined for {self.__class__}."
        )

    def load(self, data_source, regenerate=False, regenerate_plot=False):
        """
        Load data from a specific data source.
            data_object is a object inhereting the Dataset attributes and functions.
        """
        # build path to data
        data_object = self.get_object(data_source)
        fpath = data_object.setup.get_data_path()
        data = data_object.load(
            fpath, regenerate=regenerate, regenerate_plot=regenerate_plot
        )
        return data

    def get_multi_data(
        self,
        sources: List[str],
        shuffle: bool = False,
        idx_to_keep: Union[List, np.ndarray] = None,
        split_ratio: float = 1,
        transformation: str = None,
    ):
        """
        Extract data from multiple sources and concatenate them.

        Parameters:
            - sources (List[str]): The sources to extract data from.
            - shuffle (bool): Shuffle the concatenated data.
            - idx_to_keep (Union[List, np.ndarray]): Indices to keep.
            - split_ratio (float): Ratio to split the data into train and test.
            - transformation (str): Transformation to apply to the data.
                transformation types are: "binned", "relative", None.

        Returns:
            - concatenated_data_tain (np.ndarray): The concatenated training data.
            - concatenated_data_test (np.ndarray): The concatenated test data.
        """
        sources = make_list_ifnot(sources)
        concatenated_data = None
        for source in sources:
            global_logger.info(f"Loading data from {source}.")
            dataset_object = getattr(self, source)
            # data = dataset_object.data
            if transformation:
                if transformation == "binned":
                    data = dataset_object.binned_data
                elif transformation == "relative":
                    data = dataset_object.relative_data
                else:
                    global_logger.critical(
                        f"Transformation {transformation} not supported. Choose from: 'binned', 'relative'."
                    )
                    raise ValueError(
                        f"Transformation {transformation} not supported. Choose from: 'binned', 'relative'."
                    )
            else:
                data = dataset_object.data
            data = np.array([data]).transpose() if len(data.shape) == 1 else data
            if type(concatenated_data) != np.ndarray:
                concatenated_data = data
            else:
                concatenated_data, data = force_equal_dimensions(
                    concatenated_data, data
                )
                concatenated_data = np.concatenate([concatenated_data, data], axis=1)
        concatenated_data_filtered = Dataset.filter_by_idx(
            concatenated_data, idx_to_keep=idx_to_keep
        )
        concatenated_data_shuffled = (
            Dataset.shuffle(concatenated_data_filtered)
            if shuffle
            else concatenated_data_filtered
        )
        concatenated_data_tain, concatenated_data_test = Dataset.split(
            concatenated_data_shuffled, split_ratio
        )
        return concatenated_data_tain, concatenated_data_test


class Datasets_Neural(Datasets):
    def __init__(self, root_dir, metadata={}, task_id=None):
        super().__init__(root_dir=root_dir, metadata=metadata, task_id=task_id)
        self.photon_imaging_methods = ["femtonics", "thorlabs", "inscopix"]
        self.probe_imaging_methods = ["neuropixels", "tetrode"]
        # TODO: split into different datasets if needed
        self.imaging_type = self.define_imaging_type(self.metadata["setup"])
        if self.imaging_type == "photon":
            self.photon = Data_Photon(
                root_dir=root_dir, metadata=self.metadata, task_id=self.task_id
            )
        else:
            # TODO: implement probe data loading
            self.probe = Data_Probe(
                root_dir=root_dir, metadata=self.metadata, task_id=self.task_id
            )

        if "fps" not in self.metadata.keys():
            # TODO: this is not implemented for probe data
            data_object = self.get_object(self.metadata["setup"])  # photon or probe
            self.metadata["fps"] = data_object.metadata["fps"]

    def define_imaging_type(self, data_source):
        if data_source in self.photon_imaging_methods:
            imaging_type = "photon"
        elif data_source in self.probe_imaging_methods:
            imaging_type = "probe"
        else:
            raise ValueError(f"Imaging type {data_source} not supported.")
        return imaging_type

    def get_object(self, data_source:Union[None, str]=None) -> Union[Data_Photon, Data_Probe]:
        """
        Get the data object based on the data source. 
        
        WARNING: The implementation only handles on type of neural imaging method, so multiple data sources are not supported yet.

        Args:
            data_source (str, optional): One data source with a name from the list self.photon_imaging_methods or self.probe_imaging_methods
                to get the data object from. If None, the already defined imaging type is used.

        Returns:
            data_object (Data_Photon or Data_Probe): The data object based on the data source
        """
        imaging_type = self.imaging_type if data_source is None else data_source
        data_object = getattr(self, imaging_type)
        return data_object
    
    def get_process_data(self, data_source:Union[None, str]=None, type:str="processed") -> np.ndarray:
        """
        Get the data from process object based on the data source.
        
        WARNING: The implementation only handles on type of neural imaging method, so multiple data sources are not supported yet.
        
        Args:
            data_source (str, optional): One data source with a name from the list self.photon_imaging_methods or self.probe_imaging_methods
                to get the data object from. If None, the already defined imaging type is used.
                
        Returns:
            data (np.ndarray): The processed data
        """
        data_object = self.get_object(data_source)
        data = data_object.get_process_data(type=type)
        return data


class Datasets_Behavior(Datasets):
    def __init__(self, root_dir, metadata={}, task_id=None):
        super().__init__(root_dir=root_dir, metadata=metadata, task_id=task_id)
        self.data_dir = (
            self.data_dir
            if not self.data_dir == root_dir
            else self.data_dir.joinpath(f"TRD-{self.metadata['method']}")
        )
        self.position = Data_Position(
            root_dir=root_dir, metadata=self.metadata, task_id=self.task_id
        )
        self.distance = Data_Distance(
            raw_data_object=self.position,
            root_dir=root_dir,
            metadata=self.metadata,
            task_id=self.task_id,
        )
        self.stimulus = Data_Stimulus(
            raw_data_object=self.position,
            root_dir=root_dir,
            metadata=self.metadata,
            task_id=self.task_id,
        )
        self.velocity = Data_Velocity(
            raw_data_object=self.distance,
            root_dir=root_dir,
            metadata=self.metadata,
            task_id=self.task_id,
        )
        self.acceleration = Data_Acceleration(
            raw_data_object=self.velocity,
            root_dir=root_dir,
            metadata=self.metadata,
            task_id=self.task_id,
        )
        self.moving = Data_Moving(
            raw_data_object=self.velocity,
            root_dir=root_dir,
            metadata=self.metadata,
            task_id=self.task_id,
        )

    def split_by_laps(self, data=None, lap_starts=None, fps=None, sec_thr=5):
        """
        Splits data into individual laps based on lap start indices or calculates lap start indices using the provided parameters.

        Parameters:
            - data (optional): The data to be split into laps. This can be either 1D or 2D data.
            - lap_starts (optional): Indices indicating the start of each lap in the data.
            - fps (optional): Frames per second. Required if lap_starts is not provided and data is in video format.
            - sec_thr (optional): Threshold in seconds to consider as the minimum lap duration when calculating lap starts.

        Returns:
            - laps: A list containing each individual lap of the data.

        If lap_starts is not provided, lap start indices are calculated using the given fps and sec_thr parameters.
        If data is not provided, it will be expected to be accessed from the instance attribute 'position'.

        Note:
        - 1D data: If the data is 1-dimensional (e.g., time series data), lap_starts must be provided.
        - 2D data: If the data is 2-dimensional (e.g., time series data for multiple neurons).
        ```
        """
        lap_starts = lap_starts or self.position.get_lap_starts(
            fps=fps, sec_thr=sec_thr
        )
        laps = []
        for start, end in zip(lap_starts[:-1], lap_starts[1:]):
            laps.append(data[start:end])
        return laps

    def get_object(self, data_source):
        data_object = getattr(self, data_source)
        return data_object
