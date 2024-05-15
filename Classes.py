# import
from pathlib import Path

# setups and preprocessing software
from Setups import *
from Helper import *
from Visualizer import *
from Models import Models
from Datasets import Datasets_Neural, Datasets_Behavior, Dataset

# type hints
from typing import List, Union, Dict, Any, Tuple, Optional

# calculations
import numpy as np
import torch

# pip install binarize2pcalcium
# from binarize2pcalcium import binarize2pcalcium as binca

plt.style.use("dark_background")

# set seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def load_all_animals(
    root_dir,
    wanted_animal_ids=["all"],
    wanted_dates=["all"],
    model_settings=None,
    behavior_datas=None,
    regenerate_plots=False,
    **kwargs,
):
    """
    Loads animal data from the specified root directory for the given animal IDs.

    Parameters:
    - root_dir (string): The root directory path where the animal data is stored.
    - animal_ids (list, optional): A list of animal IDs to load. Default is ["all"].
    - generate (bool, optional): If True, generates new session data. Default is False.
    - regenerate (bool, optional): If True, regenerates existing session data. Default is False.
    - sessions (string, optional): Specifies the sessions. Default is "single".
    - delete (bool, optional): If True, deletes session data. Default is False.

    Returns:
    - animals_dict (dict): A dictionary containing animal IDs as keys and corresponding Animal objects as values.
    """
    root_dir = Path(root_dir)
    present_animal_ids = get_directories(root_dir, regex_search="DON-")
    animals_dict = {}
    if not model_settings:
        model_settings = kwargs

    # Search for animal_ids
    for animal_id in present_animal_ids:
        if animal_id in wanted_animal_ids or "all" in wanted_animal_ids:
            animal = Animal(
                animal_id=animal_id, root_dir=root_dir, model_settings=model_settings
            )
            animal.add_all_sessions(
                wanted_dates=wanted_dates,
                behavior_datas=behavior_datas,
                regenerate_plots=regenerate_plots,
            )
        animals_dict[animal_id] = animal
    animals_dict = sort_dict(animals_dict)
    return animals_dict


class Animal:
    def __init__(
        self, animal_id, root_dir, animal_dir=None, model_settings=None, **kwargs
    ):
        self.id = animal_id
        self.cohort_year = None
        self.dob = None
        self.sex = None
        self.root_dir = Path(root_dir)
        self.dir = animal_dir or self.root_dir.joinpath(animal_id)
        self.yaml_path = self.dir.joinpath(f"{animal_id}.yaml")
        self.sessions = {}
        self.model_settings = model_settings or kwargs
        success = self.load_metadata()
        if not success:
            return None

    def load_metadata(self, yaml_path=None, name_parts=None):
        yaml_path = yaml_path or self.yaml_path
        name_parts = name_parts or self.id
        needed_attributes = None
        success = check_correct_metadata(
            string_or_list=yaml_path, name_parts=name_parts
        )

        if success:
            animal_metadata_dict = load_yaml(yaml_path)
            # Load any additional metadata into session object
            set_attributes_check_presents(
                propertie_name_list=animal_metadata_dict.keys(),
                set_object=self,
                propertie_values=animal_metadata_dict.values(),
                needed_attributes=needed_attributes,
            )
        return success

    def add_session(
        self,
        date,
        model_settings=None,
        behavior_datas=None,
        regenerate_plots=False,
        **kwargs,
    ):
        if not model_settings:
            model_settings = kwargs if len(kwargs) > 0 else self.model_settings

        session = Session(
            self.id,
            date=date,
            animal_dir=self.dir,
            behavior_datas=behavior_datas,
            model_settings=model_settings,
            regenerate_plots=regenerate_plots,
        )

        if session:
            session.pday = (num_to_date(session.date) - num_to_date(self.dob)).days
            self.sessions[date] = session
            self.sessions = sort_dict(self.sessions)
            return self.sessions[date]
        else:
            global_logger
            print(f"Skipping {self.animal_id} {date}")
        return

    def add_all_sessions(
        self,
        wanted_dates=["all"],
        behavior_datas=None,
        model_settings=None,
        regenerate_plots=False,
        **kwargs,
    ):
        if not model_settings:
            model_settings = kwargs if len(kwargs) > 0 else self.model_settings
        # Search for Sessions
        sessions_root_path = self.root_dir.joinpath(self.animal_id)
        present_session_dates = get_directories(sessions_root_path)
        for date in present_session_dates:
            if date in wanted_dates or "all" in wanted_dates:
                self.add_session(
                    date,
                    model_settings=model_settings,
                    behavior_datas=behavior_datas,
                    regenerate_plots=regenerate_plots,
                )

    def plot_consistency_scores(
        self,
        wanted_stimulus_types,
        wanted_embeddings,
        exclude_properties=None,
        figsize=(7, 7),
    ):
        # TODO: change to only inter Session?
        viz = Vizualizer(root_dir=self.root_dir)
        labels = {}  # {wanted_embedding 1: {animal_session_task_id: embedding}, ...}
        for wanted_embedding in wanted_embeddings:
            labels[wanted_embedding] = {"embeddings": {}, "labels": {}}
            for wanted_stimulus_type in wanted_stimulus_types:
                for session_date, session in self.sessions.items():
                    for task_name, task in session.tasks.items():
                        if (
                            task.behavior_metadata["stimulus_type"]
                            == wanted_stimulus_type
                        ):
                            wanted_embeddings_dict = filter_dict_by_properties(
                                task.embeddings,
                                include_properties=wanted_embedding,
                                exclude_properties=exclude_properties,
                            )
                            for (
                                embedding_key,
                                embedding,
                            ) in wanted_embeddings_dict.items():
                                labels_id = f"{session_date[-3:]}_{task.task} {wanted_stimulus_type}"
                                position_lables = task.behavior.position.data
                                position_lables, embedding = (
                                    Datasets.force_equal_dimensions(
                                        position_lables, embedding
                                    )
                                )
                                labels[wanted_embedding]["embeddings"][
                                    labels_id
                                ] = embedding
                                labels[wanted_embedding]["labels"][
                                    labels_id
                                ] = position_lables

            dataset_ids = list(labels[wanted_embedding]["embeddings"].keys())
            embeddings = list(labels[wanted_embedding]["embeddings"].values())
            labeling = list(labels[wanted_embedding]["labels"].values())

            title = f"CEBRA-{wanted_embedding} embedding consistency"
            fig = plt.figure(figsize=figsize)
            ax1 = plt.subplot(111)
            ax1 = viz.plot_consistency_scores(
                ax1, title, embeddings, labeling, dataset_ids
            )
            plt.show()


class Session:
    def __init__(
        self,
        animal_id,
        date,
        animal_dir,
        session_dir=None,
        data_dir=None,
        model_dir=None,
        behavior_datas=None,
        regenerate_plots=False,
        model_settings={},
        **kwargs,
    ):
        self.animal_id = animal_id
        self.date = date
        self.id = f"{self.animal_id}_{self.date}"
        self.animal_dir = animal_dir
        self.dir = Path(session_dir or self.animal_dir.joinpath(date))
        self.data_dir = data_dir or self.dir
        self.model_dir = Path(model_dir or self.dir.joinpath("models"))
        self.model_settings = model_settings
        self.yaml_path = self.dir.joinpath(f"{self.date}.yaml")
        self.tasks_infos = None  # loaded from yaml
        self.tasks = {}
        success = self.load_metadata()
        if not success:
            return
        if behavior_datas:
            self.add_all_tasks(model_settings=self.model_settings, **kwargs)
            self.load_all_data(
                behavior_datas=behavior_datas, regenerate_plots=regenerate_plots
            )

    def load_metadata(self, yaml_path=None, name_parts=None):
        name_parts = name_parts or self.date
        yaml_path = yaml_path or self.yaml_path
        success = check_correct_metadata(
            string_or_list=yaml_path, name_parts=name_parts
        )
        if success:
            # Load any additional metadata into session object
            session_metadata_dict = load_yaml(yaml_path)
            needed_attributes = ["tasks_infos"]
            set_attributes_check_presents(
                propertie_name_list=session_metadata_dict.keys(),
                set_object=self,
                propertie_values=session_metadata_dict.values(),
                needed_attributes=needed_attributes,
            )
        return success

    def add_task(self, task, metadata=None, model_settings=None, **kwargs):
        success = check_correct_metadata(
            string_or_list=self.tasks_infos.keys(), name_parts=task
        )
        if success:
            task_object = Task(
                self.id,
                task,
                self.dir,
                self.data_dir,
                self.model_dir,
                metadata=metadata,
            )
            if not model_settings:
                model_settings = kwargs if len(kwargs) > 0 else self.model_settings
            task_object.init_models(model_settings)
            self.tasks[task] = task_object
        else:
            global_logger
            print(f"Skipping Task {task}")

    def add_all_tasks(self, model_settings=None, **kwargs):
        for task, metadata in self.tasks_infos.items():
            if not model_settings:
                model_settings = kwargs if len(kwargs) > 0 else self.model_settings
            self.add_task(task, metadata=metadata, model_settings=model_settings)
        return self.tasks[task]

    def load_all_data(self, behavior_datas=["position"], regenerate_plots=False):
        data = {}
        for task_name, task in self.tasks.items():
            data[task_name] = task.load_all_data(
                behavior_datas=behavior_datas, regenerate_plots=regenerate_plots
            )
        return data

    def plot_multiple_consistency_scores(
        self,
        animals,
        wanted_stimulus_types=["time", "behavior"],
        wanted_embeddings=["A", "B", "A'"],
        exclude_properties=None,
        figsize=(7, 7),
    ):
        # TODO: implement consistency session plot
        pass


class Task:
    def __init__(
        self,
        session_id,
        task,
        session_dir,
        data_dir=None,
        model_dir=None,
        metadata: dict = {},
    ):
        self.session_id = session_id
        self.id = f"{session_id}_{task}"
        self.name = task

        self.neural_metadata, self.behavior_metadata = self.load_metadata(metadata)
        self.data_dir = data_dir or session_dir
        self.neural = Datasets_Neural(
            root_dir=self.data_dir, metadata=self.neural_metadata, task_id=self.id
        )
        self.behavior_metadata = self.fit_behavior_metadata(self.neural)
        self.behavior = Datasets_Behavior(
            root_dir=self.data_dir, metadata=self.behavior_metadata, task_id=self.id
        )
        self.model_dir = model_dir or self.data_dir.joinpath("models")
        self.models = None
        self.embeddings = {}

    def load_metadata(self, metadata: dict = {}):
        needed_attributes = ["neural_metadata", "behavior_metadata"]
        set_attributes_check_presents(
            propertie_name_list=metadata.keys(),
            set_object=self,
            propertie_values=metadata.values(),
            needed_attributes=needed_attributes,
        )
        self.neural_metadata["task_id"] = self.id
        self.behavior_metadata["task_id"] = self.id
        return self.neural_metadata, self.behavior_metadata

    def fit_behavior_metadata(self, neural):
        self.behavior_metadata["method"] = neural.metadata["method"]
        self.behavior_metadata["imaging_fps"] = neural.metadata["fps"]
        if "area" in neural.metadata.keys():
            self.behavior_metadata["area"] = neural.metadata["area"]
        return self.behavior_metadata

    def load_data(self, data_source, data_type="neural", regenerate_plot=False):
        # loads neural or behaviour data
        datasets_object = getattr(self, data_type)
        data = datasets_object.load(
            self.id, data_source=data_source, regenerate_plot=regenerate_plot
        )
        return data

    def load_all_data(self, behavior_datas=["position"], regenerate_plots=False):
        """
        neural_Data = ["femtonics", "inscopix"]
        behavior_datas = ["position", "cam"]
        """
        data = {"neural": {}, "behavior": {}}
        for mouse_data in ["neural", "behavior"]:
            if mouse_data == "neural":
                data_to_load = self.neural_metadata["setup"]
            else:
                data_to_load = behavior_datas
            data_to_load = make_list_ifnot(data_to_load)
            for data_source in data_to_load:
                data[mouse_data][data_source] = self.load_data(
                    data_source=data_source,
                    data_type=mouse_data,
                    regenerate_plot=regenerate_plots,
                )
        return data

    def init_models(self, model_settings=None, **kwargs):
        if not model_settings:
            model_settings = kwargs
        self.models = Models(
            self.model_dir, model_id=self.name, model_settings=model_settings
        )

    # Cebra
    def get_multi_data(
        self,
        datasets_object: Datasets,
        data_types,
        data=None,
        movement_state="all",
        shuffled=False,
        split_ratio=1,
    ):
        idx_to_keep = self.behavior.moving.get_idx_to_keep(movement_state)

        # get data
        if not isinstance(data, np.ndarray):
            data, _ = datasets_object.get_multi_data(
                data_types,
                idx_to_keep=idx_to_keep,
                # shuffle=shuffled,
                # split_ratio=split_ratio
            )
        data_filtered = Dataset.filter_by_idx(data, idx_to_keep=idx_to_keep)
        data_shuffled = Dataset.shuffle(data_filtered) if shuffled else data_filtered
        data_train, data_test = Dataset.split(data_shuffled, split_ratio)
        data_train = Dataset.force_2d(data_train)
        # data_train = force_1_dim_larger(data_train)
        return data_train, data_test

    def set_model_name(
        self,
        model_type,
        model_name=None,
        shuffled=False,
        movement_state="all",
        split_ratio=1,
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
        return model_name

    def get_model(self, models_class, model_name, model_type, model_settings=None):
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

    def train_model(
        self,
        model_type: str,  # types: time, behavior, hybrid
        regenerate: bool = False,
        shuffled: bool = False,
        movement_state: str = "all",
        split_ratio: float = 1,
        model_name: str = None,
        neural_data: np.ndarray = None,
        behavior_data: np.ndarray = None,
        neural_data_types: List[str] = None,  # ["femtonics"],
        behavior_data_types: List[str] = None,  # ["position"],
        manifolds_pipeline: str = "cebra",
        model_settings: dict = None,
    ):
        """
        available model_types are: time, behavior, hybrid
        """
        model_name = self.set_model_name(
            model_type, model_name, shuffled, movement_state, split_ratio
        )

        # TODO: add other manifolds pipelines
        if manifolds_pipeline == "cebra":
            models_class = self.models.cebras
        else:
            raise ValueError(f"Manifolds pipeline {manifolds_pipeline} not supported.")

        model = self.get_model(
            models_class=models_class,
            model_name=model_name,
            model_type=model_type,
            model_settings=model_settings,
        )

        # get neural data
        neural_data_types = (
            neural_data_types or self.neural_metadata["preprocessing_software"]
        )
        neural_data_train, neural_data_test = self.get_multi_data(
            datasets_object=self.neural,
            data_types=self.neural.imaging_type,
            data=neural_data,
            movement_state=movement_state,
            shuffled=shuffled,
            split_ratio=split_ratio,
        )

        # get behavior data
        if behavior_data_types:
            behavior_data_train, behavior_data_test = self.get_multi_data(
                datasets_object=self.behavior,
                data_types=behavior_data_types,  # e.g. ["position", "cam"]
                data=behavior_data,
                movement_state=movement_state,
                shuffled=shuffled,
                split_ratio=split_ratio,
            )

        if not model.fitted or regenerate:
            # skip if no neural data available
            if neural_data_train.shape[0] == 0:
                global_logger.error(f"No neural data to use. Skipping {self.id}")
                print(f"Skipping {self.id}: No neural data given.")
            else:
                # train model
                global_logger.info(f"{self.id}: Training  {model.name} model.")
                print(f"{self.id}: Training  {model.name} model.")
                if model_type == "time":
                    model.fit(neural_data_train)
                else:
                    if not behavior_data_types:
                        raise ValueError(
                            f"No behavior data types given for {model_type} model."
                        )
                    neural_data_train, behavior_data_train = (
                        Datasets.force_equal_dimensions(
                            neural_data_train, behavior_data_train
                        )
                    )
                    model.fit(neural_data_train, behavior_data_train)
                model.fitted = models_class.is_fitted(model)
                model.save(model.save_path)
        else:

            global_logger.info(
                f"{self.id}: {model.name} model already trained. Skipping."
            )
            print(f"{self.id}: {model.name} model already trained. Skipping.")

        # TODO: add saving test dataset somewhere and maybe provide decoded results or decode directly.....

        if split_ratio != 1:
            return (
                model,
                neural_data_train,
                neural_data_test,
                behavior_data_train,
                behavior_data_test,
            )
        return model

    def create_embeddings(
        self,
        models=None,
        to_transform_data=None,
        model_naming_filter_include: List[List[str]] = None,  # or [str] or str
        model_naming_filter_exclude: List[List[str]] = None,  # or [str] or str
        manifolds_pipeline="cebra",
    ):
        # TODO: create function to integrate train and test embeddings into models
        if not type(to_transform_data) == np.ndarray:
            global_logger.warning(f"No data to transform given. Using default labels.")
            print(f"No neural data given. Using default labels. photon")
            to_transform_data = self.neural.photon.data
            if not isinstance(to_transform_data, np.ndarray):
                raise ValueError("No data to neural transform given.")

        filtered_models = models or self.get_pipeline_models(
            manifolds_pipeline, model_naming_filter_include, model_naming_filter_exclude
        )
        embeddings = {}
        for model_name, model in filtered_models.items():
            embedding_title = f"{model_name} - {model.max_iterations}"
            if model.fitted:
                embedding = model.transform(to_transform_data)
                self.embeddings[embedding_title] = embedding
                embeddings[embedding_title] = embedding
            else:
                global_logger.error(f"{model_name} model. Not fitted.")
                global_logger.warning(
                    f"Skipping {model_name} model. May because of no statioary frames"
                )
                print(
                    f"Skipping {model_name} model. May because of no statioary frames"
                )
        return embeddings

    def plot_model_embeddings(
        self,
        model_naming_filter_include: List[List[str]] = None,  # or [str] or str
        model_naming_filter_exclude: List[List[str]] = None,  # or [str] or str
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        to_transform_data: Optional[np.ndarray] = None,
        embedding_labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        behavior_data_types: List[str] = ["position"],
        manifolds_pipeline: str = "cebra",
        set_title: Optional[str] = None,
        title_comment: Optional[str] = None,
        markersize: float = 0.05,
        alpha: float = 0.4,
        dpi: int = 300,
    ):
        if not embeddings:
            embeddings = self.create_embeddings(
                to_transform_data=to_transform_data,
                model_naming_filter_include=model_naming_filter_include,
                model_naming_filter_exclude=model_naming_filter_exclude,
                manifolds_pipeline="cebra",
            )

        # get embedding lables
        if not isinstance(embedding_labels, np.ndarray) and not isinstance(
            embedding_labels, dict
        ):
            global_logger.error("No embedding labels given.")
            global_logger.warning(f"Using behavior_data_types: {behavior_data_types}")
            print(f"Using behavior_data_types: {behavior_data_types}")
            behavior_datas, _ = self.behavior.get_multi_data(behavior_data_types)
            embedding_labels
            embedding_labels_dict = {}
            for behavior_data_type, behavior_data in zip(
                behavior_data_types, behavior_datas.transpose()
            ):
                embedding_labels_dict[behavior_data_type] = behavior_data
        else:
            if isinstance(embedding_labels, np.ndarray):
                embedding_labels_dict = {"Provided_labels": embedding_labels}
            else:
                embedding_labels_dict = embedding_labels

        viz = Vizualizer(self.data_dir.parent.parent)

        for embedding_title, embedding_labels in embedding_labels_dict.items():
            if set_title:
                title = set_title
            else:
                title = f"{manifolds_pipeline.upper()} embeddings {self.id}"
                descriptive_metadata_keys = [
                    "stimulus_type",
                    "method",
                    "processing_software",
                ]
                title += (
                    get_str_from_dict(
                        dictionary=self.behavior_metadata,
                        keys=descriptive_metadata_keys,
                    )
                    + f" - {embedding_title}{' '+str(title_comment) if title_comment else ''}"
                )
            labels_dict = {"name": embedding_title, "labels": embedding_labels}
            viz.plot_multiple_embeddings(
                embeddings,
                labels=labels_dict,
                title=title,
                markersize=markersize,
                alpha=alpha,
                dpi=dpi,
            )
        return embeddings

    def get_pipeline_models(
        self,
        manifolds_pipeline="cebra",
        model_naming_filter_include: List[List[str]] = None,  # or [str] or str
        model_naming_filter_exclude: List[List[str]] = None,  # or [str] or str
    ):
        if manifolds_pipeline == "cebra":
            models = self.models.cebras.get_models(
                model_naming_filter_include=model_naming_filter_include,
                model_naming_filter_exclude=model_naming_filter_exclude,
            )
        else:
            raise ValueError(
                f"Manifolds Pipeline {manifolds_pipeline} is not implemented. Use 'cebra'."
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

    def plot_model_losses(
        self,
        models=None,
        title=None,
        manifolds_pipeline="cebra",
        coloring_type="rainbow",
        plot_original=True,
        plot_shuffled=True,
        num_iterations=None,
        plot_model_iterations=False,
        model_naming_filter_include: List[List[str]] = None,  # or [str] or str
        model_naming_filter_exclude: List[List[str]] = None,  # or [str] or str
        alpha=0.8,
        figsize=(10, 10),
    ):
        models_original, models_shuffled = self.get_models_splitted_original_shuffled(
            models=models,
            manifolds_pipeline=manifolds_pipeline,
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
        )
        stimulus_type = self.behavior_metadata["stimulus_type"]
        num_iterations = (
            models_original[0].max_iterations if not num_iterations else num_iterations
        )
        title = title or f"Losses {self.session_id} {stimulus_type}"
        title += f" - {num_iterations} Iterartions" if not plot_model_iterations else ""

        viz = Vizualizer(self.data_dir.parent.parent)
        viz.plot_losses(
            models=models_original,
            models_shuffled=models_shuffled,
            title=title,
            coloring_type=coloring_type,
            plot_original=plot_original,
            plot_shuffled=plot_shuffled,
            alpha=alpha,
            figsize=figsize,
            plot_model_iterations=plot_model_iterations,
        )

    def plot_multiple_consistency_scores(
        self,
        animals,
        wanted_stimulus_types,
        wanted_embeddings,
        exclude_properties=None,
        figsize=(7, 7),
    ):
        # TODO: implement consistency task plot
        pass

    # Place Cells
    # TODO: implement 1D or 2D differentiation
    def get_rate_time_map(
        self,
        movement_state="moving",
        smooth=True,
        window_size=2,
    ):
        """
        Plots the rate maps of the place cells.
        """
        idx_to_keep = self.behavior.moving.get_idx_to_keep(movement_state)
        activity = Dataset.filter_by_idx(
            self.neural.photon.data, idx_to_keep=idx_to_keep
        )
        binned_pos = Dataset.filter_by_idx(
            self.behavior.position.binned_data, idx_to_keep=idx_to_keep
        )
        rate_map, time_map = self.models.place_cell.get_maps(
            activity=activity,
            binned_pos=binned_pos,
            smooth=smooth,
            window_size=window_size,
        )
        return rate_map, time_map, activity, binned_pos

    def plot_rate_map(
        self,
        rate_map=None,
        time_map=None,
        movement_state="moving",
        norm_rate=True,
        smooth=True,
        window_size=2,
        sorting_indices=None,
    ):
        """
        Plots the rate maps of the place cells.
        """
        if rate_map is None or time_map is None:
            rm, tm, _, _ = self.get_rate_time_map(
                movement_state=movement_state,
                smooth=smooth,
                window_size=window_size,
            )
        rate_map = rate_map or rm
        time_map = time_map or tm

        only_moving = True if movement_state == "moving" else False
        additional_title = f"{self.id} Belt: {self.behavior_metadata['stimulus_type']}"
        if only_moving:
            additional_title += " (Moving only)"

        sorted_rate_map, sorting_indices = Vizualizer.plot_cell_activites_heatmap(
            rate_map,
            additional_title=additional_title,
            norm_rate=norm_rate,
            sorting_indices=sorting_indices,
        )
        return rate_map, time_map

    def plot_rate_map_per_cell(
        self,
        rate_map=None,
        time_map=None,
        provided_zscore=None,
        movement_state="moving",
        norm=True,
        smooth=True,
        window_size=2,
        sorting_indices=None,
        n_tests=1000,
        sort_by="peak",
        top_n=10,
        zscore_threshold=2.5,
    ):
        """
        parameter:
        - norm: bool
            If True, normalizes the rate map.
        - smooth: bool
            If True, smooths the rate map if no rate_map is given
        - sort_by: str
            "peak", "spatial_information" or "zscore"
        - top_n: int or str
            If int, plots the top n rate maps.
            If "all", plots all rate maps.
            if "significant", plots all significant rate maps.
        """
        only_moving = True if movement_state == "moving" else False
        additional_title = f"{self.id} Belt: {self.behavior_metadata['stimulus_type']} sorted by {sort_by if top_n != 'significant' else 'zscore'}"
        if only_moving:
            additional_title += " (Moving only)"

        if rate_map is None or time_map is None:
            rm, tm, activity, binned_pos = self.get_rate_time_map(
                movement_state=movement_state,
                smooth=smooth,
                window_size=window_size,
            )
        fps = self.behavior_metadata["imaging_fps"]
        rate_map = rate_map or rm
        time_map = time_map or tm

        if sort_by == "peak":
            sorting_indices = np.argsort(np.argmax(rate_map, axis=1))
            zscore = None
        elif sort_by in ["spatial_information", "spatial_content", "zscore"]:
            if sort_by in ["spatial_information", "spatial_content"]:
                n_test = (
                    n_tests if top_n == "significant" and provided_zscore is None else 1
                )
                zscore, si_rate, si_content = (
                    self.models.place_cell.si_model.compute_si_zscores(
                        # rate_map, time_map, n_tests=n_test, fps=fps
                        activity=activity,
                        binned_pos=binned_pos,
                        n_tests=n_test,
                        fps=fps,
                        max_pos=self.behavior_metadata["environment_dimensions"],
                    )
                )
            elif sort_by == "zscore":
                n_test = n_tests if provided_zscore is None else 1
                zscore, si_rate, si_content = (
                    self.models.place_cell.si_model.compute_si_zscores(
                        # rate_map, time_map, n_tests=n_test, fps=fps
                        activity=activity,
                        binned_pos=binned_pos,
                        n_tests=n_test,
                        fps=fps,
                        max_pos=self.behavior_metadata["environment_dimensions"],
                    )
                )
            zscore = zscore if provided_zscore is None else provided_zscore
        else:
            raise ValueError(f"Sorting of the rate_map by {sort_by} not supported.")

        # define labels
        if sort_by == "peak":
            labels = sorting_indices
        elif sort_by == "spatial_information":
            labels = si_rate
        elif sort_by == "spatial_content":
            labels = si_content
        elif sort_by == "zscore":
            labels = zscore

        sorting_indices = np.argsort(labels)[::-1]
        num_nan = np.sum(np.isnan(labels)) if top_n != "significant" else 0

        # get top n rate maps
        if top_n == "all":
            top_n = len(sorting_indices) - num_nan
        elif top_n == "significant" and sort_by != "peak":
            # sorted_zscores = zscore[np.argsort(zscore)[::-1]]
            significant_zscore_indices = np.where(zscore >= zscore_threshold)[0]
            significant_zscores = zscore[significant_zscore_indices]
            zscore_sorting_indices = np.argsort(significant_zscores)[::-1]
            sorted_significant_zscore_indices = significant_zscore_indices[
                zscore_sorting_indices
            ]
            sorting_indices = sorted_significant_zscore_indices
            num_significant = sorting_indices.shape[0]
            labels = np.array(
                [
                    f"si: {spatial_info:.1f}  sc: {spatial_cont:.1f}  (zscore: {z_val:.1f})"
                    for spatial_info, spatial_cont, z_val in zip(
                        si_rate, si_content, zscore
                    )
                ]
            )
            top_n = num_significant
        elif isinstance(top_n, int):
            top_n = min(top_n, len(sorting_indices) - num_nan)
        else:
            raise ValueError("top_n must be an integer, 'all' or 'significant'.")

        to_plot_rate_map = rate_map[sorting_indices][num_nan : num_nan + top_n]
        to_plot_labels = labels[sorting_indices][num_nan : num_nan + top_n]

        if sort_by != "zscore":
            to_plot_labels_list = []
            for cell_id, label in zip(sorting_indices, to_plot_labels):
                label = label if isinstance(label, str) else f"{label:.1f}"
                to_plot_labels_list.append(f"Cell {cell_id:>4}: {label}")

        Vizualizer.plot_traces_shifted(
            to_plot_rate_map,
            labels=to_plot_labels,
            additional_title=additional_title,
            norm=norm,
        )
        outputs = {
            "rate_map": rate_map[sorting_indices],
            "time_map": time_map,
            "zscore": zscore[sorting_indices],
        }
        return outputs

    def plot_cell_activity_pos_by_time(self, cell_ids, sec_thr=5):  # int or [int]
        """
        Plots the cell activity by position and time.
        """

        # get data
        cell_activity = self.neural.photon.data[:, cell_ids]
        binned_pos = self.behavior.position.binned_data
        
        # split data by laps
        cell_activity_by_lap = self.behavior.split_by_laps(cell_activity)
        binned_pos_by_lap = self.behavior.split_by_laps(binned_pos)

        # count spikes at position
        ...... use spike counting from models.....................
        max_pos = self.behavior_metadata["environment_dimensions"]
        lap_activity = np.zeros((len(cell_ids), pos.shape[1]))
        Vizualizer.plot_cell_activity_by_lap(cell_activity)

    def plot_place_cell_si_scores(
        self,
        movement_state="moving",
        smooth=True,
        window_size=2,
        n_tests=1000,
        colors=["red", "tab:blue"],
        method="skaggs",
    ):
        """
        Plots the spatial information scores of the place cells.
        """
        rate_map, time_map, activity, binned_pos = self.get_rate_time_map(
            movement_state=movement_state,
            smooth=smooth,
            window_size=window_size,
        )

        fps = self.behavior_metadata["imaging_fps"]

        zscore, si_rate, si_content = (
            self.models.place_cell.si_model.compute_si_zscores(
                activity=activity,
                binned_pos=binned_pos,
                # rate_map=rate_map,
                # time_map=time_map,
                n_tests=n_tests,
                spatial_information_method=method,
                fps=fps,
                max_pos=self.behavior_metadata["environment_dimensions"],
            )
        )
        additional_title = f"{self.id} Belt: {self.behavior_metadata['stimulus_type']}"
        additional_title += f" ({method})"

        Vizualizer.plot_zscore(
            zscore,
            additional_title=additional_title,
            color=colors,
        )
        Vizualizer.plot_si_rates(
            si_rate,
            zscores=zscore,
            additional_title=additional_title,
            color=colors,
        )

        return zscore, si_rate, si_content
