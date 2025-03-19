# type hints
from __future__ import annotations
from typing import List, Union, Dict, Any, Tuple, Optional
from tqdm import tqdm, trange

# paths
from pathlib import Path

# setups and preprocessing software
from Setups import *
from Helper import *
from Visualizer import *
from Models import Models, PlaceCellDetectors, decode
from Datasets import Datasets_Neural, Datasets_Behavior, Dataset

# calculations
import numpy as np
import torch

# pip install binarize2pcalcium
# from binarize2pcalcium import binarize2pcalcium as binca

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
    regenerate=False,
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
    present_animal_ids = get_directories(root_dir, regex_search="DON-[A-Za-z0-9-_]*")
    animals_dict = {}
    if not model_settings:
        model_settings = kwargs

    # Search for animal_ids
    for animal_id in present_animal_ids:
        if animal_id in wanted_animal_ids or "all" in wanted_animal_ids:
            animal = Animal(
                animal_id=animal_id, root_dir=root_dir, model_settings=model_settings
            )
            animal.add_sessions(
                wanted_dates=wanted_dates,
                behavior_datas=behavior_datas,
                regenerate=regenerate,
                regenerate_plots=regenerate_plots,
            )
        animals_dict[animal_id] = animal
    animals_dict = sort_dict(animals_dict)
    return animals_dict


class Multi:
    def __init__(
        self,
        animals_dict,
        name=None,
        wanted_properties=None,
        model_settings=None,
        **kwargs,
    ):
        self.wanted_properties: Optional[Dict[str, Any]] = wanted_properties
        self.animals: Dict[str, Animal] = animals_dict
        self.filtered_tasks: Dict[str, Task] = (
            self.animals if not wanted_properties else self.filter()
        )
        self.model_settings: Dict[str, Any] = model_settings or kwargs
        self.name: str = self.define_name(name)
        self.id: str = self.define_id(self.name)
        self.model_dir: Path = self.animals[
            list(self.animals.keys())[0]
        ].dir.parent.joinpath("models")
        self.models: Models = self.init_models(model_settings=model_settings)

    def define_name(self, name):
        name = name if name else "UNDEFINED_NAME"
        name = f"multi_set_{name}"
        return name

    def define_id(self, name):
        id = f"{name}_{self.model_settings}_{self.wanted_properties}"
        remove_ilegal_chars = [" ", "[", "]", "{", "}", ":", ",", "'", "\\", "/"]
        for char in remove_ilegal_chars:
            id = id.replace(char, "")
        return id

    def filter(self, wanted_properties: Dict[str, Dict[str, Union[str, List]]] = None):
        """
        Filters the tasks based on the wanted properties.

        Parameters
        ----------
        wanted_properties : dict, optional
            A dictionary containing the properties to filter by. The dictionary should have the following structure:
            Example:
                wanted_properties = {  # "animal": {
                    "animal_id": ["DON-007021"],
                    "sex": "male",
                    },
                    "session": {
                        # "date": ["20211022", "20211030", "20211031"],
                    },
                    "task": {
                        "name": ["FS1"],
                    },
                    "neural_metadata": {
                        "area": "CA3",
                        "method": "1P",
                    },
                    "behavior_metadata": {
                        "setup": "openfield",
                    },
                }

        Returns
        -------
        filtered_tasks : dict
            dict containing the filtered tasks based on the wanted properties with the task id as key and the task object as value.
        """
        if not wanted_properties:
            if self.wanted_properties:
                wanted_properties = self.wanted_properties
            else:
                print("No wanted properties given. Returning all tasks")
                wanted_properties = {}
        else:
            if self.wanted_properties == wanted_properties:
                return self.filtered_tasks
            else:
                self.wanted_properties = wanted_properties

        filtered_tasks = {}
        for animal_id, animal in self.animals.items():
            wanted = True
            if "animal" in wanted_properties:
                wanted = wanted_object(animal, wanted_properties["animal"])

            if wanted:
                filtered_animal_tasks = animal.filter_sessions(wanted_properties)
                filtered_tasks.update(filtered_animal_tasks)
        self.filtered_tasks = filtered_tasks
        return self.filtered_tasks

    def init_models(self, model_settings=None, **kwargs):
        if not model_settings:
            model_settings = kwargs
        models = Models(
            self.model_dir, model_id=self.name, model_settings=model_settings
        )
        return models

    def train_model(
        self,
        model_type: str,  # types: time, behavior, hybrid
        tasks=None,
        wanted_properties=None,
        regenerate: bool = False,
        shuffle: bool = False,
        movement_state: str = "all",
        split_ratio: float = 1,
        model_name: str = None,
        neural_data: np.ndarray = None,
        behavior_data: np.ndarray = None,
        binned: bool = True,
        relative: bool = False,
        neural_data_types: List[str] = None,  #
        behavior_data_types: List[str] = None,  # ["position"],
        manifolds_pipeline: str = "cebra",
        model_settings: dict = None,
        create_embeddings: bool = True,
    ):
        if not tasks:
            tasks = self.filter(wanted_properties)

        datas = []
        labels = []
        for task_id, task in tasks.items():
            # get neural data
            idx_to_keep = task.behavior.moving.get_idx_to_keep(movement_state)
            neural_data, _ = task.neural.get_multi_data(
                sources=task.neural.imaging_type,
                idx_to_keep=idx_to_keep,
                binned=binned,
            )

            # get behavior data
            if behavior_data_types:
                behavior_data, _ = task.behavior.get_multi_data(
                    sources=behavior_data_types,  # e.g. ["position", "stimulus"]
                    idx_to_keep=idx_to_keep,
                    binned=binned,
                )

            datas.append(neural_data)
            labels.append(behavior_data)

        print(self.id)
        multi_model = self.models.train_model(
            neural_data=datas,
            behavior_data=labels,
            model_type=model_type,
            model_name=model_name,
            movement_state=movement_state,
            shuffle=shuffle,
            binned=binned,
            relative=relative,
            split_ratio=split_ratio,
            model_settings=model_settings,
            pipeline=manifolds_pipeline,
            create_embeddings=create_embeddings,
            regenerate=regenerate,
        )

        return multi_model

    def plot_embeddings(
        self,
        model_naming_filter_include: Union[str, List[str], List[List[str]]] = None,
        model_naming_filter_exclude: Union[str, List[str], List[List[str]]] = None,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        to_2d: bool = False,
        show_hulls: bool = False,
        to_transform_data: Optional[np.ndarray] = None,
        colorbar_ticks: Optional[List] = None,
        labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        behavior_data_types: List[str] = ["position"],
        manifolds_pipeline: str = "cebra",
        title: Optional[str] = None,
        title_comment: Optional[str] = None,
        markersize: float = None,
        alpha: float = None,
        figsize: Tuple[int, int] = None,
        dpi: int = 300,
        as_pdf: bool = False,
    ):
        # FIXME: This function is outdated
        # FIXME: merge this function with tasks plot_embeddings

        # models = self.get_pipeline_models(manifolds_pipeline, model_naming_filter_include, model_naming_filter_exclude)
        if embeddings is not None and to_transform_data is not None:
            global_logger.error(
                "Either provide embeddings or to_transform_data, not both."
            )
            raise ValueError(
                "Either provide embeddings or to_transform_data, not both."
            )
        if to_transform_data is not None:
            embeddings = self.models.create_embeddings(
                to_transform_data=to_transform_data,
                to_2d=to_2d,
                model_naming_filter_include=model_naming_filter_include,
                model_naming_filter_exclude=model_naming_filter_exclude,
                manifolds_pipeline="cebra",
            )

        if not embeddings:
            model_class, models = self.models.get_pipeline_models(
                manifolds_pipeline=manifolds_pipeline,
                model_naming_filter_include=model_naming_filter_include,
                model_naming_filter_exclude=model_naming_filter_exclude,
            )
            embeddings = {}
            labels_dict = {}
            for model_name, model in models.items():
                embeddings[model_name] = model.data["train"]["embedding"]
                labels_dict[model_name] = model.data["train"]["behavior"]

        # get embedding lables
        if not isinstance(labels, np.ndarray) and not isinstance(labels, dict):
            behavior_label = []
            labels_dict = {}
            all_embeddings = {}
            behavior_data_types = make_list_ifnot(behavior_data_types)

            # create labels for all behavior data types
            for behavior_data_type in behavior_data_types:
                models_embeddings = print("Dont know how this should look like...")
                for embedding_model_name, embeddings in models_embeddings.items():
                    global_logger.warning(
                        f"Using behavior_data_types: {behavior_data_types}"
                    )
                    print(f"Using behavior_data_types: {behavior_data_types}")

                    # extract behavior labels from corresponding task
                    for (task_id, task), (embedding_title, embedding) in zip(
                        self.filtered_tasks.items(), embeddings.items()
                    ):
                        task_behavior_labels_dict = task.get_behavior_labels(
                            [behavior_data_type], idx_to_keep=None
                        )
                        if not equal_number_entries(
                            embedding, task_behavior_labels_dict
                        ):
                            task_behavior_labels_dict = task.get_behavior_labels(
                                [behavior_data_type], movement_state="moving"
                            )
                            if not equal_number_entries(
                                embedding, task_behavior_labels_dict
                            ):
                                task_behavior_labels_dict = task.get_behavior_labels(
                                    [behavior_data_type], movement_state="stationary"
                                )
                                if not equal_number_entries(
                                    embedding, task_behavior_labels_dict
                                ):
                                    raise ValueError(
                                        f"Number of labels is not equal to all, moving or stationary number of frames."
                                    )
                        behavior_label.append(
                            task_behavior_labels_dict[behavior_data_type]
                        )
                    all_embeddings.update(embeddings)
                labels_dict[behavior_data_type] = behavior_label
        else:
            if isinstance(labels, np.ndarray):
                labels_dict = {"Provided_labels": labels}
            else:
                labels_dict = labels

        # get ticks
        if len(behavior_data_types) == 1 and colorbar_ticks is None:
            dataset_object = getattr(task.behavior, behavior_data_types[0])
            # TODO: ticks are not always equal for all tasks, so this is not a good solution
            colorbar_ticks = dataset_object.plot_attributes["yticks"]

        viz = Vizualizer(self.model_dir.parent)
        self.id = self.define_id(self.name)
        for embedding_title, labels in labels_dict.items():
            if title:
                title = title
            else:
                title = f"{manifolds_pipeline.upper()} embeddings {self.id}"
                descriptive_metadata_keys = [
                    "stimulus_type",
                    "method",
                    "processing_software",
                ]
                title += (
                    get_str_from_dict(
                        dictionary=task.behavior_metadata,
                        keys=descriptive_metadata_keys,
                    )
                    + f"{' '+str(title_comment) if title_comment else ''}"
                )
            projection = "2d" if to_2d else "3d"
            labels_dict = {"name": embedding_title, "labels": labels}
            viz.plot_multiple_embeddings(
                all_embeddings,
                labels=labels_dict,
                ticks=colorbar_ticks,
                title=title,
                projection=projection,
                show_hulls=show_hulls,
                markersize=markersize,
                figsize=figsize,
                alpha=alpha,
                dpi=dpi,
                as_pdf=as_pdf,
            )
        return embeddings


class Animal:
    """Represents an animal in the dataset."""

    descriptive_metadata_keys = []
    needed_attributes: List[str] = ["animal_id", "dob"]

    def __init__(
        self, animal_id, root_dir, animal_dir=None, model_settings=None, **kwargs
    ):
        self.id: str = animal_id
        self.dob: str = None
        self.sex: str = None
        self.root_dir: Path = Path(root_dir)
        self.dir: Path = animal_dir or self.root_dir.joinpath(animal_id)
        self.yaml_path: Path = self.dir.joinpath(f"{animal_id}.yaml")
        self.sessions: Dict[str, Session] = {}
        self.model_settings: Dict[str, Any] = model_settings or kwargs
        self.load_metadata()

    def load_metadata(self, yaml_path=None, name_parts=None):
        load_yaml_data_into_class(
            cls=self,
            yaml_path=yaml_path,
            name_parts=name_parts,
            needed_attributes=Animal.needed_attributes,
        )

    def add_session(
        self,
        date,
        model_settings=None,
        behavior_datas=None,
        regenerate=False,
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
            regenerate=regenerate,
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

    def add_sessions(
        self,
        wanted_dates=["all"],
        behavior_datas=None,
        model_settings=None,
        regenerate=False,
        regenerate_plots=False,
        **kwargs,
    ):
        if not model_settings:
            model_settings = kwargs if len(kwargs) > 0 else self.model_settings
        # Search for Sessions
        sessions_root_path = self.root_dir.joinpath(self.id)
        # search for directories with in date format
        present_session_dates = get_directories(
            sessions_root_path, regex_search="[0-9]{8}"
        )
        for date in present_session_dates:
            if date in wanted_dates or "all" in wanted_dates:
                self.add_session(
                    date,
                    model_settings=model_settings,
                    behavior_datas=behavior_datas,
                    regenerate=regenerate,
                    regenerate_plots=regenerate_plots,
                )

    def get_pipeline_models(
        self,
        model_naming_filter_include: Union[str, List[str], List[List[str]]] = None,
        model_naming_filter_exclude: Union[str, List[str], List[List[str]]] = None,
        manifolds_pipeline: str = "cebra",
    ):
        session: Session
        models = {}
        for session_date, session in self.sessions.items():
            session_models = session.get_pipeline_models(
                manifolds_pipeline=manifolds_pipeline,
                model_naming_filter_include=model_naming_filter_include,
                model_naming_filter_exclude=model_naming_filter_exclude,
            )
            models[session_date] = session_models
        return models

    def filter_sessions(
        self, wanted_properties: Dict[str, Dict[str, Union[str, List]]] = None
    ):
        """

        Filter the available session tasks based on the wanted properties.

        Parameters
        ----------
        wanted_properties : dict, optional
            A dictionary containing the properties to filter by. The dictionary should have the following structure:
            Example:
                wanted_properties = {
                    "session": {
                        # "date": ["20211022", "20211030", "20211031"],
                    },
                    "task": {
                        "name": ["FS1"],
                    },
                    "neural_metadata": {
                        "area": "CA3",
                        "method": "1P",
                    },
                    "behavior_metadata": {
                        "setup": "openfield",
                    },
                }

        Returns
        -------
        filtered_tasks : dict
            dict containing the filtered tasks based on the wanted properties with the task id as key and the task object as value.
        """
        if not wanted_properties:
            print("No wanted properties given. Returning tasks sessions")
            wanted_properties = {}
        filtered_tasks = {}

        for session_date, session in self.sessions.items():
            wanted = True
            if "session" in wanted_properties:
                wanted = wanted_object(session, wanted_properties["session"])

            if wanted:
                filtered_session_tasks = session.filter_tasks(wanted_properties)
                filtered_tasks.update(filtered_session_tasks)
        return filtered_tasks

    def get_unique_model_information(
        self,
        labels_name: str = "",
        wanted_information: List[str] = ["embedding", "loss"],
        manifolds_pipeline: str = "cebra",
        model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
        model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
        train_or_test: str = "train",
    ):
        """
        Get the unique model information all sessions and tasks.

        Only possible if a unique model is found for each task and labels_name is in the model name.

        Parameters
        ----------
        labels_name : str
            The name of the labels describing the behavior used for labeling the embeddings.
        wanted_information : list
            A list containing the wanted information to extract from the models (default is ["embedding", "loss"]).
            Options are
                - "model": the unique models
                - "embedding": the embeddings of the unique models
                - "loss": the training losses of the unique models
                - "raw": the raw data used for training the unique models, which is binarized neural data for models based on cebra
                - "labels": the labels of the data points used for training the unique models
                - "fluorescence": the fluorescence data which is the base for the raw binarized data
                - "all": all of the above
        manifolds_pipeline : str, optional
            The name of the manifolds pipeline to use for decoding (default is "cebra").
        model_naming_filter_include : list, optional
            A list of lists containing the model naming parts to include (default is None).
            If None, all models will be included, which will result in an error if more than one model is found.
        model_naming_filter_exclude : list, optional
            A list of lists containing the model naming parts to exclude (default is None).
            If None, no models will be excluded.

        Returns
        -------
        embeddings : dict
            A dictionary containing the embeddings of the unique models with the task identifier as key and the embeddings as value.
        losses : dict
            A dictionary containing the losses of the unique models with the task identifier as key and the losses as value.
        labels : dict
            A dictionary containing the labels variable name and a list of labels.

        """
        # extract embeddings and losses
        models = {}
        raws = {}
        embeddings = {}
        losss = {}
        fluorescences = {}
        labels = {"name": labels_name, "labels": []}

        for session_date, session in self.sessions.items():
            for task_name, task in session.tasks.items():
                models_class, task_models = task.models.get_pipeline_models(
                    manifolds_pipeline=manifolds_pipeline,
                    model_naming_filter_include=model_naming_filter_include,
                    model_naming_filter_exclude=model_naming_filter_exclude,
                )
                if len(task_models) > 1:
                    for model_name in task_models.keys():
                        if labels_name not in model_name:
                            task_models.pop(model_name)
                    if len(task_models) > 1:
                        raise ValueError(
                            "More than one model found. Improve filtering."
                        )

                if len(task_models) == 0:
                    continue
                task_model = next(iter(task_models.values()))
                models[task.id] = task_model
                raws[task.id] = task_model.get_data(
                    train_or_test=train_or_test, type="neural"
                )
                embeddings[task.id] = task_model.get_data(
                    train_or_test=train_or_test, type="embedding"
                )
                labels["labels"].append(
                    task_model.get_data(train_or_test=train_or_test, type="behavior")
                )
                losss[task.id] = task_model.get_loss()

                if "fluorescence" in wanted_information and train_or_test == "test":
                    global_logger.warning(
                        "WARNING: RAW Fluorescence data is not available, for test data. Only Binarized Fluoresence"
                    )
                    fluorescences[task.id] = None
                else:
                    fluorescences[task.id] = task.neural.get_process_data(
                        type="unprocessed"
                    )

        information = {}
        for wanted_info in wanted_information:
            information[wanted_info] = locals()[wanted_info + "s"]

        return information

    def plot_task_models(
        self,
        model_naming_filter_include: Union[str, List[str], List[List[str]]] = None,
        model_naming_filter_exclude: Union[str, List[str], List[List[str]]] = None,
        train_or_test: str = "train",
        plot: Union[str, List[str]] = ["embedding", "loss"],
        to_2d: bool = False,
        behavior_type: str = "position",
        manifolds_pipeline: str = "cebra",
        embeddings_title: Optional[str] = None,
        losses_title: Optional[str] = None,
        losses_coloring: Optional[str] = "rainbow",
        title_comment: Optional[str] = None,
        markersize: float = None,
        alpha: float = None,
        figsize: Tuple[int, int] = None,
        dpi: int = 300,
        as_pdf: bool = False,
    ):
        """
        Plot model embeddings and losses nearby each other for every task in every session.

        Only possible if a unique model is found for each task. Losses can be
        colored by rainbow, distinct, or mono colors.

        Parameters
        ----------
        model_naming_filter_include : list, optional
            A list of lists containing the model naming parts to include (default is None).
            If None, all models will be included, which will result in an error if more than one model is found.
            Options:
                - single string: only one property has to be included
                - list of strings: all properties have to be included
                - list of lists of strings: Either one of the properties in the inner list has to be included

        model_naming_filter_exclude : list, optional
            A list of lists containing the model naming parts to exclude (default is None).
            If None, no models will be excluded.
            Options:
                - single string: only one property has to be excluded
                - list of strings: all properties have to be excluded
                - list of lists of strings: Either one of the properties in the inner list has to be excluded

        plot : str or list oi str, optional
            A list containing the plots to show (default is ["embedding", "loss"]).

        train_or_test : str, optional
            The data type to plot (default is "train").

        to_2d : bool, optional
            If True, the embeddings will be plotted in 2D (default is False).

        behavior_type : str, optional
            The behavior type to use for labeling the embeddings (default is "position").
        """
        plot = make_list_ifnot(plot)

        info = self.get_unique_model_information(
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
            manifolds_pipeline=manifolds_pipeline,
            labels_name=behavior_type,
            train_or_test=train_or_test,
            wanted_information=["embedding", "loss", "label"],
        )
        embeddings, losses, labels = info["embedding"], info["loss"], info["label"]

        train_info = self.get_unique_model_information(
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
            manifolds_pipeline=manifolds_pipeline,
            labels_name=behavior_type,
            train_or_test="train",
            wanted_information=["label"],
        )
        train_labels = train_info["label"]
        min_val_labels = np.min(train_labels)
        max_val_labels = np.max(train_labels)

        # plot embeddings
        embeddings_title = (
            f"{manifolds_pipeline.upper()} embeddings {self.id}"
            if not embeddings_title
            else embeddings_title
        )
        embeddings_title = add_descriptive_metadata(
            text=embeddings_title, comment=title_comment, metadata=None, keys=None
        )

        viz = Vizualizer(root_dir=self.root_dir)
        if "embedding" in plot:
            # embeddings nearby each other
            viz.plot_multiple_embeddings(
                embeddings=embeddings,
                labels=labels,
                min_val=min_val_labels,
                max_val=max_val_labels,
                title=embeddings_title,
                projection="2d" if to_2d else "3d",
                show_hulls=False,
                markersize=markersize,
                figsize=figsize,
                alpha=alpha,
                dpi=dpi,
                as_pdf=as_pdf,
            )

        if "loss" in plot:
            losses_title = (
                f"{manifolds_pipeline.upper()} losses {self.id}"
                if not losses_title
                else losses_title
            )
            losses_title = add_descriptive_metadata(
                text=losses_title, comment=title_comment, metadata=None, keys=None
            )
            # losses nearby each other
            viz.plot_losses(
                losses=losses,
                title=losses_title,
                alpha=0.8,
                figsize=figsize,
                coloring_type=losses_coloring,
                plot_iterations=False,
                as_pdf=as_pdf,
            )

    def session_model_cross_decode(
        self,
        manifolds_pipeline: str = "cebra",
        model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
        model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
        n_neighbors: Optional[int] = 36,
    ):
        """
        Calculates the decoding performance between models from all tasks based on the wanted model names.

        Parameters
        ----------
        manifolds_pipeline : str, optional
            The name of the manifolds pipeline to use for decoding (default is "cebra").
        model_naming_filter_include : list, optional
            A list of lists containing the model naming parts to include (default is None).
            If None, all models will be included.
        model_naming_filter_exclude : list, optional
            A list of lists containing the model naming parts to exclude (default is None).
            If None, no models will be excluded.
        n_neighbors : int, optional
            The number of neighbors to use for the KNN algorithm (default is 36).

        Returns
        -------
        task_decoding_statistics : dict
            A dictionary containing the decoding statistics between models based on the wanted model names.
        """
        info = self.get_unique_model_information(
            wanted_information=["model"],
            manifolds_pipeline=manifolds_pipeline,
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
        )
        unique_models = info["model"]
        
        data_labels = []
        task_decoding_statistics = {}
        global_logger.info(
            f"""Start calculating decoding statistics for all sessions, tasks and models found from {manifolds_pipeline} pipeline using naming filter including {model_naming_filter_include} and excluding {model_naming_filter_exclude}"""
        )
        for task_name, task_model in iter_dict_with_progress(unique_models):
            global_logger.info(f"Decoding statistics for {task_name}")
            task_model.define_decoding_statistics(regenerate=True, n_neighbors=n_neighbors)
            mean = task_model.get_decoding_statistics("mean")
            variance = task_model.get_decoding_statistics("variance")

            if "relative" in task_model.name:
                print("WARNING: Numbers are relavtive")
                """global_logger.warning(
                    f"Detected relative in task_model name {task_model.name}. Converting relative performance to absolute using max possible value possible."
                )
                behavior_type = task_model.name.split("_")[1]
                absolute_data = self.tasks[task_name].behavior.get_multi_data(
                    sources=behavior_type
                )[0]
                # convert percentage to cm
                max_position_absolute_model_data = np.max(absolute_data)
                mean = mean * max_position_absolute_model_data
                variance = variance * max_position_absolute_model_data"""

            #stimulus_type = self.tasks[task_name].behavior_metadata["stimulus_type"]
            stimulus_type = ""
            task_name_type = f"{task_name} ({stimulus_type})"
            task_decoding_statistics[task_name_type] = {}
            data_labels.append(task_name_type)
            task_decoding_statistics[task_name_type][stimulus_type] = {
                "mean": mean,
                "variance": variance,
            }

            for task_name2, task_models2 in unique_models.items():
                if task_name == task_name2:
                    continue
                #stimulus_type2 = self.tasks[task_name2].behavior_metadata[
                #    "stimulus_type"
                #]
                stimulus_type2 = ""
                model2 = task_models2 # [list(task_models2.keys())[0]]
                stimulus_decoding = f"{task_name_type}_{stimulus_type2}"

                neural_data_test_to_embedd = model2.get_data(train_or_test="test", type="neural")
                labels_test = model2.get_data(train_or_test="test", type="behavior")
                if task_model.get_data(type="neural").shape[1] != model2.get_data(type="neural").shape[1]:
                    print(f"WARNING: Number of Neurons in Datasets not equal. ADAPTING MODEL")
                    adapted_task_model = task_model.fit(neural_data_test_to_embedd, labels_test, adapt=True)
                    decoding_of_other_task = adapted_task_model.define_decoding_statistics(
                        neural_data_test_to_embedd=neural_data_test_to_embedd,
                        labels_test=labels_test,
                        n_neighbors=n_neighbors,
                    )
                else:
                    decoding_of_other_task = task_model.define_decoding_statistics(
                        neural_data_test_to_embedd=neural_data_test_to_embedd,
                        labels_test=labels_test,
                        n_neighbors=n_neighbors,
                    )

                mean = decoding_of_other_task["rmse"]["mean"]
                variance = decoding_of_other_task["rmse"]["variance"]

                if "relative" in task_model.name:
                    print("WARNING: Numbers are relavtive")
                    """global_logger.warning(
                        f"Detected relative in model name {model.name}. Converting relative performance to absolute using max possible value possible."
                    )
                    behavior_type = model.name.split("_")[1]
                    absolute_data = self.tasks[task_name].behavior.get_multi_data(
                        sources=behavior_type
                    )[0]
                    # convert percentage to cm
                    max_position_absolute_model_data = np.max(absolute_data)
                    mean = mean * max_position_absolute_model_data
                    variance = variance * max_position_absolute_model_data"""

                if stimulus_decoding in task_decoding_statistics[task_name_type]:
                    stimulus_decoding = f"{stimulus_decoding}_2"
                task_decoding_statistics[task_name_type][stimulus_decoding] = {
                    "mean": mean,
                    "variance": variance,
                }
                data_labels.append(stimulus_decoding)

        global_logger.info(task_decoding_statistics)

        Vizualizer.barplot_from_dict_of_dicts(
            task_decoding_statistics,
            data_labels=data_labels,
            figsize=(20, 6),
            additional_title=f"- Decoding perfomance between Models based on Environments (moving) - {self.id}",
        )
        return task_decoding_statistics

    def structural_indices(
        self,
        manifolds_pipeline: str = "cebra",
        model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
        model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
        use_raw: bool = False,
        params: Dict[str, Union[int, bool, List[int]]] = None,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        to_transform_data: Optional[np.ndarray] = None,
        behavior_data_types: List[str] = None,
        to_2d: bool = False,
        regenerate: bool = False,
        plot: bool = True,
        as_pdf: bool = False,
    ):
        animal_structrual_indices = {}
        for date, session in self.sessions.items():
            session: Session
            global_logger.info(f"Calculating structural indices for {session.id}")
            session_structrual_indices = session.structural_indices(
                manifolds_pipeline=manifolds_pipeline,
                model_naming_filter_include=model_naming_filter_include,
                model_naming_filter_exclude=model_naming_filter_exclude,
                regenerate=regenerate,
                embeddings=embeddings,
                labels=labels,
                to_transform_data=to_transform_data,
                behavior_data_types=behavior_data_types,
                to_2d=to_2d,
                params=params,
                plot=plot,
                use_raw=use_raw,
                as_pdf=as_pdf,
            )
            animal_structrual_indices[date] = session_structrual_indices
        # TODO: plotting
        return animal_structrual_indices


class Session:
    """Represents a session in the dataset."""

    tasks: Dict[str, Task]
    needed_attributes: List[str] = ["tasks_infos"]

    def __init__(
        self,
        animal_id,
        date,
        animal_dir=None,
        session_dir=None,
        data_dir=None,
        model_dir=None,
        behavior_datas=None,
        regenerate=False,
        regenerate_plots=False,
        model_settings={},
        **kwargs,
    ):
        if not animal_dir and not session_dir:
            raise ValueError(
                f"No animal_dir or session_dir given. for {self.__class__}"
            )

        self.animal_id: str = animal_id
        self.date: str = date
        self.id: str = f"{self.animal_id}_{self.date}"
        self.dir: Path = Path(session_dir or animal_dir.joinpath(date))
        self.data_dir: Path = data_dir or self.dir
        self.model_dir: Path = Path(model_dir or self.dir.joinpath("models"))
        self.model_settings: Dict[str, Dict] = model_settings
        self.yaml_path: Path = self.dir.joinpath(f"{self.date}.yaml")
        self.tasks_infos: Dict[str, Dict] = None  # loaded from yaml
        self.tasks: Dict[str, Task] = {}
        self.load_metadata()
        if behavior_datas:
            self.add_tasks(model_settings=self.model_settings, **kwargs)
            self.load_all_data(
                behavior_datas=behavior_datas,
                regenerate=regenerate,
                regenerate_plots=regenerate_plots,
            )

    def load_metadata(self, yaml_path=None, name_parts=None):
        load_yaml_data_into_class(
            cls=self,
            yaml_path=yaml_path,
            name_parts=name_parts,
            needed_attributes=Session.needed_attributes,
        )

    def add_task(self, task_name, metadata=None, model_settings=None, **kwargs):
        success = check_correct_metadata(
            string_or_list=self.tasks_infos.keys(), name_parts=task_name
        )
        if success:
            task_object = Task(
                session_id=self.id,
                task_name=task_name,
                session_dir=self.dir,
                # data_dir=self.data_dir,
                model_dir=self.model_dir,  # FIXME: comment this line, if models should be saved inside task folders
                metadata=metadata,
            )
            if not model_settings:
                model_settings = kwargs if len(kwargs) > 0 else self.model_settings
            task_object.init_models(model_settings)
            self.tasks[task_name] = task_object
        else:
            global_logger
            print(f"Skipping Task {task_name}")

    def add_tasks(
        self, model_settings=None, task_name_list: List[str] = None, **kwargs
    ):
        if not task_name_list:
            global_logger.warning("No task_name_list given. Adding all tasks.")
            task_name_list = self.tasks_infos.keys()
        for task_name, metadata in self.tasks_infos.items():
            if task_name in task_name_list:
                if not model_settings:
                    model_settings = kwargs if len(kwargs) > 0 else self.model_settings
                self.add_task(
                    task_name, metadata=metadata, model_settings=model_settings
                )
        return self.tasks

    def load_all_data(
        self, behavior_datas=["position"], regenerate=False, regenerate_plots=False
    ):
        data = {}
        for task_name, task in self.tasks.items():
            data[task_name] = task.load_all_data(
                behavior_datas=behavior_datas,
                regenerate=regenerate,
                regenerate_plots=regenerate_plots,
            )
        return data

    def get_pipeline_models(
        self,
        model_naming_filter_include: Union[str, List[str], List[List[str]]] = None,
        model_naming_filter_exclude: Union[str, List[str], List[List[str]]] = None,
        manifolds_pipeline: str = "cebra",
    ):
        task: Task
        models = {}
        for task_name, task in self.tasks.items():
            models_class, task_models = task.models.get_pipeline_models(
                model_naming_filter_include=model_naming_filter_include,
                model_naming_filter_exclude=model_naming_filter_exclude,
                manifolds_pipeline=manifolds_pipeline,
            )
            models[task_name] = task_models
        return models

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

    def filter_tasks(self, wanted_properties=None):
        """
        Filters the tasks based on the wanted properties.

        Parameters
        ----------
        wanted_properties : dict, optional
            A dictionary containing the properties to filter by. The dictionary should have the following structure:
            Example:
                wanted_properties = {
                    "task": {
                        "name": ["FS1"],
                    },
                    "neural_metadata": {
                        "area": "CA3",
                        "method": "1P",
                    },
                    "behavior_metadata": {
                        "setup": "openfield",
                    },
                }

        Returns
        -------
        filtered_tasks : dict
            dict containing the filtered tasks based on the wanted properties with the task id as key and the task object as value.
        """
        if not wanted_properties:
            print("No wanted properties given. Returning tasks sessions")
            wanted_properties = {}

        filtered_tasks = {}
        for task_name, task in self.tasks.items():
            # check if task has all wanted properties
            wanted = True
            if "task" in wanted_properties:
                wanted = wanted_object(task, wanted_properties["task"])
                for metadata_type in ["behavior_metadata", "neural_metadata"]:
                    if not wanted:
                        break
                    if metadata_type in wanted_properties:
                        wanted = wanted and wanted_object(
                            getattr(task, metadata_type),
                            wanted_properties[metadata_type],
                        )
            if wanted:
                filtered_tasks[task.id] = task
        return filtered_tasks

    def task_model_cross_decode(
        self,
        manifolds_pipeline: str = "cebra",
        model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
        model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
        n_neighbors: Optional[int] = 36,
    ):
        """
        Calculates the decoding performance between models from all tasks based on the wanted model names.

        Parameters
        ----------
        manifolds_pipeline : str, optional
            The name of the manifolds pipeline to use for decoding (default is "cebra").
        model_naming_filter_include : list, optional
            A list of lists containing the model naming parts to include (default is None).
            If None, all models will be included.
        model_naming_filter_exclude : list, optional
            A list of lists containing the model naming parts to exclude (default is None).
            If None, no models will be excluded.
        n_neighbors : int, optional
            The number of neighbors to use for the KNN algorithm (default is 36).

        Returns
        -------
        task_decoding_statistics : dict
            A dictionary containing the decoding statistics between models based on the wanted model names.
        """
        session_models = self.get_pipeline_models(
            manifolds_pipeline=manifolds_pipeline,
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
        )

        data_labels = []
        task_decoding_statistics = {}
        global_logger.info(
            f"""Start calculating decoding statistics for all tasks and models found from {manifolds_pipeline} pipeline using naming filter including {model_naming_filter_include} and excluding {model_naming_filter_exclude}"""
        )
        for task_name, task_models in iter_dict_with_progress(session_models):
            global_logger.info(f"Decoding statistics for {task_name}")
            if len(task_models) > 1:
                global_logger.error(
                    "Only one model per task is allowed for cross decoding"
                )
                raise ValueError(
                    "Only one model per task is allowed for cross decoding"
                )
            model = task_models[list(task_models.keys())[0]]
            model.define_decoding_statistics(regenerate=True, n_neighbors=n_neighbors)
            mean = model.get_decoding_statistics("mean")
            variance = model.get_decoding_statistics("variance")

            if "relative" in model.name:
                global_logger.warning(
                    f"Detected relative in model name {model.name}. Converting relative performance to absolute using max possible value possible."
                )
                behavior_type = model.name.split("_")[1]
                absolute_data = self.tasks[task_name].behavior.get_multi_data(
                    sources=behavior_type
                )[0]
                # convert percentage to cm
                max_position_absolute_model_data = np.max(absolute_data)
                mean = mean * max_position_absolute_model_data
                variance = variance * max_position_absolute_model_data

            stimulus_type = self.tasks[task_name].behavior_metadata["stimulus_type"]
            task_name_type = f"{task_name} ({stimulus_type})"
            task_decoding_statistics[task_name_type] = {}
            data_labels.append(task_name_type)
            task_decoding_statistics[task_name_type][stimulus_type] = {
                "mean": mean,
                "variance": variance,
            }

            stimulus_typ2_list = [
                self.tasks[task_name2].behavior_metadata["stimulus_type"] for task_name2 in session_models.keys()
            ]

            for task_name2, task_models2 in session_models.items():
                if task_name == task_name2:
                    continue
                stimulus_type2 = self.tasks[task_name2].behavior_metadata[
                    "stimulus_type"
                ]
                model2 = task_models2[list(task_models2.keys())[0]]
                stimulus_decoding = f"{task_name_type}_{stimulus_type2}"

                neural_data_test_to_embedd = np.concatenate(
                    (
                        model2.get_data(type="neural"),
                        model2.get_data(train_or_test="test", type="neural"),
                    )
                )
                labels_test = np.concatenate(
                    (
                        model2.get_data(type="behavior"),
                        model2.get_data(train_or_test="test", type="behavior"),
                    )
                )

                decoding_of_other_task = model.define_decoding_statistics(
                    neural_data_test_to_embedd=neural_data_test_to_embedd,
                    labels_test=labels_test,
                    n_neighbors=n_neighbors,
                )

                mean = decoding_of_other_task["rmse"]["mean"]
                variance = decoding_of_other_task["rmse"]["variance"]

                if "relative" in model.name:
                    global_logger.warning(
                        f"Detected relative in model name {model.name}. Converting relative performance to absolute using max possible value possible."
                    )
                    behavior_type = model.name.split("_")[1]
                    absolute_data = self.tasks[task_name].behavior.get_multi_data(
                        sources=behavior_type
                    )[0]
                    # convert percentage to cm
                    max_position_absolute_model_data = np.max(absolute_data)
                    mean = mean * max_position_absolute_model_data
                    variance = variance * max_position_absolute_model_data

                if stimulus_decoding in task_decoding_statistics[task_name_type]:
                    stimulus_decoding = f"{stimulus_decoding}_2"
                task_decoding_statistics[task_name_type][stimulus_decoding] = {
                    "mean": mean,
                    "variance": variance,
                }
                data_labels.append(stimulus_decoding)

        global_logger.info(task_decoding_statistics)

        Vizualizer.barplot_from_dict_of_dicts(
            task_decoding_statistics,
            data_labels=data_labels,
            figsize=(20, 6),
            additional_title=f"- Decoding perfomance between Models based on Environments (moving) - {self.id}",
        )
        return task_decoding_statistics

    def structural_indices(
        self,
        manifolds_pipeline: str = "cebra",
        model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
        model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
        use_raw: bool = False,
        params: Dict[str, Union[int, bool, List[int]]] = None,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        to_transform_data: Optional[np.ndarray] = None,
        behavior_data_types: List[str] = None,
        to_2d: bool = False,
        regenerate: bool = False,
        plot: bool = True,
        as_pdf: bool = False,
    ):
        task: Task
        session_structrual_indices = {}
        for task_name, task in self.tasks.items():
            global_logger.info(f"Calculating structural indices for {task.id}")
            task_structrual_indices = task.structural_indices(
                manifolds_pipeline=manifolds_pipeline,
                model_naming_filter_include=model_naming_filter_include,
                model_naming_filter_exclude=model_naming_filter_exclude,
                regenerate=regenerate,
                embeddings=embeddings,
                labels=labels,
                to_transform_data=to_transform_data,
                behavior_data_types=behavior_data_types,
                to_2d=to_2d,
                params=params,
                plot=plot,
                use_raw=use_raw,
                as_pdf=as_pdf,
            )
            session_structrual_indices[task_name] = task_structrual_indices

            # TODO: plotting
        return session_structrual_indices

    # Place Cells for all sessions
    def plot_cell_activity_pos_by_time(
        self,
        cell_ids: int = None,  # overrides the top_n parameter and plots only the cell with the given ids in the given order
        task_ids: str = None,
        movement_state: str = "moving",
        sort_by: str = "zscore",  # peak, spatial_information, spatial_content, zscore or indices
        reference_task: str = None,  # task id to use for sorting
        top_n: str = 10,  # if set to "significant" will use zscore_thr to get top n cells
        n_tests: int = 1000,
        provided_zscore: np.ndarray = None,
        zscore_thr: float = 2.5,
        smooth: bool = True,
        norm: bool = True,
        window_size: int = 5,
        lines_per_y: int = 3,
        figsize_x=20,
        use_discrete_colors=False,
        cmap: str = "inferno",
        show: bool = False,
        save_pdf: bool = True,
    ):
        """
        Plots the cell activity by position and time.

        Parameters
        ----------
        cell_ids : int or list, optional
            The IDs of the cells to plot. Overrides the top_n parameter (default is None).
        task_ids : str or list, optional
            The IDs of the tasks to include in the plot (default is None).
        movement_state : str, optional
            The movement state to filter by (default is "moving").
        sort_by : str, optional
            The criterion to sort by: "zscore", "peak", "spatial_information", "spatial_content", or "indices" (default is "zscore").
        reference_task : str, optional
            The task ID to use for sorting (default is None).
        top_n : int or str, optional
            The number of top cells to plot, or "significant" to plot all significant cells (default is 10).
        n_tests : int, optional
            The number of tests to perform for z-score calculation (default is 1000).
        provided_zscore : np.ndarray, optional
            An array of precomputed z-scores (default is None).
        zscore_thr : float, optional
            The z-score threshold for significance (default is 2.5).
        smooth : bool, optional
            Whether to smooth the data (default is True).
        norm : bool, optional
            Whether to normalize the data (default is True).
        window_size : int, optional
            The size of the smoothing window (default is 5).
        lines_per_y : int, optional
            The number of lines per y-axis unit (default is 3).
        figsize_x : int, optional
            The width of the figure in inches (default is 20).
        use_discrete_colors : bool, optional
            Whether to use discrete colors for the traces (default is False).
        cmap : str, optional
            The colormap to use for the traces (default is "inferno").
            Colormap to use for the plot. Default is 'inferno' (black to yellow) for better visibility.
            colormaps for dark backgrounds: 'gray', 'inferno', 'magma', 'plasma', 'viridis'
            colormaps for light backgrounds: 'binary', 'cividis', 'spring', 'summer', 'autumn', 'winter'
        show : bool, optional
            Whether to show the plot (default is False).
        save_pdf : bool, optional
            Whether to save the plot as a PDF (default is True).

        Returns
        -------
        list
            A list of figure objects containing the plots.

        Raises
        ------
        ValueError
            If the reference task is not found in the session.
        """

        sort_by = (
            "custom"
            if isinstance(sort_by, list) or isinstance(sort_by, np.ndarray)
            else sort_by
        )

        # get rate map and time map and other info for sorting
        if not reference_task and cell_ids is None:
            print("No reference task or cell ids given. Printing all cells.")
        elif reference_task and cell_ids is None:
            if reference_task not in self.tasks.keys():
                raise ValueError(
                    f"Reference task {reference_task} not found. in session {self.id}"
                )
            task = (
                self.tasks[reference_task]
                if reference_task
                else self.tasks[list(self.tasks.keys())[0]]
            )
            if not provided_zscore:
                print(f"Extracting rate map info for reference task {task.id}")
                (
                    reference_rate_map,
                    reference_time_map,
                    reference_zscore,
                    reference_si_rate,
                    reference_si_content,
                    reference_sorting_indices,
                    reference_labels,
                ) = task.extract_rate_map_info_for_sorting(
                    movement_state=movement_state,
                    smooth=smooth,
                    window_size=window_size,
                    n_tests=n_tests,
                    top_n=top_n,
                    provided_zscore=provided_zscore,
                    zscore_thr=zscore_thr,
                )
            cell_ids = reference_sorting_indices

        cell_ids = make_list_ifnot(cell_ids) if cell_ids is not None else None
        task_cell_dict = {}
        for task_id, task in self.tasks.items():
            if task_ids and task_id not in task_ids:
                continue

            if task_id == reference_task:
                zscore = reference_zscore
                si_rate = reference_si_rate
                labels = reference_labels

            else:
                cell_ids = cell_ids or np.arange(task.neural.photon.data.shape[1])

                # extract zscore and spatial information for labels
                print(f"Extracting rate map info for task {task_id}")
                _, _, zscore, si_rate, _, _, _ = task.extract_rate_map_info_for_sorting(
                    movement_state=movement_state,
                    smooth=smooth,
                    window_size=window_size,
                    n_tests=n_tests,
                )

                labels = [
                    f"zscore: {zscore[cell]:.2f}, SI: {si_rate[cell]:.2f}"
                    for cell in cell_ids
                ]

            cell_dict = PlaceCellDetectors.get_spike_maps_per_laps(
                cell_ids=cell_ids,
                neural_data=task.neural.photon.data,
                behavior=task.behavior,
            )

            for cell_id, label in zip(cell_ids, labels):
                additional_title = (
                    f"{task_id} Stimulus: {task.behavior_metadata['stimulus_type']}"
                )
                cell_dict[cell_id]["additional_title"] = additional_title
                cell_dict[cell_id]["label"] = label

            task_cell_dict[task_id] = cell_dict

        # plotting
        only_moving = True if movement_state == "moving" else False

        cell_task_activity_dict = {}
        for cell_id in cell_ids:
            cell_task_activity_dict[cell_id] = {}
            for task_id, cell_dict in task_cell_dict.items():
                cell_task_activity_dict[cell_id][task_id] = cell_dict[cell_id]

        figures = [None] * len(cell_task_activity_dict)
        for cell_num, (cell_id, cells_task_activity_dict) in enumerate(
            cell_task_activity_dict.items()
        ):
            additional_title = f"{self.id} sorted by {sort_by if top_n != 'significant' else 'zscore'} Cell {cell_id}"
            if only_moving:
                additional_title += " (Moving only)"

            fig = Vizualizer.plot_multi_task_cell_activity_pos_by_time(
                cells_task_activity_dict,
                figsize_x=figsize_x,
                norm=norm,
                smooth=smooth,
                window_size=window_size,
                additional_title=additional_title,
                savepath=None,
                lines_per_y=lines_per_y,
                use_discrete_colors=use_discrete_colors,
                cmap=cmap,
                show=show,
            )

            figures[cell_num] = fig

        if save_pdf:
            with PdfPages(f"{self.id}_cell_task_activity.pdf") as pdf:
                for fig in figures:
                    pdf.savefig(fig)


class Task:
    """Represents a task in the dataset."""

    needed_attributes: List[str] = ["neural_metadata", "behavior_metadata"]
    descriptive_metadata_keys = [
        "stimulus_type",
        "method",
        "processing_software",
    ]

    def __init__(
        self,
        session_id,
        task_name,
        session_dir,
        data_dir=None,
        model_dir=None,
        metadata: dict = {},
    ):
        self.session_id: str = session_id
        self.id: str = f"{session_id}_{task_name}"
        self.name: str = task_name

        self.neural_metadata: Dict[str, Dict]
        self.behavior_metadata: Dict[str, Dict]
        self.neural_metadata, self.behavior_metadata = self.load_metadata(metadata)

        self.data_dir: Path = data_dir or self.define_data_dir(session_dir)

        self.neural: Datasets_Neural = Datasets_Neural(
            root_dir=self.data_dir, metadata=self.neural_metadata, task_id=self.id
        )
        self.behavior_metadata = self.fit_behavior_metadata(self.neural)
        self.behavior: Datasets_Behavior = Datasets_Behavior(
            root_dir=self.data_dir, metadata=self.behavior_metadata, task_id=self.id
        )
        self.model_dir: Path = model_dir or self.data_dir.joinpath("models")
        self.models: Models = None

    def define_data_dir(self, session_dir):
        data_dir = session_dir
        if self.name in get_directories(data_dir):
            data_dir = data_dir.joinpath(self.name)
        return data_dir

    def load_metadata(self, metadata: dict = {}):
        set_attributes_check_presents(
            propertie_name_list=metadata.keys(),
            set_object=self,
            propertie_values=metadata.values(),
            needed_attributes=Task.needed_attributes,
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

    def load_data(
        self, data_source, data_type="neural", regenerate=False, regenerate_plot=False
    ):
        # loads neural or behaviour data
        datasets_object = getattr(self, data_type)
        data = datasets_object.load(
            data_source=data_source,
            regenerate=regenerate,
            regenerate_plot=regenerate_plot,
        )
        return data

    def load_all_data(
        self, behavior_datas=["position"], regenerate=False, regenerate_plots=False
    ):
        """
        neural_Data = ["photon"]
        behavior_datas = ["position", "velocity", "stimulus"]
        """
        data = {"neural": {}, "behavior": {}}
        if "velocity" in behavior_datas and "moving" not in behavior_datas:
            behavior_datas.append("moving")
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
                    regenerate=regenerate,
                    regenerate_plot=regenerate_plots,
                )
        return data

    def init_models(self, model_settings=None, **kwargs):
        if not model_settings:
            model_settings = kwargs
        self.models = Models(
            self.model_dir, model_id=self.name, model_settings=model_settings
        )

    @property
    def random_model(
        self,
        manifolds_pipeline: str = "cebra",
        model_naming_filter_include: List[List[str]] = "12800",  # or [str] or str
        model_naming_filter_exclude: List[List[str]] = "0.",  # or [str] or str
    ):
        model_class, models = self.models.get_pipeline_models(
            manifolds_pipeline=manifolds_pipeline,
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
        )
        model = models[list(models.keys())[np.random.randint(len(models))]]
        global_logger.info(f"Got random model: {model.name}")
        print(f"Random model: {model.name}")
        return model

    # Cebra
    def train_model(
        self,
        model_type: str,  # types: time, behavior, hybrid
        regenerate: bool = False,
        shuffle: bool = False,
        movement_state: str = "all",
        split_ratio: float = 1,
        model_name: str = None,
        neural_data: np.ndarray = None,
        behavior_data: np.ndarray = None,
        transformation: str = None,
        behavior_data_types: List[str] = None,  # ["position"],
        manifolds_pipeline: str = "cebra",
        model_settings: dict = None,
        create_embeddings: bool = True,
    ):
        """
        Train a model for a task using neural, behavior data and the choosen pipeline.

        A created model will inherit the train and test data as well as the embedded data.

        Parameters
        ----------
        model_type : str
            The type of model to train. The available model_types are: time, behavior, hybrid
            time: models the neural data over time
            behavior: models the neural data over behavior data
            hybrid: models the neural data over time and behavior data
        regenerate : bool, optional
            Whether to regenerate and overwrite the model (default is False).
        shuffle : bool, optional
            Whether to shuffle the data before training (default is False).
        movement_state : str, optional
            The movement state to filter by (default is "all").
            The available movement_states are: "all", "moving", "stationary"
        split_ratio : float, optional
            The ratio to data used for training the model, the rest is used for the testing sets. (default is 1).
        model_name : str, optional
            The name of the model (default is None). The name is modified by other parameters to create a more unique name.
        neural_data : np.ndarray, optional
            The neural data to use for training the model (default is None). If None, the neural data is loaded from the task.
            The shape of the neural data should be (n_samples, n_features).
        behavior_data : np.ndarray, optional
            The behavior data to use for training the model (default is None). If None, the behavior data is loaded from the task.
            The shape of the behavior data should be (n_samples, n_features).
        transformation : str, optional
            Whether the behavior data should be transformed.
            Transformation types: "binned", "relative" (default is None).
        behavior_data_types : list, optional
            The types of behavior data to use for training the model (default is None).
            Available types are: "position", "velocity", "stimulus", "moving", "acceleration"
        manifolds_pipeline : str, optional
            The pipeline to use for creating the embeddings (default is "cebra").
        model_settings : dict, optional
            The settings for the model (default is None). If not defined the default settings of the pipeline are used
            The dictionary should contain a key for the model type and the corresponding settings.
        create_embeddings : bool, optional
            Whether to create embeddings for the model (default is True).

        Returns
        -------
        Model
            The trained model.
        """
        # get neural data
        idx_to_keep = self.behavior.moving.get_idx_to_keep(movement_state)

        # neural_data_types = neural_data_types or self.neural_metadata["preprocessing"]
        if neural_data is None:
            neural_data, _ = self.neural.get_multi_data(
                sources=self.neural.imaging_type,
            )
        elif not isinstance(neural_data, np.ndarray):
            raise ValueError("neural_data must be a numpy array.")

        # get behavior data
        if behavior_data is None:
            behavior_data, _ = self.behavior.get_multi_data(
                sources=behavior_data_types,
                transformation=transformation,
            )
        elif not isinstance(behavior_data, np.ndarray):
            raise ValueError("behavior_data must be a numpy array.")

        print(self.id)
        global_logger.info(
            f"Training model {model_type} for task {self.id} using pipeline {manifolds_pipeline}"
        )
        model = self.models.train_model(
            neural_data=neural_data,
            behavior_data=behavior_data,
            idx_to_keep=idx_to_keep,
            model_type=model_type,
            model_name=model_name,
            movement_state=movement_state,
            shuffle=shuffle,
            transformation=transformation,
            split_ratio=split_ratio,
            model_settings=model_settings,
            pipeline=manifolds_pipeline,
            create_embeddings=create_embeddings,
            regenerate=regenerate,
        )
        return model

    def get_behavior_labels(
        self,
        behavior_data_types,
        transformation: str = None,
        idx_to_keep: np.ndarray = None,
        movement_state="all",
    ):
        if idx_to_keep is None and movement_state != "all":
            idx_to_keep = self.behavior.moving.get_idx_to_keep(movement_state)
        labels_dict = {}
        for behavior_data_type in behavior_data_types:
            behavior_data, _ = self.behavior.get_multi_data(
                behavior_data_type,
                transformation=transformation,
                idx_to_keep=idx_to_keep,
            )
            labels_dict[behavior_data_type] = behavior_data
        return labels_dict

    def plot_embeddings(
        self,
        model_naming_filter_include: Union[str, List[str], List[List[str]]] = None,
        model_naming_filter_exclude: Union[str, List[str], List[List[str]]] = None,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        train_or_test: str = "train",
        to_2d: bool = False,
        show_hulls: bool = False,
        to_transform_data: Optional[np.ndarray] = None,
        given_colorbar_ticks: Optional[List] = None,
        labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        behavior_data_types: List[str] = ["position"],
        manifolds_pipeline: str = "cebra",
        title: Optional[str] = None,
        title_comment: Optional[str] = None,
        markersize: float = None,
        alpha: float = None,
        figsize: Tuple[int, int] = None,
        dpi: int = 300,
        as_pdf: bool = False,
    ):
        """
        behavior_data_types : list, optional
            The types of behavior data to use when extracting the labels (default is ["position"]).
            The available types are: "position", "velocity", "stimulus", "moving", "acceleration".
        """
        embeddings, labels_dict = self.extract_wanted_embedding_and_labels(
            cls=self,
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
            embeddings=embeddings,
            train_or_test=train_or_test,
            manifolds_pipeline=manifolds_pipeline,
            to_transform_data=to_transform_data,
            labels=labels,
            to_2d=to_2d,
        )

        _, train_labels_dict = self.extract_wanted_embedding_and_labels(
            cls=self,
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
            embeddings=embeddings,
            train_or_test=train_or_test,
            manifolds_pipeline=manifolds_pipeline,
            to_transform_data=to_transform_data,
            labels=labels,
            to_2d=to_2d,
        )

        viz = Vizualizer(self.data_dir.parent.parent)
        title = (
            f"{manifolds_pipeline.upper()} embeddings {self.id}" if not title else title
        )
        title = add_descriptive_metadata(
            text=title,
            comment=title_comment,
            metadata=self.behavior_metadata,
            keys=Task.descriptive_metadata_keys,
        )

        projection = "2d" if to_2d else "3d"

        # plot embeddings if behavior_data_type is in embedding_title
        for behavior_data_type in behavior_data_types:
            # get ticks
            if given_colorbar_ticks is None:
                dataset_object = getattr(self.behavior, behavior_data_type)
                colorbar_ticks = dataset_object.plot_attributes["yticks"]
            else:
                colorbar_ticks = given_colorbar_ticks
            labels_dict = {"name": behavior_data_type, "labels": []}
            embeddings_to_plot = {}
            for embedding_title, labels in labels_dict.items():
                if behavior_data_type not in embedding_title:
                    continue
                labels_dict["labels"].append(labels)
                min_val_labels = np.min(labels)
                max_val_labels = np.max(labels)
                embeddings_to_plot[embedding_title] = embeddings[embedding_title]

            viz.plot_multiple_embeddings(
                embeddings_to_plot,
                labels=labels_dict,
                min_val=min_val_labels,
                max_val=max_val_labels,
                ticks=colorbar_ticks,
                title=title,
                projection=projection,
                show_hulls=show_hulls,
                markersize=markersize,
                figsize=figsize,
                alpha=alpha,
                dpi=dpi,
                as_pdf=as_pdf,
            )
        return embeddings

    def get_spike_map_per_lap(self, cell_id):
        """
        Returns the spike map for a given cell_id. Used for parallel processing.
        """
        # get data
        cell_activity = self.neural.photon.data[:, cell_id]
        binned_pos = self.behavior.position.binned_data

        # split data by laps
        cell_activity_by_lap = self.behavior.split_by_laps(cell_activity)
        binned_pos_by_lap = self.behavior.split_by_laps(binned_pos)

        # count spikes at position
        max_bin = self.behavior.position.max_bin
        cell_lap_activity = np.zeros((len(cell_activity_by_lap), max_bin))
        for i, (lap_act, lap_pos) in enumerate(
            zip(cell_activity_by_lap, binned_pos_by_lap)
        ):
            counts_at = PlaceCellDetectors.get_spike_map(lap_act, lap_pos, max_bin)
            cell_lap_activity[i] = counts_at

        additional_title = f"{self.id} Belt: {self.behavior_metadata['stimulus_type']} - Cell {cell_id}"

        return cell_lap_activity, additional_title

    def plot_model_losses(
        self,
        models=None,
        title=None,
        manifolds_pipeline="cebra",
        coloring_type="rainbow",
        plot_original=True,
        plot_shuffled=True,
        num_iterations=None,
        plot_iterations=False,
        model_naming_filter_include: Union[str, List[str], List[List[str]]] = None,
        model_naming_filter_exclude: Union[str, List[str], List[List[str]]] = None,
        alpha=0.8,
        figsize=(10, 10),
        as_pdf=False,
    ):
        models_original, models_shuffled = (
            self.models.get_models_splitted_original_shuffled(
                models=models,
                manifolds_pipeline=manifolds_pipeline,
                model_naming_filter_include=model_naming_filter_include,
                model_naming_filter_exclude=model_naming_filter_exclude,
            )
        )

        stimulus_type = (
            self.behavior_metadata["stimulus_type"]
            if "stimulus_type" in self.behavior_metadata.keys()
            else ""
        )
        num_iterations = (
            models_original[0].max_iterations if not num_iterations else num_iterations
        )
        title = title or f"Losses {self.id} {stimulus_type}" if not title else title
        comment = f" - {num_iterations} Iterartions" if not plot_iterations else ""
        title = add_descriptive_metadata(
            text=title,
            comment=comment,
            metadata=self.behavior_metadata,
            keys=Task.descriptive_metadata_keys,
        )

        viz = Vizualizer(self.data_dir.parent.parent)
        losses_original = {}
        for model in models_original:
            if model.fitted:
                losses_original[model.name] = model.state_dict_["loss"]
            else:
                global_logger.warning(
                    f"{model.name} Not fitted. Skipping model {model.name}."
                )
                print(f"Skipping model {model.name}.")

        losses_shuffled = {}
        for model in models_shuffled:
            if model.fitted:
                losses_shuffled[model.name] = model.state_dict_["loss"]
            else:
                global_logger.warning(
                    f"{model.name} Not fitted. Skipping model {model.name}."
                )
                print(f"Skipping model {model.name}.")

        viz.plot_losses(
            losses=losses_original,
            losses_shuffled=losses_shuffled,
            title=title,
            coloring_type=coloring_type,
            plot_original=plot_original,
            plot_shuffled=plot_shuffled,
            alpha=alpha,
            figsize=figsize,
            plot_iterations=plot_iterations,
            as_pdf=as_pdf,
        )

    @staticmethod
    def extract_wanted_embedding_and_labels(
        cls: Union[Task, Multi],
        model_naming_filter_include: Union[str, List[str], List[List[str]]] = None,
        model_naming_filter_exclude: Union[str, List[str], List[List[str]]] = None,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        manifolds_pipeline: str = "cebra",
        train_or_test: str = "train",
        to_transform_data: Optional[np.ndarray] = None,
        use_raw: bool = False,
        labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        to_2d: bool = False,
    ):
        """
        Extracts the wanted data for plotting embeddings.

        Parameters
        ----------
        model_naming_filter_include: List[List[str]] = None,  # or [str] or str
            Filter for model names to include. If None, all models will be included. 3 levels of filtering are possible.
            1. Include all models containing a specific string: "string"
            2. Include all models containing a specific combination of strings: ["string1", "string2"]
            3. Include all models containing one of the string combinations: [["string1", "string2"], ["string3", "string4"]]
        model_naming_filter_exclude: List[List[str]] = None,  # or [str] or str
            Same as model_naming_filter_include but for excluding models.
        use_raw : bool, optional
            Whether to use raw data neural data instead of embeddings (default is False).
        embeddings : dict, optional
            A dictionary containing the embeddings to plot (default is None). If None, the embeddings are created from the models,
            which should inherit the default embedding data (data that was defined to train the model) based on neural data.
        manifolds_pipeline : str, optional
            The pipeline to search for models (default is "cebra").
        to_transform_data : np.ndarray, optional
            The data to transform to embeddings (default is None). If None, the embeddings are created from the models inheriting the default data (data that was defined to train the model)
        labels : np.ndarray or dict, optional
            The labels for the embeddings (default is None). If None, the labels are extracted from the default behavior data (data that was defined to train the model).
        """
        if embeddings is not None and to_transform_data is not None:
            global_logger.error(
                "Either provide embeddings or to_transform_data, not both."
            )
            raise ValueError(
                "Either provide embeddings or to_transform_data, not both."
            )
        if to_transform_data is not None:
            embeddings = cls.models.create_embeddings(
                to_transform_data=to_transform_data,
                to_2d=to_2d,
                model_naming_filter_include=model_naming_filter_include,
                model_naming_filter_exclude=model_naming_filter_exclude,
                manifolds_pipeline="cebra",
            )

        if not isinstance(embeddings, dict):
            models_class, models = cls.models.get_pipeline_models(
                manifolds_pipeline=manifolds_pipeline,
                model_naming_filter_include=model_naming_filter_include,
                model_naming_filter_exclude=model_naming_filter_exclude,
            )
            embeddings = {}
            labels_dict = {}
            for model_name, model in models.items():
                embeddings[model_name] = (
                    model.get_data()
                    if not use_raw
                    else model.get_data(train_or_test=train_or_test, type="neural")
                )
                labels_dict[model_name] = model.get_data(
                    train_or_test=train_or_test, type="behavior"
                )

        # get embedding lables
        if labels is not None:
            if not isinstance(labels, np.ndarray) and not isinstance(labels, dict):
                raise ValueError(f"Provided labels is not a numpy array or dictionary.")
            else:
                if isinstance(labels, np.ndarray):
                    labels_dict = {"Provided_labels": labels}
                else:
                    labels_dict = labels
                    if not equal_number_entries(embeddings, labels_dict):
                        global_logger.warning(
                            f"Number of labels is not equal to all, moving or stationary number of frames. You could extract labels corresponding to only moving using the function self.get_behavior_labels(behavior_data_types, movement_state='moving'\)"
                        )

        return embeddings, labels_dict

    def structural_indices(
        self,
        params: Dict[str, Union[int, bool, List[int]]],
        manifolds_pipeline: str = "cebra",
        model_naming_filter_include: Union[str, List[str], List[List[str]]] = None,
        model_naming_filter_exclude: Union[str, List[str], List[List[str]]] = None,
        use_raw: bool = False,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        to_transform_data: Optional[np.ndarray] = None,
        to_2d: bool = False,
        regenerate: bool = False,
        plot: bool = True,
        as_pdf: bool = False,
        plot_save_dir: Optional[Path] = None,
    ) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:  # TODO: finish return types
        """
        Calculate structural indices for the task given a model.


        Raw or Embedded data as well as labels are extracted from the models that fit the naming filter.
        If n_neighbors is a list, then a parameter sweep is performed.

        This Method is based on a graph-based topological metric able to quantify the amount of structure
        present at the distribution of a given feature over a point cloud in an arbitrary D-dimensional space.
        See the publication(https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011768)
        for specific details and follow this notebook (https://colab.research.google.com/github/PridaLab/structure_index/blob/main/demos/structure_index_demo.ipynb)
        for a step by step demo.
        https://github.com/PridaLab/structure_index

        Parameters:
        -----------
        First Parameters:
            Are explained in the extract_wanted_embedding_and_labels function.
        as_pdf: bool, optional
            Whether to save the plot as a PDF (default is False).
        params: dict
            n_bins: integer (default: 10)
                number of bin-groups the label will be divided into (they will
                become nodes on the graph). For vectorial features, if one wants
                different number of bins for each entry then specify n_bins as a
                list (i.e. [10,20,5]). Note that it will be ignored if
                'discrete_label' is set to True.

            n_neighbors: integer (default: 15) or list of integers
                Number of neighbors used to compute the overlapping between
                bin-groups. This parameter controls the tradeoff between local and
                global structure.

            discrete_label: boolean (default: False)
                If the label is discrete, then one bin-group will be created for
                each discrete value it takes. Note that if set to True, 'n_bins'
                parameter will be ignored.

            num_shuffles: int (default: 100)
                Number of shuffles to be computed. Note it must fall within the
                interval [0, np.inf).

            verbose: boolean (default: False)
                Boolean controling whether or not to print internal process.

        Returns:
        --------
        structure_indices: dict
            SI: float
                structure index

            bin_label: tuple
                Tuple containing:
                    [0] Array indicating the bin-group to which each data point has
                        been assigned.
                    [1] Array indicating feature limits of each bin-group. Size is
                    [number_bin_groups, n_features, 3] where the last dimension
                    contains [bin_st, bin_center, bin_en]

            overlap_mat: numpy 2d array of shape [n_bins, n_bins]
                Array containing the overlapping between each pair of bin-groups.

            shuf_SI: numpy 1d array of shape [num_shuffles,]
                Array containing the structure index computed for each shuffling
                iteration.
        """
        model_class, models = self.models.get_pipeline_models(
            manifolds_pipeline=manifolds_pipeline,
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
        )

        parameter_sweep = is_array_like(params["n_neighbors"])

        task_structural_indices = {}
        for model_name, model in models.items():
            global_logger.info(f"Calculating structural indices for {model_name}")
            model_structural_indices = model.structure_index(
                params=params,
                labels=labels,
                to_transform_data=to_transform_data,
                to_2d=to_2d,
                use_raw=use_raw,
                regenerate=regenerate,
                plot=False if parameter_sweep else plot,
                as_pdf=as_pdf,
                plot_save_dir=plot_save_dir or self.data_dir.parent.joinpath("figures"),
            )
            task_structural_indices[model_name] = model_structural_indices

        if parameter_sweep and plot:
            # TODO: create plotting for parameter sweep
            pass

        return task_structural_indices

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
    def get_rate_time_map(
        self,
        movement_state="moving",
        smooth=True,
        window_size=2,
    ):
        """
        Gets the rate and time maps for the place cells.

        Parameters
        ----------
        movement_state : str, optional
            The movement state to filter by (default is "moving").
        smooth : bool, optional
            Whether to smooth the data (default is True).
        window_size : int, optional
            The size of the smoothing window (default is 2).

        Returns
        -------
        tuple
            rate_map : np.ndarray
                The rate map of the place cells.
            time_map : np.ndarray
                The time map of the place cells.
            activity : np.ndarray
                The filtered neural activity data.
            binned_pos : np.ndarray
                The filtered binned position data.
        """
        idx_to_keep = self.behavior.moving.get_idx_to_keep(movement_state)
        activity = Dataset.filter_by_idx(
            self.neural.photon.data, idx_to_keep=idx_to_keep
        )
        binned_pos = Dataset.filter_by_idx(
            self.behavior.position.binned_data, idx_to_keep=idx_to_keep
        )

        max_bin = self.behavior.position.max_bin

        rate_map, time_map = self.models.place_cell.get_maps(
            activity=activity,
            binned_pos=binned_pos,
            smooth=smooth,
            window_size=window_size,
            max_bin=max_bin,
        )
        return rate_map, time_map, activity, binned_pos

    def extract_rate_map_info_for_sorting(
        self,
        sort_by="zscore",
        movement_state="moving",
        smooth=True,
        window_size=2,
        n_tests=1000,
        top_n=10,
        provided_zscore=None,
        zscore_thr=2.5,
    ):
        """
        Extracts rate map information for sorting.

        Parameters
        ----------
        sort_by : str, optional
            The criterion to sort by: "zscore", "peak", "spatial_information", or "spatial_content" (default is "zscore").
        movement_state : str, optional
            The movement state to filter by (default is "moving").
        smooth : bool, optional
            Whether to smooth the data (default is True).
        window_size : int, optional
            The size of the smoothing window (default is 2).
        n_tests : int, optional
            The number of tests to perform for z-score calculation (default is 1000).
        top_n : int or str, optional
            The number of top cells to return, or "significant" to return all significant cells (default is 10).
        provided_zscore : np.ndarray, optional
            An array of precomputed z-scores (default is None).
        zscore_thr : float, optional
            The z-score threshold for significance (default is 2.5).

        Returns
        -------
        tuple
            rate_map : np.ndarray
                The rate map of the place cells.
            time_map : np.ndarray
                The time map of the place cells.
            zscore : np.ndarray
                The z-scores of the place cells.
            si_rate : np.ndarray
                The spatial information rate of the place cells.
            si_content : np.ndarray
                The spatial information content of the place cells.
            real_sorting_indices : np.ndarray
                The indices of the sorted cells.
            labels : np.ndarray
                The labels for the sorted cells.

        Raises
        ------
        ValueError
            If the sorting criterion is not supported.
        """
        # get rate map and time map
        rate_map, time_map, activity, binned_pos = self.get_rate_time_map(
            movement_state=movement_state,
            smooth=smooth,
            window_size=window_size,
        )

        # get zscore, si_rate and si_content
        max_bin = self.behavior.position.max_bin
        fps = self.behavior_metadata["imaging_fps"]

        if sort_by == "peak":
            zscore = None
        elif sort_by in ["spatial_information", "spatial_content", "zscore"]:
            if sort_by in ["spatial_information", "spatial_content"]:
                n_test = (
                    n_tests if top_n == "significant" and provided_zscore is None else 1
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
                    max_bin=max_bin,
                )
            )
            zscore = zscore if provided_zscore is None else provided_zscore
        else:
            raise ValueError(f"Sorting of the rate_map by {sort_by} not supported.")

        # define labels
        if sort_by == "peak":
            sorting_indices = np.argsort(np.argmax(rate_map, axis=1))
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
            significant_zscore_indices = np.where(zscore >= zscore_thr)[0]
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

        real_sorting_indices = sorting_indices[num_nan : num_nan + top_n]

        return (
            rate_map,
            time_map,
            zscore,
            si_rate,
            si_content,
            real_sorting_indices,
            labels,
        )

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
        provided_zscore=None,
        movement_state="moving",
        norm=True,
        smooth=True,
        window_size=2,
        n_tests=1000,
        sort_by="zscore",  # peak, spatial_information, spatial_content, zscore or indices
        top_n=10,
        zscore_thr=2.5,
        use_discrete_colors=True,
        cmap="inferno",
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
        sort_by = (
            "custom"
            if isinstance(sort_by, list) or isinstance(sort_by, np.ndarray)
            else sort_by
        )
        additional_title = f"{self.id} Belt: {self.behavior_metadata['stimulus_type']} sorted by {sort_by if top_n != 'significant' else 'zscore'}"
        if only_moving:
            additional_title += " (Moving only)"

        rate_map, time_map, zscore, si_rate, si_content, sorting_indices, labels = (
            self.extract_rate_map_info_for_sorting(
                sort_by=sort_by,
                movement_state=movement_state,
                smooth=smooth,
                window_size=window_size,
                n_tests=n_tests,
                top_n=top_n,
                provided_zscore=provided_zscore,
                zscore_thr=zscore_thr,
            )
        )

        to_plot_rate_map = rate_map[sorting_indices]
        to_plot_labels = labels[sorting_indices]

        if sort_by != "zscore":
            to_plot_labels_list = []
            for cell_id, label in zip(sorting_indices, to_plot_labels):
                label = label if isinstance(label, str) else f"{label:.1f}"
                to_plot_labels_list.append(f"Cell {cell_id:>4}: {label}")

        Vizualizer.plot_traces_shifted(
            to_plot_rate_map,
            labels=to_plot_labels,
            additional_title=additional_title,
            use_discrete_colors=use_discrete_colors,
            norm=norm,
            cmap=cmap,
        )
        outputs = {
            "rate_map": rate_map,
            "time_map": time_map,
            "zscore": zscore,
            "sorting_indices": sorting_indices,
            "si_rate": si_rate,
        }
        return outputs

    def plot_cell_activity_pos_by_time(
        self,
        cell_ids,
        labels=None,
        norm=True,
        smooth=True,
        window_size=5,
        lines_per_y=3,
        cmap="inferno",
        show=False,
        save_pdf=False,
    ):
        """
        Plots the cell activity by position and time.
        cmap: str
            Colormap to use for the plot. Default is 'inferno' (black to yellow) for better visibility.
        """
        cell_ids = make_list_ifnot(cell_ids)
        labels = labels or [None] * len(cell_ids)
        if len(labels) != len(cell_ids):
            raise ValueError("Labels must be the same length as cell_ids.")

        if "all" in cell_ids:
            cell_ids = np.arange(self.neural.photon.data.shape[1])

        cell_dict = PlaceCellDetectors.get_spike_maps_per_laps(
            cell_ids=cell_ids,
            neural_data=self.neural.photon.data,
            behavior=self.behavior,
        )  # cell_dict[cell_id]["lap_activity"] = cell_lap_activity

        for cell_id in cell_ids:
            additional_title = f"{self.id} Stimulus: {self.behavior_metadata['stimulus_type']} - Cell {cell_id}"
            cell_dict[cell_id]["additional_title"] = additional_title

        figures = [None] * len(cell_dict)
        for i, (cell_id, cell_data) in enumerate(cell_dict.items()):
            label = labels[i]
            fig = Vizualizer.plot_single_cell_activity(
                cell_data["lap_activity"],
                additional_title=cell_data["additional_title"],
                labels=label,
                norm=norm,
                smooth=smooth,
                window_size=window_size,
                lines_per_y=lines_per_y,
                cmap=cmap,
                show=show,
            )
            figures[i] = fig

        if save_pdf:
            with PdfPages(f"{self.id}_cells_activity.pdf") as pdf:
                for fig in figures:
                    pdf.savefig(fig)

        return

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
        max_bin = self.behavior.position.max_bin

        zscore, si_rate, si_content = (
            self.models.place_cell.si_model.compute_si_zscores(
                activity=activity,
                binned_pos=binned_pos,
                # rate_map=rate_map,
                # time_map=time_map,
                n_tests=n_tests,
                spatial_information_method=method,
                fps=fps,
                max_bin=max_bin,
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
