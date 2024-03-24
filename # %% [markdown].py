# %% [markdown]
# # Hypothesis-driven and discovery-driven analysis with CEBRA
# 
# In this notebook, we show how to:
# 
# - use CEBRA-Time and CEBRA-Behavior and CEBRA-Hybrid in an hypothesis-driven or discovery-driven analysis.
# 
# - use CEBRA-Behavior more specifically in an hypothesis-driven analysis, by testing different hypothesis on positon and direction encoding. 
# 
# It is mostly based on what we present in [Figure 2](https://cebra.ai/docs/cebra-figures/figures/Figure2.html) in Schneider, Lee, Mathis.

# %% [markdown]
# ## Load the data

# %%
"""# install
%pip install matplotlib numpy scipy sklearn pyyaml tqdm torch torchvision
%pip install --pre cebra[datasets,demos]

# you should also install cuda for 10x faster computation 
%pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# for mat to py conversion
#%pip install mat4py ???? #TODO: test and improve matlab files to python conversion

"""

# %%
# install dependencies
#!pip install --pre --user cebra[datasets,demos]

# import 
import os
import numpy as np
from Classes import Animal, Vizualizer, load_all_animals, global_logger_object
from Classes import yield_animal_session_task, yield_animal_session, yield_animal, filter_dict_by_properties
from Classes import correlate_vectors
import matplotlib.pyplot as plt
from cebra import CEBRA
import cebra
import torch
print(f"Cuda activated: {torch.cuda.is_available()}")

plt.style.use('dark_background')
%matplotlib inline
#%matplotlib tk
%autosave 180
%load_ext autoreload
%autoreload 2
%config Completer.use_jedi = False

if not os.path.exists("logs"):
    os.mkdir("logs")
global_logger_object.set_save_dir(os.path.join(os.getcwd(), "logs"))

# %% [markdown]
# Standard cebra models are initialized with default parameters:
#  - `model_architecture="offset10-model"`
#  - `batch_size=512`
#  - `learning_rate=3e-4`
#  - `temperature=1` or `temperature=1.12`
#  - `output_dimension=3`
#  - `max_iterations=5000`
#  - `distance="cosine"`
#  - `conditional="time_delta"` or `conditional="time"`
#  - `device="cuda_if_available"`
#  - `verbose=True`
#  - `time_offsets=10`

# %%
# settings paths
root_dir = "D:\\Experiments"
experimentalist = "Steffen"
paradigm = "Rigid_Plastic" #Intrinsic Imaging
animal_root_dir = os.path.join(root_dir, experimentalist, paradigm)

# setting data to use
behavior_datas = ["position", "distance", "velocity", "moving", "acceleration", "stimulus"]
#movement_states = ["moving", "stationary", "all"]
wanted_animal_ids=["all"]
wanted_dates=["all"]
load_all = False
regenerate_plots=True
#regenerate_behavior_data=True

# cebra setting
regenerate=False
manifolds_pipeline="cebra"
# quick CPU run-time demo, you can drop `max_iterations` to 100-500; otherwise set to 5000
model_settings = {"max_iterations" : 20000, "output_dimension" : 3}
#model_settings = {"max_iterations" : 20000, "output_dimension" : 3}

# %%
#initialize session
if load_all:
    animals = load_all_animals(animal_root_dir, 
             wanted_animal_ids=wanted_animal_ids, 
             wanted_dates=wanted_dates, 
             model_settings=model_settings,#) dont load data 
             behavior_datas=behavior_datas,
             regenerate_plots=regenerate_plots) # Load neural, behavior, data for all session parts.
    
    # load_data manually
    #for animal_id, animal in animals.items(): 
    #    animal.load_all_data(behavior_datas=behavior_datas)
    #    for session_date, session in animal.sessions.items():
    #        session.load_all_data(behavior_datas=behavior_datas)
else:
    animal_id = "DON-004366"
    session_date = "20210228"
    animal = Animal(animal_id, root_dir=animal_root_dir)
    animal.add_session(date=session_date, 
                        model_settings=model_settings, 
                        behavior_data=behavior_datas)
    for session_date, session in animal.sessions.items():
        session.add_all_tasks() #tasks=["S1", "S2", "S3"])
        # Load neural, behavior, data for all session parts.
        data = session.load_all_data(behavior_datas=behavior_datas, 
                                     regenerate_plots=regenerate_plots)

# session specific data #TODO: combine all data for session
#session = None
# animal specific data #TODO: combine all data for animal\
#animal = None

# animals["DON-004366"].sessions["20210228"].tasks["S2"].behavior.stimulus.data

# %% [markdown]
# ## Testing

# %% [markdown]
# ### Synthetic Data

# %%
behavior_types = ["position", "stimulus", "distance"]

# %% [markdown]
# ### Train Models on all data

# %%
# Settings 
## Task
if "animals" in globals():
    animal = animals["DON-004366"]
session = animal.sessions["20210228"]
task = session.tasks["S1"]
print(task.behavior_metadata)

## Dataset settings
behavior_data_types = ["position", "stimulus", "distance"]

## Model settings
model_settings = {"max_iterations" : 20000, "output_dimension" : 3}
movement_state = "moving" #"all" # "stationary"
manifolds_pipeline = "cebra"
regenerate = False


# train models
models = {}
for behavior_type in behavior_data_types:
    models[behavior_type] = task.train_model(model_type="behavior",
                                    model_name=behavior_type,
                                    regenerate=regenerate,
                                    movement_state=movement_state,
                                    behavior_data_types=behavior_type, #behavior_data_types, #auxilarry variables
                                    manifolds_pipeline=manifolds_pipeline,
                                    model_settings=model_settings)

# filter models for plotting
model_naming_filter_exclude = ["shuffled"]
model_naming_filter_include = []
for behavior_type in behavior_data_types:
    model_naming_filter_include.append([behavior_type])

# Create Embeddings for visualization
embeddings = task.plot_model_embeddings(to_transform_data=task.neural.suite2p, 
                                        behavior_data_types=behavior_data_types,
                                        #behavior_data_types=["position"],
                                        model_naming_filter_include = model_naming_filter_include,  # or [str] or str #example [["behavior -"], ["time"], ["hybrid"], ["position"]]
                                        model_naming_filter_exclude = model_naming_filter_exclude,  # or [str] or str
                                        )

# %%
# get vector size
feature_vector_size = task.neural.suite2p.data.shape[1]
print("Feature Vectore size: ", feature_vector_size)

# create a zero vector for the feature vector of no cells active
zero_vector = np.zeros((1, feature_vector_size))
# create a single cell activity matrix
neural_with_zero = np.concatenate([zero_vector, task.neural.suite2p.data[:100]], axis=0)

neural_with_zero_embeddings = {}
for behavior_type in behavior_data_types:
    neural_with_zero_embeddings[behavior_type] = models[behavior_type].transform(neural_with_zero)

label_vector = np.ones(neural_with_zero.shape[0])
label_vector[0] = 0
labels = {"all_1_but_0": label_vector}

embeddings_out = task.plot_model_embeddings(embeddings=neural_with_zero_embeddings, embedding_labels=labels,
                                            markersize=10,
                                            alpha=1,
                                            dpi=300,)
........continue here........

# %%
# get vector size
feature_vector_size = task.neural.suite2p.data.shape[1]
print("Feature Vectore size: ", feature_vector_size)

# create identity matrix for the feature vector of only 1 cell active
identity_matrix = np.identity(feature_vector_size)

# create a zero vector for the feature vector of no cells active
zero_vector = np.zeros((1, feature_vector_size))

# create a single cell activity matrix
single_cell_activity_matrix = np.concatenate([zero_vector, identity_matrix], axis=0)

labels = {"Cell IDs": np.arange(feature_vector_size)}

# use vectors to predict the position, distance, and stimulus
embeddings = {}
for behavior_type in behavior_data_types:
    embeddings[behavior_type] = models[behavior_type].transform(single_cell_activity_matrix)

embeddings_out = task.plot_model_embeddings(embeddings=embeddings, embedding_labels=labels,
                                            markersize=5,
                                            alpha=0.8,
                                            dpi=300,)

# %%
from Classes import normalize_vector_01, normalize_matrix_01

# center the embeddings
centered_embeddings = {}
for behavior_type, embedding in embeddings_out.items():
    all_off_position = embedding[0]
    centered_embedding = embedding - all_off_position
    
    centered_embedding_without_zero = centered_embedding[1:]
    # compute the distance from the origin
    distances = np.linalg.norm(centered_embedding_without_zero, axis=1)
    #TODO: explain salience, use other calculations?

    # compute the salience of each dimension
    normalized_dim_distances = normalize_matrix_01(centered_embedding_without_zero)

    # normalize the salience distances
    normalized_saliences_distances = np.linalg.norm(normalized_dim_distances, axis=1)
    normalized_saliences_distances = normalize_vector_01(normalized_saliences_distances)
    
    # compute cosine similarity between all vectors
    #TODO: explain cosine similarity and correlation connection
    correlation_matrix = correlate_vectors(centered_embedding_without_zero)
    
    centered_embeddings[behavior_type] = {"embedding": centered_embedding_without_zero, 
                                          "distances": distances, 
                                          "normalized_dim_distances": normalized_dim_distances,
                                          "saliences": normalized_saliences_distances,
                                          "correlations": correlation_matrix}

# %%
# determine if 0-vector is in the center

for behavior_type, centered_embedding in centered_embeddings.items():
    print(f"Behavior Type: {behavior_type}")
    print(f"Zero vector in center: {np.all(centered_embedding['embedding'][0] == 0)}")
    print(f"Zero vector distance: {centered_embedding['distances'][0]}")
    print(f"Zero vector salience: {centered_embedding['saliences'][0]}")
    print(f"Zero vector correlation: {centered_embedding['correlations'][0]}")
    print()

# %%
import seaborn as sns
import pandas as pd

def plot_histogram(data, title, bins=100, figsize=(10, 5)):
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.hist(data, bins=bins)
    plt.show()
    plt.close()

def histogam_subplot(data:np.ndarray, title:str, ax, bins=100, xlim=[0,1], xlabel="", ylabel="Frequency", xticklabels=None, color=None):
    ax.set_title(title)
    ax.hist(data.flatten(), bins=bins, color=color)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticklabels=="empty":
        ax.set_xticklabels("")

def heatmap_subplot(matrix, title, ax, sort=False, xlabel="Cell ID", ylabel="Cell ID", cmap="YlGnBu"):
    if sort:
        # Assuming correlations is your correlation matrix as a NumPy array
        # Convert it to a Pandas DataFrame
        correlations_df = pd.DataFrame(matrix)
        # sort the correlation matrix
        matrix = correlations_df.sort_values(by=0, axis=1, ascending=False)

    # Creating a heatmap with sort correlations
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sns.heatmap(matrix, annot=False, cmap=cmap, ax=ax)

def plot_end(plt, fig):
    plt.show()
    plt.close()

def plot_corr_hist_heat_salience(correlation: np.ndarray, saliences, title: str, bins: int = 100, sort=False, figsize=(17, 5)):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title)
    histogam_subplot(correlation, "Correlation", ax1, bins=bins, xlim=[-1, 1], xlabel="Correlation Value", ylabel="Frequency")
    heatmap_subplot(correlation, "Correlation Heatmap", ax2, sort=sort)
    histogam_subplot(saliences, "Saliences", ax3, xlim=[0, 2], bins=bins, xlabel="n", ylabel="Frequency")
    plot_end(plt, fig)

def plot_dist_sal_dims(distances, saliences, normalized_saliences, title, bins=100):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(17, 10))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    fig.suptitle(title+" Histograms")

    histogam_subplot(distances, "Distance from Origin", ax1, bins=bins, color=colors[0], xlim=[0, 2], xticklabels="empty")
    histogam_subplot(saliences, "Normalized Distances", ax2, bins=bins, color=colors[1], xticklabels="empty")
    histogam_subplot(normalized_saliences[:, 0], "normalized X", ax3, bins=bins, color=colors[2], xticklabels="empty")
    histogam_subplot(normalized_saliences[:, 1], "normalized Y", ax4, bins=bins, color=colors[3], xticklabels="empty")
    histogam_subplot(normalized_saliences[:, 2], "normalized Z", ax5, bins=bins, color=colors[4])
    plot_end(plt, fig)

def plot_dist_sal_dims_2(distances, saliences, normalized_saliences, 
                         distances2, saliences2, normalized_saliences2,
                         title, bins=100):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 2, figsize=(17, 10))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    fig.suptitle(title+" Histograms")

    histogam_subplot(distances, "Distance from Origin", ax1[0], bins=bins, color=colors[0], xlim=[0, 2])
    histogam_subplot(saliences, "Normalized Distances", ax2[0], bins=bins, color=colors[1], xticklabels="empty")
    histogam_subplot(normalized_saliences[:, 0], "normalized X", ax3[0], bins=bins, color=colors[2], xticklabels="empty")
    histogam_subplot(normalized_saliences[:, 1], "normalized Y", ax4[0], bins=bins, color=colors[3], xticklabels="empty")
    histogam_subplot(normalized_saliences[:, 2], "normalized Z", ax5[0], bins=bins, color=colors[4])

    histogam_subplot(distances2, "Distance from Origin", ax1[1], bins=bins, color=colors[0], xlim=[0, 2])
    histogam_subplot(saliences2, "Normalized Distances", ax2[1], bins=bins, color=colors[1], xticklabels="empty")
    histogam_subplot(normalized_saliences2[:, 0], "normalized X", ax3[1], bins=bins, color=colors[2], xticklabels="empty")
    histogam_subplot(normalized_saliences2[:, 1], "normalized Y", ax4[1], bins=bins, color=colors[3], xticklabels="empty")
    histogam_subplot(normalized_saliences2[:, 2], "normalized Z", ax5[1], bins=bins, color=colors[4])
    plot_end(plt, fig)

def plot_corr_heat_corr_heat(correlation1, correlation2, title1, title2, sort=False, figsize=(17, 5)):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
    fig.suptitle(title1+" vs "+title2)
    histogam_subplot(correlation1, title1+" Correlation", ax1, bins=100, xlim=[-1, 1], xlabel="Correlation Value", ylabel="Frequency", color="tab:blue")
    heatmap_subplot(correlation1, title1, ax2, sort=sort)
    histogam_subplot(correlation2, title2+" Correlation", ax3, bins=100, xlim=[-1, 1], xlabel="Correlation Value", ylabel="Frequency", color="tab:orange")
    heatmap_subplot(correlation2, title2, ax4, sort=sort)
    plot_end(plt, fig)


# %%
for behavior_type, values in centered_embeddings.items():
    embedding = values["embedding"]
    distances = values["distances"]
    normalized_dim_distances = values["normalized_dim_distances"]
    saliences = values["saliences"]
    correlations = values["correlations"]
    #plot_histogram(saliences, bins=100, title=f"{behavior_type} saliences")
    plot_corr_hist_heat_salience(correlations, distances, title=f"{behavior_type.upper()}", bins=100, sort=True)
    plot_dist_sal_dims(distances, saliences, normalized_dim_distances, title=f"{behavior_type.upper()}", bins=100)
    
    labels = {f"{behavior_type} salience": saliences}
    embeddings_out = task.plot_model_embeddings(embeddings=embeddings, to_transform_data=single_cell_activity_matrix, embedding_labels=labels,
                                            markersize=5,
                                            alpha=0.8,
                                            dpi=300,)
    

# %%
behavior_type = "stimulus"
stim_data = centered_embeddings[behavior_type]
embedding = stim_data["embedding"]
distances = stim_data["distances"]
normalized_dim_distances = stim_data["normalized_dim_distances"]
saliences = stim_data["saliences"]
correlations = stim_data["correlations"]

# %%
def fill_vector(vector, indices, fill_value=np.nan):
    filled_vector = vector.copy()
    filled_vector[indices] = fill_value
    return filled_vector

def fill_matrix(matrix, indices, fill_value=np.nan):
    filled_matrix = matrix.copy()
    filled_matrix[indices, :] = fill_value
    return filled_matrix

def fill(matrix, indices, fill_value=np.nan):
    if len(matrix.shape) == 1:
        return fill_vector(matrix, indices, fill_value)
    else:
        return fill_matrix(matrix, indices, fill_value)
    
def filter_inputs(inputs: dict, indices:np.ndarray, fill_value=np.nan):
    filtered_inputs = inputs.copy()
    for key, value in filtered_inputs.items():
        if len(value.shape) == 1:
            filtered_inputs[key] = fill_vector(value, indices)
        else:
            filtered_inputs[key] = fill_matrix(value, indices)
    return filtered_inputs

def get_top_percentile_indices(vector, percentile=5, indices_smaller=True):
    cutoff = np.percentile(vector, 100-percentile)
    if indices_smaller:
        indices = np.where(vector < cutoff)[0]
    else:
        indices = np.where(vector > cutoff)[0]
    return indices

# %%
minimum_perceniles = np.arange(10, 90.1, 10)
for perenctile in minimum_perceniles[::-1]:
    # filter saliences
    not_saliences_indexes = get_top_percentile_indices(saliences, percentile=100-perenctile)
    saliences_indexes = get_top_percentile_indices(saliences, percentile=perenctile)
    
    inputs = {"distances": distances, 
            "saliences": saliences, 
            "normalized_dim_distances": normalized_dim_distances}
    
    filtered_inputs = filter_inputs(inputs, not_saliences_indexes)
    filtered_distance_values = filtered_inputs["distances"]
    filtered_salience_values = filtered_inputs["saliences"]
    filtered_normalized_dim = filtered_inputs["normalized_dim_distances"]

    filter = np.ones(feature_vector_size)
    filter[not_saliences_indexes] = np.nan
    filtered_correlation_matrix = (correlations.transpose() * filter).transpose() * filter
    ######################################################################################

    not_distances_indexes = get_top_percentile_indices(distances, percentile=100-perenctile)
    distances_indexes = get_top_percentile_indices(distances, percentile=perenctile)
    
    inputs_distances = {"distances": distances, 
            "saliences": saliences, 
            "normalized_dim_distances": normalized_dim_distances}
    
    filtered_inputs_distances = filter_inputs(inputs_distances, not_distances_indexes)
    filtered_distance_values_distances = filtered_inputs_distances["distances"]
    filtered_salience_values_distances = filtered_inputs_distances["saliences"]
    filtered_normalized_dim_distances = filtered_inputs_distances["normalized_dim_distances"]

    filter = np.ones(feature_vector_size)
    filter[not_distances_indexes] = np.nan
    filtered_correlation_matrix_distances = (correlations.transpose() * filter).transpose() * filter
    ######################################################################################

    num_cells_distances = len(distances_indexes)
    num_cells_saliences = len(saliences_indexes)
    equal_cells = 0
    for index in distances_indexes:
        if index in saliences_indexes:
            equal_cells += 1
    plot_corr_heat_corr_heat(filtered_correlation_matrix_distances, filtered_correlation_matrix, "Normalized Distances", f"Distances > {perenctile}% - Same cells {equal_cells}")
    
    title = f"{behavior_type} percentage > {perenctile}% - Same {equal_cells}"
    plot_dist_sal_dims_2(distances=filtered_distance_values_distances, 
                         saliences=filtered_salience_values_distances, 
                         normalized_saliences=filtered_normalized_dim_distances, 
                         distances2=filtered_distance_values, 
                         saliences2=filtered_salience_values, 
                         normalized_saliences2=filtered_normalized_dim, 
                         title=f"Distances {num_cells_distances} | {title} | Saliences {num_cells_saliences} - Same {equal_cells} -")
    
    #plot_dist_sal_dims(filtered_distance_values_distances, filtered_salience_values_distances, filtered_normalized_dim_distances, title=f"{'Distance '+title}", bins=100)
    labels = {"Distance "+title+f" num_cells {num_cells_distances}": filtered_distance_values_distances}
    embeddings_out = task.plot_model_embeddings(embeddings=embeddings, to_transform_data=single_cell_activity_matrix, embedding_labels=labels,
                                        markersize=5,
                                        alpha=0.8,
                                        dpi=300,)
    
    #plot_dist_sal_dims(filtered_distance_values, filtered_salience_values, filtered_normalized_dim, title=f"{'Salience '+title}", bins=100)
    labels = {"Salience "+title+f" num_cells {num_cells_saliences}": filtered_salience_values}
    embeddings_out = task.plot_model_embeddings(embeddings=embeddings, to_transform_data=single_cell_activity_matrix, embedding_labels=labels,
                                        markersize=5,
                                        alpha=0.8,
                                        dpi=300,)

# %% [markdown]
# ##### Decoding

# %%
#Train
split = 0.8
regenerate = False
movement_state = "moving"
manifolds_pipeline = "cebra"
model_settings = {"max_iterations" : 20000, "output_dimension" : 3}

model, neural_data_train, neural_data_test, behavior_data_train, behavior_data_test  = task.train_model(model_type="behavior",
                                                                                                        model_name=behavior_type,
                                                                                                        behavior_data_types=[behavior_type],
                                                                                                        shuffled=False,
                                                                                                        split_ratio=split,
                                                                                                        regenerate=regenerate,
                                                                                                        manifolds_pipeline=manifolds_pipeline,
                                                                                                        movement_state=movement_state,
                                                                                                        model_settings=model_settings)

# %%
neural_data_tests = {"distances": {}, "saliences": {}}
minimum_perceniles = np.arange(10, 90.1, 10)
for perenctile in minimum_perceniles[::-1]:
    # filter saliences

    not_saliences_indexes = get_top_percentile_indices(saliences, percentile=100-perenctile)
    saliences_indexes = get_top_percentile_indices(saliences, percentile=perenctile)


    not_distances_indexes = get_top_percentile_indices(distances, percentile=100-perenctile)
    distances_indexes = get_top_percentile_indices(distances, percentile=perenctile)

    #remove information from the neural data
    percentage_string = f"{100-perenctile}"
    neural_data_tests["distances"][percentage_string] = fill_matrix(neural_data_test, not_distances_indexes, fill_value=0)
    neural_data_tests["saliences"][percentage_string] = fill_matrix(neural_data_test, not_saliences_indexes, fill_value=0)

# %%
# Transform
embedded_train = cebra_behavior_model.transform(neural_data_train)
embedded_test = cebra_behavior_model.transform(neural_data_test)

embedded_test_datasets = {"distances": {}, "saliences": {}}
for scoring_type, neural_data_sets in neural_data_tests.items():
    for percentage, neural_data_test_set in neural_data_sets.items():
        embedded_test_datasets[scoring_type][percentage] = cebra_behavior_model.transform(neural_data_test_set)
    

# %%
from Classes import decode
#decode
decoder_results = decode(embedded_train, embedded_test, behavior_data_train, behavior_data_test)

decoded_test_datasets = {"distances": {}, "saliences": {}}
for scoring_type, embedded_test_sets in embedded_test_datasets.items():
    for percentage, embedded_test_set in embedded_test_sets.items():
        decoded_test_datasets[scoring_type][percentage] = decode(embedded_train, embedded_test_set, behavior_data_train, behavior_data_test)

# %%
print(task.id)
print(task.behavior_metadata)

# %%
decoded_lists = [[decoder_results]]
labels = [["All Cells"]]
labels_flattened = ["All Cells"]
for scoring_type, decoded_test_sets in decoded_test_datasets.items():
    decoded_lists.append([])
    labels.append([])
    for percentage, decoded_test_set in decoded_test_sets.items():
        decoded_lists[-1].append(decoded_test_set)
        label = f"{scoring_type} - {percentage}% cells"
        labels[-1].append(label)
        labels_flattened.append(label)
labels

# %%
#viz = Vizualizer(root_dir=root_dir)
#TODO: Is this working????????

# viz.plot_decoding_score(decoded_model_lists=decoded_model_lists, labels=labels, figsize=(13, 5))
fig = plt.figure(figsize=(13, 5))
fig.suptitle(f"{task.id} top % cells fro Behavioral Decoding of Stimulus", fontsize=16)
colors=["green", "red", "deepskyblue"]
ax1 = plt.subplot(111)
#ax2 = plt.subplot(211)

overall_num = 0
for color, docoded_model_list, labels_list in zip(colors, decoded_lists, labels):
    for num, (decoded, label) in enumerate(zip(docoded_model_list, labels_list)):
        #color = "deepskyblue" if "A\'" == "".join(label[:2]) else "red" if "B" == label[0] else "green"
        alpha = 1 - ((1 / len(docoded_model_list)) * num / 1.3)
        x_pos = overall_num + num
        width = 0.4  # Width of the bars
        ax1.bar(
            #x_pos, decoded[1], width=0.4, color=color, alpha=alpha, label = label
            x_pos, decoded[1], width=0.4, color=color, alpha=alpha, label = label
        )
        #ax2.bar(
            #x_pos, decoded[1], width=0.4, color=color, alpha=alpha, label = label
            #x_pos, decoded[2], width=0.4, color=color, alpha=alpha, label = label
        #)
        ##ax2.scatter(
        #    middle_A_model.state_dict_["loss"][-1],
        #    decoded[1],
        #    s=50,
        #    c=color,
        #    alpha=alpha,
        #    label=label,
        #)
    overall_num = x_pos + 1

ylabel = "Mean stimulus error"

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_ylabel(ylabel)
ax1.grid(axis="y", alpha=0.2)
print_labels = labels_flattened
label_pos = np.arange(len(labels_flattened))
ax1.set_xticks(label_pos)
ax1.set_ylim([0, 1])
ax1.set_xticklabels(print_labels, rotation=45, ha="right")


ylabel = "coefficient of determination (r2 score)"

#ax2.spines["top"].set_visible(False)
#ax2.spines["right"].set_visible(False)
#ax2.set_ylabel(ylabel)
#ax2.grid(axis="y", alpha=0.5)
#print_labels = labels_flattened
#label_pos = np.arange(len(labels_flattened))
#ax2.set_xticks(label_pos)
##ax2.set_ylim([0, 130])
#ax2.set_xticklabels(print_labels, rotation=45, ha="right")

#plt.legend()
plt.show()

# %%
#viz = Vizualizer(root_dir=root_dir)
#TODO: Is this working????????

# viz.plot_decoding_score(decoded_model_lists=decoded_model_lists, labels=labels, figsize=(13, 5))
fig = plt.figure(figsize=(13, 5))
fig.suptitle(f"{task.id} top % cells fro Behavioral Decoding of Stimulus", fontsize=16)
colors=["green", "red", "deepskyblue"]
ax1 = plt.subplot(111)
#ax2 = plt.subplot(211)

overall_num = 0
for color, docoded_model_list, labels_list in zip(colors, decoded_lists, labels):
    for num, (decoded, label) in enumerate(zip(docoded_model_list, labels_list)):
        #color = "deepskyblue" if "A\'" == "".join(label[:2]) else "red" if "B" == label[0] else "green"
        alpha = 1 - ((1 / len(docoded_model_list)) * num / 1.3)
        x_pos = overall_num + num
        width = 0.4  # Width of the bars
        ax1.bar(
            #x_pos, decoded[1], width=0.4, color=color, alpha=alpha, label = label
            x_pos, decoded[2], width=0.4, color=color, alpha=alpha, label = label
        )
        #ax2.bar(
            #x_pos, decoded[1], width=0.4, color=color, alpha=alpha, label = label
            #x_pos, decoded[2], width=0.4, color=color, alpha=alpha, label = label
        #)
        ##ax2.scatter(
        #    middle_A_model.state_dict_["loss"][-1],
        #    decoded[1],
        #    s=50,
        #    c=color,
        #    alpha=alpha,
        #    label=label,
        #)
    overall_num = x_pos + 1

ylabel = "coefficient of determination (r2 score)"

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_ylabel(ylabel)
ax1.grid(axis="y", alpha=0.2)
print_labels = labels_flattened
label_pos = np.arange(len(labels_flattened))
ax1.set_xticks(label_pos)
ax1.set_ylim([0, 1])
ax1.set_xticklabels(print_labels, rotation=45, ha="right")



#ax2.spines["top"].set_visible(False)
#ax2.spines["right"].set_visible(False)
#ax2.set_ylabel(ylabel)
#ax2.grid(axis="y", alpha=0.5)
#print_labels = labels_flattened
#label_pos = np.arange(len(labels_flattened))
#ax2.set_xticks(label_pos)
##ax2.set_ylim([0, 130])
#ax2.set_xticklabels(print_labels, rotation=45, ha="right")

#plt.legend()
plt.show()

# %% [markdown]
# ##### everything on position

# %%
behavior_type = "position"
stim_data = centered_embeddings[behavior_type]
embedding = stim_data["embedding"]
distances = stim_data["distances"]
normalized_dim_distances = stim_data["normalized_dim_distances"]
saliences = stim_data["saliences"]
correlations = stim_data["correlations"]

minimum_perceniles = np.arange(10, 90.1, 10)
for perenctile in minimum_perceniles[::-1]:
    # filter saliences

    not_saliences_indexes = get_top_percentile_indices(saliences, percentile=100-perenctile)
    saliences_indexes = get_top_percentile_indices(saliences, percentile=perenctile)
    
    inputs = {"distances": distances, 
            "saliences": saliences, 
            "normalized_dim_distances": normalized_dim_distances}
    
    filtered_inputs = filter_inputs(inputs, not_saliences_indexes)
    filtered_distance_values = filtered_inputs["distances"]
    filtered_salience_values = filtered_inputs["saliences"]
    filtered_normalized_dim = filtered_inputs["normalized_dim_distances"]

    filter = np.ones(feature_vector_size)
    filter[not_saliences_indexes] = np.nan
    filtered_correlation_matrix = (correlations.transpose() * filter).transpose() * filter
    ######################################################################################

    not_distances_indexes = get_top_percentile_indices(distances, percentile=100-perenctile)
    distances_indexes = get_top_percentile_indices(distances, percentile=perenctile)
    
    inputs_distances = {"distances": distances, 
            "saliences": saliences, 
            "normalized_dim_distances": normalized_dim_distances}
    
    filtered_inputs_distances = filter_inputs(inputs_distances, not_distances_indexes)
    filtered_distance_values_distances = filtered_inputs_distances["distances"]
    filtered_salience_values_distances = filtered_inputs_distances["saliences"]
    filtered_normalized_dim_distances = filtered_inputs_distances["normalized_dim_distances"]

    filter = np.ones(feature_vector_size)
    filter[not_distances_indexes] = np.nan
    filtered_correlation_matrix_distances = (correlations.transpose() * filter).transpose() * filter
    ######################################################################################

    num_cells_distances = len(distances_indexes)
    num_cells_saliences = len(saliences_indexes)
    equal_cells = 0
    for index in distances_indexes:
        if index in saliences_indexes:
            equal_cells += 1
    plot_corr_heat_corr_heat(filtered_correlation_matrix_distances, filtered_correlation_matrix, "Normalized Distances", f"Distances > {perenctile}% - Same cells {equal_cells}")
    
    title = f"{behavior_type} percentage > {perenctile}% - Same {equal_cells}"
    plot_dist_sal_dims_2(distances=filtered_distance_values_distances, 
                         saliences=filtered_salience_values_distances, 
                         normalized_saliences=filtered_normalized_dim_distances, 
                         distances2=filtered_distance_values, 
                         saliences2=filtered_salience_values, 
                         normalized_saliences2=filtered_normalized_dim, 
                         title=f"Distances {num_cells_distances} | {title} | Saliences {num_cells_saliences} - Same {equal_cells} -")
    
    #plot_dist_sal_dims(filtered_distance_values_distances, filtered_salience_values_distances, filtered_normalized_dim_distances, title=f"{'Distance '+title}", bins=100)
    labels = {"Distance "+title+f" num_cells {num_cells_distances}": filtered_distance_values_distances}
    embeddings_out = task.plot_model_embeddings(embeddings=embeddings, to_transform_data=single_cell_activity_matrix, embedding_labels=labels,
                                        markersize=5,
                                        alpha=0.8,
                                        dpi=300,)
    
    #plot_dist_sal_dims(filtered_distance_values, filtered_salience_values, filtered_normalized_dim, title=f"{'Salience '+title}", bins=100)
    labels = {"Salience "+title+f" num_cells {num_cells_saliences}": filtered_salience_values}
    embeddings_out = task.plot_model_embeddings(embeddings=embeddings, to_transform_data=single_cell_activity_matrix, embedding_labels=labels,
                                        markersize=5,
                                        alpha=0.8,
                                        dpi=300,)

# %% [markdown]
# ##### position decoding

# %%
#Train
split = 0.8
regenerate = False
movement_state = "moving"
manifolds_pipeline = "cebra"
model_settings = {"max_iterations" : 20000, "output_dimension" : 3}

model, neural_data_train, neural_data_test, behavior_data_train, behavior_data_test  = task.train_model(model_type="behavior",
                                                                                                        model_name=behavior_type,
                                                                                                        behavior_data_types=[behavior_type],
                                                                                                        shuffled=False,
                                                                                                        split_ratio=split,
                                                                                                        regenerate=regenerate,
                                                                                                        manifolds_pipeline=manifolds_pipeline,
                                                                                                        movement_state=movement_state,
                                                                                                        model_settings=model_settings)

neural_data_tests = {"distances": {}, "saliences": {}}
minimum_perceniles = np.arange(10, 90.1, 10)
for perenctile in minimum_perceniles[::-1]:
    # filter saliences

    not_saliences_indexes = get_top_percentile_indices(saliences, percentile=100-perenctile)
    saliences_indexes = get_top_percentile_indices(saliences, percentile=perenctile)


    not_distances_indexes = get_top_percentile_indices(distances, percentile=100-perenctile)
    distances_indexes = get_top_percentile_indices(distances, percentile=perenctile)

    #remove information from the neural data
    percentage_string = f"{100-perenctile}"
    neural_data_tests["distances"][percentage_string] = fill_matrix(neural_data_test, not_distances_indexes, fill_value=0)
    neural_data_tests["saliences"][percentage_string] = fill_matrix(neural_data_test, not_saliences_indexes, fill_value=0)

# Transform
embedded_train = cebra_behavior_model.transform(neural_data_train)
embedded_test = cebra_behavior_model.transform(neural_data_test)

embedded_test_datasets = {"distances": {}, "saliences": {}}
for scoring_type, neural_data_sets in neural_data_tests.items():
    for percentage, neural_data_test_set in neural_data_sets.items():
        embedded_test_datasets[scoring_type][percentage] = cebra_behavior_model.transform(neural_data_test_set)
    
from Classes import decode
#decode
decoder_results = decode(embedded_train, embedded_test, behavior_data_train, behavior_data_test)

decoded_test_datasets = {"distances": {}, "saliences": {}}
for scoring_type, embedded_test_sets in embedded_test_datasets.items():
    for percentage, embedded_test_set in embedded_test_sets.items():
        decoded_test_datasets[scoring_type][percentage] = decode(embedded_train, embedded_test_set, behavior_data_train, behavior_data_test)

print(task.id)
print(task.behavior_metadata)

# %%
decoded_lists = [[decoder_results]]
labels = [["All Cells"]]
labels_flattened = ["All Cells"]
for scoring_type, decoded_test_sets in decoded_test_datasets.items():
    decoded_lists.append([])
    labels.append([])
    for percentage, decoded_test_set in decoded_test_sets.items():
        decoded_lists[-1].append(decoded_test_set)
        label = f"{scoring_type} - {percentage}% cells"
        labels[-1].append(label)
        labels_flattened.append(label)
labels

# %%
#viz = Vizualizer(root_dir=root_dir)
#TODO: Is this working????????

# viz.plot_decoding_score(decoded_model_lists=decoded_model_lists, labels=labels, figsize=(13, 5))
fig = plt.figure(figsize=(13, 5))
fig.suptitle(f"{task.id} top % cells fro Behavioral Decoding of Position", fontsize=16)
colors=["green", "red", "deepskyblue"]
ax1 = plt.subplot(111)
#ax2 = plt.subplot(211)

overall_num = 0
for color, docoded_model_list, labels_list in zip(colors, decoded_lists, labels):
    for num, (decoded, label) in enumerate(zip(docoded_model_list, labels_list)):
        #color = "deepskyblue" if "A\'" == "".join(label[:2]) else "red" if "B" == label[0] else "green"
        alpha = 1 - ((1 / len(docoded_model_list)) * num / 1.3)
        x_pos = overall_num + num
        width = 0.4  # Width of the bars
        ax1.bar(
            #x_pos, decoded[1], width=0.4, color=color, alpha=alpha, label = label
            x_pos, decoded[1], width=0.4, color=color, alpha=alpha, label = label
        )
        #ax2.bar(
            #x_pos, decoded[1], width=0.4, color=color, alpha=alpha, label = label
            #x_pos, decoded[2], width=0.4, color=color, alpha=alpha, label = label
        #)
        ##ax2.scatter(
        #    middle_A_model.state_dict_["loss"][-1],
        #    decoded[1],
        #    s=50,
        #    c=color,
        #    alpha=alpha,
        #    label=label,
        #)
    overall_num = x_pos + 1

ylabel = "Mean stimulus error"

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_ylabel(ylabel)
ax1.grid(axis="y", alpha=0.2)
print_labels = labels_flattened
label_pos = np.arange(len(labels_flattened))
ax1.set_xticks(label_pos)
#ax1.set_ylim([0, 1])
ax1.set_xticklabels(print_labels, rotation=45, ha="right")


ylabel = "coefficient of determination (r2 score)"

#ax2.spines["top"].set_visible(False)
#ax2.spines["right"].set_visible(False)
#ax2.set_ylabel(ylabel)
#ax2.grid(axis="y", alpha=0.5)
#print_labels = labels_flattened
#label_pos = np.arange(len(labels_flattened))
#ax2.set_xticks(label_pos)
##ax2.set_ylim([0, 130])
#ax2.set_xticklabels(print_labels, rotation=45, ha="right")

#plt.legend()
plt.show()

# %%
#viz = Vizualizer(root_dir=root_dir)
#TODO: Is this working????????

# viz.plot_decoding_score(decoded_model_lists=decoded_model_lists, labels=labels, figsize=(13, 5))
fig = plt.figure(figsize=(13, 5))
fig.suptitle(f"{task.id} top % cells fro Behavioral Decoding of Position", fontsize=16)
colors=["green", "red", "deepskyblue"]
ax1 = plt.subplot(111)
#ax2 = plt.subplot(211)

overall_num = 0
for color, docoded_model_list, labels_list in zip(colors, decoded_lists, labels):
    for num, (decoded, label) in enumerate(zip(docoded_model_list, labels_list)):
        #color = "deepskyblue" if "A\'" == "".join(label[:2]) else "red" if "B" == label[0] else "green"
        alpha = 1 - ((1 / len(docoded_model_list)) * num / 1.3)
        x_pos = overall_num + num
        width = 0.4  # Width of the bars
        ax1.bar(
            #x_pos, decoded[1], width=0.4, color=color, alpha=alpha, label = label
            x_pos, decoded[2], width=0.4, color=color, alpha=alpha, label = label
        )
        #ax2.bar(
            #x_pos, decoded[1], width=0.4, color=color, alpha=alpha, label = label
            #x_pos, decoded[2], width=0.4, color=color, alpha=alpha, label = label
        #)
        ##ax2.scatter(
        #    middle_A_model.state_dict_["loss"][-1],
        #    decoded[1],
        #    s=50,
        #    c=color,
        #    alpha=alpha,
        #    label=label,
        #)
    overall_num = x_pos + 1

ylabel = "coefficient of determination (r2 score)"

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_ylabel(ylabel)
ax1.grid(axis="y", alpha=0.2)
print_labels = labels_flattened
label_pos = np.arange(len(labels_flattened))
ax1.set_xticks(label_pos)
ax1.set_ylim([0, 1])
ax1.set_xticklabels(print_labels, rotation=45, ha="right")



#ax2.spines["top"].set_visible(False)
#ax2.spines["right"].set_visible(False)
#ax2.set_ylabel(ylabel)
#ax2.grid(axis="y", alpha=0.5)
#print_labels = labels_flattened
#label_pos = np.arange(len(labels_flattened))
#ax2.set_xticks(label_pos)
##ax2.set_ylim([0, 130])
#ax2.set_xticklabels(print_labels, rotation=45, ha="right")

#plt.legend()
plt.show()

# %%


# %% [markdown]
# ##### lowest contributing cells for position

# %%
behavior_type = "position"
stim_data = centered_embeddings[behavior_type]
embedding = stim_data["embedding"]
distances = stim_data["distances"]
normalized_dim_distances = stim_data["normalized_dim_distances"]
saliences = stim_data["saliences"]
correlations = stim_data["correlations"]

minimum_perceniles = np.arange(10, 90.1, 10)
for perenctile in minimum_perceniles[::-1]:
    # filter saliences

    not_saliences_indexes = get_top_percentile_indices(indices_smaller=False, vector=saliences, percentile=perenctile)
    saliences_indexes = get_top_percentile_indices(vector=saliences, percentile=100-perenctile)
    
    inputs = {"distances": distances, 
            "saliences": saliences, 
            "normalized_dim_distances": normalized_dim_distances}
    
    filtered_inputs = filter_inputs(inputs, not_saliences_indexes)
    filtered_distance_values = filtered_inputs["distances"]
    filtered_salience_values = filtered_inputs["saliences"]
    filtered_normalized_dim = filtered_inputs["normalized_dim_distances"]

    filter = np.ones(feature_vector_size)
    filter[not_saliences_indexes] = np.nan
    filtered_correlation_matrix = (correlations.transpose() * filter).transpose() * filter
    ######################################################################################

    not_distances_indexes = get_top_percentile_indices(indices_smaller=False, vector=distances, percentile=perenctile)
    distances_indexes = get_top_percentile_indices(indices_smaller=False, vector=distances, percentile=100-perenctile)
    
    inputs_distances = {"distances": distances, 
            "saliences": saliences, 
            "normalized_dim_distances": normalized_dim_distances}
    
    filtered_inputs_distances = filter_inputs(inputs_distances, not_distances_indexes)
    filtered_distance_values_distances = filtered_inputs_distances["distances"]
    filtered_salience_values_distances = filtered_inputs_distances["saliences"]
    filtered_normalized_dim_distances = filtered_inputs_distances["normalized_dim_distances"]

    filter = np.ones(feature_vector_size)
    filter[not_distances_indexes] = np.nan
    filtered_correlation_matrix_distances = (correlations.transpose() * filter).transpose() * filter
    ######################################################################################

    num_cells_distances = len(distances_indexes)
    num_cells_saliences = len(saliences_indexes)
    equal_cells = 0
    for index in distances_indexes:
        if index in saliences_indexes:
            equal_cells += 1
    plot_corr_heat_corr_heat(filtered_correlation_matrix_distances, filtered_correlation_matrix, "Normalized Distances", f"Distances > {perenctile}% - Same cells {equal_cells}")
    
    title = f"{behavior_type} percentage > {perenctile}% - Same {equal_cells}"
    plot_dist_sal_dims_2(distances=filtered_distance_values_distances, 
                         saliences=filtered_salience_values_distances, 
                         normalized_saliences=filtered_normalized_dim_distances, 
                         distances2=filtered_distance_values, 
                         saliences2=filtered_salience_values, 
                         normalized_saliences2=filtered_normalized_dim, 
                         title=f"Distances {num_cells_distances} | {title} | Saliences {num_cells_saliences} - Same {equal_cells} -")
    
    #plot_dist_sal_dims(filtered_distance_values_distances, filtered_salience_values_distances, filtered_normalized_dim_distances, title=f"{'Distance '+title}", bins=100)
    labels = {"Distance "+title+f" num_cells {num_cells_distances}": filtered_distance_values_distances}
    embeddings_out = task.plot_model_embeddings(embeddings=embeddings, to_transform_data=single_cell_activity_matrix, embedding_labels=labels,
                                        markersize=5,
                                        alpha=0.8,
                                        dpi=300,)
    
    #plot_dist_sal_dims(filtered_distance_values, filtered_salience_values, filtered_normalized_dim, title=f"{'Salience '+title}", bins=100)
    labels = {"Salience "+title+f" num_cells {num_cells_saliences}": filtered_salience_values}
    embeddings_out = task.plot_model_embeddings(embeddings=embeddings, to_transform_data=single_cell_activity_matrix, embedding_labels=labels,
                                        markersize=5,
                                        alpha=0.8,
                                        dpi=300,)

# %%
#Train
split = 0.8
regenerate = False
movement_state = "moving"
manifolds_pipeline = "cebra"
model_settings = {"max_iterations" : 20000, "output_dimension" : 3}

model, neural_data_train, neural_data_test, behavior_data_train, behavior_data_test  = task.train_model(model_type="behavior",
                                                                                                        model_name=behavior_type,
                                                                                                        behavior_data_types=[behavior_type],
                                                                                                        shuffled=False,
                                                                                                        split_ratio=split,
                                                                                                        regenerate=regenerate,
                                                                                                        manifolds_pipeline=manifolds_pipeline,
                                                                                                        movement_state=movement_state,
                                                                                                        model_settings=model_settings)

neural_data_tests = {"distances": {}, "saliences": {}}
minimum_perceniles = np.arange(10, 90.1, 10)
for perenctile in minimum_perceniles[::-1]:
    # filter saliences

    not_saliences_indexes = get_top_percentile_indices(indices_smaller=False, vector=saliences, percentile=perenctile)
    saliences_indexes = get_top_percentile_indices(indices_smaller=False, vector=saliences, percentile=100-perenctile)

    not_distances_indexes = get_top_percentile_indices(indices_smaller=False, vector=distances, percentile=perenctile)
    distances_indexes = get_top_percentile_indices(indices_smaller=False, vector=distances, percentile=100-perenctile)

    #remove information from the neural data
    percentage_string = f"{100-perenctile}"
    neural_data_tests["distances"][percentage_string] = fill_matrix(neural_data_test, not_distances_indexes, fill_value=0)
    neural_data_tests["saliences"][percentage_string] = fill_matrix(neural_data_test, not_saliences_indexes, fill_value=0)

# Transform
embedded_train = cebra_behavior_model.transform(neural_data_train)
embedded_test = cebra_behavior_model.transform(neural_data_test)

embedded_test_datasets = {"distances": {}, "saliences": {}}
for scoring_type, neural_data_sets in neural_data_tests.items():
    for percentage, neural_data_test_set in neural_data_sets.items():
        embedded_test_datasets[scoring_type][percentage] = cebra_behavior_model.transform(neural_data_test_set)
    
from Classes import decode
#decode
decoder_results = decode(embedded_train, embedded_test, behavior_data_train, behavior_data_test)

decoded_test_datasets = {"distances": {}, "saliences": {}}
for scoring_type, embedded_test_sets in embedded_test_datasets.items():
    for percentage, embedded_test_set in embedded_test_sets.items():
        decoded_test_datasets[scoring_type][percentage] = decode(embedded_train, embedded_test_set, behavior_data_train, behavior_data_test)

print(task.id)
print(task.behavior_metadata)

decoded_lists = [[decoder_results]]
labels = [["All Cells"]]
labels_flattened = ["All Cells"]
for scoring_type, decoded_test_sets in decoded_test_datasets.items():
    decoded_lists.append([])
    labels.append([])
    for percentage, decoded_test_set in decoded_test_sets.items():
        decoded_lists[-1].append(decoded_test_set)
        label = f"{scoring_type} - {percentage}% cells"
        labels[-1].append(label)
        labels_flattened.append(label)
print(labels)






#viz = Vizualizer(root_dir=root_dir)
#TODO: Is this working????????

# viz.plot_decoding_score(decoded_model_lists=decoded_model_lists, labels=labels, figsize=(13, 5))
fig = plt.figure(figsize=(13, 5))
fig.suptitle(f"{task.id} lowest % cells fro Behavioral Decoding of Position", fontsize=16)
colors=["green", "red", "deepskyblue"]
ax1 = plt.subplot(111)
#ax2 = plt.subplot(211)

overall_num = 0
for color, docoded_model_list, labels_list in zip(colors, decoded_lists, labels):
    for num, (decoded, label) in enumerate(zip(docoded_model_list, labels_list)):
        #color = "deepskyblue" if "A\'" == "".join(label[:2]) else "red" if "B" == label[0] else "green"
        alpha = 1 - ((1 / len(docoded_model_list)) * num / 1.3)
        x_pos = overall_num + num
        width = 0.4  # Width of the bars
        ax1.bar(
            #x_pos, decoded[1], width=0.4, color=color, alpha=alpha, label = label
            x_pos, decoded[1], width=0.4, color=color, alpha=alpha, label = label
        )
        #ax2.bar(
            #x_pos, decoded[1], width=0.4, color=color, alpha=alpha, label = label
            #x_pos, decoded[2], width=0.4, color=color, alpha=alpha, label = label
        #)
        ##ax2.scatter(
        #    middle_A_model.state_dict_["loss"][-1],
        #    decoded[1],
        #    s=50,
        #    c=color,
        #    alpha=alpha,
        #    label=label,
        #)
    overall_num = x_pos + 1

ylabel = "Mean stimulus error"

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_ylabel(ylabel)
ax1.grid(axis="y", alpha=0.2)
print_labels = labels_flattened
label_pos = np.arange(len(labels_flattened))
ax1.set_xticks(label_pos)
#ax1.set_ylim([0, 1])
ax1.set_xticklabels(print_labels, rotation=45, ha="right")


ylabel = "mean position in cm"

#ax2.spines["top"].set_visible(False)
#ax2.spines["right"].set_visible(False)
#ax2.set_ylabel(ylabel)
#ax2.grid(axis="y", alpha=0.5)
#print_labels = labels_flattened
#label_pos = np.arange(len(labels_flattened))
#ax2.set_xticks(label_pos)
##ax2.set_ylim([0, 130])
#ax2.set_xticklabels(print_labels, rotation=45, ha="right")

#plt.legend()
plt.show()




#viz = Vizualizer(root_dir=root_dir)
#TODO: Is this working????????

# viz.plot_decoding_score(decoded_model_lists=decoded_model_lists, labels=labels, figsize=(13, 5))
fig = plt.figure(figsize=(13, 5))
fig.suptitle(f"{task.id} lowest % cells fro Behavioral Decoding of Position", fontsize=16)
colors=["green", "red", "deepskyblue"]
ax1 = plt.subplot(111)
#ax2 = plt.subplot(211)

overall_num = 0
for color, docoded_model_list, labels_list in zip(colors, decoded_lists, labels):
    for num, (decoded, label) in enumerate(zip(docoded_model_list, labels_list)):
        #color = "deepskyblue" if "A\'" == "".join(label[:2]) else "red" if "B" == label[0] else "green"
        alpha = 1 - ((1 / len(docoded_model_list)) * num / 1.3)
        x_pos = overall_num + num
        width = 0.4  # Width of the bars
        ax1.bar(
            #x_pos, decoded[1], width=0.4, color=color, alpha=alpha, label = label
            x_pos, decoded[2], width=0.4, color=color, alpha=alpha, label = label
        )
        #ax2.bar(
            #x_pos, decoded[1], width=0.4, color=color, alpha=alpha, label = label
            #x_pos, decoded[2], width=0.4, color=color, alpha=alpha, label = label
        #)
        ##ax2.scatter(
        #    middle_A_model.state_dict_["loss"][-1],
        #    decoded[1],
        #    s=50,
        #    c=color,
        #    alpha=alpha,
        #    label=label,
        #)
    overall_num = x_pos + 1

ylabel = "coefficient of determ ination (r2 score)"

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_ylabel(ylabel)
ax1.grid(axis="y", alpha=0.2)
print_labels = labels_flattened
label_pos = np.arange(len(labels_flattened))
ax1.set_xticks(label_pos)
#ax1.set_ylim([0, 1])
ax1.set_xticklabels(print_labels, rotation=45, ha="right")



#ax2.spines["top"].set_visible(False)
#ax2.spines["right"].set_visible(False)
#ax2.set_ylabel(ylabel)
#ax2.grid(axis="y", alpha=0.5)
#print_labels = labels_flattened
#label_pos = np.arange(len(labels_flattened))
#ax2.set_xticks(label_pos)
##ax2.set_ylim([0, 130])
#ax2.set_xticklabels(print_labels, rotation=45, ha="right")

#plt.legend()
plt.show()



# %%


# %%


# %%


# %%
# only testing multiple iterations!!!!!!!!!!!!!!!!!
#task = session.tasks["S1"]
#behavior_data_types = ["position"]
#behavior_data_types = ["position", "distance", "velocity", "acceleration", "stimulus"]
behavior_data_types = ["position", "stimulus"]
#movement_states = ["moving", "stationary", "all"]
movement_states = ["moving"]
regenerate = False
#movement_states = ["stationary"]
#for max_iter in [1000, 5000, 10000, 20000, 50000, 100000]:
for max_iter in [5000]:
    model_settings = {"max_iterations" : max_iter, "output_dimension" : 3}
    #model_settings = {"max_iterations" : 100, "output_dimension" : 3}
    for animal_id, animal in animals.items():
        for session_date, session in animal.sessions.items():
            for task_name, task in session.tasks.items():
                for movement_state in movement_states:
                    # train model if it is not trained
                    cebra_time_model = task.train_model(model_type="time", 
                                                        regenerate=regenerate, 
                                                        manifolds_pipeline=manifolds_pipeline,
                                                        shuffled=False,
                                                        movement_state=movement_state,
                                                        split_ratio=1,
                                                        model_name=None, # will be set automatically based on some parameters
                                                        neural_data=None, # if None, will use task.neural.suite2p.data
                                                        behavior_data=None, # if None, will use task.behavior.position.data
                                                        behavior_data_types=None, # not needed for time model
                                                        model_settings=model_settings)
                    # CEBRA-Behavior: Train a model with 3D output that uses positional information (position).
                    cebra_behavior_model = task.train_model(model_type="behavior", 
                                                            regenerate=regenerate,
                                                            movement_state=movement_state,
                                                            behavior_data_types=behavior_data_types, #behavior_data_types, #auxilarry variables
                                                            manifolds_pipeline=manifolds_pipeline,
                                                            model_settings=model_settings)
                    # CEBRA-Hybrid: Train a model that uses both time and behavior information. 
                    cebra_hybrid_model = task.train_model(model_type="hybrid", 
                                                                regenerate=regenerate, 
                                                                movement_state=movement_state,
                                                                behavior_data_types=behavior_data_types, #behavior_data_types, #auxilarry variables
                                                                manifolds_pipeline=manifolds_pipeline,
                                                                model_settings=model_settings)
                    # CEBRA-Shuffled Behavior: Train a control model with shuffled neural data.
                    cebra_behavior_shuffled_model = task.train_model(model_type="behavior", 
                                                                    shuffled=True,
                                                                    regenerate=regenerate, 
                                                                    movement_state=movement_state,
                                                                    behavior_data_types=behavior_data_types, #auxilarry variables
                                                                    manifolds_pipeline=manifolds_pipeline,
                                                                    model_settings=model_settings)
                print("Generated models:", list(task.models.cebras.models.keys()))

                #Intra Task
                # Getting embeddings and creating plots
                neural_data = task.neural.suite2p.data
                position_data = task.behavior.position.data
                embeddings = task.plot_model_embeddings(to_transform_data=neural_data, 
                                                        behavior_data_types=["position"], #, "distance", "velocity"] #, "acceleration", "stimulus"], #example [["behavior -"], ["time"], ["hybrid"], ["position"]]
                                                        #behavior_data_types=["position"],
                                                        model_naming_filter_include = None,  #[[str], [str]]  or [str] or str
                                                        model_naming_filter_exclude = "shuffled", #"shuffled",  #[[str], [str]]  or [str] or str
                                                        title_comment=model_settings["max_iterations"])
                task.plot_model_losses(models=None, # if "None" plots all models of task
                                        coloring_type='rainbow', #possible colors: rainbow, distinct, mono
                                        plot_original=True,  # plot original data
                                        plot_shuffled=True,  # plot shuffled data
                                        model_naming_filter_include = None,  # [[str], [str]] or [str] or str
                                        model_naming_filter_exclude = [["hybrid"],["shuffled"]],  # [[str], [str]] or [str] or str
                                        alpha=0.8)
                """
                #TODO: add consistency
                task.plot_consistency_scores(....)
                """
                #TODO: add decoding
                #TODO: add cohomology

            # Inter Task
            #TODO: add decoding
            #TODO: add cohomology
            # Consistency Inter-Task, Inter-Session
            #TODO: add only task consistency
            """
            session.plot_consistency_scores(.....)
            """

        # Inter Session
        #TODO: add only session consistency
        """
        animal.plot_consistency_scores(wanted_stimulus_types=["time", "behavior"]
                                        wanted_embeddings=["A", "B", "A\'"], 
                                        exclude_properties="shuffle", 
                                        figsize=(12, 12))
        """
        #TODO: add decoding
        #TODO: add cohomology

    # Inter Animal
    # consistency Inter-Animal, maybe also Inter-Session
    #TODO: add consistency
    #TODO: add decoding
    #TODO: add cohomology


# %% [markdown]
# ## Summary
# Create for every session:
# 1. models:
#    1. time
#    2. behavior 
#    3. hybrid
#    4. behavior shuffled
# 2. plot
#    1. embeddings
#    2. losses

# %%
#task = session.tasks["S1"]
#behavior_data_types = ["position"]
behavior_data_types = ["position", "distance", "velocity", "acceleration", "stimulus"]
movement_states = ["moving", "stationary", "all"]
#movement_states = ["stationary"]
model_settings = {"max_iterations" : 5000, "output_dimension" : 3}
#model_settings = {"max_iterations" : 100, "output_dimension" : 3}
regenerate = True
#max_iterarions = [1000, 5000, 10000, 20000, 50000, 100000]
#for max_iter in max_iterarions:
#    model_settings = {"max_iterations" : max_iter, "output_dimension" : 3}
for animal_id, animal in animals.items():
    for session_date, session in animal.sessions.items():
        for task_name, task in session.tasks.items():
            for movement_state in movement_states:
                # train model if it is not trained
                cebra_time_model = task.train_model(model_type="time", 
                                                    regenerate=regenerate, 
                                                    manifolds_pipeline=manifolds_pipeline,
                                                    shuffled=False,
                                                    movement_state=movement_state,
                                                    split_ratio=1,
                                                    model_name=None, # will be set automatically based on some parameters
                                                    neural_data=None, # if None, will use task.neural.suite2p.data
                                                    behavior_data=None, # if None, will use task.behavior.position.data
                                                    behavior_data_types=None, # not needed for time model
                                                    model_settings=model_settings)
                # CEBRA-Behavior: Train a model with 3D output that uses positional information (position).
                cebra_behavior_model = task.train_model(model_type="behavior", 
                                                        regenerate=regenerate,
                                                        movement_state=movement_state,
                                                        behavior_data_types=["position", "stimulus"], #behavior_data_types, #auxilarry variables
                                                        manifolds_pipeline=manifolds_pipeline,
                                                        model_settings=model_settings)
                # CEBRA-Hybrid: Train a model that uses both time and behavior information. 
                cebra_hybrid_model = task.train_model(model_type="hybrid", 
                                                            regenerate=regenerate, 
                                                            movement_state=movement_state,
                                                            behavior_data_types=["position", "stimulus"], #behavior_data_types, #auxilarry variables
                                                            manifolds_pipeline=manifolds_pipeline,
                                                            model_settings=model_settings)
                # CEBRA-Shuffled Behavior: Train a control model with shuffled neural data.
                cebra_behavior_shuffled_model = task.train_model(model_type="behavior", 
                                                                shuffled=True,
                                                                regenerate=regenerate, 
                                                                movement_state=movement_state,
                                                                behavior_data_types=["position", "stimulus"], #auxilarry variables
                                                                manifolds_pipeline=manifolds_pipeline,
                                                                model_settings=model_settings)
            print("Generated models:", list(task.models.cebras.models.keys()))

            #Intra Task
            # Getting embeddings and creating plots
            neural_data = task.neural.suite2p.data
            position_data = task.behavior.position.data
            embeddings = task.plot_model_embeddings(to_transform_data=neural_data, 
                                                    behavior_data_types=["position"], #, "distance", "velocity"] #, "acceleration", "stimulus"], #example [["behavior -"], ["time"], ["hybrid"], ["position"]]
                                                    #behavior_data_types=["position"],
                                                    model_naming_filter_include = None,  #[[str], [str]]  or [str] or str
                                                    model_naming_filter_exclude = "shuffled", #"shuffled",  #[[str], [str]]  or [str] or str
                                                    title_comment=model_settings["max_iterations"])
            task.plot_model_losses(models=None, # if "None" plots all models of task
                                    coloring_type='rainbow', #possible colors: rainbow, distinct, mono
                                    plot_original=True,  # plot original data
                                    plot_shuffled=True,  # plot shuffled data
                                    model_naming_filter_include = None,  # [[str], [str]] or [str] or str
                                    model_naming_filter_exclude = [["hybrid"],["shuffled"]],  # [[str], [str]] or [str] or str
                                    alpha=0.8)
            """
            #TODO: add consistency
            task.plot_consistency_scores(....)
            """
            #TODO: add decoding
            #TODO: add cohomology

        # Inter Task
        #TODO: add decoding
        #TODO: add cohomology
        # Consistency Inter-Task, Inter-Session
        #TODO: add only task consistency
        """
        session.plot_consistency_scores(.....)
        """

    # Inter Session
    #TODO: add only session consistency
    """
    animal.plot_consistency_scores(wanted_stimulus_types=["time", "behavior"]
                                    wanted_embeddings=["A", "B", "A\'"], 
                                    exclude_properties="shuffle", 
                                    figsize=(12, 12))
    """
    #TODO: add decoding
    #TODO: add cohomology

# Inter Animal
# consistency Inter-Animal, maybe also Inter-Session
#TODO: add consistency
#TODO: add decoding
#TODO: add cohomology


# %% [markdown]
# ## CEBRA workflow: Discovery-driven and Hypothesis-driven analysis.
# 
# - We will compare CEBRA-Time (discovery-driven), CEBRA-Behavior and CEBRA-Hybrid models (hypothesis-driven) as in the recommended [CEBRA workflow](https://cebra.ai/docs/usage.html#the-cebra-workflow).

# %% [markdown]
# **------------------- BEGINNING OF TRAINING SECTION -------------------**
# ### Train the models
# *[You can skip this section if you already have the models saved]*

# %% [markdown]
# #### CEBRA-Time: Discovery-driven. Train a model that uses time without the behavior information. 
# - We can use CEBRA -Time mode by setting conditional = 'time':
#   - Discovery-driven: time contrastive learning. 
#   - No assumption on the behaviors that are influencing neural activity. 
#   - It can be used as a first step into the data analysis for instance, or as a comparison point to multiple hypothesis-driven analyses.

# %%
print(f"Neural data shape: {task.neural.suite2p.data.shape}")
print(f"Position data shape: {task.behavior.position.data.shape}")

# train model if it is not trained
cebra_time_model = task.train_model(model_type="time", 
                                            regenerate=False, 
                                            neural_data_type="suite2p",
                                            manifolds_pipeline="cebra"
                                            )


# %% [markdown]
# #### CEBRA-Behavior: Train a model with 3D output that uses positional information (position).
# - Setting conditional = 'time_delta' means we will use :
#   - CEBRA-Behavior mode and use 
#   - auxiliary behavior variable for the model training.

# %%
print(f"Neural data shape: {task.neural.suite2p.data.shape}")
print(f"Position data shape: {task.behavior.position.data.shape}")

# train model if it is not trained
cebra_behavior_model = task.train_model(model_type="behavior", 
                                                regenerate=False, 
                                                neural_data_type="suite2p",
                                                behavior_data_types=["position"], #auxilarry variables
                                                manifolds_pipeline="cebra"
                                                )

# %% [markdown]
# #### CEBRA-Hybrid: Train a model that uses both time and positional information. 

# %%
print(f"Neural data shape: {task.neural.suite2p.data.shape}")
print(f"Position data shape: {task.behavior.position.data.shape}") 
behavior_data_types = ["position", "distance", "velocity", "acceleration", "stimulus"]

cebra_hybrid_model = task.train_model(model_type="hybrid", 
                                            regenerate=False, 
                                            shuffled=False, 
                                            neural_data_type="suite2p",
                                            behavior_data_types=behavior_data_types, #auxilarry variables
                                            manifolds_pipeline="cebra"
                                            )

# %% [markdown]
# #### CEBRA-Shuffled Behavior: Train a control model with shuffled neural data.
# - The model specification is the same as the CEBRA-Behavior above.

# %%
cebra_behavior_shuffled_model = task.train_model(model_type="behavior", 
                                                         shuffled=True,
                                                         regenerate=False, 
                                                         neural_data_type="suite2p",
                                                         behavior_data_types=["position"], #auxilarry variables
                                                         manifolds_pipeline="cebra"
                                                         )

# %% [markdown]
# ### Get corresponding embeddings

# %%
neural_data = task.neural.suite2p.data
cebra_time = cebra_time_model.transform(neural_data)
cebra_behavior = cebra_behavior_model.transform(neural_data)
cebra_hybrid = cebra_hybrid_model.transform(neural_data)
cebra_behavior_shuffled = cebra_behavior_shuffled_model.transform(neural_data)

# %% [markdown]
# ### Visualize the embeddings from CEBRA-Behavior, CEBRA-Time and CEBRA-Hybrid

# %% [markdown]
# **Note to Google Colaboratory users:** replace the first line of the next cell (``%matplotlib notebook``) with ``%matplotlib inline``.

# %%
neural_data = task.neural.suite2p.data
position_data = task.behavior.position.data
embeddings = task.plot_model_embeddings(to_transform_data=neural_data, embedding_labels=position_data)
embeddings = task.plot_model_embeddings(to_transform_data=task.neural.suite2p.data, 
                                        behavior_data_types=["position", "distance"])#, "velocity", "acceleration", "stimulus"])

# %% [markdown]
# **------------------- BEGINNING OF TRAINING SECTION -------------------**
# ### Train the models

# %% [markdown]
# #### Train CEBRA-Behavior with ["position", "distance", "velocity", "acceleration", "stimulus"] and shuffeled

# %%
# create models
cebra_position_model = task.models.cebras.behavior(shuffled=False, name="position")
cebra_distance_model = task.models.cebras.behavior(shuffled=False, name="distance")
cebra_velocity_model = task.models.cebras.behavior(shuffled=False, name="velocity")
cebra_acceleration_model = task.models.cebras.behavior(shuffled=False, name="acceleration")
cebra_stimulus_model = task.models.cebras.behavior(shuffled=False, name="stimulus")

cebra_position_shuffled_model = task.models.cebras.behavior(shuffled=True, name="position")
cebra_distance_shuffled_model = task.models.cebras.behavior(shuffled=True, name="distance")
cebra_velocity_shuffled_model = task.models.cebras.behavior(shuffled=True, name="velocity")
cebra_acceleration_shuffled_model = task.models.cebras.behavior(shuffled=True, name="acceleration")
cebra_stimulus_shuffled_model = task.models.cebras.behavior(shuffled=True, name="stimulus")

# %%
# Train CEBRA-Behavior models if not trained
cebra_position_model = task.train_model(model_type="behavior", 
                                                shuffled=False,
                                                regenerate=False,
                                                model_name="position", 
                                                behavior_data_types="position", 
                                                manifolds_pipeline="cebra")
cebra_distance_model = task.train_model(model_type="behavior", 
                                                model_name="distance", 
                                                behavior_data_types="distance")
cebra_velocity_model = task.train_model(model_type="behavior", 
                                                model_name="velocity", 
                                                behavior_data_types="velocity")
cebra_acceleration_model = task.train_model(model_type="behavior", 
                                                model_name="acceleration", 
                                                behavior_data_types="acceleration")
cebra_stimulus_model = task.train_model(model_type="behavior", 
                                                model_name="stimulus", 
                                                behavior_data_types="stimulus")

cebra_position_shuffled_model = task.train_model(model_type="behavior", 
                                                shuffled=True, 
                                                model_name="position", 
                                                behavior_data_types="position")
cebra_distance_shuffled_model = task.train_model(model_type="behavior", 
                                                shuffled=True, 
                                                model_name="distance", 
                                                behavior_data_types="distance")
cebra_velocity_shuffled_model = task.train_model(model_type="behavior", 
                                                shuffled=True, 
                                                model_name="velocity", 
                                                behavior_data_types="velocity")
cebra_acceleration_shuffled_model = task.train_model(model_type="behavior", 
                                                shuffled=True, 
                                                model_name="acceleration", 
                                                behavior_data_types="acceleration")
cebra_stimulus_shuffled_model = task.train_model(model_type="behavior", 
                                                shuffled=True, 
                                                model_name="stimulus", 
                                                behavior_data_types="stimulus")


# %% [markdown]
# **------------------- END OF TRAINING SECTION -------------------**
# ### Visualize embeddings from different hypothesis

# %%
# Load the model and get the corresponding embeddings coloring by only position
#embeddings = task.plot_model_embeddings(to_transform_data=neural_data, embedding_labels=position_data)
embeddings = task.plot_model_embeddings(to_transform_data=neural_data, 
                                        behavior_data_types=["position", "distance", "velocity", "acceleration", "stimulus"],
                                        #behavior_data_types=["position"],
                                        model_naming_filter_include = None,  # or [str] or str #example [["behavior -"], ["time"], ["hybrid"], ["position"]]
                                        model_naming_filter_exclude = "shuffled",  # or [str] or str
                                        )

# %% [markdown]
# ### Visualize the loss of models trained with different hypothesis

# %%
models_to_plot = [cebra_time_model, cebra_behavior_model, cebra_distance_model, cebra_velocity_model, cebra_acceleration_model, cebra_stimulus_model,cebra_behavior_shuffled_model, cebra_position_shuffled_model, cebra_distance_shuffled_model, cebra_velocity_shuffled_model, cebra_acceleration_shuffled_model, cebra_stimulus_shuffled_model]
#task.plot_model_losses(coloring_type='rainbow', plot_original=True, plot_shuffled=True, alpha=0.8)
task.plot_model_losses(models_to_plot, coloring_type='mono', plot_original=True, plot_shuffled=True, alpha=0.8)
#task.plot_model_losses(models_to_plot, coloring_type='mono', plot_original=True, plot_shuffled=True, plot_model_iterations=True, alpha=0.8)
#task.plot_model_losses(models_to_plot, coloring_type='rainbow', plot_original=True, plot_shuffled=True, alpha=0.8)
#task.plot_model_losses(models_to_plot, coloring_type='distinct', plot_original=True, plot_shuffled=True, alpha=0.8)
#task.plot_model_losses(models_to_plot, coloring_type='rainbow', plot_original=True, plot_shuffled=False, alpha=0.8)
#task.plot_model_losses(models_to_plot, coloring_type='rainbow', plot_original=False, plot_shuffled=True, alpha=0.8)


# %% [markdown]
# ## Consistency
# **Compute the consistency maps:**
# Correlation matrices depict the $R^2$ after fitting a linear model between behavior-aligned embeddings of two animals, one as the target one as the source (mean, n=10 runs). Parameters were picked by optimizing average run consistency across rats.

# %% [markdown]
# ### Consistency Inter + Intra -Session

# %%
wanted_embeddings = ["time", "behavior"]
wanted_stimulus_types = ["A", "B", "A\'"]
exclude_properties = None#["'"]
animal = animals["DON-004366"]
animal.plot_consistency_scores(wanted_stimulus_types, wanted_embeddings, exclude_properties=exclude_properties, figsize=(12, 12))

# %% [markdown]
# ### Consistency Inter-Animals

# %%
# TODO: plot consistency scores for all animals

# %% [markdown]
# ## Decoding
# - We will compare CEBRA-Behavior models trained on position and on shuffed position
# - Here, we use the set model dimension; in the paper we used 3-64 on the hippocampus data (and found a consistent topology across these dimensions).

# %% [markdown]
# ### Decoding Intra-Task

# %% [markdown]
# #### Split data

# %%

def split_data(data, test_ratio):
    split_idx = int(len(data) * (1 - test_ratio))
    train = data[:split_idx]
    test = data[split_idx:]
    return np.array(train), np.array(test)

# %%
cebra_behavior_model = task.train_model(model_type="behavior", 
                                                shuffled=False,
                                                regenerate=regenerate,
                                                model_name="all_train", 
                                                behavior_data=label_train_behavior, 
                                                neural_data=neural_train)

# %%
for sname, task in session.tasks.items():
    print(task.behavior_metadata["stimulus_type"])

# %%
##TODO: change notebook code
#..... introduce splitting into train model and transform model
#..... introduce splitting into train model and transform model
#..... introduce splitting into train model and transform model
#..... introduce splitting into train model and transform model
#..... introduce splitting into train model and transform model
#..... introduce splitting into train model and transform model
#..... introduce splitting into train model and transform model
#..... introduce splitting into train model and transform model
#..... introduce splitting into train model and transform model
#..... introduce splitting into train model and transform model
#neural_train, neural_test = split_data(neural_data, 0.8)
task = session.tasks["S1"]
behavior_datas = ["position", "distance", "velocity", "acceleration", "stimulus"]
neural_train, neural_test = split_data(task.neural.suite2p.data, 0.8)
all_behavior = task.behavior.get_multi_data(behavior_datas)
label_train_behavior, label_test_behavior = split_data(all_behavior, 0.8)
label_train_position, label_test_position = split_data(task.behavior.position.data, 0.8)
label_train_distance, label_test_distance = split_data(task.behavior.distance.data, 0.8)
label_train_velocity, label_test_velocity = split_data(task.behavior.velocity.data, 0.8)
label_train_acceleration, label_test_acceleration = split_data(task.behavior.acceleration.data, 0.8)
label_train_stimulus, label_test_stimulus = split_data(task.behavior.stimulus.data, 0.8)

# %% [markdown]
# #### retrain the models on only training data

# %%
# Train CEBRA-Behavior models if not trained
regenerate = False
cebra_behavior_model = task.train_model(model_type="behavior", 
                                                shuffled=False,
                                                regenerate=regenerate,
                                                model_name="all_train", 
                                                behavior_data=label_train_behavior, 
                                                neural_data=neural_train)
cebra_position_model = task.train_model(model_type="behavior", 
                                                shuffled=False,
                                                regenerate=regenerate,
                                                model_name="position_train", 
                                                behavior_data=label_train_position, 
                                                neural_data=neural_train,
                                                manifolds_pipeline="cebra")
cebra_distance_model = task.train_model(model_type="behavior", 
                                                regenerate=regenerate,
                                                model_name="distance_train", 
                                                behavior_data=label_train_distance,
                                                neural_data=neural_train)
cebra_velocity_model = task.train_model(model_type="behavior", 
                                                regenerate=regenerate,
                                                model_name="velocity_train", 
                                                behavior_data=label_train_velocity,
                                                neural_data=neural_train)
cebra_acceleration_model = task.train_model(model_type="behavior", 
                                                model_name="acceleration_train", 
                                                behavior_data=label_train_acceleration,
                                                neural_data=neural_train)
cebra_stimulus_model = task.train_model(model_type="behavior", 
                                                regenerate=regenerate,
                                                model_name="stimulus_train", 
                                                behavior_data=label_train_stimulus,
                                                neural_data=neural_train)

cebra_behavior_shuffled_model = task.train_model(model_type="behavior", 
                                                shuffled=True,
                                                regenerate=regenerate,
                                                model_name="all_train", 
                                                behavior_data=label_train_behavior, 
                                                neural_data=neural_train)
cebra_position_shuffled_model = task.train_model(model_type="behavior", 
                                                regenerate=regenerate,
                                                shuffled=True, 
                                                model_name="position_train", 
                                                behavior_data=label_train_position,
                                                neural_data=neural_train)
cebra_distance_shuffled_model = task.train_model(model_type="behavior", 
                                                regenerate=regenerate,
                                                shuffled=True, 
                                                model_name="distance_train", 
                                                behavior_data=label_train_distance,
                                                neural_data=neural_train)
cebra_velocity_shuffled_model = task.train_model(model_type="behavior", 
                                                shuffled=True, 
                                                regenerate=regenerate,
                                                model_name="velocity_train", 
                                                behavior_data=label_train_velocity,
                                                neural_data=neural_train)
cebra_acceleration_shuffled_model = task.train_model(model_type="behavior", 
                                                regenerate=regenerate,
                                                shuffled=True, 
                                                model_name="acceleration_train", 
                                                behavior_data=label_train_acceleration,
                                                neural_data=neural_train)
cebra_stimulus_shuffled_model = task.train_model(model_type="behavior", 
                                                regenerate=regenerate,
                                                shuffled=True, 
                                                model_name="stimulus_train", 
                                                behavior_data=label_train_stimulus,
                                                neural_data=neural_train)


# %% [markdown]
# #### create embeddings on train and test neural data

# %%
cebra_behavior_model_train = cebra_behavior_model.transform(neural_train)
cebra_behavior_model_test = cebra_behavior_model.transform(neural_test)
cebra_position_model_train = cebra_position_model.transform(neural_train)
cebra_position_model_test = cebra_position_model.transform(neural_test)
cebra_distance_model_train = cebra_distance_model.transform(neural_train)
cebra_distance_model_test = cebra_distance_model.transform(neural_test)
cebra_velocity_model_train = cebra_velocity_model.transform(neural_train)
cebra_velocity_model_test = cebra_velocity_model.transform(neural_test)
cebra_acceleration_model_train = cebra_acceleration_model.transform(neural_train)
cebra_acceleration_model_test = cebra_acceleration_model.transform(neural_test)
cebra_stimulus_model_train = cebra_stimulus_model.transform(neural_train)
cebra_stimulus_model_test = cebra_stimulus_model.transform(neural_test)

cebra_behavior_shuffled_model_train = cebra_behavior_shuffled_model.transform(neural_train)
cebra_behavior_shuffled_model_test =  cebra_behavior_shuffled_model.transform(neural_test)
cebra_position_shuffled_model_train = cebra_position_shuffled_model.transform(neural_train)
cebra_position_shuffled_model_test = cebra_position_shuffled_model.transform(neural_test)
cebra_distance_shuffled_model_train = cebra_distance_shuffled_model.transform(neural_train)
cebra_distance_shuffled_model_test = cebra_distance_shuffled_model.transform(neural_test)
cebra_velocity_shuffled_model_train = cebra_velocity_shuffled_model.transform(neural_train)
cebra_velocity_shuffled_model_test = cebra_velocity_shuffled_model.transform(neural_test)
cebra_acceleration_shuffled_model_train = cebra_acceleration_shuffled_model.transform(neural_train)
cebra_acceleration_shuffled_model_test = cebra_acceleration_shuffled_model.transform(neural_test)
cebra_stimulus_shuffled_model_train = cebra_stimulus_shuffled_model.transform(neural_train)
cebra_stimulus_shuffled_model_test = cebra_stimulus_shuffled_model.transform(neural_test)

# %% [markdown]
# #### Decode the labels from the embeddings
# We evaluate decoding performance of the different hypothesis models.

# %%
from Classes import decode

# %%
# decode position
cebra_behavior_model_decode = decode(cebra_behavior_model_train, cebra_behavior_model_test, label_train_position, label_test_position)
cebra_position_model_decode = decode(cebra_position_model_train, cebra_position_model_test, label_train_position, label_test_position)
cebra_distance_model_decode = decode(cebra_distance_model_train, cebra_distance_model_test, label_train_position, label_test_position)
cebra_velocity_model_decode = decode(cebra_velocity_model_train, cebra_velocity_model_test, label_train_position, label_test_position)
cebra_acceleration_model_decode = decode(cebra_acceleration_model_train, cebra_acceleration_model_test, label_train_position, label_test_position)
cebra_stimulus_model_decode = decode(cebra_stimulus_model_train, cebra_stimulus_model_test, label_train_position, label_test_position)
cebra_behavior_shuffled_model_decode = decode(cebra_behavior_shuffled_model_train, cebra_behavior_shuffled_model_test, label_train_position, label_test_position)
cebra_position_shuffled_model_decode = decode(cebra_position_shuffled_model_train, cebra_position_shuffled_model_test, label_train_position, label_test_position)
cebra_distance_shuffled_model_decode = decode(cebra_distance_shuffled_model_train, cebra_distance_shuffled_model_test, label_train_position, label_test_position)
cebra_velocity_shuffled_model_decode = decode(cebra_velocity_shuffled_model_train, cebra_velocity_shuffled_model_test, label_train_position, label_test_position)
cebra_acceleration_shuffled_model_decode = decode(cebra_acceleration_shuffled_model_train, cebra_acceleration_shuffled_model_test, label_train_position, label_test_position)
cebra_stimulus_shuffled_model_decode = decode(cebra_stimulus_shuffled_model_train, cebra_stimulus_shuffled_model_test, label_train_position, label_test_position)

# %%
cebra_behavior_model.decoded = cebra_behavior_model_decode
cebra_position_model.decoded = cebra_position_model_decode
cebra_distance_model.decoded = cebra_distance_model_decode
cebra_velocity_model.decoded = cebra_velocity_model_decode
cebra_acceleration_model.decoded = cebra_acceleration_model_decode
cebra_stimulus_model.decoded = cebra_stimulus_model_decode
cebra_behavior_shuffled_model.decoded = cebra_behavior_shuffled_model_decode
cebra_position_shuffled_model.decoded = cebra_position_shuffled_model_decode
cebra_distance_shuffled_model.decoded = cebra_distance_shuffled_model_decode
cebra_velocity_shuffled_model.decoded = cebra_velocity_shuffled_model_decode
cebra_acceleration_shuffled_model.decoded = cebra_acceleration_shuffled_model_decode
cebra_stimulus_shuffled_model.decoded = cebra_stimulus_shuffled_model_decode

# %%
task.id

# %%

decoded_model_original = [cebra_behavior_model, cebra_position_model, cebra_distance_model, cebra_velocity_model, cebra_acceleration_model, cebra_stimulus_model]
decoded_model_shuffled = [cebra_behavior_shuffled_model, cebra_position_shuffled_model, cebra_distance_shuffled_model, cebra_velocity_shuffled_model, cebra_acceleration_shuffled_model, cebra_stimulus_shuffled_model]
decoded_model_lists = [decoded_model_original, decoded_model_shuffled]
labels = ["".join(model.name.split("_train")).split("behavior_")[-1] for model in decoded_model_original+decoded_model_shuffled]

viz = Vizualizer(root_dir=root_dir)
task
viz.plot_decoding_score(title=f"{task.id} {task.behavior_metadata['stimulus_type']} Decoding of Position", decoded_model_lists=decoded_model_lists, labels=labels, figsize=(13, 5))


# %%
# decode individual labels
cebra_behavior_model_decode = decode(cebra_behavior_model_train, cebra_behavior_model_test, )
cebra_position_model_decode = decode(cebra_position_model_train, cebra_position_model_test, )
cebra_distance_model_decode = decode(cebra_distance_model_train, cebra_distance_model_test, )
cebra_velocity_model_decode = decode(cebra_velocity_model_train, cebra_velocity_model_test, )
cebra_acceleration_model_decode = decode(cebra_acceleration_model_train, cebra_acceleration_model_test, )
cebra_stimulus_model_decode = decode(cebra_stimulus_model_train, cebra_stimulus_model_test, )
cebra_behavior_shuffled_model_decode = decode(cebra_behavior_shuffled_model_train, cebra_behavior_shuffled_model_test, )
cebra_position_shuffled_model_decode = decode(cebra_position_shuffled_model_train, cebra_position_shuffled_model_test, )
cebra_distance_shuffled_model_decode = decode(cebra_distance_shuffled_model_train, cebra_distance_shuffled_model_test, )
cebra_velocity_shuffled_model_decode = decode(cebra_velocity_shuffled_model_train, cebra_velocity_shuffled_model_test, )
cebra_acceleration_shuffled_model_decode = decode(cebra_acceleration_shuffled_model_train, cebra_acceleration_shuffled_model_test, )
cebra_stimulus_shuffled_model_decode = decode(cebra_stimulus_shuffled_model_train, cebra_stimulus_shuffled_model_test, )

# %%
# decode all variables []
cebra_behavior_model_decode = decode(cebra_behavior_model_train, cebra_behavior_model_test, )
cebra_position_model_decode = decode(cebra_position_model_train, cebra_position_model_test, )
cebra_distance_model_decode = decode(cebra_distance_model_train, cebra_distance_model_test, )
cebra_velocity_model_decode = decode(cebra_velocity_model_train, cebra_velocity_model_test, )
cebra_acceleration_model_decode = decode(cebra_acceleration_model_train, cebra_acceleration_model_test, )
cebra_stimulus_model_decode = decode(cebra_stimulus_model_train, cebra_stimulus_model_test, )
cebra_behavior_shuffled_model_decode = decode(cebra_behavior_shuffled_model_train, cebra_behavior_shuffled_model_test, )
cebra_position_shuffled_model_decode = decode(cebra_position_shuffled_model_train, cebra_position_shuffled_model_test, )
cebra_distance_shuffled_model_decode = decode(cebra_distance_shuffled_model_train, cebra_distance_shuffled_model_test, )
cebra_velocity_shuffled_model_decode = decode(cebra_velocity_shuffled_model_train, cebra_velocity_shuffled_model_test, )
cebra_acceleration_shuffled_model_decode = decode(cebra_acceleration_shuffled_model_train, cebra_acceleration_shuffled_model_test, )
cebra_stimulus_shuffled_model_decode = decode(cebra_stimulus_shuffled_model_train, cebra_stimulus_shuffled_model_test, )

# %%
label_train_behavior, label_test_behavior
label_train_position, label_test_position
label_train_distance, label_test_distance
label_train_velocity, label_test_velocity
label_train_acceleration, label_test_acceleration
label_train_stimulus, label_test_stimulus
label_train_behavior, label_test_behavior
label_train_position, label_test_position
label_train_distance, label_test_distance
label_train_velocity, label_test_velocity
label_train_acceleration, label_test_acceleration
label_train_stimulus, label_test_stimulus

# %% [markdown]
# ### Decoding Intra-Session (Inter currently not possible because of different amount of neurons)

# %%
# settings paths
root_dir = "D:\\Experiments"
experimentalist = "Steffen"
paradigm = "Rigid_Plastic" #Intrinsic Imaging
animal_root_dir = os.path.join(root_dir, experimentalist, paradigm)

# setting data to use
behavior_datas = ["position", "distance", "velocity", "acceleration", "stimulus"]
wanted_animal_ids=["all"]
wanted_dates=["all"]
load_all = True
regenerate_plots=False

# cebra setting
regenerate=False
manifolds_pipeline="cebra"
# quick CPU run-time demo, you can drop `max_iterations` to 100-500; otherwise set to 5000.
model_settings = {"max_iterations" : 5000, "output_dimension" : 3}

animals = load_all_animals(animal_root_dir, 
            wanted_animal_ids=wanted_animal_ids, 
            wanted_dates=wanted_dates, 
            model_settings=model_settings,#) dont load data 
            behavior_datas=behavior_datas,
            regenerate_plots=regenerate_plots) # Load neural, behavior, data for all session parts.

# %% [markdown]
# #### train models

# %%
# Train CEBRA-Behavior models if not trained
regenerate = False
animal_id = "DON-004366"
session_date = "20210301"
task_id = "S1"
animal_id = "DON-004366"
animal = animals[animal_id]
middle_modles = {}
for session_name, session in animal.sessions.items():
    for task_name, task in session.tasks.items():
        neural_train, neural_test = split_data(task.neural.suite2p.data, 0.8)
        label_train_position, label_test_position = split_data(task.behavior.position.data, 0.8)

        task.trained_model = task.train_model(model_type="behavior", 
                                        shuffled=False,
                                        regenerate=False,
                                        model_name="position_train", 
                                        behavior_data=label_train_position, 
                                        neural_data=neural_train,
                                        manifolds_pipeline="cebra")

        task.trained_model.neural_train = neural_train
        task.trained_model.neural_test = neural_test
        task.trained_model.label_train_position = label_train_position
        task.trained_model.label_test_position = label_test_position

# %% [markdown]
# #### Create embeddings 
# - own train data and 
# - other test neural data

# %%

neural_label_data = {}
for animal, session, task in yield_animal_session_task(animals):
    neural_label_data[session.id] = {}
    for wanted_stimulus_type in ["A", "B", "A\'"]:
        neural_label_data[session.id][wanted_stimulus_type] = {}
        if wanted_stimulus_type == task.behavior_metadata["stimulus_type"]:
            neural_label_data[session.id][wanted_stimulus_type][session.date] = {"neural": task.neural.suite2p.data, 
                                                               "position_lables": task.behavior.position.data}

# %%
#Inter-Task embeddings
for session_name, session in animal.sessions.items():
    for task_name, task in session.tasks.items():
        neural_train, neural_test = split_data(task.neural.suite2p.data, 0.2)
        label_train_position, label_test_position = split_data(task.behavior.position.data, 0.2)
        
        embeddings = {task_name: task.trained_model.transform(task.trained_model.neural_test)}
        for task_name2, task2 in session.tasks.items():
            if not task_name==task_name2:
                embeddings[task_name2] = task.trained_model.transform(task2.neural.suite2p.data)

        task.middle_embeddings = embeddings
        #Inter-Session
#TODO: currently not possible because of channel differences (different amount of neurons)
#A Intra-Session

# %% [markdown]
# #### Decoding labels from embeddings
# We evaluate decoding performance of the different hypothesis models.

# %%
# decode position
for session_name, session in animal.sessions.items():
    for task_name, task in session.tasks.items():
        ground_truth_embedding = task.trained_model.transform(task.trained_model.neural_train)
        label_train_position = task.trained_model.label_train_position
        label_test_position = task.trained_model.label_test_position

        belt_type = task.behavior_metadata["stimulus_type"]
        decodings = {belt_type: decode(ground_truth_embedding, 
                                        task.middle_embeddings[task_name], 
                                        label_train_position, 
                                        label_test_position)}
        
        for task_name2, task2 in session.tasks.items():
            if not task_name==task_name2:
                label_test_position = task.trained_model.label_test_position
                belt_type = task2.behavior_metadata["stimulus_type"]
                decodings[belt_type] = decode(ground_truth_embedding, 
                                                task.middle_embeddings[task_name2], 
                                                label_train_position, 
                                                task2.neural.suite2p.data)
        task.decodings = decodings

# %%
#decoded_model_original_decoded = [cebra_middle_model_A_train_decoded, cebra_middle_model_B_decoded, cebra_middle_model_Aprime_decoded]
#decoded_lists = [decoded_model_original_decoded, decoded_model_shuffled]
#labels = ["A - A", "A - B", "A - A\'"]

for session_name, session in animal.sessions.items():
    decoded_lists = []
    labels = []
    for task_name, task in session.tasks.items():
        decoded_lists.append([])
        labels.append([])
        for embedded_to, decoding in task.decodings.items():
            decoded_lists[-1].append(decoding)
            labels[-1].append(f"{task.behavior_metadata['stimulus_type']} - {embedded_to}")

# %%
labels

# %%
#viz = Vizualizer(root_dir=root_dir)
#TODO: Is this working????????


for session_name, session in animal.sessions.items():
    decoded_lists = []
    labels = []
    for task_name, task in session.tasks.items():
        decoded_lists.append([])
        labels.append([])
        for embedded_to, decoding in task.decodings.items():
            decoded_lists[-1].append(decoding)
            labels[-1].append(f"{task.behavior_metadata['stimulus_type']} - {embedded_to}")

    # viz.plot_decoding_score(decoded_model_lists=decoded_model_lists, labels=labels, figsize=(13, 5))
    fig = plt.figure(figsize=(13, 5))
    fig.suptitle(f"{session.id} Intra-Session Behavioral Decoding of Position", fontsize=16)
    colors=["green", "red", "deepskyblue"]
    ax1 = plt.subplot(111)
    #ax2 = plt.subplot(122)

    overall_num = 0
    for color, docoded_model_list, labels_list in zip(colors, decoded_lists, labels):
        for num, (decoded, label) in enumerate(zip(docoded_model_list, labels_list)):
            color = "deepskyblue" if "A\'" == "".join(label[:2]) else "red" if "B" == label[0] else "green"
            alpha = 1 - ((1 / len(docoded_model_list)) * num / 1.3)
            x_pos = overall_num + num
            width = 0.4  # Width of the bars
            ax1.bar(
                x_pos, decoded[1], width=0.4, color=color, alpha=alpha, label = label
            )
            #ax2.scatter(
            #    middle_A_model.state_dict_["loss"][-1],
            #    decoded[1],
            #    s=50,
            #    c=color,
            #    alpha=alpha,
            #    label=label,
            #)
        overall_num = x_pos + 1

    x_label = "InfoNCE Loss (contrastive learning)"
    ylabel = "Median position error in cm"

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_ylabel(ylabel)
    labels = np.array(labels).flatten()
    label_pos = np.arange(len(np.array(labels).flatten()))
    ax1.set_xticks(label_pos)
    ax1.set_ylim([0, 130])
    ax1.set_xticklabels(labels, rotation=45, ha="right")

    #ax2.spines["top"].set_visible(False)
    #ax2.spines["right"].set_visible(False)
    #ax2.set_xlabel(x_label)
    #ax2.set_ylabel(ylabel)
    plt.legend()
    plt.show()

# %%
animal_id = "DON-004366"
session_date = "20210301"
for i, j in animals[animal_id].sessions[session_date].tasks.items():
    print(j.task, j.behavior_metadata["stimulus_type"])

# %% [markdown]
# ### Deconding Inter-Animals

# %%
#TODO: decoding inter-animals

# %% [markdown]
# ## Co-Homology and circular coordinates

# %% [markdown]
# # Testing

# %%
animal.sessions["20210228"].tasks["S1"].behavior.velocity.plot(regenerate_plot=True)

# %%
animal.sessions["20210228"].tasks["S1"].behavior.stimulus.figsize=(20,3)
animal.sessions["20210228"].tasks["S1"].behavior.stimulus.plot(regenerate_plot=True)

# %%
# settings paths
root_dir = "D:\\Experiments"
experimentalist = "Steffen"
paradigm = "Rigid_Plastic" #Intrinsic Imaging
animal_root_dir = os.path.join(root_dir, experimentalist, paradigm)

# setting data to use
behavior_datas = ["position", "distance", "velocity", "acceleration", "stimulus"]
wanted_animal_ids=["all"]
wanted_dates=["all"]
load_all = True
regenerate_plots=True

# cebra setting
regenerate=False
manifolds_pipeline="cebra"
# quick CPU run-time demo, you can drop `max_iterations` to 100-500; otherwise set to 5000.
model_settings = {"max_iterations" : 5000, "output_dimension" : 3}

animal_id = "DON-004366"
session_date = "20210228"
animal = Animal(animal_id, root_dir=animal_root_dir)
animal.add_session(date=session_date, 
                    model_settings=model_settings, 
                    behavior_data=behavior_datas)
for session_date, session in animal.sessions.items():
    session.add_all_tasks(["S1"])
    data = session.load_all_data(behavior_data=behavior_datas, regenerate_plots=regenerate_plots)

# animals["DON-004366"].sessions["20210228"].tasks["S1"].behavior.stimulus.plot()


