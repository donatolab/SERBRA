# Description
Fully automated batch loading, preprocessing, processing and plotting. This pipeline was initially developed for running [Cebra](https://github.com/AdaptiveMotorControlLab/CEBRA), but is now the base for running any analysis pipeline.

The pipelines implemented in this repository strongly depend on 
- [metadata variables defined in the yaml files](#yaml-metadata-variables) and
- [folder structure](#folder-structure)

# Setup Environment
The following steps are required to setup the environment for the analysis pipeline.

```bash
# Clone the repositories
git clone https://github.com/Dynamics-of-Neural-Systems-Lab/MARBLE.git
#git clone https://github.com/AdaptiveMotorControlLab/CEBRA.git
git clone https://github.com/Ch3fUlrich/CEBRA_own.git
conda create -n cebra python=3.10
conda activate cebra

# Manifold embedding without labels
cd MARBLE
pip install .
pip install pot umap ptu torch_scatter torch_sparse torch_geometric torch_cluster
cd ..

# this repository requirements
cd CEBRA_own
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

# Folder Structure
The folder structure is important for the pipeline to work. The following names used are eamples and the files are still accepted if they have different names as long as they have **specific information in the file names**. The Important information is marked using **[square brackets]**, which should not really be in the file names. The structure is as follows:

## Inscopix Data
```bash
───DON-019608
    │   DON-019608.yaml
    │   
    ├───20240126
    │   │   20240126.yaml
    │   │   
    │   ├───FS1 (task data)
    │   │   │
    │   │   ├───001P-I (Inscopix data)
    │   │   │   │     2021-10-31-17-58-12_video_sched_0_[binarized_traces]_V3_curated.[npz]
    │   │   │   │   
    │   │   │   └───tif
    │   │   │       │      
    │   │   │       └───suite2p
    │   │   │           │      
    │   │   │           └───plane0
    │   │   │               F.npy
    │   │   │               Fneu.npy
    │   │   │               iscell.npy
    │   │   │               ops.npy
    │   │   │               spks.npy
    │   │   │               stat.npy
    │   │   │               
    │   │   └───TR-BSL (Open field data)
    │   │            [DON-007021_20211031_TR-BSL_NS1]-ACQ(crop)DLC_resnet50_open_arena_white_floorSep8shuffle1_200000_fixedlocs_[locs.npy]
    │   │
    │   └───NS1 (task data)
    │       │   ...
    │      ...
    │
    └───20240127
        │   ...
       ...
```

## 2P Data
```bash
───DON-019608
    │   DON-019608.yaml
    │   
    ├───20240126
    │   │   20240126.yaml
    │   │   
    │   ├───002P-F (2P data from femtonics)
    │   │   │     [DON-019608_20240126_002P-F_S1-S2-ACQ.mesc]
    │   │   │   
    │   │   └───tif
    │   │       │      
    │   │       └───suite2p
    │   │           │      
    │   │           └───plane0
    │   │               F.npy
    │   │               Fneu.npy
    │   │               iscell.npy
    │   │               ops.npy
    │   │               spks.npy
    │   │               stat.npy
    │   ├───TRD-2P (treadmil movement data)
    │   │        DON-019608_202401261_TRD-2P_S1-ACQ.mat
    │   │        DON-019608_202401261_TRD-2P_S2-ACQ.mat
    │   │
    │   └───0000MC (optional: movies folder)
    │
    └───20240127
        │   ...
       ...
```

# Metadata Variables
## Python Variables
The pipeline has some important variables that need to be defined in the python script. These variables are important for defining what data to load and how to analyze it. 

### Required Variables
Table
| Variable     | Default | Description                        | Type    | Example    | Input Options              |
|--------------|--------|------------------------------------|---------|------------|----------------------------|
| `animal_root_dir` |  | Directory path for the animal's data | String | `"D:\\Experiments\\Nathalie\\OpenFieldDynamics"` | Built using `os.path.join` |
| `behavior_datas` | ["position"] | List of behavioral data types to analyze | List | ["position", "distance", "velocity", "acceleration", "stimulus", "moving"] | "position", "distance", "velocity", etc. |
| `wanted_animal_ids` |  | List of specific animal IDs to analyze | List | ["all"] | List of animal IDs or "all" |
| `wanted_dates` |  | List of specific experiment dates to include | List | ["all"] | List of dates or "all" |
| `load_all` |  | Flag to load all available data | Boolean | False | True/False |
| `regenerate_plots` |  | Flag to regenerate plots from the data | Boolean | True | True/False |
| `regenerate` | | Flag to regenerate existing data | Boolean | False | True/False |
| `manifolds_pipeline` | `"cebra"` | Name of the processing pipeline for manifolds | String | `"cebra"`  | `"cebra"`  |
| `model_settings` |  | Dictionary specifying model settings | Dict | see [Model Settings](#model-settings) | Dictionary with model settings |

### Model Settings
The model settings variable is a dictionary that specifies the settings for the model used in the analysis pipeline. The dictionary should contain the following keys:

| Key          | Default | Description                        | Type    | Example    | Input Options              |
|--------------|--------|------------------------------------|---------|------------|----------------------------|
| `place_cell` | see [Place Cell Settings](#place-cell-settings) | Place Cell settings | Dict | see [Place Cell Settings](#place-cell-settings) | Dictionary with place cell settings |
| `cebra`      | see [CEBRA Settings](#cebra-settings) | CEBRA settings | Dict | see [CEBRA Settings](#cebra-settings) | Dictionary with CEBRA settings |

#### Example Model Settings
```python
model_settings = {"place_cell": {"method" : "skaggs"}, 
                  "cebra": {"max_iterations" : 20000,  "output_dimension" : 3}} 
```

### Place Cell Settings
The place cell settings variable is a dictionary that specifies the settings for the place cell model used in the analysis pipeline. 

| Key          | Default | Description                        | Type    | Example    | Input Options              |
|--------------|--------|------------------------------------|---------|------------|----------------------------|
| `method`     | "skaggs" | Method for place cell analysis | String | "skaggs" | "skaggs" |

### CEBRA Settings
The CEBRA settings variable is a dictionary that specifies the settings for the CEBRA model used in the analysis pipeline.

| Key          | Default | Description                        | Type    | Example    | Input Options              |
|--------------|--------|------------------------------------|---------|------------|----------------------------|
| `model_architecture` | "offset10-model" | Name of the model architecture to use | String | "offset10-model" | 'offset10-model-mse', 'offset5-model', 'offset1-model', and others |
| `batch_size` | 512 | Batch size for the model training | Integer | 512 | Any integer |
| `learning_rate` | 3e-4 | Learning rate for the model training | Float | 3e-4 | Any float |
| `temperature` | 1 | Temperature for the model training | Float | 1 | Any float |
| `output_dimension` | 3 | Output dimension for the model training | Integer | 3 | Any integer |
| `max_iterations` | 5000 | Maximum number of iterations for the model training | Integer | 5000 | Any integer |
| `distance` | "cosine" | Distance metric for the model training | String | "cosine" | "cosine", "euclidean", "constant" |
| `conditional` | "time_delta" | Conditional for the model training | String | "time_delta" | "time_delta", "time", "delta" |
| `device` | "cuda_if_available" | Device for the model training | String | "cuda_if_available" | "cuda_if_available", "cuda", "cpu" |
| `verbose` | True | Verbosity for the model training | Boolean | True | True/False |
| `time_offsets` | 10 | Time offsets for the model training | Integer | 10 | Any integer |

## Yaml Metadata Variables
Metadata variables need to be named and valued exactly like defined or errors will occure. The variables are used for defining the analysis pipeline.

It is possible to add metadata variables additionally to the ones described in this document. Those can be used further for personalized analysis.

### Animal
The following metadata variables are used to describe the animal. The file should be saved in root directory of the animal folder. e.g. `.../DON-013199/DON-013199.yaml`

##### Required Metadata Variables
| Variable     | Default | Description                                   | Type    | Example    | Input Options              |
|--------------|--------|-----------------------------------------------|---------|------------|----------------------------|
| `animal_id`  | | Unique identifier for the animal              | String  | DON-013199 | Combination of letters and numbers |
| `dob`        | | Date of birth of the animal                   | String  | 20230419   | Date in 'YYYYMMDD' format |

### Session
The following metadata variables are used to describe the session. The file should be saved in the root directory of the session folder. e.g. `.../DON-013199/20230419/20230419.yaml`

#### Required Metadata Variables
| Variable     | Default |  Description                      | Type    | Example    | Input Options              |
|--------------|--| -----------------------------------------------|---------|------------|----------------------------|
| `date`       | |  The date when the experiment was conducted.   | String  | '20210228' | Date in 'YYYYMMDD' format  |
| `tasks_infos`| |  Metadata of tasks performed in the session.   | dict    | [Tasks section](#tasks) | Dictionary with task metadata |          

### Tasks
The following metadata variables are used to describe the tasks performed in the session. The metadata should be saved inside the session metadata file at the root directory of the session folder. e.g. `.../DON-013199/20230419/20230419.yaml`. More specifically, the task metadata should be saved inside the `tasks_infos` variable, which is a dictionary of further dictionaries. Each key in the `tasks_infos` dictionary should be the name of the task, and the value should be a dictionary containing the task metadata. The `tasks_infos` variable explanaiton is given in the [Session Metadata Variables](#session) section.

#### Required Metadata Variables
| Variable     | Default |  Description                      | Type    | Example    | Input Options              |
|--------------|--| -----------------------------------------------|---------|------------|----------------------------|
| `<any_name>` | |  Dictionary containing task metadata           | dict    | [Task metadata](#task-metadata) | Dictionary with task metadata |

### Task Metadata
The following metadata variables are used to describe the task. The metadata should be saved inside the `tasks_infos` variable in the session metadata file at the root directory of the session folder. e.g. `.../DON-013199/20230419/20230419.yaml`. More specifically, the task metadata should be saved inside a dictionary with the name of the task as the key (`<any_name>`). The `<any_name>` variable is explained in the [Tasks](#tasks) section.

#### Required Metadata Variables
| Variable     | Default |  Description                      | Type    | Example    | Input Options              |
|--------------|--| -----------------------------------------------|---------|------------|----------------------------|
|`neural_metadata`| |  Metadata of the neural data recorded during the task | dict | [Neural metadata](#neural-metadata) | Dictionary with neural metadata |
|`behavior_metadata`| |  Metadata of the behavior data recorded during the task | dict | [Behavior metadata](#behavior-metadata) | Dictionary with behavior metadata | 

### Neural Metadata
The following metadata variables are used to describe the neural data recorded during the task. The metadata should be saved inside the task metadata dictionary in the session metadata file at the root directory of the session folder. e.g. `.../DON-013199/20230419/20230419.yaml`. More specifically, the neural metadata should be saved inside a dictionary with the key `neural_metadata`. The `neural_metadata` variable is explained in the [Task Metadata](#task-metadata) section.

#### Required Metadata Variables
| Variable     | Default |  Description                      | Type    | Example    | Input Options              |
|--------------|--| -----------------------------------------------|---------|------------|----------------------------|
| `method`     | | The method used to record the neural data     | String  | `2P`       | `2P`, `1P`                 |
| `area`       | | The brain area where the neural data was recorded | String | `CA3`   | `CA3`, `CA1`       |
| `setup`      | | The setup used for neural recording           | String  | `femtonics` | `femtonics`, `thorlabs`, `inscopix`   |
| `preprocessing` | | The preprocessing pipeline used for neural data | String | `suite2p` | `suite2p`, `opexebo`    |
| `processing` | | The processing pipeline used for neural data  | String  | `cabincorr` | `cabincorr`  |

#### Optional Metadata Variables
| Variable     | Default |  Description                      | Type    | Example    | Input Options              |
|--------------|--| -----------------------------------------------|---------|------------|----------------------------|
| `fps`        | None | Frames per second of the neural recording. This value will be extracted if not present given     | float | 30.9      | Any float |

### Behavior Metadata
The following metadata variables are used to describe the behavior data recorded during the task. The metadata should be saved inside the task metadata dictionary in the session metadata file at the root directory of the session folder. e.g. `.../DON-013199/20230419/20230419.yaml`. More specifically, the behavior metadata should be saved inside a dictionary with the key `behavior_metadata`. The `behavior_metadata` variable is explained in the [Task Metadata](#task-metadata) section.

#### Always Required Metadata Variables
| Variable     | Default |  Description                      | Type    | Example    | Input Options              |
|--------------|--| -----------------------------------------------|---------|------------|----------------------------|
| `setup`     |  | Setup used for behavior recordings            | String  | see [1D](#1d-environment) or [2D](#2d-environment) | see [1D](#1d-environment) or [2D](#2d-environment)                  |
| `preprocessing` |  | Preprocessing pipeline for behavior data  | String  | see [1D](#1d-environment) or [2D](#2d-environment) | see [1D](#1d-environment) or [2D](#2d-environment)             |
| `environment_dimensions` |  | Dimensions of the environment    | see [1D](#1d-environment) or [2D](#2d-environment) | see [1D](#1d-environment) or [2D](#2d-environment)  | Size in meters             |

#### Optional Metadata Variables
| Variable     | Default |  Description                      | Type    | Example    | Input Options              |
|--|--------|------------------------------------|---------|------------|----------------------------|
| `fps` |  | Frames per second of the behavior recording. This value will be extracted if possible| float | 10000      | Any float  |

#### 1D Environment
This section describes the metadata variables used to describe the behavior data recorded in a 1D environment.

##### Required Metadata Variables
| Variable     | Default |  Description                      | Type    | Example    | Input Options              |
|--------------|--| -----------------------------------------------|---------|------------|----------------------------|
| `setup`      |  | Setup used for behavior recordings            | String  | `treadmill` | `treadmill`, `wheel`          |
| `binning_size` | 0.01 | Size of binning used for spatial data in meter  | Float   | 0.01       | Size in meters |
| `environment_dimensions` |  | Dimensions of the environment         | Float   | 1.8        | Size in meters             |


#### 2D Environment
| Variable     | Default |  Description                      | Type    | Example    | Input Options              |
|--------------|--|------------------------------------------|---------|------------|----------------------------|
| `setup`      |  | Setup used for behavior recordings            | String  | `openfield` | `openfield`          |


# Needs to be done
#TODO: Add more details to the following sections
..... create stimulus dict ? integrate camera??
| Variable     | Default |  Description                      | Type    | Example    | Input Options              |
|--------------|--| -----------------------------------------------|---------|------------|----------------------------|
| `stimulus_type` |  | Type (Name) of stimulus presented    | String  | `A`        | `A`, `B`, etc.             |
| `stimulus_sequence` |  | Sequence of stimulus presentations        | List    | `[1, 2, 3, 4, 5, 6]` | List of integers |
| `stimulus_dimensions` |  | Dimensions of each stimulus              | List    | `[0.3, 0.3, 0.3, 0.3, 0.3, 0.3]` | List of floats in meters |
| `stimulus_by` |  | Determines how stimulus is presented       | String  | `location` | `location`, `time`         |


...... how to create code for this????
| Variable     | Default |  Description                      | Type    | Example    | Input Options              |
|--------------|--| -----------------------------------------------|---------|------------|----------------------------|
| `preprocessing` | Preprocessing pipeline for behavior data    | String  | `rotary_encoder` | `rotary_encoder`, `wheel`, `cam` |
| `pixel_per_meter` | Number of pixels per meter in the video    | Float   | 1000       | Defaults to 1000           |
| `preprocessing` | Preprocessing pipeline for behavior data   | String  | see | `cam`, `tracking` |
| `radius`     |  | Radius of the wheel or treadmill used        | Float   | 0.05       | Size in meters (e.g., 0.05, 0.1) |
| `clicks_per_rotation` |  | Number of encoder clicks per wheel rotation | Integer | 500        | Defaults to 500            |




### Cam Metadata
The following metadata variables are used to describe the camera data recorded during the task. The metadata should be saved inside the task metadata dictionary in the session metadata file at the root directory of the session folder. e.g. `.../DON-013199/20230419/20230419.yaml`. More specifically, the camera metadata should be saved inside a dictionary with the key `cam`. The `cam` variable is explained in the