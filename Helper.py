import os
from pathlib import Path


# type hints
from typing import List, Union, Dict, Any, Tuple, Optional


# calculations
import numpy as np
from scipy.signal import butter, filtfilt  # , lfilter, freqz
import sklearn
from sklearn.metrics import (
    mutual_info_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    v_measure_score,
    fowlkes_mallows_score,
    rand_score,
    adjusted_rand_score,
)
from scipy.stats import wasserstein_distance, ks_2samp, entropy, energy_distance
from scipy.spatial.distance import cdist, mahalanobis
from scipy.spatial import distance, ConvexHull
from sklearn.covariance import EllipticEnvelope

# parallelization
from numba import njit, prange, jit

# import cupy as cp  # numpy using CUDA for faster computation
import yaml
import re

from datetime import datetime

# debugging
import logging
from time import time
from pyinstrument import Profiler

class GlobalLogger:
    def __init__(self, save_dir=""):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.configure_logger(save_dir=save_dir)

    def configure_logger(self, save_dir=""):
        self.logger.setLevel(logging.DEBUG)  # Set the desired level here

        # Create a file handler which logs even debug messages.
        log_file_path = os.path.join(save_dir, "Global_log.log")
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.DEBUG)

        # Create a console handler with a higher log level.
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        # Create a formatter and add it to the handlers.
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to the logger.
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def set_save_dir(self, save_dir):
        # Get the old handler
        old_handler = self.logger.handlers[0]

        # Create a new handler with the updated filename
        new_handler = logging.FileHandler(save_dir + "/Global_log.log")
        new_handler.setLevel(old_handler.level)
        new_handler.setFormatter(old_handler.formatter)

        # Remove the old handler and add the new one
        for old_handler in self.logger.handlers:
            self.logger.removeHandler(old_handler)
        self.logger.addHandler(new_handler)


global_logger_object = GlobalLogger()
global_logger = global_logger_object.logger

# global_logger_object.set_save_dir(os.path.join(os.getcwd(), "logs"))
# global_logger = global_logger_object.logger
# global_logger.info('check if log file is created in new working directory')
# global_logger
# print(global_logger_object.logger.handlers)
# global_logger_object.logger.handlers[0].baseFilename


def npz_loader(file_path, fname=None):
    data = None
    if os.path.exists(file_path):
        with np.load(file_path) as data:
            if fname:
                if fname in data.files:
                    data = data[fname]
                else:
                    print(f"File {fname} not found in {file_path}, returning all data")
    else:
        print(f"File {file_path} not found, returning None")
    return data


def yield_animal(animals_dict):
    for animal_id, animal in animals_dict.items():
        yield animal


def yield_animal_session(animals_dict):
    for animal in yield_animal(animals_dict):
        for session_date, session in animal.sessions.items():
            yield animal, session


def yield_animal_session_task(animals_dict):
    for animal, session in yield_animal_session(animals_dict):
        for task_id, task in session.tasks.items():
            yield animal, session, task


#### directory, file search
def dir_exist_create(directory):
    """
    Checks if a directory exists and creates it if it doesn't.

    Parameters:
    dir (str): Path of the directory to check and create.

    Returns:
    None
    """
    # Check if the directory exists
    directory = Path(directory)
    if not directory.exists():
        # Create the directory
        directory.mkdir()


def create_dirs(dirs):
    """
    Create a new directory hierarchy.

    Args:
        dirs (list): A list of strings representing the path to the new directory.

    Returns:
        str: The path to the newly created directory.
    """
    new_path = dirs[0]
    for path_part in dirs[1:]:
        new_path = os.path.join(new_path, path_part)
        dir_exist_create(new_path)
    return new_path


def get_directories(directory, regex_search=""):
    """
    This function returns a list of directories from the specified directory that match the regular expression search pattern.

    Parameters:
    directory (str): The directory path where to look for directories.
    regex_search (str, optional): The regular expression pattern to match. Default is an empty string, which means all directories are included.

    Returns:
    list: A list of directory names that match the regular expression search pattern.
    """
    directory = Path(directory)
    directories = None
    if directory.exists():
        directories = []
        for name in os.listdir(directory):
            if directory.joinpath(name).is_dir():
                if fname_match_search(name, regex_search=regex_search):
                    directories.append(name)
    else:
        global_logger.warning(f"Directory does not exist: {directory}")
        print(f"Directory does not exist: {directory}")
    return directories


def fname_match_search(name, ending="", regex_search=""):
    return len(re.findall(regex_search, name)) > 0 and name.endswith(ending)


def get_files(directory, ending="", regex_search=""):
    """
    This function returns a list of files from the specified directory that match the regular expression search pattern and have the specified file ending.

    Parameters:
    directory (str): The directory path where to look for files.
    ending (str, optional): The file ending to match. Default is '', which means all file endings are included.
    regex_search (str, optional): The regular expression pattern to match. Default is an empty string, which means all files are included.

    Returns:
    list: A list of file names that match the regular expression search pattern and have the specified file ending.
    """
    directory = Path(directory)
    files_list = None
    if directory.exists():
        files_list = []
        for name in os.listdir(directory):
            if directory.joinpath(name).is_file():
                if fname_match_search(name, ending, regex_search):
                    files_list.append(name)
    # else:
    #    global_logger.warning(f"Directory does not exist: {directory}")
    #    print(f"Directory does not exist: {directory}")
    return files_list


def search_file(directory, filename):
    """
    This function searches for a file with a given filename within a specified directory and its subdirectories.

    :param directory: The directory in which to search for the file.
    :type directory: str
    :param filename: The name of the file to search for.
    :type filename: str
    :return: The full path of the file if found, otherwise returns the string "Not found".
    :rtype: str
    """
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return Path(root).joinpath(filename)
    return None


def make_list_ifnot(string_or_list):
    if isinstance(string_or_list, np.ndarray):
        return string_or_list.tolist()
    return [string_or_list] if type(string_or_list) != list else string_or_list


def save_file_present(fpath, show_print=False):
    fpath = Path(fpath)
    file_present = False
    if fpath.exists():
        if show_print:
            global_logger.warning(f"File already present {fpath}")
            print(f"File already present {fpath}")
        file_present = True
    else:
        if show_print:
            global_logger.info(f"Saving {fpath.name} to {fpath}")
            print(f"Saving {fpath.name} to {fpath}")
    return file_present


# Math
def calc_cumsum_distances(positions, length, distance_threshold=0.30):
    """
    Calculates the cumulative sum of distances between positions along a track.

    Args:
    - positions (array-like): List or array of positions along the track.
    - length (numeric): Length of the track.
    - distance_threshold (numeric, optional): Threshold for considering a frame's distance change.
                                            Defaults to 30.

    Returns:
    - cumsum_distance (numpy.ndarray): Cumulative sum of distances between positions.
    """
    cumsum_distances = []
    cumsum_distance = 0
    old_position = positions[0]
    for position in positions[1:]:
        frame_distance = position - old_position
        if abs(frame_distance) < distance_threshold:
            cumsum_distance += frame_distance
        else:
            # check if mouse moves from end to start of the track
            if frame_distance < 0:
                cumsum_distance += frame_distance + length
            # check if mouse moves from start to end of the track
            else:
                cumsum_distance += frame_distance - length
        cumsum_distances.append(cumsum_distance)
        old_position = position
    return np.array(cumsum_distances)


def moving_average(data, window_size=30):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, "valid")


def smooth_array(data, window_size=5, axis=0):
    """
    Smooth a NumPy array along the specified axis using convolution.

    Args:
        data (np.ndarray): The input 2D array.
        window_size (int, optional): Size of the smoothing window (default is 5).
        axis (int, optional): Axis along which to apply smoothing (default is 0).

    Returns:
        np.ndarray: The smoothed array.
    """
    weights = np.ones(window_size) / window_size
    smoothed_data = np.apply_along_axis(
        lambda x: np.convolve(x, weights, "same"), axis, data
    )
    return smoothed_data


def butter_lowpass(cutoff, fs, order=5):
    """
    Design a lowpass Butterworth filter.

    Parameters:
        cutoff (float): The cutoff frequency in Hertz.
        fs (float): The sampling frequency in Hertz.
        order (int, optional): The filter order. Defaults to 5.

    Returns:
        b (array-like): Numerator (zeros) coefficients of the filter.
        a (array-like): Denominator (poles) coefficients of the filter.
    """

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Apply a lowpass Butterworth filter to the input data.

    Parameters:
        data (array-like): The input data to filter.
        cutoff (float): The cutoff frequency in Hertz.
        fs (float): The sampling frequency in Hertz.
        order (int, optional): The filter order. Defaults to 5.

    Returns:
        y (array-like): The filtered output data.
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def may_butter_lowpass_filter(data, smooth=True, cutoff=2, fps=None, order=2):
    data_smoothed = data
    if smooth:
        if not fps or fps == 0:
            print(f"WARNING: No fps provided smoothing not possible")
        else:
            for i in range(data.shape[1]):
                data_smoothed[:, i] = butter_lowpass_filter(
                    data[:, i], cutoff=cutoff, fs=fps, order=order
                )
    return data_smoothed

@njit(nopython=True)
def cosine_similarity(v1, v2):
    """
    A cosine similarity can be seen as the correlation between two vectors or point distributions.
    Returns:
        float: The cosine similarity between the two vectors.
    """   
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot_product / norm_product


def compute_mutual_information(
    true_labels: List[int], predicted_labels: List[int], metric="adjusted"
):
    """
    Mutual Information is a measure of the similarity between two labels of the same data.
    This metric is independent of the absolute values of the labels: a permutation of the class or
    cluster label values won’t change the score value in any way.
    Also note that the metric is not symmetric: switching true and predicted labels will return the same score value.

    parameters:
        true_labels: List[int] - true labels of the data
        predicted_labels: List[int] - predicted labels of the data
        metric: str - metric to use for the computation. Options: "mutual", "normalized", "adjusted"
            - "mutual": Mutual Information: Mutual Information (MI) is a measure of the similarity between two labels of the same data.

            - "normalized": Normalized Mutual Information: Normalized Mutual Information (NMI) is a normalization of the Mutual Information (MI)
            score to scale the results between 0 (no mutual information) and 1 (perfect correlation)

            - "adjusted": Adjusted Mutual Information: Adjusted Mutual Information is an adjustment of the Normalized Mutual Information (NMI)
            score to account for chance. -1 <= AMI <= 1.0 (1.0 is the perfect match, 0 is the random match, and -1 is the worst match)

            - "v": V-measure: The V-measure is the harmonic mean between homogeneity and completeness: v = (1 + beta) * homogeneity * completeness / (beta * homogeneity + completeness).

            - "fmi": Fowlkes-Mallows Index: The Fowlkes-Mallows index (FMI) is defined as the geometric mean between precision and recall: FMI = sqrt(precision * recall).

            - "rand": Rand Index: The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.

            - "adjrand": Adjusted Rand Index: The Adjusted Rand Index is the corrected-for-chance version of the Rand Index.

    returns:
        mi: float - mutual information score
    """
    if metric == "mutual":
        mi = mutual_info_score(true_labels.flatten(), predicted_labels.flatten())
    elif metric == "normalized":
        mi = normalized_mutual_info_score(
            true_labels.flatten(), predicted_labels.flatten()
        )
    elif metric == "ami":
        mi = adjusted_mutual_info_score(
            true_labels.flatten(), predicted_labels.flatten()
        )
    elif metric == "v":
        mi = v_measure_score(true_labels.flatten(), predicted_labels.flatten())
    elif metric == "fmi":
        # particularly useful for evaluating clustering algorithms where the pairwise clustering structure is important. It provides a balanced view by taking into account both precision and recall.
        mi = fowlkes_mallows_score(true_labels.flatten(), predicted_labels.flatten())
    elif metric == "rand":
        mi = rand_score(true_labels.flatten(), predicted_labels.flatten())
    elif metric == "adjrand":
        mi = adjusted_rand_score(true_labels.flatten(), predicted_labels.flatten())
    else:
        raise ValueError(
            f"Metric {metric} is not supported. Use 'mutual', 'normalized', or 'adjusted'."
        )
    return mi


def same_distribution(points1, points2):
    """
    Check if two distributions are the same.

    Parameters:
    - points1: array-like, first distribution
    - points2: array-like, second distribution

    Returns:
    - bool: True if the distributions are the same, False otherwise.
    """
    if points1.shape != points2.shape:
        return False
    return np.allclose(points1, points2)


def compare_distributions(
    points1, points2, metric="cosine", filter_outliers=True, parallel=True, neighbor_distance=None, out_det_method="density"
):
    """
    Compare two distributions using the specified metric.

    Parameters:
    - points1: array-like, first distribution
    - points2: array-like, second distribution
    - metric: str, the metric to use for comparison ('wasserstein', 'ks', 'chi2', 'kl', 'js', 'energy', 'mahalanobis')
        - 'wasserstein': Wasserstein Distance (Earth Mover's Distance) energy needed to move one distribution to the other
        - 'ks': Kolmogorov-Smirnov statistic for each dimension and take the maximum (typically used for 1D distributions)
        - 'chi2': Chi-Squared test (requires binned data, here we just compare histograms) - sum of squared differences between observed and expected frequencies
        - 'kl': Kullback-Leibler Divergence - measure of how one probability distribution diverges from a second, expected probability distribution
        - 'js': Jensen-Shannon Divergence - measure of similarity between two probability distributions
        - 'energy': Energy Distance - measure of the distance between two probability distributions (typically used for 1D distributions)
        - 'mahalanobis': Mahalanobis Distance - measure of the distance between two probability distributions
        - "cosine": Cosine Similarity only for mean vector of distribution- measure of the cosine of the angle between two non-zero vectors. This metric is equivalent to the Pearson correlation coefficient for normalized vectors.

    Wasserstein Distance (smaller = more similar)
    In your case, a Wasserstein distance of 3.099 indicates that it would take an average of 3.099 units of “work” (moving mass) to transform one distribution into the other.
    Commonly used for comparing probability distributions, especially in optimal transport problems.
    Breakage: When the distributions have disjoint support (no overlap), the Wasserstein distance becomes infinite.

    Kolmogorov-Smirnov Distance (smaller = more similar)
    Measures the maximum vertical difference between the cumulative distribution functions (CDFs) of the two distributions.
    A KS distance of 0.43 suggests that the two distributions differ significantly in terms of their shape or location.
    Breakage: It assumes continuous distributions and may not work well for discrete data.


    Chi-Squared Distance (smaller = more similar)
    Compares the observed and expected frequencies in a contingency table.
    Breakage: It is sensitive to sample size and may not work well for small datasets.

    Kullback-Leibler Divergence (smaller = more similar)
    Measures the relative entropy between two probability distributions.
    Breakage: Not symmetric; it can be infinite if one distribution has zero probability where the other doesn’t.

    Jensen-Shannon Divergence (smaller = more similar)
    A smoothed version of KL divergence that is symmetric and bounded.
    Breakage: Still sensitive to zero probabilities.

    Energy Distance (smaller = more similar)
    Measures the distance between two distributions using the energy distance.
    Breakage: It is sensitive to outliers and may not work well for skewed data and small datasets.

    Mahalanobis Distance (smaller = more similar)
    Mahalanobis distance accounts for correlations between variables.
    Breakage: It assumes that the data is normally distributed and may not work well for non-normal data.

    Returns:
    - The computed distance between the two distributions.
    """
    assert (
        points1.shape[1] == points2.shape[1]
    ), "Distributions must have the same number of dimensions"

    similarity_range = {
        "euclidean": {"lowest": np.inf,"highest": 0.0},
        "wasserstein": {"lowest": np.inf,"highest": 0.0},
        "ks": {"lowest": np.inf,"highest": 0.0},
        "chi2": {"lowest": np.inf,"highest": 0.0},
        "kullback-leibler": {"lowest": np.inf,"highest": 0.0},
        "jensen-shannon": {"lowest": np.inf,"highest": 0.0},
        "energy": {"lowest": np.inf,"highest": 0.0},
        "mahalanobis": {"lowest": np.inf,"highest": 0.0},
        "cosine": {"lowest": 0.0,"highest": 1.0},
        "overlap": {"lowest": 0.0,"highest": 1.0},
    }

    same_dist = False
    if same_distribution(points1, points2):
        same_dist = True

    # Filter out outliers from the distributions
    if filter_outliers and not same_dist:
        if parallel:
            points1 = filter_outlier_numba(points1, method=out_det_method, neighbor_distance=neighbor_distance)
            points2 = filter_outlier_numba(points2, method=out_det_method, neighbor_distance=neighbor_distance)
        else:
            points1 = filter_outlier(points1, method=out_det_method)
            points2 = filter_outlier(points2, method=out_det_method)
        
    if same_dist:
        for key in similarity_range.keys():
            if key in metric:
                if points1.shape[0] == 0 or points2.shape[0] == 0:
                    return None
                else:
                    return similarity_range[key]["highest"]
        # if metric_name has no implemented function
        raise NotImplementedError(f"Metric {metric} not implemented")

    if metric == "wasserstein":
        distances = [
            wasserstein_distance(points1[:, i], points2[:, i])
            for i in range(points1.shape[1])
        ]
        return np.sum(distances)

    elif metric == "kolmogorov-smirnov":
        # Calculate the Kolmogorov-Smirnov statistic for each dimension and take the maximum
        ks_statistics = [
            ks_2samp(points1[:, i], points2[:, i]).statistic
            for i in range(points1.shape[1])
        ]
        return np.max(ks_statistics)

    elif metric == "chi2":
        # Chi-Squared test (requires binned data, here we just compare histograms)
        hist1, _ = np.histogram(points1, bins=10, density=True)
        hist2, _ = np.histogram(points2, bins=10, density=True)
        return np.sum((hist1 - hist2) ** 2 / hist2 + 1e-10)

    elif metric == "kullback-leibler":
        # Kullback-Leibler Divergence
        hist1, _ = np.histogram(points1, bins=10, density=True)
        hist2, _ = np.histogram(points2, bins=10, density=True)
        return entropy(
            hist1 + 1e-10, hist2 + 1e-10
        )  # Adding a small value to avoid division by zero

    elif metric == "jensen-shannon":
        # Jensen-Shannon Divergence
        hist1, _ = np.histogram(points1, bins=10, density=True)
        hist2, _ = np.histogram(points2, bins=10, density=True)
        m = 0.5 * (hist1 + hist2)
        return 0.5 * (
            entropy(hist1 + 1e-10, m + 1e-10) + entropy(hist2 + 1e-10, m + 1e-10)
        )

    elif metric == "energy":
        # Energy Distance
        distances = [
            energy_distance(points1[:, i], points2[:, i])
            for i in range(points1.shape[1])
        ]
        return np.mean(distances)

    elif metric == "euclidean":
        return euclidean_distance_between_distributions(points1, points2)

    elif metric == "mahalanobis":
        return mahalanobis_distance_between_distributions(points1, points2)

    elif metric == "cosine":
        if points1.ndim == 2:
            v1 = np.mean(points1, axis=0)
        if points2.ndim == 2:
            v2 = np.mean(points2, axis=0)
        return cosine_similarity(v1, v2)

    elif "overlap" in metric:
        if "2d" in metric:
            # 1. Convert surface of 3D sphere to 2D plane
            points1 = sphere_to_plane(points1)
            points2 = sphere_to_plane(points2)
        else:
            raise ValueError(f"Calculation of overlap only possible in 2d space, name metric as overlap_2d")

        # 3. Calculate overlap
        hull1 = ConvexHull(points1)
        hull2 = ConvexHull(points2)

        overlap = area_of_intersection(hull1, hull2) / (
            hull1.area + hull2.area - area_of_intersection(hull1, hull2)
        )
        return overlap

    else:
        raise ValueError(f"Unsupported metric: {metric}")

@njit(nopython=True)
def get_covariance_matrix(data: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    cov = np.cov(data, rowvar=False)
    return regularized_covariance(cov, epsilon=epsilon)

@njit(nopython=True)
def regularized_covariance(cov_matrix, epsilon=1e-6):
    return cov_matrix + np.eye(cov_matrix.shape[0]) * epsilon

@njit(nopython=True)
def pca_numba(data, n_components=2):
    """
    Perform Principal Component Analysis (PCA) on the input data.

    Args:
    data (np.ndarray): Input data, shape (n_samples, n_features)
    n_components (int): Number of components to keep

    Returns: 
        data_pca (np.ndarray): Transformed data, shape (n_samples, n_components)
    """
    # Compute the covariance matrix
    cov = get_covariance_matrix(data)
    # Compute the eigenvectors and eigenvalues
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx[:n_components]]  # extract only the first n_components principal components
    data_pca = np.dot(data, eigvecs)
    return data_pca

@njit(nopython=True)
def sphere_to_plane(points: np.ndarray):
    """
    Convert 3D points on a sphere to 2D points on a plane.
    """
    points = points.astype(np.float64)
    # convert spherical coordinates to 2D plane
    phi = np.arctan2(points[:, 1], points[:, 0])
    theta = np.arccos(points[:, 2])
    points_2d = np.column_stack((phi, theta))
    return points_2d


@njit(nopython=True)
def compute_mean_and_cov(X):
    """
    Compute the mean vector and covariance matrix of the input data.

    Args:
    X (np.ndarray): Input data, shape (n_samples, n_features)

    Returns:
    tuple: (mean, cov)
        mean (np.ndarray): Mean vector, shape (n_features,)
        cov (np.ndarray): Covariance matrix, shape (n_features, n_features)
    """
    n, d = X.shape
    mean = np.zeros(d)
    for i in range(d):
        mean[i] = np.sum(X[:, i]) / n
    cov = np.zeros((d, d))
    for i in range(n):
        diff = X[i] - mean
        for j in range(d):
            for k in range(d):
                cov[j, k] += diff[j] * diff[k]
    cov /= n - 1
    return mean, cov


@njit(nopython=True)
def all_finite(arr):
    """Check if all elements in the array are finite."""
    for x in arr.flat:
        if not np.isfinite(x):
            return False
    return True


@njit(nopython=True)
def is_positive_definite(A):
    """Check if a matrix is positive definite."""
    try:
        np.linalg.cholesky(A)
        return True
    except:
        return False

@njit(nopython=True)
def mahalanobis_distance_between_distributions(points1, points2):
    # Mahalanobis Distance
    # Compute the covariance matrix for each distribution (can be estimated from data)
    cov_matrix1 = get_covariance_matrix(points1)
    cov_matrix2 = get_covariance_matrix(points2)
    if cov_matrix1.ndim < 2 or cov_matrix2.ndim < 2:
        return np.inf
    # Compute the inverse of the covariance matrices
    cov_inv1 = np.linalg.inv(cov_matrix1)
    cov_inv2 = np.linalg.inv(cov_matrix2)

    # Compute the Mahalanobis distance between the means of the dists
    mean1 = np.mean(points1, axis=0)
    mean2 = np.mean(points2, axis=0)
    return mahalanobis_distance(mean1, mean2, cov_inv1 + cov_inv2)

@njit(nopython=True)
def euclidean_distance_between_distributions(points1: np.ndarray, points2: np.ndarray) -> float:
    """
    Computes a distance metric between two distributions using pairwise Euclidean distances. 
    This is done by computing the pairwise Euclidean distances between all points in the two distributions,
    and then taking the mean of these distances. 

    Args:
        points1 (np.ndarray): A 2D array of shape (n_samples1, n_features) representing the first distribution.
        points2 (np.ndarray): A 2D array of shape (n_samples2, n_features) representing the second distribution.
        metric (str): A string specifying the metric to use. Only "euclidean" is supported in this implementation.

    Returns:
        float: A non-negative float representing the distance between the two distributions.
    """
    points1 = np.atleast_2d(points1)
    points2 = np.atleast_2d(points2)
    d1 = pairwise_euclidean_distance(points1, points1)
    d2 = pairwise_euclidean_distance(points2, points2)
    d3 = pairwise_euclidean_distance(points1, points2)
    return np.abs(np.mean(d1) + np.mean(d2) - 2 * np.mean(d3))

@njit(nopython=True)
def pairwise_euclidean_distance(points1, points2):
    """
    Compute the pairwise Euclidean distance between two sets of points.

    Args:
    points1 (np.ndarray): First set of points, shape (n_samples1, n_features)
    points2 (np.ndarray): Second set of points, shape (n_samples2, n_features)

    Returns:
    np.ndarray: Pairwise Euclidean distance matrix, shape (n_samples1, n_samples2)
    """
    n1, d = points1.shape
    n2 = points2.shape[0]
    distances = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            distances[i, j] = euclidean_distance(points1[i], points2[j])        
    return distances

@njit(nopython=True)
def euclidean_distance(x, y):
    """
    Compute the Euclidean distance between two points.

    Args:
    x (np.ndarray): First point, shape (n_features,)
    y (np.ndarray): Second point, shape (n_features,)

    Returns:
    float: Euclidean distance
    """
    diff = x - y
    distances = np.sqrt(np.dot(diff, diff))
    return distances

@njit(nopython=True)
def mahalanobis_distance(x, mean, inv_cov):
    """
    Compute the Mahalanobis distance between a point and the distribution.

    Args:
    x (np.ndarray): Input point, shape (n_features,)
    mean (np.ndarray): Mean of the distribution, shape (n_features,)
    inv_cov (np.ndarray): Inverse of the covariance matrix, shape (n_features, n_features)

    Returns:
    float: Mahalanobis distance
    """
    x = x.astype(np.float64)
    mean = mean.astype(np.float64)
    inv_cov = inv_cov.astype(np.float64)
    diff = x - mean
    distance = np.sqrt(diff.dot(inv_cov).dot(diff))
    return distance

@njit(nopython=True)
def compute_density(points, neighbor_distance, inv_cov=None):
    """
    Compute the density of each high-dimensional point in the distribution using a neighbor_distance and Mahalanobis distance.
    """
    n = points.shape[0]
    densities = np.zeros(n)

    for i in range(n):
        count = 0
        for j in range(n):
            if i != j:
                ## compute euclidean distance for low dimensional data
                if points.ndim <= 3:
                    dist = euclidean_distance(points[i], points[j])
                else:
                    dist = euclidean_distance(points[i], points[j])
                    # proper way to define neighbor_distance for outlier detection does not work with mahalanobis distance
                    # Compute Mahalanobis distance for high-dimensional data
                    #dist = mahalanobis_distance(points[i], points[j], inv_cov)

                if dist < neighbor_distance:
                    count += 1

        densities[i] = count
    
    densities = densities / np.max(densities)
    # remove nan
    densities = np.nan_to_num(densities, nan=0.0)

    if densities.sum() == 0:
        print("No inliers found")
    return densities

@njit(nopython=True)
def get_outlier_mask_numba(points, contamination=0.2, neighbor_distance=None, method="contamination"):
    n, d = points.shape

    # Remove any rows with NaN or inf
    mask_valid = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        mask_valid[i] = all_finite(points[i])
    valid_points = points[mask_valid]

    if len(valid_points) < d + 1:
        # Not enough valid points to compute covariance
        return np.ones(len(valid_points), dtype=np.bool_)


    # Create mask for non-outlier points
    mask_inliers = np.zeros(len(valid_points), dtype=np.bool_)
    if method=="contamination":
        # Compute mean and covariance
        mean, cov = compute_mean_and_cov(valid_points)

        # Check if covariance matrix is positive definite
        if not is_positive_definite(cov):
            # If not, add a small value to the diagonal
            cov += np.eye(d) * 1e-6

        try:
            inv_cov = np.linalg.inv(cov)
        except:
            # If inversion fails, return valid points
            return np.ones(len(valid_points), dtype=np.bool_)
        # Compute Mahalanobis distances
        distances = np.zeros(len(valid_points))
        for i in range(len(valid_points)):
            distances[i] = mahalanobis_distance(valid_points[i], mean, inv_cov)

        # Determine threshold based on contamination (percentage of outliers)
        threshold = np.percentile(distances, 100 * (1 - contamination))

        for i in range(len(valid_points)):
            mask_inliers[i] = distances[i] <= threshold

    elif method=="density":
        # Determine threshold based on density estimation
        # Compute densities for each point
        if neighbor_distance is None:
            raise ValueError("neighbor_distance must be specified for density-based outlier detection")
        densities = compute_density(valid_points, neighbor_distance)
        
        # Determine threshold based on contamination (percentage of lowest density points)
        threshold = np.percentile(densities, 100 * contamination)
    
        for i in range(len(valid_points)):
            mask_inliers[i] = densities[i] > threshold

    return mask_inliers


@njit(nopython=True)
def filter_outlier_numba(points, contamination=0.2, neighbor_distance=None, method="contamination"):
    """
    Filter out outliers from a set of points using a simplified Elliptic Envelope method.

    This function implements a basic form of outlier detection using Mahalanobis distances.
    It's optimized for speed using Numba's just-in-time compilation.

    Args:
    points (np.ndarray): Input points, shape (n_samples, n_features)
    contamination (float): The proportion of outliers in the data set. Default is 0.2.

    Returns:
    np.ndarray: Filtered points with outliers removed

    Note:
    - This is a simplified implementation and may not be as robust as sklearn's EllipticEnvelope
      for all datasets.
    - The first run will include compilation time; subsequent runs will be much faster.
    """
    n, d = points.shape
    mask_valid = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        mask_valid[i] = all_finite(points[i])
    valid_points = points[mask_valid]

    mask_inliers = get_outlier_mask_numba(points, contamination, neighbor_distance=neighbor_distance, method=method)
    return valid_points[mask_inliers]

def filter_outlier(points, contamination=0.2, only_mask=False, method="contamination"):
    """
    Filter out outliers from a 2D distribution using an Elliptic Envelope.
    The algorithm fits an ellipse to the data, trying to encompass the most concentrated 80% of the points (since we set contamination to 0.2).
    Points outside this ellipse are labeled as outliers.
        - It estimates the robust covariance of the dataset.
        - It computes the Mahalanobis distances of the points.
        - Points with a Mahalanobis distance above a certain threshold (determined by the contamination parameter) are considered outliers.
    """
    if method == "density":
        raise NotImplementedError(f"Method 'density' not implemented for not parallel filter_outlier")
    elif method == "contamination":
        outlier_detector = EllipticEnvelope(contamination=contamination, random_state=42)
    else:
        raise ValueError(f"Method {method} not supported")
    num_points = points.shape[0]
    mask = (
        outlier_detector.fit_predict(points) != -1
        if num_points > 2
        else np.ones(num_points, dtype=bool)
    )
    if only_mask:
        return mask
    return points[mask]


import numpy as np

@njit(nopython=True)
def numba_cross(a, b):
    """
    Numba implementation of np.cross function.

    Parameters:
    a (np.ndarray): First input array.
    b (np.ndarray): Second input array.

    Returns:
    np.ndarray: Cross product of a and b.

    Note:
    - This function assumes that a and b are 2D or 3D arrays.
    """
    if a.shape[-1] == 2 and b.shape[-1] == 2:
        result = np.zeros_like(a)
        result[0] = a[0] * b[1] - a[1] * b[0]
    elif a.shape[-1] == 3 and b.shape[-1] == 3:
        result = np.zeros_like(a)
        result[0] = a[1] * b[2] - a[2] * b[1]
        result[1] = a[2] * b[0] - a[0] * b[2]
        result[2] = a[0] * b[1] - a[1] * b[0]
    else:
        raise ValueError("Input arrays must be 2D or 3D.")
    return result


def intersect_segments(seg1, seg2):
    """
    Determine if two line segments intersect in any number of dimensions (2D or 3D) and return the point of intersection.

    Parameters:
    seg1 (list of tuples): The first line segment, represented by two points [(p1), (p2)].
    seg2 (list of tuples): The second line segment, represented by two points [(q1), (q2)].

    Returns:
    tuple or None: If the segments intersect, returns the coordinates of the
                   intersection point. If they don't intersect or are parallel, returns None.

    Note:
    - The function assumes that the input segments are valid (i.e., two distinct points for each segment).
    - Parallel or collinear segments are considered as non-intersecting unless their projections overlap.
    """
    # Convert points to numpy arrays for easier manipulation
    p1, p2 = np.array(seg1[0]), np.array(seg1[1])
    q1, q2 = np.array(seg2[0]), np.array(seg2[1])

    # Direction vectors for each segment
    r = p2 - p1
    s = q2 - q1

    # Check if the segments are parallel (determinant == 0)
    r_cross_s = numba_cross(r, s)
    r_cross_s_magnitude = np.linalg.norm(r_cross_s)

    if r_cross_s_magnitude == 0:
        return None  # Segments are parallel or collinear

    # Calculate the parameters t and u for the parametric equations
    qp = q1 - p1
    t = np.sum(numba_cross(qp, s)) / r_cross_s_magnitude
    u = np.sum(numba_cross(qp, r)) / r_cross_s_magnitude

    # Check if the intersection point lies on both line segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        # Calculate the point of intersection
        intersection_point = p1 + t * r
        return tuple(intersection_point)
    else:
        return None

def area_of_intersection(hull1, hull2):
    """
    Calculate the area of intersection between two convex hulls.

    This function finds the intersection points between the edges of two convex hulls
    and calculates the area of the resulting intersection polygon.

    Parameters:
    hull1 (scipy.spatial.ConvexHull): The first convex hull
    hull2 (scipy.spatial.ConvexHull): The second convex hull

    Returns:
    float: The area of intersection between the two convex hulls.
           Returns 0 if there's no intersection or if the intersection is a point or a line.

    Note:
    - This function assumes that the hulls are 2D convex hulls.
    - It uses the `intersect_segments` function to find edge intersections.
    - It also includes points from one hull that lie inside the other hull.
    - The function depends on an `in_hull` function (not provided) to check if a point is inside a hull.
    - It uses scipy's ConvexHull to calculate the final intersection area.

    Raises:
    Any exceptions raised by the ConvexHull constructor if the intersection points are invalid.
    """
    intersect_points = []
    for simplex1 in hull1.simplices:
        for simplex2 in hull2.simplices:
            intersect = intersect_segments(
                hull1.points[simplex1], hull2.points[simplex2]
            )
            if intersect is not None:
                intersect_points.append(intersect)

    # Also include points from hull1 that are inside hull2 and vice versa
    for point in hull1.points:
        if in_hull(point, hull2):
            intersect_points.append(point)
    for point in hull2.points:
        if in_hull(point, hull1):
            intersect_points.append(point)

    area = 0
    if len(intersect_points) > 2:
        try:
            if hull1.points.shape[1] == 2:
                intersection_hull = ConvexHull(intersect_points)
                area = intersection_hull.area
            elif hull1.points.shape[1] == 3:
                intersection_hull = ConvexHull(intersect_points)
                area = intersection_hull.volum
        except:
            area = 0
    return area


def in_hull(point, hull):
    # Check if a point is inside a convex hull
    return all((np.dot(eq[:-1], point) + eq[-1] <= 0) for eq in hull.equations)


def correlate_vectors(vectors: np.ndarray, metric="pearson"):
    """
    Matrix Multiplikation is used to calculate the correlation matrix.

    Equivalent to cosine measure for normalized vectors. Example code:
        similarity_matrix = np.zeros((vectors.shape[0], vectors.shape[0]))
        for i, v1 in enumerate(vectors):
            for j, v2 in enumerate(vectors):
                similarity_matrix[i, j] = cosine_similarity(v1, v2)

    Returns:
        np.ndarray: A matrix of correlations/cosine similarities between each pair of vectors.

    Example:
        vectors = np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]])
        correlation_matrix = correlation_matrix(vectors)
        print(correlation_matrix)
    """
    if metric == "pearson":
        correlation_matrix = np.corrcoef(vectors)
    elif metric == "cosine":
        # normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        # correlation_matrix = normalized_vectors @ normalized_vectors.T
        correlation_matrix = sklearn.metrics.pairwise.cosine_similarity(vectors)
    else:
        raise ValueError("metric must be 'pearson' or 'cosine'.")
    return correlation_matrix


## normalization
def normalize_01(vector, axis=1):
    axis = 0 if axis == 1 and len(vector.shape) == 1 else axis
    min_val = np.min(vector, axis=axis, keepdims=True)
    max_val = np.max(vector, axis=axis, keepdims=True)
    normalized_vector = (vector - min_val) / (max_val - min_val)
    return normalized_vector


# strings
def filter_strings_by_properties(
    strings, include_properties=None, exclude_properties=None
):
    """
    Filters a list of strings based on given properties.

    Args:
    - strings (list): List of strings to filter.
    - include_properties (list or str, optional): List of properties to include in filtered strings.
        Each property can be a string or a list of strings.
    - exclude_properties (list or str, optional): List of properties to exclude from filtered strings.
        Each property can be a string or a list of strings.

    Returns:
    - filtered_strings (list): List of strings filtered based on the given properties.
    """
    filtered_strings = []

    if include_properties:
        if not isinstance(include_properties, list):
            include_properties = [include_properties]
        elif isinstance(include_properties, list) and all(
            isinstance(prop, str) for prop in include_properties
        ):
            include_properties = [include_properties]

    if exclude_properties:
        if isinstance(exclude_properties, str):
            exclude_properties = [exclude_properties]
        elif isinstance(exclude_properties, list) and all(
            isinstance(prop, str) for prop in exclude_properties
        ):
            exclude_properties = [exclude_properties]

    for string in strings:
        if include_properties:
            # Check if any of the include properties lists are present in the string
            include_check = any(
                all(prop.lower() in string.lower() for prop in props)
                for props in include_properties
            )
        else:
            include_check = True

        if exclude_properties:
            # Check if any of the exclude properties lists are present in the string
            exclude_check = any(
                all(prop.lower() in string.lower() for prop in props)
                for props in exclude_properties
            )
        else:
            exclude_check = False

        # Only include the string if it matches include properties and does not match exclude properties
        if include_check and not exclude_check:
            filtered_strings.append(string)

    return filtered_strings


def check_correct_metadata(string_or_list, name_parts):
    name_parts = make_list_ifnot(name_parts)
    success = True
    if isinstance(string_or_list, Path):
        if not string_or_list.exists():
            success = False
            global_logger.error(f"No matching file found")
            print(f"No matching file found")
        else:
            string_or_list = string_or_list.name

    if success:
        for name_part in name_parts:
            if not name_part in string_or_list:
                success = False
            if not success:
                global_logger.error(
                    f"Metadata naming does not match Object Metadata: {string_or_list} != {name_parts}"
                )
                print(
                    f"Metadata naming does not match Object Metadata: {string_or_list} != {name_parts}"
                )
    return success


# array
def encode_categorical(data, categories=None, return_category_map=False):
    """
    Encode categorical data into numerical values.

    Parameters:
    -----------
    data : array-like
        The input data to encode. If the data is 1D, it will be treated as a single category.
    categories : list, optional, default=None
        The list of categories to encode. If None, the unique values in the data will be used.

    Returns:
    --------
    encoded_data : array-like
        The encoded data.
    """
    if categories is None:
        # 1D categories
        categories = np.unique(data, axis=0)
        if categories[0].shape[0] == 1:
            # Single-dimensional categories
            category_map = {category:category for category in categories}
            encoded_data = np.array([category_map[category] for category in data])
        elif categories[0].shape[0] > 1:
            # Multi-dimensional categories
            category_map = {tuple(category): i for i, category in enumerate(categories)}
            encoded_data = np.array([category_map[tuple(category)] for category in data])
    if return_category_map:
        return encoded_data, category_map
    else:
        return encoded_data

def is_rgba(value):
    if isinstance(value, list) or isinstance(value, np.ndarray):
        return all(is_single_rgba(v) for v in value)
    else:
        return is_single_rgba(value)


def is_single_rgba(val):
    if isinstance(val, tuple) or isinstance(val, list) or isinstance(val, np.ndarray):
        if len(val) == 4:
            return all(isinstance(c, (int, float)) and 0 <= c <= 1 for c in val)
    return False


def values_to_groups(
    values, points, filter_outliers=True, contamination=0.2, parallel=True
):
    """
    Group points based on corresponding values, with optional outlier filtering.

    Parameters:
    -----------
    values : array-like
        An array of values used for grouping the points. Each value corresponds to a point.

    points : array-like
        An array of points to be grouped. Each point corresponds to a value.

    filter_outliers : bool, optional, default=True
        If True, outliers in the points will be filtered out before grouping.
        Outliers are determined using the `filter_outlier` function.

    Returns:
    --------
    groups : dict
        A dictionary where keys are unique values from `values` and values are arrays of points
        corresponding to each unique value.
    """
    if len(values) != len(points):
        raise ValueError("Values and points must have the same length.")

    if filter_outliers:
        filter_mask = (
            get_outlier_mask_numba(points, contamination)
            if parallel
            else filter_outlier(points, contamination=contamination, only_mask=True)
        )
    else:
        filter_mask = None
    if filter_mask is not None:
        filtered_points = points[filter_mask]
        filtered_values = values[filter_mask]
    else:
        filtered_points = points
        filtered_values = values

    groups = {}
    unique_labels = np.unique(filtered_values, axis=0)
    for label in unique_labels:
        matching_values = np.all(filtered_values == label, axis=1)
        groups[tuple(label)] = filtered_points[matching_values]

    return groups


def is_integer(array: np.ndarray) -> bool:
    return np.issubdtype(array.dtype, np.integer)


def is_floating(array: np.ndarray) -> bool:
    return np.issubdtype(array.dtype, np.floating)

def is_list_of_ndarrays(variable):
    # Check if the variable is a list
    if isinstance(variable, list):
        # Check if every element in the list is an instance of np.ndarray
        return all(isinstance(element, np.ndarray) for element in variable)
    return False

def force_1_dim_larger(data: np.ndarray):
    if len(data.shape) == 1 or data.shape[0] < data.shape[1]:
        # global_logger.warning(
        #    f"Data is probably transposed. Needed Shape [Time, cells] Transposing..."
        # )
        # print("Data is probably transposed. Needed Shape [Time, cells] Transposing...")
        return data.T  # Transpose the array if the condition is met
    else:
        return data  # Return the original array if the condition is not met


def force_equal_dimensions(array1: np.ndarray, array2: np.ndarray):
    """
    Force two arrays to have the same dimensions.
    By cropping the larger array to the size of the smaller array.
    """
    if is_list_of_ndarrays(array1) and is_list_of_ndarrays(array2):
        for i in range(len(array1)):
            array1[i], array2[i] = force_equal_dimensions(array1[i], array2[i])
        return array1, array2
    elif isinstance(array1, np.ndarray) and isinstance(array2, np.ndarray):
        shape_0_diff = array1.shape[0] - array2.shape[0]
        if shape_0_diff > 0:
            array1 = array1[:-shape_0_diff]
        elif shape_0_diff < 0:
            array2 = array2[:shape_0_diff]
    return array1, array2


def sort_arr_by(arr, axis=1, sorting_indices=None):
    """
    if no sorting indices are given array is sorted by maximum value of 2d array
    """
    if sorting_indices is not None:
        indices = sorting_indices
    else:
        maxes = np.argmax(arr, axis=axis)
        indices = np.argsort(maxes)
    sorted_arr = arr[indices]
    return sorted_arr, indices


def split_array_by_zscore(array, zscore, threshold=2.5):
    above_threshold = np.where(zscore >= threshold)[0]
    below_threshold = np.where(zscore < threshold)[0]
    return array[above_threshold], array[below_threshold]


def bin_array_1d(
    arr: List[float],
    bin_size: float,
    min_bin: float = None,
    max_bin: float = None,
    restrict_to_range=True,
):
    """
    Bin a 1D array of floats based on a given bin size.

    Parameters:
    arr (numpy.ndarray): Input array of floats.
    bin_size (float): Size of each bin.
    min_bin (float, optional): Minimum bin value. If None, use min value of the array.
    max_bin (float, optional): Maximum bin value. If None, use max value of the array.

    Returns:
    numpy.ndarray: Binned array, starting at bin 0 to n-1.
    """
    min_bin = min_bin or np.min(arr)
    max_bin = max_bin or np.max(arr)

    # Calculate the number of bins
    num_bins = int(np.ceil((max_bin - min_bin) / bin_size))

    # Calculate the edges of the bins
    bin_edges = np.linspace(min_bin, min_bin + num_bins * bin_size, num_bins + 1)

    # Bin the array
    binned_array = np.digitize(arr, bin_edges) - 1

    if restrict_to_range:
        binned_array = np.clip(binned_array, 0, num_bins - 1)

    return binned_array


def bin_array(
    arr: List[float],
    bin_size: List[float],
    min_bin: List[float] = None,
    max_bin: List[float] = None,
):
    """
    Bin an array of floats based on a given bin size.

    Parameters:
    arr (numpy.ndarray): Input array of floats. Can be 1D or 2D.
    bin_size (float): Size of each bin.
    min_bin (float, optional): Minimum bin value. If None, use min value of the array.
    max_bin (float, optional): Maximum bin value. If None, use max value of the array.

    Returns:
    numpy.ndarray: Binned array, starting at bin 0 to n-1.
    """
    min_bin = make_list_ifnot(min_bin)
    max_bin = make_list_ifnot(max_bin)
    bin_size = make_list_ifnot(bin_size)
    np_arr = np.array(arr)
    if np_arr.ndim == 1:
        np_arr = np_arr.reshape(-1, 1)
    else:
        num_dims = min(arr.shape)
        if len(min_bin) == 1:
            min_bin = [min_bin[0]] * num_dims
        if len(max_bin) == 1:
            max_bin = [max_bin[0]] * num_dims
        if len(bin_size) == 1:
            bin_size = [bin_size[0]] * num_dims

    # Transform the array if 1st dimension is smaller than 2nd dimension
    if np_arr.shape[0] < np_arr.shape[1]:
        np_arr = np_arr.T

    # Bin each dimension of the array
    binned_array = np.zeros_like(np_arr, dtype=int)
    for i in range(np_arr.shape[1]):
        binned_array[:, i] = bin_array_1d(
            np_arr[:, i], bin_size[i], min_bin[i], max_bin[i]
        )

    return binned_array


def fill_continuous_array(data_array, fps, time_gap):
    frame_gap = fps * time_gap
    # Find indices where values change
    value_changes = np.where(np.abs(np.diff(data_array)) > np.finfo(float).eps)[0] + 1
    filled_array = data_array.copy()
    # Fill gaps after continuous 2 seconds of the same value
    for i in range(len(value_changes) - 1):
        start = value_changes[i]
        end = value_changes[i + 1]
        if (
            end - start <= frame_gap
            and filled_array[start - 1] == filled_array[end + 1]
        ):
            filled_array[start:end] = [filled_array[start - 1]] * (end - start)
    return filled_array


def convert_values_to_binary(vec: np.ndarray, threshold=2.5):
    smaller_idx = np.where(vec < threshold)
    bigger_idx = np.where(vec > threshold)
    vec[smaller_idx] = 0
    vec[bigger_idx] = 1
    vec = vec.astype(int)
    return vec


def per_frame_to_per_second(data, fps=None):
    if not fps:
        data = data
        print(f"WARNING: No fps provided. Output is value per frame")
    else:
        data *= fps
    return data


def fill_vector(vector, indices, fill_value=np.nan):
    filled_vector = vector.copy()
    filled_vector[indices] = fill_value
    return filled_vector


def fill_matrix(matrix, indices, fill_value=np.nan, axis=0):
    filled_matrix = matrix.copy()
    if axis == 0:
        filled_matrix[indices, :] = fill_value
    elif axis == 1:
        filled_matrix[:, indices] = fill_value
    else:
        raise ValueError("Axis must be 0 or 1. 3D not supported.")
    return filled_matrix


def fill(matrix, indices, fill_value=np.nan):
    if len(matrix.shape) == 1:
        return fill_vector(matrix, indices, fill_value)
    else:
        return fill_matrix(matrix, indices, fill_value)


def fill_inputs(inputs: dict, indices: np.ndarray, fill_value=np.nan):
    filtered_inputs = inputs.copy()
    for key, value in filtered_inputs.items():
        if len(value.shape) == 1:
            filtered_inputs[key] = fill_vector(value, indices, fill_value=fill_value)
        else:
            filtered_inputs[key] = fill_matrix(value, indices, fill_value=fill_value)
    return filtered_inputs


def get_top_percentile_indices(vector, percentile=5, indices_smaller=True):
    cutoff = np.percentile(vector, 100 - percentile)
    if indices_smaller:
        indices = np.where(vector < cutoff)[0]
    else:
        indices = np.where(vector > cutoff)[0]
    return indices


# dict
def init_dict_in_dict(dict, key):
    if key not in dict:
        dict[key] = {}
    return dict[key]


def add_to_list_in_dict(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)


def dict_value_keylist(dict, keylist):
    for key in keylist:
        dict = dict[key]
    return dict

def is_dict_of_dicts(dict):
    return all(type(value) == type(dict) for value in dict.values())

def equal_number_entries(embeddings, embedding_labels):
    # check if number labels is the same as number of embedded frames
    for key, labels in embedding_labels.items():
        if isinstance(embeddings, np.ndarray):
            embeddings = {"dummy_name": embeddings}
        for embedding_name, embedding in embeddings.items():
            if labels.shape[0] != embedding.shape[0]:
                return False
    return True

def group_by_binned_data(
    binned_data: np.ndarray,
    category_map: dict = None,
    data: np.ndarray = None,
    as_array: bool = False,
    group_values: str = "raw",
    max_bin=None,
):
    """
    Group data based on binned data.
    If category_map is provided, the binned data will be used as keys to group the data. Mostly used to map 1D discrete labels to multi dimensional position data.

    Args:
        - binned_data: The binned data used for grouping.
        - category_map: A dictionary mapping binned data to categories.
        - group_by: The method used for grouping the data.
            - 'symmetric_matrix' is in the group by options. It is filtering in 2D dimensions
            - 'raw': Return the raw data.
            - 'mean': Return the mean of the grouped data.
            - 'count': Return the count of the grouped data.

    """
    if data is None and group_values != "count":
        raise ValueError("Data needed for grouping.")

    if group_values != "count":
        data, binned_data = force_equal_dimensions(data, binned_data)
    # Define bins and counts
    if category_map is None:
        bins = unique_locations = np.unique(binned_data, axis=0, return_counts=False)
        uncoded_binned_data = binned_data
    else:
        unique_locations = list(category_map.keys())
        bins = list(category_map.keys())
        uncoded_binned_data = uncode_categories(binned_data, category_map)

    max_bin = np.max(bins, axis=0) + 1 if max_bin is None else max_bin

    # create groups
    groups = {}
    for unique_loc in unique_locations:
        idx = np.all(
            force_1_dim_larger(np.atleast_2d(uncoded_binned_data == unique_loc)), axis=1
        )
        # calculation is for cells x cells matrix if symmetric_matrix
        if group_values == "count":
            values_at_bin = np.sum(idx)
        elif group_values == "count_norm":
            values_at_bin = np.sum(idx) / len(uncoded_binned_data)
        else:
            filtered_data = (
                data[idx][:, idx] if "symmetric_matrix" in group_values else data[idx]
            )
            if "raw" in group_values:
                values_at_bin = filtered_data
            elif "mean" in group_values:
                values_at_bin = np.mean(filtered_data)

        groups[unique_loc] = values_at_bin

    if as_array:
        groups_array = np.zeros(max_bin.astype(int))
        for coordinates, values in groups.items():
            groups_array[coordinates] = values
        return groups_array, bins
    return groups, bins
 
def uncode_categories(data: Union[dict, np.ndarray, list], category_map: dict):
    """
    Converts a dictionary with categorical keys to a dictionary with tuple keys.
    """
    rev_category_map = {v: k for k, v in category_map.items()}        
    if isinstance(data, dict):
        uncoded = {rev_category_map[key] : value for key, value in data.items()}
    elif isinstance(data, np.ndarray) or isinstance(data, list):
        uncoded = np.array([rev_category_map[encoded_category] for encoded_category in data.flatten()])
    return uncoded


def filter_dict_by_properties(
    dictionary,
    include_properties: List[List[str]] = None,  # or [str] or str
    exclude_properties: List[List[str]] = None,  # or [str] or str):
):
    """
    Filter a dictionary with descriptive keys based on given properties.
    """
    dict_keys = dictionary.keys()
    if include_properties or exclude_properties:
        dict_keys = filter_strings_by_properties(
            dict_keys,
            include_properties=include_properties,
            exclude_properties=exclude_properties,
        )
    filtered_dict = {key: dictionary[key] for key in dict_keys}
    return filtered_dict

def wanted_object(obj, wanted_keys_values):
    """
    Check if an object has the wanted keys and values.
    """
    if isinstance(obj, dict):
        dictionary = obj 
    else:
        if hasattr(obj, "__dict__"):
            dictionary = obj.__dict__
        else:
            raise ValueError("Object has no dictionary")
        
    for key, value in wanted_keys_values.items():
        if key not in dictionary:
            return False
        else:
            obj_val = dictionary[key]
            if isinstance(value, list):
                if obj_val not in value:
                    return False
            elif obj_val != value:
                return False
    return True

def sort_dict(dictionary, reverse=False):
    return dict(sorted(dictionary.items(), reverse=reverse))


def load_yaml(path, mode="r"):
    if not Path(path).exists():
        # check for .yml file
        if Path(path).with_suffix(".yml").exists():
            path = Path(path).with_suffix(".yml")
        else:
            raise FileNotFoundError(f"No yaml file found: {path}")
    with open(path, mode) as file:
        dictionary = yaml.safe_load(file)
    return dictionary


def get_str_from_dict(dictionary, keys):
    present_keys = dictionary.keys()
    string = ""
    for variable in keys:
        if variable in present_keys:
            string += f" {dictionary[variable]}"
    return string


def keys_missing(dictionary, keys):
    present_keys = dictionary.keys()
    missing = []
    for key in keys:
        if key not in present_keys:
            missing.append(key)
    if len(missing) > 0:
        return missing
    return False


def check_needed_keys(metadata, needed_attributes):
    missing = keys_missing(metadata, needed_attributes)
    if missing:
        raise NameError(f"Missing metadata for: {missing} not defined")


def add_missing_keys(metadata, needed_attributes, fill_value=None):
    missing = keys_missing(metadata, needed_attributes)
    if missing:
        for key in missing:
            metadata[key] = fill_value
    return metadata


def traverse_dicts(dicts, keys=None):
    """
    Traverse a nested dictionary and yield each key-value pair with a key list.
    """
    for key, value in dicts.items():
        keys_list = keys + [key] if keys else [key]
        if isinstance(value, dict):
            yield from traverse_dicts(value, keys=keys_list)
        else:
            yield keys_list, value


def delete_nested_key(d, keys):
    """
    Deletes a key from a nested dictionary.

    Parameters:
    d (dict): The dictionary to modify.
    keys (list): A list of keys to traverse the dictionary.

    Example:
    delete_nested_key(my_dict, ['a', 'b', 'c']) will delete my_dict['a']['b']['c']
    """
    for key in keys[:-1]:
        d = d[key]
    del d[keys[-1]]


# class
def define_cls_attributes(cls_object, attributes_dict, override=False):
    """
    Defines attributes for a class object based on a dictionary.

    Args:
        cls_object (object): The class object to define attributes for.
        attributes_dict (dict): A dictionary containing attribute names as keys and their corresponding values.
        override (bool, optional): If True, existing attributes will be overridden. Defaults to False.

    Returns:
        object: The modified class object with the defined attributes.
    """
    for key, value in attributes_dict.items():
        if key not in cls_object.__dict__.keys() or override:
            setattr(cls_object, key, value)
    return cls_object


def copy_attributes_to_object(
    propertie_name_list,
    set_object,
    get_object=None,
    propertie_values=None,
    override=True,
):
    """
    Set attributes of a target object based on a list of property names and values.

    This function allows you to set attributes on a target object (the 'set_object') based on a list of
    property names provided in 'propertie_name_list' and corresponding values. You can specify these
    values directly through 'propertie_values' or retrieve them from another object ('get_object').
    If 'propertie_values' is not provided, this function will attempt to fetch the values from the
    'get_object' using the specified property names.

    Args:
        propertie_name_list (list): A list of property names to set on the 'set_object.'
        set_object (object): The target object for attribute assignment.
        get_object (object, optional): The source object to retrieve property values from. Default is None.
        propertie_values (list, optional): A list of values corresponding to the property names.
            Default is None.

    Returns:
        None

    Raises:
        ValueError: If the number of properties in 'propertie_name_list' does not match the number of values
            provided in 'propertie_values' (if 'propertie_values' is specified).

    Example Usage:
        # Example 1: Set attributes directly with values
        copy_attributes_to_object(["attr1", "attr2"], my_object, propertie_values=[value1, value2])

        # Example 2: Retrieve attribute values from another object
        copy_attributes_to_object(["attr1", "attr2"], my_object, get_object=source_object)
    """
    propertie_name_list = list(propertie_name_list)
    if propertie_values:
        propertie_values = list(propertie_values)
        if len(propertie_values) != len(propertie_name_list):
            raise ValueError(
                f"Number of properties does not match given propertie values: {len(propertie_name_list)} != {len(propertie_values)}"
            )
    elif get_object:
        propertie_values = []
        for propertie in propertie_name_list:
            if propertie in get_object.__dict__.keys():
                propertie_values.append(getattr(get_object, propertie))
            else:
                propertie_values.append(None)

    propertie_values = (
        propertie_values if propertie_values else [None] * len(propertie_name_list)
    )
    for propertie, value in zip(propertie_name_list, propertie_values):
        if propertie in set_object.__dict__.keys() and not override:
            continue
        setattr(set_object, propertie, value)


def attributes_present(attribute_names, object):
    for attribute_name in attribute_names:
        defined_variable = getattr(object, attribute_name)
        if defined_variable == None:
            return attribute_name
    return True


def set_attributes_check_presents(
    propertie_name_list,
    set_object,
    get_object=None,
    propertie_values=None,
    needed_attributes=None,
):
    copy_attributes_to_object(
        propertie_name_list=propertie_name_list,
        set_object=set_object,
        get_object=get_object,
        propertie_values=propertie_values,
    )

    if needed_attributes:
        present = attributes_present(needed_attributes, set_object)
        if present != True:
            raise NameError(
                f"Variable {present} is not defined in yaml file for {set_object.id}"
            )


# matlab
def load_matlab_metadata(mat_fpath):
    import scipy.io as sio

    mat = sio.loadmat(mat_fpath)
    # get the metadata
    dtypes = mat["metadata"][0][0].dtype.fields
    values = mat["metadata"][0][0]
    for attribute, value in zip(dtypes, values):
        value = value[0] if len(value) > 0 else " "
        print(f"{attribute:>12}: {str(value)}")


# date time
def num_to_date(date_string):
    if type(date_string) != str:
        date_string = str(date_string)
    date = datetime.strptime(date_string, "%Y%m%d")
    return date


def range_to_seconds(end: int, fps: float, start: int = 0):
    return np.arange(start, end) / fps


def second_to_time(second: float, striptime_format: str = "%M:%S"):
    return datetime.fromtimestamp(second).strftime("%M:%S")


def seconds_to_times(seconds: List[float]):
    if seconds[-1] > 60 * 60:  # if more than 1 hour
        times = np.full(len(seconds), "00:00:00")
        striptime_format = "%H:%M:%S"
    else:
        times = np.full(len(seconds), "00:00")
        striptime_format = "%M:%S"

    for i, sec in enumerate(seconds):
        times[i] = second_to_time(sec, striptime_format)
    return times


def range_to_times(end: int, fps: float, start: int = 0):
    seconds = range_to_seconds(end, fps, start)
    times = seconds_to_times(seconds)
    return times


def range_to_times_xlables_xpos(
    end: int, fps: float, start: int = 0, seconds_per_label: float = 30
):
    seconds = range_to_seconds(end, fps, start)
    if seconds[-1] > 60 * 60:  # if more than 1 hour
        striptime_format = "%H:%M:%S"
    else:
        striptime_format = "%M:%S"
    minutes = seconds_to_times(seconds)
    xticks = [minutes[0]]
    xpos = [0]
    for i, minute in enumerate(minutes[1:]):
        time_before = datetime.strptime(xticks[-1], striptime_format)
        time_current = datetime.strptime(minute, striptime_format)
        if (time_current - time_before).seconds > seconds_per_label:
            xticks.append(minute)
            xpos.append(i + 1)
    return xticks, xpos


# decorator functions
def timer(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func

def profile_function(func):
    def wrap_func(*args, **kwargs):
        profiler = Profiler()
        profiler.start()
        result = func(*args, **kwargs)
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        # Save the output to an HTML file
        with open('profile_report.html', 'w') as f:
            f.write(profiler.output_html())
        return result
    return wrap_func