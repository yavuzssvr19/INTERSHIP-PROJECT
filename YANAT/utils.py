from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Generator, List, Optional, Union
import pandas as pd
import _pickle as pk # type: ignore
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from copy import deepcopy
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator

def find_density(adjacency_matrix: np.ndarray) -> float:
    """Finds the density of the given adjacency matrix. It's the ratio of the number of edges to the number of possible edges.

    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix of the network.

    Returns:
        float: The density of the network.
    """
    return np.where(adjacency_matrix != 0, 1, 0).sum() / adjacency_matrix.shape[0] ** 2


def minmax_normalize(
    data: Union[pd.DataFrame, np.ndarray],
) -> Union[pd.DataFrame, np.ndarray]:
    """Normalizes data between 0 and 1.

    Args:
        data (Union[pd.DataFrame, np.ndarray]): Data to be normalized. Can be a DataFrame or an np array but in both cases it should be at most 2D.

    Returns:
        Union[pd.DataFrame, np.ndarray]: Normalized data with the same shape as the input.
    """
    return (data - data.min()) / (data.max() - data.min())


def log_normalize(adjacency_matrix: np.ndarray) -> np.ndarray:
    """Returns the logarithm of the data (adjacency_matrix) but also takes care of the infinit values.

    Args:
        adjacency_matrix (np.ndarray): Adjacency matrix of the network. Technically can be any matrix but I did it for the adjacency matrices.

    Returns:
        np.ndarray: Normalized data with the same shape as the input.
    """
    return np.nan_to_num(np.log(adjacency_matrix), neginf=0, posinf=0)


def log_minmax_normalize(adjacency_matrix: np.ndarray) -> np.ndarray:
    """It first takes the logarithm of the data and then normalizes it between 0 and 1. It also takes care of the infinit values and those nasty things.

    Args:
        adjacency_matrix (np.ndarray): Adjacency matrix of the network. Technically can be any matrix but I did it for the adjacency matrices.

    Returns:
        np.ndarray: Normalized data with the same shape as the input.
    """
    lognorm_adjacency_matrix = minmax_normalize(log_normalize(adjacency_matrix))
    np.fill_diagonal(lognorm_adjacency_matrix, 0.0)
    return np.where(lognorm_adjacency_matrix != 1.0, lognorm_adjacency_matrix, 0.0)


def spectral_normalization(
    target_radius: float, adjacency_matrix: np.ndarray
) -> np.ndarray:
    """Normalizes the adjacency matrix to have a spectral radius of the target_radius. Good to keep the system stable.

    Args:
        target_radius (float): A value below 1.0. It's the spectral radius that you want to achieve. But use 1.0 if you're planning to change the global coupling strength somewhere.
        adjacency_matrix (np.ndarray): Adjacency matrix of the network.

    Returns:
        np.ndarray: Normalized adjacency matrix with the same shape as the input.
    """
    spectral_radius = np.max(np.abs(np.linalg.eigvals(adjacency_matrix)))
    return adjacency_matrix * target_radius / spectral_radius


def strength_normalization(adjacency_matrix: np.ndarray) -> np.ndarray:
    """
    Normalizes the adjacency matrix to subside the effect of high-strength (or high-degree) nodes.
    This function implements the strength normalization algorithm described in [1].
    The algorithm aims to reduce the influence of high-strength nodes in a network by scaling the adjacency matrix.

    Parameters:
        adjacency_matrix (np.ndarray): The adjacency matrix of the network. It should be a square matrix of shape (n, n), where n is the number of nodes in the network.

    Returns:
        np.ndarray: The normalized adjacency matrix with the same shape as the input.

    References:
        [1] https://royalsocietypublishing.org/doi/full/10.1098/rsif.2008.0484
    """
    strength: np.ndarray = adjacency_matrix.sum(1)
    normalized_strength: np.ndarray = np.power(strength, -0.5)
    diagonalized_normalized_strength: np.ndarray = np.diag(normalized_strength)
    normalized_adjacency_matrix: np.ndarray = (
        diagonalized_normalized_strength
        @ adjacency_matrix
        @ diagonalized_normalized_strength
    )
    return normalized_adjacency_matrix

# Kısaca Simulasyonu hazır hale getiren parametreleri üreten bir fonksiyondur. 
def optimal_influence_default_values(
    adjacency_matrix: np.ndarray,
    location: str = "adjacency_matrices_for_oi",
    random_seed: int = 11,
) -> dict:
    """
    Returns the default values for the parameters of the optimal_influence function.

    Parameters:
        adjacency_matrix (np.ndarray): The adjacency matrix representing the network structure.
        location (str, optional): The location to save the adjacency matrix file. Defaults to "adjacency_matrices_for_oi".
        random_seed (int, optional): The random seed for generating input noise. Defaults to 11.

    Returns:
        dict: Default values for the parameters of the optimal_influence function.
    """
    rng: Generator = np.random.default_rng(seed=random_seed)
    NOISE_STRENGTH: float = 1
    DELTA: float = 0.01
    TAU: float = 0.02
    G: float = 0.5
    DURATION: int = 10
    input_noise: np.ndarray = rng.normal(
        0, NOISE_STRENGTH, (adjacency_matrix.shape[0], int(DURATION / DELTA))
    ) # Simullasyonu yapılmasını sağlayacak girdi gürültüsünün oluşturulması, node sayısına göre bir girdi gürültüsü üretilir
    model_params: dict = { #simulasyonumuzda bize yardımcı olacak model parametre sözlüğü 
        "dt": DELTA,
        "timeconstant": TAU,
        "coupling": G,
        "duration": DURATION,
    }
    timestamp: str = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    """   
    Fonksiyon, komşuluk matrisini belirli bir konumda .pkl formatında kaydeder. Bu, Python’un pickle modülü kullanılarak yapılan bir serileştirme işlemidir ve 
    Python nesnelerinin disk üzerinde saklanmasını ve daha sonra bu nesnelerin tam olarak aynı durumda geri yüklenmesini sağlar. Zaman damgası ve düğüm sayısı 
    gibi ek bilgilerle dosya adı oluşturulması, farklı simülasyonlar için dosyalar arasında kolay ayrım yapılmasına olanak tanır.
    """
    base_path: Path = Path(location)
    base_path.mkdir(parents=True, exist_ok=True)
    file_location: str = (
        base_path / f"adjmat_{adjacency_matrix.shape[0]}_nodes_{timestamp}"
    ) #matriximiz belirtilen konuma piksel formatında kaydedilir 

    with open(f"{file_location}.pkl", "wb") as f:
        pk.dump(adjacency_matrix, f)

    game_params: dict = {
        "adjacency_matrix": f"{file_location}.pkl",
        "input_noise": input_noise,
        "model_params": model_params,
    }

    # TODO: allow already pickled adjacency matrices to be used as input.
    # TODO: allow the user to specify an arbitrary parameter while keeping the rest as default.
    return game_params
# TODLAR YAPILDIKTAN SONRA FONKSİYONUN YENİ HALİ 
# We need the function to be able to read a pickled file if it is already there instead of making one. bu cümleyi sağladık gibi 
def optimal_influence_default_values_NEW(
    adjacency_matrix=None,
    location="adjacency_matrices_for_oi",
    random_seed=11,
    path_to_pickle=None,
    **kwargs    
) -> dict:
    # The function allows the user to load an existing adjacency matrix from a pickle file if path_to_pickle is provided and adjacency_matrix is not specified.
    # The function uses **kwargs to let users override default model parameters like noise strength, time constant, and duration, while keeping unspecified parameters at their default values.
    # I can use the function like: optimal_influence_default_values(path_to_pickle, noise=5) 
    """
    Returns the default values for the parameters of the optimal_influence function.

    Parameters:
        adjacency_matrix (np.ndarray, optional): The adjacency matrix representing the network structure.
        location (str, optional): The location to save the adjacency matrix file. Defaults to "adjacency_matrices_for_oi".
        random_seed (int, optional): The random seed for generating input noise. Defaults to 11.
        path_to_pickle (str, optional): Path to an existing pickle file of the adjacency matrix.
        **kwargs: Arbitrary keyword arguments for overwriting default model parameters.

    Returns:
        dict: Default values for the parameters of the optimal_influence function.
    """
    # Generate input noise
    """  
    Fonksiyon, **kwargs aracılığıyla alınan anahtar kelimeleri kullanarak model parametrelerini (örneğin, gürültü gücü, zaman aralığı, zaman sabiti, 
    bağlantı kuvveti ve süre) özelleştirmeye izin verir. Kullanıcı, varsayılan değerlerin üzerine yazabilir ve bu özellik fonksiyonun esnekliğini artırır.
    """
    rng = np.random.default_rng(seed=random_seed)
    NOISE_STRENGTH = kwargs.get("noise", 1)  # default noise strength is 1 unless overridden
    DELTA = kwargs.get("dt", 0.01)           # default delta
    TAU = kwargs.get("timeconstant", 0.02)   # default timeconstant
    G = kwargs.get("coupling", 0.5)          # default coupling strength
    DURATION = kwargs.get("duration", 10)    # default duration
    
    # Load or save the adjacency matrix
    """  
    Aşağıdaki if li yapının açıklaması 
    Fonksiyon, path_to_pickle parametresi verilerek çağrıldığında, bu dosya yolu üzerinden mevcut bir pickle dosyasını yükler. 
    Eğer path_to_pickle belirtilmiş ve adjacency_matrix verilmemişse, belirtilen pickle dosyasından komşuluk matrisini yükler. 
    Bu, kullanıcının her seferinde matrisi yeniden sağlamasını gerektirmeyen esnek bir kullanım sağlar.
    """ 
    if path_to_pickle and not adjacency_matrix:
        with open(path_to_pickle, "rb") as f:
            adjacency_matrix = pk.load(f)
    elif adjacency_matrix is not None:
        timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        base_path = Path(location)
        base_path.mkdir(parents=True, exist_ok=True)
        file_location = base_path / f"adjmat_{adjacency_matrix.shape[0]}_nodes_{timestamp}.pkl"
        with open(file_location, "wb") as f:
            pk.dump(adjacency_matrix, f)
        path_to_pickle = str(file_location)
    else:
        raise ValueError("Either 'adjacency_matrix' or 'path_to_pickle' must be provided.")
    
    input_noise = rng.normal(
        0, NOISE_STRENGTH, (adjacency_matrix.shape[0], int(DURATION / DELTA))
    )

    model_params = {
        "dt": DELTA,
        "timeconstant": TAU,
        "coupling": G,
        "duration": DURATION,
    }

    game_params = {
        "adjacency_matrix": path_to_pickle,
        "input_noise": input_noise,
        "model_params": model_params,
    }

    return game_params

"""*******************            simple_fit fonksiyonunun parelelize ve skcitklarn kütüphanesine uygun hali                  ***********************************************"""
def _process_parameter(parameter: Dict, model: Callable, model_kwargs: Dict, target_matrix: np.ndarray, normalize: Union[bool, Callable]) -> Dict:
    # process_parameter function during parallel processing to make predictions for each parameter combination using the model, compare these predictions with the target matrix, and then calculate and store the resulting correlation value
    """Processes a single parameter set, running the model and calculating correlation.

    Args:
        parameter (Dict): The set of parameters to use for the model.
        model (Callable): The model function to use.
        model_kwargs (Dict): Additional arguments for the model function.
        target_matrix (np.ndarray): The target matrix to compare the model's output to.
        normalize (Union[bool, Callable]): If the output needs to be normalized before taking correlation.

    Returns:
        Dict: The parameter set with the added correlation value.
    """
    # Evet, process_parameter fonksiyonunu, paralel işleme sırasında her bir parametre kombinasyonu için modelin tahminini yapıp bu tahmini hedef matrisle karşılaştırmak ve sonuç olarak korelasyon değerini hesaplayıp kaydetmek amacıyla kullanıyorsunuz
    
    estimation = model(**parameter, **model_kwargs)
    if normalize:
        estimation: np.ndarray = normalize(estimation)
    r = _matrix_correlation(target_matrix, estimation)
    parameter.update({"correlation": r})
    
    return parameter
# parallel processing with joblib's Parallel and delayed functions, making it efficient for larger datasets, and compatible with Scikit-learn's model and parameter grid structures.
def simple_fit( #  Paralel işlemeyi destekler ve bu nedenle daha büyük veri kümeleri veya daha fazla parametre kombinasyonu için daha verimlidir.
    model: Callable,
    X: np.ndarray,
    parameter_space: List[Dict],
    model_kwargs: Optional[Dict] = None,
    normalize: Union[bool, Callable] = False,
    n_jobs: int = -1
) -> List[Dict]:

    # verilen bir model ve hedef matris ile en iyi uyumu sağlayan parametre kombinasyonunu bulmayı amaçlar.
    """
    Simple fitting function to find the best parameters for a model.

    Args:
        model (callable): Which model to use.
        X (np.ndarray): Target matrix to compare the model's output to.
        parameter_space (list): List of parameter grids to search in.
        model_kwargs (Optional[dict], optional): Extra things that the model wants. Defaults to None.
        normalize (Union[bool, callable], optional): If the output needs to be normalized before taking correlation. Defaults to False.
        n_jobs (int, optional): Number of jobs for parallel processing. Defaults to -1.

    Returns:
        list: Updated copy of the parameter space with the correlation values.
    """
    if model_kwargs is None:
        model_kwargs = {}

    results = deepcopy(parameter_space)
    processed_results = Parallel(n_jobs=n_jobs)(
        delayed(_process_parameter)(param, model, model_kwargs, X, normalize)
        for param in tqdm(results, total=len(parameter_space), desc="C3PO noises: ")
    )
    return processed_results

# A Function That Enables Parallel Computation
# yeni yazdırdığın kodda tam olarak ne var eskisinden farkı ne? ve istediği sckitlearn e benzetme ve parelelizasyon işlemleri tam olarak sağlandı mı? 
def _matrix_correlation(one_matrix: np.ndarray, another_matrix: np.ndarray) -> float:
    """Computes the Pearson's correlation between two matrices (not just the upper-triangle).

    Args:
        one_matrix (np.ndarray): One of the matrices.
        another_matrix (np.ndarray): Guess what, the other matrix.

    Returns:
        float: Pearson's correlation between the two matrices.
    """
    return np.corrcoef(one_matrix.flatten(), another_matrix.flatten())[0, 1]


# TODO: add a function to create example adjacency matrices for demonstration purposes.
