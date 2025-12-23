
"""



Functions for generating nonlinear synthetic data. It is used to generated the correlated concepts dataset.

This module provides functions to generate synthetic datasets with nonlinear relationships between covariates, concepts, and labels.
It includes utilities for creating random nonlinear mappings, generating synthetic data, and constructing dataset objects for training, validation, and testing.

Classes:
    SyntheticDataset: Custom Dataset class for the nonlinear synthetic data.

Functions:
    random_nonlin_map: Creates a random nonlinear function parameterized by an MLP.
    generate_synthetic_data_correlated_c: Generates a synthetic dataset with correlated concepts.
    get_synthetic_datasets: Constructs dataset objects for the synthetic data.
"""

import numpy as np
from numpy.random import multivariate_normal

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_low_rank_matrix

import torch
from torch.utils import data
import pandas as pd


def random_nonlin_map(n_in, n_out, n_hidden, rank=100):
    """
    Create a random nonlinear function parameterized by an MLP.
    This is used to create the mapping between concepts and covariates.

    Args:
        n_in (int): Number of input features.
        n_out (int): Number of output features.
        n_hidden (int): Number of hidden units.
        rank (int, optional): Effective rank for the low-rank matrices. Default is 100.

    Returns:
        function: A random nonlinear mapping.
    """
    # Random MLP mapping
    W_0 = make_low_rank_matrix(n_in, n_hidden, effective_rank=rank)
    W_1 = make_low_rank_matrix(n_hidden, n_hidden, effective_rank=rank)
    W_2 = make_low_rank_matrix(n_hidden, n_out, effective_rank=rank)
    # No biases
    b_0 = np.random.uniform(0, 0, (1, n_hidden))
    b_1 = np.random.uniform(0, 0, (1, n_hidden))
    b_2 = np.random.uniform(0, 0, (1, n_out))

    nlin_map = lambda x: np.matmul(
        ReLU(
            np.matmul(ReLU(np.matmul(x, W_0) + np.tile(b_0, (x.shape[0], 1))), W_1)
            + np.tile(b_1, (x.shape[0], 1))
        ),
        W_2,
    ) + np.tile(b_2, (x.shape[0], 1))

    return nlin_map


def ReLU(x):
    return x * (x > 0)


#tei 추가 12/21 
def generate_synthetic_data_scm(p: int, n: int, k: int, seed: int):
    """
    Generate a synthetic dataset based on a specific Structural Causal Model (SCM).
    Structure:
    C0 (Root)
    C1 <- C0 (Chain)
    C2 <- C0 (Fork with C1)
    C3 <- C1, C2 (Collider)
    Y  <- C3
    
    Args:
        p (int): Number of covariates (X dimension).
        n (int): Number of data points.
        k (int): Number of concepts (Ignored, fixed to 4 for this SCM).
        seed (int): Random generator seed.

    Returns:
        tuple: X, c, y
    """
    np.random.seed(seed)
    
    # 1. Generate Concepts (SCM)
    # C0: Root variable ~ Bernoulli(0.5)
    c0 = np.random.binomial(1, 0.5, n)

    # C1: Depends on C0 (Chain: C0 -> C1)
    # If C0=1, 80% chance C1=1. If C0=0, 20% chance C1=1.
    prob_c1 = 0.8 * c0 + 0.2 * (1 - c0)
    c1 = np.random.binomial(1, prob_c1)

    # C2: Depends on C0 (Fork: C1 <- C0 -> C2)
    # If C0=1, 70% chance C2=1.
    prob_c2 = 0.7 * c0 + 0.3 * (1 - c0)
    c2 = np.random.binomial(1, prob_c2)

    # C3: Depends on C1 and C2 (Collider: C1 -> C3 <- C2)
    # Let's use a noisy AND gate: C3 is likely 1 if C1=1 AND C2=1
    prob_c3 = 0.9 * (c1 * c2) + 0.1 * (1 - (c1 * c2))
    c3 = np.random.binomial(1, prob_c3)

    # Stack concepts (n, 4)
    c = np.stack([c0, c1, c2, c3], axis=1)

    # 2. Generate Label Y
    # Y depends primarily on C3 (and maybe some noise)
    # Y = C3
    prob_y = 0.95 * c3 + 0.05 * (1 - c3)
    y = np.random.binomial(1, prob_y).reshape(-1, 1)

    # 3. Generate Features X
    # Map discrete concepts to continuous latent space z
    # Add noise so X isn't just a perfect copy of C
    z = c.astype(float) + np.random.normal(0, 0.5, c.shape)
    
    # Non-linear mapping to high-dimensional space
    # Input dimension is 4 (number of concepts), Output is p
    g = random_nonlin_map(
        n_in=4,
        n_out=p,
        n_hidden=10, # Slightly more complex mapping
    )
    X = g(z)

    # Standardize and add observation noise
    ss = StandardScaler()
    X = ss.fit_transform(X)
    X = X + np.random.normal(0, 0.5, X.shape) # Add noise to features
    X = ss.fit_transform(X)

    return X, c, y



def generate_scm_noise_structural(p: int, n: int, k:int, seed: int):
    np.random.seed(seed)

    u0 = np.random.normal(0, 1, n)
    u1 = np.random.normal(0, 1, n)
    u2 = np.random.normal(0, 1, n)
    u3 = np.random.normal(0, 1, n)
    uy = np.random.normal(0, 1, n)

    c0 = (u0 > 0).astype(int)
    c1 = (1.5 * c0 + u1 > 0).astype(int)
    c2 = (1.2 * c0 + u2 > 0).astype(int)
    c3 = (1.3 * c1 + 1.3 * c2 + u3 > 1.5).astype(int)

    c = np.stack([c0, c1, c2, c3], axis=1)

    y = ((1.8 * c3 + 0.8 * c1 + uy) > 1.0).astype(int).reshape(-1, 1)

    z = c + np.random.normal(0, 0.3, c.shape)
    g = random_nonlin_map(4, p, 12)
    X = g(z)
    X = StandardScaler().fit_transform(X)

    return X, c, y



def generate_synthetic_data_correlated_c(p: int, n: int, k: int, seed: int):
    """
    Generate a synthetic dataset with correlated concepts.

    Args:
        p (int): Number of covariates.
        n (int): Number of data points.
        k (int): Number of concepts.
        seed (int): Random generator seed.

    Returns:
        tuple: A tuple containing the design matrix (X), concept values (c), and labels (y).
    """
    # Generative process: x <-- z --> c --> y

    # Replicability
    np.random.seed(seed)

    # Generate concepts
    # Generate a k x 5 matrix of random values from a standard normal distribution
    W = np.random.randn(k, 10)

    # Compute the product W * W' (W times its transpose) and add a diagonal matrix of random values
    S = np.dot(W, W.T) + np.diag(np.random.rand(k))

    sigma = S

    try:
        torch.linalg.cholesky(torch.tensor(sigma))
    except:
        assert False, "Matrix not positive definite"

    z = multivariate_normal(mean=np.zeros((k,)), cov=sigma, size=n)
    c = (z >= 0) * 1

    # Generate balanced labels from concepts
    lin_weights = np.random.uniform(size=(1, k))
    lin_weights = np.tile(lin_weights, (n, 1))
    y = np.sum(lin_weights * c, 1, keepdims=True)
    tmp = np.median(y, 0)
    y = (y >= tmp) * 1

    # Nonlinear maps. The number of hidden units is fixed to 5 to make the task harder
    g = random_nonlin_map(
        n_in=k,
        n_out=p,
        n_hidden=5,
    )
    # Generate covariates
    # Concept groups get mapped to one
    X = g(z)

    ss = StandardScaler()
    X = ss.fit_transform(X)
    # Add Gaussian noise after standardization to ensure similar effect on all covariates and have intuition on how much noise is added
    X = X + np.random.normal(0, 1.0, X.shape)
    ss = StandardScaler()
    X = ss.fit_transform(X)

    return X, c, y


class SyntheticDataset(data.dataset.Dataset):
    """
    Dataset class for the nonlinear synthetic data
    """

    def __init__(
        self,
        num_vars: int,
        num_points: int,
        num_predicates: int,
        type: str = None,
        indices: np.ndarray = None,
        seed: int = 42,
    ):
        """
        Initialize the SyntheticDataset.

        Args:
            num_vars (int): Number of covariates.
            num_points (int): Number of data points.
            num_predicates (int): Number of concepts.
            type (str, optional): Type of synthetic data to generate. Default is None.
            indices (numpy.ndarray, optional): Indices of the data points to be kept. Default is None.
            seed (int, optional): Random generator seed. Default is 42.
        """
        # Shall a partial predicate set be used?
        self.predicate_idx = np.arange(0, num_predicates)
        if type == "correlated_c":
            generate_synthetic_data = generate_synthetic_data_correlated_c
        
        
        #tei 추가 12/21
        elif type == "scm":
            generate_synthetic_data = generate_synthetic_data_scm
        elif type == "Structural-Noise SCM":
            generate_synthetic_data = generate_scm_noise_structural
        else:
            ValueError("Simulation type not implemented!")

        self.X, self.c, self.y = generate_synthetic_data(
            p=num_vars, n=num_points, k=num_predicates, seed=seed
        )

        if indices is not None:
            self.X = self.X[indices]
            self.c = self.c[indices]
            self.y = self.y[indices]

        self.graph = []

    def register_graph(self, graph):
        self.graph = graph

    def __getitem__(self, index):
        """
        Returns points from the dataset

        @param index: index
        @return: a dictionary with the data; dict['features'] contains features, dict['label'] contains
        target labels, dict['concepts'] contains concepts
        """
        labels = self.y[index, 0]
        concepts = self.c[index, self.predicate_idx]
        features = self.X[index].astype("f")

        return {"x": torch.Tensor(features), 
                "y": torch.Tensor([labels]), 
                "c": torch.Tensor(concepts),
                "graph": []}

    def __len__(self):
        return self.X.shape[0]


def get_synthetic_datasets(
    num_vars: int,
    num_points: int,
    num_predicates: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    ftune_size: float = 0.,
    seed: int = 42,
    type: str = None,
):
    """
    Construct dataset objects for the synthetic data.

    Args:
        num_vars (int): Number of covariates.
        num_points (int): Number of data points.
        num_predicates (int): Number of concepts.
        train_ratio (float, optional): Ratio of training set size. Default is 0.6.
        val_ratio (float, optional): Ratio of validation set size. Default is 0.2.
        seed (int, optional): Random generator seed. Default is 42.
        type (str, optional): Type of synthetic data to generate. Default is None.

    Returns:
        tuple: Dataset objects for the training, validation, and test sets.
    """
    # Train-validation-test split
    indices_train, indices_valtest = train_test_split(
        np.arange(0, num_points), train_size=train_ratio, random_state=seed
    )
    indices_val, indices_test = train_test_split(
        indices_valtest,
        train_size=val_ratio / (1.0 - train_ratio),
        random_state=2 * seed,
    )

    # Datasets
    print("generating synthetic data...")
    train_data = SyntheticDataset(
            num_vars=num_vars,
            num_points=num_points,
            num_predicates=num_predicates,
            indices=indices_train,
            seed=seed,
            type=type,
        )
    print("train data generated")
    val_data = SyntheticDataset(
            num_vars=num_vars,
            num_points=num_points,
            num_predicates=num_predicates,
            indices=indices_val,
            seed=seed,
            type=type,
        )
    print("val data generated")
    test_data = SyntheticDataset(
            num_vars=num_vars,
            num_points=num_points,
            num_predicates=num_predicates,
            indices=indices_test,
            seed=seed,
            type=type,
        )
    print("test data generated")

    # Create a dictionary to hold the datasets
    synthetic_datasets = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }

    return synthetic_datasets

class SyntheticDatasetContainer():
    def __init__(self, 
                 data,
                 c_info=None,
                 y_info=None,):

        self.data = data
        self.c_info = c_info
        self.y_info = y_info
        self.graph = []

    def load_ground_truth_graph(self):
        # The only edges that have to be different from 0 are the ones
        # that go from the concepts to the label.
        # adj_mat = np.zeros((len(self.c_info['names']) + 1, len(self.c_info['names']) + 1), 
        #                    dtype=int)
        # adj_mat[:-1, -1] = 1

        #tei 추가 12/21
        
        # If using the SCM dataset, we know the specific graph structure
        if len(self.c_info['names']) == 4:

            adj_mat = np.zeros((5, 5), dtype=int)
            # Indices: 0:C0, 1:C1, 2:C2, 3:C3, 4:Y

            # Common SCM backbone
            adj_mat[0, 1] = 1  # C0 -> C1
            adj_mat[0, 2] = 1  # C0 -> C2
            adj_mat[1, 3] = 1  # C1 -> C3
            adj_mat[2, 3] = 1  # C2 -> C3

            # Y <- C3
            adj_mat[3, 4] = 1


            # Y <- C3 + C1  (generate_scm_v2_structural)
            adj_mat[1, 4] = 1


        else:
            # Default star graph (all C -> Y)
            adj_mat = np.zeros((len(self.c_info['names']) + 1, len(self.c_info['names']) + 1), 
                            dtype=int)
            adj_mat[:-1, -1] = 1



        graph = pd.DataFrame(adj_mat,
                            columns=self.c_info['names'] + self.y_info['names'],
                            index=self.c_info['names'] + self.y_info['names'])
        return graph
