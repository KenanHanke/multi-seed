import numpy as np
from numba import njit


@njit
def corr_coef(time_series_1, time_series_2):
    """
    Computes the correlation coefficient between two time series.
    This is a measure of functional connectivity.

    Does the calculation manually instead of with np.corrcoef in order
    to improve performance, as this function is called many times: The
    numpy function builds a matrix, which is a small but substantial
    overhead, and by manually calculating the correlation coefficient,
    we get a 2x speedup.
    
    This implementation was tested against integer overflows, which were
    a possible concern because the time series are of type np.int16.

    Args:
        time_series_1 (np.array): The first time series
        time_series_2 (np.array): The second time series (must have same length as first)

    Returns:
        float: The correlation coefficient.
    """
    x = time_series_1
    y = time_series_2

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))

    if denominator == 0:
        r = 0
    else:
        r = numerator / denominator

    return r


class PCA:
    """
    A PCA implementation that partially implements the interface of
    sklearn.decomposition.PCA. Its results are identical to sklearn's PCA
    class when it has been initialized with the argument svd_solver='full',
    except for signs of the principal component basis vectors (which are
    arbitrary).
    """

    def __init__(self, n_components):
        """
        Initialize the PCA instance with the specified number of principal components.

        Args:
            n_components (int): Number of principal components to be retained.
        """
        self.n_components = n_components
        self.components = None

        self.mean = None

    def fit(self, X):
        """
        Compute the principal components of the input data.

        Args:
            X (numpy array-like): Input data, shape (n_samples, n_features).
        """
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # compute the covariance matrix
        cov_matrix = np.cov(X, rowvar=False)

        # compute the eigenvectors and eigenvalues using Hermitian
        # eigendecomposition because covariance matrix is symmetric
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # sort eigenvectors based on eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        # select the top eigenvectors (amount determined by n_components)
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        """
        Apply the PCA transformation to the input data.

        Args:
            X (numpy array-like): Input data, shape (n_samples, n_features).

        Returns:
            numpy array-like: Transformed data, shape (n_samples, n_components).
        """
        X = X - self.mean

        # the order of the arguments seems backwards, but it's correct
        # because the vectors are horizontal and we're therefore
        # essentially dealing with transposed matrices (AB=C <=> BtAt=Ct)
        return np.dot(X, self.components)
