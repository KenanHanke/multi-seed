import numpy as np
from numba import njit


@njit
def abs_corr_coef(time_series_1, time_series_2):
    """
    Computes the absolute value of the correlation coefficient between
    two time series. This is a measure of functional connectivity. The
    absolute value is used because a strong negative correlation is
    equally indicative of a connection as a strong positive correlation.

    Does the calculation manually instead of with np.corrcoef in order
    to improve performance, as this function is called many times. (The
    numpy function builds a matrix, which is a small but substantial
    overhead.)

    Args:
        time_series_1 (np.array): The first time series
        time_series_2 (np.array): The second time series (must have same length as first)

    Returns:
        float: The absolute value of the correlation coefficient.
    """
    n = len(time_series_1)

    sum_1 = np.sum(time_series_1)
    sum_2 = np.sum(time_series_2)

    sum_sq_1 = np.sum(np.square(time_series_1))
    sum_sq_2 = np.sum(np.square(time_series_2))

    sum_prod = np.sum(time_series_1 * time_series_2)

    numerator = n * sum_prod - sum_1 * sum_2
    denominator = ((n * sum_sq_1 - sum_1**2) * (n * sum_sq_2 - sum_2**2))**0.5

    if denominator == 0:
        return 0
    else:
        return np.abs(numerator / denominator)


class UntranslatedPCA:
    """
    A PCA implementation that does not translate the data before
    performing the transformation. Partially implements the interface
    of sklearn.decomposition.PCA. When run on pre-centered data, the
    results are identical to sklearn's PCA class when it has been
    initialized with the argument svd_solver='full', except for signs
    of the principal component basis vectors (which are arbitrary).
    """

    # Lines that would perform a translation are included for reference
    # but are commented out. It has been verified that the results of
    # running this class with the translation lines not commented out
    # are identical to the results of running sklearn's PCA class with
    # the argument svd_solver='full', except for signs of the principal
    # component basis vectors, which are arbitrary.

    def __init__(self, n_components):
        """
        Initialize the UntranslatedPCA instance with the specified number of principal components.

        Args:
            n_components (int): Number of principal components to be retained.
        """
        self.n_components = n_components
        self.components = None

        # FOLLOWING LINE IS INCLUDED ONLY FOR REFERENCE
        # self.mean = None

    def fit(self, X):
        """
        Compute the principal components of the input data without centering it.

        Args:
            X (numpy array-like): Input data, shape (n_samples, n_features).
        """
        # FOLLOWING LINES ARE INCLUDED ONLY FOR REFERENCE
        # self.mean = np.mean(X, axis=0)
        # X = X - self.mean

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
        Apply the UntranslatedPCA transformation to the input data without centering it.

        Args:
            X (numpy array-like): Input data, shape (n_samples, n_features).

        Returns:
            numpy array-like: Transformed data, shape (n_samples, n_components).
        """
        # FOLLOWING LINE IS INCLUDED ONLY FOR REFERENCE
        # X = X - self.mean

        return np.dot(X, self.components)
