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
    to improve performance, as this function is called many times. The
    numpy function builds a matrix, which is a small but substantial
    overhead.

    Args:
        time_series_1 (np.array): The first time series.
        time_series_2 (np.array): The second time series. Must have the same length as time_series_1.

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
    denominator = ((n * sum_sq_1 - sum_1 ** 2) * (n * sum_sq_2 - sum_2 ** 2)) ** 0.5

    if denominator == 0:
        return 0
    else:
        return np.abs(numerator / denominator)
