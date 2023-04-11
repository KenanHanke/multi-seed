import logging
from dataset import Dataset
from typing import Iterable
from rbloom import Bloom
import numpy as np
from typing import Optional
from image import Mask


def sample_points(n_points: int, *, mask: Optional[Mask], rng=None):
    """
    Sample n_points from the given mask. If no mask is given, sample from the whole space.
    Each point is guaranteed to be unique.

    Args:
        n_points (int): The number of points to sample.
        mask (Mask, optional): A mask object to filter points based on certain criteria.
        rng (np.random.Generator, optional): A random number generator to use for sampling points.

    Returns:
        np.array: An array containing the 3D coordinates of the sampled points. The shape of the array is (n_points, 3).
    """
    logging.info("Sampling %d unique points", n_points)

    points = np.empty((n_points, 3), dtype=np.int16)

    if rng is None:
        rng = np.random.default_rng()

    points_so_far = Bloom(n_points * 1000, 0.01)

    i = 0
    while i < n_points:
        point = tuple(rng.integers(mask.dimensions, size=3))
        if mask is None or mask.data[point]:
            if point not in points_so_far:
                points[i] = point
                points_so_far.add(point)
                i += 1

    return points
