from rbloom import Bloom
import numpy as np
import logging
from image import Mask


class ReferenceBuilder:
    """
    A class to create a set of reference time series by sampling points from a dataset.

    This class samples points from a dataset based on the provided dimensions and an optional mask. 
    The sampled points can be used to build a Reference object, which contains a set of reference 
    time series that can be used to compute correlation coefficients with other time series.

    Attributes:
        dimensions (tuple[int]): The dimensions of the data to sample points from.
        mask (Mask, optional): A mask object to filter points based on certain criteria.
        points (np.array): An array containing the sampled points.
    """

    def __init__(self, *, dimensions: tuple[int] = None, mask: Mask = None):
        """
        Initialize a ReferenceBuilder instance.

        Args:
            dimensions (tuple[int], optional): The dimensions of the data to sample points from.
            mask (Mask, optional): A mask object to filter points based on certain criteria.

        Note: Either dimensions or mask must be specified.
        """
        self.mask = mask
        if mask is None:
            self.dimensions = dimensions
        else:
            self.dimensions = mask.dimensions
        self.points = None

    def sample(self, n: int, rng=None):
        """
        Sample n points from the data space.

        Args:
            n (int): The number of points to sample.
            rng (np.random.Generator, optional): A random number generator to use for sampling points.
        """
        logging.info('Sampling %d points', n)

        self.points = np.empty((n, 3), dtype=np.int16)

        if rng is None:
            rng = np.random.default_rng()

        points_so_far = Bloom(n*1000, 0.01)

        i = 0
        while i < n:
            point = tuple(rng.integers(self.dimensions, size=3))
            if self.mask is None or self.mask.data[point]:
                if point not in points_so_far:
                    self.points[i] = point
                    points_so_far.add(point)
                    i += 1

    def save(self, path):
        """
        Save the sampled points to a compressed file.

        Args:
            path (str): The path to save the compressed file.
        """
        np.savez_compressed(path, points=self.points)

    @classmethod
    def load(cls, path):
        """
        Load a ReferenceBuilder instance from a saved file.

        Args:
            path (str): The path to the saved file.

        Returns:
            ReferenceBuilder: An instance of ReferenceBuilder with the loaded data.
        """
        data = np.load(path)
        reference_builder = cls()
        reference_builder.points = data['points']
        return reference_builder

    def __len__(self):
        return len(self.points)

    def build(self, dataset):
        """
        Build a Reference object from the sampled points.

        Args:
            dataset: The dataset to extract the reference time series from.

        Returns:
            Reference: A Reference object containing the reference time series.
        """
        logging.info('Building reference from %d points', len(self))

        reference = Reference(len(self), dataset.time_series_length)
        reference.source = self
        for i, point in enumerate(self.points):
            reference.data[i] = dataset.data[tuple(point)]
        return reference


class Reference:
    """
    Contains a set of reference time series to which correlation
    coefficients can be computed.

    Attributes:
        data (np.array): A 2D array containing the reference time series.
        source (ReferenceBuilder): The ReferenceBuilder instance used to generate the reference time series.
    """

    def __init__(self, n: int, time_series_length: int):
        """
        Initialize a Reference instance.

        Args:
            n (int): The number of reference time series.
            time_series_length (int): The length of each time series.
        """
        self.data = np.zeros((n, time_series_length), dtype=np.float64)
        self.source: ReferenceBuilder = None
