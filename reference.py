from rbloom import Bloom
import numpy as np
import logging


class ReferenceBuilder:
    def __init__(self, *, dimensions=None, mask=None):
        """
        Either dimensions or mask must be specified.
        """
        self.mask = mask
        if mask is None:
            self.dimensions = dimensions
        else:
            self.dimensions = mask.dimensions
        self.points = None

    def sample(self, n, rng=None):
        """
        Sample n points.
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
        np.savez_compressed(path, points=self.points)

    @ classmethod
    def load(cls, path):
        data = np.load(path)
        reference_builder = cls()
        reference_builder.points = data['points']
        return reference_builder

    def __len__(self):
        return len(self.points)

    def build(self, dataset):
        """
        Build a reference from the sampled points.
        """
        logging.info('Building reference from %d points', len(self))

        reference = Reference(len(self), dataset.time_series_length)
        for i, point in enumerate(self.points):
            reference.data[i] = dataset.data[tuple(point)]
        return reference


class Reference:
    """
    Contains a set of reference time series to which correlation
    coefficients can be computed.
    """

    def __init__(self, n, time_series_length):
        self.data = np.zeros((n, time_series_length), dtype=np.float64)
