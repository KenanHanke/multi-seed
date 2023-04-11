from rbloom import Bloom
import numpy as np
from numba import njit, prange
import logging
from image import Mask
from common import sample_points


class ReferenceBuilder:
    """
    A class to create a set of reference time series by sampling points from a dataset.

    This class samples points from a dataset based on the provided dimensions and an optional mask.
    The sampled points can be used to build a Reference object, which contains a set of reference
    time series that can be used to compute correlation coefficients with other time series.

    If the radius is greater than 0, the sampled points will be interpreted as the center of a
    spherical seed region and the time series for that region will be the weighted mean of all
    voxels within that sphere. The weights are determined by w = - d^2 / r^2 + 1, where d is the
    distance from the center of the sphere and r is the radius of the sphere. This means that
    voxels closer to the center of the sphere will have a higher weight than voxels further away.

    Attributes:
        dimensions (tuple[int]): The dimensions of the data to sample points from.
        mask (Mask, optional): A mask object to filter points based on certain criteria.
        points (np.array): An array containing the sampled points.
        radius (float): The radius of the seed regions to sample. Must be greater than or equal to 0.
    """

    def __init__(self,
                 *,
                 radius: float,
                 dimensions: tuple[int] = None,
                 mask: Mask = None):
        """
        Initialize a ReferenceBuilder instance.

        Args:
            dimensions (tuple[int], optional): The dimensions of the data to sample points from.
            mask (Mask, optional): A mask object to filter points based on certain criteria.

        Note: Either dimensions or mask must be specified.
        """
        self.radius = float(radius)
        self.mask = mask
        if mask is None:
            self.dimensions = dimensions
        else:
            self.dimensions = mask.dimensions
        self.points = None
        self._loaded = False

    def sample(self, n_points: int, rng=None):
        """
        Sample n points from the data space.

        Args:
            n (int): The number of points to sample.
            rng (np.random.Generator, optional): A random number generator to use for sampling points.
        """
        if self._loaded:
            raise RuntimeError(
                "Cannot sample new points from a loaded ReferenceBuilder instance"
            )

        self.points = sample_points(n_points, mask=self.mask, rng=rng)

    def save(self, path):
        """
        Save the sampled points to a compressed file.

        Args:
            path (str): The path to save the compressed file.
        """
        np.savez_compressed(path,
                            points=self.points,
                            radius=self.radius,
                            dimensions=self.dimensions)

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
        points = data["points"]
        radius = data["radius"]
        dimensions = tuple(data["dimensions"])
        reference_builder = cls(radius=radius, dimensions=dimensions)
        reference_builder.points = points
        reference_builder._loaded = True
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
        logging.info("Building reference from %d points", len(self))

        reference = Reference(len(self), dataset.n_images)
        reference.source = self
        _build_reference(reference.data, dataset.data, self.points,
                         self.radius)
        return reference


############ START OF COMPILED REFERENCEBUILDER HELPER FUNCTIONS ############


@njit(parallel=True)
def _build_reference(reference_data, dataset_data, points, radius):
    for i in prange(points.shape[0]):
        point = points[i]
        point = (
            int(point[0]),
            int(point[1]),
            int(point[2]),
        )  # necessary syntax for numba
        if radius <= 0:
            reference_data[i] = dataset_data[point]
        else:
            reference_data[i] = _build_reference_seed(dataset_data, point,
                                                      radius)


@njit
def _build_reference_seed(dataset_data, point, radius):
    # Returns a time series where each value is the weighted mean of all voxels of the
    # corresponding time point within a sphere of radius r centered at the given point.

    x, y, z = point  # not necessarily MNI x, y, z coordinates
    ceil_radius = int(np.ceil(radius))

    # Prevent going out of bounds of the dataset
    min_x = max(0, x - ceil_radius)
    max_x = min(dataset_data.shape[0] - 1, x + ceil_radius)
    min_y = max(0, y - ceil_radius)
    max_y = min(dataset_data.shape[1] - 1, y + ceil_radius)
    min_z = max(0, z - ceil_radius)
    max_z = min(dataset_data.shape[2] - 1, z + ceil_radius)

    total_weight = 0
    time_series = np.zeros(dataset_data.shape[3], dtype=np.float64)

    for i in range(min_x, max_x + 1):
        for j in range(min_y, max_y + 1):
            for k in range(min_z, max_z + 1):
                dist = np.sqrt((i - x)**2 + (j - y)**2 + (k - z)**2)

                # only use voxels within the sphere
                if dist > radius:
                    continue

                weight = 1 - dist**2 / radius**2

                total_weight += weight
                time_series += weight * dataset_data[i, j, k]

    return time_series / total_weight


############ END OF COMPILED REFERENCEBUILDER HELPER FUNCTIONS ############


class Reference:
    """
    Contains a set of reference time series to which correlation
    coefficients can be computed.

    Attributes:
        data (np.array): A 2D array containing the reference time series.
        source (ReferenceBuilder): The ReferenceBuilder instance used to generate the reference time series.
    """

    def __init__(self, n_seeds: int, time_series_length: int):
        """
        Initialize an empty Reference instance.

        Args:
            n_seeds (int): The number of reference time series.
            time_series_length (int): The length of each time series.
        """
        self.data = np.zeros((n_seeds, time_series_length), dtype=np.float64)
        self.source: ReferenceBuilder = None
