import numpy as np
from numba import njit, prange
import logging
from image import Image, Mask
from os import sched_getaffinity
from math_utils import corr_coef


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
        # raise if not (dimensions specified XOR mask specified)
        if dimensions is None and mask is None:
            raise ValueError(
                "Either dimensions or mask must be specified for ReferenceBuilder"
            )
        if dimensions is not None and mask is not None:
            raise ValueError(
                "Only one of dimensions or mask can be specified for ReferenceBuilder"
            )
        
        self.radius = float(radius)
        self.mask = mask
        if mask is None:
            self.dimensions = dimensions
        else:
            self.dimensions = mask.dimensions
        self.points = None
        self._LOADED = False

    def sample(self, n_points: int, rng=None):
        """
        Sample n points from the data space.

        Args:
            n (int): The number of points to sample.
            rng (np.random.Generator, optional): A random number generator to use for sampling points.
        """
        if self._LOADED:
            raise RuntimeError(
                "Cannot sample new points from a loaded ReferenceBuilder instance"
            )

        self.points = self.mask.sample(n_points, rng=rng)

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
        reference_builder._LOADED = True
        return reference_builder

    def __len__(self):
        return len(self.points)

    @property
    def n_seeds(self):
        return len(self)

    def visualized(self) -> Image:
        """
        Visualize the sampled reference regions.

        Returns:
            Image: A floating point Image object containing the visualization.
        """
        image = Image(self.dimensions, dtype=np.float32)

        self.__class__._visualize(image.data, self.points, self.radius,
                                  len(sched_getaffinity(0)))

        return image

    @staticmethod
    @njit(parallel=True)
    def _visualize(image_data, points, radius, availabe_cpus):
        point_arrs = np.array_split(points, availabe_cpus)

        # If the radius is 0, just put in the point at each voxel
        if radius <= 0:
            for point in points:
                point = (int(point[0]), int(point[1]), int(point[2])
                         )  # necessary syntax for numba
                image_data[point] += 1
            return

        reduction_var = np.zeros_like(image_data)

        # At this point, the radius is guaranteed to be greater than 0
        for i in prange(availabe_cpus):
            points = point_arrs[i]
            data = np.zeros_like(image_data)

            for index, value in np.ndenumerate(data):
                x, y, z = index

                for point in points:
                    dist = np.sqrt((x - point[0])**2 + (y - point[1])**2 +
                                   (z - point[2])**2)

                    if dist <= radius:
                        value += -dist**2 / radius**2 + 1

                data[index] = value

            reduction_var += data

        image_data += reduction_var

    def build(self, dataset):
        """
        Build a Reference object from the sampled points.

        Args:
            dataset: The dataset to extract the reference time series from.

        Returns:
            Reference: A Reference object containing all reference time series.
        """
        logging.info("Building reference from %d points", len(self))

        reference = Reference(len(self), dataset.n_images)
        reference.source = self
        self.__class__._build_reference(reference.data, dataset.data,
                                        self.points, self.radius)
        return reference

    @staticmethod
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
                reference_data[i] = _build_reference_seed(
                    dataset_data, point, radius)


# A COMPILED REFERENCEBUILDER HELPER FUNCTION
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


class Reference:
    """
    Contains a set of reference time series to which correlation
    coefficients can be computed.

    Attributes:
        data (np.array): A 2D array containing the reference time series. Shape is (n_seeds, time_series_length).
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

    @property
    def n_seeds(self):
        """
        Returns the number of reference time series.

        Returns:
            int: The number of reference time series.
        """
        return self.data.shape[0]

    @property
    def time_series_length(self):
        """
        Returns the length of each time series.

        Returns:
            int: The length of each time series.
        """
        return self.data.shape[1]

    def apply(self, time_series_array: np.ndarray) -> np.ndarray:
        """
        Compute correlation coefficients for each pair of reference time series and point time series.

        Args:
            time_series_array (np.ndarray): An array of point time series, of shape (n_points, time_series_length).

        Returns:
            np.ndarray: An array of correlation coefficients.
        """
        return self.__class__._apply(self.data, time_series_array)

    @staticmethod
    @njit(parallel=True)
    def _apply(reference_data: np.ndarray,
               time_series_array: np.ndarray) -> np.ndarray:
        output = np.zeros(
            (time_series_array.shape[0], reference_data.shape[0]),
            dtype=np.float32)
        for i in prange(time_series_array.shape[0]):
            point_time_series = time_series_array[i]

            if not np.any(point_time_series):
                continue  # empty time series have no correlation

            for j, reference_time_series in enumerate(reference_data):
                output[i, j] = corr_coef(point_time_series,
                                         reference_time_series)
        return output
