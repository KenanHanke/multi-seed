from typing import Iterator
import numpy as np
import logging
from rbloom import Bloom


class Image:
    """
    A class to represent an image object.
    """

    def __init__(self,
                 dimensions: tuple[int] = None,
                 *,
                 dtype=None,
                 data=None):
        """
        Initialize the Image object.
        
        Args:
            dimensions (tuple[int], optional): Tuple representing the shape of the image.
            dtype (optional): Data type for the image.
            data (optional): Pre-existing image data.

        Raises:
            ValueError: If both dimensions and dtype are not provided.
            
        Note: Either data or dimensions and dtype must be provided.
        """
        if data is not None:
            self.data = data
        else:
            if dimensions is None or dtype is None:
                raise ValueError(
                    "Either data or dimensions and dtype must be given")
            self.data = np.zeros(dimensions, dtype=dtype)

    @classmethod
    def load(cls, path):
        """
        Load image data from a given file.

        Args:
            path (str): Path to the file.

        Returns:
            Image: An Image object containing the loaded data.
        """
        logging.debug("Loading image from %s", path)
        data = np.load(path)["data"]
        return cls(data=data)

    def save(self, path):
        """
        Save image data to a file.

        Args:
            path (str): Path to the file.
        """
        logging.debug("Saving image to %s", path)
        np.savez_compressed(path, data=self.data)

    def normalized(self) -> "Image":
        """
        Create a new Image object normalized to the range [0, 1].
        Can only be used on floating-point images to avoid data loss.

        Returns:
            Image: A new Image object with normalized data.

        Raises:
            TypeError: If the image is not of floating-point type.
        """
        if not np.issubdtype(self.dtype, np.floating):
            raise TypeError("Image must be floating-point to normalize")

        minimum = np.min(self.data)
        maximum = np.max(self.data)

        new_data = (self.data - minimum) / (maximum - minimum)
        return self.__class__(data=new_data)

    def scaled(self, factor: float) -> "Image":
        """
        Create a new Image object scaled by a given factor.
        Can only be used on floating-point images to avoid overflow.

        Args:
            factor (float): The scaling factor.

        Returns:
            Image: A new Image object with scaled data.

        Raises:
            TypeError: If the image is not of floating-point type.
        """
        if not np.issubdtype(self.dtype, np.floating):
            raise TypeError("Image must be floating-point to scale")
        new_data = self.data * factor
        return self.__class__(data=new_data)

    def converted(self, dtype) -> "Image":
        """
        Create a new Image object with a different data type.

        Args:
            dtype: The target data type.

        Returns:
            Image: A new Image object with converted data.
        """
        new_data = self.data.astype(dtype)
        return self.__class__(data=new_data)

    @classmethod
    def normalize_all(cls, images: list["Image"]) -> list["Image"]:
        """
        Normalize all images in the provided list to the range [0, 1]
        with respect to the minimum and maximum values in any of the
        images. Applies the same transformation to all images in the list.
        
        Can only be used on floating-point images to avoid data loss.

        Args:
            images (list[Image]): A list of Image objects.

        Returns:
            list[Image]: A list of normalized Image objects.
            
        Raises:
            TypeError: If any of the images are not of floating-point type.
        """
        if not all(
                np.issubdtype(image.dtype, np.floating) for image in images):
            raise TypeError("All images must be floating-point to normalize")

        minimum = min(np.min(image.data) for image in images)
        maximum = max(np.max(image.data) for image in images)

        return [
            image.__class__(data=(image.data - minimum) / (maximum - minimum))
            for image in images
        ]

    @classmethod
    def normalize_all_lazy(cls, image_iter_func) -> Iterator["Image"]:
        """
        Similar to normalize_all(), but takes a function that returns an iterator
        over Image objects instead of a list of Image objects. This allows the
        images to be loaded lazily from disk instead of all at once, which may
        be necessary depending on the number of images and the available memory.
        A function returning an iterator is used in order to allow iterating
        over the images multiple times.
        
        Args:
            image_iter_func: A function that returns an iterator over Image objects.
        
        Returns:
            iter[Image]: An iterator over normalized Image objects.
        
        Raises:
            TypeError: If any of the images are not of floating-point type.
        """
        first = True
        for image in image_iter_func():
            if not np.issubdtype(image.dtype, np.floating):
                raise TypeError(
                    "All images must be floating-point to normalize")
            if first:
                minimum = np.min(image.data)
                maximum = np.max(image.data)
                first = False
            else:
                minimum = min(minimum, np.min(image.data))
                maximum = max(maximum, np.max(image.data))
        for image in image_iter_func():
            yield image.__class__(data=(image.data - minimum) /
                                  (maximum - minimum))

    def __add__(self, other: "Image"):
        """
        Add another Image to this one. The result is a new Image object
        containing the sum of the data.

        Args:
            other (Image): The other Image object to add.

        Returns:
            Image: A new Image object with the summed data.
        """
        if self.dtype != other.dtype:
            raise ValueError("Data types must match for addition")
        new_data = self.data + other.data
        return self.__class__(data=new_data)

    def __iadd__(self, other: "Image"):
        """
        Perform an in-place addition with another Image. The data of this
        image is modified to be the sum of its own data and the other image's data.

        Args:
            other (Image): The other Image object to add.

        Returns:
            Image: The modified Image object after the addition.
        """
        if self.dtype != other.dtype:
            raise ValueError("Data types must match for in-place addition")
        self.data += other.data
        return self

    @property
    def dimensions(self):
        """
        Get the dimensions of the image.

        Returns:
            tuple[int]: Tuple representing the shape of the image.
        """
        return self.data.shape

    @property
    def dtype(self):
        """
        Get the data type of the image.

        Returns:
            The data type of the image.
        """
        return self.data.dtype

    def copy(self):
        """
        Create a copy of the Image object.

        Returns:
            Image: A new Image object with the same data.
        """
        new_image = self.__class__(self.dimensions, dtype=self.dtype)
        new_image.data[:] = self.data
        return new_image


class Mask(Image):
    """
    A class to represent a mask object, derived from the Image class.
    """

    def __init__(self,
                 dimensions: tuple[int] = None,
                 *,
                 dtype=bool,
                 data=None):
        """
        Initialize the Mask object.

        Args:
            dimensions (tuple[int], optional): Tuple representing the shape of the mask.
            dtype (optional): Data type for the mask, defaults to bool.
            data (optional): Pre-existing mask data.
        """
        super().__init__(dimensions, dtype=dtype, data=data)

    def sample(self, n_points: int, *, rng=None) -> np.array:
        """
        Sample n_points. Each point is guaranteed to be unique.

        Args:
            n_points (int): The number of points to sample.
            rng (np.random.Generator, optional): A random number generator to use for sampling points.

        Returns:
            np.array: An array containing the 3D coordinates of the sampled points. The shape of the array is (n_points, 3).
        """
        logging.info("Sampling %d unique points", n_points)

        points = np.empty((n_points, 3), dtype=np.int16)

        if rng is None:
            rng = np.random.default_rng()

        points_so_far = Bloom(n_points * 100, 0.01)

        i = 0
        while i < n_points:
            point = tuple(rng.integers(self.dimensions, size=3))
            if self.data[point]:
                if point not in points_so_far:
                    points[i] = point
                    points_so_far.add(point)
                    i += 1

        return points

    def __ior__(self, other: "Mask"):
        """
        Perform an in-place bitwise OR operation with another Mask. This
        results in a mask that contains all voxels that are in at least one
        of the masks.

        Args:
            other (Mask): The other Mask object.

        Returns:
            Mask: The modified Mask object after the bitwise OR operation.
        """
        self.data |= other.data
        return self

    def __or__(self, other: "Mask"):
        """
        Perform a bitwise OR operation with another Mask. This results in a
        mask that contains all voxels that are in at least one of the masks.

        Args:
            other (Mask): The other Mask object.

        Returns:
            Mask: A new Mask object with the result of the bitwise OR operation.
        """
        new_mask = self.copy()
        new_mask |= other
        return new_mask

    @classmethod
    def normalize_all(cls, images):
        raise NotImplementedError("Masks cannot be group normalized")