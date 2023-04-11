import numpy as np
from image import Image, Mask
import logging
from collections.abc import Iterable


class Dataset:
    """
    Represent a sequence of images as a single object.
    """

    def __init__(self,
                 dimensions: tuple[int] = None,
                 n_images: int = None,
                 *,
                 dtype=None,
                 data=None):

        if data is not None:
            self.data = data

        else:
            if dimensions is None or dtype is None:
                raise ValueError(
                    "Either data or the other parameters must be given")

            # all data for a voxel is contiguous in memory (last axis)
            # this is important to prevent cache misses
            self.data = np.zeros((*dimensions, n_images), dtype=dtype)

    @classmethod
    def load(cls, file_path):
        logging.info("Loading dataset from %s", file_path)
        data = np.load(file_path)["data"]
        return cls(data=data)

    def save(self, file_path):
        logging.info("Saving dataset to %s", file_path)
        np.savez_compressed(file_path, data=self.data)

    def extract_mask(self):
        mask = Mask(self.dimensions)
        mask.data[:] = np.any(self.data, axis=-1)
        return mask

    @property
    def dimensions(self):
        return self.data.shape[:-1]

    @property
    def n_images(self):
        return self.data.shape[-1]

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return self.n_images

    def __getitem__(self, key: int):
        if not isinstance(key, int):
            raise TypeError("Only integer indexing is supported")

        return Image(data=self.data[..., key])

    def __setitem__(self, key: int, value: Image):
        if not isinstance(key, int):
            raise TypeError("Only integer indexing is supported")

        if not self.data.dtype == value.dtype:
            raise ValueError("Data type mismatch")

        self.data[..., key] = value.data

    def copy(self):
        new_dataset = self.__class__(self.dimensions,
                                     self.n_images,
                                     dtype=self.dtype)
        new_dataset.data[:] = self.data
        return new_dataset


class DatasetLoader(Iterable[Dataset]):
    """
    Reusable iterable that lazily loads datasets. Instances should only
    be created by internal functions.
    """

    def __init__(self, paths: list, generator_func):
        self.paths = paths
        self.generator_func = generator_func

    def __iter__(self):
        return self.generator_func(self.paths)

    def __len__(self):
        return len(self.paths)

    def extract_mask(self):
        """
        Accumulate a mask from the given datasets.
        """
        logging.info("Creating combined mask from datasets")
        datasets = iter(self)
        try:
            first = next(datasets)
        except StopIteration:
            raise ValueError("No datasets given")
        mask = first.extract_mask()
        for dataset in datasets:
            mask |= dataset.extract_mask()
        return mask