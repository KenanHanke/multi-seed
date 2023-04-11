import os
import numpy as np
from image import Image, Mask
import logging


class Dataset:

    def __init__(self,
                 dimensions: tuple[int] = None,
                 time_series_length: int = None,
                 *,
                 dtype=None,
                 data=None):

        if data is not None:
            self.data = data

        else:
            if dimensions is None or dtype is None:
                raise ValueError(
                    "Either data or the other parameters must be given")

            # time series data is contiguous in memory (last axis)
            # this is important to prevent cache misses
            self.data = np.zeros((*dimensions, time_series_length),
                                 dtype=dtype)

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
    def time_series_length(self):
        return self.data.shape[-1]

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return self.time_series_length

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
                                     self.time_series_length,
                                     dtype=self.dtype)
        new_dataset.data[:] = self.data
        return new_dataset
