import re
import os
import numpy as np
from image import Image, Mask
import logging

from constants import TIFT_DTYPE


class Dataset:
    def __init__(self, dimensions=None, time_series_length=None, *, dtype=None, data=None):
        if data is not None:
            self.data = data
        else:
            # time series data is contiguous in memory (last axis)
            # this is important to prevent cache misses
            self.data = np.zeros((*dimensions, time_series_length), dtype=dtype)

    @classmethod
    def load(cls, file_path):
        logging.info('Loading dataset from %s', file_path)
        data = np.load(file_path)['data']
        return cls(data=data)

    def save(self, file_path):
        logging.info('Saving dataset to %s', file_path)
        np.savez_compressed(file_path, data=self.data)

    @classmethod
    def load_tift(cls, folder_path):
        logging.info('Loading dataset from folder %s', folder_path)

        # retrieve all image paths in folder
        pattern = r'f\d{10}\.img\.z'
        listing = os.listdir(folder_path)
        img_paths = [os.path.join(folder_path, entry)
                     for entry in listing if re.fullmatch(pattern, entry)]
        img_paths.sort()

        time_series_length = len(img_paths)
        dimensions = (256,)*3

        # initialize dataset
        dataset = cls(dimensions, time_series_length, dtype=TIFT_DTYPE)

        # load images
        for i, img_path in enumerate(img_paths):
            dataset.data[..., i] = Image.load_tift(img_path).data

        return dataset

    def save_tift(self, folder_path, filename_format='f{one_based_index:010}.img.z'):
        logging.info('Saving dataset to folder %s', folder_path)

        # create folder if it does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # save images
        for i in range(self.time_series_length):
            img = Image(data=self.data[..., i])
            img_path = os.path.join(
                folder_path, filename_format.format(one_based_index=i+1))
            img.save_tift(img_path)

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

    def copy(self):
        new_dataset = self.__class__(self.dimensions, self.time_series_length,
                                     dtype=self.dtype)
        new_dataset.data[:] = self.data
        return new_dataset
