from tift import TIFT_DTYPE
import numpy as np
import gzip
import logging


class Image:
    def __init__(self, dimensions: tuple[int] = None, *, dtype=None, data=None):
        if data is not None:
            self.data = data
        else:
            self.data = np.zeros(dimensions, dtype=dtype)

    @classmethod
    def load(cls, path):
        logging.debug('Loading image from %s', path)
        data = np.load(path)['data']
        return cls(data=data)

    def save(self, path):
        logging.debug('Saving image to %s', path)
        np.savez_compressed(path, data=self.data)

    @classmethod
    def load_tift(cls, path):
        logging.debug('Loading image from %s', path)

        dimensions = (256,)*3
        image = cls(dimensions, dtype=TIFT_DTYPE)
        with gzip.open(path, 'rb') as f:
            raw_data = np.frombuffer(f.read(), dtype=TIFT_DTYPE)
            image.data[:] = raw_data.reshape(dimensions)
        return image

    def save_tift(self, path):
        logging.debug('Saving image to %s', path)

        with gzip.open(path, 'wb') as f:
            f.write(self.data.tobytes())

    @property
    def dimensions(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def copy(self):
        new_image = self.__class__(self.dimensions, dtype=self.dtype)
        new_image.data[:] = self.data
        return new_image


class Mask(Image):
    def __init__(self, dimensions: tuple[int] = None, *, dtype=bool, data=None):
        super().__init__(dimensions, dtype=dtype, data=data)

    def __ior__(self, other: 'Mask'):
        self.data |= other.data
        return self

    def __or__(self, other: 'Mask'):
        new_mask = self.copy()
        new_mask |= other
        return new_mask
