import numpy as np
import logging


class Image:

    def __init__(self,
                 dimensions: tuple[int] = None,
                 *,
                 dtype=None,
                 data=None):
        if data is not None:
            self.data = data
        else:
            if dimensions is None or dtype is None:
                raise ValueError(
                    "Either data or dimensions and dtype must be given")
            self.data = np.zeros(dimensions, dtype=dtype)

    @classmethod
    def load(cls, path):
        logging.debug("Loading image from %s", path)
        data = np.load(path)["data"]
        return cls(data=data)

    def save(self, path):
        logging.debug("Saving image to %s", path)
        np.savez_compressed(path, data=self.data)

    def normalized(self) -> "Image":
        if not np.issubdtype(self.dtype, np.floating):
            raise TypeError("Image must be floating-point to normalize")

        minimum = np.min(self.data)
        maximum = np.max(self.data)

        new_data = (self.data - minimum) / (maximum - minimum)
        return self.__class__(data=new_data)

    def scaled(self, factor: float) -> "Image":
        new_data = self.data * factor
        return self.__class__(data=new_data)

    def converted(self, dtype) -> "Image":
        new_data = self.data.astype(dtype)
        return self.__class__(data=new_data)

    @classmethod
    def normalize_all(cls, images: list["Image"]) -> list["Image"]:
        minimum = min(np.min(image.data) for image in images)
        maximum = max(np.max(image.data) for image in images)

        return [
            image.__class__(data=(image.data - minimum) / (maximum - minimum))
            for image in images
        ]

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

    def __init__(self,
                 dimensions: tuple[int] = None,
                 *,
                 dtype=bool,
                 data=None):
        super().__init__(dimensions, dtype=dtype, data=data)

    def __ior__(self, other: "Mask"):
        self.data |= other.data
        return self

    def __or__(self, other: "Mask"):
        new_mask = self.copy()
        new_mask |= other
        return new_mask

    def normalized(self):
        raise NotImplementedError("Mask cannot be normalized")

    def scaled(self, factor):
        raise NotImplementedError("Mask cannot be scaled")

    def converted(self, dtype):
        raise NotImplementedError("Mask cannot be converted")

    @classmethod
    def normalize_all(cls, images):
        raise NotImplementedError("Masks cannot be normalized")