from abc import ABC, abstractmethod
from typing import Iterable
import numpy as np
import sklearn.decomposition
from dataset import Dataset
from reference import Reference
from math_utils import UntranslatedPCAInterface
from functools import reduce
from operator import mul


class FeatureMapper(ABC):
    """
    An abstract base class that defines the interface for feature extraction
    models that map the high-dimensional absolute value correlation coefficient
    vector space to a lower-dimensional feature space.
    """

    @abstractmethod
    def __init__(self, n_features):
        """
        Initialize the feature extractor with the number of features
        it should extract.

        Args:
            n_features: The number of features to extract.
        """
        ...

    @abstractmethod
    def fit(self,
            datasets: Iterable[Dataset],
            reference: Reference,
            *,
            samples_per_dataset: int,
            rng=None):
        ...

    @abstractmethod
    def transform(self, dataset: Dataset, reference: Reference) -> Dataset:
        ...


class TranslatedPCA(FeatureMapper):
    """
    A mapping that performs PCA on the dataset. No prior standardization
    is performed because all dimensions are correlation coefficients.

    Uses sklearn's PCA implementation, which automatically centers the data.
    """

    def __init__(self, n_components):
        self.pca = sklearn.decomposition.PCA(
            n_components=n_components
        )  # use svd_solver='full' for more accuracy but worse performance

    ...


class UntranslatedPCA(FeatureMapper):

    def __init__(self, n_components):
        self.pca = UntranslatedPCAInterface(n_components)
        self.n_features = n_components

    def fit(self,
            datasets: Iterable[Dataset],
            reference: Reference,
            *,
            samples_per_dataset: int,
            rng=None):
        X = np.empty((samples_per_dataset * len(datasets), reference.n_seeds),
                     dtype=np.float32)

        for i, dataset in enumerate(datasets):
            points = dataset.extract_mask().sample(samples_per_dataset,
                                                   rng=rng)

            time_series_length = dataset.n_images
            time_series_array = np.empty((len(points), time_series_length),
                                         dtype=dataset.dtype)

            for j, point in enumerate(points):
                time_series_array[j] = dataset.data[point]

            X[i * samples_per_dataset:(i + 1) *
              samples_per_dataset] = reference.apply(time_series_array)

        self.pca.fit(X)

    def transform(self, dataset: Dataset, reference: Reference) -> Dataset:
        float_result = Dataset(dataset.dimensions,
                               self.n_features,
                               dtype=np.float32)
        time_series_arrays = dataset.data.reshape(
            (dataset.dimensions[0], -1, dataset.n_images))
        for i, time_series_array in enumerate(time_series_arrays):
            X = reference.apply(time_series_array)
            float_result.data[i] = self.pca.transform(X).reshape(
                (-1, self.n_features))

        result = Dataset(float_result.dimensions,
                         float_result.n_images,
                         dtype=dataset.dtype)

        for i, image in enumerate(float_result):
            image = image.normalized().scaled(2**16 - 1)
            image = image.converted(dataset.dtype)
            result[i] = image

        return result
