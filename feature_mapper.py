from abc import ABC, abstractmethod
from typing import Iterable
import numpy as np
import sklearn.decomposition
from dataset import Dataset
from reference import Reference


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

    def fit(self,
            datasets: Iterable[Dataset],
            reference: Reference,
            *,
            samples_per_dataset: int,
            rng=None):
        X = np.empty((samples_per_dataset * len(datasets), reference.n_seeds),
                     dtype=np.float32)

        for dataset in datasets:
            points = dataset.extract_mask().sample(samples_per_dataset,
                                                   rng=rng)
