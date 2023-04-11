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
    def fit(self, *, datasets: Iterable[Dataset], reference: Reference, rng=None):
        ...

    @abstractmethod
    def transform(self, X):
        ...


class TranslatedPCA(FeatureMapper):
    """
    A mapping that performs PCA on the dataset. No prior standardization
    is performed because all dimensions are correlation coefficients.

    Uses sklearn's PCA implementation, which automatically centers the data.
    """
