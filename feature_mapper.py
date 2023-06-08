from abc import ABC, abstractmethod
from typing import Iterable
import numpy as np
import sklearn.decomposition
from dataset import Dataset
from reference import ReferenceBuilder
import logging


def name_to_mapper_class(mapper_name) -> type['FeatureMapper']:
    match mapper_name:
        case "PCA":
            return PCAMapper
        case _:
            raise ValueError("Inappropriate mapper algorithm name")


class FeatureMapper(ABC):
    """
    An abstract base class that defines the interface for feature extraction
    models that map the high-dimensional absolute value correlation coefficient
    vector space to a lower-dimensional feature space.
    """

    def __init__(self, n_features, reference_builder: ReferenceBuilder):
        """
        Initialize the feature extractor with the number of features
        it should extract.

        Args:
            n_features: The number of features to extract.
        """
        self.n_features = n_features
        self.reference_builder = reference_builder
        self.reduction_model: type = self.reduction_impl(n_features)

    def fit(self,
            datasets: Iterable[Dataset],
            *,
            samples_per_dataset: int,
            rng=None):
        """
        Fit the model to the given datasets.
        
        Args:
            datasets: The datasets to fit the model to.
            samples_per_dataset: The number of samples to extract from each dataset.
            rng: The random number generator to use.
        """

        # initialize the model
        X = np.empty(
            (samples_per_dataset * len(datasets),
             self.reference_builder.n_seeds),
            dtype=np.float32,
        )

        logging.info("Extracting samples from datasets...")
        for i, dataset in enumerate(datasets):
            points = dataset.extract_mask().sample(samples_per_dataset,
                                                   rng=rng)

            time_series_length = dataset.n_images
            time_series_collection = np.empty(
                (len(points), time_series_length), dtype=dataset.dtype)

            for j, point in enumerate(points):
                point = tuple(point)
                # copy the time series at the point into
                # the time_series_collection at index j
                time_series_collection[j] = dataset.data[point]

            reference = self.reference_builder.build(dataset)
            res = reference.apply(time_series_collection)

            # copy the result into X
            X[i * samples_per_dataset:(i + 1) * samples_per_dataset] = res

        logging.info("Fitting model "
                     f"({self.reduction_model.__class__.__name__})...")
        self.reduction_model.fit(X)

    def transform(self, dataset: Dataset) -> Dataset:
        float_result = Dataset(dataset.dimensions,
                               self.n_features,
                               dtype=np.float32)
        """
        Transform the given dataset into the feature space defined by the
        fitted model.
        
        Args:
            dataset: The dataset to transform.
        """
        reference = self.reference_builder.build(dataset)

        # essentially a bunch of slices, each as raw data
        # (no coordinate info; second dimension is merely time series length)
        time_series_collections = dataset.data.reshape((
            dataset.dimensions[0],
            -1,
            dataset.n_images,
        ))

        logging.info("Transforming dataset...")
        for i, time_series_collection in enumerate(time_series_collections):
            logging.debug(f"Transforming slice {i+1}/"
                          f"{len(time_series_collections)}...")
            X = reference.apply(time_series_collection)

            float_result.data[i] = self.reduction_model.transform(X).reshape((
                *float_result.dimensions[1:],
                self.n_features,
            ))

        logging.debug("Converting result to original data type...")
        result = Dataset(float_result.dimensions,
                         float_result.n_images,
                         dtype=dataset.dtype)

        for i, image in enumerate(float_result):
            image = image.normalized().scaled(2**16 - 1)
            image = image.converted(dataset.dtype)
            result[i] = image

        return result

    @property
    @abstractmethod
    def reduction_impl(self):
        """
        Returns the class of the reduction model to use.
        
        Returns:
            The class of the reduction model to use.
        """
        pass


class PCAMapper(FeatureMapper):
    """
    A mapping that performs PCA on the dataset. No prior standardization
    is performed because all dimensions are correlation coefficients.

    Uses sklearn's PCA implementation, which automatically centers the data.
    """

    @property
    def reduction_impl(self):
        return sklearn.decomposition.PCA
