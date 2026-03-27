# src/multiseed/dataset.py

from collections.abc import Callable, Iterable, Iterator
import logging

import numpy as np

from .image import Image, Mask


class Dataset:
    """In-memory 4D dataset of voxel time series.

    The underlying array is stored with shape `(*dimensions, n_images)`, so the
    final axis corresponds to image/time index and each voxel's full time series is
    contiguous in memory.
    """

    def __init__(
        self,
        dimensions: tuple[int, ...] | None = None,
        n_images: int | None = None,
        *,
        dtype=None,
        data=None,
    ) -> None:
        """Create a dataset from an existing array or allocate a new one.

        Args:
            dimensions: Spatial shape of the dataset, excluding the image axis.
            n_images: Number of images/time points to allocate.
            dtype: NumPy dtype for a newly allocated dataset.
            data: Existing array with shape `(*dimensions, n_images)`.

        Raises:
            ValueError: If `data` is not provided and the allocation parameters
                are incomplete.
        """
        if data is not None:
            self.data = data
        else:
            if dimensions is None or dtype is None:
                raise ValueError("Either data or the other parameters must be given")

            # Keep the image/time axis last so each voxel time series is contiguous
            # in memory, which reduces cache misses during analysis.
            self.data = np.zeros((*dimensions, n_images), dtype=dtype)

    @classmethod
    def load(cls, file_path: str) -> "Dataset":
        """Load a dataset from a compressed NumPy archive.

        The archive must contain a `data` array with shape
        `(*dimensions, n_images)`.

        Args:
            file_path: Path to the `.npz` file.

        Returns:
            The loaded dataset.
        """
        logging.info("Loading dataset from %s", file_path)
        data = np.load(file_path)["data"]
        return cls(data=data)

    def save(self, file_path: str) -> None:
        """Save the dataset as a compressed NumPy archive.

        Args:
            file_path: Destination path for the `.npz` file.
        """
        logging.info("Saving dataset to %s", file_path)
        np.savez_compressed(file_path, data=self.data)

    def extract_mask(self) -> Mask:
        """Return a mask of voxels with any nonzero value across images.

        Returns:
            A boolean mask with the dataset's spatial dimensions.
        """
        mask = Mask(self.dimensions)
        mask.data[:] = np.any(self.data, axis=-1)
        return mask

    @property
    def dimensions(self) -> tuple[int, ...]:
        """Spatial dimensions of the dataset, excluding the image axis."""
        return self.data.shape[:-1]

    @property
    def n_images(self) -> int:
        """Number of images/time points in the dataset."""
        return self.data.shape[-1]

    @property
    def dtype(self):
        """NumPy dtype of the underlying dataset array."""
        return self.data.dtype

    def __len__(self) -> int:
        """Return the number of images/time points."""
        return self.n_images

    def __getitem__(self, key: int) -> Image:
        """Return the image at a single integer index.

        Args:
            key: Zero-based image index.

        Returns:
            The selected image view wrapped as an `Image`.

        Raises:
            TypeError: If `key` is not an integer.
        """
        if not isinstance(key, int):
            raise TypeError("Only integer indexing is supported")

        return Image(data=self.data[..., key])

    def __setitem__(self, key: int, value: Image) -> None:
        """Replace the image at a single integer index.

        Args:
            key: Zero-based image index.
            value: Image whose data will be written into the dataset.

        Raises:
            TypeError: If `key` is not an integer.
            ValueError: If the image dtype does not match the dataset dtype.
        """
        if not isinstance(key, int):
            raise TypeError("Only integer indexing is supported")

        if self.data.dtype != value.dtype:
            raise ValueError("Data type mismatch")

        self.data[..., key] = value.data

    def copy(self) -> "Dataset":
        """Return a deep copy of the dataset."""
        new_dataset = self.__class__(
            self.dimensions,
            self.n_images,
            dtype=self.dtype,
        )
        new_dataset.data[:] = self.data
        return new_dataset


class DatasetLoader(Iterable[Dataset]):
    """Reusable lazy iterable over datasets.

    `DatasetLoader` defers dataset construction to `generator_func` and can be
    iterated multiple times. It is intended to be created by internal helper
    functions such as `load_datasets_lazy()` rather than instantiated directly by
    callers.
    """

    def __init__(
        self,
        paths: list[str],
        generator_func: Callable[[list[str]], Iterator[Dataset]],
    ) -> None:
        """Initialize a lazy dataset loader.

        Args:
            paths: Dataset source paths that will be passed to `generator_func`.
            generator_func: Callable that returns an iterator yielding datasets for
                the provided paths.
        """
        self.paths = paths
        self.generator_func = generator_func

    def __iter__(self) -> Iterator[Dataset]:
        """Yield datasets lazily for the configured paths."""
        return self.generator_func(self.paths)

    def __len__(self) -> int:
        """Return the number of dataset paths managed by this loader."""
        return len(self.paths)

    def __iadd__(self, other: "DatasetLoader") -> "DatasetLoader":
        """Append another loader's paths to this loader in place."""
        self.paths += other.paths
        return self

    def __add__(self, other: "DatasetLoader") -> "DatasetLoader":
        """Return a new loader covering the concatenated path lists."""
        paths = self.paths + other.paths
        return DatasetLoader(paths, self.generator_func)

    def extract_mask(self) -> Mask:
        """Accumulate the union mask across all lazily loaded datasets.

        Returns:
            A mask containing every voxel that is nonzero in at least one dataset.

        Raises:
            ValueError: If the loader contains no datasets.
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