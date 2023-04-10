import logging
from dataset import Dataset
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable
import gc
from rbloom import Bloom
import numpy as np
from typing import Optional
from image import Mask


def sample_points(n_points: int, *, mask: Optional[Mask], rng=None):
    """
    Sample n_points from the given mask. If no mask is given, sample from the whole space.
    Each point is guaranteed to be unique.

    Args:
        n_points (int): The number of points to sample.
        mask (Mask, optional): A mask object to filter points based on certain criteria.
        rng (np.random.Generator, optional): A random number generator to use for sampling points.

    Returns:
        np.array: An array containing the 3D coordinates of the sampled points. The shape of the array is (n_points, 3).
    """
    logging.info('Sampling %d unique points', n_points)

    points = np.empty((n_points, 3), dtype=np.int16)

    if rng is None:
        rng = np.random.default_rng()

    points_so_far = Bloom(n_points*1000, 0.01)

    i = 0
    while i < n_points:
        point = tuple(rng.integers(mask.dimensions, size=3))
        if mask is None or mask.data[point]:
            if point not in points_so_far:
                points[i] = point
                points_so_far.add(point)
                i += 1

    return points


def create_combined_mask(mask_path, datasets: Iterable[Dataset]):
    """
    Accumulate a mask from the given datasets.
    """
    logging.info('Creating combined mask from datasets')
    datasets = iter(datasets)
    try:
        first = next(datasets)
    except StopIteration:
        raise ValueError('No datasets given')
    mask = first.extract_mask()
    for dataset in datasets:
        mask |= dataset.extract_mask()
    mask.save(mask_path)


def tift_dataset_generator_sync(folder_paths: Iterable[str]):
    folder_paths = sorted(folder_paths)
    for path in folder_paths:
        yield Dataset.load_tift(path)
        gc.collect()  # free memory


def tift_dataset_generator_async(folder_paths: Iterable[str]):
    """
    Load datasets asynchronously. Uses more memory, but makes sure
    that the next dataset is already loaded when the current one is
    finished being processed.
    """
    folder_paths = sorted(folder_paths)

    with ThreadPoolExecutor() as executor:
        folder_paths = iter(folder_paths)

        # load first dataset
        try:
            current_dataset = executor.submit(Dataset.load_tift, next(folder_paths))
        except StopIteration:  # no datasets to load
            return

        for path in folder_paths:
            # wait for current dataset to finish loading
            current_dataset = current_dataset.result()

            # start loading next dataset in background
            next_dataset = executor.submit(Dataset.load_tift, path)

            # yield and continue with next dataset
            yield current_dataset
            current_dataset = next_dataset

            # free memory
            gc.collect()

        # yield last dataset
        yield current_dataset.result()
