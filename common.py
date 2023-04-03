import logging
from dataset import Dataset


def create_combined_mask(mask_path, dataset_generator: iter[Dataset]):
    """
    Accumulate a mask from the given datasets.
    """
    logging.info('Creating combined mask from datasets')
    first = next(dataset_generator)
    mask = first.extract_mask()
    for dataset in dataset_generator:
        mask |= dataset.extract_mask()
    mask.save(mask_path)
