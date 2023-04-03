import logging
from dataset import Dataset
from concurrent.futures import ThreadPoolExecutor


def create_combined_mask(mask_path, dataset_generator: iter[Dataset]):
    """
    Accumulate a mask from the given datasets.
    """
    logging.info('Creating combined mask from datasets')
    try:
        first = next(dataset_generator)
    except StopIteration:
        raise ValueError('No datasets given')
    mask = first.extract_mask()
    for dataset in dataset_generator:
        mask |= dataset.extract_mask()
    mask.save(mask_path)


def tift_dataset_generator_sync(folder_paths: iter[str]):
    for path in folder_paths:
        yield Dataset.load_tift(path)


def tift_dataset_generator_async(folder_paths: iter[str]):
    """
    Load datasets asynchronously. Uses more memory, but makes sure
    that the next dataset is already loaded when the current one is
    finished being processed.
    """
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

        # yield last dataset
        yield current_dataset.result()
