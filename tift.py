from concurrent.futures import ThreadPoolExecutor
import gc
from typing import Iterable
import numpy as np
from image import Image
from dataset import Dataset, DatasetLoader
import gzip
import logging
import os
import nibabel as nib


def load_image(path):
    """
    Load image from the specified file path.

    Args:
        path (str): Path to the image file.

    Returns:
        Image: Loaded Image object.
    """
    logging.debug("Loading image from %s", path)

    try:
        # try to load using nibabel's default loader
        data = np.asarray(nib.load(path).dataobj)
    except (FileNotFoundError, nib.filebasedimages.ImageFileError):
        # If nibabel's default loader fails with these errors,
        # it's likely because the image is compressed as a .img.z
        # file, which isn't standard:
        # (1) A FileNotFoundError arises when the .hdr is provided
        # and the image can't be found because a .img is expected
        # and the .img.z is not recognized.
        # (2) An ImageFileError arises when the .img.z is provided
        # directly to nibabel, which expects a .hdr or .img file.
        if path.lower().endswith(".img.z"):
            img_path = path
            hdr_path = path[:-6] + ".hdr"  # remove .img.z and add .hdr
        elif path.lower().endswith(".hdr"):
            img_path = path[:-4] + ".img.z"  # remove .hdr and add .img.z
            hdr_path = path
        else:
            raise

        # Read the image manually from the header and gzipped image file
        hdr_holder = nib.FileHolder(filename=hdr_path)
        with gzip.open(img_path, "rb") as gz_file:
            fmap = {'header': hdr_holder,
                    'image' : nib.FileHolder(fileobj=gz_file)}
            img  = nib.AnalyzeImage.from_file_map(fmap)
            
            # Because    data = img.get_fdata()   results in a float64 array,
            # we use img.dataobj to preserve the original data type
            data = np.asarray(img.dataobj)
        
    image = Image(data=np.array(data))
    return image


def save_image(image: Image, path):
    """
    Save an Image object as an SPM2 Analyze image at the specified file path.

    Args:
        image (Image): Image object to save.
        path (str): Path to save the SPM2 Analyze image file.
    """
    
    data = image.data
    dtype = data.dtype
    
    voxel_sizes = (1.0, 1.0, 1.0)          # each voxel is 1 mm
    affine = np.diag(voxel_sizes + (1.0,))

    img = nib.Spm2AnalyzeImage(data, affine)

    # modify header
    hdr = img.header
    hdr.set_data_dtype(dtype)  # ensures correct on-disk dtype
    hdr.set_zooms(voxel_sizes)

    # save (path can be .hdr or .img)
    logging.debug("Saving image to %s", path)
    nib.save(img, path)


def load_dataset(folder_path):
    """
    Load dataset of images in the Analyze format from the specified
    folder in alphabetical order.

    Args:
        folder_path (str): Path to the folder containing the dataset.

    Returns:
        Dataset: Loaded Dataset object.
    """
    logging.info("Loading dataset from folder %s", folder_path)
    
    EXTENSIONS = (
        ".img",       # Analyze image
        ".img.gz",    # gzipped Analyze image
        ".img.z",     # gzipped Analyze image with alternative extension
        ".nii",       # NIfTI
        ".nii.gz",    # gzipped NIfTI
    )
    image_paths = sorted(
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if filename.lower().endswith(EXTENSIONS)
    )

    n_images = len(image_paths)
    
    # load images
    for i, image_path in enumerate(image_paths):
        image = load_image(image_path)
        if i == 0:
            # Initialize the dataset with the first image's dimensions and dtype
            dataset = Dataset(image.dimensions, n_images, dtype=image.dtype)
        dataset[i] = image

    return dataset


def save_dataset(dataset: Dataset,
                 folder_path,
                 filename_format="image{one_based_index:010}.img"):
    """
    Save a Dataset object to the specified folder. Supports saving as
    files as .img, .img.gz, .nii, .nii.gz depending on the provided
    filename format.

    Args:
        dataset (Dataset): Dataset object to save.
        folder_path (str): Path to save the dataset.
        filename_format (str, optional): Filename format for images in the dataset.
                                         Defaults to "image{one_based_index:010}.img".
    """
    logging.info("Saving dataset to folder %s", folder_path)

    # create folder if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    # save images
    for i in range(dataset.n_images):
        img = dataset[i]
        img_path = os.path.join(folder_path,
                                filename_format.format(one_based_index=i + 1))
        save_image(img, img_path)


# This function returns an iterable that can lazily load many datasets
# given by the argument paths.
def load_datasets_lazy(paths: list, *, asynchronous: bool) -> DatasetLoader:
    """
    Load multiple datasets from the specified paths.
    
    Args:
        paths (list): List of paths to load datasets from.
        asynchronous (bool): Whether to load datasets asynchronously. Setting this to True
                             uses twice as much memory, but lets the respective subsequent
                             dataset load in the background while the current dataset is
                             still being processed.

    Returns:
        DatasetLoader: Iterable that can be reused to load datasets.
    """

    def _dataset_generator_sync(folder_paths: Iterable[str]):
        folder_paths = sorted(folder_paths)
        for path in folder_paths:
            yield load_dataset(path)
            gc.collect()  # free memory

    def _dataset_generator_async(folder_paths: Iterable[str]):
        folder_paths = sorted(folder_paths)

        with ThreadPoolExecutor() as executor:
            folder_paths = iter(folder_paths)

            # load first dataset
            try:
                current_dataset = executor.submit(load_dataset,
                                                  next(folder_paths))
            except StopIteration:  # no datasets to load
                return

            for path in folder_paths:
                # wait for current dataset to finish loading
                current_dataset = current_dataset.result()

                # start loading next dataset in background
                next_dataset = executor.submit(load_dataset, path)

                # yield and continue with next dataset
                yield current_dataset
                current_dataset = next_dataset

                # free memory
                gc.collect()

            # yield last dataset
            yield current_dataset.result()

    if asynchronous:
        generator_func = _dataset_generator_async
    else:
        generator_func = _dataset_generator_sync

    return DatasetLoader(paths, generator_func)
