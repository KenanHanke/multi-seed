# src/multiseed/__init__.py

"""
# multiseed

The `multiseed` library exposes an extensive API encompassing almost all parts of its algorithm.

A few select functions and classes are also imported into the top-level namespace for convenience:
- `Image`: the main image class, representing a 3D MRI image (see `multiseed.image.Image`).
- `Mask`: a subclass of `Image` representing a binary mask (see `multiseed.image.Mask`).
- `Dataset`: a class representing a dataset, which is a collection of images and masks from a single scan (see `multiseed.dataset.Dataset`).
- `load_image` and `save_image`: functions for loading and saving images (see `multiseed.ioutils.load_image` and `multiseed.ioutils.save_image`).
- `load_dataset` and `save_dataset`: functions for loading and saving datasets (see `multiseed.ioutils.load_dataset` and `multiseed.ioutils.save_dataset`).

See the submodule documentation pages for details.
"""

import logging
from .image import Image, Mask
from .dataset import Dataset
from .ioutils import load_image, save_image, load_dataset, save_dataset

# suppress numba's excessive debug messages
logging.getLogger("numba").setLevel(logging.WARNING)
