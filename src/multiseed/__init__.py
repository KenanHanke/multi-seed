# src/multiseed/__init__.py

"""
# multiseed

The `multiseed` library exposes an extensive API encompassing almost all parts of its algorithm.
See the submodule documentation for details.
"""

import logging
from .image import Image, Mask
from .dataset import Dataset
from .ioutils import load_image, save_image, load_dataset, save_dataset

# suppress numba's excessive debug messages
logging.getLogger("numba").setLevel(logging.WARNING)
