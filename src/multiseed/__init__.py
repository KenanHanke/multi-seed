# src/multiseed/__init__.py

"""
multiseed

.. include:: README.md
"""

import logging
from .image import Image, Mask
from .dataset import Dataset
from .ioutils import load_image, save_image, load_dataset, save_dataset

# suppress numba's excessive debug messages
logging.getLogger("numba").setLevel(logging.WARNING)
