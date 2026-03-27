# multiseed/__init__.py

import logging
from .image import Image, Mask
from .dataset import Dataset
from .ioutils import load_image, save_image, load_dataset, save_dataset

# suppress numba's excessive debug messages
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)
