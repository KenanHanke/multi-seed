# multiseed/__init__.py

import logging


# suppress numba's excessive debug messages
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)
