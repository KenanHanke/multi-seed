from rbloom import Bloom
import numpy as np
from image import Image
from dataset import Dataset
import gzip
import logging
import os
import re

TIFT_DTYPE = np.uint16


def load_image(path):
    logging.debug("Loading TIFT image from %s", path)

    dimensions = (256, ) * 3
    with gzip.open(path, "rb") as f:
        raw_data = np.frombuffer(f.read(), dtype=TIFT_DTYPE)
        image = Image(data=raw_data.reshape(dimensions))
    return image


def save_image(image: Image, path):
    if not path.endswith(".img.z"):
        raise ValueError("TIFT image path must end with .img.z")

    # save header
    hdr_path = path[:-6] + ".hdr"  # remove .img.z and add .hdr
    logging.debug("Generating TIFT header at %s", hdr_path)
    create_header(hdr_path)

    # save image
    logging.debug("Saving TIFT image to %s", path)
    with gzip.open(path, "wb") as f:
        f.write(image.data.tobytes())


def load_dataset(folder_path):
    """
    Load fMRI dataset from folder.
    """
    logging.info("Loading dataset from folder %s", folder_path)

    # retrieve all image paths in folder
    pattern = r"f\d{10}\.img\.z"
    listing = os.listdir(folder_path)
    img_paths = [
        os.path.join(folder_path, entry) for entry in listing
        if re.fullmatch(pattern, entry)
    ]
    img_paths.sort()

    time_series_length = len(img_paths)
    dimensions = (256, ) * 3

    # initialize dataset
    dataset = Dataset(dimensions, time_series_length, dtype=TIFT_DTYPE)

    # load images
    for i, img_path in enumerate(img_paths):
        dataset[i] = load_image(img_path)

    return dataset


def save_dataset(dataset: Dataset,
                 folder_path,
                 filename_format="image{one_based_index:010}.img.z"):
    logging.info("Saving dataset to folder %s", folder_path)

    # create folder if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    # save images
    for i in range(dataset.time_series_length):
        img = dataset[i]
        img_path = os.path.join(folder_path,
                                filename_format.format(one_based_index=i + 1))
        save_image(img, img_path)


# This is a generic MPRAGE header file that has been cleaned of all
# uniquely identifying information. Each byte is equivalent to the
# mode of all bytes at that position in hundreds of MPRAGE header
# files. Byte frequency analysis was used to determine that bytes
# 201-204 can be used to store a unique tag, which the function that
# uses this generic byte string generates randomly and inserts at that
# position.
# This reverse engineering was performed in good faith and the author
# has made an effort to be compliant with Article 6 of the EU Software
# Directive (Directive 2009/24/EC), which makes specific allowances for
# reverse engineering for the purpose of interoperability. The author
# will, however, remove all reverse engineered format information if
# requested to do so by a proprietor of TIFT.
_GENERIC_MPRAGE_HEADER = (
    b"\\\x01\x00\x00dsr    \x00\x00\x00" + b"/redacted/origin" +
    b"l(\x00\x00\x00\x00\x00\x00r0" +
    b"\x03\x00\x00\x01\x00\x01\x00\x01\x01\x00\x01\x00\x01\x00\x01\x00" +
    b"mm\x00\x00\x00\x00\x00\x00\x00\x00\x9c\xff\x00\x00\x04\x00\x10" +
    b"\x00\x00\x00\x00\x00\x80\xbf\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?" +
    b"\x00" * 56 + b"SPM compatible    \x00\x00\xee\xf076\x9e\x7f" +
    b"\x00" * 10 +
    b"\x80\xf3\xcd\xfa\xfc\x7f\x00\x00\xf0\xf3\xcd\xbe\xfc\x7f" +
    b"\x00\x00\x00\xd7q\xcb\x8cHl(" + b"\x00" * 16 +
    b"\xf0\xf3\xcd\xbenone    \x00\x7f" + b"\x00" * 13 +
    b"\x04\x01\x80\x00\x80\x00\x80" + b"\x00" * 14 +
    b"P\x00\xcd\xbe\xfc\x7f\x00\x00K\x00\x00\x00\x00\x00\x00\x00" +
    b",\xf9\xcd\xbe\xfc\x00\x00\x00\xb1i\x986\x9e\x7f\x00\x00" +
    b"\x01\x80\xad\xfb\x00\x00\x00\x00,\x00\xcd\xbe" + b"\x00" * 32)


def create_header(path):
    """
    Creates an MPRAGE-equivalent header file at the given path, which
    allows any image with the same filename (excepting its extension)
    to be viewed from inside TIFT as a grayscale 3D image.
    """
    # set up static variable
    if not hasattr(create_header, "filter"):
        create_header.filter = Bloom(100_000, 0.01)  # uses around 100 KB

    # generate a unique tag
    while True:
        tag = np.random.randint(0, 2**32 - 1)
        if tag not in create_header.filter:
            create_header.filter.add(tag)
            break

    header = bytearray(_GENERIC_MPRAGE_HEADER)
    header[201:205] = tag.to_bytes(4, "big")

    with open(path, "wb") as f:
        f.write(header)
