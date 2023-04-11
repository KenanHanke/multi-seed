from rbloom import Bloom
import numpy as np

TIFT_DTYPE = np.uint16


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
    b"\\\x01\x00\x00dsr    \x00\x00\x00"
    + b"/redacted/origin"
    + b"l(\x00\x00\x00\x00\x00\x00r0"
    + b"\x03\x00\x00\x01\x00\x01\x00\x01\x01\x00\x01\x00\x01\x00\x01\x00"
    + b"mm\x00\x00\x00\x00\x00\x00\x00\x00\x9c\xff\x00\x00\x04\x00\x10"
    + b"\x00\x00\x00\x00\x00\x80\xbf\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?"
    + b"\x00" * 56
    + b"SPM compatible    \x00\x00\xee\xf076\x9e\x7f"
    + b"\x00" * 10
    + b"\x80\xf3\xcd\xfa\xfc\x7f\x00\x00\xf0\xf3\xcd\xbe\xfc\x7f"
    + b"\x00\x00\x00\xd7q\xcb\x8cHl("
    + b"\x00" * 16
    + b"\xf0\xf3\xcd\xbenone    \x00\x7f"
    + b"\x00" * 13
    + b"\x04\x01\x80\x00\x80\x00\x80"
    + b"\x00" * 14
    + b"P\x00\xcd\xbe\xfc\x7f\x00\x00K\x00\x00\x00\x00\x00\x00\x00"
    + b",\xf9\xcd\xbe\xfc\x00\x00\x00\xb1i\x986\x9e\x7f\x00\x00"
    + b"\x01\x80\xad\xfb\x00\x00\x00\x00,\x00\xcd\xbe"
    + b"\x00" * 32
)


def create_header(path):
    """
    Creates an MPRAGE-equivalent header file at the given path, which
    allows any image with the same filename (excepting its extension)
    to be viewed from inside TIFT as a grayscale 3D image.
    """
    # set up static variable
    if not hasattr(create_header, "filter"):
        create_header.filter = Bloom(1000000, 0.01)  # uses around 1 MB of memory

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
