import shutil
import glob
from common import ROOT
import os


def generate_header(path):
    """
    Generates an MPRAGE-equivalent header, which allows any image
    file named identically (except for its extension) to be viewed
    from inside TIFT as a grayscale 3D image.

    Currently copies headers from templates; will ideally be
    replaced by generating headers from scratch.
    """
    # set up static variables
    if not hasattr(generate_header, "index"):
        generate_header.index = 0
        template_folder = os.path.join(ROOT, "header_templates", "*")
        generate_header.templates = glob.glob(template_folder)
    # increment static variable
    else:
        generate_header.index += 1
        generate_header.index %= len(generate_header.templates)

    # copy template to path
    shutil.copyfile(generate_header.templates[generate_header.index], path)
