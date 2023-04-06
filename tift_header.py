import shutil
import glob


def generate_header(path):
    """
    Currently copies headers from templates; will be replaced by
    generating headers from scratch.
    """
    # set up static variables
    if not hasattr(generate_header, "index"):
        generate_header.index = 0
        generate_header.templates = glob.glob("header_templates/*")
    # increment static variable
    else:
        generate_header.index += 1
        generate_header.index %= len(generate_header.templates)

    # copy template to path
    shutil.copyfile(generate_header.templates[generate_header.index], path)
