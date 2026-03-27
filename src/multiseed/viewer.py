#!/usr/bin/env python3

# src/multiseed/viewer.py

"""

Simple, quick and dirty image viewer for viewing 256x256x256 3D MRI images in the
Analyze format, optionally overlaid on each other as RGB channels.

Only supports int16 and float32 images.

Pass it an image file via the command line (ending in ".img" or ".img.z").
Alternatively, pass three images to view them as RGB channels overlaid on each other.

"""

import os
import FreeSimpleGUI as sg
import PIL.Image, PIL.ImageOps
import io
import sys
import numpy as np
import gzip
import tkinter as tk
from .image import Image


######################################################################


def _load_image(path):
    # If the path ends with .hdr, we need to find the corresponding .img or .img.z file
    if path.endswith(".hdr"):
        # find out if the corresponding .img file is gzipped or not
        img_path = path[:-4] + ".img"
        gzipped_img_path = img_path + ".z"
        if os.path.exists(gzipped_img_path):
            path = gzipped_img_path
        elif os.path.exists(img_path):
            path = img_path
        else:
            raise FileNotFoundError(
                f"Could not find corresponding .img or .img.z file for {path}")

    if path.endswith(".z"):
        with gzip.open(path, "rb") as f:
            raw_data = f.read()
    else:
        with open(path, "rb") as f:
            raw_data = f.read()

    if len(raw_data) == 256 * 256 * 256 * 2:
        raw_data = np.frombuffer(raw_data, dtype=np.int16)
    elif len(raw_data) == 256 * 256 * 256 * 4:
        raw_data = np.frombuffer(raw_data, dtype=np.float32)
    else:
        raise ValueError(
            f"Unexpected file size {len(raw_data)} for image data. Expected a 256x256x256 image of type int16 or float32.")

    dimensions = (256, ) * 3
    image = Image(data=raw_data.reshape(dimensions))
    return image


######################################################################
#  THE REMAINDER OF THIS PROGRAM IS DIFFICULT TO UNDERSTAND; IT WAS  #
#  WRITTEN AS A QUICK AND DIRTY PROOF OF CONCEPT.                    #
######################################################################


def _get_mri_images(imgs, axial_coordinate,
                   coronal_coordinate,
                   sagittal_coordinate,
                   show_crosshairs=True):
    axial_coordinate = 256 - axial_coordinate
    coronal_coordinate -= 1
    sagittal_coordinate = 256 - sagittal_coordinate

    axial_rgb = np.stack([imgs[i][axial_coordinate, :, :] for i in range(3)],
                         axis=-1)
    saggital_rgb = np.stack(
        [imgs[i][:, coronal_coordinate, :] for i in range(3)], axis=-1)
    sagittal_rgb = np.stack(
        [imgs[i][:, :, sagittal_coordinate] for i in range(3)], axis=-1)

    if show_crosshairs:
        axial_rgb[coronal_coordinate, :, :] = 255
        axial_rgb[:, sagittal_coordinate, :] = 255
        saggital_rgb[axial_coordinate, :, :] = 255
        saggital_rgb[:, sagittal_coordinate, :] = 255
        sagittal_rgb[axial_coordinate, :, :] = 255
        sagittal_rgb[:, coronal_coordinate, :] = 255

    axial_image = PIL.Image.fromarray(axial_rgb).resize(
        (axial_rgb.shape[1] * SCALE_FACTOR, axial_rgb.shape[0] * SCALE_FACTOR),
        PIL.Image.NEAREST)
    coronal_image = PIL.Image.fromarray(saggital_rgb).resize(
        (saggital_rgb.shape[1] * SCALE_FACTOR,
         saggital_rgb.shape[0] * SCALE_FACTOR), PIL.Image.NEAREST)
    sagittal_image = PIL.Image.fromarray(sagittal_rgb).resize(
        (sagittal_rgb.shape[1] * SCALE_FACTOR,
         sagittal_rgb.shape[0] * SCALE_FACTOR), PIL.Image.NEAREST)

    axial_image = PIL.ImageOps.mirror(axial_image)
    coronal_image = PIL.ImageOps.mirror(coronal_image)

    return axial_image, coronal_image, sagittal_image


def _convert_to_bytes(image):
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    return bio.getvalue()


def main():
    
    # Scale factor for the images (nearest neighbor interpolation)
    # Located within main() to prevent issues with importing tkinter in environments where it may not be available
    global SCALE_FACTOR
    SCALE_FACTOR = 1  # Default value; will be updated based on screen height if possible
    try:
        root = tk.Tk()
        SCALE_FACTOR = root.winfo_screenheight() // 700
    finally:
        try:
            root.destroy()
        except Exception:
            pass
    
    sg.theme('DarkAmber')

    img_paths = sys.argv[1:4] if len(sys.argv) >= 4 else [sys.argv[1]] * 3
    imgs: list[Image] = [_load_image(f) for f in img_paths]
    imgs: list[np.ndarray] = [
        img.converted(np.float32).normalized().scaled(255).converted(np.uint8).data
        for img in imgs
    ]
    imgs = [img[::-1, ::-1, ::-1] for img in imgs]



    # GUI layout
    layout = [[
        sg.Checkbox("Show crosshairs",
                    default=True,
                    key="show_crosshairs",
                    enable_events=True),
    ],
            [
                sg.Column([[sg.Image(key="coronal_image")],
                            [
                                sg.Slider(range=(1, 256),
                                        orientation="h",
                                        key="coronal_slider",
                                        enable_events=True,
                                        default_value=128,
                                        expand_x=True)
                            ]]),
                sg.Column([[sg.Image(key="sagittal_image")],
                            [
                                sg.Slider(range=(1, 256),
                                        orientation="h",
                                        key="sagittal_slider",
                                        enable_events=True,
                                        default_value=128,
                                        expand_x=True)
                            ]]),
                sg.Column([[sg.Image(key="axial_image")],
                            [
                                sg.Slider(range=(1, 256),
                                        orientation="h",
                                        key="axial_slider",
                                        enable_events=True,
                                        default_value=128,
                                        expand_x=True)
                            ]])
            ]]

    window = sg.Window("Image Viewer", layout, finalize=True)

    axial_image, coronal_image, sagittal_image = _get_mri_images(
        imgs, 128, 128, 128, True)

    window["axial_image"].update(data=_convert_to_bytes(axial_image))
    window["coronal_image"].update(data=_convert_to_bytes(coronal_image))
    window["sagittal_image"].update(data=_convert_to_bytes(sagittal_image))

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break

        if event in ("axial_slider", "coronal_slider", "sagittal_slider",
                    "show_crosshairs"):
            axial_coordinate = int(values["axial_slider"])
            coronal_coordinate = int(values["coronal_slider"])
            sagittal_coordinate = int(values["sagittal_slider"])
            show_crosshairs = values["show_crosshairs"]

            axial_image, coronal_image, sagittal_image = _get_mri_images(
                imgs, axial_coordinate, coronal_coordinate, sagittal_coordinate,
                show_crosshairs)

            window["axial_image"].update(data=_convert_to_bytes(axial_image))
            window["coronal_image"].update(data=_convert_to_bytes(coronal_image))
            window["sagittal_image"].update(data=_convert_to_bytes(sagittal_image))

    window.close()

    return 0

if __name__ == "__main__":
    sys.exit(main())
