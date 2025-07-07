#!/usr/bin/env python3

import FreeSimpleGUI as sg
import PIL.Image, PIL.ImageOps
import io
import sys
import tift
import numpy as np
from image import Image

# Scale factor for the images (nearest neighbor interpolation)
SCALE_FACTOR = 1

######################################################################
# I FORMALLY APOLOGIZE TO ANYONE WHO EVER HAS TO DECIPHER ANY CODE   #
# PAST THIS POINT. THIS PROGRAM WAS MEANT AS A QUICK AND DIRTY PROOF #
# OF CONCEPT IMAGE VIEWER. IT DOES WORK FULLY AS INTENDED, HOWEVER.  #
######################################################################

sg.theme('DarkAmber')

img_paths = sys.argv[1:4] if len(sys.argv) >= 4 else [sys.argv[1]] * 3
imgs: list[Image] = [tift.load_image(f) for f in img_paths]
imgs: list[np.ndarray] = [
    img.converted(np.float32).normalized().scaled(255).converted(np.uint8).data
    for img in imgs
]
imgs = [img[::-1, ::-1, ::-1] for img in imgs]


def get_mri_images(axial_coordinate,
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


def convert_to_bytes(image):
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    return bio.getvalue()


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

axial_image, coronal_image, sagittal_image = get_mri_images(
    128, 128, 128, True)

window["axial_image"].update(data=convert_to_bytes(axial_image))
window["coronal_image"].update(data=convert_to_bytes(coronal_image))
window["sagittal_image"].update(data=convert_to_bytes(sagittal_image))

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

        axial_image, coronal_image, sagittal_image = get_mri_images(
            axial_coordinate, coronal_coordinate, sagittal_coordinate,
            show_crosshairs)

        window["axial_image"].update(data=convert_to_bytes(axial_image))
        window["coronal_image"].update(data=convert_to_bytes(coronal_image))
        window["sagittal_image"].update(data=convert_to_bytes(sagittal_image))

window.close()
