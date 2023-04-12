#!/usr/bin/env python3

import PySimpleGUI as sg
import PIL.Image
import io
import sys
import tift
import numpy as np
from image import Image

sg.theme('DarkAmber')

img_paths = sys.argv[1:4] if len(sys.argv) >= 4 else [sys.argv[1]] * 3
imgs: list[Image] = [tift.load_image(f) for f in img_paths]
imgs: list[np.ndarray] = [
    img.converted(np.float32).normalized().scaled(255).converted(np.uint8).data
    for img in imgs
]


def get_mri_images(coronal_coordinate,
                   sagittal_coordinate,
                   axial_coordinate,
                   show_crosshairs=True):
    coronal_coordinate -= 1
    sagittal_coordinate -= 1
    axial_coordinate -= 1

    coronal_rgb = np.stack([imgs[0][coronal_coordinate, :, :]] * 3, axis=-1)
    saggital_rgb = np.stack([imgs[1][:, sagittal_coordinate, :]] * 3, axis=-1)
    axial_rgb = np.stack([imgs[2][:, :, axial_coordinate]] * 3, axis=-1)

    if show_crosshairs:
        coronal_rgb[sagittal_coordinate, :, :] = 255
        coronal_rgb[:, axial_coordinate, :] = 255
        saggital_rgb[coronal_coordinate, :, :] = 255
        saggital_rgb[:, axial_coordinate, :] = 255
        axial_rgb[coronal_coordinate, :, :] = 255
        axial_rgb[:, sagittal_coordinate, :] = 255

    coronal_image = PIL.Image.fromarray(coronal_rgb)
    sagittal_image = PIL.Image.fromarray(saggital_rgb)
    axial_image = PIL.Image.fromarray(axial_rgb)

    return coronal_image, sagittal_image, axial_image


def convert_to_bytes(image):
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    return bio.getvalue()


# GUI layout
layout = [[
    sg.Checkbox("Show crosshairs", default=True, key="show_crosshairs")
],
          [
              sg.Column([[sg.Image(key="coronal_image")],
                         [
                             sg.Slider(range=(1, 256),
                                       orientation="h",
                                       key="coronal_slider",
                                       enable_events=True,
                                       default_value=128)
                         ]]),
              sg.Column([[sg.Image(key="sagittal_image")],
                         [
                             sg.Slider(range=(1, 256),
                                       orientation="h",
                                       key="sagittal_slider",
                                       enable_events=True,
                                       default_value=128)
                         ]]),
              sg.Column([[sg.Image(key="axial_image")],
                         [
                             sg.Slider(range=(1, 256),
                                       orientation="h",
                                       key="axial_slider",
                                       enable_events=True,
                                       default_value=128)
                         ]])
          ]]

window = sg.Window("Color Viewer", layout, finalize=True)

coronal_image, sagittal_image, axial_image = get_mri_images(
    128, 128, 128, True)

window["coronal_image"].update(data=convert_to_bytes(coronal_image))
window["sagittal_image"].update(data=convert_to_bytes(sagittal_image))
window["axial_image"].update(data=convert_to_bytes(axial_image))

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break

    if event in ("coronal_slider", "sagittal_slider", "axial_slider",
                 "show_crosshairs"):
        coronal_coordinate = int(values["coronal_slider"])
        sagittal_coordinate = int(values["sagittal_slider"])
        axial_coordinate = int(values["axial_slider"])
        show_crosshairs = values["show_crosshairs"]

        coronal_image, sagittal_image, axial_image = get_mri_images(
            coronal_coordinate, sagittal_coordinate, axial_coordinate,
            show_crosshairs)

        window["coronal_image"].update(data=convert_to_bytes(coronal_image))
        window["sagittal_image"].update(data=convert_to_bytes(sagittal_image))
        window["axial_image"].update(data=convert_to_bytes(axial_image))

window.close()
