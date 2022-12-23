#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Show case the GenICam class."""

import numpy as np

from ipcv_tools.camera import GenICam
from ipcv_tools.viewer import ImageViewer


def _get_frame() -> np.ndarray:
    frame = cam.get_next_element()
    f = np.mean(frame, -1)  # Convert to grayscale
    f = np.round(f).astype("uint8")  # Convert to uint8

    g = mapping[f]
    res_img = np.repeat(np.expand_dims(g, -1), 3, -1)
    return res_img


port = None
factor = 0.4
decimation = 2
original_height = 1536
original_width = 2048
height = original_height // decimation
width = original_width // decimation

gamma = 0.5
r = np.arange(256)
mapping = 255 ** (1 - gamma) * r**gamma  # gamma
# transform = 255 * (0.5 - 0.5 * np.cos(2 * np.pi * r / 255))  # Hann
mapping = np.array(mapping, dtype="uint8")

with GenICam() as cam:
    assert cam.handler is not None
    node_map = cam.handler.remote_device.node_map
    node_map.DecimationHorizontal.value = decimation
    node_map.DecimationVertical.value = decimation
    node_map.Width.value = width
    node_map.Height.value = height
    node_map.PixelFormat.value = "RGB8"

    viewer = ImageViewer(height, 2 * width, _get_frame, font={"color": (0, 0, 255)})
    cam.start()

    viewer.run()
