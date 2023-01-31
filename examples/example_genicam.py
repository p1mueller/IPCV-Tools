#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Show case the GenICam class."""

import cv2
import numpy as np

from ipcv_tools.camera import GenICam
from ipcv_tools.ui import CameraUI
from ipcv_tools.viewer import ImageViewer


def _get_frame() -> np.ndarray:
    frame = cam.get_next_element()
    f = np.mean(frame, -1)  # Convert to grayscale
    f = np.round(f).astype("uint8")  # Convert to uint8

    g = mapping[f]
    return frame, f, g


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
    node_map.Gain.set_value(0.0)
    node_map.ExposureTime.set_value(1e4)
    node_map.DecimationHorizontal.set_value(decimation)
    node_map.DecimationVertical.set_value(decimation)
    node_map.Width.set_value(node_map.Width.max)
    node_map.Height.set_value(node_map.Height.max)
    node_map.PixelFormat.set_value("RGB8")

    viewer = CameraUI(
        width, height, ["Original", "Grayscale", "Transformed"], _get_frame
    )
    cam.start()

    viewer.run()
