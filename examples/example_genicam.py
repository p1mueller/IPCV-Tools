#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Show case the GenICam class."""

from argparse import ArgumentParser
from typing import Sequence, Union

import numpy as np

from ipcv_tools.camera import GenICam
from ipcv_tools.ui import CameraUI
from ipcv_tools.viewer import ImageViewer


def _get_frame() -> Union[np.ndarray, Sequence[np.ndarray]]:
    frame = cam.get_next_element()
    f = np.mean(frame, -1)  # Convert to grayscale
    f = np.round(f).astype("uint8")  # Convert to uint8

    g = mapping[f]
    if args.viewer:
        output = np.concatenate([frame, np.repeat(g[..., None], 3, -1)], 1)
        return output
    return frame, f, g


parser = ArgumentParser()
parser.add_argument("-d", "--decimation", default=1, type=int)
parser.add_argument("-g", "--gain", default=0.0, type=float)
parser.add_argument("-e", "--exposure", default=1e4, type=float)
parser.add_argument("-v", "--viewer", action="store_true")
args = parser.parse_args()

original_height = 1536
original_width = 2048
height = original_height // args.decimation
width = original_width // args.decimation

gamma = 0.5
r = np.arange(256)
mapping = 255 ** (1 - gamma) * r**gamma  # gamma
# mapping = 255 * (0.5 - 0.5 * np.cos(2 * np.pi * r / 255))  # Hann
mapping = np.array(mapping, dtype="uint8")
with GenICam() as cam:
    assert cam.handler is not None
    node_map = cam.handler.remote_device.node_map
    node_map.Gain.set_value(0.0)
    node_map.ExposureTime.set_value(1e4)
    node_map.DecimationHorizontal.set_value(args.decimation)
    node_map.DecimationVertical.set_value(args.decimation)
    node_map.Width.set_value(node_map.Width.max)
    node_map.Height.set_value(node_map.Height.max)
    node_map.PixelFormat.set_value("RGB8")

    viewer: Union[ImageViewer, CameraUI]
    if args.viewer:
        viewer = ImageViewer(2 * width, height, _get_frame)
    else:
        viewer = CameraUI(
            width, height, ["Original", "Grayscale", "Transformed"], _get_frame
        )
    cam.start()
    viewer.run()
