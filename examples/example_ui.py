#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test case to show case the CameraUI class."""

import sys
from argparse import ArgumentParser

import utilities

from ipcv_tools.pipeline import Pipeline
from ipcv_tools.plotting import HistogramPlotter
from ipcv_tools.ui import CameraUI

parser = ArgumentParser()
parser.add_argument("-d", "--decimation", default=1, type=int)
parser.add_argument("-e", "--exposure", default=1e4, type=float)
parser.add_argument("-g", "--gain", default=0.0, type=float)
parser.add_argument("-m", "--monochrome", action="store_true")
args = parser.parse_args()

decimation = args.decimation
width = 1920 // decimation
height = 1080 // decimation

contour_resampler = utilities.create_contour_resampler()

pipeline = Pipeline(
    contour_resampler.__call__,
    width=width,
    height=height,
    use_ui=True,
    img_names=contour_resampler.output_names,
)

if isinstance(pipeline.viewer, CameraUI):
    utilities.initialize_controls(pipeline.viewer, contour_resampler)
    hist_plotter = HistogramPlotter(0, args.monochrome)
    pipeline.viewer.add_plot("Histogram", hist_plotter)

ret = pipeline.run(
    {
        "decimation": args.decimation,
        "width": width,
        "height": height,
        "exposure": args.exposure,
        "gain": args.gain,
    }
)
sys.exit(ret)
