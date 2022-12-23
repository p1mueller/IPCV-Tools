#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test case to show case the CameraUI class."""

import sys

import numpy as np
from example_viewer import ContourResampler

from ipcv_tools.pipeline import Pipeline
from ipcv_tools.plotting import HistogramPlotter
from argparse import ArgumentParser

factor = 0.4
sigma = 3.0

parser = ArgumentParser()
parser.add_argument("-d", "--decimation", default=1, type=int)
parser.add_argument("-g", "--grayscale", action="store_true")
args = parser.parse_args()

decimation = args.decimation
width = 1920 // decimation
height = 1080 // decimation

contour_resampler = ContourResampler(factor, sigma)

pipeline = Pipeline(
    contour_resampler.__call__,
    width=width,
    height=height,
    use_ui=True,
    img_names=contour_resampler.output_names,
)
pipeline.viewer.add_slider(
    "Sigma", 1.0, 10.0, 101, 300, 10, contour_resampler.set_sigma, value=sigma
)
pipeline.viewer.add_slider(
    "Coeffs %", 0.0, 1.0, 101, 300, 10, contour_resampler.set_keep_coeffs, value=factor
)

hist_plotter = HistogramPlotter(0, args.grayscale)
graph = pipeline.viewer.add_plot("Histogram", hist_plotter)

ret = pipeline.run()
sys.exit(ret)
