#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Show case the GenICam class."""

import sys
from argparse import ArgumentParser

import utilities

import ipcv_tools.camera as ipcam
from ipcv_tools.pipeline import Pipeline
from ipcv_tools.plotting import HistogramPlotter
from ipcv_tools.ui import CameraUI

parser = ArgumentParser()
parser.add_argument("-d", "--decimation", default=1, type=int)
parser.add_argument("-p", "--path", default="ahornblatt", type=str)
parser.add_argument("-m", "--monochrome", action="store_true")
parser.add_argument("-r", "--rel_std", default=0, type=float)
args = parser.parse_args()

decimation = args.decimation

contour_resampler = utilities.create_contour_resampler()

if args.rel_std <= 0:
    camera = ipcam.DataCam(args.path, not args.monochrome)
else:
    camera = ipcam.NoisyDataCam(args.path, not args.monochrome)
height, width = camera.get_shape()

pipeline = Pipeline(
    contour_resampler.__call__,
    width=width,
    height=height,
    use_ui=True,
    camera=camera,
    img_names=contour_resampler.output_names,
)

if isinstance(pipeline.viewer, CameraUI):
    utilities.initialize_controls(pipeline.viewer, contour_resampler)
    hist_plotter = HistogramPlotter(0, args.monochrome)
    pipeline.viewer.add_plot("Histogram", hist_plotter)

ret = pipeline.run({"decimation": args.decimation, "rel_std": args.rel_std})
sys.exit(ret)
