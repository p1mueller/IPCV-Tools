#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test case to show case the CameraUI class."""

import sys
from argparse import ArgumentParser

from example_viewer import ContourResampler

from ipcv_tools.pipeline import Pipeline
from ipcv_tools.plotting import HistogramPlotter
from ipcv_tools.ui import CameraUI

factor = 0.4
abs_val = 11
sigma = 3.0
use_percent = True
reduce_bp = False

parser = ArgumentParser()
parser.add_argument("-d", "--decimation", default=1, type=int)
parser.add_argument("-e", "--exposure", default=1e4, type=float)
parser.add_argument("-g", "--gain", default=0.0, type=float)
parser.add_argument("-m", "--monochrome", action="store_true")
args = parser.parse_args()

decimation = args.decimation
width = 1920 // decimation
height = 1080 // decimation

contour_resampler = ContourResampler(
    rel_coeffs=factor,
    abs_coeffs=abs_val,
    sigma=sigma,
    use_percent=use_percent,
    reduce_boundary_points=reduce_bp,
)

pipeline = Pipeline(
    contour_resampler.__call__,
    width=width,
    height=height,
    use_ui=True,
    img_names=contour_resampler.output_names,
)

if isinstance(pipeline.viewer, CameraUI):
    pipeline.viewer.add_checkbox(
        "Use %",
        func=contour_resampler.set_use_percent,
        value=contour_resampler.use_percent,
    )
    pipeline.viewer.add_checkbox(
        "Reduce Boundary Points",
        func=contour_resampler.set_reduce_boundary_points,
        value=contour_resampler.reduce_boundary_points,
    )
    pipeline.viewer.add_slider(
        "Sigma",
        1.0,
        10.0,
        101,
        300,
        10,
        contour_resampler.set_sigma,
        value=contour_resampler.sigma,
    )
    pipeline.viewer.add_slider(
        "Absolute Coeffs",
        1,
        101,
        51,
        100,
        10,
        contour_resampler.set_abs_coeffs,
        value=contour_resampler.abs_coeffs,
    )
    pipeline.viewer.add_slider(
        "Coeffs %",
        0.0,
        1.0,
        501,
        300,
        10,
        contour_resampler.set_rel_coeffs,
        value=contour_resampler.rel_coeffs,
    )

    hist_plotter = HistogramPlotter(0, args.monochrome)
    graph = pipeline.viewer.add_plot("Histogram", hist_plotter)
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
