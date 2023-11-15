#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test case to show case the ImageViewer class."""

from argparse import ArgumentParser

import utilities

from ipcv_tools.pipeline import Pipeline

parser = ArgumentParser()
parser.add_argument("-d", "--decimation", default=1, type=int)
parser.add_argument("-c", "--coefficients", default=0.4, type=float)
parser.add_argument("-s", "--sigma", default=3.0, type=float)
parser.add_argument("-p", "--port", default=None)
args = parser.parse_args()

original_width = 1920
original_height = 1080
height = original_height // args.decimation
width = original_width // args.decimation

contour_resampler = utilities.ContourResampler(args.coefficients, args.sigma)
pipeline = Pipeline(
    contour_resampler.__call__,
    port=args.port,
    width=width,
    height=height,
    use_ui=False,
)
pipeline.run({"decimation": args.decimation, "width": width, "height": height})
