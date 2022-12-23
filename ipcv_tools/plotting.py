#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plotting utilities for the camera UI."""

from abc import ABC, abstractmethod
import warnings

import numpy as np
import pyqtgraph as pg
from skimage.exposure import histogram


def compute_histogram(img, grayscale=False):
    """Compute the histogram for an RGB image.

    Args:
        img (np.ndarray): RGB image

    Returns:
        np.ndarray: Center points of bins
        np.ndarray: Histogram values
    """
    x = np.arange(256)
    channel_axis = -1 if (img.ndim > 2) and not grayscale else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist, _ = histogram(img, 256, "dtype", True, channel_axis=channel_axis)
    return x, np.atleast_2d(hist)


class Plotter(ABC):
    """Abstract class to plot within the UI."""

    def __init__(self) -> None:
        """Initialize.

        Args:
            graph (PlotWidget): Plot
        """
        self.graph = None

    def set_graph(self, graph, example):
        """Set graph object.

        Args:
            graph: Graph
            example: Example image to initialize graph.
        """
        self.graph = graph
        self.init_graph(example)

    @abstractmethod
    def init_graph(self, example):
        """Initialize graph with line dummies.

        Args:
            graph: Graph
            example: Example image to initialize graph.
        """
        pass

    @abstractmethod
    def update(self, *args):
        """Update plot."""
        pass


class HistogramPlotter(Plotter):
    """Plot histogram of a frame."""

    def __init__(self, frame_index=0, grayscale=False):
        """Initialize.

        Args:
            frame_index (int): Index of frame to use. Defaults to 0
            grayscale (bool): Make histogram for grayscale images. Defaults to False.
        """
        super().__init__()
        self.frame_index = frame_index
        self.grayscale = grayscale

    def init_graph(self, example):
        """Initialize graph with line dummies.

        Args:
            graph: Graph
            example: Example image to initialize graph.
        """
        assert self.graph is not None
        self.lines = []
        self.graph.addLegend()

        x, hists = compute_histogram(example, self.grayscale)
        if hists.shape[0] == 1:
            self.lines.append(self.graph.plot(x, hists[0]))
        else:
            for hist_oc, color in zip(hists, ["r", "g", "b"]):
                pen = pg.mkPen(color=color)
                self.lines.append(
                    self.graph.plot(x, hist_oc, name=color.upper(), pen=pen)
                )

    def _ensure_dim(self, img):
        if self.grayscale and (img.ndim > 2):
            return np.round(img.mean(-1)).astype(np.uint8)
        elif not self.grayscale and (img.ndim < 3):
            return img[..., None].repeat(3, -1)
        return img

    def update(self, frames):
        """Update plot.

        Args:
            frames (Tuple[np.ndarray]): Frames from video source
        """
        x, hists = compute_histogram(frames[self.frame_index], self.grayscale)
        for line, hist in zip(self.lines, hists):
            line.setData(x, hist)
