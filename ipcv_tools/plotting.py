#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plotting utilities for the camera UI."""

import warnings
from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from skimage.exposure import histogram


def compute_histogram(
    img: np.ndarray, grayscale: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the histogram for an RGB image.

    Args:
        img: RGB image
        grayscale: Image is grayscale. Defaults to False.

    Returns:
        Center points of bins
        Histogram values
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
        """Initialize."""
        self.graph = None

    def set_graph(self, graph: pg.PlotWidget, example: np.ndarray) -> None:
        """Set graph object.

        Args:
            graph: Graph
            example: Example image to initialize graph.
        """
        self.graph = graph
        self._init_graph(example)

    @abstractmethod
    def _init_graph(self, example: np.ndarray) -> None:
        """Initialize graph with line dummies.

        Args:
            graph: Graph
            example: Example image to initialize graph.
        """

    @abstractmethod
    def update(self, frames: Sequence[np.ndarray]) -> None:
        """Update plot."""


class HistogramPlotter(Plotter):
    """Plot histogram of a frame."""

    def __init__(self, frame_index: int = 0, grayscale: bool = False):
        """Initialize.

        Args:
            frame_index: Index of frame to use. Defaults to 0
            grayscale: Make histogram for grayscale images. Defaults to False.
        """
        super().__init__()
        self.frame_index = frame_index
        self.grayscale = grayscale
        self.lines: List[pg.PlotCurveItem] = []

    def _init_graph(self, example: np.ndarray) -> None:
        """Initialize graph with line dummies.

        Args:
            graph: Graph
            example: Example image to initialize graph.
        """
        assert self.graph is not None
        self.lines.clear()
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

    def _ensure_dim(self, img: np.ndarray) -> np.ndarray:
        if self.grayscale and (img.ndim > 2):
            return np.round(img.mean(-1)).astype(np.uint8)
        if not self.grayscale and (img.ndim < 3):
            return img[..., None].repeat(3, -1)
        return img

    def update(self, frames: Sequence[np.ndarray]) -> None:
        """Update plot.

        Args:
            frames: Frames from video source
        """
        x, hists = compute_histogram(frames[self.frame_index], self.grayscale)
        for line, hist in zip(self.lines, hists):
            line.setData(x, hist)
