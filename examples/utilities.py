#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities of examples."""

from typing import Sequence

import cv2
import numpy as np

from ipcv_tools.plotting import HistogramPlotter
from ipcv_tools.ui import CameraUI


def initialize_controls(viewer: CameraUI, contour_resampler: "ContourResampler") -> None:
    """Initialize UI controls.

    Makes sure different UI examples have the same controls.

    Args:
        viewer: UI
        contour_resampler: Contour resampler
    """
    assert isinstance(viewer, CameraUI)
    assert isinstance(contour_resampler, ContourResampler)
    viewer.add_checkbox(
        "Use %",
        func=contour_resampler.set_use_percent,
        value=contour_resampler.use_percent,
    )
    viewer.add_checkbox(
        "Reduce Boundary Points",
        func=contour_resampler.set_reduce_boundary_points,
        value=contour_resampler.reduce_boundary_points,
    )
    viewer.add_slider(
        "Sigma",
        1.0,
        10.0,
        101,
        300,
        10,
        contour_resampler.set_sigma,
        value=contour_resampler.sigma,
    )
    viewer.add_slider(
        "Absolute Coeffs",
        1,
        101,
        51,
        100,
        10,
        contour_resampler.set_abs_coeffs,
        value=contour_resampler.abs_coeffs,
    )
    viewer.add_slider(
        "Coeffs %",
        0.0,
        100.0,
        501,
        300,
        10,
        contour_resampler.set_rel_coeffs,
        value=100.0 * contour_resampler.rel_coeffs,
    )


def initialize_plots(
    viewer: CameraUI, frame_index: int = 0, monochrome: bool = False
) -> None:
    """Initialize UI plots.

    Args:
        viewer: UI
        frame_index: Index of frame of which the histogram is calculated. Defaults to 0.
        monochrome: Flag if histogram is calculated monochrome. Defaults to False.
    """
    hist_plotter = HistogramPlotter(frame_index, monochrome)
    viewer.add_plot("Histogram", hist_plotter)


def create_contour_resampler() -> "ContourResampler":
    """Create contour resampler object.

    Makes sure all examples use the same parameters.

    Returns:
        Contour resampler
    """
    factor = 0.4
    abs_val = 11
    sigma = 3.0
    use_percent = True
    reduce_bp = False
    return ContourResampler(
        rel_coeffs=factor,
        abs_coeffs=abs_val,
        sigma=sigma,
        use_percent=use_percent,
        reduce_boundary_points=reduce_bp,
    )


class ContourResampler:
    """Resamples contour using Fourier coefficients."""

    def __init__(
        self,
        rel_coeffs: float = 1.0,
        abs_coeffs: int = 11,
        sigma: float = 3.0,
        use_percent: bool = True,
        reduce_boundary_points: bool = True,
    ) -> None:
        """Initialize.

        Args:
            rel_coeffs: Percentage of Fourier coefficients kept. Between 0 to 1.
                Defaults to 1.0.
            abs_coeffs: Absolute number of Fourier coefficients kept. Defaults to 11.
            sigma: Sigma of Gaussian blur. Defaults to 3.0.
            use_percent: Use relative number of Fourier coefficients. Defaults to True.
            reduce_boundary_points: Flag if number of boundary points will be reduced
                according to the number of coefficients. Defaults to True.
        """
        self.sigma = sigma
        self.use_percent = use_percent
        self.reduce_boundary_points = reduce_boundary_points
        self.rel_coeffs = rel_coeffs
        self.abs_coeffs = int(abs_coeffs)
        self.output_names = ["Original", "Blurred", "Thresholded", "Result"]

    def set_use_percent(self, value: int) -> None:
        """Set if absolute or relative number of Fourier coefficients are kept.

        Args:
            value: New value
        """
        self.use_percent = value > 0

    def set_reduce_boundary_points(self, value: int) -> None:
        """Set if the number of boundary points is reduced.

        The number of boundary points will be reduced
        according to the number of coefficients.

        Args:
            value: New value
        """
        self.reduce_boundary_points = value > 0

    def set_rel_coeffs(self, value: float) -> None:
        """Set the percentage of Fourier coefficients kept.

        Args:
            value: New value
        """
        self.rel_coeffs = value / 100.0

    def set_abs_coeffs(self, value: float) -> None:
        """Set the absolute number of Fourier coefficients kept.

        Args:
            value: New value
        """
        self.abs_coeffs = int(np.round(value))

    def set_sigma(self, value: float) -> None:
        """Set the sigma of the Gaussian blur.

        Args:
            value: New value
        """
        self.sigma = value

    def __call__(self, frame: np.ndarray) -> Sequence[np.ndarray]:
        """Process frame.

        1. Blur image (Gaussian blur)
        2. Threshold image (Otsu)
        3. Find contours
        4. Describe contours with Fourier coefficients.
        5. Remove percentage of coefficients
        6. Recalculate contours with missing coefficients set to zero.
        7. Overlay the 'optimized' contours over the original image

        Args:
            frame: Frame from a video source

        Returns:
            Original frame
            Blurred frame
            Frame after thresholding
            Frame with optimized contour overlay
        """
        # # Pre-processing
        if frame.ndim > 2:
            res_img = frame.copy()
            f = np.mean(frame, -1)
        else:
            f = frame
            res_img = np.stack(3 * [frame], -1)
        f = np.clip(np.round(f), 0, 255).astype("uint8")

        # # Image processing
        # Segmentation
        g = np.round(cv2.GaussianBlur(f, (0, 0), self.sigma))
        _, g_thresh = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU)

        # Boundary Extraction
        contours, _ = cv2.findContours(g_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Fourier Descriptors
        opt_contours = []
        for contour in contours:
            s = contour[:, 0, 0] + 1j * contour[:, 0, 1]
            if self.use_percent:
                nr_coeffs = int(0.5 + self.rel_coeffs * s.shape[0])
            else:
                nr_coeffs = min(self.abs_coeffs, s.shape[0])
            v = nr_coeffs // 2
            coeffs = np.fft.fft(s)  # * (2 * v + 1) / s.shape[0]
            if self.reduce_boundary_points:
                x = np.arange(v + 1)
                x = np.concatenate([x, -np.flipud(x[1:])])
                coeffs = coeffs[x] * (2 * v + 1) / s.shape[0]
            else:
                top = -v if v > 0 else None
                coeffs[v + 1 : top] = 0.0
            res = np.fft.ifft(coeffs)
            opt_contours.append(np.column_stack([np.real(res), np.imag(res)]).astype(int))

        # Overlay image with boundaries
        cv2.polylines(res_img, opt_contours, True, (255, 0, 0), 3)
        return frame, g, g_thresh, res_img
