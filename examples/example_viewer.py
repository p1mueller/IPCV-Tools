#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test case to show case the ImageViewer class."""

from typing import Sequence

import cv2
import numpy as np


class ContourResampler:
    """Resamples contour using Fourier coefficients."""

    def __init__(self, keep_coeffs: float = 1.0, sigma: float = 3.0) -> None:
        """Initialize.

        Args:
            keep_coeffs: Percentage of Fourier coefficients kept. Between 0 to 1.
                Defaults to 1.0.
            sigma: Sigma of Gaussian blur. Defaults to 3.0.
        """
        self.sigma = sigma
        self.keep_coeffs = keep_coeffs
        self.output_names = ["Original", "Blurred", "Thresholded", "Result"]

    def set_keep_coeffs(self, value: float) -> None:
        """Set the percentage of Fourier coefficients kept.

        Args:
            value: New value
        """
        self.keep_coeffs = value

    def set_sigma(self, value: float) -> None:
        """Set the sigma of the Gaussian blur.

        Args:
            value: New value
        """
        self.sigma = value

    def __call__(self, frame: np.ndarray) -> Sequence[np.ndarray]:
        """Processing function.

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
        f = np.mean(frame, -1)
        f = f.astype("uint8")

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
            nr_coeffs = int(0.5 + self.keep_coeffs * s.shape[0])
            v = nr_coeffs // 2
            x = np.arange(v + 1)
            x = np.concatenate([x, -np.flipud(x[1:])])
            coeffs = np.fft.fft(s)[x] * (2 * v + 1) / s.shape[0]
            res = np.fft.ifft(coeffs)
            opt_contours.append(
                np.column_stack([np.real(res), np.imag(res)]).astype(int)
            )

        # Overlay image with boundaries
        res_img = frame.copy()
        cv2.polylines(res_img, opt_contours, True, (255, 0, 0), 3)
        return frame, g, g_thresh, res_img


if __name__ == "__main__":
    from ipcv_tools.pipeline import Pipeline

    port = None
    factor = 0.4
    decimation = 2
    original_width = 1920
    original_height = 1080
    height = original_height // decimation
    width = original_width // decimation

    contour_resampler = ContourResampler()
    pipeline = Pipeline(
        contour_resampler.__call__, port=port, width=width, height=height, use_ui=False
    )
    pipeline.run({"decimation": decimation, "width": width, "height": height})
