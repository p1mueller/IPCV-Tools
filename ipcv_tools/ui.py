#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UI to display frames from a video source and control processing parameters."""

import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

from ipcv_tools.controls import Slider
from ipcv_tools.plotting import Plotter
from ipcv_tools.utilities import FPS, fix_cv2_issue


def array_to_qimage(array: np.ndarray) -> QImage:
    """Convert an numpy array to QImage.

    Args:
        array: Array to convert. 2D or 3D arrays allowed.

    Returns:
         Converted image
    """
    if array.ndim == 2:
        array = array[..., None]
    h, w, ch = array.shape
    bytes_per_line = ch * w
    if ch == 3:
        qimg = QImage(array.data, w, h, bytes_per_line, QImage.Format_RGB888)
    else:
        qimg = QImage(array.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
    return qimg.scaled(w, h, Qt.KeepAspectRatio)


class VideoThread(QThread):
    """Thread which polls from a video source."""

    changed_image = pyqtSignal(tuple)

    def __init__(self, source: Callable, parent: QObject) -> None:
        """Initialize.

        Args:
            source: Function to get new frame(s)
            parent: Parent
        """
        super().__init__(parent)
        self.fps = FPS()
        self.running = False
        self.source = source

    def run(self) -> None:
        """Task of thread.

        Polls frame(s) from video source and emits signal, when new elements present.
        """
        self.running = True
        while self.running:
            frames = self.source()
            self.fps.update()
            self.changed_image.emit(frames)


class CameraUI(QtWidgets.QWidget):
    """UI to display camera images and results of image pipeline."""

    def __init__(
        self,
        width: int,
        height: int,
        img_names: Sequence[str],
        source: Callable,
        title: str = "IPCV Cam",
    ):
        """Initialize.

        Args:
            width: Width of images.
            height: Height of images.
            img_names: Name of images (used in tabs)
            source: Video source pipeline.
            title: Window title. Defaults to "IPCV Cam".
        """
        fix_cv2_issue()
        self.app = QtWidgets.QApplication(sys.argv)

        super().__init__()
        self.save_request = False
        self._width = width
        self._height = height
        self.img_names = img_names

        self.setWindowTitle(title)

        # Image tab widgets
        self.img_widgets = []
        tab = QtWidgets.QTabWidget()
        tab.setTabPosition(QtWidgets.QTabWidget.West)
        for img_name in self.img_names:
            img_widget = QtWidgets.QLabel()
            img_widget.setBaseSize(self._width, self._height)
            tab.addTab(img_widget, img_name)
            self.img_widgets.append(img_widget)
        tab.setCurrentIndex(len(img_names) - 1)

        # Label to display image shape and FPS
        self.info_label = QtWidgets.QLabel()

        # Button to save the current images
        save_button = QtWidgets.QPushButton()
        save_button.setText("Save")
        save_button.clicked.connect(self.save_image_request)

        # Button to quit the UI
        quit_button = QtWidgets.QPushButton()
        quit_button.setText("Quit")
        quit_button.clicked.connect(self.quit)

        # Layout
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(save_button)
        button_layout.addWidget(quit_button)

        self.tools_layout = QtWidgets.QVBoxLayout()
        self.tools_layout.setAlignment(Qt.AlignTop)
        self.tools_layout.addWidget(self.info_label)
        self.tools_layout.addLayout(button_layout)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setAlignment(Qt.AlignTop)
        main_layout.addWidget(tab)
        main_layout.addLayout(self.tools_layout)
        self.setLayout(main_layout)

        # Create thread which reads video data
        # Sends signal when new data is present
        self.video_thread = VideoThread(source, self)
        self.video_thread.changed_image.connect(self.update_images)

        self.app.aboutToQuit.connect(self.quit)

    @pyqtSlot(tuple)
    def update_images(self, frames: Sequence[np.ndarray]) -> None:
        """Update images.

        Args:
            frames: New frames
        """
        for img_widget, frame in zip(self.img_widgets, frames):
            qimg = array_to_qimage(frame)
            img_widget.setPixmap(QPixmap.fromImage(qimg))
        self.set_info_label()

        if self.save_request:
            self.save_request = False
            file, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save", filter="Image files"
            )

            path = Path(file)
            folder = path.parent
            base = path.stem
            suffix = path.suffix

            if len(self.img_names) > 1:
                for name, frame in zip(self.img_names, frames):
                    file_name = f"{base}_{name}{suffix}"
                    target = folder / file_name
                    plt.imsave(target, frame, cmap="gray")
            else:
                plt.imsave(file, frame, cmap="gray")

    def set_info_label(self) -> None:
        """Set text of info label."""
        infos = [
            f"Shape: ({self._height}, {self._width})",
            f"FPS: {self.video_thread.fps.value:.1f}",
        ]
        self.info_label.setText(", ".join(infos))

    def quit(self) -> None:
        """Quits UI and all underlying processes."""
        self.video_thread.running = False
        while self.video_thread.isRunning():
            pass
        self.app.quit()

    def save_image_request(self) -> None:
        """Set request for image saving."""
        self.save_request = True

    def run(self) -> int:
        """Start the UI."""
        null_img = np.zeros((self._height, self._width, 3), np.uint8)
        self.update_images(len(self.img_widgets) * [null_img])
        self.video_thread.start()
        self.show()
        return self.app.exec_()

    def add_slider(
        self,
        name: str,
        start: float,
        end: float,
        steps: int = 100,
        width: int = 300,
        height: int = 10,
        func: Optional[Callable] = None,
        value: Optional[float] = None,
    ) -> Slider:
        """Add a slider to the right side of the UI.

        Args:
            name: Name of slider
            start: Minimum value of slider.
            end: Maximum value of slider.
            steps: Number of increments the slider has. Defaults to 100.
            width: Minimum width of the slider. Defaults to 300.
            height: Minimal height of the slider. Defaults to 10.
            func: Callback function when slider value changes.
                Defaults to None.
            value: Initial value. If None the value is set to the middle
                of the range. Defaults to None.
        """
        if value is None:
            value = (start + end) // 2
        slider = Slider(name, start, end, steps, value, width, height)
        self.tools_layout.addWidget(slider)
        if func is not None:
            slider.value_changed.connect(func)
        return slider

    def add_checkbox(
        self,
        name: str,
        width: int = 100,
        height: int = 10,
        func: Optional[Callable] = None,
        value: bool = False,
    ) -> QtWidgets.QCheckBox:
        """Add a checkbox to the right side of the UI.

        Args:
            name: Name of checkbox. Also acts as label
            width: Minimum width of the slider. Defaults to 300.
            height: Minimal height of the slider. Defaults to 10.
            func: Callback function when checkbox value changes.
                Defaults to None.
            value: Initial state of checkbox. Defaults to False.
        """
        checkbox = QtWidgets.QCheckBox()
        checkbox.setText(name)
        checkbox.setMinimumSize(width, height)
        checkbox.setChecked(value)
        if func is not None:
            checkbox.stateChanged.connect(func)
        self.tools_layout.addWidget(checkbox)
        return checkbox

    def add_plot(
        self,
        name: str,
        plotter: Plotter,
        width: int = 400,
        height: int = 150,
        signal: Optional[Any] = None,
        func: Optional[Callable] = None,
    ) -> pg.PlotWidget:
        """Add plot widget to the right side ot the UI.

        Args:
            name: Name of plot
            plotter: Handles plotting from new frames.
            width: Minimum width of the plot. Defaults to 400.
            height: Minimum height of the plot. Defaults to 150.
            signal: Update signal used to call callback function (func).
                Defaults to None.
            func: Callback function. Defaults to None.

        Returns:
            Plot
        """
        assert isinstance(plotter, Plotter)
        graph = pg.PlotWidget()
        graph.setMinimumSize(width, height)

        graph.showGrid(x=True, y=True)
        graph.setTitle(name)

        self.tools_layout.addWidget(graph)

        null_img = np.ones((self._height, self._width, 3), np.uint8)
        plotter.set_graph(graph, null_img)
        self.video_thread.changed_image.connect(plotter.update)
        if (signal is not None) and (func is not None):
            signal.connect(func)
        return graph


if __name__ == "__main__":
    from ipcv_tools.plotting import HistogramPlotter

    size = (600, 600)
    mean = 128
    min_sigma = sigma = 1
    max_sigma = 30

    def _random_noise_img() -> Tuple[np.ndarray, np.ndarray]:
        uniform_noise = np.random.randint(0, 256, size + (3,), np.uint8)
        imgs = [np.random.normal(noise.mean, noise.std, size) for noise in noises]
        normal_noise = np.clip(np.stack(imgs, -1), 0, 255).astype(np.uint8)
        return uniform_noise, normal_noise

    class _Noise:
        def __init__(self, mean: float, std: float) -> None:
            self.mean = mean
            self.std = std

        def set_mean(self, value: float) -> None:
            self.mean = value

        def set_std(self, value: float) -> None:
            self.std = value

    viewer = CameraUI(*size[::-1], ["Uniform Noise", "Gaussian Noise"], _random_noise_img)

    noises = []
    for c in ["R", "G", "B"]:
        noise = _Noise(
            np.random.randint(50, 205), np.random.uniform(min_sigma, max_sigma)
        )
        viewer.add_slider(f"{c} Mean", 0, 255, 256, func=noise.set_mean, value=noise.mean)
        viewer.add_slider(
            f"{c} STD", min_sigma, max_sigma, 100, func=noise.set_std, value=noise.std
        )
        noises.append(noise)

    hist_plotter = HistogramPlotter(1)
    graph = viewer.add_plot("Histogram", hist_plotter)

    viewer.run()
