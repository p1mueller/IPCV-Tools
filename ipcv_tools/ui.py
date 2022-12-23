#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UI to display frames from a video source and control processing parameters."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from ipcv_tools.plotting import Plotter

from ipcv_tools.utilities import FPS


def array_to_qimage(array):
    """Convert an numpy array to QImage.

    Args:
        array (np.ndarray): Array to convert. 2D or 3D arrays allowed.

    Returns:
         QImage: Converted image
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


class Slider(QtWidgets.QWidget):
    """Composed slider widget.

    Extension to allow floating values in slider. (QT slider only allows for int)
    """

    value_changed = pyqtSignal(float)

    def __init__(
        self, name="", min_val=0, max_val=1, steps=100, value=None, width=300, height=10
    ):
        """Initialize.

        Args:
            name: Name of slider. Defaults to "".
            min_val: Minimum value of slider. Defaults to 0.
            max_val: Maximum value of slider. Defaults to 1.
            steps: Number of steps of slider. Defaults to 100.
            value: Initial value of slider. If None the value in the middle is taken.
                Defaults to None.
            width: Minimal width of the slider. Defaults to 300.
            height: Minimal height of the slider. Defaults to 10.
        """
        super().__init__()
        self.step = (max_val - min_val) / (steps - 1)
        self.min_val = min_val

        if value is None:
            value = steps // 2
        else:
            value = np.clip(int(0.5 + (value - min_val) / self.step), 0, steps - 1)

        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setMinimumSize(width, height)
        self.slider.setMinimum(0)
        self.slider.setMaximum(steps - 1)
        self.slider.setValue(value)
        self.slider.valueChanged.connect(self.value_change_slider)

        slider_label = QtWidgets.QLabel()
        slider_label.setText(name)
        slider_label.setFixedWidth(90)
        self.slider_text = QtWidgets.QLabel()
        self.slider_text.setFixedWidth(50)
        self.change_text(self.value())

        slider_layout = QtWidgets.QHBoxLayout()
        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.slider_text)
        self.setLayout(slider_layout)

    def value(self):
        """Get current value of slider."""
        return self.min_val + self.step * self.slider.value()

    def change_text(self, value):
        """Change text of slider value label.

        Args:
            value: New value to display
        """
        self.slider_text.setText(f"{value:-.3g}")

    def value_change_slider(self, value):
        """Slot for internal slider value change.

        Args:
            value: Internal slider value
        """
        val = self.value()
        self.change_text(val)
        self.value_changed.emit(val)


class VideoThread(QThread):
    """Thread which polls from a video source."""

    changed_image = pyqtSignal(tuple)

    def __init__(self, source, parent) -> None:
        """Initialize.

        Args:
            source (function): Function to get new frame(s)
            parent (QObject): Parent
        """
        super().__init__(parent)
        self.fps = FPS()
        self.running = False
        self.source = source

    def run(self):
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

    def __init__(self, width, height, img_names, source, title="IPCV Cam"):
        """Initialize.

        Args:
            width (int): Width of images.
            height (int): Height of images.
            img_names (List[str]): Name of images (used in tabs)
            source (Worker): Video source pipeline.
            title (str): Window title. Defaults to "IPCV Cam".
        """
        self.app = QtWidgets.QApplication(sys.argv)

        super().__init__()
        self.save_request = False
        self.width = width
        self.height = height
        self.img_names = img_names

        self.setWindowTitle(title)

        # Image tab widgets
        self.img_widgets = []
        tab = QtWidgets.QTabWidget()
        tab.setTabPosition(QtWidgets.QTabWidget.West)
        for img_name in self.img_names:
            img_widget = QtWidgets.QLabel()
            img_widget.setBaseSize(self.width, self.height)
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
    def update_images(self, frames):
        """Update images.

        Args:
            frames (Tuple(np.ndarray)): New frames
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

    def set_info_label(self):
        """Set text of info label."""
        self.info_label.setText(
            f"Shape: ({self.height}, {self.width}), FPS: {self.video_thread.fps.value:.1f}"
        )

    def quit(self):
        """Quits UI and all underlying processes."""
        self.video_thread.running = False
        while self.video_thread.isRunning():
            pass
        self.app.quit()

    def save_image_request(self):
        """Set request for image saving."""
        self.save_request = True

    def run(self):
        """Start the UI."""
        null_img = np.zeros((self.height, self.width, 3), np.uint8)
        self.update_images(len(self.img_widgets) * [null_img])
        self.video_thread.start()
        self.show()
        self.app.exec_()

    def add_slider(
        self, name, start, end, steps=100, width=300, height=10, func=None, value=None
    ):
        """Add a slider to the right side of the UI.

        Args:
            name (str): Name of slider
            start (float): Minimum value of slider.
            end (float): Maximum value of slider.
            steps (int): Number of increments the slider has. Defaults to 100.
            width (int): Minimum width of the slider. Defaults to 300.
            height (int): Minimal height of the slider. Defaults to 10.
            func (function): Callback function when slider value changes.
                Defaults to None.
            value (float): Initial value. If None the value is set to the middle
                of the range. Defaults to None.
        """
        if value is None:
            value = (start + end) // 2
        slider = Slider(name, start, end, steps, value, width, height)
        self.tools_layout.addWidget(slider)
        if func is not None:
            slider.value_changed.connect(func)

    def add_plot(self, name, plotter, width=400, height=150, signal=None, func=None):
        """Add plot widget to the right side ot the UI.

        Args:
            name (str): Name of plot
            plotter (Plotter): Handles plotting from new frames.
            width (int): Minimum width of the plot. Defaults to 400.
            height (int): Minimum height of the plot. Defaults to 150.
            signal (pyqtSignal): Update signal used to call callback function (func).
                Defaults to None.
            func (function): Callback function. Defaults to None.

        Returns:
            PlotWidget: Plot
        """
        assert isinstance(plotter, Plotter)
        graph = pg.PlotWidget()
        graph.setMinimumSize(width, height)

        graph.showGrid(x=True, y=True)
        graph.setTitle(name)

        self.tools_layout.addWidget(graph)

        null_img = np.ones((self.height, self.width, 3), np.uint8)
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

    def _random_noise_img():
        noisy_img = np.random.randint(0, 256, size + (3,), np.uint8)
        imgs = [np.random.normal(noise.mean, noise.std, size) for noise in noises]
        img = np.clip(np.stack(imgs, -1), 0, 255).astype(np.uint8)
        return noisy_img, img

    class _Noise:
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def set_mean(self, value):
            self.mean = value

        def set_std(self, value):
            self.std = value

    viewer = CameraUI(
        *size[::-1], ["Uniform Noise", "Gaussian Noise"], _random_noise_img
    )

    noises = []
    for c in ["R", "G", "B"]:
        noise = _Noise(
            np.random.randint(50, 205), np.random.uniform(min_sigma, max_sigma)
        )
        viewer.add_slider(
            f"{c} Mean", 0, 255, 256, func=noise.set_mean, value=noise.mean
        )
        viewer.add_slider(
            f"{c} STD", min_sigma, max_sigma, 100, func=noise.set_std, value=noise.std
        )
        noises.append(noise)

    hist_plotter = HistogramPlotter(1)
    graph = viewer.add_plot("Histogram", hist_plotter)

    viewer.run()
