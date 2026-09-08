"""Tests for the CameraUI widget in ipcv_tools.ui.

CameraUI reuses the shared QApplication (see conftest.py), so tests must not
call ``app.quit()`` themselves — Qt objects are released at process exit.
"""

import numpy as np
import pyqtgraph as pg
import pytest

from ipcv_tools.plotting import HistogramPlotter
from ipcv_tools.ui import CameraUI, array_to_qimage


def _rgb(shape=(32, 32)):
    """Create a random 3-channel uint8 image."""
    return np.random.randint(0, 255, shape + (3,), dtype=np.uint8)


def _make_ui(img_names=("frame",), title="UI"):
    """Helper to construct a CameraUI for a single test."""
    return CameraUI(32, 32, list(img_names), lambda: _rgb(), title=title)


def test_array_to_qimage_rgb(qapp):
    """A 3D array should convert to an RGB QImage of matching size."""
    img = _rgb((24, 18))
    qimg = array_to_qimage(img)
    assert qimg.width() == 18
    assert qimg.height() == 24


def test_cameraui_construction(qapp):
    """CameraUI should build one QLabel tab per image name."""
    ui = _make_ui(img_names=("frame", "result"), title="My UI")
    assert ui.windowTitle() == "My UI"
    assert len(ui.img_widgets) == 2
    assert ui.video_thread.source is not None
    assert ui.plotters == []


def test_cameraui_update_images_sets_pixmap_and_label(qapp):
    """update_images should populate the info label with shape and FPS."""
    ui = _make_ui()
    ui.update_images([_rgb()])
    text = ui.info_label.text()
    assert "Shape: (32, 32)" in text
    assert "FPS:" in text


def test_cameraui_set_info_label(qapp):
    """set_info_label should produce a comma-separated info string."""
    ui = _make_ui()
    ui.set_info_label()
    text = ui.info_label.text()
    assert "," in text
    assert "Shape" in text


def test_cameraui_add_slider_default_midpoint(qapp):
    """add_slider without an explicit value must use midpoint."""
    ui = _make_ui()
    s = ui.add_slider("gain", 0, 100, steps=101)
    assert s.value() == 50.0


def test_cameraui_add_slider_with_callback(qapp):
    """add_slider should connect the callback to value_changed."""
    ui = _make_ui()
    box = []
    s = ui.add_slider("gain", 0, 100, steps=101, func=box.append)
    s.slider.setValue(42)
    assert box and box[-1] == s.value()


def test_cameraui_add_checkbox_state(qapp):
    """add_checkbox should set initial state and text."""
    ui = _make_ui()
    cb = ui.add_checkbox("invert", value=True)
    assert cb.isChecked() is True
    assert cb.text() == "invert"
    cb2 = ui.add_checkbox("auto", value=False)
    assert cb2.isChecked() is False


def test_cameraui_add_plot_with_histogram(qapp):
    """add_plot should register a HistogramPlotter and a PlotWidget."""
    ui = _make_ui()
    plotter = HistogramPlotter()
    graph = ui.add_plot("Histogram", plotter)
    assert isinstance(graph, pg.PlotWidget)
    assert plotter in ui.plotters
    # Emitting a frame through the slot should update the plot without error.
    ui.update_images([_rgb()])
    assert len(plotter.lines) == 3


def test_cameraui_save_single_image(qapp, tmp_path, monkeypatch):
    """A single-image save should call the dialog once and write a file."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from PyQt5 import QtWidgets

    target = tmp_path / "out.png"
    calls = {"n": 0}

    def fake_dialog(*a, **k):
        calls["n"] += 1
        return (str(target), "Image files")

    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", staticmethod(fake_dialog)
    )

    ui = _make_ui()
    ui.save_request = True
    ui.update_images([_rgb()])
    assert calls["n"] == 1
    assert ui.save_request is False
    assert target.exists()


def test_cameraui_save_multiple_images(qapp, tmp_path, monkeypatch):
    """A multi-image save should append each image name to the base stem."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from PyQt5 import QtWidgets

    def fake_dialog(*a, **k):
        return (str(tmp_path / "base.png"), "Image files")

    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", staticmethod(fake_dialog)
    )

    ui = _make_ui(img_names=("raw", "proc"))
    ui.save_request = True
    ui.update_images([_rgb(), _rgb()])
    files = sorted(p.name for p in tmp_path.glob("base_*.png"))
    assert files == ["base_proc.png", "base_raw.png"]
