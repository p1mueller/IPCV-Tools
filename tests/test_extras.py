"""Extra tests to push coverage on small branches of ipcv_tools modules."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from ipcv_tools.camera import DataCam
from ipcv_tools.plotting import HistogramPlotter
from ipcv_tools.processing import Processor
from ipcv_tools.viewer import ImageViewer


def _rgb(shape=(16, 16)):
    return np.random.randint(0, 255, shape + (3,), dtype=np.uint8)


def _gray(shape=(16, 16)):
    return np.random.randint(0, 255, shape, dtype=np.uint8)


def test_datacam_loads_by_bundled_name():
    """Passing the stem of a bundled image should resolve to the file."""
    cam = DataCam("lena", colored=True)
    assert cam._image.ndim == 3


def test_datacam_file_path_grayscale():
    """colored=False on a file path should collapse to a 2D grayscale image."""
    cam = DataCam("lena", colored=False)
    assert cam._image.ndim == 2


def test_ensure_dim_passthrough_2d_grayscale():
    """2D input + grayscale=True is already correct, so _ensure_dim returns it unchanged."""
    plotter = HistogramPlotter(grayscale=True)
    img = _gray()
    assert plotter._ensure_dim(img) is img


def test_ensure_dim_passthrough_3d_not_grayscale():
    """3D input + grayscale=False is already correct, so _ensure_dim returns it unchanged."""
    plotter = HistogramPlotter(grayscale=False)
    img = _rgb()
    assert plotter._ensure_dim(img) is img


def test_plotter_set_graph_stores_reference(qapp):
    """set_graph should store the graph and call _init_graph on the concrete subclass."""
    plotter = HistogramPlotter()
    import pyqtgraph as pg

    graph = pg.PlotWidget()
    plotter.set_graph(graph, _rgb())
    assert plotter.graph is graph


def test_processor_is_ready_reflects_buffer(qapp):
    """Processor._is_ready should reflect whether its buffer has data."""
    got = {"n": 0}

    def src():
        got["n"] += 1
        if got["n"] == 1:
            return np.zeros((8, 8, 3), np.uint8)
        # Idle after first element so the worker thread does not spin.
        return None

    p = Processor(src_func=src, proc_func=lambda x: x)
    p.start()
    import time

    time.sleep(0.05)
    assert p._is_ready() is True
    p.stop()


def test_imageviewer_run_until_quit(qapp):
    """ImageViewer.run should loop while is_running, then return 0."""
    import pygame

    img = _rgb((8, 8))
    state = {"n": 0}

    def source():
        state["n"] += 1
        if state["n"] >= 2:
            viewer.stop()
        return img

    viewer = ImageViewer(8, 8, source, title="T")
    try:
        assert viewer.is_running() is True
        ret = viewer.run()
        assert ret == 0
        assert viewer.is_running() is False
        assert state["n"] >= 2
    finally:
        pygame.quit()


def test_imageviewer_set_title(qapp):
    """set_title should call pygame.display.set_caption with the new title."""
    pygame = pytest.importorskip("pygame")
    img = np.zeros((4, 4, 3), np.uint8)
    viewer = ImageViewer(4, 4, lambda: img, title="Start")
    try:
        assert pygame.display.get_caption()[0] == "Start"
        viewer.set_title("Changed")
        assert pygame.display.get_caption()[0] == "Changed"
    finally:
        pygame.quit()


def test_array_to_qimage_grayscale(qapp):
    """A 2D array should convert to a grayscale QImage of matching size."""
    from ipcv_tools.ui import array_to_qimage

    img = np.random.randint(0, 255, (10, 12), dtype=np.uint8)
    qimg = array_to_qimage(img)
    assert qimg.width() == 12
    assert qimg.height() == 10


def test_videothread_run_emits_frames(qapp):
    """VideoThread.run should poll the source repeatedly and emit frames."""
    from PyQt5.QtCore import QEventLoop, QTimer

    from ipcv_tools.ui import VideoThread

    img = np.zeros((4, 4, 3), np.uint8)
    thread = VideoThread(lambda: (img,), parent=None)
    seen = []
    thread.changed_image.connect(lambda f: seen.append(f))

    loop = QEventLoop()
    timer = QTimer()
    timer.timeout.connect(loop.quit)
    timer.start(150)
    thread.start()
    loop.exec_()
    thread.wait(1000)
    thread.running = False
    # The worker busy-loops, so at least one (usually many) frames emit.
    assert len(seen) >= 1


def test_cameraui_save_image_request(qapp):
    """save_image_request should toggle the save_request flag on."""
    from ipcv_tools.ui import CameraUI

    img = _rgb()
    ui = CameraUI(32, 32, ["frame"], lambda: (img,), title="T")
    assert ui.save_request is False
    ui.save_image_request()
    assert ui.save_request is True


def test_cameraui_quit_stops_thread(qapp):
    """quit() should set the video thread's running flag to False."""
    from ipcv_tools.ui import CameraUI

    img = _rgb()
    ui = CameraUI(32, 32, ["frame"], lambda: (img,), title="T")
    ui.video_thread.running = True
    ui.quit()
    assert ui.video_thread.running is False


def test_cameraui_add_checkbox_with_callback(qapp):
    """add_checkbox should wire the stateChanged slot when a func is given."""
    from PyQt5.QtCore import Qt

    from ipcv_tools.ui import CameraUI

    img = _rgb()
    ui = CameraUI(32, 32, ["frame"], lambda: (img,), title="T")
    hits = []
    cb = ui.add_checkbox("toggle", value=False, func=lambda state: hits.append(state))
    cb.setChecked(True)
    assert hits and hits[-1] == int(Qt.Checked)


def test_cameraui_add_plot_with_signal_and_func(qapp):
    """add_plot should connect a signal to a func when both are provided."""
    from PyQt5.QtCore import QObject, pyqtSignal

    from ipcv_tools.plotting import HistogramPlotter
    from ipcv_tools.ui import CameraUI

    class _Src(QObject):
        sig = pyqtSignal(float)

    img = _rgb()
    ui = CameraUI(32, 32, ["frame"], lambda: (img,), title="T")
    src = _Src()
    hits = []
    plotter = HistogramPlotter()
    ui.add_plot("h", plotter, signal=src.sig, func=lambda v: hits.append(v))
    src.sig.emit(3.0)
    assert hits == [3.0]
