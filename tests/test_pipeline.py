"""Tests for Pipeline (construction only — run() blocks) and mocked cameras."""

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")

from ipcv_tools.camera import DataCam, Webcam, find_camera_handler
from ipcv_tools.pipeline import Pipeline
from ipcv_tools.ui import CameraUI


def _rgb(shape=(32, 32)):
    """Create a random 3-channel uint8 image."""
    return np.random.randint(0, 255, shape + (3,), dtype=np.uint8)


def _proc(frame):
    """Small processing function: return the frame as a uint8 ndarray."""
    return np.asarray(frame, np.uint8)


def test_pipeline_with_camera_use_ui(qapp):
    """Pipeline with use_ui=True should build a CameraUI and wire the processor."""
    cam = DataCam(_rgb(), colored=True, buffer_size=4)
    pipe = Pipeline(_proc, camera=cam, width=32, height=32, img_names=["raw", "proc"])
    assert isinstance(pipe.viewer, CameraUI)
    assert pipe.processor is not None
    assert pipe.camera is cam


def test_pipeline_with_camera_use_image_viewer(qapp):
    """Pipeline with use_ui=False should build an ImageViewer (pygame)."""
    import pygame

    from ipcv_tools.viewer import ImageViewer

    cam = DataCam(_rgb(), colored=True, buffer_size=4)
    pipe = Pipeline(_proc, camera=cam, width=16, height=16, use_ui=False)
    try:
        assert isinstance(pipe.viewer, ImageViewer)
    finally:
        pygame.quit()


def test_find_camera_handler_falls_back_to_webcam(monkeypatch):
    """Without GenICam devices the fallback must be Webcam."""
    from ipcv_tools import camera as cam_mod

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no geni")

    monkeypatch.setattr(cam_mod, "GenICam", _Boom)
    assert find_camera_handler() is Webcam


def test_find_camera_handler_prefers_genicam_when_present(monkeypatch):
    """When GenICam reports a device we must return it."""
    from ipcv_tools import camera as cam_mod

    class _FakeGeni:
        def __init__(self, port=None, cti_file=None, buffer_size=1):
            self.port = port
            self.cti_file = cti_file

        def check_devices(self):
            return [object()]

    monkeypatch.setattr(cam_mod, "GenICam", _FakeGeni)
    assert find_camera_handler() is _FakeGeni


class _FakeCaptureDevice:
    """Mimic the subset of cv2.VideoCapture Webcam uses."""

    def set(self, key, value):
        return True

    def get(self, key):
        if key == cv2.CAP_PROP_FRAME_WIDTH:
            return 320.0
        if key == cv2.CAP_PROP_FRAME_HEIGHT:
            return 240.0
        return 0.0

    def release(self):
        self.released = True


def _enter_cam(cam):
    """Attach a cv2.VideoCapture without starting the worker thread."""
    cam.__enter__()
    return cam


def test_webcam_init_and_shape():
    """Webcam.__init__ should accept a port and keep buffer size."""
    cam = Webcam(port=0, cti_file=None, buffer_size=3)
    assert cam.port == 0
    assert cam.buffer.size == 3


def test_webcam_enter_reads_shape_and_frames(monkeypatch):
    """Webcam.__enter__ should attach a cv2.VideoCapture and expose its shape."""

    class _Dev(_FakeCaptureDevice):
        def read(self):
            return True, np.zeros((240, 320, 3), np.uint8)

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: _Dev())
    cam = _enter_cam(Webcam(port=0, buffer_size=3))
    try:
        assert cam.get_shape() == (240, 320)
    finally:
        cam.handler.release()


def test_webcam_acquire_converts_to_rgb(monkeypatch):
    """Webcam._acquire_element should return the BGR->RGB conversion result."""
    frame = np.zeros((1, 1, 3), np.uint8)
    frame[0, 0] = (255, 0, 0)  # BGR order: B=255 (pure blue)

    class _Dev(_FakeCaptureDevice):
        def read(self):
            return True, frame

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: _Dev())
    cam = _enter_cam(Webcam(port=0))
    try:
        result = cam._acquire_element()
    finally:
        cam.handler.release()
    # BGR (255, 0, 0) is pure blue; after BGR->RGB it becomes [0, 0, 255].
    assert result[0, 0].tolist() == [0, 0, 255]


def test_webcam_settings_passes_known_and_ignores_unknown_keys(monkeypatch):
    """Webcam.settings should forward known keys and skip unknown ones."""
    log = []

    class _Dev(_FakeCaptureDevice):
        def set(self, key, value):
            log.append((key, value))
            return True

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: _Dev())
    cam = _enter_cam(Webcam(port=0))
    try:
        cam.settings(width=123, unknown="nope")
    finally:
        cam.handler.release()
    keys = {k for k, _ in log}
    assert cv2.CAP_PROP_FRAME_WIDTH in keys
    # "unknown" must not be forwarded to cv2.
    assert cv2.CAP_PROP_GAIN not in keys


def test_webcam_default_port_on_linux():
    """On linux a None port should default to /dev/video0."""
    cam = Webcam(port=None, buffer_size=2)
    assert cam.port == "/dev/video0"


def test_webcam_string_port_open(monkeypatch):
    """A string port should open cv2.VideoCapture by passing the device path."""
    opened = {}

    class _Dev(_FakeCaptureDevice):
        def read(self):
            return True, np.zeros((4, 4, 3), np.uint8)

    def _capt(*a, **k):
        opened.update(a=a, k=k)
        return _Dev()

    monkeypatch.setattr(cv2, "VideoCapture", _capt)
    cam = _enter_cam(Webcam(port="/dev/video0"))
    try:
        cam._acquire_element()
    finally:
        cam.handler.release()
    assert opened["a"] == ("/dev/video0",)


def test_webcam_read_failure_requests_stop(monkeypatch):
    """On read() failure _acquire_element should return None and stop the camera."""

    class _Dev(_FakeCaptureDevice):
        def read(self):
            return False, None

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: _Dev())
    cam = _enter_cam(Webcam(port=0))
    try:
        stopped = {"n": 0}
        cam.stop = lambda: stopped.update(n=stopped["n"] + 1)
        got = cam._acquire_element()
        assert got is None
        assert stopped["n"] == 1
    finally:
        cam.handler.release()
