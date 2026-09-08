"""Tests for GenICam camera paths, mocked against harvesters.

Harvester is patched at class level so GenICam runs deterministically without
real GenTL device files.
"""

import time

import numpy as np
import pytest

from ipcv_tools import camera as cam_mod


class _FakeNode:
    """Mimic a genicam node-map entry: set_value / value / min / max."""

    def __init__(self, value=0):
        self._v = value

    def set_value(self, v):
        self._v = v

    @property
    def value(self):
        return self._v

    @property
    def min(self):
        return 0

    @property
    def max(self):
        return 1000

    def execute(self):
        pass


class _FakeNodeMap:
    """Aggregates a fake node-map with realistic width/height values."""

    def __init__(self):
        self.PixelFormat = _FakeNode()
        self.DecimationHorizontal = _FakeNode()
        self.DecimationVertical = _FakeNode()
        self.Width = _FakeNode(value=32)
        self.Height = _FakeNode(value=16)
        self.AcquisitionMode = _FakeNode()
        self.TriggerMode = _FakeNode()
        self.TriggerSource = _FakeNode()
        self.TriggerSoftware = _FakeNode()
        self.AcquisitionFrameRateMode = _FakeNode()
        self.AcquisitionFrameRate = _FakeNode()
        self.Gain = _FakeNode()
        self.ExposureTime = _FakeNode()


class _FakeRemote:
    node_map = _FakeNodeMap()


class _FakeBuffer:
    class _C:
        data = np.zeros((4, 5), dtype="uint8")
        height = 4
        width = 5

    payload = type("payload", (), {"components": [_C()]})

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeHandler:
    keep_latest = False
    remote_device = _FakeRemote()

    def __init__(self):
        self.started = False
        self.stopped = False
        self.destroyed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def destroy(self):
        self.destroyed = True

    def fetch(self, timeout=0):
        return _FakeBuffer()


class _FakeHarvester:
    created_handlers = []

    def add_file(self, p):
        self.last_file = p

    def update(self):
        pass

    def create(self, port):
        h = _FakeHandler()
        _FakeHarvester.created_handlers.append(h)
        return h

    def reset(self):
        pass

    def __enter__(self):
        self.device_info_list = [{"serial": "mock"}]
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def geni(tmp_path, monkeypatch):
    """Provide a GenICam instance backed by mocked harvesters."""
    cti = tmp_path / "mock.cti"
    cti.write_bytes(b"x")
    monkeypatch.setattr(cam_mod, "Harvester", _FakeHarvester)
    return cam_mod.GenICam(port=0, cti_file=str(cti), buffer_size=2)


def test_genicam_check_devices(geni):
    """check_devices should return the fake device list from the mock harvesters."""
    assert len(geni.check_devices()) == 1


def test_genicam_enter_and_get_shape(geni):
    """__enter__ should create a handler and get_shape should read dimensions."""
    geni.__enter__()
    assert isinstance(geni.handler, _FakeHandler)
    assert geni.get_shape() == (16, 32)


def test_genicam_settings_soft_trigger(geni):
    """Settings with soft_trigger should set the trigger source to Software."""
    geni.__enter__()
    geni.settings(gain=1.0, exposure=10.0, decimation=2, soft_trigger=True)
    assert geni.soft_trigger is True
    nm = geni.handler.remote_device.node_map
    assert nm.TriggerSource.value == "Software"
    assert nm.Gain.value == 1.0
    assert nm.ExposureTime.value == 10.0
    assert nm.DecimationHorizontal.value == 2
    assert nm.DecimationVertical.value == 2
    assert nm.AcquisitionMode.value == "Continuous"


def test_genicam_settings_continuous_trigger_off(geni):
    """Settings with soft_trigger=False should turn trigger mode Off."""
    geni.__enter__()
    geni.settings(soft_trigger=False)
    assert geni.soft_trigger is False
    nm = geni.handler.remote_device.node_map
    assert nm.TriggerMode.value == "Off"
    assert nm.AcquisitionFrameRateMode.value == "On"


def test_genicam_acquire_element(geni):
    """_acquire_element should fetch a buffer and return a reshaped ndarray."""
    geni.__enter__()
    img = geni._acquire_element()
    assert img.shape == (4, 5)
    assert img.dtype == np.uint8


def test_genicam_discovery_from_gentl_path_env(tmp_path, monkeypatch):
    """With cti_file=None, GenICam should discover a .cti from the env var."""
    d = tmp_path / "gentl"
    d.mkdir()
    cti = d / "mock.cti"
    cti.write_bytes(b"x")
    monkeypatch.setenv("GENICAM_GENTL64_PATH", str(d))
    monkeypatch.setattr(cam_mod, "Harvester", _FakeHarvester)
    cam = cam_mod.GenICam(port=0, cti_file=None, buffer_size=1)
    assert cam.cti_file == str(cti)


def test_genicam_start_stop_lifecycle(geni):
    """start/stop should drive the handler and release harvesters."""
    with geni:
        geni.start()
        time.sleep(0.1)
        handler = geni.handler
        assert handler.started is True
    assert handler.stopped is True
    assert handler.destroyed is True
