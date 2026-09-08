"""Tests for the Buffer/FPS utilities in ipcv_tools.utilities."""

import time

import numpy as np
import pytest

import ipcv_tools.utilities as utils
from ipcv_tools.utilities import FPS

target_fps = 20
sleep_time = 1 / target_fps


class FakeClock:
    """Deterministic clock that we advance manually."""

    def __init__(self):
        """Start the clock at time zero."""
        self.t = 0.0

    def __call__(self):
        """Return the current fake time."""
        return self.t

    def advance(self, dt):
        """Move the fake clock forward by ``dt`` seconds."""
        self.t += dt


@pytest.fixture
def clock(monkeypatch):
    """Patch the ``time`` used by FPS with a manually-advanced clock."""
    fake = FakeClock()
    monkeypatch.setattr(utils, "time", fake)
    return fake


def test_update():
    """Real-time: the running value should track the target fps closely."""
    fps = FPS()
    fps.update()
    for _ in range(20):
        time.sleep(sleep_time)
        fps.update()
        rel_error = np.abs((fps.value - target_fps) / target_fps)
        assert rel_error < 0.01


def test_value_is_zero_before_two_updates(clock):
    """Value must remain 0.0 until at least two measurements exist."""
    fps = FPS()
    fps.update()
    assert fps.value == 0.0
    assert fps.initialized is False


def test_restart_refreshes_baseline(clock):
    """restart() should reset the last time so the next update re-baselines."""
    fps = FPS(alpha=0.5)
    fps.update()
    clock.advance(10)
    fps.update()
    assert fps.initialized is True
    assert fps.value == pytest.approx(0.1)

    fps.restart()
    assert fps.last_time is None
    clock.advance(10)
    fps.update()
    # After restart the update only re-baselines (last_time was None).
    assert fps.value == pytest.approx(0.1)


def test_converges_to_true_rate(clock):
    """With a constant dt the estimator converges to the true rate 1/dt.

    Convergence holds regardless of alpha given enough samples.
    """
    fps = FPS(alpha=0.9)
    fps.update()
    for _ in range(200):
        clock.advance(10)  # dt = 10 -> true rate = 0.1 fps
        fps.update()
    assert fps.value == pytest.approx(0.1, rel=1e-3)


if __name__ == "__main__":
    test_update()
