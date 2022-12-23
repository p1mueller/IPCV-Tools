"""Test Buffer class."""

import time

import numpy as np

from ipcv_tools.utilities import FPS

target_fps = 30
sleep_time = 1 / target_fps


def test_update():
    fps = FPS()
    fps.update()
    for _ in range(20):
        time.sleep(sleep_time)
        fps.update()
        rel_error = np.abs((fps.value - target_fps) / target_fps)
        assert rel_error < 0.01


if __name__ == "__main__":
    test_update()
