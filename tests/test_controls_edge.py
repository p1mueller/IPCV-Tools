"""Tests for the controls module edge cases not covered by test_slider."""

import numpy as np

from ipcv_tools.controls import Slider


def test_slider_clips_value_above_max(qapp):
    """A value above max_val should be clipped to the top step."""
    slider = Slider(min_val=0, max_val=10, steps=11, value=99)
    # int(0.5 + (99 - 0)/1) = 99 -> clipped to 10
    assert slider.slider.value() == 10
    assert slider.value() == 10


def test_slider_clips_value_below_min(qapp):
    """A value below min_val should be clipped to the bottom step."""
    slider = Slider(min_val=0, max_val=10, steps=11, value=-50)
    assert slider.slider.value() == 0
    assert slider.value() == 0


def test_slider_step_math_float_range(qapp):
    """Non-integer ranges should still round-trip within one step."""
    slider = Slider(min_val=0.5, max_val=2.5, steps=41, value=1.5)
    # step = (2.5 - 0.5) / 40 = 0.05
    assert np.isclose(slider.step, 0.05)
    assert np.isclose(slider.value(), 1.5, atol=1e-9)


def test_slider_change_text_negative(qapp):
    """change_text should render negative values using the ``-`` format flag."""
    slider = Slider(min_val=-10, max_val=10, steps=21, value=0)
    slider.change_text(-1.2345)
    # "-.3g" -> 3 sig digits, negative sign preserved.
    assert slider.slider_text.text() == "-1.23"
