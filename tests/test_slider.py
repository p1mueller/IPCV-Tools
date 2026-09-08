"""Tests for the Slider control widget in ipcv_tools.controls.

Uses the shared offscreen QApplication fixture (conftest.py) so widgets can
be constructed headless on the CI runner.
"""

from ipcv_tools.controls import Slider


def test_default_value_is_midpoint(qapp):
    """Without an explicit value the slider should sit at its midpoint."""
    slider = Slider(min_val=0, max_val=100, steps=101)
    # No explicit value -> middle step -> 50 steps * (100/100).
    assert slider.value() == 50.0


def test_value_within_range_and_step_math(qapp):
    """The mapping between internal steps and external value is linear."""
    slider = Slider(min_val=-10, max_val=10, steps=21)
    # step = (10 - (-10)) / (21 - 1) = 1
    assert slider.step == 1
    slider.slider.setValue(0)
    assert slider.value() == -10
    slider.slider.setValue(10)
    assert slider.value() == 0
    slider.slider.setValue(20)
    assert slider.value() == 10


def test_value_change_emits_signal(qapp):
    """Moving the internal slider should emit value_changed with new value."""
    slider = Slider(min_val=0, max_val=1, steps=101)
    seen = []
    slider.value_changed.connect(lambda v: seen.append(v))
    slider.slider.setValue(75)
    assert len(seen) == 1
    assert seen[0] == slider.value()


def test_change_text_formats_value(qapp):
    """change_text should render the value with 3 significant digits."""
    slider = Slider(min_val=0, max_val=10, steps=11)
    slider.change_text(3.14159)
    # "{value:-.3g}" -> 3 significant digits.
    assert slider.slider_text.text() == "3.14"
