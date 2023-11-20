#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Control widgets for UI."""

from typing import Optional

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal


class Slider(QtWidgets.QWidget):
    """Composed slider widget.

    Extension to allow floating values in slider. (QT slider only allows for int)
    """

    value_changed = pyqtSignal(float)

    def __init__(
        self,
        name: str = "",
        min_val: float = 0,
        max_val: float = 1,
        steps: int = 100,
        value: Optional[float] = None,
        width: int = 300,
        height: int = 10,
    ) -> None:
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
            value = int(np.clip(int(0.5 + (value - min_val) / self.step), 0, steps - 1))

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

    def value(self) -> float:
        """Get current value of slider."""
        return self.min_val + self.step * self.slider.value()

    def change_text(self, value: float) -> None:
        """Change text of slider value label.

        Args:
            value: New value to display
        """
        self.slider_text.setText(f"{value:-.3g}")

    def value_change_slider(self, value: int) -> None:
        """Slot for internal slider value change.

        Args:
            value: Internal slider value
        """
        val = self.value()
        self.change_text(val)
        self.value_changed.emit(val)
