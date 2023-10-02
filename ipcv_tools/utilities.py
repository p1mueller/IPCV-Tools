#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""General purpose utilities for the package."""

import os
import sys
import threading
from time import time
from typing import Any, List, Optional


def delete_variable(name: str) -> None:
    """Delete environmental variable if it exists.

    Args:
        name (str): Name of environment variable
    """
    if os.environ.get(name) is not None:
        os.environ.pop(name)


def fix_cv2_issue() -> None:
    """Fix XCB issue when opencv an Qt are used together."""
    ci_and_not_headless = False
    try:
        from cv2.version import ci_build, headless

        ci_and_not_headless = ci_build and not headless
    except Exception:
        pass
    if sys.platform.startswith("linux") and ci_and_not_headless:
        delete_variable("QT_QPA_PLATFORM_PLUGIN_PATH")
    if sys.platform.startswith("linux") and ci_and_not_headless:
        delete_variable("QT_QPA_FONTDIR")


class FPS:
    """FPS evaluator."""

    def __init__(self, alpha: float = 0.98) -> None:
        """Initialize.

        Args:
            alpha: IIR filter remember factor. Defaults to 0.98.
        """
        self.alpha = alpha
        self.value = 0.0
        self.last_time: Optional[float] = None
        self.initialized = False

    def restart(self) -> None:
        """Restart current measurement."""
        self.last_time = None

    def update(self) -> None:
        """Take new measurement and add to value."""
        t = time()
        if self.last_time is not None:
            dt = t - self.last_time
            if self.initialized:
                self.value = self.alpha * self.value + (1 - self.alpha) / dt
            else:
                self.initialized = True
                self.value = 1 / dt
        self.last_time = t


class Buffer:
    """Shared buffer."""

    def __init__(self, size: int) -> None:
        """Initialize.

        Args:
            size: Maximum buffer size
        """
        self.size = size
        self._buf: List[Any] = []
        self.not_empty_event = threading.Event()
        self.lock = threading.Lock()

    def add(self, element: Any) -> None:
        """Add new element to buffer.

        If full the oldest element will be discarded.

        Args:
            element: New element
        """
        with self.lock:
            if len(self._buf) >= self.size:
                self._buf.pop(0)
            self._buf.append(element)
            self.not_empty_event.set()

    def pop(self) -> Any:
        """Return and remove oldest element from the buffer.

        Returns:
            Oldest element
        """
        self.not_empty_event.wait()
        with self.lock:
            element = self._buf.pop(0)
            if len(self._buf) <= 0:
                self.not_empty_event.clear()
        return element

    def __len__(self) -> int:
        return len(self._buf)
