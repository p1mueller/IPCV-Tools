#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""General purpose utilities for the package."""

import threading
from time import time


class FPS:
    """FPS evaluator."""

    def __init__(self, alpha=0.98) -> None:
        """Initialize.

        Args:
            alpha (float): IIR filter remember factor. Defaults to 0.98.
        """
        self.alpha = alpha
        self.value = 0.0
        self.last_time = None
        self.initialized = False

    def restart(self):
        """Restart current measurement."""
        self.last_time = None

    def update(self):
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

    def __init__(self, size) -> None:
        """Initialize.

        Args:
            size (int): Maximum buffer size
        """
        self.size = size
        self._buf = []
        self.not_empty_event = threading.Event()
        self.lock = threading.Lock()

    def add(self, element):
        """Add new element to buffer.

        If full the oldest element will be discarded.

        Args:
            element (Any): New element
        """
        with self.lock:
            if len(self._buf) >= self.size:
                self._buf.pop(0)
            self._buf.append(element)
            self.not_empty_event.set()

    def pop(self):
        """Return and remove oldest element from the buffer.

        Returns:
            Any: oldest element
        """
        self.not_empty_event.wait()
        with self.lock:
            element = self._buf.pop(0)
            if len(self._buf) <= 0:
                self.not_empty_event.clear()
        return element

    def __len__(self):
        return len(self._buf)
