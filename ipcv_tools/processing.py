#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Worker utilities that help to run processing asynchronuously."""

import threading
from abc import ABC, abstractmethod

from ipcv_tools.utilities import Buffer


class Worker(ABC):
    """Worker thread with shared buffer."""

    def __init__(self, buffer_size=1, daemonic=False) -> None:
        """Initialize.

        Args:
            buffer_size (int): Buffer size. Defaults to 1.
            daemonic (bool): Thread is daemon. Defaults to False.
        """
        self.started = False
        self.buffer = Buffer(buffer_size)
        self.thread = threading.Thread(target=self._run)
        self.thread.setDaemon(daemonic)
        self.stop_event = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    @abstractmethod
    def _acquire_element(self):
        return

    def _is_ready(self):
        return len(self.buffer) > 0

    def _run(self):
        assert self.started
        while not self.stopped():
            element = self._acquire_element()
            self.buffer.add(element)

    def start(self):
        """Start thread."""
        assert self.thread is not None
        self.started = True
        self.thread.start()

    def stop(self):
        """Stop thread."""
        self.started = False
        self.stop_event.set()
        self.thread.join()

    def stopped(self):
        """Thread has stopped.

        Returns:
            bool: Is stopped
        """
        return self.stop_event.is_set()

    def get_next_element(self):
        """Get next element in buffer.

        Returns:
            Any: Next element
        """
        return self.buffer.pop()


class Processor(Worker):
    """Wrapper to encapsulate processing part of pipeline.

    Is executed in a thread and has a shared buffer.
    """

    def __init__(self, src_func, proc_func, buffer_size=1) -> None:
        """Initialize.

        Args:
            src_func: Function which polls frames from video stream
            proc_func: Function which processes the polled frames.
            buffer_size: Output buffer size. Defaults to 1.
        """
        super().__init__(buffer_size)
        assert callable(src_func)
        assert callable(proc_func)
        self.src_func = src_func
        self.proc_func = proc_func

    def _acquire_element(self):
        src = self.src_func()
        return self.proc_func(src)
