#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Worker utilities that help to run processing asynchronuously."""

import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np

from ipcv_tools.utilities import Buffer


class Worker(ABC):
    """Worker thread with shared buffer."""

    def __init__(self, buffer_size: int = 1, daemonic: bool = False) -> None:
        """Initialize.

        Args:
            buffer_size: Buffer size. Defaults to 1.
            daemonic: Thread is daemon. Defaults to False.
        """
        self.started = False
        self.buffer = Buffer(buffer_size)
        self.thread = threading.Thread(target=self._run)
        self.thread.setDaemon(daemonic)
        self.stop_event = threading.Event()

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    @abstractmethod
    def _acquire_element(self) -> Optional[np.ndarray]:
        pass

    def _is_ready(self) -> bool:
        return len(self.buffer) > 0

    def _run(self) -> None:
        assert self.started
        while not self.stopped():
            element = self._acquire_element()
            self.buffer.add(element)

    def start(self) -> None:
        """Start thread."""
        assert self.thread is not None
        self.started = True
        self.thread.start()

    def stop(self) -> None:
        """Stop thread."""
        self.started = False
        self.stop_event.set()
        self.thread.join()

    def stopped(self) -> bool:
        """Thread has stopped.

        Returns:
            Is stopped
        """
        return self.stop_event.is_set()

    def get_next_element(self) -> Any:
        """Get next element in buffer.

        Returns:
            Next element
        """
        return self.buffer.pop()


class Processor(Worker):
    """Wrapper to encapsulate processing part of pipeline.

    Is executed in a thread and has a shared buffer.
    """

    def __init__(
        self, src_func: Callable, proc_func: Callable, buffer_size: int = 1
    ) -> None:
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

    def _acquire_element(self) -> Any:
        src = self.src_func()
        return self.proc_func(src)
