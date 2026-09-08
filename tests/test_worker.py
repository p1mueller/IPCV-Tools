"""Tests for the Worker and Processor classes in ipcv_tools.processing."""

import time

from ipcv_tools.processing import Processor, Worker


class _FakeWorker(Worker):
    """Concrete Worker that emits a fixed sequence of elements."""

    def __init__(self, values, buffer_size=1):
        super().__init__(buffer_size=buffer_size)
        self.values = list(values)

    def _acquire_element(self):
        if len(self.values) > 0:
            return self.values.pop(0)
        # No more data: block until stopped to avoid burning the CPU.
        self.stop_event.wait(timeout=0.05)
        return None


def test_worker_processes_elements_in_order():
    """Worker should buffer everything the source emits until it is stopped."""
    worker = _FakeWorker([1, 2, 3], buffer_size=10)
    worker.start()
    deadline = time.time() + 2
    while len(worker.buffer) < 3 and time.time() < deadline:
        time.sleep(0.01)
    assert len(worker.buffer) == 3
    got = [worker.buffer.pop() for _ in range(3)]
    worker.stop()
    assert got == [1, 2, 3]


def test_worker_stopped_flag():
    """stopped() should reflect the state of the stop_event."""
    worker = _FakeWorker([1])
    assert worker.stopped() is False
    worker.stop_event.set()
    assert worker.stopped() is True


def test_processor_applies_func_to_every_element():
    """Processor should apply proc_func to the output of src_func."""
    counter = {"n": 0}
    total = 4
    state = {}

    def src_func():
        counter["n"] += 1
        if counter["n"] <= total:
            return counter["n"]
        # Source exhausted: idle until the worker is stopped so we do not
        # flood the buffer with sentinel values.
        stop = state.get("stop")
        if stop is not None:
            stop.wait()
        return -1

    proc = Processor(src_func=src_func, proc_func=lambda x: x * 10, buffer_size=10)
    state["stop"] = proc.stop_event
    proc.start()
    deadline = time.time() + 2
    while len(proc.buffer) < total and time.time() < deadline:
        time.sleep(0.01)
    assert len(proc.buffer) == total
    proc.stop()
    got = [proc.get_next_element() for _ in range(total)]
    assert got == [10, 20, 30, 40]


def test_worker_context_manager_stops_thread():
    """Exiting the context manager should call stop() and join the thread."""
    worker = _FakeWorker([1, 2], buffer_size=10)
    with worker:
        worker.start()
        time.sleep(0.05)
        assert worker.thread.is_alive()
    assert not worker.thread.is_alive()
