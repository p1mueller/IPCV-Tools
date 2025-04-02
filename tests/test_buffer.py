"""Test Buffer class."""

import threading
import time

import numpy as np

from ipcv_tools.processing import Buffer

size = 20


def fill_buffer(buffer):
    """Fill buffer with data."""
    data = np.arange(size, dtype=int)
    for i in data:
        buffer.add(i)
        assert len(buffer._buf) == i + 1
    assert np.sum(np.abs(np.array(buffer._buf) - data)) == 0
    return data


def test_add():
    """Test adding elements to the buffer."""
    buffer = Buffer(size)
    assert len(buffer) == 0
    data = fill_buffer(buffer)
    buffer.add(size)
    assert len(buffer) == size
    assert np.sum(np.abs(np.array(buffer._buf) - (data + 1))) == 0


def test_pop():
    """Test popping elements from the buffer."""
    buffer = Buffer(size)
    assert len(buffer) == 0
    data = fill_buffer(buffer)

    # Check if elements are returned in the right order
    for d in data:
        assert buffer.pop() == d

    last_element = -1

    def pop():
        nonlocal last_element
        last_element = buffer.pop()

    # Buffer should be empty and popping should take forever
    thread = threading.Thread(target=pop)
    thread.start()
    thread.join(0.05)
    assert thread.is_alive()

    # Add an element to check if the pop thread finishes
    buffer.add(size)
    time.sleep(0.01)
    assert not thread.is_alive()
    assert last_element == size


def _random_sleep():
    return time.sleep(np.random.uniform(1e-3, 1e-2))


class TestAsync:
    """Test asynchronous read and write operations on the buffer."""

    data = list(range(size))[::-1]
    consumed_data = []
    finished_producing = False
    buffer = Buffer(size)

    def producing(self):
        """Produce data and add it to the buffer."""
        for d in self.data:
            _random_sleep()
            self.buffer.add(d)
        self.finished_producing = True

    def consuming(self):
        """Consume data from the buffer."""
        while (not self.finished_producing) or (self.buffer.not_empty_event.is_set()):
            _random_sleep()
            self.consumed_data.append(self.buffer.pop())

    def test_async_read_write(self):
        """Test asynchronous read and write operations on the buffer."""
        threads = [threading.Thread(target=f) for f in (self.producing, self.consuming)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert np.sum(np.abs(np.array(self.data) - np.array(self.consumed_data))) == 0


if __name__ == "__main__":
    test = TestAsync()
    test.test_async_read_write()
