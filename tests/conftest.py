"""Shared fixtures and setup for the test suite.

Creates a single shared QApplication at import time so that any module which
touches Qt (directly or via pyqtgraph) can be imported safely in headless CI.
"""

import os

import pytest
from PyQt5.QtWidgets import QApplication

import ipcv_tools.ui as _ui_mod

# Run Qt in offscreen mode so widgets can be constructed headlessly.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Run pygame headlessly (SDL dummy video/audio drivers).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def _ensure_qapp():
    """Create (or return) a single QApplication for the process."""

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


_QAPP = _ensure_qapp()

# CameraUI calls QApplication(sys.argv) in its constructor. Reuse the single
# shared instance instead so tests never spawn (or quit) an extra QApplication,
# which would destabilise the Qt event loop in-process.

_ui_mod.QtWidgets.QApplication = lambda *a, **k: _QAPP


@pytest.fixture(scope="session")
def qapp():
    """Provide the shared QApplication instance."""
    return _QAPP
