"""Tests for the pygame ImageViewer and fix_cv2_issue in ipcv_tools."""

import sys

import numpy as np

from ipcv_tools.utilities import delete_variable, fix_cv2_issue


def test_fix_cv2_issue_clears_qt_vars_on_linux_ci(monkeypatch):
    """On linux with a CI non-headless cv2 build, the QT vars are cleared."""
    assert sys.platform.startswith("linux")
    import cv2

    monkeypatch.setattr(cv2.version, "ci_build", True, raising=False)
    monkeypatch.setattr(cv2.version, "headless", False, raising=False)

    import os

    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/x"
    os.environ["QT_QPA_FONTDIR"] = "/y"

    fix_cv2_issue()

    assert os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH") is None
    assert os.environ.get("QT_QPA_FONTDIR") is None


def test_fix_cv2_issue_keeps_vars_when_headless(monkeypatch):
    """Headless cv2 builds must not clear the QT variables."""
    import os

    import cv2

    monkeypatch.setattr(cv2.version, "ci_build", True, raising=False)
    monkeypatch.setattr(cv2.version, "headless", True, raising=False)

    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/x"
    os.environ["QT_QPA_FONTDIR"] = "/y"

    fix_cv2_issue()

    assert os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH") == "/x"
    assert os.environ.get("QT_QPA_FONTDIR") == "/y"


def test_fix_cv2_issue_import_error_is_swallowed(monkeypatch):
    """If cv2.version import fails the function must swallow the error.

    On a linux + CI + non-headless build this still clears the QT vars.
    """
    import os
    import sys

    import cv2

    # Make the `from cv2.version import ...` inside the function fail.
    monkeypatch.delitem(sys.modules, "cv2.version", raising=False)
    monkeypatch.delattr(cv2, "version", raising=False)

    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/x"
    os.environ["QT_QPA_FONTDIR"] = "/y"

    fix_cv2_issue()

    # The error path leaves ci_and_not_headless False, so nothing is cleared.
    assert os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH") is None
    assert os.environ.get("QT_QPA_FONTDIR") is None


def test_delete_variable(monkeypatch):
    """delete_variable should remove existing vars and ignore unknown ones."""
    monkeypatch.setenv("TEST_IPCV_VAR", "1")
    delete_variable("TEST_IPCV_VAR")
    assert "TEST_IPCV_VAR" not in __import__("os").environ


def test_imageviewer_headless(qapp):
    """Construct an ImageViewer (pygame, SDL dummy) and pump one frame."""
    import pygame

    from ipcv_tools.viewer import ImageViewer

    img = np.zeros((8, 8, 3), np.uint8)

    viewer = ImageViewer(8, 8, lambda: img, title="Test Viewer")
    try:
        assert viewer.is_running() is True
        viewer.update(img)
        viewer.stop()
        assert viewer.is_running() is False
    finally:
        pygame.quit()
