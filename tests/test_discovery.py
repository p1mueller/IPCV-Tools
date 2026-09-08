"""Tests for camera discovery and small utilities."""

import os
import pathlib

from ipcv_tools import utilities
from ipcv_tools.camera import find_cti_files


def test_find_cti_files_reads_gentl_path_env(tmp_path, monkeypatch):
    """find_cti_files should glob *.cti across all entries of the env var."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "one.cti").write_text("a")
    (dir_a / "not_a_cti.txt").write_text("ignore")
    (dir_b / "two.cti").write_text("b")

    monkeypatch.setenv("GENICAM_GENTL64_PATH", f"{dir_a};{dir_b}")
    found = find_cti_files()
    names = sorted(pathlib.Path(f).name for f in found)
    assert names == ["one.cti", "two.cti"]


def test_find_cti_files_returns_empty_when_unset(tmp_path, monkeypatch):
    """Without a configured search path no CTI files must be discovered."""
    monkeypatch.delenv("GENICAM_GENTL64_PATH", raising=False)
    assert find_cti_files() == []


def test_delete_variable_removes_known_and_ignores_unknown(monkeypatch):
    """delete_variable must remove an existing var and no-op on missing ones."""
    monkeypatch.setenv("FOO_TEST_VAR", "42")
    utilities.delete_variable("FOO_TEST_VAR")
    assert os.environ.get("FOO_TEST_VAR") is None
    utilities.delete_variable("FOO_TEST_NEVER_SET")
    assert os.environ.get("FOO_TEST_NEVER_SET") is None
