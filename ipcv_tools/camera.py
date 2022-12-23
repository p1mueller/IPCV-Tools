#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Camera interfaces for easy use."""

import os
from abc import abstractmethod
from glob import glob
from pathlib import Path

import cv2
from harvesters.core import Harvester

from ipcv_tools.processing import Worker


def find_camera_handler(cti_file=None):
    """Get appropriate camera handler.

    Args:
        cti_file (str): CTI file to use (only when GenICam). Defaults to None.

    Returns:
        Capture: Appropriate capture class
    """
    geni = GenICam(cti_file=cti_file)
    if len(geni.check_devices()) > 0:
        return GenICam
    return Webcam


def find_cti_files():
    """Find CTI file which defines the interface for the genicam.

    Returns:
        str: Path to file
    """
    gentl_path = os.getenv("GENICAM_GENTL64_PATH")
    if gentl_path is None:
        gentl_path = ""
    base_paths = gentl_path.split(";")

    match_groups = [glob(f"{p}/*.cti") for p in base_paths]
    files = []
    for matches in match_groups:
        files.extend(matches)
    return files


class Capture(Worker):
    """Abstract capture class.

    Is used as base class for all possible camera interfaces.
    """

    @abstractmethod
    def settings(self, **kwargs):
        """Set settings of camera."""
        pass

    @abstractmethod
    def get_shape(self):
        """Get shape of output frames."""
        return


class GenICam(Capture):
    """Class to control a genicam (see https://en.wikipedia.org//wiki/GenICam).

    Cameras in the IPCV are genicams.
    """

    def __init__(self, port=None, cti_file=None, buffer_size=1) -> None:
        """Initialize.

        Args:
            port (int|str): Camera port. Defaults to None.
            cti_file (str): Path to CTI file. Defaults to None.
            buffer_size (int): Buffer size. Defaults to 1.
        """
        super().__init__(buffer_size)
        if cti_file is None:
            files = find_cti_files()
            if len(files) > 0:
                cti_file = files[0]
            else:
                root = Path(__file__).parent
                cti_file = root / "mvGenTLProducer.cti"
        assert os.path.exists(cti_file)

        self.port = port
        self.cti_file = cti_file
        self.harvester = None
        self.handler = None

    def check_devices(self):
        """Check if genicam devices are found.

        Returns:
            List[str]: Available genicam devices.
        """
        with Harvester() as harvester:
            harvester.add_file(str(self.cti_file))
            harvester.update()
            device_info = harvester.device_info_list
        return device_info

    def __enter__(self):
        self.harvester = Harvester()
        self.harvester.add_file(str(self.cti_file))
        self.harvester.update()
        self.handler = self.harvester.create(self.port)
        self.handler.keep_latest = True
        return super().__enter__()

    def _acquire_element(self):
        with self.handler.fetch(timeout=1.0) as buffer:
            component = buffer.payload.components[0]
            img = component.data.reshape(component.height, component.width, 3).copy()
        return img

    def settings(
        self,
        width=None,  # TODO: See how it is handled
        height=None,
        decimation=None,
        gain=None,
        exposure=None,
        pixel_format="RGB8",
    ):
        """Set settings of camera."""
        node_map = self.handler.remote_device.node_map
        node_map.PixelFormat.value = pixel_format
        if decimation is not None:
            node_map.DecimationHorizontal.value = decimation
            node_map.DecimationVertical.value = decimation
            # TODO: Must width and height also be set?
        if gain is not None:
            node_map.Gain.value = gain  # TODO: What type and range?
        if exposure is not None:
            node_map.ExposureTime = exposure  # TODO: What type and range?

    def get_shape(self):
        """Get shape of frames.

        Returns:
            Tuple[int]: Shape of frames (Channels are leftout).
        """
        node_map = self.handler.remote_device.node_map
        height = node_map.Height.value
        width = node_map.Width.value
        return (height, width)

    def start(self):
        """Start capturing of frames from camera."""
        assert self.harvester is not None
        assert self.handler is not None
        self.handler.start()
        super().start()

    def stop(self):
        """Stop capturing frames from camera."""
        super().stop()
        self.handler.stop()
        self.handler.destroy()
        self.harvester.reset()


class Webcam(Capture):
    """Class to control a webcam with opencv."""

    _settings_map = {
        "width": cv2.CAP_PROP_FRAME_WIDTH,
        "height": cv2.CAP_PROP_FRAME_HEIGHT,
        "gain": cv2.CAP_PROP_GAIN,
        "exposure": cv2.CAP_PROP_EXPOSURE,
    }

    def __init__(self, port=None, buffer_size=1) -> None:
        """Initialize.

        Args:
            port (int|str): Camera port. Defaults to None.
            buffer_size (int): Buffer size. Defaults to 1.
        """
        super().__init__(buffer_size)
        self.port = port
        self.handler = None

    def __enter__(self):
        self.handler = cv2.VideoCapture(self.port)
        self.handler.set(
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G")
        )
        return super().__enter__()

    def _acquire_element(self):
        ret, frame = self.handler.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.stop()
        return None

    def get_shape(self):
        """Get shape of frames.

        Returns:
            Tuple[int]: Shape of frames (Channels are leftout).
        """
        height, width = [
            self.handler.get(self._settings_map[k]) for k in ["height", "width"]
        ]
        return (height, width)

    def settings(self, **kwargs):
        """Set settings of camera."""
        assert self.handler is not None
        for key, value in kwargs.items():
            cv_key = self._settings_map.get(key)
            if cv_key is None:
                continue
            self.handler.set(cv_key, value)

    def stop(self):
        """Stop capturing frames from camera."""
        super().stop()
        self.handler.release()
