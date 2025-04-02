#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Camera interfaces for easy use."""

import os
import pathlib
import sys
import time
from abc import abstractmethod
from glob import glob
from typing import Any, Optional, Sequence, Tuple, Type, Union

import cv2
import numpy as np
from harvesters.core import Harvester, ImageAcquirer

from ipcv_tools.processing import Worker

Port = Union[int, str]


class Capture(Worker):
    """Abstract capture class.

    Is used as base class for all possible camera interfaces.
    """

    def __init__(
        self,
        port: Optional[Port] = None,
        buffer_size: int = 1,
    ) -> None:
        """Initialize.

        Args:
            port: Camera port. Defaults to None.
            buffer_size: Buffer size. Defaults to 1.
        """
        super().__init__(buffer_size)
        self.port = port
        self.handler: Any = None

    @abstractmethod
    def settings(self, **kwargs: Any) -> None:
        """Set settings of camera."""

    @abstractmethod
    def get_shape(self) -> Sequence[int]:
        """Get shape of output frames."""


def find_camera_handler(
    port: Optional[Port] = None, cti_file: Optional[str] = None
) -> Type[Capture]:
    """Get appropriate camera handler.

    Args:
        port: Camera port. If None will search for the first best camera. Defaults to None.
        cti_file: CTI file to use (only when GenICam). Defaults to None.

    Returns:
        Appropriate capture class
    """
    geni = GenICam(port=port, cti_file=cti_file)
    if len(geni.check_devices()) > 0:
        return GenICam
    return Webcam


def find_cti_files() -> Sequence[str]:
    """Find CTI file which defines the interface for the genicam.

    Returns:
        Path to file
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


class GenICam(Capture):
    """Class to control a genicam (see https://en.wikipedia.org//wiki/GenICam).

    Cameras in the IPCV are genicams.
    """

    def __init__(
        self,
        port: Optional[Port] = None,
        cti_file: Optional[str] = None,
        buffer_size: int = 1,
    ) -> None:
        """Initialize.

        Args:
            port: Camera port. Defaults to None.
            cti_file: Path to CTI file. Defaults to None.
            buffer_size: Buffer size. Defaults to 1.
        """
        super().__init__(port, buffer_size)
        if cti_file is None:
            files = find_cti_files()
            if len(files) > 0:
                cti_file = files[0]
            else:
                root = pathlib.Path(__file__).parent
                cti_file = str(root / "mvGenTLProducer.cti")
        assert os.path.exists(cti_file)

        self.cti_file = cti_file
        self.harvester: Optional[Harvester] = None
        self.handler: Optional[ImageAcquirer]

    def check_devices(self) -> Sequence[Any]:
        """Check if genicam devices are found.

        Returns:
            Available genicam devices.
        """
        with Harvester() as harvester:
            harvester.add_file(str(self.cti_file))
            harvester.update()
            device_info = harvester.device_info_list.copy()
        return device_info

    def __enter__(self) -> Capture:
        self.harvester = Harvester()
        assert self.harvester is not None
        self.harvester.add_file(str(self.cti_file))
        self.harvester.update()
        self.handler = self.harvester.create(self.port)
        assert self.handler is not None
        self.handler.keep_latest = True
        return super().__enter__()

    def _acquire_element(self) -> np.ndarray:
        assert self.handler is not None
        self.handler.remote_device.node_map.TriggerSoftware.execute()
        with self.handler.fetch(timeout=5.0) as buffer:
            component = buffer.payload.components[0]
            img = component.data.reshape(component.height, component.width).copy()
        return img

    def settings(
        self,
        decimation: Optional[int] = None,
        gain: Optional[float] = None,
        exposure: Optional[float] = None,
        pixel_format: str = "RGB8",
        **_: Any,
    ) -> None:
        """Set settings of camera.

        Args:
            decimation: Decimation factor. Range 1 - 4. Defaults to None.
            gain: Gain in dB. Range 0.0 - 28.0. Defaults to None.
            exposure: Exposure time in us. Range 30.0 - 1000000.0. Defaults to None.
            pixel_format: Format of pixels. "RGB8" recommended. Defaults to None.
        """
        assert self.handler is not None
        node_map = self.handler.remote_device.node_map
        if pixel_format is not None:
            node_map.PixelFormat.set_value(pixel_format)
        if decimation is not None:
            node_map.DecimationHorizontal.set_value(decimation)
            node_map.DecimationVertical.set_value(decimation)
            node_map.Width.set_value(node_map.Width.max)
            node_map.Height.set_value(node_map.Height.max)
        if gain is not None:
            node_map.Gain.set_value(float(gain))
        if exposure is not None:
            node_map.ExposureTime.set_value(float(exposure))

    def get_shape(self) -> Sequence[int]:
        """Get shape of frames.

        Returns:
            Shape of frames (Channels are leftout).
        """
        assert self.handler is not None
        node_map = self.handler.remote_device.node_map
        height = node_map.Height.value
        width = node_map.Width.value
        return (height, width)

    def start(self) -> None:
        """Start capturing of frames from camera."""
        assert self.harvester is not None
        assert self.handler is not None
        self.handler.start()
        super().start()

    def stop(self) -> None:
        """Stop capturing frames from camera."""
        super().stop()
        assert self.handler is not None
        assert self.harvester is not None
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

    def __init__(self, port: Optional[Port] = None, buffer_size: int = 1) -> None:
        """Initialize.

        Args:
            port: Camera port. Defaults to None.
            buffer_size: Buffer size. Defaults to 1.
        """
        if port is None:
            if sys.platform == "linux":
                port = "/dev/video0"
            else:
                port = 0
        super().__init__(port, buffer_size)

    def __enter__(self) -> Capture:
        self.handler: Optional[cv2.VideoCapture]
        if self.port is None:
            self.handler = cv2.VideoCapture()
        elif isinstance(self.port, int):
            self.handler = cv2.VideoCapture(index=self.port)
        else:
            self.handler = cv2.VideoCapture(self.port)
        assert self.handler is not None
        self.handler.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
        return super().__enter__()

    def _acquire_element(self) -> Optional[np.ndarray]:
        assert self.handler is not None
        ret, frame = self.handler.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.stop()
        return None

    def get_shape(self) -> Tuple[int, int]:
        """Get shape of frames.

        Returns:
            Shape of frames (Channels are leftout).
        """
        assert self.handler is not None
        height, width = [
            int(np.round(self.handler.get(self._settings_map[k])))
            for k in ["height", "width"]
        ]
        return (height, width)

    def settings(self, **kwargs: Any) -> None:
        """Set settings of camera."""
        assert self.handler is not None
        for key, value in kwargs.items():
            cv_key = self._settings_map.get(key)
            if cv_key is None:
                continue
            self.handler.set(cv_key, value)

    def stop(self) -> None:
        """Stop capturing frames from camera."""
        assert self.handler is not None
        super().stop()
        self.handler.release()


class DataCam(Capture):
    """Camera mock that displays a given static image."""

    def __init__(
        self,
        image: Union[np.ndarray, str, os.PathLike],
        colored: bool = False,
        buffer_size: int = 1,
    ) -> None:
        """Initialize.

        Args:
            image: Array or path of the static image
            colored: Use color image. Defaults to False.
            buffer_size: Buffer size. Defaults to 1.
        """
        super().__init__(None, buffer_size)

        data_folder = pathlib.Path(__file__).parent / "data"
        data_paths = list(data_folder.glob("*"))
        names = [p.stem for p in data_paths]
        flag = None
        if isinstance(image, np.ndarray):
            ndim = image.ndim
            if (2 > ndim) or (ndim > 3):
                raise ValueError(
                    "Image must have at have 2 to 3 dimensions, "
                    f"but has dimensionality {ndim} with shape {image.shape}."
                )
            if (ndim == 2) and colored:
                flag = cv2.COLOR_GRAY2RGB
            elif (ndim == 3) and not colored:
                flag = cv2.COLOR_RGB2GRAY
            _img = image.copy()
        else:
            if image in names:
                index = names.index(str(image))
                image = data_paths[index]
            _img = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if _img is None:
                raise ValueError(f"Image path '{image}' does not exist.")

            if colored:
                flag = cv2.COLOR_BGR2RGB
            else:
                flag = cv2.COLOR_BGR2GRAY
        if flag is not None:
            self._image = cv2.cvtColor(_img, flag)
        else:
            self._image = _img.copy()
        self._orig_image = self._image.copy()

    def _acquire_element(self) -> np.ndarray:
        time.sleep(5e-3)
        return self._image

    def settings(self, decimation: int = 1, **_: Any) -> None:
        """Set settings of camera.

        Args:
            decimation: Decimation factor. Range 1 - 4. Defaults to None.
        """
        self._image = self._orig_image[::decimation, ::decimation]

    def get_shape(self) -> Sequence[int]:
        """Get shape of output frames."""
        return self._image.shape[:2]


class NoisyDataCam(DataCam):
    """Camera mock that displays a given static image with added (dynamic) noise."""

    def __init__(
        self,
        image: Union[np.ndarray, str, os.PathLike],
        colored: bool = False,
        buffer_size: int = 1,
    ) -> None:
        """Initialize.

        Args:
            image: Array or path of the static image
            colored: Use color image. Defaults to False.
            buffer_size: Buffer size. Defaults to 1.
        """
        super().__init__(image, colored, buffer_size)
        self._std = 0.0

    def settings(self, decimation: int = 1, rel_std: float = 0.1, **kwargs: Any) -> None:
        """Set settings of camera.

        Args:
            decimation: Decimation factor. Range 1 - 4. Defaults to None.
            rel_std: Relative standard deviation of noise compared to image STD.
                Defaults to 0.1.
            kwargs: Additional arguments for DataCam.
        """
        super().settings(decimation, **kwargs)
        self._std = np.abs(rel_std) * self._image.std()

    def _acquire_element(self) -> np.ndarray:
        noise = np.random.normal(scale=self._std, size=self._image.shape)
        return np.clip(np.round(self._image + noise), 0, 255).astype(np.uint8)
