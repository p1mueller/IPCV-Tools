#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Container that incorperates acquisition, processing and display."""

from typing import Any, Callable, Dict, Optional, Sequence, Union

from ipcv_tools.camera import Capture, Port, find_camera_handler
from ipcv_tools.processing import Processor
from ipcv_tools.ui import CameraUI
from ipcv_tools.viewer import ImageViewer


class Pipeline:
    """Container that incorperates acquisition, processing and display."""

    def __init__(
        self,
        func: Callable,
        cti_file: Optional[str] = None,
        port: Optional[Port] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        use_ui: Optional[bool] = True,
        img_names: Optional[Sequence[str]] = None,
        camera: Optional[Capture] = None,
        processor_buf_size: int = 1,
        camera_buf_size: int = 1,
        title: str = "IPCV Viewer",
    ) -> None:
        """Initialize.

        Args:
            func: Processing function.
            cti_file: Path to CTI file. Only needed for genicams. Defaults to None.
            port: Camera port. Defaults to None.
            width: Width of the received frame. Defaults to None.
            height: Height of the received frame. Defaults to None.
            camera: Camera object. Defaults to None.
            use_ui: Use UI or if false the image viewer. Defaults to True.
            img_names: Names of received images.
                Only needed when use_ui=True. Defaults to None.
            processor_buf_size: Buffer size for processor output. Defaults to 1.
            camera_buf_size: Buffer size for camera output. Defaults to 1.
            title: Title of window. Defaults to "IPCV Viewer".
        """

        self.width = width
        self.height = height

        if camera is None:
            cam_handler = find_camera_handler(port, cti_file)
            self.camera = cam_handler(port, cti_file, buffer_size=camera_buf_size)
        else:
            self.camera = camera
        self.processor = Processor(self.camera.get_next_element, func, processor_buf_size)

        if (width is None) or (height is None):
            height, width = self.camera.get_shape()

        self.viewer: Union[CameraUI, ImageViewer]
        if use_ui:
            if img_names is None:
                img_names = ["Frame"]
            self.viewer = CameraUI(
                width,
                height,
                img_names,
                self.processor.get_next_element,
                title,
            )
        else:
            self.viewer = ImageViewer(
                width, height, self.processor.get_next_element, title=title
            )

    def run(self, camera_settings: Optional[Dict[str, Any]] = None) -> int:
        """Run the computer vision pipeline.

        Args:
            camera_settings: Additional camera settings.
                Defaults to None.

        Returns:
            System exit code
        """
        with self.camera as cam, self.processor as proc:
            if (self.width is not None) and (self.height is not None):
                if camera_settings is None:
                    camera_settings = {}
                camera_settings["width"] = self.width
                camera_settings["height"] = self.height
            if camera_settings is not None:
                cam.settings(**camera_settings)
            cam.start()
            proc.start()
            ret = self.viewer.run()
        return ret
