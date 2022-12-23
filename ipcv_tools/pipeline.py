#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Container that incorperates acquisition, processing and display."""

import sys

from ipcv_tools.camera import find_camera_handler
from ipcv_tools.processing import Processor
from ipcv_tools.ui import CameraUI
from ipcv_tools.viewer import ImageViewer


class Pipeline:
    """Container that incorperates acquisition, processing and display."""

    def __init__(
        self,
        func,
        cti_file=None,
        port=None,
        width=None,
        height=None,
        use_ui=True,
        img_names=None,
        processor_buf_size=1,
        camera_buf_size=1,
        title="IPCV Viewer",
    ) -> None:
        """Initialize.

        Args:
            func (function): Processing function.
            cti_file (str): Path to CTI file. Only needed for genicams. Defaults to None.
            port (int|str): Camera port. Defaults to None.
            width (int): Width of the received frame. Defaults to None.
            height (int): Height of the received frame. Defaults to None.
            use_ui (bool): Use UI or if false the image viewer. Defaults to True.
            img_names (List[str]): Names of received images.
                Only needed when use_ui=True. Defaults to None.
            processor_buf_size (int): Buffer size for processor output. Defaults to 1.
            camera_buf_size (int): Buffer size for camera output. Defaults to 1.
            title (str): Title of window. Defaults to "IPCV Viewer".
        """
        if port is None:
            if sys.platform == "linux":
                port = "/dev/video0"
            else:
                port = 0

        self.width = width
        self.height = height

        cam_handler = find_camera_handler(cti_file)
        self.camera = cam_handler(port, buffer_size=camera_buf_size)
        self.processor = Processor(
            self.camera.get_next_element, func, processor_buf_size
        )

        if (width is None) or (height is None):
            height, width = self.camera.get_shape()

        if use_ui:
            self.viewer = CameraUI(
                width,
                height,
                img_names,
                self.processor.get_next_element,
                title,
            )
        else:
            self.viewer = ImageViewer(width, height, self.processor.get_next_element)
            self.viewer.set_title(title)

    def run(self, camera_settings=None):
        """Run the computer vision pipeline.

        Args:
            camera_settings (Dict[str, Any]): Additional camera settings.
                Defaults to None.

        Returns:
            int: System exit code
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
