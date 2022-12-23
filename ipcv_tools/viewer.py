#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Viewer to display frames from a video source."""

import numpy as np
import pygame

from ipcv_tools.utilities import FPS


class _Font:
    defaults = {"name": "freesansbold.ttf", "size": 24, "color": (0, 0, 255)}

    def __init__(self, values) -> None:
        if values is None:
            values = {}
        params = {}
        for k, v in self.defaults.items():
            params[k] = values.get(k, v)
        self.params = params

    def as_dict(self):
        return self.params


class ImageViewer:
    """Display video stream with pygame."""

    def __init__(self, width, height, func, font=None):
        """Initialize.

        Args:
            width (int): Width of images
            height (int): Height of images
            func (function): Function to get next frame from video stream
            font (Dict[str, Any]): Font to display FPS. Defaults to None.
        """
        parsed_font = _Font(font).as_dict()
        pygame.init()
        self.quit = False
        self.fps = FPS()
        self.width = width
        self.height = height
        self.func = func

        self.display = pygame.display.set_mode((width, height))
        self.font = pygame.font.Font(parsed_font["name"], parsed_font["size"])
        self.font_color = parsed_font["color"]

        text = self._create_fps_text()
        rect = text.get_rect()
        size = self.display.get_size()
        self.txt_coord = (size[0] - 1.05 * rect.width, size[1] - 1.05 * rect.height)
        self.update(np.zeros((height, width, 3), np.uint8))

    def _create_fps_text(self):
        return self.font.render(f"FPS: {self.fps.value:5.1f}", True, self.font_color)

    def is_running(self):
        """Get if running.

        Returns:
            bool: Is running
        """
        return not self.quit

    def set_title(self, title):
        """Set window title.

        Args:
            title: Window title
        """
        pygame.display.set_caption(title)

    def update(self, img):
        """Update displayed image.

        Args:
            img: New image
        """
        img = np.swapaxes(img, 0, 1)
        surf = pygame.surfarray.make_surface(img)
        text = self._create_fps_text()
        self.display.blits([(surf, (0, 0)), (text, self.txt_coord)])

        events = pygame.event.get()
        self.quit = any([e.type == pygame.QUIT for e in events])

        pygame.display.update()
        self.fps.update()

    def stop(self):
        """Stop viewing process."""
        self.quit = True
        pygame.quit()

    def run(self):
        """Run the acquisition process."""
        while self.is_running():
            frames = self.func()
            if isinstance(frames, tuple):
                frame = frames[-1]
            else:
                frame = frames
            self.update(frame)
        return


if __name__ == "__main__":
    size = (600, 600)

    def _random_noise_img():
        image = np.random.random(size + (3,)) * 255.0
        return image.astype("uint8")

    viewer = ImageViewer(*size, _random_noise_img)
    viewer.set_title("Random Noise")
    viewer.run()
