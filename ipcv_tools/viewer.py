#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Viewer to display frames from a video source."""

from threading import Event
from typing import Any, Callable, Dict

import numpy as np
import pygame

from ipcv_tools.utilities import FPS


class _Font:
    defaults = {"name": "freesansbold.ttf", "size": 24, "color": (0, 0, 255)}

    def __init__(self, values: Dict[str, Any] = None) -> None:
        if values is None:
            values = {}
        params = {}
        for k, v in self.defaults.items():
            params[k] = values.get(k, v)
        self.params = params

    def as_dict(self) -> Dict[str, Any]:
        return self.params


class ImageViewer:
    """Display video stream with pygame."""

    def __init__(
        self,
        width: int,
        height: int,
        func: Callable,
        font: Dict[str, Any] = None,
        title: str = "IPCV Viewer",
    ) -> None:
        """Initialize.

        Args:
            width: Width of images
            height: Height of images
            func: Function to get next frame from video stream
            font: Font to display FPS. Defaults to None.
            title: Window title. Defaults to "IPCV Viewer".
        """
        parsed_font = _Font(font).as_dict()
        pygame.init()
        self.quit = Event()
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
        self.set_title(title)

    def _create_fps_text(self) -> pygame.surface.Surface:
        return self.font.render(f"FPS: {self.fps.value:5.1f}", True, self.font_color)

    def is_running(self) -> bool:
        """Get if running.

        Returns:
            Is running
        """
        return not self.quit.is_set()

    def set_title(self, title: str) -> None:
        """Set window title.

        Args:
            title: Window title
        """
        pygame.display.set_caption(title)

    def update(self, img: np.ndarray) -> None:
        """Update displayed image.

        Args:
            img: New image
        """
        img = np.swapaxes(img, 0, 1)
        surf = pygame.surfarray.make_surface(img)
        text = self._create_fps_text()
        self.display.blits([(surf, (0, 0)), (text, self.txt_coord)])

        events = pygame.event.get()
        if any([e.type == pygame.QUIT for e in events]):
            self.quit.set()

        pygame.display.update()
        self.fps.update()

    def stop(self) -> None:
        """Stop viewing process."""
        self.quit.set()

    def run(self) -> int:
        """Run the acquisition process."""
        while self.is_running():
            frames = self.func()
            if isinstance(frames, tuple):
                frame = frames[-1]
            else:
                frame = frames
            self.update(frame)
        pygame.quit()
        return 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage.exposure import histogram

    def _random_noise_img() -> np.ndarray:
        image = np.random.random(size + (3,))
        image *= 255.999
        return image.astype("uint8")

    def _run() -> np.ndarray:
        frame = _random_noise_img()
        hist, _ = histogram(
            frame, source_range="dtype", normalize=True, channel_axis=-1
        )
        for line, hist_i in zip(lines, hist):
            line.set_ydata(hist_i)
        if plt.fignum_exists(fig.number):
            fig.canvas.draw()
            fig.canvas.flush_events()
        else:
            viewer.stop()
        return frame

    fig_num = 1
    size = (600, 600)

    hist, mids = histogram(
        _random_noise_img(), source_range="dtype", normalize=True, channel_axis=-1
    )
    fig, ax = plt.subplots(num=fig_num, clear=True, constrained_layout=True)
    lines = []
    for hist_i, color in zip(hist, ["r", "g", "b"]):
        lines.extend(ax.plot(mids, hist_i))
    ax.grid(1)
    ax.set_xlim(-0.5, 255.5)
    ax.set_ylim(0, None)
    plt.show(block=False)

    viewer = ImageViewer(*size, _run)
    viewer.set_title("Random Noise")
    viewer.run()
