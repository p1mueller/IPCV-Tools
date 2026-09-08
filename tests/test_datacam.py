"""Tests for the DataCam / NoisyDataCam mock cameras in ipcv_tools.camera."""

import pathlib

import numpy as np
import pytest

from ipcv_tools.camera import DataCam, NoisyDataCam

DATA_FOLDER = pathlib.Path(__file__).parent.parent / "ipcv_tools" / "data"


def _gray_image(shape: tuple = (48, 64)) -> np.ndarray:
    """Create a random grayscale image of the given 2D shape."""
    return np.random.randint(0, 255, shape, dtype=np.uint8)


def test_datacam_from_gray_array_keeps_grayscale():
    """A 2D input without ``colored`` should stay 2D grayscale."""
    img = _gray_image()
    cam = DataCam(img, colored=False)
    assert cam._image.shape == img.shape
    assert cam._image.dtype == np.uint8
    assert cam.get_shape() == img.shape


def test_datacam_colored_converts_gray_to_rgb():
    """A 2D input with ``colored`` should become 3-channel RGB."""
    img = _gray_image()
    cam = DataCam(img, colored=True)
    assert cam._image.ndim == 3
    assert cam._image.shape[2] == 3


def test_datacam_from_rgb_array_converts_to_gray_when_not_colored():
    """A 3D input without ``colored`` should collapse to 2D grayscale."""
    img = np.random.randint(0, 255, (20, 30, 3), dtype=np.uint8)
    cam = DataCam(img, colored=False)
    assert cam._image.ndim == 2


def test_datacam_from_file_path():
    """Loading a bundled sample image should keep its 3-channel color form."""
    path = str(DATA_FOLDER / "lena.bmp")
    cam = DataCam(path, colored=True)
    assert cam._image.ndim == 3
    acquired = cam._acquire_element()
    assert acquired.shape == cam._image.shape


def test_datacam_invalid_path_raises(tmp_path):
    """A non-existent image path should raise a ValueError."""
    with pytest.raises(ValueError):
        DataCam(str(tmp_path / "does_not_exist.png"))


def test_datacam_rejects_wrong_ndim():
    """Images with too few dimensions should be rejected."""
    with pytest.raises(ValueError):
        DataCam(np.zeros((10,), dtype=np.uint8))


def test_datacam_decimation_reduces_resolution():
    """A decimation factor of 2 should halve the frame size."""
    h, w = 100, 120
    img = (np.arange(h * w) % 256).astype(np.uint8).reshape(h, w)
    cam = DataCam(img, colored=False, buffer_size=2)
    cam.settings(decimation=2)
    assert cam.get_shape() == (h // 2, w // 2)


def test_datacam_returns_static_image():
    """Consecutive acquisitions must return the identical frame."""
    img = _gray_image()
    cam = DataCam(img, colored=False, buffer_size=2)
    cam.settings(decimation=1)
    first = cam._acquire_element()
    second = cam._acquire_element()
    assert np.array_equal(first, second)


def test_noisy_datacam_adds_noise():
    """With a non-zero relative std each frame must differ from the base."""
    img = _gray_image((64, 64))
    np.random.seed(0)
    cam = NoisyDataCam(img, colored=False, buffer_size=2)
    cam.settings(decimation=1, rel_std=0.5)
    samples = [cam._acquire_element() for _ in range(20)]
    assert all(not np.array_equal(s, cam._image) for s in samples)


def test_noisy_datacam_zero_rel_std_is_deterministic():
    """Zero relative std must reproduce the undistorted base image."""
    img = _gray_image((32, 32))
    cam = NoisyDataCam(img, colored=False, buffer_size=2)
    cam.settings(decimation=1, rel_std=0.0)
    assert np.array_equal(cam._acquire_element(), cam._image)


def test_noisy_datacam_output_is_uint8():
    """The framed output must always be a uint8 array of the base shape."""
    img = _gray_image()
    np.random.seed(1)
    cam = NoisyDataCam(img, colored=False, buffer_size=2)
    cam.settings(decimation=1, rel_std=0.2)
    out = cam._acquire_element()
    assert out.dtype == np.uint8
    assert out.shape == img.shape
