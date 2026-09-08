"""Tests for the plotting utilities in ipcv_tools.plotting."""

import numpy as np
import pyqtgraph as pg
import pytest

from ipcv_tools.plotting import HistogramPlotter, Plotter, compute_histogram


def _rgb_img(shape=(16, 16)):
    """Create a random 3-channel uint8 image."""
    return np.random.randint(0, 255, shape + (3,), dtype=np.uint8)


def _gray_img(shape=(16, 16)):
    """Create a random 2D uint8 image."""
    return np.random.randint(0, 255, shape, dtype=np.uint8)


def test_compute_histogram_rgb_returns_three_channels():
    """RGB image without grayscale flag -> three channels (r, g, b)."""
    img = _rgb_img()
    x, hists = compute_histogram(img, grayscale=False)
    assert x.shape == (256,)
    assert hists.shape == (3, 256)
    # Histograms are normalized: each channel sums to 1.
    for i in range(3):
        assert hists[i].sum() == pytest.approx(1.0)


def test_compute_histogram_grayscale_flag_collapses_channels():
    """grayscale=True on an RGB image should return a single aggregated histogram."""
    img = _rgb_img()
    x, hists = compute_histogram(img, grayscale=True)
    assert hists.shape[0] == 1


def test_compute_histogram_gray_2d_returns_single_channel():
    """2D grayscale image -> single channel histogram."""
    img = _gray_img()
    x, hists = compute_histogram(img, grayscale=False)
    assert hists.shape[0] == 1


def test_histogram_plotter_init_graph_rgb():
    """RGB image: the plotter should register three lines (r/g/b)."""
    plotter = HistogramPlotter(frame_index=0, grayscale=False)
    graph = pg.PlotWidget()
    plotter.set_graph(graph, _rgb_img())
    assert len(plotter.lines) == 3


def test_histogram_plotter_update_grayscale():
    """update() on a grayscale (2D) frame should not raise and should refresh lines."""
    plotter = HistogramPlotter(frame_index=0, grayscale=True)
    graph = pg.PlotWidget()
    plotter.set_graph(graph, _gray_img())
    plotter.update([_gray_img()])
    assert len(plotter.lines) == 1


def test_histogram_plotter_update_rgb():
    """update() on an RGB frame should refresh three lines."""
    plotter = HistogramPlotter(frame_index=0, grayscale=False)
    graph = pg.PlotWidget()
    plotter.set_graph(graph, _rgb_img())
    plotter.update([_rgb_img()])
    assert len(plotter.lines) == 3


def test_plotter_abc_is_not_instantiable():
    """Plotter is an abstract base and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Plotter()


def test_histogram_plotter_ensure_dim_promotes_2d_when_not_grayscale():
    """With grayscale=False and 2D input _ensure_dim should promote to 3D RGB."""
    plotter = HistogramPlotter(grayscale=False)
    promoted = plotter._ensure_dim(_gray_img())
    assert promoted.ndim == 3 and promoted.shape[-1] == 3


def test_histogram_plotter_ensure_dim_collapses_3d_when_grayscale():
    """With grayscale=True and 3D input _ensure_dim should collapse to 2D."""
    plotter = HistogramPlotter(grayscale=True)
    collapsed = plotter._ensure_dim(_rgb_img())
    assert collapsed.ndim == 2
