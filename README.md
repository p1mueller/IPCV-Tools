# ipcv_tools — Image Processing & Computer Vision Tools

Utilities for building image-processing pipelines on top of GenICam / USB
webcams, with two interchangeable display layers:

- **`CameraUI`** (PyQt5 + pyqtgraph): tabbed viewer with live FPS counter,
  sliders / checkboxes for tuning, and attached plot widgets (e.g. histograms).
- **`ImageViewer`** (pygame): lightweight single-frame display, ideal for
  embedded / minimal deployments.

A single `Pipeline` wires together the _camera → processor → viewer_ thread
chain in a few lines of Python.

## Features

- **Abstract camera layer** — `GenICam`, `Webcam` (OpenCV), and two mock
  sources (`DataCam` / `NoisyDataCam`) share a common `Capture` interface.
- **Threaded pipeline** — acquisition, processing, and display each run on
  their own thread with a shared bounded buffer; no back-pressure on the
  camera.
- **Live UI** — sliders, checkboxes, and histogram plots are wired to a
  callback API; each `CameraUI` tab can hold a different output frame.
- **GenICam soft-trigger** — supports software-triggered acquisition as well
  as continuous mode with configurable frame-rate.
- **Bundled sample images** — `ipcv_tools/data/` ships a handful of
  grayscale/color test images (`ahornblatt.jpg`, `dog.jpg`, `lions.jpg`, …)
  for offline development.

## Installation

The project targets Python 3.9 – 3.11 and is managed with [uv](https://github.com/astral-sh/uv),
but a plain pip install also works.

### With `uv` (recommended)

```bash
uv sync            # dev group (tests, lint, mypy) is included
uv pip install -e .
```

### With pip

```bash
python3 -m pip install .
```

## Project layout

```
ipcv_tools/
├── camera.py        # Capture / GenICam / Webcam / DataCam / NoisyDataCam
├── controls.py      # Floating-point Slider widget (PyQt5)
├── pipeline.py      # Pipeline: acquisition + processing + display
├── plotting.py      # Plotter / HistogramPlotter (pyqtgraph)
├── processing.py    # Worker / Processor threaded helpers
├── ui.py            # CameraUI, VideoThread
├── utilities.py     # Buffer, FPS, cv2/Qt env fixups
├── viewer.py        # ImageViewer (pygame)
├── data/            # Bundled sample images
├── examples/        # Runnable demo scripts
└── tests/           # pytest suite (~90 tests, 92 % coverage)
```

### Dependency graph & class diagram

![packages](https://gitlab.ost.ch/patrik.mueller/ipcv-tools/-/jobs/artifacts/master/raw/packages.png?job=dependencies)

![classes](https://gitlab.ost.ch/patrik.mueller/ipcv-tools/-/jobs/artifacts/master/raw/classes.png?job=dependencies)

## Quick start

The simplest way to run a pipeline on a **GenICam** camera with the PyQt5 UI:

```python
import ipcv_tools.camera as ipcam
from ipcv_tools.pipeline import Pipeline

def process(frame):
    # frame: numpy.ndarray  (BGR by default for Webcams, RGB for GenICam)
    return frame

pipeline = Pipeline(process, use_ui=True)
pipeline.run()
```

### On a USB webcam

```python
pipeline = Pipeline(process, port=0, width=640, height=480, use_ui=False)
```

### On one of the bundled sample images (no camera required)

```python
import ipcv_tools.camera as ipcam
from ipcv_tools.pipeline import Pipeline

cam = ipcam.DataCam("ahornblatt", colored=True)

def process(frame):
    return frame

pipeline = Pipeline(
    process,
    camera=cam,
    use_ui=True,
    img_names=["Frame"],
)
pipeline.run({"decimation": 1})
```

### Tunable parameters

`Pipeline.run(camera_settings: dict)` forwards settings to the camera:

```python
pipeline.run({
    "decimation": 2,
    "gain": 10.0,
    "exposure": 2e4,
    "frame_rate": 30.0,
    "soft_trigger": True,
    "pixel_format": "RGB8",
})
```

Unknown keys are ignored by `Webcam` and pass through via `**kwargs` for
vendor-specific extensions.

## Building a UI

`CameraUI` exposes three helper methods for wiring controls and plots.
See `examples/example_ui.py` for a complete example.

```python
ui = CameraUI(width=1280, height=720, img_names=["Raw", "Processed"], source=source)

# Slider (float values via steps)
ui.add_slider("Gain", 0., 64., steps=641, value=6., func=lambda v: cam.settings(gain=v))

# Checkbox
ui.add_checkbox("Apply CLAHE", value=True)

# Plot (e.g. live histogram)
from ipcv_tools.plotting import HistogramPlotter
ui.add_plot("Histogram", HistogramPlotter(frame_index=0, grayscale=False))
```

## Examples

| File                           | Shows                                        |
| ------------------------------ | -------------------------------------------- |
| `examples/example_data_cam.py` | `DataCam` + `CameraUI` + sliders + histogram |
| `examples/example_ui.py`       | Real GenICam with `CameraUI`, exposure/gain  |
| `examples/example_viewer.py`   | Real GenICam + single `ImageViewer`          |
| `examples/example_genicam.py`  | Custom `GenICam` usage with gamma map        |
| `examples/utilities.py`        | Shared `ContourResampler` demo processor     |

Run any of them from the repo root:

```bash
cd examples
python example_data_cam.py -d 2 -m        # 2× decimation, monochrome, no noise
python example_ui.py      -g 12 -e 3e4
```

## Running the test suite

Tests are run with `pytest` and `pytest-cov`. Coverage is configured via
`[tool.coverage]` in `pyproject.toml`.

```bash
uv run pytest tests --cov=ipcv_tools --cov-report=term-missing
```

The CI pipeline (`.github/workflows/ci.yml`, `.gitlab-ci.yml`) runs lint
(`ruff`), type checking (`mypy`), the test suite with coverage, and publishes
artifacts.

## License

Licensed under the MIT License — see [`LICENSE`](LICENSE).
