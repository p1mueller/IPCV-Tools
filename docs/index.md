# ipcv_tools — Image Processing & Computer Vision Tools

`ipcv_tools` is a small Python library for building **real-time image
processing pipelines** on top of GenICam / USB webcams. Each pipeline is the
simple three-part composition:

```
camera  ──▶  processor  ──▶  viewer
```

…where the processor is *your* processing function (any `callable`), and the
viewer is one of two interchangeable UI layers.

---

## What it does

| Concern            | Choice                                                   | Why                                                                  |
| ------------------ | -------------------------------------------------------- | -------------------------------------------------------------------- |
| Camera source      | GenICam, Webcam (OpenCV), or a mock source               | `Capture` is abstracted, so your code stays source-agnostic          |
| Threading          | One thread per stage, shared bounded buffer              | No back-pressure on acquisition, no blocking between stages          |
| Display            | `ImageViewer` (pygame) or `CameraUI` (PyQt5 + pyqtgraph) | Lightweight embedded display or a full Qt tabbed UI with plots        |
| Controls           | `add_slider`, `add_checkbox`, `add_plot`                 | Wire a Qt slider to `cam.settings(...)` in a single line             |
| Sample data        | `ipcv_tools/data/*.{jpg,tif,bmp,png}`                    | 6 bundled images, perfect for offline development                    |

---

## Documentation table of contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Core concepts](#core-concepts)
- [Building a UI](#building-a-ui)
- [Working with cameras](#working-with-cameras)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

### With `uv` (recommended)

```bash
uv sync            # resolves the dev group (mypy, ruff, pytest, …)
uv pip install -e .
```

### With plain pip

```bash
python3 -m pip install .
```

### System dependencies (Linux)

If you install from source on Linux, the following native libraries must be
available for OpenCV / Qt to load at runtime:

```bash
# Debian / Ubuntu
sudo apt update
sudo apt install -y ffmpeg libsm6 libxext6
```

---

## Quickstart

The `Pipeline` class is the main entry point. Three lines is all it takes:

```python
import ipcv_tools.camera as ipcam
from ipcv_tools.pipeline import Pipeline

def process(frame):
    return frame

pipeline = Pipeline(process, use_ui=True)
pipeline.run()
```

Run on a bundled sample image (no camera needed):

```python
cam = ipcam.DataCam("ahornblatt", colored=True)
pipeline = Pipeline(process, camera=cam, use_ui=True, img_names=["Frame"])
pipeline.run({"decimation": 1, "rel_std": 0.0})
```

Run on a USB webcam:

```python
pipeline = Pipeline(process, port=0, width=640, height=480, use_ui=False)
```

See the [examples](#examples) below for a complete working example.

---

## Core concepts

### `Capture` (abstract)

Every source in `ipcv_tools.camera` is a subclass of `Capture`:

- **`Webcam`** — OpenCV-backed, `cv2.VideoCapture`.
- **`GenICam`** — GenICam/GenTL-backed via `harvesters`.
- **`DataCam`** — Mock source that replays a single image.
- **`NoisyDataCam`** — `DataCam` + synthetic Gaussian noise on every frame.

All four share the same API: `settings(**kwargs)`, `start()`, `stop()`,
`get_next_element()`, `get_shape()`, and context-manager support (`with
camera as cam: …`).

### `Worker` / `Processor`

`Worker` (in `ipcv_tools.processing`) is a threading primitive:

```
acquire_element() ──▶ Buffer ──▶ consumer
```

`Processor(Worker)` is the stage that polls your `process(frame)` callable
and pushes the result into the next stage. It also runs on its own thread
(`Worker._run` → your callable in a loop).

For a full walk-through of the threading model, see `ipcv_tools/processing.py`.

### `Pipeline` (the top-level API)

```python
Pipeline(
    func=process_fn,            # Callable[[ndarray], ndarray | Sequence[ndarray]]
    cti_file=None,              # Optional GenICam .cti
    port=0,                     # Camera port
    width=1280,                 # Optional
    height=720,                 # Optional
    use_ui=True,                # True → CameraUI, False → ImageViewer
    img_names=None,             # Tab labels when use_ui=True
    camera=None,                # Pre-built Capture (e.g. DataCam)
    processor_buf_size=1,
    camera_buf_size=1,
    title="IPCV Viewer",
)
```

Call `pipeline.run(camera_settings: dict | None = None)` to start. Settings
are forwarded to `camera.settings(**kwargs)`.

---

## Building a UI

The `CameraUI` (PyQt5) helper methods:

| Method             | Returns            | Notes                                                 |
| ------------------ | ------------------ | ----------------------------------------------------- |
| `add_slider`       | `Slider`           | Wraps `QtWidgets.QSlider` with float support          |
| `add_checkbox`     | `QtWidgets.QCheckBox` | Standard QCheckBox                                   |
| `add_plot`         | `pg.PlotWidget`    | Add any subclass of `Plotter`, e.g. `HistogramPlotter` |

Example (full version in `examples/example_ui.py`):

```python
from ipcv_tools.ui import CameraUI
from ipcv_tools.plotting import HistogramPlotter

ui = CameraUI(1280, 720, img_names=["Raw", "Processed"], source=source)
ui.add_slider("Gain", 0, 64, 641, 100, 10, lambda v: cam.settings(gain=v), value=5)
ui.add_checkbox("Use Otsu", value=True)
ui.add_plot("Histogram", HistogramPlotter(frame_index=0, grayscale=True))
```

---

## Working with cameras

### GenICam

```python
cam = ipcam.GenICam(port=0, cti_file="path/to/producer.cti")
with cam:
    cam.settings(decimation=2, gain=10, exposure=2e4,
                 pixel_format="Mono8", soft_trigger=True, frame_rate=60)
```

`soft_trigger=True` enables `TriggerMode=On + TriggerSource=Software`.
`soft_trigger=False` enables continuous mode at the given frame rate.

### Webcam

```python
cam = ipcam.Webcam(port=0)   # or "/dev/video0" on Linux
cam.settings(width=1280, height=720)
```

### Mock sources

```python
cam = ipcam.DataCam("ahornblatt", colored=True)
cam = ipcam.NoisyDataCam("ahornblatt", colored=True)
```

`DataCam` exposes a bundled image and accepts `decimation=`; `NoisyDataCam`
additionally accepts `rel_std=` to scale Gaussian noise.

---

## Examples

| File                            | Shows                                                    |
| ------------------------------- | -------------------------------------------------------- |
| `examples/example_data_cam.py`  | `DataCam` + `CameraUI` + sliders + histogram             |
| `examples/example_ui.py`        | GenICam with `CameraUI`, exposure/gain, mono/RGB toggle  |
| `examples/example_viewer.py`    | GenICam + single `ImageViewer` (pygame)                  |
| `examples/example_genicam.py`   | Custom `GenICam` usage with gamma mapping                |
| `examples/utilities.py`         | Shared `ContourResampler` processor used across examples |

Run any of them:

```bash
cd examples
python example_data_cam.py -d 2 -m
```

---

## Contributing

```bash
# Run the test suite
uv run pytest tests --cov=ipcv_tools --cov-report=term-missing

# Lint + type check
uv run ruff check ipcv_tools tests
uv run mypy ipcv_tools --ignore-missing-imports
```

---

## License

MIT — see [`LICENSE`](../LICENSE).
