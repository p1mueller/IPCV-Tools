# Getting started with `ipcv_tools`

This is a 5-minute tour of the library. By the end you will have

1. installed the package,
2. run a pipeline on a bundled image,
3. wired up a slider, checkbox, and plot.

---

## 1. Install

### Prerequisites

- Python **3.9 – 3.11**
- Linux: system packages `ffmpeg`, `libsm6`, `libxext6`

```bash
sudo apt update
sudo apt install -y ffmpeg libsm6 libxext6
```

### Install the package

Either of these works:

```bash
uv sync
uv pip install -e .
```

…or

```bash
python3 -m pip install .
```

---

## 2. First pipeline (no camera needed)

The simplest pipeline processes one of the bundled images and shows it in the
`ImageViewer` (pygame). Open a REPL in the project root and run:

```python
import ipcv_tools.camera as ipcam
from ipcv_tools.pipeline import Pipeline

def process(frame):
    # Return the frame unchanged.
    return frame

cam = ipcam.DataCam("ahornblatt", colored=True)
pipeline = Pipeline(
    process,
    camera=cam,
    width=cam.get_shape()[1],
    height=cam.get_shape()[0],
    use_ui=False,
)
pipeline.run({"decimation": 1, "rel_std": 0.0})
```

A pygame window should open, and `Esc` / closing it quits.

If you prefer PyQt, swap `use_ui=False` → `use_ui=True` and `img_names=["Frame"]`:

```python
pipeline = Pipeline(
    process,
    camera=cam,
    width=cam.get_shape()[1],
    height=cam.get_shape()[0],
    use_ui=True,
    img_names=["Frame"],
)
pipeline.run({"decimation": 1, "rel_std": 0.0})
```

`Pipeline.run(camera_settings)` forwards every entry of the dict to
`camera.settings(**kwargs)`. Keys like `"decimation"` and `"rel_std"` are
recognised by `DataCam` / `NoisyDataCam`; other keys are silently ignored
unless a camera's `settings` accepts them.

---

## 3. Real camera

### USB webcam

```python
pipeline = Pipeline(
    process,
    port=0,                       # or "/dev/video0" on Linux
    width=640,
    height=480,
    use_ui=True,
    img_names=["Frame"],
)
pipeline.run()
```

### GenICam

```python
pipeline = Pipeline(
    process,
    cti_file="path/to/producer.cti",   # optional; discovered if missing
    port=0,
    use_ui=True,
    img_names=["Frame"],
)
pipeline.run({
    "decimation": 1,
    "gain": 5.0,
    "exposure": 5e3,
    "frame_rate": 30.0,
    "soft_trigger": False,
    "pixel_format": "RGB8",
})
```

---

## 4. Adding live controls

`CameraUI` exposes three helpers that return the underlying Qt widget, so
you can wire them up like any other Qt component:

```python
from ipcv_tools.ui import CameraUI
from ipcv_tools.plotting import HistogramPlotter

ui = CameraUI(width=1280, height=720, img_names=["Raw", "Processed"], source=source)

# Slider
slider = ui.add_slider(
    "Sigma", 1.0, 10.0, steps=101,
    value=3.0,
    func=lambda v: cam.settings(gain=v),
)

# Checkbox
ui.add_checkbox("Use Otsu", value=True)

# Plot
ui.add_plot("Histogram", HistogramPlotter(frame_index=0, grayscale=False))
```

Each slider is an instance of `ipcv_tools.controls.Slider` (a `QtWidgets`
wrapper around `QSlider` that emits a `float` on change).

---

## 5. Next steps

- Walk through the full demo (`examples/example_data_cam.py`) with live
  sliders + histogram.
- Read `ipcv_tools/pipeline.py` for the threading model.
- Run the test suite (`uv run pytest tests --cov`).

## 6. Troubleshooting

| Symptom                                            | Fix                                                                  |
| -------------------------------------------------- | -------------------------------------------------------------------- |
| `pygame.error: video system not initialized`        | `sudo apt install -y libsm6 libxext6` and/or use `QT_QPA_PLATFORM=offscreen` |
| `harvesters._harvester.GenICam` errors               | Make sure `GENICAM_GENTL64_PATH` is set and points at a valid CTI file |
| No UI window                                       | Set `QT_QPA_PLATFORM=offscreen` and check your `DISPLAY` env var      |
| `cv2.error: Could not load plugin`                  | Run `sudo apt install -y libxext6` and reinstall `opencv-python`      |
