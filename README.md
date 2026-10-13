# Digital Kinect: Live Webcam Depth Heat Map

A Python app that turns an ordinary webcam into a "digital Kinect". It estimates how near or far everything in the picture is and shows it live as a colour heat map.

- **Red / yellow** = close to the camera
- **Green** = medium distance
- **Blue / purple** = far from the camera

It runs on an **integrated GPU** (Intel, AMD or any DirectX 12 GPU on Windows) using ONNX Runtime with DirectML. No NVIDIA card and no PyTorch needed.

> The depth is **relative**: it shows what is nearer or farther than other things in the same image, not exact distances in metres.

## Features

- Live depth heat map from a single webcam
- Runs on an integrated GPU (DirectML / OpenVINO), with automatic CPU fallback
- Smooth video: capture, depth estimation and display run in separate threads
- Four colour maps and three view modes
- Mouse probe that shows closeness (0 to 100%) at any point
- Screenshot saving

## Requirements

- Windows 10 or 11 (for GPU acceleration; Linux and macOS run on CPU)
- Python 3.9 or newer
- A webcam
- Internet on the first run only (to download the ~67 MB model)

## Installation

```
pip uninstall onnxruntime onnxruntime-directml -y
pip install opencv-python numpy onnxruntime-directml
```

The `uninstall` line removes old ONNX Runtime packages that can conflict. If it says they are not installed, ignore that and continue.

If `pip` is not recognised, use `python -m pip` instead of `pip`.

## Run

Go to the folder containing the script, then run it:

```
cd %USERPROFILE%\Downloads
python webcam_depth_igpu.py
```

(In PowerShell, use `cd $env:USERPROFILE\Downloads`.)

On the first run the script downloads `model-small.onnx` into the same folder. Later runs start straight away.

Check the status line at the top of the window. If it shows `Dml`, your GPU is being used. If it shows `CPU`, the GPU package did not install correctly.

### Optional arguments

| Argument   | Default | Meaning                                                  |
| ---------- | ------- | -------------------------------------------------------- |
| `--camera` | 0       | Webcam index. Try `1` if the default camera does not open |
| `--width`  | 640     | Capture width in pixels                                   |
| `--height` | 480     | Capture height in pixels                                  |

Example: `python webcam_depth_igpu.py --camera 1`

## Controls

| Input       | Action                                                     |
| ----------- | ---------------------------------------------------------- |
| `c`         | Cycle colour maps (JET, TURBO, INFERNO, HOT)               |
| `v`         | Cycle views (side-by-side, heat map only, overlay)         |
| `f`         | Toggle mirror on or off                                    |
| `s`         | Save a screenshot as `depth_<timestamp>.png`               |
| `q` / `Esc` | Quit                                                       |
| Mouse hover | Show the closeness percentage at the pointer               |

## How it works

1. A camera thread continuously grabs the latest webcam frame.
2. A depth thread resizes the frame to 256 x 256, normalises it, and runs the **MiDaS-small** model through ONNX Runtime on the GPU.
3. The depth output is smoothed over time and scaled to a stable 0 to 1 closeness range.
4. The main thread colours the result with an OpenCV colour map and draws it with a legend and status line.

Because inference runs in its own thread, the video stays smooth even if depth estimation is slower than the camera. The heat map may lag the video by a fraction of a second.

## Troubleshooting

| Problem | Fix |
| ------- | --- |
| Status line says `CPU` | Run `pip uninstall onnxruntime onnxruntime-directml -y`, then `pip install onnxruntime-directml` again. Install only one ONNX Runtime package |
| Webcam does not open | Close other apps using the camera, or run with `--camera 1` |
| Model download fails | Download `model-small.onnx` from https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx and save it next to the script |
| Low frame rate | Lower `SIZE` (for example to `192`) near the top of the script, or use a lower `--width` / `--height` |
| `ModuleNotFoundError` | Re-run the `pip install` line from the Installation section |

## Limitations

- Depth is relative, not real-world distance.
- Accuracy drops on glass, mirrors, shiny surfaces and plain featureless walls.
- Colours are normalised per scene, so they can shift slightly when objects enter or leave the frame.
- Speed depends on your GPU, drivers and camera.

## Project files

| File                     | Purpose                                                   |
| ------------------------ | --------------------------------------------------------- |
| `webcam_depth_igpu.py`   | The application                                           |
| `model-small.onnx`       | MiDaS-small model (downloaded automatically on first run) |
| `depth_<timestamp>.png`  | Screenshots you save with the `s` key                     |

## Credits

- Depth model: [MiDaS](https://github.com/isl-org/MiDaS) by Intel ISL (Ranftl et al., 2020)
- [ONNX Runtime](https://onnxruntime.ai) for GPU inference
- [OpenCV](https://opencv.org) for camera capture and display
