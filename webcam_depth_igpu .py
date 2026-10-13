"""
Live "Digital Kinect" for INTEGRATED GPUs (Intel / AMD / any DirectX 12 GPU on Windows).

Depth is estimated from a normal webcam with the MiDaS-small model, run through
ONNX Runtime so it uses your iGPU instead of the CPU.

    RED / YELLOW = close        BLUE / PURPLE = far

-------------------------------------------------------------------------------
SETUP (one time)

  Windows (Intel / AMD / NVIDIA, via DirectML):
      pip install opencv-python numpy onnxruntime-directml

  Intel iGPU alternative (often fastest on Intel):
      pip install opencv-python numpy onnxruntime-openvino

  (Install only ONE onnxruntime variant. If you installed plain `onnxruntime`
   before, run `pip uninstall onnxruntime` first.)

  Run:   python webcam_depth_igpu.py
  The first run downloads the official MiDaS-small ONNX model (~67 MB).
  No PyTorch needed.
-------------------------------------------------------------------------------
Keys:  q/ESC quit | c colour map | v view mode | f mirror | s screenshot
Mouse: hover to read closeness (0-100%) at that point.
"""

import argparse
import os
import threading
import time
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "model-small.onnx"
MODEL_URL = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
SIZE = 256  # network input resolution (256 = fast on iGPU; try 192 for more FPS)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

COLORMAPS = [
    ("JET", cv2.COLORMAP_JET),
    ("TURBO", cv2.COLORMAP_TURBO),
    ("INFERNO", cv2.COLORMAP_INFERNO),
    ("HOT", cv2.COLORMAP_HOT),
]
VIEWS = ["side-by-side", "heatmap", "overlay"]


# ----------------------------------------------------------------------------
# Model download (runs once)
# ----------------------------------------------------------------------------
def download_model(path: str, url: str):
    print("Downloading MiDaS-small ONNX model (~67 MB, one-time)...")
    tmp = path + ".part"

    def progress(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            print(f"\r  {pct}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, tmp, progress)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise SystemExit(
            f"\nDownload failed: {e}\n"
            f"Download this file manually in your browser:\n  {url}\n"
            f"and save it next to this script as '{path}'."
        )
    os.replace(tmp, path)
    print("\nSaved", path)


# ----------------------------------------------------------------------------
# ONNX Runtime session on the best available accelerator
# ----------------------------------------------------------------------------
def make_session(path: str):
    available = ort.get_available_providers()
    print("Available providers:", available)

    preferred = []
    if "OpenVINOExecutionProvider" in available:
        preferred.append(("OpenVINOExecutionProvider", {"device_type": "GPU"}))
    if "DmlExecutionProvider" in available:
        preferred.append("DmlExecutionProvider")
    if "CoreMLExecutionProvider" in available:
        preferred.append("CoreMLExecutionProvider")
    preferred.append("CPUExecutionProvider")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if "DmlExecutionProvider" in available:
        # DirectML requirements
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    sess = ort.InferenceSession(path, opts, providers=preferred)
    used = sess.get_providers()[0]
    print("Running on:", used)
    if used == "CPUExecutionProvider":
        print("WARNING: no GPU provider found - running on CPU. "
              "Install onnxruntime-directml (Windows) or onnxruntime-openvino (Intel).")
    return sess, used


def preprocess(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])  # 1x3xHxW


# ----------------------------------------------------------------------------
# Threads: capture + depth inference run independently so the display stays live
# ----------------------------------------------------------------------------
class Camera(threading.Thread):
    def __init__(self, index, w, h):
        super().__init__(daemon=True)
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise SystemExit("Could not open webcam. Try --camera 1")
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.join(timeout=2)
        self.cap.release()


class DepthWorker(threading.Thread):
    def __init__(self, session, camera, mirror_getter):
        super().__init__(daemon=True)
        self.sess = session
        self.cam = camera
        self.get_mirror = mirror_getter
        self.input_name = session.get_inputs()[0].name
        self.depth = None      # latest normalised closeness map (SIZE x SIZE, 0..1)
        self.ms = 0.0          # inference time
        self.lock = threading.Lock()
        self.running = True
        self._lo = self._hi = None
        self._prev = None

    def run(self):
        while self.running:
            frame = self.cam.read()
            if frame is None:
                time.sleep(0.005)
                continue
            if self.get_mirror():
                frame = cv2.flip(frame, 1)

            t0 = time.time()
            out = self.sess.run(None, {self.input_name: preprocess(frame)})[0]
            out = np.squeeze(out).astype(np.float32)  # inverse depth: bigger = closer
            ms = (time.time() - t0) * 1000

            # temporal smoothing (less flicker)
            self._prev = out if self._prev is None else 0.6 * self._prev + 0.4 * out
            sm = self._prev

            # smoothed normalisation range so colours stay stable
            lo, hi = np.percentile(sm, 2), np.percentile(sm, 98)
            self._lo = lo if self._lo is None else 0.9 * self._lo + 0.1 * lo
            self._hi = hi if self._hi is None else 0.9 * self._hi + 0.1 * hi
            closeness = np.clip((sm - self._lo) / max(self._hi - self._lo, 1e-6), 0, 1)

            with self.lock:
                self.depth = closeness
                self.ms = ms

    def get(self):
        with self.lock:
            return self.depth, self.ms


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    if not os.path.exists(MODEL_PATH):
        download_model(MODEL_PATH, MODEL_URL)

    session, provider = make_session(MODEL_PATH)

    state = {"mirror": True}
    cam = Camera(args.camera, args.width, args.height)
    cam.start()
    worker = DepthWorker(session, cam, lambda: state["mirror"])
    worker.start()

    cmap_idx, view_idx = 0, 0
    mouse = {"x": -1, "y": -1}
    fps, last = 0.0, time.time()

    win = "Digital Kinect (iGPU) - live depth"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, lambda e, x, y, f, p: mouse.update(x=x, y=y))

    while True:
        frame = cam.read()
        depth, ms = worker.get()
        if frame is None or depth is None:
            if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                break
            continue

        if state["mirror"]:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        closeness = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        gray = (closeness * 255).astype(np.uint8)  # 255 = nearest
        name, cmap = COLORMAPS[cmap_idx]
        heat = cv2.applyColorMap(gray, cmap)

        view = VIEWS[view_idx]
        if view == "side-by-side":
            out = np.hstack([frame, heat])
        elif view == "heatmap":
            out = heat
        else:
            out = cv2.addWeighted(frame, 0.45, heat, 0.55, 0)

        # legend (far -> near)
        bar_h = 160
        bar = np.linspace(255, 0, bar_h).astype(np.uint8).reshape(-1, 1)
        bar = cv2.applyColorMap(np.repeat(bar, 18, axis=1), cmap)
        x0, y0 = out.shape[1] - 40, 40
        out[y0:y0 + bar_h, x0:x0 + 18] = bar
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out, "NEAR", (x0 - 14, y0 - 8), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, "FAR", (x0 - 4, y0 + bar_h + 16), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # status line
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        status = f"display {fps:4.0f} fps | depth {ms:4.0f} ms ({1000 / max(ms, 1):.0f} fps) | {provider.replace('ExecutionProvider', '')} | {name}"
        cv2.putText(out, status, (10, 22), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, status, (10, 22), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # mouse probe
        mx, my = mouse["x"], mouse["y"]
        px = mx - w if (view == "side-by-side" and mx >= w) else mx
        if 0 <= px < w and 0 <= my < h:
            label = f"closeness: {closeness[my, px] * 100:.0f}%"
            cv2.circle(out, (mx, my), 5, (255, 255, 255), 2)
            cv2.putText(out, label, (mx + 10, my - 10), font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label, (mx + 10, my - 10), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win, out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("c"):
            cmap_idx = (cmap_idx + 1) % len(COLORMAPS)
        elif key == ord("v"):
            view_idx = (view_idx + 1) % len(VIEWS)
        elif key == ord("f"):
            state["mirror"] = not state["mirror"]
        elif key == ord("s"):
            fn = f"depth_{int(time.time())}.png"
            cv2.imwrite(fn, out)
            print("Saved", fn)

    worker.running = False
    worker.join(timeout=2)
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
