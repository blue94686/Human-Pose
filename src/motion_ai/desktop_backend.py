from __future__ import annotations

import gc
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from .analyzer import ActionAnalyzer
from .config import DEFAULT_FRAME_WIDTH, DEFAULT_TEMPLATE_FILE, DEFAULT_WEIGHTS, OUTPUTS_DIR


@contextmanager
def _silence_native_stderr():
    """临时静默 OpenCV 原生层 stderr，避免摄像头探测时刷屏。"""
    if os.name != "nt":
        yield
        return
    try:
        stderr_fd = sys.stderr.fileno()
    except Exception:
        yield
        return
    saved_fd = None
    null_fd = None
    try:
        saved_fd = os.dup(stderr_fd)
        null_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(null_fd, stderr_fd)
        yield
    finally:
        if saved_fd is not None:
            os.dup2(saved_fd, stderr_fd)
            os.close(saved_fd)
        if null_fd is not None:
            os.close(null_fd)


def resize_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= DEFAULT_FRAME_WIDTH:
        return frame
    ratio = DEFAULT_FRAME_WIDTH / float(width)
    return cv2.resize(frame, (DEFAULT_FRAME_WIDTH, int(height * ratio)), interpolation=cv2.INTER_AREA)


def open_camera(index: int):
    if os.name == "nt":
        backend_candidates = [
            getattr(cv2, "CAP_DSHOW", None),
            getattr(cv2, "CAP_MSMF", None),
            None,
        ]
    else:
        backend_candidates = [
            getattr(cv2, "CAP_AVFOUNDATION", None),
            getattr(cv2, "CAP_V4L2", None),
            None,
        ]

    backends: list[int | None] = []
    for backend in backend_candidates:
        if backend in backends:
            continue
        backends.append(backend)

    for backend in backends:
        with _silence_native_stderr():
            capture = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
        if capture is not None and capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            capture.set(cv2.CAP_PROP_FPS, 20)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return capture
        release_capture(capture)
    with _silence_native_stderr():
        return cv2.VideoCapture(index)


def release_capture(capture) -> None:
    if capture is not None:
        try:
            if capture.isOpened():
                capture.release()
        except Exception:
            pass
    gc.collect()


def scan_cameras(max_index: int = 8) -> list[dict[str, Any]]:
    cameras: list[dict[str, Any]] = []
    for index in range(max_index + 1):
        capture = None
        try:
            capture = open_camera(index)
            if capture is None or not capture.isOpened():
                continue
            readable = False
            width = 0
            height = 0
            for _ in range(10):
                ok, frame = capture.read()
                if ok and frame is not None and frame.size > 0:
                    readable = True
                    height, width = frame.shape[:2]
                    break
            cameras.append(
                {
                    "index": index,
                    "name": f"摄像头 {index}",
                    "readable": readable,
                    "width": int(width),
                    "height": int(height),
                }
            )
            # 只要当前编号已经无法打开，后续更高编号通常也无设备，避免无意义探测。
            if not readable and index > 0:
                break
        finally:
            release_capture(capture)
    return cameras


def print_camera_scan(max_index: int = 8) -> int:
    for item in scan_cameras(max_index=max_index):
        print(f"{item['index']}\t{1 if item['readable'] else 0}\t{item['name']}")
    return 0


def capture_camera_frame(index: int) -> np.ndarray:
    capture = open_camera(index)
    if capture is None or not capture.isOpened():
        raise RuntimeError(f"无法打开摄像头 {index}")
    frame = None
    try:
        for _ in range(8):
            ok, frame = capture.read()
            if ok and frame is not None and frame.size > 0:
                break
        if frame is None:
            raise RuntimeError(f"摄像头 {index} 未返回有效画面")
        return resize_frame(frame)
    finally:
        release_capture(capture)


def create_video_thumbnail(video_path: str | Path, output_path: str | Path) -> int:
    source_path = Path(video_path).expanduser().resolve()
    target_path = Path(output_path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source_path))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"无法读取视频首帧：{source_path}")
    cv2.imwrite(str(target_path), resize_frame(frame))
    return 0


class VideoAnalyzeWorker(QThread):
    progress_changed = Signal(dict)
    finished_success = Signal(object)
    finished_error = Signal(str)

    def __init__(
        self,
        *,
        source: str,
        description_text: str,
        weights_path: str,
        template_file: str,
        output_dir: Path,
        frame_stride: int,
        max_frames: int | None,
        analysis_mode: str,
        baseline_payload: dict | None,
        evaluator_level: str,
    ) -> None:
        super().__init__()
        self.source = source
        self.description_text = description_text
        self.weights_path = weights_path
        self.template_file = template_file
        self.output_dir = output_dir
        self.frame_stride = frame_stride
        self.max_frames = max_frames
        self.analysis_mode = analysis_mode
        self.baseline_payload = baseline_payload
        self.evaluator_level = evaluator_level
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            analyzer = ActionAnalyzer(weights_path=self.weights_path, template_file=self.template_file)
            result = analyzer.analyze_video(
                source=self.source,
                description_text=self.description_text,
                output_dir=self.output_dir,
                frame_stride=max(1, self.frame_stride),
                max_frames=self.max_frames,
                frame_callback=self._emit_progress,
                should_stop=lambda: self._stop_requested,
                analysis_mode=self.analysis_mode,
                baseline_payload=self.baseline_payload,
                evaluator_level=self.evaluator_level,
            )
            self.finished_success.emit(result)
        except Exception as exc:
            self.finished_error.emit(str(exc))

    def _emit_progress(self, frame: np.ndarray, status: dict[str, Any]) -> None:
        self.progress_changed.emit({"frame": frame, "status": status})


class CameraScanWorker(QThread):
    finished_success = Signal(list)
    finished_error = Signal(str)

    def __init__(self, max_index: int = 8) -> None:
        super().__init__()
        self.max_index = max_index

    def run(self) -> None:
        try:
            self.finished_success.emit(scan_cameras(max_index=self.max_index))
        except Exception as exc:
            self.finished_error.emit(str(exc))


class RealtimeCameraWorker(QThread):
    progress_changed = Signal(dict)
    finished_success = Signal(str)
    finished_error = Signal(str)

    def __init__(
        self,
        *,
        camera_index: int,
        description_text: str,
        weights_path: str,
        template_file: str,
        evaluator_level: str,
        target_fps: float = 20.0,
    ) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.description_text = description_text
        self.weights_path = weights_path
        self.template_file = template_file
        self.evaluator_level = evaluator_level
        self.target_fps = max(15.0, min(25.0, float(target_fps)))
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        capture = None
        try:
            capture = open_camera(self.camera_index)
            if capture is None or not capture.isOpened():
                raise RuntimeError(f"无法打开摄像头 {self.camera_index}")

            warm_frame = None
            for _ in range(12):
                ok, frame = capture.read()
                if ok and frame is not None and frame.size > 0:
                    warm_frame = resize_frame(frame)
                    break
                self.msleep(40)
            if warm_frame is None:
                raise RuntimeError(f"摄像头 {self.camera_index} 已打开，但未返回有效画面")

            analyzer = ActionAnalyzer(weights_path=self.weights_path, template_file=self.template_file)
            session = analyzer.create_realtime_session(
                self.description_text,
                evaluator_level=self.evaluator_level,
            )
            frame_interval_ms = int(1000.0 / self.target_fps)
            frame = warm_frame

            while not self._stop_requested:
                overlay, status = analyzer.analyze_realtime_frame(frame, session)
                status["device_index"] = self.camera_index
                status["device_state"] = "运行中"
                status["target_fps"] = self.target_fps
                self.progress_changed.emit({"frame": overlay, "status": status})
                self.msleep(max(1, frame_interval_ms))
                ok, next_frame = capture.read()
                if not ok or next_frame is None or next_frame.size == 0:
                    continue
                frame = resize_frame(next_frame)

            self.finished_success.emit(f"摄像头 {self.camera_index} 已停止并释放")
        except Exception as exc:
            self.finished_error.emit(str(exc))
        finally:
            release_capture(capture)


def build_output_dir(prefix: str) -> Path:
    target = OUTPUTS_DIR / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_baseline_payload(path_text: str) -> dict | None:
    text = (path_text or "").strip()
    if not text:
        return None
    baseline_path = Path(text).expanduser().resolve()
    if not baseline_path.exists():
        raise RuntimeError(f"模板基线不存在：{baseline_path}")
    return json.loads(baseline_path.read_text(encoding="utf-8"))


__all__ = [
    "CameraScanWorker",
    "RealtimeCameraWorker",
    "VideoAnalyzeWorker",
    "build_output_dir",
    "capture_camera_frame",
    "create_video_thumbnail",
    "load_baseline_payload",
    "open_camera",
    "print_camera_scan",
    "release_capture",
    "resize_frame",
    "scan_cameras",
]
