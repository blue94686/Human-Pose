from __future__ import annotations

import os
import sys
from pathlib import Path


def _frozen_base_dir() -> Path | None:
    if getattr(sys, "frozen", False):
        executable = Path(sys.executable).resolve()
        return executable.parent
    return None


def _source_base_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _resource_root() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")).resolve()
    frozen_dir = _frozen_base_dir()
    if frozen_dir is not None:
        return frozen_dir
    return _source_base_dir()


def _runtime_root() -> Path:
    env_root = os.environ.get("MOTION_AI_HOME", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    frozen_dir = _frozen_base_dir()
    if frozen_dir is not None:
        return frozen_dir
    return _source_base_dir()


PROJECT_ROOT = _runtime_root()
RESOURCE_ROOT = _resource_root()
TEMPLATES_DIR = PROJECT_ROOT / "templates"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
FONTS_DIR = PROJECT_ROOT / "fonts"
RESOURCE_TEMPLATES_DIR = RESOURCE_ROOT / "templates"
RESOURCE_WEIGHTS_DIR = RESOURCE_ROOT / "weights"
RESOURCE_FONTS_DIR = RESOURCE_ROOT / "fonts"

YOLO_POSE_MODELS = [
    {
        "name": "yolov8n-pose.pt",
        "label": "YOLOv8n-Pose | 640x640 | 最快",
        "imgsz": 640,
        "speed": "最快（约 1.18ms）",
        "profile": "推理速度最快，适合普通电脑实时摄像头分析与边缘端部署。",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt",
    },
    {
        "name": "yolov8s-pose.pt",
        "label": "YOLOv8s-Pose | 640x640 | 较快",
        "imgsz": 640,
        "speed": "较快",
        "profile": "精度与速度均衡，适合常规视频与摄像头分析。",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-pose.pt",
    },
    {
        "name": "yolov8m-pose.pt",
        "label": "YOLOv8m-Pose | 640x640 | 中等",
        "imgsz": 640,
        "speed": "中等",
        "profile": "精度更高，对算力有一定要求，适合离线分析或较强设备。",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-pose.pt",
    },
    {
        "name": "yolov8l-pose.pt",
        "label": "YOLOv8l-Pose | 640x640 | 较慢",
        "imgsz": 640,
        "speed": "较慢",
        "profile": "适合追求更高精度的离线分析场景，尤其是关注动作细节时。",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-pose.pt",
    },
    {
        "name": "yolov8x-pose.pt",
        "label": "YOLOv8x-Pose | 640x640 | 慢",
        "imgsz": 640,
        "speed": "慢",
        "profile": "640 输入下精度极高，适合对精度要求严格的离线分析。",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-pose.pt",
    },
    {
        "name": "yolov8x-pose-p6.pt",
        "label": "YOLOv8x-Pose-P6 | 1280x1280 | 最慢",
        "imgsz": 1280,
        "speed": "最慢（约 10.04ms）",
        "profile": "精度最高，可处理更多细节，适合高精度离线分析。",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-pose-p6.pt",
    },
]
YOLO_POSE_MODEL_BY_NAME = {item["name"]: item for item in YOLO_POSE_MODELS}

WEIGHTS_CANDIDATES: list[Path] = []
for model in YOLO_POSE_MODELS:
    model_name = model["name"]
    WEIGHTS_CANDIDATES.extend(
        [
            PROJECT_ROOT / model_name,
            WEIGHTS_DIR / model_name,
            RESOURCE_ROOT / model_name,
            RESOURCE_WEIGHTS_DIR / model_name,
        ]
    )

DEFAULT_WEIGHTS = next((path for path in WEIGHTS_CANDIDATES if path.exists()), WEIGHTS_CANDIDATES[0])
DEFAULT_TEMPLATE_FILE = (
    TEMPLATES_DIR / "action_templates.json"
    if (TEMPLATES_DIR / "action_templates.json").exists()
    else RESOURCE_TEMPLATES_DIR / "action_templates.json"
)
DEFAULT_RULES_FILE = (
    TEMPLATES_DIR / "error_rules.json"
    if (TEMPLATES_DIR / "error_rules.json").exists()
    else RESOURCE_TEMPLATES_DIR / "error_rules.json"
)
TEMPLATE_LIBRARY_FILE = TEMPLATES_DIR / "template_library.json"
DEFAULT_CHINESE_FONT = (
    FONTS_DIR / "simhei.ttf"
    if (FONTS_DIR / "simhei.ttf").exists()
    else RESOURCE_FONTS_DIR / "simhei.ttf"
)

DEFAULT_FRAME_WIDTH = 960
FLOW_WIDTH = 320
