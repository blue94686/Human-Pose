from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtGui import QImage

from .config import DEFAULT_CHINESE_FONT, PROJECT_ROOT


def qimage_from_bgr(frame: np.ndarray) -> QImage:
    """将 OpenCV 图像转成 QImage。"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()


def fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """按比例缩放画面，避免拉伸。"""
    if frame is None or frame.size == 0:
        return np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)
    src_h, src_w = frame.shape[:2]
    scale = min(width / max(src_w, 1), height / max(src_h, 1))
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def draw_text_cn(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int] = (20, 32, 52),
    font_size: int = 22,
) -> np.ndarray:
    """使用 Pillow 绘制中文。"""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = load_ui_font(font_size)
    draw.text(origin, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def load_ui_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """加载项目自带中文字体，失败时再回退系统字体。"""
    candidates = [
        DEFAULT_CHINESE_FONT,
        PROJECT_ROOT / "fonts" / "simhei.ttf",
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for candidate in candidates:
        try:
            if candidate and Path(candidate).exists():
                return ImageFont.truetype(str(candidate), size)
        except Exception:
            continue
    return ImageFont.load_default()


__all__ = ["draw_text_cn", "fit_frame", "load_ui_font", "qimage_from_bgr"]
