"""
中文文本渲染辅助模块
使用Pillow解决OpenCV中文显示问题
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional


def draw_chinese_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: int = 24,
    color: Tuple[int, int, int] = (255, 255, 255),
    background: Optional[Tuple[int, int, int, int]] = None
) -> np.ndarray:
    """
    在图像上绘制中文文本
    
    Args:
        frame: OpenCV图像（BGR格式）
        text: 要绘制的文本
        position: 文本位置 (x, y)
        font_size: 字体大小
        color: 文本颜色 (R, G, B)
        background: 背景颜色 (R, G, B, A)，None表示无背景
    
    Returns:
        绘制后的图像
    """
    # 转换为PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img, 'RGBA' if background else 'RGB')
    
    # 加载中文字体
    font = _load_chinese_font(font_size)
    
    # 绘制背景
    if background:
        bbox = draw.textbbox(position, text, font=font)
        padding = 5
        draw.rectangle(
            [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
            fill=background
        )
    
    # 绘制文本
    draw.text(position, text, font=font, fill=color)
    
    # 转回OpenCV格式
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_multiline_chinese_text(
    frame: np.ndarray,
    lines: list[str],
    origin: Tuple[int, int],
    font_size: int = 20,
    color: Tuple[int, int, int] = (255, 255, 255),
    background: Optional[Tuple[int, int, int, int]] = None,
    line_spacing: int = 8
) -> np.ndarray:
    """
    绘制多行中文文本
    
    Args:
        frame: OpenCV图像
        lines: 文本行列表
        origin: 起始位置 (x, y)
        font_size: 字体大小
        color: 文本颜色
        background: 背景颜色
        line_spacing: 行间距
    
    Returns:
        绘制后的图像
    """
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img, 'RGBA' if background else 'RGB')
    font = _load_chinese_font(font_size)
    
    x, y = origin
    
    # 计算总体边界框
    if background:
        all_bboxes = [draw.textbbox((x, y + i * (font_size + line_spacing)), line, font=font) 
                      for i, line in enumerate(lines)]
        min_x = min(bbox[0] for bbox in all_bboxes)
        min_y = min(bbox[1] for bbox in all_bboxes)
        max_x = max(bbox[2] for bbox in all_bboxes)
        max_y = max(bbox[3] for bbox in all_bboxes)
        padding = 8
        draw.rectangle(
            [min_x - padding, min_y - padding, max_x + padding, max_y + padding],
            fill=background
        )
    
    # 绘制每一行
    for i, line in enumerate(lines):
        y_pos = y + i * (font_size + line_spacing)
        draw.text((x, y_pos), line, font=font, fill=color)
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _load_chinese_font(size: int) -> ImageFont.FreeTypeFont:
    """加载中文字体"""
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",      # 黑体
        "C:/Windows/Fonts/msyh.ttf",        # 微软雅黑
        "C:/Windows/Fonts/simsun.ttc",      # 宋体
        "/System/Library/Fonts/PingFang.ttc",  # Mac
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux
    ]
    
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            continue
    
    # 如果都失败，使用默认字体
    return ImageFont.load_default()
