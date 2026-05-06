from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout

from .view_helpers import fit_frame, qimage_from_bgr


class CameraDialog(QDialog):
    """实时摄像头预览窗口。"""

    stop_requested = Signal()
    start_requested = Signal()
    start_record_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("摄像头实时分析控制台")
        self.resize(1100, 760)
        self.setMinimumSize(900, 640)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        toolbar = QHBoxLayout()
        self.start_button = QPushButton("▶ 开始分析")
        self.start_button.clicked.connect(self._request_start)
        self.stop_button = QPushButton("⏹ 停止分析")
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.stop_button.setEnabled(False)
        self.record_button = QPushButton("⏺ 实时分析并录制")
        self.record_button.clicked.connect(self._request_start_record)
        self.hint_label = QLabel("先确认摄像头编号与人物站位，再手动点击“开始分析”或“实时分析并录制”。")
        self.hint_label.setWordWrap(True)
        toolbar.addWidget(self.stop_button)
        toolbar.addWidget(self.start_button)
        toolbar.addWidget(self.record_button)
        toolbar.addStretch(1)
        toolbar.addWidget(self.hint_label, 1)
        layout.addLayout(toolbar)

        self.preview_label = QLabel("新窗口已打开，点击“开始分析”后进入实时识别。")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(480)
        self.preview_label.setStyleSheet("background:#0E1A2B;border-radius:12px;color:#FFFFFF;font-size:16px;")
        layout.addWidget(self.preview_label, 1)

        self.status_text = QPlainTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(180)
        self.status_text.setStyleSheet("font-size:13px;")
        layout.addWidget(self.status_text)

        self._last_pixmap: QPixmap | None = None
        self._frame_count = 0
        self._last_phase = None
        self._analysis_started = False

    def _request_start(self) -> None:
        self.start_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._analysis_started = True
        self.start_requested.emit()

    def _request_start_record(self) -> None:
        self.start_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._analysis_started = True
        self.start_record_requested.emit()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._analysis_started:
            self.stop_requested.emit()
        event.accept()

    def update_payload(self, payload: dict) -> None:
        frame = payload.get("frame")
        if isinstance(frame, np.ndarray):
            self._frame_count += 1
            fitted = fit_frame(frame, max(320, self.preview_label.width()), max(240, self.preview_label.height()))
            self._last_pixmap = QPixmap.fromImage(qimage_from_bgr(fitted))
            self.preview_label.setPixmap(self._last_pixmap)
            self.preview_label.setText("")

        current_phase = payload.get("phase_label") or "分析中"
        phase_changed = self._last_phase is not None and self._last_phase != current_phase
        self._last_phase = current_phase

        lines = [
            "【实时状态】",
            f"设备状态：{payload.get('device_state', '待命')}",
            f"摄像头编号：{payload.get('camera_index', '--')}",
            f"已处理帧数：{self._frame_count}",
            f"FPS：{float(payload.get('fps') or payload.get('camera_fps') or 0.0):.1f}",
            f"录制状态：{'录制中' if payload.get('record_enabled') else '未录制'}",
            "",
            "【动作阶段】",
        ]
        record_path = payload.get("record_path")
        if record_path:
            lines.insert(5, f"录制文件：{record_path}")
        lines.append(f"🔄 阶段切换 → {current_phase}" if phase_changed else f"当前阶段：{current_phase}")
        lines.extend(
            [
                "",
                "【动作指标】",
                f"运动强度：{float(payload.get('motion_mean') or 0.0):.3f}",
                f"左右平衡差：{float(payload.get('left_right_balance') or 0.0):.3f}",
                f"姿态质量：{payload.get('pose_quality_label') or '分析中'}",
                f"关键点数：{payload.get('visible_keypoint_count') or 0}",
                f"衔接评分：{float(payload.get('transition_score') or 0.0):.1f}",
            ]
        )
        torso_tilt = payload.get("torso_tilt_deg")
        if torso_tilt is not None:
            lines.append(f"躯干倾斜：{float(torso_tilt):.1f}°")
        arm_symmetry = payload.get("arm_symmetry_error")
        if arm_symmetry is not None:
            lines.append(f"双臂对称误差：{float(arm_symmetry):.1f}°")
        if payload.get("issues"):
            lines.append("")
            lines.append("【实时提示】")
            for issue in payload.get("issues", [])[:3]:
                lines.append(f"提示：{issue}")
        self.status_text.setPlainText("\n".join(lines))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._last_pixmap is not None:
            self.preview_label.setPixmap(self._last_pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


__all__ = ["CameraDialog"]
