from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from html import escape
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Qt, QSize, Signal
from PySide6.QtGui import QAction, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .analyzer import ActionAnalyzer
from .app_identity import APP_NAME, APP_SUBTITLE, APP_VERSION
from .camera_dialog import CameraDialog
from .config import (
    DEFAULT_TEMPLATE_FILE,
    DEFAULT_WEIGHTS,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    RESOURCE_ROOT,
    RESOURCE_WEIGHTS_DIR,
    TEMPLATE_LIBRARY_FILE,
    YOLO_POSE_MODELS,
)
from .desktop_backend import build_output_dir, open_camera, release_capture, scan_cameras
from .intake import EXCEL_SUFFIXES, IMAGE_SUFFIXES, TEXT_SUFFIXES, analyze_image_source, analyze_spreadsheet_source, analyze_text_source
from .left_panel import build_left_panel
from .pose import COCO_KEYPOINTS, SKELETON_EDGES, PoseEstimator
from .center_panel import build_center_panel
from .result_presenter import clear_result_widgets, populate_result_widgets
from .right_panel import build_right_panel
from .template_library import delete_template_entry, list_template_entries, save_template_result
from .test_user_state import TestUserEntry, format_test_user_display, format_test_user_subject
from .view_helpers import draw_text_cn as _draw_text_cn
from .view_helpers import fit_frame as _fit_frame
from .view_helpers import qimage_from_bgr as _qimage_from_bgr
from .exporter import build_prepost_metric_row, export_prepost_summary_xlsx


def _safe_open_path(path: Path) -> None:
    """打开本地文件或目录。"""
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def _clean_display_text(value: object) -> str:
    text = str(value or "").strip()
    replacements = {
        "⚠️": "",
        "⚠": "",
        "✅": "",
        "•": "-",
        "**": "",
        "‘": "",
        "’": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.split())

class AnalysisWorker(QThread):
    """后台视频分析线程。"""

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
        # [模式隔离] Worker 内部保存明确模式，避免模板生成和测试分析共用一套含糊状态。
        self.mode = analysis_mode
        self.analysis_mode = analysis_mode
        self.baseline_payload = baseline_payload
        self.evaluator_level = evaluator_level
        self._stop_requested = False

    def request_stop(self) -> None:
        """请求停止分析。"""
        self._stop_requested = True

    def run(self) -> None:
        """执行视频分析。"""
        try:
            analyzer = ActionAnalyzer(weights_path=self.weights_path, template_file=self.template_file)
            result = analyzer.analyze_video(
                source=self.source,
                description_text=self.description_text,
                output_dir=self.output_dir,
                frame_stride=1,
                frame_callback=self._emit_frame,
                progress_callback=self._emit_status,
                should_stop=lambda: self._stop_requested,
                analysis_mode=self.mode,
                baseline_payload=self.baseline_payload,
                evaluator_level=self.evaluator_level,
            )
        except Exception as exc:
            self.finished_error.emit(str(exc))
            return
        self.finished_success.emit(result)

    def _emit_frame(self, frame: np.ndarray, status: dict) -> None:
        payload = dict(status)
        payload["frame"] = frame
        self.progress_changed.emit(payload)

    def _emit_status(self, status: dict) -> None:
        if "frame" in status:
            status = dict(status)
            status.pop("frame", None)
        self.progress_changed.emit(dict(status))


class CameraWorker(QThread):
    """后台实时摄像头分析线程。"""

    frame_changed = Signal(dict)
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
        baseline_payload: dict | None,
        target_fps: int = 20,
        record_output_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.description_text = description_text
        self.weights_path = weights_path
        self.template_file = template_file
        self.evaluator_level = evaluator_level
        self.baseline_payload = baseline_payload
        self.target_fps = max(15, min(25, int(target_fps)))
        self.record_output_path = record_output_path
        self._stop_requested = False

    def request_stop(self) -> None:
        """请求停止摄像头线程。"""
        self._stop_requested = True

    def run(self) -> None:
        """执行实时摄像头分析。"""
        capture = None
        writer = None
        try:
            capture = open_camera(self.camera_index)
            if capture is None or not capture.isOpened():
                raise RuntimeError("未检测到摄像头设备")

            first_frame = None
            for _ in range(12):
                ok, frame = capture.read()
                if ok and frame is not None and frame.size > 0:
                    first_frame = frame
                    break
                self.msleep(40)
            if first_frame is None:
                raise RuntimeError("摄像头启动失败：未检测到有效画面")

            analyzer = ActionAnalyzer(weights_path=self.weights_path, template_file=self.template_file)
            session = analyzer.create_realtime_session(
                self.description_text,
                evaluator_level=self.evaluator_level,
            )

            frame_interval_ms = max(1, int(1000 / self.target_fps))

            while not self._stop_requested:
                # 先读取新帧
                ok, current_frame = capture.read()
                if not ok or current_frame is None or current_frame.size == 0:
                    self.msleep(10)
                    continue

                # 分析当前帧
                overlay, status = analyzer.analyze_realtime_frame(current_frame, session)
                if self.record_output_path is not None:
                    if writer is None:
                        frame_h, frame_w = overlay.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(self.record_output_path), fourcc, float(self.target_fps), (frame_w, frame_h))
                        if not writer.isOpened():
                            writer.release()
                            writer = None
                            raise RuntimeError(f"录制文件创建失败：{self.record_output_path}")
                    writer.write(overlay)
                status["frame"] = overlay
                status["camera_index"] = self.camera_index
                status["device_state"] = "运行中"
                status["target_fps"] = self.target_fps
                status["record_enabled"] = self.record_output_path is not None
                status["record_path"] = str(self.record_output_path) if self.record_output_path is not None else ""
                self.frame_changed.emit(status)

                # 控制帧率
                self.msleep(frame_interval_ms)

            if self.record_output_path is not None:
                self.finished_success.emit(f"摄像头 {self.camera_index} 已停止，分析录制文件已保存：{self.record_output_path}")
            else:
                self.finished_success.emit(f"摄像头 {self.camera_index} 已停止，资源已释放。")
        except Exception as exc:
            self.finished_error.emit(str(exc))
        finally:
            if writer is not None:
                writer.release()
            release_capture(capture)


class MotionAnalysisQtWindow(QMainWindow):
    """桌面三栏主界面。"""

    def __init__(self, *, app_name: str = APP_NAME, app_version: str = APP_VERSION) -> None:
        super().__init__()
        self.app_name = app_name.strip() or APP_NAME
        self.app_version = app_version.strip() or APP_VERSION
        self.app_title = f"{self.app_name} {self.app_version}".strip()
        self.setWindowTitle(self.app_title)
        self.resize(1720, 960)
        self.setMinimumSize(QSize(1380, 820))

        self.analysis_worker: AnalysisWorker | None = None
        self.camera_worker: CameraWorker | None = None
        self.camera_dialog: CameraDialog | None = None
        self.current_result = None
        self.current_output_dir = OUTPUTS_DIR
        self.template_entries: list[dict] = []
        self.current_template_source: Path | None = None
        self.current_test_source: Path | None = None
        self.test_users: list[TestUserEntry] = []
        self.current_test_user_index: int | None = None
        self.running_test_user_index: int | None = None
        self.running_analysis_mode: str | None = None
        self.current_reference_frame: np.ndarray | None = None
        self.current_test_frame: np.ndarray | None = None
        self.current_overlay_frame: np.ndarray | None = None
        self.current_baseline_payload: dict | None = None
        self.current_baseline_path: Path | None = None
        self.running_baseline_payload: dict | None = None
        self.running_source: str | None = None
        self.current_weights_path = str(DEFAULT_WEIGHTS)
        self.output_buttons: dict[str, QPushButton] = {}
        self.playback_speed = "1.0x"
        self.left_panel_visible = True
        self.right_panel_visible = True
        self.current_camera_record_path: Path | None = None
        self.camera_record_purpose: str | None = None
        self._analysis_start_time: float | None = None
        self._last_live_suggestions_text = ""
        self._last_live_suggestions_frame = -1
        self._last_added_suggestion = ""
        self._last_pose_backend_reason = ""
        self._pose_backend_available = False

        self._build_ui()
        self._apply_styles()
        self._refresh_template_library()
        self._update_output_buttons({})
        self.statusBar().showMessage("待机中")
        self._append_log("系统启动完成，默认进入桌面模式。")
        self._append_log("流程提示：1️⃣ 导入模板 → 2️⃣ 选择测试 → 3️⃣ 开始分析 → 4️⃣ 查看结果")
        self._show_pose_backend_notice()

    def _compose_analysis_description(self, base_text: str, extra_terms: str) -> str:
        """组合动作描述与手动术语词汇。"""
        base = (base_text or "").strip()
        terms = (extra_terms or "").strip()
        if not terms:
            return base
        if not base:
            return f"动作术语与要求：{terms}"
        return f"{base}\n补充术语与动作词汇：{terms}"

    def _inject_project_type(self, description_text: str, project_type: str) -> str:
        """将手动指定的项目类型并入描述文本。"""
        project_type = (project_type or "").strip()
        if not project_type or project_type == "自动识别":
            return description_text
        mapping = {
            "八段锦": "项目类型：八段锦（健身气功）",
            "武术": "项目类型：武术基本功",
            "民族舞": "项目类型：民族舞",
            "太极": "项目类型：太极",
        }
        prefix = mapping.get(project_type, f"项目类型：{project_type}")
        text = (description_text or "").strip()
        if not text:
            return prefix
        return f"{prefix}\n{text}"

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("RootPage")
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(18, 16, 18, 16)
        main_layout.setSpacing(16)

        header = QFrame()
        header.setObjectName("DashboardHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(20, 16, 20, 16)
        header_layout.setSpacing(14)

        top_row = QHBoxLayout()
        top_row.setSpacing(16)
        title_wrap = QVBoxLayout()
        title_wrap.setSpacing(4)
        title = QLabel(self.app_title)
        title.setObjectName("HeroTitle")
        subtitle = QLabel(APP_SUBTITLE)
        subtitle.setObjectName("HeroSubTitle")
        title_wrap.addWidget(title)
        title_wrap.addWidget(subtitle)
        top_row.addLayout(title_wrap, 1)

        header_info = QVBoxLayout()
        header_info.setSpacing(4)
        self.header_hint_label = QLabel("待机中，先导入标准模板或打开摄像头")
        self.header_hint_label.setObjectName("HeaderHint")
        self.header_ratio_label = QLabel("模板匹配度 --   偏差指数 --")
        self.header_ratio_label.setObjectName("HeaderRatio")
        header_info.addWidget(self.header_hint_label, 0, Qt.AlignRight)
        header_info.addWidget(self.header_ratio_label, 0, Qt.AlignRight)
        top_row.addLayout(header_info)

        self.status_chip = QLabel("空闲")
        self.status_chip.setObjectName("StatusChip")
        top_row.addWidget(self.status_chip, 0, Qt.AlignTop)
        header_layout.addLayout(top_row)

        step_row = QHBoxLayout()
        step_row.setSpacing(10)
        self.step_labels: list[QLabel] = []
        steps = [
            ("导入素材 / 摄像头", True),
            ("选择标准动作法", False),
            ("姿态分析", False),
            ("生成整改报告", False),
        ]
        for index, (text, is_active) in enumerate(steps, start=1):
            step_label = QLabel(f"{index} {text}")
            step_label.setObjectName("ActiveStepChip" if is_active else "StepChip")
            step_label.setAlignment(Qt.AlignCenter)
            step_row.addWidget(step_label)
            self.step_labels.append(step_label)
            if index < len(steps):
                arrow = QLabel("→")
                arrow.setObjectName("StepArrow")
                step_row.addWidget(arrow)
        step_row.addStretch(1)
        header_layout.addLayout(step_row)
        main_layout.addWidget(header)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(16)
        main_layout.addLayout(body_layout, 1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(350)
        left_scroll.setObjectName("LeftScroll")
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_container = QWidget()
        left_scroll.setWidget(left_container)
        left_layout = QVBoxLayout(left_container)
        left_layout.setSpacing(14)
        left_layout.setContentsMargins(0, 0, 0, 0)
        build_left_panel(self, left_layout)
        self.left_scroll = left_scroll
        body_layout.addWidget(left_scroll, 3)

        self.toggle_left_panel_btn = QToolButton()
        self.toggle_left_panel_btn.setObjectName("PanelToggleButton")
        self.toggle_left_panel_btn.clicked.connect(self._toggle_left_panel)
        left_toggle_layout = QVBoxLayout()
        left_toggle_layout.setContentsMargins(0, 36, 0, 0)
        left_toggle_layout.setSpacing(0)
        left_toggle_layout.addWidget(self.toggle_left_panel_btn, 0, Qt.AlignTop)
        left_toggle_layout.addStretch(1)
        body_layout.addLayout(left_toggle_layout)

        center_scroll = QScrollArea()
        center_scroll.setWidgetResizable(True)
        center_scroll.setObjectName("CenterScroll")
        center_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        center_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        center_container = QWidget()
        center_scroll.setWidget(center_container)
        center_layout = QVBoxLayout(center_container)
        center_layout.setSpacing(14)
        center_layout.setContentsMargins(0, 0, 0, 0)
        build_center_panel(self, center_layout)
        body_layout.addWidget(center_scroll, 7)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setObjectName("RightScroll")
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_container = QWidget()
        right_scroll.setWidget(right_container)
        right_container.setMinimumWidth(420)
        right_container.setMaximumWidth(540)
        right_layout = QVBoxLayout(right_container)
        right_layout.setSpacing(14)
        right_layout.setContentsMargins(0, 0, 0, 0)
        build_right_panel(self, right_layout)
        self.right_scroll = right_scroll
        self.toggle_right_panel_btn = QToolButton()
        self.toggle_right_panel_btn.setObjectName("PanelToggleButton")
        self.toggle_right_panel_btn.clicked.connect(self._toggle_right_panel)
        right_toggle_layout = QVBoxLayout()
        right_toggle_layout.setContentsMargins(0, 36, 0, 0)
        right_toggle_layout.setSpacing(0)
        right_toggle_layout.addWidget(self.toggle_right_panel_btn, 0, Qt.AlignTop)
        right_toggle_layout.addStretch(1)
        body_layout.addLayout(right_toggle_layout)
        body_layout.addWidget(right_scroll, 4)
        self._set_left_panel_visible(True)
        self._set_right_panel_visible(True)

        footer = QFrame()
        footer.setObjectName("CommandBar")
        footer_layout = QGridLayout(footer)
        footer_layout.setContentsMargins(16, 10, 16, 10)
        footer_layout.setHorizontalSpacing(12)
        footer_layout.setVerticalSpacing(8)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.task_status_label = QLabel("当前任务：空闲")
        self.camera_status_label = QLabel("设备状态：未启动")
        self.stop_button = QPushButton("⏹️ 停止分析")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_running_task)
        self.open_output_button = QPushButton("📂 打开输出目录")
        self.open_output_button.clicked.connect(self._open_output_dir)
        footer_layout.addWidget(self.progress_bar, 0, 0, 1, 2)
        footer_layout.addWidget(self.task_status_label, 0, 2, 1, 2)
        footer_layout.addWidget(self.camera_status_label, 1, 0, 1, 2)
        footer_layout.addWidget(self.stop_button, 1, 2)
        footer_layout.addWidget(self.open_output_button, 1, 3)
        main_layout.addWidget(footer)

        self._create_menu()

    def _create_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("文件")
        open_output_action = QAction("打开输出目录", self)
        open_output_action.triggered.connect(self._open_output_dir)
        file_menu.addAction(open_output_action)
        clear_action = QAction("清空日志", self)
        clear_action.triggered.connect(self.log_text.clear)
        file_menu.addAction(clear_action)

    def _create_card(self, title: str) -> QGroupBox:
        card = QGroupBox(title)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        return card

    def _create_video_stage(self, subtitle: str, title: str, preview_label: QLabel) -> QWidget:
        stage = QFrame()
        stage.setObjectName("VideoStageCard")
        layout = QVBoxLayout(stage)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("VideoStageHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(4)
        title_label = QLabel(title)
        title_label.setObjectName("VideoStageTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setWordWrap(True)
        subtitle_label.setObjectName("VideoStageSubtitle")
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        layout.addWidget(header)
        layout.addWidget(preview_label, 1)
        return stage

    def _make_action_button(self, text: str, tooltip: str, slot) -> QPushButton:
        button = QPushButton(text)
        button.setToolTip(tooltip)
        button.clicked.connect(slot)
        return button

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            #RootPage {
                background: #eef3f7;
            }
            QWidget {
                color: #16324f;
                font-family: "Microsoft YaHei UI", "Segoe UI";
                font-size: 14px;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
            #LeftScroll, #CenterScroll, #RightScroll {
                border: none;
                background: transparent;
            }
            #DashboardHeader, #CommandBar, #TransportBar {
                background: #ffffff;
                border: 1px solid #dbe5ef;
                border-radius: 18px;
            }
            #HeroTitle {
                font-size: 30px;
                font-weight: 700;
                color: #102d4c;
            }
            #HeroSubTitle {
                font-size: 14px;
                color: #66819b;
            }
            #HeaderHint {
                color: #f07e39;
                font-size: 15px;
                font-weight: 700;
            }
            #HeaderRatio {
                color: #6b8196;
                font-size: 13px;
            }
            #AnalysisSplitter::handle {
                background: #d9e5ef;
                width: 8px;
                border-radius: 4px;
            }
            #AnalysisSplitter::handle:hover {
                background: #9ec7e8;
            }
            #AnalysisColumn {
                background: transparent;
            }
            #CameraScanStatus {
                color: #6b8196;
                font-size: 13px;
                padding: 2px 4px;
            }
            #CameraScanStatus[scanState="success"] {
                color: #177a3c;
                font-weight: 700;
            }
            #CameraScanStatus[scanState="error"] {
                color: #c24030;
                font-weight: 700;
            }
            #StatusChip {
                background: #0f88d8;
                color: white;
                border-radius: 14px;
                padding: 10px 18px;
                font-weight: 600;
                font-size: 14px;
            }
            #StepChip, #ActiveStepChip {
                border-radius: 14px;
                padding: 12px 20px;
                font-size: 15px;
                font-weight: 700;
                min-width: 150px;
            }
            #StepChip {
                background: #ffffff;
                border: 1px solid #dbe6f0;
                color: #30485f;
            }
            #ActiveStepChip {
                background: #1678d3;
                border: 1px solid #1678d3;
                color: #ffffff;
            }
            #StepArrow {
                color: #6b8196;
                font-size: 24px;
                font-weight: 700;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #dbe5ef;
                border-radius: 18px;
                margin-top: 14px;
                padding-top: 18px;
                font-weight: 700;
                font-size: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                color: #143652;
            }
            #CollapsibleSection {
                background: #ffffff;
                border: 1px solid #dbe5ef;
                border-radius: 18px;
            }
            #SectionToggle {
                text-align: left;
                background: #ffffff;
                color: #143652;
                border: none;
                border-bottom: 1px solid #ebf1f7;
                border-top-left-radius: 18px;
                border-top-right-radius: 18px;
                padding: 12px 16px;
                font-size: 15px;
                font-weight: 700;
            }
            #SectionToggle:hover:!disabled {
                background: #f5f9fd;
            }
            #SectionBody {
                background: transparent;
                border: none;
            }
            #OverviewCard {
                background: #f7fbff;
                border: 1px solid #d6e6f5;
                border-radius: 14px;
            }
            #OverviewTitle {
                color: #688197;
                font-size: 13px;
            }
            #OverviewValue {
                color: #1284d8;
                font-size: 22px;
                font-weight: 800;
            }
            QPushButton {
                background: #1678d3;
                color: white;
                border: none;
                border-radius: 12px;
                padding: 10px 14px;
                font-weight: 600;
                font-size: 14px;
                min-height: 20px;
            }
            QPushButton:disabled {
                background: #c7d5e3;
                color: #6b8196;
            }
            QPushButton:hover:!disabled {
                background: #0f68bc;
            }
            QLineEdit, QPlainTextEdit, QTextEdit, QComboBox {
                background: #ffffff;
                border: 1px solid #ccdae7;
                border-radius: 12px;
                padding: 10px 12px;
                font-size: 14px;
                min-height: 24px;
            }
            QListWidget {
                background: #ffffff;
                border: 1px solid #ccdae7;
                border-radius: 12px;
                padding: 6px;
            }
            QListWidget::item {
                min-height: 34px;
                padding: 8px 10px;
                border-radius: 8px;
                color: #16324f;
            }
            QListWidget::item:selected {
                background: #e8f3ff;
                color: #0f68bc;
                font-weight: 700;
            }
            QTabWidget::pane {
                border: 1px solid #d8e2ec;
                border-radius: 14px;
                background: #ffffff;
                top: -1px;
            }
            QTabBar::tab {
                min-width: 92px;
                padding: 10px 14px;
                background: #f2f7fb;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                margin-right: 4px;
                color: #486176;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                border: 1px solid #d8e2ec;
                border-bottom-color: #ffffff;
                color: #143652;
                font-weight: 700;
            }
            QProgressBar {
                background: #eff5fa;
                border: 1px solid #d1deea;
                border-radius: 10px;
                text-align: center;
                min-height: 20px;
            }
            QProgressBar::chunk {
                background: #1992de;
                border-radius: 9px;
            }
            QToolButton {
                background: #eef5fb;
                border: 1px solid #d8e4ef;
                border-radius: 10px;
                padding: 8px 12px;
                color: #143652;
                font-weight: 600;
            }
            #HelpToolButton {
                background: #f5f9fd;
                border: 1px solid #c9d8e6;
                border-radius: 9px;
                color: #1678d3;
                font-weight: 800;
                min-width: 22px;
                max-width: 22px;
                min-height: 22px;
                max-height: 22px;
                padding: 0;
            }
            #HelpToolButton:hover {
                background: #e6f2ff;
                border-color: #96c5ed;
            }
            #PanelToggleButton {
                background: #ffffff;
                border: 1px solid #dbe5ef;
                border-radius: 14px;
                color: #143652;
                font-size: 22px;
                font-weight: 700;
                min-width: 42px;
                max-width: 42px;
                min-height: 74px;
                padding: 6px 4px;
            }
            #PanelToggleButton:hover {
                background: #f5f9fd;
                border-color: #c8d7e6;
            }
            QMenuBar {
                background: #ffffff;
                border: 1px solid #dbe5ef;
                border-radius: 12px;
                padding: 4px 8px;
            }
            QMenuBar::item:selected {
                background: #eef5fb;
                border-radius: 8px;
            }
            #VideoStageCard {
                background: #ffffff;
                border: 1px solid #dbe5ef;
                border-radius: 18px;
            }
            #VideoStageHeader {
                background: #0b5973;
                border-top-left-radius: 18px;
                border-top-right-radius: 18px;
            }
            #VideoStageTitle {
                color: #ffffff;
                font-size: 16px;
                font-weight: 700;
            }
            #VideoStageSubtitle {
                color: #c8e6ef;
                font-size: 12px;
            }
            #StagePreview {
                background: #072e3b;
                border: 0;
                border-bottom-left-radius: 18px;
                border-bottom-right-radius: 18px;
                color: #d4edf7;
                font-size: 14px;
                min-height: 260px;
            }
            #StageHint {
                color: #6b8196;
                font-size: 13px;
            }
            #PoseNotice {
                color: #0b5973;
                background: #eef8fb;
                border: 1px solid #cfe7ef;
                border-radius: 12px;
                padding: 10px 12px;
                font-weight: 700;
                font-size: 13px;
            }
            QTextBrowser {
                background: #ffffff;
                border: 1px solid #d2dee9;
                border-radius: 12px;
                padding: 12px;
                font-size: 14px;
                line-height: 1.6;
            }
            QLabel {
                font-size: 14px;
            }
            #ScoreLabel {
                font-size: 42px;
                font-weight: 800;
                color: #1678d3;
            }
            #GradeLabel {
                color: #143652;
                font-size: 16px;
                font-weight: 700;
            }
            #ResultSubject {
                color: #5e7488;
                font-size: 13px;
                font-weight: 700;
            }
            """
        )
    def _sync_weights_by_combo(self) -> None:
        mapping = {
            0: "yolov8n-pose.pt",
            1: "yolov8s-pose.pt",
            2: "yolov8m-pose.pt",
            3: "yolov8l-pose.pt",
            4: "yolov8x-pose.pt",
            5: "yolov8x-pose-p6.pt",
        }
        target = mapping.get(self.model_combo.currentIndex(), "yolov8n-pose.pt")
        self.current_weights_path = str(self._resolve_weights_candidate(target))
        self._show_pose_backend_notice()

    def _resolve_weights_candidate(self, model_name: str) -> Path:
        candidates = [
            RESOURCE_ROOT / model_name,
            RESOURCE_WEIGHTS_DIR / model_name,
            PROJECT_ROOT / model_name,
            PROJECT_ROOT / "weights" / model_name,
        ]
        return next((path for path in candidates if path.exists() and path.is_file()), candidates[0])

    def _set_active_step(self, step_index: int) -> None:
        """刷新顶部步骤条。"""
        for index, label in enumerate(self.step_labels, start=1):
            label.setObjectName("ActiveStepChip" if index == step_index else "StepChip")
            label.style().unpolish(label)
            label.style().polish(label)

    def _show_pose_backend_notice(self) -> None:
        """刷新姿态后端提示。"""
        estimator = PoseEstimator(weights_path=self.current_weights_path)
        self._pose_backend_available = bool(estimator.status.available)
        self._last_pose_backend_reason = str(estimator.status.reason or "").strip()
        notice_text = self._last_pose_backend_reason or "当前为光流兜底模式，无骨架可视化"
        self.pose_notice_label.setText(notice_text)
        self.pose_notice_label.setToolTip(
            f"权重路径：{estimator.status.model_path or self.current_weights_path}\n状态：{notice_text}"
        )

    def _infer_test_type_from_source(self, source_path: Path) -> str:
        stem_text = str(source_path.stem or "").strip().lower()
        if "后测" in stem_text or "post" in stem_text:
            return "后测"
        if "前测" in stem_text or "pre" in stem_text:
            return "前测"
        return "未标记"

    def _current_baseline_name(self) -> str | None:
        """返回当前选中的模板名称。"""
        entry = self.baseline_combo.currentData() or self.template_combo.currentData()
        if isinstance(entry, dict):
            template_name = (entry.get("template_name") or "").strip()
            if template_name:
                return template_name
        template_name = self.template_name_edit.text().strip()
        return template_name or None

    def _test_user_display_text(self, index: int, test_user: TestUserEntry) -> str:
        """构造测试用户列表展示文本。"""
        return format_test_user_display(index, test_user)

    def _refresh_test_user_item(self, index: int) -> None:
        """刷新单个测试用户列表项。"""
        if index < 0 or index >= len(self.test_users):
            return
        item = self.test_user_list.item(index)
        if item is None:
            return
        item.setData(Qt.UserRole, index)
        item.setText(self._test_user_display_text(index, self.test_users[index]))

    def _refresh_all_test_user_items(self) -> None:
        """刷新测试用户列表显示。"""
        for index in range(len(self.test_users)):
            self._refresh_test_user_item(index)

    def _clear_result_panels(self, subject_text: str = "当前对象：未选择测试用户") -> None:
        """清空右侧结果区。"""
        clear_result_widgets(self, subject_text)

    def _scan_cameras(self) -> None:
        """扫描可用摄像头。"""
        self._set_camera_scan_status("正在扫描摄像头...", "pending")
        try:
            cameras = scan_cameras(max_index=4)
        except Exception as exc:
            self._set_camera_scan_status("扫描失败，请检查权限或设备连接。", "error")
            QMessageBox.warning(self, "错误", f"摄像头扫描失败：{exc}")
            return
        self.camera_combo.clear()
        if not cameras:
            QMessageBox.warning(self, "提示", "未检测到摄像头设备")
            self.camera_status_label.setText("设备状态：未检测到摄像头设备")
            self._set_camera_scan_status("未检测到摄像头设备。", "error")
            return
        for item in cameras:
            label = f"摄像头 {item['index']} ({item['width']}x{item['height']})"
            self.camera_combo.addItem(label, item["index"])
        if self.camera_combo.count() > 0:
            self.camera_combo.setCurrentIndex(0)
        self.camera_status_label.setText(f"设备状态：已检测到 {len(cameras)} 个摄像头")
        self._set_camera_scan_status("设备在线：扫描成功，请选择摄像头。", "success")
        self._append_log(f"摄像头扫描完成，共检测到 {len(cameras)} 个设备。")

    def _set_camera_scan_status(self, text: str, state: str = "pending") -> None:
        """刷新摄像头扫描提示文本。"""
        label = getattr(self, "camera_scan_status_label", None)
        if label is None:
            return
        label.setText(text)
        label.setProperty("scanState", state)
        label.style().unpolish(label)
        label.style().polish(label)

    def _pick_template_video(self) -> None:
        self._pick_template_source("视频文件 (*.mp4 *.avi *.mov *.mkv)")

    def _pick_template_image(self) -> None:
        self._pick_template_source("图片文件 (*.png *.jpg *.jpeg *.bmp *.webp)")

    def _pick_template_excel(self) -> None:
        self._pick_template_source("Excel 文件 (*.xlsx *.xlsm *.xltx *.xltm)")

    def _pick_template_text(self) -> None:
        self._pick_template_source("文本文件 (*.txt *.md *.json)")

    def _pick_test_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择测试视频", str(PROJECT_ROOT), "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        source_path = Path(path)
        self._add_test_user_from_video(source_path)

    def _add_test_user_from_video(self, source_path: Path) -> None:
        """将本地或摄像头录制视频加入测试用户列表。"""
        inferred_test_type = self._infer_test_type_from_source(source_path)
        test_user = TestUserEntry(
            name=source_path.stem,
            source=source_path,
            description=f"测试动作视频：{source_path.stem}",
            terminology="",
            project_type=self.test_project_combo.currentText().strip() or "自动识别",
            test_type=inferred_test_type,
        )
        self.test_users.append(test_user)
        item = QListWidgetItem(self._test_user_display_text(len(self.test_users) - 1, test_user))
        item.setData(Qt.UserRole, len(self.test_users) - 1)
        self.test_user_list.addItem(item)
        self.test_user_list.setCurrentRow(len(self.test_users) - 1)
        if hasattr(self, "test_type_combo"):
            self.test_type_combo.blockSignals(True)
            self.test_type_combo.setCurrentText(inferred_test_type)
            self.test_type_combo.blockSignals(False)
        self._append_log(f"已加入测试用户视频：{source_path}")

    def _switch_test_user(self, row: int) -> None:
        """切换当前测试用户。"""
        if row < 0 or row >= len(self.test_users):
            self.current_test_user_index = None
            self.current_test_source = None
            self.result_subject_label.setText("当前对象：未选择测试用户")
            return
        self.current_test_user_index = row
        test_user = self.test_users[row]
        source_path = test_user.source
        self.current_test_source = source_path
        self.test_desc_edit.blockSignals(True)
        self.test_desc_edit.setPlainText(test_user.description or f"测试动作视频：{source_path.stem}")
        self.test_desc_edit.blockSignals(False)
        self.test_terms_edit.blockSignals(True)
        self.test_terms_edit.setPlainText(test_user.terminology or "")
        self.test_terms_edit.blockSignals(False)
        self.test_project_combo.blockSignals(True)
        self.test_project_combo.setCurrentText(test_user.project_type or "自动识别")
        self.test_project_combo.blockSignals(False)
        self.test_type_combo.blockSignals(True)
        self.test_type_combo.setCurrentText((test_user.test_type or "未标记").strip() or "未标记")
        self.test_type_combo.blockSignals(False)
        subject_text = format_test_user_subject(test_user)
        self.result_subject_label.setText(subject_text)
        self._load_video_preview(source_path, target="test")
        result = test_user.result
        if result is not None:
            self.current_result = result
            self._populate_result_panels(result)
            self._update_output_buttons(result.artifacts)
            self.current_output_dir = Path(result.output_dir)
        else:
            if test_user.last_output_dir is not None:
                self.current_output_dir = test_user.last_output_dir
            self._clear_result_panels(subject_text)

    def _sync_current_test_user_description(self) -> None:
        """同步当前测试用户描述。"""
        if self.current_test_user_index is None or self.current_test_user_index >= len(self.test_users):
            return
        self.test_users[self.current_test_user_index].description = self.test_desc_edit.toPlainText().strip()

    def _sync_current_test_user_terminology(self) -> None:
        """同步当前测试用户术语词汇。"""
        if self.current_test_user_index is None or self.current_test_user_index >= len(self.test_users):
            return
        self.test_users[self.current_test_user_index].terminology = self.test_terms_edit.toPlainText().strip()

    def _sync_current_test_user_project_type(self) -> None:
        """同步当前测试用户项目类型。"""
        if self.current_test_user_index is None or self.current_test_user_index >= len(self.test_users):
            return
        self.test_users[self.current_test_user_index].project_type = self.test_project_combo.currentText().strip() or "自动识别"

    def _sync_current_test_user_test_type(self) -> None:
        """同步当前测试用户测试类型。"""
        if self.current_test_user_index is None or self.current_test_user_index >= len(self.test_users):
            return
        test_user = self.test_users[self.current_test_user_index]
        test_user.test_type = self.test_type_combo.currentText().strip() or "未标记"
        self._refresh_test_user_item(self.current_test_user_index)
        self.result_subject_label.setText(format_test_user_subject(test_user))

    def _remove_current_test_user(self) -> None:
        """删除当前测试用户。"""
        if self.current_test_user_index is None or not self.test_users:
            QMessageBox.warning(self, "提示", "当前没有可删除的测试对象。")
            return
        current_index = int(self.current_test_user_index)
        if current_index < 0 or current_index >= len(self.test_users):
            QMessageBox.warning(self, "提示", "当前测试对象索引无效。")
            return
        removed = self.test_users.pop(current_index)
        self.test_user_list.takeItem(current_index)
        self._refresh_all_test_user_items()
        self._append_log(f"已删除测试对象：{removed.name}")
        if not self.test_users:
            self.current_test_user_index = None
            self.current_test_source = None
            self.test_desc_edit.clear()
            self.test_terms_edit.clear()
            self.test_project_combo.setCurrentText("自动识别")
            self.test_type_combo.setCurrentText("未标记")
            self._clear_result_panels()
            return
        next_index = min(current_index, len(self.test_users) - 1)
        self.test_user_list.setCurrentRow(next_index)

    def _open_current_test_output_dir(self) -> None:
        """打开当前测试用户的输出目录。"""
        if self.current_test_user_index is None or self.current_test_user_index >= len(self.test_users):
            QMessageBox.warning(self, "提示", "请先选择一个测试用户。")
            return
        result = self.test_users[self.current_test_user_index].result
        if result is None:
            QMessageBox.warning(self, "提示", "当前测试用户尚未生成分析结果。")
            return
        _safe_open_path(Path(result.output_dir))

    def _pick_template_source(self, filter_text: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择模板数据", str(PROJECT_ROOT), filter_text)
        if not path:
            return
        source_path = Path(path)
        self.current_template_source = source_path
        self.template_name_edit.setText(source_path.stem)
        self._append_log(f"已选择模板数据：{source_path}")

        try:
            if source_path.suffix.lower() in IMAGE_SUFFIXES:
                suggestion, frame = analyze_image_source(source_path, self.current_weights_path, DEFAULT_TEMPLATE_FILE)
                self.template_desc_edit.setPlainText(suggestion.suggested_description)
                self.current_reference_frame = frame
                self._set_preview_image(self.reference_preview, frame)
            elif source_path.suffix.lower() in EXCEL_SUFFIXES:
                suggestion = analyze_spreadsheet_source(source_path, DEFAULT_TEMPLATE_FILE)
                self.template_desc_edit.setPlainText(suggestion.suggested_description)
            elif source_path.suffix.lower() in TEXT_SUFFIXES:
                suggestion = analyze_text_source(source_path, DEFAULT_TEMPLATE_FILE)
                self.template_desc_edit.setPlainText(suggestion.suggested_description)
            else:
                self.template_desc_edit.setPlainText(f"{source_path.stem} 动作模板，建议补充节奏、躯干、手臂和下盘要求。")
                self.template_terms_edit.setPlainText("")
                self._load_video_preview(source_path, target="reference")
        except Exception as exc:
            QMessageBox.warning(self, "错误", f"模板数据读取失败：{exc}")

    def _load_video_preview(self, path: Path, *, target: str) -> None:
        """读取视频首帧预览。"""
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"无法打开视频文件：{path}")
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise RuntimeError(f"无法读取视频文件：{path}")
        if target == "reference":
            self.current_reference_frame = frame
            self._set_preview_image(self.reference_preview, frame)
        else:
            self.current_test_frame = frame
            self._set_preview_image(self.test_preview, frame)

    def _set_preview_image(self, label: QLabel, frame: np.ndarray) -> None:
        """刷新预览图。"""
        target_width = max(180, label.contentsRect().width() - 8)
        target_height = max(180, label.contentsRect().height() - 8)
        fitted = _fit_frame(frame, target_width, target_height)
        pixmap = QPixmap.fromImage(_qimage_from_bgr(fitted))
        label.setPixmap(pixmap)
        label.setText("")

    def _current_evaluator_level(self) -> str:
        """将熟练度下拉框映射到分析器等级。"""
        mapping = {
            "初学者": "初习者",
            "中级": "进阶者",
            "专业运动员": "熟练者",
        }
        return mapping.get(self.level_combo.currentText(), "熟练者")

    def _start_test_analysis(self) -> None:
        """启动测试动作分析。"""
        if self.analysis_worker is not None:
            QMessageBox.warning(self, "提示", "当前已有分析任务正在执行。")
            return
        if self.current_test_user_index is None:
            QMessageBox.warning(self, "提示", "请先选择需要分析的测试对象。")
            return
        if self.current_test_source is None or not self.current_test_source.exists():
            QMessageBox.warning(self, "提示", "请先导入测试视频。")
            return
        if self.current_baseline_payload is None:
            QMessageBox.warning(self, "提示", "请先加载标准模板基线。")
            return
        if self.current_test_user_index >= len(self.test_users):
            QMessageBox.warning(self, "提示", "当前测试对象索引超出范围。")
            return
        test_user = self.test_users[self.current_test_user_index]
        source_snapshot = Path(test_user.source)
        if self.current_test_source is None or Path(self.current_test_source) != source_snapshot:
            self.current_test_source = source_snapshot
        baseline_snapshot = dict(self.current_baseline_payload)
        validation_message = self._validate_analysis_inputs(
            source=source_snapshot,
            analysis_mode="test",
            baseline_payload=baseline_snapshot,
        )
        if validation_message:
            QMessageBox.warning(self, "提示", validation_message)
            return
        description_text = self._compose_analysis_description(
            self.test_desc_edit.toPlainText().strip() or f"{source_snapshot.stem} 测试动作分析",
            self.test_terms_edit.toPlainText().strip(),
        )
        description_text = self._inject_project_type(description_text, self.test_project_combo.currentText())
        test_user.description = self.test_desc_edit.toPlainText().strip() or f"{source_snapshot.stem} 测试动作分析"
        test_user.terminology = self.test_terms_edit.toPlainText().strip()
        test_user.project_type = self.test_project_combo.currentText().strip() or "自动识别"
        test_user.test_type = self.test_type_combo.currentText().strip() or "未标记"
        test_user.status = "分析中"
        test_user.baseline_name = self._current_baseline_name()
        self.running_test_user_index = self.current_test_user_index
        self.running_analysis_mode = "test"
        self._refresh_test_user_item(self.current_test_user_index)
        output_dir = build_output_dir("test")
        self.current_output_dir = output_dir
        self._set_active_step(3)
        self.current_test_frame = None
        self.current_overlay_frame = None
        self._append_log(f"开始分析测试视频：{source_snapshot.name}，对象：{test_user.name}")
        self._show_toast("操作已生效")
        self._launch_analysis_worker(
            source=str(source_snapshot),
            description_text=description_text,
            output_dir=output_dir,
            analysis_mode="test",
            baseline_payload=baseline_snapshot,
        )

    def _launch_analysis_worker(
        self,
        *,
        source: str,
        description_text: str,
        output_dir: Path,
        analysis_mode: str,
        baseline_payload: dict | None,
    ) -> None:
        """启动后台视频分析线程。"""
        if analysis_mode not in {"template", "test"}:
            QMessageBox.warning(self, "提示", f"未知分析模式：{analysis_mode}")
            return
        if analysis_mode == "template":
            baseline_payload = None
        validation_message = self._validate_analysis_inputs(
            source=Path(source),
            analysis_mode=analysis_mode,
            baseline_payload=baseline_payload,
        )
        if validation_message:
            QMessageBox.warning(self, "提示", validation_message)
            return
        self.running_analysis_mode = analysis_mode
        self.running_source = source
        self.running_baseline_payload = dict(baseline_payload) if isinstance(baseline_payload, dict) else None
        self._analysis_start_time = time.time()
        self._last_live_suggestions_text = ""
        self._last_live_suggestions_frame = -1
        self._last_added_suggestion = ""
        self.statusBar().showMessage("正在分析中，请稍候...")
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_chip.setText("分析中")
        self.header_hint_label.setText("系统正在执行姿态分析，请观察双画面关键点识别情况")
        self.header_ratio_label.setText("模板匹配度 --   偏差指数 --")
        self.task_status_label.setText("当前任务：分析中")
        self.analysis_worker = AnalysisWorker(
            source=source,
            description_text=description_text,
            weights_path=self.current_weights_path,
            template_file=str(DEFAULT_TEMPLATE_FILE),
            output_dir=output_dir,
            analysis_mode=analysis_mode,
            baseline_payload=self.running_baseline_payload,
            evaluator_level=self._current_evaluator_level(),
        )
        self.analysis_worker.progress_changed.connect(self._handle_analysis_progress)
        self.analysis_worker.finished_success.connect(self._handle_analysis_success)
        self.analysis_worker.finished_error.connect(self._handle_analysis_error)
        self.analysis_worker.finished.connect(self._on_analysis_worker_finished)
        # [修复分析完成后自动退出]
        # 线程 finished 只做 deleteLater，不再级联任何关闭主窗口的动作。
        self.analysis_worker.finished.connect(self.analysis_worker.deleteLater)
        self.analysis_worker.start()

    def _safe_int(self, value: object, default: int = 0) -> int:
        try:
            if value is None:
                return default
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def _resolve_suggestion_text_edit(self):
        for name in (
            "suggestion_text",
            "error_text",
            "issues_text_edit",
            "suggestion_text_edit",
            "suggestions_text_edit",
            "problem_text_edit",
            "advice_text_edit",
        ):
            widget = getattr(self, name, None)
            if widget is not None and (hasattr(widget, "setPlainText") or hasattr(widget, "setHtml")):
                return widget
        return None

    def _show_toast(self, message: str) -> None:
        self.statusBar().showMessage(message, 2000)

    def _format_seconds_to_mm_ss(self, seconds: object) -> str:
        try:
            total_seconds = max(0, int(float(seconds)))
        except (TypeError, ValueError):
            total_seconds = 0
        return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"

    def _set_live_feedback_text(self, widget, html_text: str, plain_text: str) -> None:
        if widget is None:
            return
        if hasattr(widget, "setHtml"):
            widget.setHtml(html_text)
            return
        if hasattr(widget, "setMarkdown"):
            widget.setMarkdown(plain_text)
            return
        if hasattr(widget, "setPlainText"):
            widget.setPlainText(plain_text)

    def _update_live_suggestions(self, payload: dict) -> None:
        issues_raw = payload.get("issues", [])
        suggestions_raw = payload.get("suggestions", [])
        live_feedback = payload.get("live_feedback") if isinstance(payload.get("live_feedback"), dict) else {}
        issues = [
            str(item).strip()
            for item in (issues_raw if isinstance(issues_raw, list) else [])
            if str(item).strip()
        ]
        suggestions = [
            _clean_display_text(item)
            for item in (suggestions_raw if isinstance(suggestions_raw, list) else [])
            if _clean_display_text(item)
        ]
        phase_label = _clean_display_text(payload.get("phase_label") or "未识别阶段") or "未识别阶段"
        phase_guidance = _clean_display_text(payload.get("phase_guidance") or "")
        action_name = _clean_display_text(live_feedback.get("action_name") or "当前动作") or "当前动作"
        focus_summary = _clean_display_text(live_feedback.get("focus_summary") or "")
        manifestation = _clean_display_text(live_feedback.get("manifestation") or "")
        possible_cause = _clean_display_text(live_feedback.get("possible_cause") or "")
        adjustment_method = _clean_display_text(live_feedback.get("adjustment_method") or "")
        training_focus = _clean_display_text(live_feedback.get("training_focus") or "")
        detail_items = [
            _clean_display_text(item)
            for item in (live_feedback.get("detail_items") if isinstance(live_feedback.get("detail_items"), list) else [])
            if _clean_display_text(item)
        ]
        metric_lines = [
            _clean_display_text(item)
            for item in (live_feedback.get("metric_lines") if isinstance(live_feedback.get("metric_lines"), list) else [])
            if _clean_display_text(item)
        ]
        if isinstance(suggestions_raw, list):
            raw_suggestion_items = [_clean_display_text(item) for item in suggestions_raw if _clean_display_text(item)]
        else:
            raw_suggestion_items = []
        new_suggestion_text = " ".join(raw_suggestion_items).strip()
        if new_suggestion_text and new_suggestion_text != getattr(self, "_last_added_suggestion", ""):
            fps_value = float(payload.get("fps", 30) or 30.0)
            processed_frames = self._safe_int(payload.get("processed_frames"), 0)
            time_str = self._format_seconds_to_mm_ss(processed_frames / max(1.0, fps_value))
            log_lines = [new_suggestion_text]
            if adjustment_method:
                log_lines.append(f"调整方法：{adjustment_method}")
            if training_focus:
                log_lines.append(training_focus)
            if metric_lines:
                log_lines.append(" | ".join(metric_lines))
            log_html = f"<b>[{time_str}]</b> " + "<br>".join(escape(line) for line in log_lines if line.strip())
            if hasattr(self, "live_suggestion_log") and self.live_suggestion_log is not None:
                self.live_suggestion_log.append(log_html)
                scrollbar = self.live_suggestion_log.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
            self._last_added_suggestion = new_suggestion_text

        issue_plain = "\n".join(f"- {item}" for item in issues) if issues else "- 当前未发现明显实时问题。"
        suggestion_items = list(suggestions)
        if phase_guidance:
            suggestion_items.append(f"【{phase_label}】{phase_guidance}")
        suggestion_items.extend(item for item in detail_items if item not in suggestion_items)
        suggestion_plain = (
            "\n".join(f"- {item}" for item in suggestion_items)
            if suggestion_items else
            "- 当前暂无新的优化建议，请继续保持动作稳定。"
        )
        combined_plain = (
            f"定位分析\n动作：{action_name}\n当前阶段：{phase_label}\n"
            f"{focus_summary or '请结合当前阶段关注动作路线与身体控制。'}\n"
            f"问题表现：{manifestation or '当前未发现明显实时问题。'}\n"
            f"可能原因：{possible_cause or '请继续观察动作细节。'}\n"
            f"调整方法：{adjustment_method or suggestion_plain}\n"
            f"训练重点：{training_focus or '继续保持动作稳定。'}\n"
            + ("\n".join(metric_lines) if metric_lines else "")
        )
        frame_index = self._safe_int(payload.get("frame_index"), 0)
        should_refresh = combined_plain != self._last_live_suggestions_text or frame_index % 15 == 0
        if not should_refresh:
            return
        issue_html_parts = [
            "<h3>实时定位分析</h3>",
            f"<p><b>动作：</b>{escape(action_name)}<br>",
            f"<b>当前阶段：</b>{escape(phase_label)}</p>",
        ]
        if focus_summary:
            issue_html_parts.append(f"<p><b>阶段重点：</b>{escape(focus_summary)}</p>")
        issue_html_parts.append(
            "<div style='margin-top:6px;padding:10px 12px;background:#f8fbfe;border:1px solid #dbe7f1;border-radius:10px;'>"
            f"<b>问题表现：</b>{escape(manifestation or '当前未发现明显实时问题。')}<br>"
            f"<b>可能原因：</b>{escape(possible_cause or '请继续观察动作细节。')}<br>"
            f"<b>调整方法：</b>{escape(adjustment_method or '请继续保持动作稳定。')}<br>"
            f"<b>训练重点：</b>{escape(training_focus or '继续保持动作稳定。')}"
            "</div>"
        )
        if issues:
            issue_html_parts.append("<h4>即时问题标签</h4><ul>")
            issue_html_parts.extend(f"<li>{escape(item)}</li>" for item in issues)
            issue_html_parts.append("</ul>")
        if metric_lines:
            issue_html_parts.append("<h4>当前关键指标</h4><ul>")
            issue_html_parts.extend(f"<li>{escape(item)}</li>" for item in metric_lines)
            issue_html_parts.append("</ul>")
        issue_html = "".join(issue_html_parts)
        suggestion_html_parts = [
            "<h3>实时优化建议</h3>",
            f"<p><b>动作：</b>{escape(action_name)}<br><b>当前阶段：</b>{escape(phase_label)}</p>",
        ]
        if suggestions:
            suggestion_html_parts.append("<ul>")
            suggestion_html_parts.extend(f"<li>{escape(item)}</li>" for item in suggestions)
            suggestion_html_parts.append("</ul>")
        else:
            suggestion_html_parts.append("<p>当前暂无新的优化建议，请继续保持动作稳定。</p>")
        if phase_guidance:
            suggestion_html_parts.append("<h4>阶段训练提示</h4>")
            suggestion_html_parts.append(f"<p>{escape(phase_guidance)}</p>")
        if detail_items:
            suggestion_html_parts.append("<h4>动作细化建议</h4><ul>")
            suggestion_html_parts.extend(f"<li>{escape(item)}</li>" for item in detail_items)
            suggestion_html_parts.append("</ul>")
        suggestion_html = "".join(suggestion_html_parts)

        error_widget = getattr(self, "error_text", None)
        suggestion_widget = getattr(self, "suggestion_text", None)
        if error_widget is not None:
            self._set_live_feedback_text(error_widget, issue_html, issue_plain)
        if suggestion_widget is not None:
            self._set_live_feedback_text(suggestion_widget, suggestion_html, suggestion_plain)
        if error_widget is None and suggestion_widget is None:
            widget = self._resolve_suggestion_text_edit()
            self._set_live_feedback_text(widget, suggestion_html, combined_plain)
        self._last_live_suggestions_text = combined_plain
        self._last_live_suggestions_frame = frame_index

    def _handle_analysis_progress(self, payload: dict) -> None:
        """处理分析进度并刷新界面。"""
        frame = payload.get("frame")
        if isinstance(frame, np.ndarray):
            if self.running_analysis_mode == "template":
                self.current_reference_frame = frame
                self._set_preview_image(self.reference_preview, frame)
            else:
                self.current_test_frame = frame
                self._set_preview_image(self.test_preview, frame)
            self._refresh_overlay_preview(payload)
        processed = self._safe_int(payload.get("processed_frames"), 0)
        total = self._safe_int(payload.get("total_frames"), 0)
        percent = int(processed * 100 / total) if total > 0 else min(99, processed % 100)
        progress = self._safe_int(payload.get("progress"), percent)
        phase = str(payload.get("phase_label") or "未识别阶段")
        pose_backend = _clean_display_text(payload.get("pose_backend") or self._last_pose_backend_reason or "")
        if 0 < progress < 100 and self._analysis_start_time is not None:
            elapsed_real = max(0.0, time.time() - self._analysis_start_time)
            eta_sec = max(0, int(elapsed_real / max(progress / 100.0, 1e-6) - elapsed_real))
            eta_str = f"{eta_sec // 60:02d}:{eta_sec % 60:02d}"
        else:
            eta_str = "--:--"
        self.statusBar().showMessage(
            f"进度: {progress}% | 剩余时间: {eta_str} | 当前阶段: {phase}"
            + (f" | 姿态后端: {pose_backend}" if pose_backend else "")
        )
        self.progress_bar.setValue(percent)
        self._update_live_suggestions(payload)
        self.header_hint_label.setText(f"当前阶段：{phase}")
        self.task_status_label.setText(f"当前任务：正在分析 {processed}/{total or '--'} 帧")
        self.camera_status_label.setText(
            f"实时状态：FPS {float(payload.get('fps') or 0.0):.1f} | 阶段 {phase} | 识别质量 {payload.get('pose_quality_label') or '--'}"
            + (f" | 后端 {pose_backend}" if pose_backend else "")
        )
        issues = payload.get("issues") or []
        self.center_status_text.setHtml(
            "<h3>实时分析状态</h3>"
            f"<p><b>当前阶段：</b>{escape(phase)}<br>"
            f"<b>处理进度：</b>{processed} / {total or '--'}<br>"
            f"<b>FPS：</b>{float(payload.get('fps') or 0.0):.1f}<br>"
            f"<b>识别质量：</b>{escape(str(payload.get('pose_quality_label') or '--'))}<br>"
            f"<b>关键点数量：</b>{payload.get('visible_keypoint_count') or 0}<br>"
            f"<b>姿态后端：</b>{escape(pose_backend or '--')}</p>"
            + (
                "<h3>实时问题</h3><ul>"
                + "".join(f"<li>{escape(str(issue))}</li>" for issue in issues[:4])
                + "</ul>"
                if issues
                else "<p>当前未发现明显实时问题。</p>"
            )
        )

    def _handle_analysis_success(self, result) -> None:
        """处理分析完成结果。"""
        finished_mode = self.running_analysis_mode
        finished_test_index = self.running_test_user_index
        self.running_analysis_mode = None
        self.running_test_user_index = None
        self.running_source = None
        self.running_baseline_payload = None
        self.stop_button.setEnabled(False)
        self.status_chip.setText("已完成")
        self._set_active_step(4)
        self.progress_bar.setValue(100)
        self.header_hint_label.setText("分析完成，可查看规范等级、偏差定位和整改建议")
        self.task_status_label.setText("当前任务：分析完成")
        self.current_result = result
        self._populate_result_panels(result)
        self._update_output_buttons(result.artifacts)
        self._append_log(f"分析完成，输出目录：{result.output_dir}")
        if result.analysis_mode == "template":
            baseline_path = Path(result.artifacts["template_baseline_json"])
            payload = self._load_json(baseline_path)
            payload["模板名称"] = self.template_name_edit.text().strip() or payload.get("模板名称") or "未命名模板"
            save_template_result(result, payload, TEMPLATE_LIBRARY_FILE)
            self.current_baseline_path = baseline_path
            self.load_template_to_dropdown(selected_baseline_path=baseline_path)
            # [模板闭环] 模板生成后自动设为当前标准模板，测试页可直接进入“开始分析”。
            self.current_baseline_payload = payload
            self._select_template_by_baseline_path(baseline_path)
            self.baseline_combo.setToolTip(str(baseline_path))
            self._set_active_step(3)
            if hasattr(self, "test_user_list"):
                self.test_user_list.setFocus()
            self.header_hint_label.setText("模板基线已生成并自动加载，可继续选择测试视频开始分析")
            self._append_log(f"已自动加载新模板基线：{baseline_path}")
        elif finished_mode == "test" and finished_test_index is not None and finished_test_index < len(self.test_users):
            test_user = self.test_users[finished_test_index]
            test_user.result = result
            test_user.status = "已完成"
            test_user.last_output_dir = Path(result.output_dir)
            test_user.last_analyzed_at = datetime.now().strftime("%H:%M:%S")
            if not test_user.baseline_name:
                test_user.baseline_name = self._current_baseline_name()
            self._refresh_test_user_item(finished_test_index)
            if self.current_test_user_index == finished_test_index:
                self.result_subject_label.setText(format_test_user_subject(test_user))
            self._export_prepost_summary_for_all_tests()
        elapsed = self._format_elapsed_time()
        self.statusBar().showMessage(f"分析完成，用时 {elapsed}")

    def _select_template_by_baseline_path(self, baseline_path: Path) -> None:
        """在模板/基线下拉框中选中新生成的模板。"""
        normalized = str(baseline_path.resolve())
        for combo in (self.template_combo, self.baseline_combo):
            combo.blockSignals(True)
            try:
                for index in range(combo.count()):
                    entry = combo.itemData(index)
                    if not isinstance(entry, dict):
                        continue
                    entry_path = entry.get("baseline_json_path")
                    if entry_path and str(Path(entry_path).resolve()) == normalized:
                        combo.setCurrentIndex(index)
                        break
            finally:
                combo.blockSignals(False)

    def _handle_analysis_error(self, message: str) -> None:
        """处理分析失败。"""
        failed_mode = self.running_analysis_mode
        failed_test_index = self.running_test_user_index
        self.running_analysis_mode = None
        self.running_test_user_index = None
        self.running_source = None
        self.running_baseline_payload = None
        self.stop_button.setEnabled(False)
        self.status_chip.setText("失败")
        self.header_hint_label.setText("分析失败，请检查输入素材、模型文件或摄像头状态")
        self.task_status_label.setText("当前任务：执行失败")
        if failed_mode == "test" and failed_test_index is not None and failed_test_index < len(self.test_users):
            self.test_users[failed_test_index].status = "失败"
            self._refresh_test_user_item(failed_test_index)
        elapsed = self._format_elapsed_time()
        self.statusBar().showMessage(f"分析失败，已结束，本次耗时 {elapsed}")
        QMessageBox.warning(self, "错误", f"分析失败：{message}")
        self._append_log(f"分析失败：{message}")

    def _on_analysis_worker_finished(self) -> None:
        """只在线程真正 finished 后再释放 Python 引用，避免 QThread 运行中被析构。"""
        self.analysis_worker = None

    def _refresh_overlay_preview(self, payload: dict) -> None:
        """刷新双骨架叠加预览。"""
        base = None
        if self.current_reference_frame is not None:
            base = self.current_reference_frame.copy()
        elif self.current_test_frame is not None:
            base = self.current_test_frame.copy()
        if base is None:
            return
        canvas = np.full((420, 760, 3), 252, dtype=np.uint8)
        if self.current_reference_frame is not None:
            left = _fit_frame(self.current_reference_frame, 360, 200)
            canvas[20:220, 20:380] = left
        if self.current_test_frame is not None:
            right = _fit_frame(self.current_test_frame, 360, 200)
            canvas[20:220, 400:760] = right
        baseline_points = {}
        mapped_points = payload.get("baseline_keypoint_positions") or {}
        if mapped_points:
            baseline_points = mapped_points
        elif self.current_baseline_payload:
            rows = self.current_baseline_payload.get("原始逐帧数据") or []
            if rows:
                baseline_points = rows[min(len(rows) - 1, max(0, int(payload.get("processed_frames") or 1) - 1))].get("keypoint_positions") or {}
        current_points = payload.get("keypoint_positions") or {}
        overlay = self._draw_skeleton_compare(canvas, baseline_points, current_points)
        if not baseline_points:
            overlay = _draw_text_cn(overlay, "请先导入标准动作模板", (24, 250), color=(20, 45, 180), font_size=24)
        if not self._pose_backend_available:
            overlay = _draw_text_cn(
                overlay,
                self._last_pose_backend_reason or "当前为光流兜底模式，无骨架可视化",
                (24, 285),
                color=(0, 0, 220),
                font_size=24,
            )
        self.current_overlay_frame = overlay
        self._set_preview_image(self.overlay_preview, overlay)

    def _draw_skeleton_compare(
        self,
        canvas: np.ndarray,
        baseline_points: dict[str, list[float]],
        current_points: dict[str, list[float]],
    ) -> np.ndarray:
        """绘制模板和测试者骨架对比。"""
        painted = canvas.copy()
        left_offset = (20, 20)
        right_offset = (400, 20)
        self._draw_named_skeleton(painted, baseline_points, left_offset, (50, 180, 90), (50, 180, 90))
        self._draw_named_skeleton(painted, current_points, right_offset, (80, 180, 80), (0, 0, 255), baseline_points)
        comparable = [name for name in COCO_KEYPOINTS if name in baseline_points and name in current_points]
        marked = 0
        for name in comparable:
            base_x, base_y = baseline_points[name][:2]
            cur_x, cur_y = current_points[name][:2]
            if not np.isfinite(np.array([base_x, base_y, cur_x, cur_y], dtype=np.float32)).all():
                continue
            diff = float(np.linalg.norm(np.array([base_x - cur_x, base_y - cur_y], dtype=np.float32)))
            if diff < 24.0:
                continue
            marked += 1
            point = (int(round(cur_x * 0.45)) + right_offset[0], int(round(cur_y * 0.45)) + right_offset[1])
            cv2.circle(painted, point, 9, (0, 0, 255), 2, cv2.LINE_AA)
            painted = _draw_text_cn(painted, f"{name}:{diff:.1f}", (point[0] + 10, point[1] - 8), color=(220, 30, 30), font_size=14)
        painted = _draw_text_cn(painted, f"红色偏差关键点：{marked}", (420, 380), color=(220, 30, 30), font_size=18)
        return painted

    def _draw_named_skeleton(
        self,
        frame: np.ndarray,
        points: dict[str, list[float]],
        offset: tuple[int, int],
        normal_color: tuple[int, int, int],
        alert_color: tuple[int, int, int],
        baseline_points: dict[str, list[float]] | None = None,
    ) -> None:
        """根据关键点名称绘制骨架。"""
        if not points:
            return
        scale = 0.45
        for start_idx, end_idx in SKELETON_EDGES:
            start_name = COCO_KEYPOINTS[start_idx]
            end_name = COCO_KEYPOINTS[end_idx]
            if start_name not in points or end_name not in points:
                continue
            sx, sy = points[start_name][:2]
            ex, ey = points[end_name][:2]
            if not np.isfinite(np.array([sx, sy, ex, ey], dtype=np.float32)).all():
                continue
            start = (int(round(sx * scale)) + offset[0], int(round(sy * scale)) + offset[1])
            end = (int(round(ex * scale)) + offset[0], int(round(ey * scale)) + offset[1])
            cv2.line(frame, start, end, normal_color, 2, cv2.LINE_AA)
        for name, point in points.items():
            if len(point) < 2 or not np.isfinite(np.array(point[:2], dtype=np.float32)).all():
                continue
            x = int(round(point[0] * scale)) + offset[0]
            y = int(round(point[1] * scale)) + offset[1]
            color = normal_color
            if baseline_points and name in baseline_points:
                if not np.isfinite(np.array(baseline_points[name][:2], dtype=np.float32)).all():
                    continue
                diff = float(np.linalg.norm(np.array(points[name][:2], dtype=np.float32) - np.array(baseline_points[name][:2], dtype=np.float32)))
                if diff >= 24.0:
                    color = alert_color
            cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)

    def _populate_result_panels(self, result) -> None:
        """刷新右侧结果区。"""
        populate_result_widgets(self, result)

    def _ensure_visual_artifact(self, result) -> None:
        """生成对比可视化图片。"""
        if self.current_overlay_frame is None:
            return
        target = Path(result.output_dir) / "07_对比可视化图片.jpg"
        cv2.imwrite(str(target), self.current_overlay_frame)
        result.artifacts["comparison_visual_image"] = str(target)
        self._update_output_buttons(result.artifacts)

    def _update_output_buttons(self, artifacts: dict[str, str]) -> None:
        """根据文件生成状态更新导出按钮。"""
        for key, button in self.output_buttons.items():
            path_text = artifacts.get(key)
            exists = bool(path_text and Path(path_text).exists())
            button.setEnabled(exists)

    def _export_prepost_summary_for_all_tests(self) -> None:
        rows: list[dict] = []
        output_path: Path | None = None
        for test_user in self.test_users:
            result = test_user.result
            if result is None:
                continue
            output_path = Path(result.output_dir) / "15_前后测改善量总表.xlsx"
            rows.append(
                build_prepost_metric_row(
                    result,
                    subject_id=test_user.name,
                    test_type=test_user.test_type or "未标记",
                    baseline_name=test_user.baseline_name or "",
                )
            )
        if not rows or output_path is None:
            return
        export_prepost_summary_xlsx(rows, output_path)
        if self.current_result is not None:
            self.current_result.artifacts["prepost_summary_xlsx"] = str(output_path)
            self._update_output_buttons(self.current_result.artifacts)

    def _refresh_template_library(self) -> None:
        """刷新模板库下拉框。"""
        self.template_entries = list_template_entries(TEMPLATE_LIBRARY_FILE)
        self.template_combo.blockSignals(True)
        self.baseline_combo.blockSignals(True)
        self.template_combo.clear()
        self.baseline_combo.clear()
        self.template_combo.addItem("请选择模板", None)
        self.baseline_combo.addItem("请选择模板", None)
        for entry in self.template_entries:
            label = f"{entry.get('template_name', '未命名模板')} · {entry.get('updated_at', '')}"
            self.template_combo.addItem(label, entry)
            self.baseline_combo.addItem(label, entry)
        self.template_combo.blockSignals(False)
        self.baseline_combo.blockSignals(False)

    def load_template_to_dropdown(self, selected_baseline_path: Path | None = None) -> None:
        """刷新模板下拉并自动选中新生成模板。"""
        self._refresh_template_library()
        if selected_baseline_path is not None:
            self._select_template_by_baseline_path(selected_baseline_path)

    def _on_template_selected(self) -> None:
        """加载选中的模板基线。"""
        combo = self.sender()
        if not isinstance(combo, QComboBox):
            combo = self.baseline_combo
        entry = combo.currentData()
        if not entry:
            # [修复状态污染] 下拉框回到“请选择模板”时清空旧基线，测试页不能沿用上一轮模板。
            self.current_baseline_payload = None
            self.current_baseline_path = None
            return
        baseline_path = entry.get("baseline_json_path")
        if not baseline_path or not Path(baseline_path).exists():
            QMessageBox.warning(self, "提示", "请先加载姿态权重文件")
            return
        try:
            baseline_path_obj = Path(baseline_path)
            self.current_baseline_payload = self._load_json(baseline_path_obj)
            self.current_baseline_path = baseline_path_obj
            self.template_name_edit.setText(entry.get("template_name") or self.template_name_edit.text())
            self._append_log(f"已加载模板基线：{entry.get('template_name')}")
            snapshots = self.current_baseline_payload.get("阶段快照") or []
            if snapshots:
                image_path = snapshots[0].get("image_path")
                if image_path and Path(image_path).exists():
                    frame = cv2.imread(str(image_path))
                    if frame is not None:
                        self.current_reference_frame = frame
                        self._set_preview_image(self.reference_preview, frame)
        except Exception as exc:
            QMessageBox.warning(self, "错误", f"模板基线加载失败：{exc}")

    def _load_local_baseline_json(self) -> None:
        """手动加载本地基线 JSON。"""
        path, _ = QFileDialog.getOpenFileName(self, "选择模板基线 JSON", str(PROJECT_ROOT), "JSON 文件 (*.json)")
        if not path:
            return
        try:
            self.current_baseline_payload = self._load_json(Path(path))
            self.current_baseline_path = Path(path)
            self.baseline_combo.setToolTip(path)
            self._append_log(f"已加载本地 JSON：{path}")
        except Exception as exc:
            QMessageBox.warning(self, "错误", f"加载失败：{exc}")

    def _delete_selected_template(self) -> None:
        """删除模板库条目。"""
        entry = self.template_combo.currentData()
        if not entry:
            return
        if QMessageBox.question(self, "确认", f"确认删除模板“{entry.get('template_name')}”吗？") != QMessageBox.Yes:
            return
        deleted = delete_template_entry(entry.get("id"), TEMPLATE_LIBRARY_FILE)
        if deleted:
            self._append_log(f"已删除模板：{entry.get('template_name')}")
            self._refresh_template_library()

    def _record_template_from_camera(self) -> None:
        """用摄像头录制标准示范，停止后挂载为模板素材。"""
        self.camera_record_purpose = "template"
        self._open_camera_dialog(record_mode=True)

    def _record_test_from_camera(self) -> None:
        """用摄像头录制待测者动作，停止后加入测试列表。"""
        self.camera_record_purpose = "test"
        self._open_camera_dialog(record_mode=True)

    def _open_camera_dialog(self, record_mode: bool = False) -> None:
        """打开摄像头控制窗口，由用户手动点击开始。"""
        if self.camera_combo.currentData() is None:
            QMessageBox.warning(self, "提示", "请先扫描并选择摄像头设备")
            return
        if not record_mode:
            self.camera_record_purpose = None
        if self.camera_dialog is None:
            self.camera_dialog = CameraDialog(self)
            self.camera_dialog.stop_requested.connect(self._stop_camera_analysis)
            self.camera_dialog.start_requested.connect(self._start_camera_analysis)
            self.camera_dialog.start_record_requested.connect(lambda: self._start_camera_analysis(enable_record=True))
        if record_mode:
            self.camera_dialog.start_button.setEnabled(False)
            self.camera_dialog.record_button.setEnabled(True)
            self.camera_dialog.hint_label.setText("当前为摄像头录制流程，请点击“实时分析并录制”，停止后会自动回填素材。")
        else:
            self.camera_dialog.start_button.setEnabled(True)
            self.camera_dialog.record_button.setEnabled(True)
            self.camera_dialog.hint_label.setText("先确认摄像头编号与人物站位，再手动点击“开始分析”或“实时分析并录制”。")
        self.camera_dialog.show()
        self.camera_dialog.raise_()
        self.camera_dialog.activateWindow()
        if record_mode:
            purpose_text = "模板" if self.camera_record_purpose == "template" else "测试视频"
            self.header_hint_label.setText(f"摄像头录制窗口已打开，请点击“实时分析并录制”生成{purpose_text}")
            self.header_ratio_label.setText("等待录制启动")
            self._append_log(f"摄像头{purpose_text}录制窗口已打开。")
        else:
            self.header_hint_label.setText("摄像头控制窗口已打开，请在新窗口中手动点击开始分析")
            self.header_ratio_label.setText("等待手动启动")
            self._append_log("摄像头控制窗口已打开，等待手动开始分析。")

    def _start_camera_analysis(self, enable_record: bool = False) -> None:
        """在摄像头控制窗口中启动实时分析。"""
        if self.camera_worker is not None:
            QMessageBox.warning(self, "提示", "摄像头分析已在运行中。")
            return

        if self.camera_combo.currentData() is None:
            QMessageBox.warning(self, "提示", "未检测到摄像头设备")
            return

        description = self._compose_analysis_description(
            self.template_desc_edit.toPlainText().strip() if self.camera_record_purpose == "template" else self.test_desc_edit.toPlainText().strip(),
            self.template_terms_edit.toPlainText().strip() if self.camera_record_purpose == "template" else self.test_terms_edit.toPlainText().strip(),
        )
        project_combo = self.template_project_combo if self.camera_record_purpose == "template" else self.test_project_combo
        description = self._inject_project_type(description, project_combo.currentText())
        if not description:
            description = "摄像头标准模板录制" if self.camera_record_purpose == "template" else "摄像头实时动作分析"

        if self.camera_dialog is None:
            self.camera_dialog = CameraDialog(self)
            self.camera_dialog.stop_requested.connect(self._stop_camera_analysis)
            self.camera_dialog.start_requested.connect(self._start_camera_analysis)
            self.camera_dialog.start_record_requested.connect(lambda: self._start_camera_analysis(enable_record=True))
            self.camera_dialog.show()
        self._set_active_step(3)
        self.current_camera_record_path = None
        if enable_record:
            record_prefix = "camera_template" if self.camera_record_purpose == "template" else "camera_test"
            output_dir = build_output_dir(record_prefix)
            self.current_output_dir = output_dir
            self.current_camera_record_path = output_dir / (
                "01_摄像头模板录制.mp4" if self.camera_record_purpose == "template" else "01_实时分析录制.mp4"
            )

        self.camera_worker = CameraWorker(
            camera_index=int(self.camera_combo.currentData()),
            description_text=description,
            weights_path=self.current_weights_path,
            template_file=str(DEFAULT_TEMPLATE_FILE),
            evaluator_level=self._current_evaluator_level(),
            baseline_payload=None if self.camera_record_purpose == "template" else self.current_baseline_payload,
            target_fps=20,
            record_output_path=self.current_camera_record_path,
        )
        self.camera_worker.frame_changed.connect(self._handle_camera_payload)
        self.camera_worker.finished_success.connect(self._handle_camera_success)
        self.camera_worker.finished_error.connect(self._handle_camera_error)
        self.camera_worker.finished.connect(self._on_camera_worker_finished)
        self.camera_worker.finished.connect(self.camera_worker.deleteLater)
        self.camera_worker.start()
        self.stop_button.setEnabled(True)
        self.status_chip.setText("摄像头分析中")
        self.header_hint_label.setText("摄像头实时分析已启动，正在持续刷新动作状态")
        self.header_ratio_label.setText("实时模式")
        self.camera_status_label.setText("设备状态：摄像头启动中")
        if self.camera_dialog is not None:
            self.camera_dialog.start_button.setEnabled(False)
            self.camera_dialog.stop_button.setEnabled(True)
            self.camera_dialog.record_button.setEnabled(False)
        if enable_record and self.camera_record_purpose == "template":
            self._append_log("摄像头标准模板录制已启动")
        elif enable_record:
            self._append_log("摄像头实时分析并录制已启动")
        else:
            self._append_log("摄像头实时分析已启动")

    def _handle_camera_payload(self, payload: dict) -> None:
        """刷新摄像头状态。"""
        if self.camera_dialog is not None:
            self.camera_dialog.update_payload(payload)
        frame = payload.get("frame")
        if isinstance(frame, np.ndarray):
            if self.camera_record_purpose == "template":
                self.current_reference_frame = frame
                self._set_preview_image(self.reference_preview, frame)
            else:
                self.current_test_frame = frame
                self._set_preview_image(self.test_preview, frame)
            self._refresh_overlay_preview(payload)
        phase = payload.get("phase_label") or "分析中"
        pose_backend = _clean_display_text(payload.get("pose_backend") or self._last_pose_backend_reason or "")
        issues = payload.get("issues") or []
        self._update_live_suggestions(payload)
        self.header_hint_label.setText(f"实时阶段：{phase}")
        self.center_status_text.setHtml(
            "<h3>实时摄像头状态</h3>"
            f"<p><b>设备：</b>摄像头 {payload.get('camera_index', '--')}<br>"
            f"<b>阶段：</b>{phase}<br>"
            f"<b>FPS：</b>{float(payload.get('fps') or payload.get('camera_fps') or 0.0):.1f}<br>"
            f"<b>姿态质量：</b>{payload.get('pose_quality_label') or '--'}<br>"
            f"<b>姿态后端：</b>{pose_backend or '--'}</p>"
            + (
                "<h3>实时提醒</h3><ul>"
                + "".join(f"<li>{issue}</li>" for issue in issues[:4])
                + "</ul>"
                if issues
                else "<p>暂未识别到显著错误动作。</p>"
            )
        )
        self.camera_status_label.setText(
            f"设备状态：运行中 · FPS {float(payload.get('fps') or payload.get('camera_fps') or 0.0):.1f} · AI {payload.get('pose_quality_label') or '--'}"
            + (f" · 后端 {pose_backend}" if pose_backend else "")
        )

    def _handle_camera_success(self, message: str) -> None:
        """摄像头线程正常结束。"""
        record_path = self.current_camera_record_path
        record_purpose = self.camera_record_purpose
        if self.camera_dialog is not None:
            self.camera_dialog.start_button.setEnabled(True)
            self.camera_dialog.stop_button.setEnabled(False)
            self.camera_dialog.record_button.setEnabled(True)
            self.camera_dialog.close()
            self.camera_dialog = None
        self.current_camera_record_path = None
        self.camera_record_purpose = None
        self.stop_button.setEnabled(False)
        self.status_chip.setText("空闲")
        self._set_active_step(1)
        self.header_hint_label.setText("摄像头分析已停止，可继续导入素材或重新开启实时分析")
        self.header_ratio_label.setText("实时模式结束")
        self.camera_status_label.setText("设备状态：未启动")
        self._append_log(message)
        if record_path is not None and record_path.exists():
            self._handle_camera_record_file(record_path, record_purpose)

    def _handle_camera_error(self, message: str) -> None:
        """摄像头线程异常结束。"""
        if self.camera_dialog is not None:
            self.camera_dialog.start_button.setEnabled(True)
            self.camera_dialog.stop_button.setEnabled(False)
            self.camera_dialog.record_button.setEnabled(True)
            self.camera_dialog.close()
            self.camera_dialog = None
        self.current_camera_record_path = None
        self.camera_record_purpose = None
        self.stop_button.setEnabled(False)
        self.status_chip.setText("失败")
        self.header_hint_label.setText("摄像头分析启动失败，请检查设备连接和权限")
        self.camera_status_label.setText("设备状态：异常")
        QMessageBox.warning(self, "错误", f"摄像头启动失败：{message}\n请检查设备连接和权限")
        self._append_log(f"摄像头异常：{message}")

    def _on_camera_worker_finished(self) -> None:
        """只在线程真正 finished 后再释放 Python 引用，避免 QThread 运行中被析构。"""
        self.camera_worker = None

    def _stop_camera_analysis(self) -> None:
        """停止摄像头分析。"""
        if self.camera_worker is not None:
            self.camera_worker.request_stop()
            self._wait_thread_safely(self.camera_worker, timeout_ms=8000)

    def _handle_camera_record_file(self, record_path: Path, purpose: str | None) -> None:
        """将摄像头录制文件回填到模板或测试流程。"""
        if purpose == "template":
            self.current_template_source = record_path
            self.template_name_edit.setText(record_path.stem.replace("01_", "") or "摄像头模板")
            if not self.template_desc_edit.toPlainText().strip():
                self.template_desc_edit.setPlainText(f"{record_path.stem} 标准动作模板，来源：摄像头录制。")
            self._load_video_preview(record_path, target="reference")
            self._set_active_step(1)
            self._append_log(f"摄像头录制已挂载为模板素材：{record_path}")
            QMessageBox.information(self, "录制完成", "摄像头模板录制已完成，可点击“生成标准模板基线”。")
            return
        if purpose == "test":
            self._add_test_user_from_video(record_path)
            self._set_active_step(2)
            self._append_log(f"摄像头录制已加入待测用户：{record_path}")
            QMessageBox.information(self, "录制完成", "摄像头测试视频已加入待测用户列表，可加载模板后开始分析。")

    def _stop_running_task(self) -> None:
        """停止当前任务。"""
        if self.analysis_worker is not None:
            self.analysis_worker.request_stop()
            self._wait_thread_safely(self.analysis_worker, timeout_ms=8000)
            self._append_log("已请求停止当前分析任务。")
        if self.camera_worker is not None:
            self.camera_worker.request_stop()
            self._wait_thread_safely(self.camera_worker, timeout_ms=8000)
            self._append_log("已请求停止摄像头分析。")
        self.stop_button.setEnabled(False)
        self.status_chip.setText("空闲")
        self._set_active_step(1)
        self.header_hint_label.setText("任务已停止，可重新选择模板、测试视频或摄像头")
        self.header_ratio_label.setText("模板匹配度 --   偏差指数 --")

    def _wait_thread_safely(self, worker: QThread, timeout_ms: int = 8000) -> bool:
        """[线程守护] 等待 QThread 完整退出，避免窗口关闭时线程对象仍在运行。"""
        if worker.isRunning():
            worker.quit()
        deadline = datetime.now().timestamp() + timeout_ms / 1000.0
        while worker.isRunning() and datetime.now().timestamp() < deadline:
            QApplication.processEvents()
            worker.wait(100)
        return not worker.isRunning()

    def _format_elapsed_time(self) -> str:
        if self._analysis_start_time is None:
            return "0分0秒"
        elapsed = max(0, int(round(time.time() - self._analysis_start_time)))
        self._analysis_start_time = None
        minutes, seconds = divmod(elapsed, 60)
        return f"{minutes}分{seconds}秒"

    def _copy_summary(self) -> None:
        """复制当前标签页内容。"""
        current_widget = self.result_tabs.currentWidget()
        if isinstance(current_widget, QTextBrowser):
            QApplication.clipboard().setText(current_widget.toPlainText())

    def _open_artifact(self, artifact_key: str) -> None:
        """打开导出文件。"""
        if self.current_result is None:
            return
        path_text = self.current_result.artifacts.get(artifact_key)
        if not path_text:
            QMessageBox.warning(self, "提示", "当前文件尚未生成。")
            return
        path = Path(path_text)
        if not path.exists():
            QMessageBox.warning(self, "提示", "当前文件已不存在，按钮状态已刷新。")
            self._update_output_buttons(self.current_result.artifacts)
            return
        _safe_open_path(path)

    def _open_output_dir(self) -> None:
        """打开输出目录。"""
        self.current_output_dir.mkdir(parents=True, exist_ok=True)
        self._show_toast("操作已生效")
        _safe_open_path(self.current_output_dir)

    def _toggle_log_area(self) -> None:
        """折叠/展开日志区域。"""
        visible = self.log_text.isVisible()
        self.log_text.setVisible(not visible)
        self.toggle_log_btn.setText("展开" if visible else "折叠")

    def _toggle_left_panel(self) -> None:
        """折叠/展开左侧控制区。"""
        self._set_left_panel_visible(not self.left_panel_visible)

    def _set_left_panel_visible(self, visible: bool) -> None:
        """更新左侧控制区显隐状态。"""
        self.left_panel_visible = visible
        self.left_scroll.setVisible(visible)
        self.toggle_left_panel_btn.setText("◂" if visible else "▸")
        self.toggle_left_panel_btn.setToolTip("隐藏左侧控制区" if visible else "展开左侧控制区")

    def _toggle_right_panel(self) -> None:
        """折叠/展开右侧结果区。"""
        self._set_right_panel_visible(not self.right_panel_visible)

    def _set_right_panel_visible(self, visible: bool) -> None:
        """更新右侧结果区显隐状态。"""
        self.right_panel_visible = visible
        self.right_scroll.setVisible(visible)
        self.toggle_right_panel_btn.setText("▸" if visible else "◂")
        self.toggle_right_panel_btn.setToolTip("隐藏右侧结果区" if visible else "展开右侧结果区")

    def _append_log(self, message: str) -> None:
        """写入右侧日志。"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{timestamp}] {message}")

    def _reset_preview_area(self) -> None:
        """重置中间预览。"""
        self.reference_preview.setText("请先导入标准动作模板")
        self.reference_preview.setPixmap(QPixmap())
        self.test_preview.setText("等待测试画面")
        self.test_preview.setPixmap(QPixmap())
        self.overlay_preview.setText("双骨架叠加对比窗口")
        self.overlay_preview.setPixmap(QPixmap())
        self.current_reference_frame = None
        self.current_test_frame = None
        self.current_overlay_frame = None

    def _set_speed(self, speed_text: str) -> None:
        """更新播放速度显示。"""
        self.playback_speed = speed_text
        self._append_log(f"播放速度切换为 {speed_text}")

    def _load_json(self, path: Path) -> dict:
        """读取 JSON 文件。"""
        if not path.exists():
            raise FileNotFoundError(f"JSON 文件不存在：{path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 格式非法：{path}，{exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"JSON 顶层必须是对象：{path}")
        return payload

    def _start_template_analysis(self) -> None:
        """启动标准模板分析。"""
        if self.analysis_worker is not None:
            QMessageBox.warning(self, "提示", "当前已有分析任务正在执行。")
            return
        if self.current_template_source is None or not self.current_template_source.exists():
            QMessageBox.warning(self, "提示", "请先导入模板素材。")
            return
        template_name = self.template_name_edit.text().strip()
        if not template_name:
            QMessageBox.warning(self, "提示", "请先填写模板名称。")
            return
        validation_message = self._validate_analysis_inputs(
            source=self.current_template_source,
            analysis_mode="template",
            baseline_payload=None,
        )
        if validation_message:
            QMessageBox.warning(self, "提示", validation_message)
            return
        description_text = self._compose_analysis_description(
            self.template_desc_edit.toPlainText().strip() or f"{template_name} 标准动作模板",
            self.template_terms_edit.toPlainText().strip(),
        )
        description_text = self._inject_project_type(description_text, self.template_project_combo.currentText())
        output_dir = build_output_dir("template")
        self.current_output_dir = output_dir
        self._set_active_step(2)
        self._append_log(f"开始生成模板基线：{template_name}")
        self._show_toast("操作已执行")
        self._launch_analysis_worker(
            source=str(self.current_template_source),
            description_text=description_text,
            output_dir=output_dir,
            analysis_mode="template",
            baseline_payload=None,
        )

    def _load_local_baseline_json(self) -> None:
        """手动加载本地基线 JSON。"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模板基线 JSON",
            str(PROJECT_ROOT),
            "JSON 文件 (*.json)",
        )
        if not path:
            return
        try:
            self.current_baseline_payload = self._load_json(Path(path))
            self.current_baseline_path = Path(path)
            self.baseline_combo.setToolTip(path)
            self._append_log(f"已加载本地 JSON：{path}")
        except Exception as exc:
            QMessageBox.warning(self, "错误", f"加载失败：{exc}")

    def _validate_analysis_inputs(
        self,
        *,
        source: Path,
        analysis_mode: str,
        baseline_payload: dict | None,
    ) -> str | None:
        if analysis_mode not in {"template", "test"}:
            return f"未知分析模式：{analysis_mode}"
        if source is None or not source.exists():
            return f"输入源不存在：{source}"
        if not source.is_file():
            return f"输入源不是文件：{source}"
        model_name = Path(self.current_weights_path).name or "yolov8n-pose.pt"
        weights_path = self._resolve_weights_candidate(model_name)
        self.current_weights_path = str(weights_path)
        if not weights_path.exists() or not weights_path.is_file():
            return f"姿态权重文件不存在：{weights_path}"
        if analysis_mode == "test":
            if baseline_payload is None:
                return "请先加载标准模板基线。"
            if not isinstance(baseline_payload, dict):
                return "模板基线 JSON 顶层必须是对象。"
            rows = baseline_payload.get("原始逐帧数据")
            if not isinstance(rows, list) or not rows:
                return "模板基线 JSON 缺少有效的“原始逐帧数据”列表。"
        return None

    def closeEvent(self, event) -> None:  # noqa: N802
        """关闭主窗口前安全回收线程。"""
        for worker_name in ("analysis_worker", "camera_worker"):
            worker = getattr(self, worker_name, None)
            if worker is None:
                continue
            if hasattr(worker, "request_stop"):
                worker.request_stop()
            worker.quit()
            if not worker.wait(2000):
                self.statusBar().showMessage("后台线程仍在退出中，已取消关闭")
                event.ignore()
                return
            setattr(self, worker_name, None)
        event.accept()
        super().closeEvent(event)


def run_qt_gui(*, app_name: str = APP_NAME, app_version: str = APP_VERSION) -> int:
    """启动桌面 GUI。"""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(app_name.strip() or APP_NAME)
    safe_font = QFont("Microsoft YaHei UI")
    safe_point_size = safe_font.pointSize()
    safe_font.setPointSize(10 if safe_point_size is None or safe_point_size <= 0 else max(10, safe_point_size))
    app.setFont(safe_font)
    window = MotionAnalysisQtWindow(app_name=app_name, app_version=app_version)
    window.show()
    return app.exec()


__all__ = ["MotionAnalysisQtWindow", "run_qt_gui"]
