"""
简化版三栏布局GUI - 专注于核心功能实现
这是一个参考实现，展示如何实现第三章和第四章的需求
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import QThread, Qt, Signal, QTimer
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QGroupBox,
)

from .app_identity import APP_NAME, APP_VERSION
from .config import DEFAULT_WEIGHTS, OUTPUTS_DIR, YOLO_POSE_MODELS
from .desktop_backend import (
    VideoAnalyzeWorker,
    CameraScanWorker,
    RealtimeCameraWorker,
    build_output_dir,
    load_baseline_payload,
)
from .pose import COCO_KEYPOINTS, SKELETON_EDGES, draw_pose
from .template_library import list_template_entries, delete_template_entry


class ThreeColumnMainWindow(QMainWindow):
    """三栏布局主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.setMinimumSize(1280, 720)
        self.resize(1440, 900)

        # 状态变量
        self.current_weights = str(DEFAULT_WEIGHTS)
        self.current_template_baseline = None
        self.current_result = None
        self.worker = None
        self.camera_worker = None
        self.available_cameras = []

        # 构建UI
        self._build_ui()
        self._apply_styles()

    def _build_ui(self):
        """构建三栏布局界面"""
        # 中心部件
        central = QWidget()
        self.setCentralWidget(central)

        # 主布局（水平三栏）
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ========== 左侧控制区（320px固定宽度）==========
        left_panel = self._build_left_panel()
        left_panel.setFixedWidth(320)
        main_layout.addWidget(left_panel)

        # ========== 中间对比展示区（自适应宽度）==========
        center_panel = self._build_center_panel()
        main_layout.addWidget(center_panel, 1)  # stretch=1

        # ========== 右侧数据结果区（400px固定宽度）==========
        right_panel = self._build_right_panel()
        right_panel.setFixedWidth(400)
        main_layout.addWidget(right_panel)

        # ========== 底部状态栏 ==========
        self._build_status_bar()

    def _build_left_panel(self) -> QWidget:
        """构建左侧控制区"""
        panel = QFrame()
        panel.setObjectName("leftPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 标题
        title = QLabel("📋 控制面板")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #165DFF;")
        layout.addWidget(title)

        # 双页签：模板用户 / 测试用户
        self.left_tabs = QTabWidget()
        self.left_tabs.addTab(self._build_template_tab(), "📋 模板用户")
        self.left_tabs.addTab(self._build_test_tab(), "🧪 测试用户")
        layout.addWidget(self.left_tabs)

        layout.addStretch()
        return panel

    def _build_template_tab(self) -> QWidget:
        """构建模板用户页签"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # 导入模板数据
        group1 = QGroupBox("导入模板数据")
        group1_layout = QVBoxLayout(group1)

        btn_import_video = QPushButton("📹 导入视频")
        btn_import_video.clicked.connect(self._import_template_video)
        group1_layout.addWidget(btn_import_video)

        btn_import_image = QPushButton("🖼️ 导入图片")
        btn_import_image.clicked.connect(self._import_template_image)
        group1_layout.addWidget(btn_import_image)

        btn_import_excel = QPushButton("📊 导入Excel")
        btn_import_excel.clicked.connect(self._import_template_excel)
        group1_layout.addWidget(btn_import_excel)

        btn_import_text = QPushButton("✏️ 文本输入")
        btn_import_text.clicked.connect(self._import_template_text)
        group1_layout.addWidget(btn_import_text)

        layout.addWidget(group1)

        # 模板名称
        group2 = QGroupBox("模板信息")
        group2_layout = QVBoxLayout(group2)

        self.template_name_input = QLineEdit()
        self.template_name_input.setPlaceholderText("请输入模板名称（如：八段锦标准动作）")
        group2_layout.addWidget(QLabel("模板名称："))
        group2_layout.addWidget(self.template_name_input)

        self.btn_generate_template = QPushButton("🔍 分析生成模板基线")
        self.btn_generate_template.clicked.connect(self._generate_template_baseline)
        group2_layout.addWidget(self.btn_generate_template)

        layout.addWidget(group2)

        # 已保存模板
        group3 = QGroupBox("已保存模板")
        group3_layout = QVBoxLayout(group3)

        self.template_list_combo = QComboBox()
        self.template_list_combo.addItem("-- 选择模板 --")
        group3_layout.addWidget(self.template_list_combo)

        btn_refresh = QPushButton("🔄 刷新列表")
        btn_refresh.clicked.connect(self._refresh_template_list)
        group3_layout.addWidget(btn_refresh)

        btn_delete = QPushButton("🗑️ 删除选中")
        btn_delete.clicked.connect(self._delete_template)
        group3_layout.addWidget(btn_delete)

        layout.addWidget(group3)

        layout.addStretch()
        return widget

    def _build_test_tab(self) -> QWidget:
        """构建测试用户页签"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # 测试数据源
        group1 = QGroupBox("测试数据源")
        group1_layout = QVBoxLayout(group1)

        self.btn_import_test_video = QPushButton("📹 导入视频文件")
        self.btn_import_test_video.clicked.connect(self._import_test_video)
        group1_layout.addWidget(self.btn_import_test_video)

        self.btn_start_camera = QPushButton("📷 启动摄像头实时分析")
        self.btn_start_camera.clicked.connect(self._start_camera_analysis)
        group1_layout.addWidget(self.btn_start_camera)

        layout.addWidget(group1)

        # 加载模板基线
        group2 = QGroupBox("加载模板基线")
        group2_layout = QVBoxLayout(group2)

        self.baseline_combo = QComboBox()
        self.baseline_combo.addItem("-- 选择模板基线 --")
        group2_layout.addWidget(QLabel("选择模板："))
        group2_layout.addWidget(self.baseline_combo)

        btn_load_local = QPushButton("📂 或加载本地JSON")
        btn_load_local.clicked.connect(self._load_local_baseline)
        group2_layout.addWidget(btn_load_local)

        layout.addWidget(group2)

        # 分析参数配置
        group3 = QGroupBox("分析参数配置")
        group3_layout = QVBoxLayout(group3)

        group3_layout.addWidget(QLabel("算法精度："))
        self.precision_combo = QComboBox()
        for model in YOLO_POSE_MODELS[:3]:  # 只显示前3个常用模型
            self.precision_combo.addItem(model["label"], model["name"])
        group3_layout.addWidget(self.precision_combo)

        group3_layout.addWidget(QLabel("适用人群："))
        self.population_combo = QComboBox()
        self.population_combo.addItems(["成人", "儿童", "老人"])
        group3_layout.addWidget(self.population_combo)

        group3_layout.addWidget(QLabel("熟练度："))
        self.skill_combo = QComboBox()
        self.skill_combo.addItems(["初学者", "中级", "专业运动员"])
        group3_layout.addWidget(self.skill_combo)

        layout.addWidget(group3)

        # 开始分析按钮
        self.btn_start_analysis = QPushButton("▶️ 开始分析")
        self.btn_start_analysis.setStyleSheet("""
            QPushButton {
                background-color: #165DFF;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4080FF;
            }
        """)
        self.btn_start_analysis.clicked.connect(self._start_analysis)
        layout.addWidget(self.btn_start_analysis)

        layout.addStretch()
        return widget

    def _build_center_panel(self) -> QWidget:
        """构建中间对比展示区"""
        panel = QFrame()
        panel.setObjectName("centerPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 标题
        title = QLabel("📊 对比展示区")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # 播放控制栏
        control_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶️ 播放")
        self.btn_pause = QPushButton("⏸️ 暂停")
        self.btn_reset = QPushButton("🔄 重置")
        self.btn_speed = QComboBox()
        self.btn_speed.addItems(["1x", "0.5x", "2x"])

        control_layout.addWidget(self.btn_play)
        control_layout.addWidget(self.btn_pause)
        control_layout.addWidget(self.btn_reset)
        control_layout.addWidget(QLabel("倍速："))
        control_layout.addWidget(self.btn_speed)
        control_layout.addStretch()

        layout.addLayout(control_layout)

        # 双画面对比区
        video_layout = QHBoxLayout()

        # 标准模板画面
        template_frame = QFrame()
        template_frame.setFrameShape(QFrame.Box)
        template_layout = QVBoxLayout(template_frame)
        template_layout.addWidget(QLabel("标准模板动作"))
        self.template_video_label = QLabel()
        self.template_video_label.setMinimumSize(400, 300)
        self.template_video_label.setStyleSheet("background-color: #F7F8FA;")
        self.template_video_label.setAlignment(Qt.AlignCenter)
        self.template_video_label.setText("等待加载模板...")
        template_layout.addWidget(self.template_video_label)

        # 测试用户画面
        test_frame = QFrame()
        test_frame.setFrameShape(QFrame.Box)
        test_layout = QVBoxLayout(test_frame)
        test_layout.addWidget(QLabel("测试用户动作"))
        self.test_video_label = QLabel()
        self.test_video_label.setMinimumSize(400, 300)
        self.test_video_label.setStyleSheet("background-color: #F7F8FA;")
        self.test_video_label.setAlignment(Qt.AlignCenter)
        self.test_video_label.setText("等待导入测试视频...")
        test_layout.addWidget(self.test_video_label)

        video_layout.addWidget(template_frame)
        video_layout.addWidget(test_frame)

        layout.addLayout(video_layout)

        # 双骨架叠加对比窗口
        skeleton_frame = QFrame()
        skeleton_frame.setFrameShape(QFrame.Box)
        skeleton_layout = QVBoxLayout(skeleton_frame)
        skeleton_layout.addWidget(QLabel("双骨架叠加对比"))
        self.skeleton_compare_label = QLabel()
        self.skeleton_compare_label.setMinimumHeight(200)
        self.skeleton_compare_label.setStyleSheet("background-color: #F7F8FA;")
        self.skeleton_compare_label.setAlignment(Qt.AlignCenter)
        self.skeleton_compare_label.setText("请先导入标准动作模板和测试视频")
        skeleton_layout.addWidget(self.skeleton_compare_label)

        layout.addWidget(skeleton_frame)

        layout.addStretch()
        return panel

    def _build_right_panel(self) -> QWidget:
        """构建右侧数据结果区"""
        panel = QFrame()
        panel.setObjectName("rightPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 标题
        title = QLabel("📈 数据结果")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # 评分卡片
        score_group = QGroupBox("📊 动作评分")
        score_layout = QVBoxLayout(score_group)

        self.score_total_label = QLabel("总分：-- / 100")
        self.score_total_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #165DFF;")
        score_layout.addWidget(self.score_total_label)

        self.score_rating_label = QLabel("评级：--")
        score_layout.addWidget(self.score_rating_label)

        score_layout.addWidget(QLabel("分项评分："))
        self.score_arm_label = QLabel("手臂动作：-- / 100")
        self.score_leg_label = QLabel("下盘稳定：-- / 100")
        self.score_core_label = QLabel("核心姿态：-- / 100")
        self.score_flow_label = QLabel("动作连贯性：-- / 100")

        score_layout.addWidget(self.score_arm_label)
        score_layout.addWidget(self.score_leg_label)
        score_layout.addWidget(self.score_core_label)
        score_layout.addWidget(self.score_flow_label)

        layout.addWidget(score_group)

        # 研判标签页
        self.result_tabs = QTabWidget()
        self.result_tabs.addTab(self._create_text_widget(), "研判摘要")
        self.result_tabs.addTab(self._create_text_widget(), "模板差异")
        self.result_tabs.addTab(self._create_text_widget(), "错误定位")
        self.result_tabs.addTab(self._create_text_widget(), "修改建议")

        layout.addWidget(self.result_tabs)

        # 输出文件按钮区
        output_group = QGroupBox("📁 导出文件")
        output_layout = QVBoxLayout(output_group)

        self.btn_export_excel = QPushButton("📊 Excel分析报表")
        self.btn_export_excel.setEnabled(False)
        self.btn_export_excel.clicked.connect(lambda: self._open_output_file("excel"))
        output_layout.addWidget(self.btn_export_excel)

        self.btn_export_csv = QPushButton("📈 逐帧关键点数据CSV")
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.clicked.connect(lambda: self._open_output_file("csv"))
        output_layout.addWidget(self.btn_export_csv)

        self.btn_export_json = QPushButton("📉 指标汇总JSON")
        self.btn_export_json.setEnabled(False)
        self.btn_export_json.clicked.connect(lambda: self._open_output_file("json"))
        output_layout.addWidget(self.btn_export_json)

        self.btn_export_image = QPushButton("🖼️ 对比可视化图片")
        self.btn_export_image.setEnabled(False)
        self.btn_export_image.clicked.connect(lambda: self._open_output_file("image"))
        output_layout.addWidget(self.btn_export_image)

        layout.addWidget(output_group)

        layout.addStretch()
        return panel

    def _create_text_widget(self) -> QWidget:
        """创建文本显示组件"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlaceholderText("分析完成后将显示结果...")
        layout.addWidget(text_edit)
        return widget

    def _build_status_bar(self):
        """构建底部状态栏"""
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(16, 8, 16, 8)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - 空闲")

        self.status_label = QLabel("当前任务：空闲")

        self.btn_stop = QPushButton("⏹️ 停止分析")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_analysis)

        self.btn_open_output = QPushButton("📂 打开输出目录")
        self.btn_open_output.clicked.connect(self._open_output_dir)

        status_layout.addWidget(self.progress_bar, 2)
        status_layout.addWidget(self.status_label, 1)
        status_layout.addWidget(self.btn_stop)
        status_layout.addWidget(self.btn_open_output)

        self.statusBar().addPermanentWidget(status_widget, 1)

    def _apply_styles(self):
        """应用样式表"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FFFFFF;
            }
            QFrame#leftPanel, QFrame#rightPanel {
                background-color: #F7F8FA;
                border-right: 1px solid #E5E6EB;
            }
            QFrame#centerPanel {
                background-color: #FFFFFF;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #E5E6EB;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #C9CDD4;
                border-radius: 4px;
                padding: 6px 12px;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #F2F3F5;
                border-color: #165DFF;
            }
            QPushButton:pressed {
                background-color: #E5E6EB;
            }
            QPushButton:disabled {
                background-color: #F7F8FA;
                color: #C9CDD4;
            }
            QLineEdit, QComboBox {
                border: 1px solid #C9CDD4;
                border-radius: 4px;
                padding: 6px;
                background-color: #FFFFFF;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #165DFF;
            }
        """)

    # ========== 槽函数实现 ==========

    def _import_template_video(self):
        """导入模板视频"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模板视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            QMessageBox.information(self, "成功", f"已导入模板视频：\n{file_path}")
            # TODO: 实际处理逻辑

    def _import_template_image(self):
        """导入模板图片"""
        QMessageBox.information(self, "提示", "图片导入功能开发中...")

    def _import_template_excel(self):
        """导入模板Excel"""
        QMessageBox.information(self, "提示", "Excel导入功能开发中...")

    def _import_template_text(self):
        """文本输入"""
        QMessageBox.information(self, "提示", "文本输入功能开发中...")

    def _generate_template_baseline(self):
        """生成模板基线"""
        template_name = self.template_name_input.text().strip()
        if not template_name:
            QMessageBox.warning(self, "警告", "请先输入模板名称")
            return
        QMessageBox.information(self, "提示", f"开始生成模板基线：{template_name}")
        # TODO: 实际处理逻辑

    def _refresh_template_list(self):
        """刷新模板列表"""
        self.template_list_combo.clear()
        self.template_list_combo.addItem("-- 选择模板 --")
        # TODO: 从template_library加载
        QMessageBox.information(self, "提示", "模板列表已刷新")

    def _delete_template(self):
        """删除选中模板"""
        current = self.template_list_combo.currentText()
        if current == "-- 选择模板 --":
            QMessageBox.warning(self, "警告", "请先选择要删除的模板")
            return
        # TODO: 实际删除逻辑

    def _import_test_video(self):
        """导入测试视频"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择测试视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            QMessageBox.information(self, "成功", f"已导入测试视频：\n{file_path}")
            # TODO: 实际处理逻辑

    def _start_camera_analysis(self):
        """启动摄像头实时分析"""
        QMessageBox.information(self, "提示", "摄像头实时分析功能开发中...")
        # TODO: 实际处理逻辑

    def _load_local_baseline(self):
        """加载本地基线JSON"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模板基线JSON", "", "JSON文件 (*.json)"
        )
        if file_path:
            try:
                self.current_template_baseline = load_baseline_payload(file_path)
                QMessageBox.information(self, "成功", "模板基线加载成功")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载失败：{e}")

    def _start_analysis(self):
        """开始分析"""
        if not self.current_template_baseline:
            QMessageBox.warning(self, "警告", "请先加载模板基线")
            return
        QMessageBox.information(self, "提示", "开始分析...")
        # TODO: 实际处理逻辑

    def _stop_analysis(self):
        """停止分析"""
        if self.worker:
            self.worker.request_stop()
        if self.camera_worker:
            self.camera_worker.request_stop()

    def _open_output_dir(self):
        """打开输出目录"""
        import subprocess
        import platform
        output_dir = str(OUTPUTS_DIR)
        if platform.system() == "Windows":
            subprocess.Popen(f'explorer "{output_dir}"')
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", output_dir])
        else:
            subprocess.Popen(["xdg-open", output_dir])

    def _open_output_file(self, file_type: str):
        """打开输出文件"""
        QMessageBox.information(self, "提示", f"打开{file_type}文件...")


def run_simple_gui() -> int:
    """启动简化版GUI"""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setFont(QFont("Microsoft YaHei UI", 10))

    window = ThreeColumnMainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(run_simple_gui())
