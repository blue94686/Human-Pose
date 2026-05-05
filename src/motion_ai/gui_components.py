"""
简化版GUI界面优化模块

提供更简洁、更易用的界面组件和辅助功能
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class SimpleCard(QFrame):
    """简洁的卡片组件"""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("simpleCard")
        self.setFrameShape(QFrame.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        if title:
            title_label = QLabel(title)
            title_label.setObjectName("cardTitle")
            layout.addWidget(title_label)

        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(8)
        layout.addLayout(self.content_layout)

    def add_widget(self, widget: QWidget):
        """添加组件到卡片"""
        self.content_layout.addWidget(widget)

    def add_layout(self, layout):
        """添加布局到卡片"""
        self.content_layout.addLayout(layout)


class QuickActionButton(QPushButton):
    """快速操作按钮"""

    def __init__(self, text: str, icon_name: str = "", parent=None):
        super().__init__(text, parent)
        self.setObjectName("quickActionButton")
        self.setMinimumHeight(44)

        if icon_name:
            # 可以设置图标
            pass


class OutputFileCard(QFrame):
    """输出文件卡片"""

    def __init__(self, file_name: str, file_path: Path, parent=None):
        super().__init__(parent)
        self.setObjectName("outputFileCard")
        self.file_path = file_path

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        # 文件图标
        icon_label = QLabel("📄")
        icon_label.setFixedSize(24, 24)
        layout.addWidget(icon_label)

        # 文件信息
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        name_label = QLabel(file_name)
        name_label.setObjectName("fileName")
        info_layout.addWidget(name_label)

        if file_path.exists():
            size = file_path.stat().st_size
            size_text = self._format_size(size)
            size_label = QLabel(size_text)
            size_label.setObjectName("fileSize")
            info_layout.addWidget(size_label)

        layout.addLayout(info_layout, 1)

        # 操作按钮
        open_btn = QPushButton("打开")
        open_btn.setObjectName("smallButton")
        open_btn.setFixedWidth(60)
        open_btn.clicked.connect(lambda: self._open_file(file_path))
        layout.addWidget(open_btn)

    def _format_size(self, size: int) -> str:
        """格式化文件大小"""
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        else:
            return f"{size / (1024 * 1024):.1f} MB"

    def _open_file(self, file_path: Path):
        """打开文件"""
        import subprocess
        import sys

        if sys.platform == "win32":
            subprocess.run(["start", "", str(file_path)], shell=True, check=False)
        elif sys.platform == "darwin":
            subprocess.run(["open", str(file_path)], check=False)
        else:
            subprocess.run(["xdg-open", str(file_path)], check=False)


class ScoreCard(QFrame):
    """评分卡片"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("scoreCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # 标题
        title = QLabel("综合评分")
        title.setObjectName("scoreCardTitle")
        layout.addWidget(title)

        # 大号分数
        self.score_label = QLabel("--")
        self.score_label.setObjectName("bigScore")
        self.score_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.score_label)

        # 评价等级
        self.grade_label = QLabel("等待分析")
        self.grade_label.setObjectName("gradeLabel")
        self.grade_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.grade_label)

        # 详细指标
        self.metrics_layout = QVBoxLayout()
        self.metrics_layout.setSpacing(6)
        layout.addLayout(self.metrics_layout)

    def update_score(self, score: float, grade: str):
        """更新评分"""
        self.score_label.setText(f"{score:.1f}")
        self.grade_label.setText(grade)

        # 根据分数设置颜色
        if score >= 90:
            self.score_label.setStyleSheet("color: #167c55;")  # 绿色
        elif score >= 75:
            self.score_label.setStyleSheet("color: #0b5cab;")  # 蓝色
        elif score >= 60:
            self.score_label.setStyleSheet("color: #c46600;")  # 橙色
        else:
            self.score_label.setStyleSheet("color: #bf1d1d;")  # 红色

    def add_metric(self, name: str, value: float):
        """添加详细指标"""
        metric_row = QHBoxLayout()
        metric_row.setSpacing(8)

        name_label = QLabel(name)
        name_label.setObjectName("metricName")
        metric_row.addWidget(name_label, 1)

        value_label = QLabel(f"{value:.1f}")
        value_label.setObjectName("metricValue")
        value_label.setAlignment(Qt.AlignRight)
        metric_row.addWidget(value_label)

        self.metrics_layout.addLayout(metric_row)

    def clear_metrics(self):
        """清空详细指标"""
        while self.metrics_layout.count():
            item = self.metrics_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


class IssueCard(QFrame):
    """问题卡片"""

    def __init__(self, category: str, description: str, severity: str = "minor", parent=None):
        super().__init__(parent)
        self.setObjectName(f"issueCard_{severity}")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        # 严重程度图标
        icon_map = {
            "critical": "🔴",
            "major": "🟡",
            "minor": "🔵",
        }
        icon_label = QLabel(icon_map.get(severity, "🔵"))
        icon_label.setFixedSize(20, 20)
        layout.addWidget(icon_label)

        # 问题信息
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        category_label = QLabel(category)
        category_label.setObjectName("issueCategory")
        info_layout.addWidget(category_label)

        desc_label = QLabel(description)
        desc_label.setObjectName("issueDescription")
        desc_label.setWordWrap(True)
        info_layout.addWidget(desc_label)

        layout.addLayout(info_layout, 1)


def create_simple_stylesheet() -> str:
    """创建简洁的样式表"""
    return """
    /* 卡片样式 */
    QFrame#simpleCard {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
    }

    QLabel#cardTitle {
        font-size: 16px;
        font-weight: bold;
        color: #102033;
    }

    /* 快速操作按钮 */
    QPushButton#quickActionButton {
        background: #0b5cab;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 20px;
        font-size: 14px;
        font-weight: 600;
    }

    QPushButton#quickActionButton:hover {
        background: #094a99;
    }

    QPushButton#quickActionButton:pressed {
        background: #083d7a;
    }

    /* 输出文件卡片 */
    QFrame#outputFileCard {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
    }

    QFrame#outputFileCard:hover {
        background: #eef4fb;
        border-color: #0b5cab;
    }

    QLabel#fileName {
        font-size: 13px;
        font-weight: 600;
        color: #102033;
    }

    QLabel#fileSize {
        font-size: 11px;
        color: #5f6f82;
    }

    QPushButton#smallButton {
        background: #e8f1fb;
        color: #0b5cab;
        border: none;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 600;
    }

    QPushButton#smallButton:hover {
        background: #d0e5f7;
    }

    /* 评分卡片 */
    QFrame#scoreCard {
        background: linear-gradient(135deg, #f8fbff 0%, #fff 100%);
        border: 2px solid #0b5cab;
        border-radius: 16px;
    }

    QLabel#scoreCardTitle {
        font-size: 14px;
        font-weight: 600;
        color: #5f6f82;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    QLabel#bigScore {
        font-size: 56px;
        font-weight: 700;
        color: #0b5cab;
    }

    QLabel#gradeLabel {
        font-size: 16px;
        font-weight: 600;
        color: #5f6f82;
    }

    QLabel#metricName {
        font-size: 13px;
        color: #5f6f82;
    }

    QLabel#metricValue {
        font-size: 15px;
        font-weight: 700;
        color: #102033;
    }

    /* 问题卡片 */
    QFrame#issueCard_critical {
        background: #ffe6e6;
        border-left: 4px solid #bf1d1d;
        border-radius: 8px;
    }

    QFrame#issueCard_major {
        background: #fff5e6;
        border-left: 4px solid #c46600;
        border-radius: 8px;
    }

    QFrame#issueCard_minor {
        background: #e8f1fb;
        border-left: 4px solid #0b5cab;
        border-radius: 8px;
    }

    QLabel#issueCategory {
        font-size: 13px;
        font-weight: 700;
        color: #102033;
    }

    QLabel#issueDescription {
        font-size: 12px;
        color: #5f6f82;
        line-height: 1.5;
    }
    """


class OutputManager:
    """输出管理器 - 统一管理所有输出文件"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.files: dict[str, Path] = {}

    def register_file(self, key: str, file_path: Path):
        """注册输出文件"""
        if file_path.exists():
            self.files[key] = file_path

    def get_file(self, key: str) -> Path | None:
        """获取输出文件"""
        return self.files.get(key)

    def get_all_files(self) -> dict[str, Path]:
        """获取所有输出文件"""
        return self.files.copy()

    def open_file(self, key: str) -> bool:
        """打开指定文件"""
        file_path = self.get_file(key)
        if not file_path or not file_path.exists():
            return False

        import subprocess
        import sys

        try:
            if sys.platform == "win32":
                subprocess.run(["start", "", str(file_path)], shell=True, check=False)
            elif sys.platform == "darwin":
                subprocess.run(["open", str(file_path)], check=False)
            else:
                subprocess.run(["xdg-open", str(file_path)], check=False)
            return True
        except Exception:
            return False

    def open_directory(self) -> bool:
        """打开输出目录"""
        if not self.output_dir.exists():
            return False

        import subprocess
        import sys

        try:
            if sys.platform == "win32":
                subprocess.run(["explorer", str(self.output_dir)], check=False)
            elif sys.platform == "darwin":
                subprocess.run(["open", str(self.output_dir)], check=False)
            else:
                subprocess.run(["xdg-open", str(self.output_dir)], check=False)
            return True
        except Exception:
            return False

    def export_all(self, target_dir: Path) -> list[Path]:
        """批量导出所有文件到指定目录"""
        import shutil

        target_dir.mkdir(parents=True, exist_ok=True)
        exported = []

        for key, file_path in self.files.items():
            if file_path.exists():
                target_path = target_dir / file_path.name
                shutil.copy2(file_path, target_path)
                exported.append(target_path)

        return exported

    def clear(self):
        """清空文件列表"""
        self.files.clear()


def create_quick_action_panel(actions: list[tuple[str, Callable]]) -> QWidget:
    """创建快速操作面板"""
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)

    for text, callback in actions:
        btn = QuickActionButton(text)
        btn.clicked.connect(callback)
        layout.addWidget(btn)

    return panel


def create_output_files_panel(output_manager: OutputManager) -> QWidget:
    """创建输出文件面板"""
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)

    # 标题
    title = QLabel("输出文件")
    title.setObjectName("cardTitle")
    layout.addWidget(title)

    # 文件列表
    files = output_manager.get_all_files()
    if not files:
        empty_label = QLabel("暂无输出文件")
        empty_label.setObjectName("emptyHint")
        empty_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(empty_label)
    else:
        for key, file_path in files.items():
            file_card = OutputFileCard(file_path.name, file_path)
            layout.addWidget(file_card)

    # 批量操作按钮
    if files:
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(8)

        open_dir_btn = QPushButton("打开目录")
        open_dir_btn.setObjectName("smallButton")
        open_dir_btn.clicked.connect(output_manager.open_directory)
        actions_layout.addWidget(open_dir_btn)

        layout.addLayout(actions_layout)

    return panel
