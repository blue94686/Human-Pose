from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSplitter, QTextBrowser, QVBoxLayout, QWidget


def build_center_panel(window, parent_layout: QVBoxLayout) -> None:
    stage_note = QLabel("标准动作与待测用户双画面对照，叠加骨架关键点与偏差提示")
    stage_note.setObjectName("StageHint")
    parent_layout.addWidget(stage_note)

    # [布局重构] 使用 QSplitter 将视频播放区与双骨骼/实时状态并排展示，避免对比窗口被挤到页面底部。
    split_view = QSplitter(Qt.Horizontal)
    split_view.setObjectName("AnalysisSplitter")
    split_view.setChildrenCollapsible(False)
    video_column = QWidget()
    video_column.setObjectName("AnalysisColumn")
    video_layout = QVBoxLayout(video_column)
    video_layout.setContentsMargins(0, 0, 0, 0)
    video_layout.setSpacing(14)
    compare_column = QWidget()
    compare_column.setObjectName("AnalysisColumn")
    compare_layout = QVBoxLayout(compare_column)
    compare_layout.setContentsMargins(0, 0, 0, 0)
    compare_layout.setSpacing(14)

    window.reference_preview = QLabel("请先导入标准动作模板")
    window.reference_preview.setAlignment(Qt.AlignCenter)
    window.reference_preview.setMinimumHeight(320)
    window.reference_preview.setObjectName("StagePreview")
    window.test_preview = QLabel("等待测试画面")
    window.test_preview.setAlignment(Qt.AlignCenter)
    window.test_preview.setMinimumHeight(320)
    window.test_preview.setObjectName("StagePreview")
    video_layout.addWidget(
        window._create_video_stage("标准八段锦示范画面，叠加绿色骨架关键点", "标准模板动作", window.reference_preview),
        1,
    )
    video_layout.addWidget(
        window._create_video_stage("待测用户动作视频画面，叠加红色识别骨骼与偏差提示", "待测用户动作", window.test_preview),
        1,
    )
    window.overlay_preview = QLabel("双骨架叠加对比窗口")
    window.overlay_preview.setAlignment(Qt.AlignCenter)
    window.overlay_preview.setMinimumHeight(360)
    window.overlay_preview.setObjectName("StagePreview")
    compare_layout.addWidget(
        window._create_video_stage("经骨架复合视图，双骨骼叠加对比画布", "双骨骼叠加对比", window.overlay_preview),
        3,
    )

    live_box = window._create_card("关键偏差 / 实时状态")
    live_layout = QVBoxLayout(live_box)
    window.center_status_text = QTextBrowser()
    window.center_status_text.setMinimumHeight(240)
    window.center_status_text.setHtml("<p>等待分析开始，右侧会同步输出结构化诊断结果。</p>")
    live_layout.addWidget(window.center_status_text)
    compare_layout.addWidget(live_box, 2)
    split_view.addWidget(video_column)
    split_view.addWidget(compare_column)
    split_view.setSizes([620, 620])
    parent_layout.addWidget(split_view, 1)


__all__ = ["build_center_panel"]
