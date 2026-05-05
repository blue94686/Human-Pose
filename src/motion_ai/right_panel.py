from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QToolTip,
    QVBoxLayout,
)


def _build_collapsible_section(parent_layout: QVBoxLayout, title: str) -> QFrame:
    card = QFrame()
    card.setObjectName("CollapsibleSection")
    card_layout = QVBoxLayout(card)
    card_layout.setContentsMargins(0, 0, 0, 0)
    card_layout.setSpacing(0)

    header = QPushButton(f"▼ {title}")
    header.setObjectName("SectionToggle")
    header.setCheckable(True)
    header.setChecked(True)
    header.setMinimumHeight(46)
    card_layout.addWidget(header)

    body = QFrame()
    body.setObjectName("SectionBody")
    card_layout.addWidget(body)

    def toggle_body(checked: bool) -> None:
        body.setVisible(checked)
        header.setText(f"{'▼' if checked else '▶'} {title}")

    header.toggled.connect(toggle_body)
    parent_layout.addWidget(card)
    return body


_METRIC_HELP = {
    "综合得分": "综合姿态标准度、稳定性、连贯性、节奏控制和完整性后的总评得分。",
    "规范等级": "将综合得分映射为完美、良好、一般、偏差等等级，用于快速判断动作规范程度。",
    "分项评估": "分项展示姿态完成度、稳定性、动作连贯性等维度，便于定位训练短板。",
    "时序相似度": "测试动作与标准模板在阶段节奏和关键点轨迹上的时间序列相似程度。",
    "问题数量": "系统识别出的姿态、节奏、对称性和稳定性问题总数。",
    "优先改进": "对总分和安全性影响更大的问题数量，应优先处理。",
    "识别模式": "当前姿态来源，如模型识别、模型加时序跟踪或光流补偿。",
}


def _help_button(text: str) -> QToolButton:
    tooltip = _METRIC_HELP.get(text, "该指标用于辅助解读当前动作分析结果。")
    button = QToolButton()
    button.setText("?")
    button.setObjectName("HelpToolButton")
    button.setToolTip(tooltip)
    button.clicked.connect(lambda: QToolTip.showText(button.mapToGlobal(button.rect().center()), tooltip, button))
    return button


def _metric_title(text: str) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(6)
    label = QLabel(text)
    label.setObjectName("OverviewTitle")
    row.addWidget(label)
    row.addStretch(1)
    row.addWidget(_help_button(text), 0)
    return row


def build_right_panel(window, parent_layout: QVBoxLayout) -> None:
    score_body = _build_collapsible_section(parent_layout, "动作评估 / 规范等级 / 数据导出")
    score_layout = QVBoxLayout(score_body)
    score_layout.setContentsMargins(14, 12, 14, 14)
    score_layout.setSpacing(10)
    window.result_subject_label = QLabel("当前对象：未选择测试用户")
    window.result_subject_label.setObjectName("ResultSubject")
    window.total_score_label = QLabel("--")
    window.total_score_label.setObjectName("ScoreLabel")
    window.grade_label = QLabel("规范等级：待评估")
    window.grade_label.setObjectName("GradeLabel")
    window.sub_scores_label = QLabel("分项评估\n姿态完成度：--\n下盘稳定性：--\n动作连贯性：--\n综合说明：待生成")
    window.sub_scores_label.setWordWrap(True)
    score_layout.addWidget(window.result_subject_label)
    score_layout.addLayout(_metric_title("综合得分"))
    score_layout.addWidget(window.total_score_label)
    score_layout.addLayout(_metric_title("规范等级"))
    score_layout.addWidget(window.grade_label)
    score_layout.addLayout(_metric_title("分项评估"))
    score_layout.addWidget(window.sub_scores_label)

    overview_body = _build_collapsible_section(parent_layout, "关键指标速览")
    overview_layout = QGridLayout(overview_body)
    overview_layout.setContentsMargins(14, 12, 14, 14)
    overview_layout.setHorizontalSpacing(10)
    overview_layout.setVerticalSpacing(10)
    window.overview_cards = {}
    overview_items = [
        ("similarity", "时序相似度"),
        ("issue_count", "问题数量"),
        ("priority_count", "优先改进"),
        ("pose_mode", "识别模式"),
    ]
    for index, (key, title) in enumerate(overview_items):
        card = QFrame()
        card.setObjectName("OverviewCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 10, 10, 10)
        title_label = QLabel(title)
        title_label.setObjectName("OverviewTitle")
        value_label = QLabel("--")
        value_label.setObjectName("OverviewValue")
        value_label.setWordWrap(True)
        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        title_row.addWidget(title_label)
        title_row.addStretch(1)
        title_row.addWidget(_help_button(title), 0)
        card_layout.addLayout(title_row)
        card_layout.addWidget(value_label)
        window.overview_cards[key] = value_label
        overview_layout.addWidget(card, index // 2, index % 2)

    diagnosis_body = _build_collapsible_section(parent_layout, "动作差异分析 / 规范性调整建议")
    diagnosis_layout = QVBoxLayout(diagnosis_body)
    diagnosis_layout.setContentsMargins(14, 12, 14, 14)
    diagnosis_layout.setSpacing(10)
    window.result_tabs = QTabWidget()
    window.result_tabs.setMinimumHeight(280)
    window.summary_text = QTextBrowser()
    window.summary_text.setMinimumHeight(220)
    window.diff_text = QTextBrowser()
    window.diff_text.setMinimumHeight(220)
    window.error_text = QTextBrowser()
    window.error_text.setMinimumHeight(220)
    window.suggestion_text = QTextBrowser()
    window.suggestion_text.setMinimumHeight(220)
    window.result_tabs.addTab(window.summary_text, "综合评定")
    window.result_tabs.addTab(window.diff_text, "指标差异")
    window.result_tabs.addTab(window.error_text, "问题定位")
    window.result_tabs.addTab(window.suggestion_text, "调整建议")

    window.live_suggestion_group = window._create_card(" 实时动作详细优化建议")
    window.live_suggestion_group.setCheckable(True)
    window.live_suggestion_group.setChecked(True)
    live_suggestion_layout = QVBoxLayout(window.live_suggestion_group)
    live_suggestion_layout.setContentsMargins(12, 10, 12, 12)
    live_suggestion_layout.setSpacing(8)
    window.live_suggestion_log = QTextBrowser()
    window.live_suggestion_log.setMinimumHeight(150)
    window.live_suggestion_log.setHtml("<p>等待实时建议流...</p>")
    window.live_suggestion_group.toggled.connect(window.live_suggestion_log.setVisible)
    live_suggestion_layout.addWidget(window.live_suggestion_log)

    action_row = QHBoxLayout()
    copy_btn = QPushButton("复制当前页")
    copy_btn.clicked.connect(window._copy_summary)
    export_txt_btn = QPushButton("导出评估摘要")
    export_txt_btn.clicked.connect(lambda: window._open_artifact("analysis_summary_txt"))
    action_row.addWidget(copy_btn)
    action_row.addWidget(export_txt_btn)
    action_row.addStretch(1)

    diagnosis_layout.addWidget(window.result_tabs)
    diagnosis_layout.addWidget(window.live_suggestion_group)
    diagnosis_layout.addLayout(action_row)

    output_body = _build_collapsible_section(parent_layout, "数据导出")
    output_layout = QGridLayout(output_body)
    output_layout.setContentsMargins(14, 12, 14, 14)
    output_layout.setHorizontalSpacing(10)
    output_layout.setVerticalSpacing(10)
    button_specs = [
        ("excel_report", "Excel分析报表"),
        ("frame_metrics_csv", "逐帧关键点数据CSV"),
        ("compact_data_json", "指标汇总JSON"),
        ("inhibition_metrics_xlsx", "科研指标Excel"),
        ("comparison_visual_image", "对比可视化图片"),
        ("action_guidance_txt", "动作提示文本"),
        ("analysis_summary_txt", "结果摘要TXT"),
        ("prepost_summary_xlsx", "前后测改善量总表"),
    ]
    for index, (key, title) in enumerate(button_specs):
        button = QPushButton(title)
        button.setEnabled(False)
        button.setMinimumHeight(42)
        button.clicked.connect(lambda _checked=False, artifact_key=key: window._open_artifact(artifact_key))
        window.output_buttons[key] = button
        output_layout.addWidget(button, index // 2, index % 2)

    log_body = _build_collapsible_section(parent_layout, "过程记录")
    log_layout = QVBoxLayout(log_body)
    log_layout.setContentsMargins(14, 12, 14, 14)
    log_layout.setSpacing(10)
    log_header = QHBoxLayout()
    log_label = QLabel("过程日志")
    window.toggle_log_btn = QToolButton()
    window.toggle_log_btn.setText("展开")
    window.toggle_log_btn.clicked.connect(window._toggle_log_area)
    log_header.addWidget(log_label)
    log_header.addStretch(1)
    log_header.addWidget(window.toggle_log_btn)
    window.log_text = QPlainTextEdit()
    window.log_text.setReadOnly(True)
    window.log_text.setMinimumHeight(140)
    window.log_text.setVisible(False)
    log_layout.addLayout(log_header)
    log_layout.addWidget(window.log_text)

    parent_layout.setStretch(0, 1)
    parent_layout.setStretch(1, 1)
    parent_layout.setStretch(2, 4)
    parent_layout.setStretch(3, 1)
    parent_layout.setStretch(4, 0)


__all__ = ["build_right_panel"]
