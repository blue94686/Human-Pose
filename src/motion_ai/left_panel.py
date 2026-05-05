from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QToolTip,
    QVBoxLayout,
)


def _build_collapsible_section(window, parent_layout: QVBoxLayout, title: str) -> QVBoxLayout:
    card = QFrame()
    card.setObjectName("CollapsibleSection")
    card_layout = QVBoxLayout(card)
    card_layout.setContentsMargins(0, 0, 0, 0)
    card_layout.setSpacing(0)

    header = QPushButton(f"▾ {title}")
    header.setObjectName("SectionToggle")
    header.setCheckable(True)
    header.setChecked(True)
    header.setMinimumHeight(46)
    card_layout.addWidget(header)

    body = QFrame()
    body.setObjectName("SectionBody")
    body_layout = QVBoxLayout(body)
    body_layout.setContentsMargins(14, 12, 14, 14)
    body_layout.setSpacing(10)
    card_layout.addWidget(body)

    def toggle_body(checked: bool) -> None:
        body.setVisible(checked)
        header.setText(f"{'▾' if checked else '▸'} {title}")

    header.toggled.connect(toggle_body)
    parent_layout.addWidget(card)
    return body_layout


def _build_help_label(text: str, tooltip: str) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(6)
    label = QLabel(text)
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    help_btn = QToolButton()
    help_btn.setText("?")
    help_btn.setObjectName("HelpToolButton")
    help_btn.setToolTip(tooltip)
    help_btn.clicked.connect(lambda: QToolTip.showText(help_btn.mapToGlobal(help_btn.rect().center()), tooltip, help_btn))
    row.addWidget(label)
    row.addWidget(help_btn, 0)
    return row


def build_left_panel(window, parent_layout: QVBoxLayout) -> None:
    device_layout = _build_collapsible_section(window, parent_layout, "设备")
    window.camera_combo = QComboBox()
    window.camera_combo.setPlaceholderText("摄像头选择下拉框")
    window.scan_camera_btn = QPushButton("扫描摄像头")
    window.scan_camera_btn.clicked.connect(window._scan_cameras)
    window.start_camera_btn = window._make_action_button("打开摄像头分析窗口", "打开摄像头实时分析控制窗口", window._open_camera_dialog)
    window.camera_scan_status_label = QLabel("请先扫描摄像头")
    window.camera_scan_status_label.setObjectName("CameraScanStatus")
    window.camera_scan_status_label.setWordWrap(True)
    device_layout.addWidget(window.camera_combo)
    device_layout.addWidget(window.scan_camera_btn)
    device_layout.addWidget(window.camera_scan_status_label)
    device_layout.addWidget(window.start_camera_btn)

    template_layout = _build_collapsible_section(window, parent_layout, "模板用户")
    template_intro = QLabel("导入标准示范素材，生成模板基线。支持视频、图片、Excel 和文字描述。")
    template_intro.setWordWrap(True)
    template_layout.addWidget(template_intro)
    window.import_template_video_btn = window._make_action_button("本地视频导入模板", "导入模板视频", window._pick_template_video)
    window.record_template_camera_btn = window._make_action_button("摄像头录制为模板", "打开摄像头并录制标准示范，停止后自动挂载为模板素材", window._record_template_from_camera)
    window.import_template_image_btn = window._make_action_button("图片导入模板", "导入模板图片", window._pick_template_image)
    window.import_template_excel_btn = window._make_action_button("Excel 导入模板", "导入模板 Excel", window._pick_template_excel)
    window.import_template_text_btn = window._make_action_button("文本描述导入模板", "导入模板文本", window._pick_template_text)
    template_layout.addWidget(window.import_template_video_btn)
    template_layout.addWidget(window.record_template_camera_btn)
    template_layout.addWidget(window.import_template_image_btn)
    template_layout.addWidget(window.import_template_excel_btn)
    template_layout.addWidget(window.import_template_text_btn)
    window.template_name_edit = QLineEdit()
    window.template_name_edit.setPlaceholderText("请输入模板名称")
    window.template_project_combo = QComboBox()
    window.template_project_combo.addItems(["自动识别", "八段锦", "武术", "民族舞", "太极"])
    window.template_desc_edit = QPlainTextEdit()
    window.template_desc_edit.setPlaceholderText("标准动作描述，会在导入后自动补全")
    window.template_desc_edit.setMaximumHeight(92)
    window.template_terms_edit = QPlainTextEdit()
    window.template_terms_edit.setPlaceholderText("可手动补充模板术语/动作词汇，例如：八段锦、沉肩坠肘、立身中正、定势停顿")
    window.template_terms_edit.setMaximumHeight(72)
    template_layout.addWidget(QLabel("模板名称"))
    template_layout.addWidget(window.template_name_edit)
    template_layout.addWidget(QLabel("项目类型"))
    template_layout.addWidget(window.template_project_combo)
    template_layout.addWidget(QLabel("模板动作描述"))
    template_layout.addWidget(window.template_desc_edit)
    template_layout.addWidget(QLabel("模板术语与动作词汇"))
    template_layout.addWidget(window.template_terms_edit)
    window.generate_template_btn = QPushButton("生成标准模板基线")
    window.generate_template_btn.clicked.connect(window._start_template_analysis)
    template_layout.addWidget(window.generate_template_btn)

    window.template_combo = QComboBox()
    window.template_combo.currentIndexChanged.connect(window._on_template_selected)
    window.template_combo.setPlaceholderText("已保存模板")
    template_btn_row = QHBoxLayout()
    refresh_btn = QPushButton("刷新模板库")
    refresh_btn.clicked.connect(window._refresh_template_library)
    delete_btn = QPushButton("删除模板")
    delete_btn.clicked.connect(window._delete_selected_template)
    template_btn_row.addWidget(delete_btn)
    template_btn_row.addWidget(refresh_btn)
    template_layout.addWidget(QLabel("模板库"))
    template_layout.addWidget(window.template_combo)
    template_layout.addLayout(template_btn_row)

    test_layout = _build_collapsible_section(window, parent_layout, "测试用户")
    test_intro = QLabel("导入一个或多个测试视频，加载模板基线后执行动作分析。")
    test_intro.setWordWrap(True)
    test_layout.addWidget(test_intro)
    window.import_test_video_btn = window._make_action_button("导入待测视频", "导入测试视频", window._pick_test_video)
    window.record_test_camera_btn = window._make_action_button("摄像头录制为测试视频", "打开摄像头并录制待测者动作，停止后自动加入测试用户列表", window._record_test_from_camera)
    window.realtime_test_camera_btn = window._make_action_button("摄像头实时评分", "用当前摄像头直接进行待测者实时评分", window._open_camera_dialog)
    window.import_test_video_btn_2 = window._make_action_button("导入更多测试视频", "为其他测试用户追加视频", window._pick_test_video)
    window.test_user_list = QListWidget()
    window.test_user_list.setMinimumHeight(132)
    window.test_user_list.currentRowChanged.connect(window._switch_test_user)
    test_action_row = QGridLayout()
    test_action_row.setHorizontalSpacing(8)
    test_action_row.setVerticalSpacing(8)
    window.remove_test_user_btn = QPushButton("删除对象")
    window.remove_test_user_btn.clicked.connect(window._remove_current_test_user)
    window.rerun_test_user_btn = QPushButton("重新分析")
    window.rerun_test_user_btn.clicked.connect(window._start_test_analysis)
    window.open_test_output_btn = QPushButton("打开结果目录")
    window.open_test_output_btn.clicked.connect(window._open_current_test_output_dir)
    test_action_row.addWidget(window.remove_test_user_btn, 0, 0)
    test_action_row.addWidget(window.rerun_test_user_btn, 0, 1)
    test_action_row.addWidget(window.open_test_output_btn, 1, 0, 1, 2)
    window.baseline_combo = QComboBox()
    window.baseline_combo.setPlaceholderText("下拉选择标准模板")
    window.baseline_combo.currentIndexChanged.connect(window._on_template_selected)
    window.load_json_btn = QPushButton("加载本地 JSON 基线")
    window.load_json_btn.clicked.connect(window._load_local_baseline_json)
    window.test_project_combo = QComboBox()
    window.test_project_combo.addItems(["自动识别", "八段锦", "武术", "民族舞", "太极"])
    window.test_project_combo.currentIndexChanged.connect(window._sync_current_test_user_project_type)
    window.test_type_combo = QComboBox()
    window.test_type_combo.addItems(["未标记", "前测", "后测"])
    window.test_type_combo.currentIndexChanged.connect(window._sync_current_test_user_test_type)
    window.test_desc_edit = QPlainTextEdit()
    window.test_desc_edit.setPlaceholderText("待测用户动作描述，可补充动作目标和限制条件")
    window.test_desc_edit.setMaximumHeight(92)
    window.test_desc_edit.textChanged.connect(window._sync_current_test_user_description)
    window.test_terms_edit = QPlainTextEdit()
    window.test_terms_edit.setPlaceholderText("可手动补充测试动作术语/动作词汇，例如：马步、冲拳、摆臂、转身、节拍点")
    window.test_terms_edit.setMaximumHeight(72)
    window.test_terms_edit.textChanged.connect(window._sync_current_test_user_terminology)
    window.start_test_btn = QPushButton("开始分析待测视频")
    window.start_test_btn.clicked.connect(window._start_test_analysis)
    test_layout.addWidget(window.import_test_video_btn)
    test_layout.addWidget(window.record_test_camera_btn)
    test_layout.addWidget(window.realtime_test_camera_btn)
    test_layout.addWidget(window.import_test_video_btn_2)
    test_layout.addWidget(QLabel("测试用户列表"))
    test_layout.addWidget(window.test_user_list)
    test_layout.addLayout(test_action_row)
    test_layout.addWidget(QLabel("标准对照模板"))
    test_layout.addWidget(window.baseline_combo)
    test_layout.addWidget(window.load_json_btn)
    test_layout.addWidget(QLabel("项目类型"))
    test_layout.addWidget(window.test_project_combo)
    test_layout.addWidget(QLabel("测试类型"))
    test_layout.addWidget(window.test_type_combo)
    test_layout.addWidget(QLabel("待测动作描述"))
    test_layout.addWidget(window.test_desc_edit)
    test_layout.addWidget(QLabel("测试术语与动作词汇"))
    test_layout.addWidget(window.test_terms_edit)
    test_layout.addWidget(window.start_test_btn)

    params_layout = _build_collapsible_section(window, parent_layout, "模型配置")
    window.model_combo = QComboBox()
    window.model_combo.addItems(
        [
            "YOLOv8-n（快速）",
            "YOLOv8-s（平衡）",
            "YOLOv8-m（高精度）",
            "YOLOv8-l（更高精度）",
            "YOLOv8-x（最高精度）",
            "YOLOv8-x-p6（超高分辨率）",
        ]
    )
    window.model_combo.setToolTip("切换推荐精度档位，精度越高速度越慢")
    window.model_combo.currentIndexChanged.connect(window._sync_weights_by_combo)
    window.population_combo = QComboBox()
    window.population_combo.addItems(["成人", "儿童", "老人"])
    window.level_combo = QComboBox()
    window.level_combo.addItems(["初学者", "中级", "专业运动员"])
    window.confidence_combo = QComboBox()
    window.confidence_combo.addItems(["0.5", "0.6", "0.7"])
    params_layout.addLayout(_build_help_label("算法精度", "YOLOv8-Pose 权重档位。n/s 更快，m/l/x 更准但耗时更高。"))
    params_layout.addWidget(window.model_combo)
    params_layout.addLayout(_build_help_label("适用人群", "用于后续阈值解释和报告措辞，不直接改变模型推理结果。"))
    params_layout.addWidget(window.population_combo)
    params_layout.addLayout(_build_help_label("熟练度", "用于调整测试者评价容忍度：初学者更宽松，专业运动员更严格。"))
    params_layout.addWidget(window.level_combo)
    params_layout.addLayout(_build_help_label("置信度阈值", "关键点低于该置信度时会视为不可靠点，并进入插值和平滑管道。"))
    params_layout.addWidget(window.confidence_combo)
    window.pose_notice_label = QLabel("当前为光流兜底模式，无骨架可视化")
    window.pose_notice_label.setObjectName("PoseNotice")
    window.pose_notice_label.setWordWrap(True)
    params_layout.addWidget(window.pose_notice_label)
    parent_layout.addStretch(1)


__all__ = ["build_left_panel"]
