from __future__ import annotations
from html import escape
from pathlib import Path


def clear_result_widgets(window, subject_text: str = "当前对象：未选择测试用户") -> None:
    """清空右侧结果区。"""
    window.current_result = None
    window.result_subject_label.setText(subject_text)
    window.total_score_label.setText("--")
    window.grade_label.setText("规范等级：待评估")
    window.sub_scores_label.setText("分项评估\n姿态完成度：--\n下盘稳定性：--\n动作连贯性：--\n综合说明：待生成")
    for label in window.overview_cards.values():
        label.setText("--")
    window.summary_text.setHtml("<p>等待选择测试对象并完成评估，系统将生成综合评定结果。</p>")
    window.diff_text.setHtml("<p>当前暂无可展示的指标差异数据。</p>")
    window.error_text.setHtml("<p>当前暂无可展示的问题定位结果。</p>")
    window.suggestion_text.setHtml("<p>当前暂无规范性调整建议。</p>")
    window._update_output_buttons({})


def populate_result_widgets(window, result) -> None:
    """刷新右侧结果区。"""
    total_score = float(result.summary.total_score)
    window.total_score_label.setText(f"{total_score:.1f}")
    grade = "规范" if total_score >= 80 else "亚标准" if total_score >= 60 else "偏差明显"
    window.grade_label.setText(f"规范等级：{grade}")
    window.sub_scores_label.setText(
        "\n".join(
            [
                "分项评估",
                f"姿态完成度：{float(result.summary.posture_score):.1f} / 100",
                f"下盘稳定性：{float(result.summary.stability_score):.1f} / 100",
                f"动作连贯性：{float(result.summary.continuity_score):.1f} / 100",
                f"综合说明：共识别 {len(result.issues)} 项待关注问题",
            ]
        )
    )
    populate_overview_cards(window, result)
    window.summary_text.setHtml(build_summary_html(result))
    window.diff_text.setHtml(build_diff_html(result))
    window.error_text.setHtml(build_error_html(result))
    window.suggestion_text.setHtml(build_suggestion_html(result))
    issue_count = len(result.issues)
    window.header_ratio_label.setText(f"总分 {total_score:.1f} · 问题 {issue_count} 项")
    window.center_status_text.setHtml(
        "<h3>分析摘要</h3>"
        f"<p><b>总分：</b>{total_score:.1f} / 100<br>"
        f"<b>规范等级：</b>{grade}<br>"
        f"<b>识别模式：</b>{'YOLOv8-Pose' if result.used_pose_estimator else '光流兜底'}<br>"
        f"<b>问题数量：</b>{issue_count}</p>"
    )
    window._ensure_visual_artifact(result)


def populate_overview_cards(window, result) -> None:
    """刷新差异速览卡片。"""
    comparison = getattr(result, "comparison", None)
    similarity = "--"
    if comparison is not None and comparison.sequence_similarity_score is not None:
        similarity = f"{float(comparison.sequence_similarity_score):.1f}"
    priority_count = 0
    if comparison is not None and hasattr(comparison, "comparison_issues") and comparison.comparison_issues:
        priority_count = sum(1 for issue in comparison.comparison_issues if getattr(issue, "priority", 0) == 1)
    window.overview_cards["similarity"].setText(similarity)
    window.overview_cards["issue_count"].setText(str(len(result.issues)))
    window.overview_cards["priority_count"].setText(str(priority_count))
    window.overview_cards["pose_mode"].setText("YOLOv8-Pose" if result.used_pose_estimator else "光流兜底")


def _fmt_value(value: float | None, digits: int = 2, unit: str = "") -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}{unit}"


def _fmt_time_range(time_range: tuple[float, float] | None) -> str:
    if not time_range:
        return "未定位具体时间段"
    return f"{time_range[0]:.1f}s - {time_range[1]:.1f}s"


def _describe_issue_direction(title: str, expected_value: float | None, actual_value: float | None) -> str:
    if expected_value is None or actual_value is None:
        return "当前动作与参考要求存在偏差，建议结合时间段回看动作细节。"

    delta = actual_value - expected_value
    abs_delta = abs(delta)
    title_text = title or ""

    if any(keyword in title_text for keyword in ["抬臂", "举臂", "手臂"]):
        if delta < 0:
            return f"手臂动作不到位，抬举幅度较参考值不足 {abs_delta:.2f}。"
        return f"手臂动作幅度偏大，高于参考值 {abs_delta:.2f}，需控制动作上举范围。"
    if any(keyword in title_text for keyword in ["对称", "双臂"]):
        if delta > 0:
            return f"左右动作一致性不足，对称误差较参考值增大 {abs_delta:.2f}。"
        return f"左右动作一致性基本可控，但仍有 {abs_delta:.2f} 的细微误差。"
    if any(keyword in title_text for keyword in ["重心", "平衡", "下盘"]):
        if delta > 0:
            return f"下盘控制不稳，重心偏移程度高于参考值 {abs_delta:.2f}。"
        return f"下盘稳定性基本达标，但仍建议继续压缩重心波动 {abs_delta:.2f}。"
    if any(keyword in title_text for keyword in ["躯干", "转体", "倾斜"]):
        if delta > 0:
            return f"躯干控制不到位，偏移或倾斜幅度高于参考值 {abs_delta:.2f}。"
        return f"躯干姿态接近参考，但仍有 {abs_delta:.2f} 的可优化空间。"
    if any(keyword in title_text for keyword in ["节奏", "速度", "衔接", "过渡"]):
        if delta > 0:
            return f"动作节奏偏快或衔接过急，当前指标高出参考值 {abs_delta:.2f}。"
        return f"动作节奏偏慢或停顿偏多，当前指标低于参考值 {abs_delta:.2f}。"
    if delta < 0:
        return f"当前动作完成度低于参考要求，差距约 {abs_delta:.2f}。"
    return f"当前动作幅度或强度高于参考要求，差距约 {abs_delta:.2f}。"


def _describe_metric_detail(metric) -> str:
    if metric.status == "缺失":
        return "当前指标缺少有效数据，建议补充完整骨架识别结果后再判断。"
    if metric.baseline_value is None or metric.current_value is None:
        return "当前指标暂无完整参考值与测试值对应关系。"

    delta_value = metric.delta_value if metric.delta_value is not None else metric.current_value - metric.baseline_value
    abs_delta = abs(float(delta_value))
    label = str(metric.label)
    if metric.status in {"完美", "极佳", "良好"}:
        return f"该项与参考值接近，当前仅有 {abs_delta:.2f}{metric.unit} 的正常波动。"

    if "抬臂" in label:
        if delta_value < 0:
            return f"抬臂幅度不足，存在动作不到位现象，低于参考值 {abs_delta:.2f}{metric.unit}。"
        return f"抬臂幅度偏大，高于参考值 {abs_delta:.2f}{metric.unit}，需控制上举范围。"
    if "对称" in label:
        return f"双侧动作协同不足，左右不对称程度较参考值多出 {abs_delta:.2f}{metric.unit}。"
    if "重心" in label or "平衡" in label:
        return f"下盘稳定性不足，重心波动较参考值偏大 {abs_delta:.2f}{metric.unit}。"
    if "躯干" in label:
        return f"躯干姿态控制不足，偏移幅度较参考值多出 {abs_delta:.2f}{metric.unit}。"
    if "节奏" in label or "过渡" in label or "连贯" in label:
        return f"动作衔接不够顺畅，节奏控制与参考值相差 {abs_delta:.2f}{metric.unit}。"
    return f"该项存在较明显偏差，当前与参考值相差 {abs_delta:.2f}{metric.unit}。"


def _describe_comparison_issue(issue) -> str:
    affected = "、".join(issue.affected_keypoints[:4]) if getattr(issue, "affected_keypoints", None) else ""
    difference_text = ""
    if getattr(issue, "difference", None) is not None:
        difference_text = f"偏差量约 {float(issue.difference):.2f}。"
    if issue.issue_type == "posture":
        base = "当前姿态与标准模板存在明显差异，动作完成位置不够准确。"
    elif issue.issue_type == "rhythm":
        base = "当前动作节奏与模板不一致，存在快慢失衡或停顿不当。"
    elif issue.issue_type == "symmetry":
        base = "当前左右侧协同不足，存在明显不对称现象。"
    elif issue.issue_type == "phase":
        base = "当前阶段衔接与模板不一致，可能存在起势、定势或收势不到位。"
    else:
        base = "当前动作与标准模板存在需要重点关注的差异。"
    if affected:
        base += f" 主要涉及：{affected}。"
    if difference_text:
        base += difference_text
    return base


def _infer_phase_text(title: str, description: str = "", time_range: tuple[float, float] | None = None) -> str:
    source_text = f"{title} {description}"
    if "起势" in source_text:
        phase_text = "主要影响起势阶段的动作建立与初始姿态进入。"
    elif "过渡" in source_text or "衔接" in source_text:
        phase_text = "主要影响过渡阶段的动作衔接与节奏连续性。"
    elif "定势" in source_text or "保持" in source_text:
        phase_text = "主要影响定势阶段的到位保持与姿态稳定。"
    elif "发力" in source_text:
        phase_text = "主要影响发力阶段的动作展开幅度与协同效率。"
    elif "收势" in source_text:
        phase_text = "主要影响收势阶段的回收控制与动作完整性。"
    else:
        phase_text = "主要影响当前问题对应时间段内的动作完成质量。"
    if time_range:
        phase_text += f" 对应时间段：{_fmt_time_range(time_range)}。"
    return phase_text


def _infer_possible_cause(title: str, expected_value: float | None, actual_value: float | None, description: str = "") -> str:
    source_text = f"{title} {description}"
    if "抬臂" in source_text or "举臂" in source_text or "手臂" in source_text:
        return "常见原因是肩肘联动不充分、上举路线不稳定，或发力后未能持续送到目标位置。"
    if "对称" in source_text or "双臂" in source_text:
        return "常见原因是左右侧启动时机不一致，或一侧代偿明显，导致动作轨迹不同步。"
    if "重心" in source_text or "平衡" in source_text or "下盘" in source_text:
        return "常见原因是支撑腿控制不足、骨盆稳定性不够，或动作过程中身体中线没有保持稳定。"
    if "躯干" in source_text or "转体" in source_text or "倾斜" in source_text:
        return "常见原因是核心控制不足、肩髋协同不够，或转动时上身先行造成代偿。"
    if "节奏" in source_text or "速度" in source_text or "衔接" in source_text or "过渡" in source_text:
        return "常见原因是分段节拍控制不稳，启动、过渡和收束之间缺少均匀的速度分配。"
    if expected_value is not None and actual_value is not None:
        if actual_value < expected_value:
            return "常见原因是动作发起不足、到位意识不强，或目标姿态保持时间不够。"
        return "常见原因是动作控制过冲、幅度控制不足，或节奏处理过急。"
    return "常见原因是动作控制细节不足，建议结合视频回看确认代偿部位和节奏变化。"


def _infer_adjustment_method(title: str, suggestion: str, description: str = "") -> str:
    source_text = f"{title} {description}"
    if "抬臂" in source_text or "举臂" in source_text or "手臂" in source_text:
        return f"建议先分解肩、肘、腕的上举路线，按慢速抬起并在到位点短暂停留，再回到完整动作。{suggestion}"
    if "对称" in source_text or "双臂" in source_text:
        return f"建议采用镜像练习，强调左右同时启动、同时到位，并对照模板检查双侧高度与角度。{suggestion}"
    if "重心" in source_text or "平衡" in source_text or "下盘" in source_text:
        return f"建议先做静态步型与支撑稳定练习，再叠加上肢动作，确保重心始终落在稳定支撑面内。{suggestion}"
    if "躯干" in source_text or "转体" in source_text or "倾斜" in source_text:
        return f"建议先固定肩髋中线，再用腰胯带动动作展开，减少上身前冲、侧倒或过早转体。{suggestion}"
    if "节奏" in source_text or "速度" in source_text or "衔接" in source_text or "过渡" in source_text:
        return f"建议将动作拆为启动、过渡、到位、收束四段，使用固定节拍反复练习，避免突然加速或停顿。{suggestion}"
    return suggestion or "建议先回看问题时间段，对照模板逐段修正动作路线、幅度和节奏。"


def _infer_training_focus(title: str, description: str = "") -> str:
    source_text = f"{title} {description}"
    if "抬臂" in source_text or "举臂" in source_text or "手臂" in source_text:
        return "训练重点放在上肢到位意识、肩肘联动顺序和末端控制。"
    if "对称" in source_text or "双臂" in source_text:
        return "训练重点放在左右同步性、双侧幅度一致性和动作协同。"
    if "重心" in source_text or "平衡" in source_text or "下盘" in source_text:
        return "训练重点放在下盘支撑稳定、中线控制和重心转移连续性。"
    if "躯干" in source_text or "转体" in source_text or "倾斜" in source_text:
        return "训练重点放在核心稳定、肩髋协同和躯干姿态保持。"
    if "节奏" in source_text or "速度" in source_text or "衔接" in source_text or "过渡" in source_text:
        return "训练重点放在分段节拍、动作衔接均匀性和整套连贯完成。"
    return "训练重点放在问题时间段复练、模板对照和整套动作稳定复现。"


def _render_guidance_block(*, manifestation: str, cause: str, phase_text: str, method: str, focus: str) -> str:
    return (
        "<div style='margin-top:6px;padding:10px 12px;background:#f8fbfe;border:1px solid #dbe7f1;border-radius:10px;'>"
        f"<b>问题表现：</b>{escape(manifestation)}<br>"
        f"<b>可能原因：</b>{escape(cause)}<br>"
        f"<b>影响阶段：</b>{escape(phase_text)}<br>"
        f"<b>调整方法：</b>{escape(method)}<br>"
        f"<b>训练重点：</b>{escape(focus)}"
        "</div>"
    )


def _category_terminology_profile(category: str, raw_text: str = "") -> dict[str, str]:
    text = raw_text or ""
    if "八段锦" in text or category == "健身气功":
        return {
            "manifestation": "当前动作在身形中正、沉肩坠肘、松腰沉胯或定势保持方面存在不足。",
            "cause": "多与双手起落不同步、肩肘路线不顺、呼吸配合不稳或关键姿态停留不充分有关。",
            "method": "建议按起势、过渡、定势、收势分段练习，突出沉肩坠肘、立身中正和动作慢而不断。",
            "focus": "训练重点放在身形中正、双侧同步、呼吸配合和关键定势质量。",
        }
    if "太极" in text or category == "太极":
        return {
            "manifestation": "当前动作在立身中正、虚实转换、上下相随或劲路连贯方面存在不足。",
            "cause": "多与腰胯带动不足、重心转换过急、上肢先行或路线圆活性不够有关。",
            "method": "建议先稳住虚实转换，再用腰胯带动上肢运行，强化转接不断劲和手法圆活。",
            "focus": "训练重点放在重心转换、腰胯主导、上下相随和劲路连贯。",
        }
    if "武术" in text or category == "武术基本功":
        return {
            "manifestation": "当前动作在步型规格、出手路线、发力定型或收放节奏方面存在不足。",
            "cause": "多与弓步、马步等基础步型不稳，力点不清，或手脚配合时机不一致有关。",
            "method": "建议先稳定步型和支撑脚，再分解出手路线、发力节点和动作定型，最后提升速度。",
            "focus": "训练重点放在步型规格、出手路线、力点清晰和手脚协同。",
        }
    if "民族舞" in text or category == "民族舞":
        return {
            "manifestation": "当前动作在节拍把握、摆臂路线、身体韵律或转身轴心方面存在不足。",
            "cause": "多与节奏点不稳、上肢线路混乱、胸腰方向控制不足或重心转移提前有关。",
            "method": "建议先按节拍点分段练习，再统一摆臂、转身和收束，逐步增强身体韵律和表现幅度。",
            "focus": "训练重点放在节拍准确、路线清晰、身体韵律和轴心稳定。",
        }
    return {
        "manifestation": "当前动作在完成度、稳定性或连贯性方面存在不足。",
        "cause": "多与动作路线不清、发力不足、节奏失衡或身体中线控制不稳有关。",
        "method": "建议对照模板分阶段回看与复练，优先修正问题最集中的时间段。",
        "focus": "训练重点放在动作路线、幅度控制、节奏分配和整套稳定复现。",
    }


def _merge_with_category_profile(*, category: str, raw_text: str, manifestation: str, cause: str, method: str, focus: str) -> tuple[str, str, str, str]:
    profile = _category_terminology_profile(category, raw_text)
    return (
        f"{profile['manifestation']} {manifestation}".strip(),
        f"{profile['cause']} {cause}".strip(),
        f"{profile['method']} {method}".strip(),
        f"{profile['focus']} {focus}".strip(),
    )


def build_summary_html(result) -> str:
    """构建研判摘要 HTML。"""
    sections = [
        "<h3>综合评定摘要</h3>",
        f"<p><b>总分：</b>{float(result.summary.total_score):.1f} / 100<br>"
        f"<b>评价标准：</b>{result.summary.evaluator_level}<br>"
        f"<b>姿态识别：</b>{'YOLOv8-Pose 已启用' if result.used_pose_estimator else result.summary.pose_backend_reason}</p>",
    ]
    if getattr(result.summary, "phases", None):
        sections.append("<h3>动作阶段划分</h3><ul>")
        for phase in result.summary.phases[:8]:
            sections.append(
                f"<li><b>{phase.get('name') or '未命名阶段'}</b>："
                f"{float(phase.get('start_sec') or 0):.1f}s - {float(phase.get('end_sec') or 0):.1f}s</li>"
            )
        sections.append("</ul>")
    sections.append("<h3>主要偏差项</h3><ul>")
    if result.issues:
        for issue in result.issues[:6]:
            extra = ""
            if issue.actual_value is not None and issue.expected_value is not None:
                extra = f"（参考值 {issue.expected_value:.1f}，当前值 {issue.actual_value:.1f}）"
            sections.append(
                f"<li><b>{escape(issue.title)}</b>{extra}"
                f"<br>判定：{escape(_describe_issue_direction(issue.title, issue.expected_value, issue.actual_value))}"
                f"<br>时间段：{escape(_fmt_time_range(issue.time_range))}</li>"
            )
    else:
        sections.append("<li>当前未发现明显规范性偏差。</li>")
    sections.append("</ul>")
    return "".join(sections)


def build_diff_html(result) -> str:
    """构建模板差异 HTML。"""
    comparison = getattr(result, "comparison", None)
    if comparison is None:
        return "<p>当前未生成详细的模板对照结果。<br>请先加载标准模板基线后，再执行测试分析。</p>"

    parts = [
        "<h3>标准模板差异总览</h3>",
        f"<p><b>基线名称：</b>{comparison.baseline_name}<br>"
        f"<b>对齐方式：</b>{comparison.alignment_mode}<br>"
        f"<b>时序相似度：</b>{comparison.sequence_similarity_score or 0:.1f}<br>"
        f"<b>总体判断：</b>{comparison.overall_assessment or '暂无'}</p>",
        "<h3>关键指标对照表</h3><table width='100%' cellspacing='0' cellpadding='6' style='border-collapse:collapse;'>"
        "<tr><th align='left'>指标</th><th align='left'>标准</th><th align='left'>当前</th><th align='left'>差值</th><th align='left'>状态</th></tr>",
    ]
    if comparison.metrics:
        for metric in comparison.metrics[:12]:
            baseline_val = f"{metric.baseline_value:.2f}" if metric.baseline_value is not None else "--"
            current_val = f"{metric.current_value:.2f}" if metric.current_value is not None else "--"
            delta_val = f"{metric.delta_value:+.2f}" if metric.delta_value is not None else "--"
            status_text = {
                "完美": "完美",
                "极佳": "极佳",
                "良好": "良好",
                "一般": "一般",
                "注意": "注意",
                "偏差": "偏差",
                "较差": "较差",
                "缺失": "数据缺失",
            }.get(metric.status, metric.status)
            color = {
                "完美": "#15803D",
                "极佳": "#16A34A",
                "良好": "#22C55E",
                "一般": "#CA8A04",
                "注意": "#D97706",
                "偏差": "#DC2626",
                "较差": "#B91C1C",
                "缺失": "#DC2626",
            }.get(metric.status, "#475467")
            parts.append(
                f"<tr><td>{metric.label}</td><td>{baseline_val}</td><td>{current_val}</td>"
                f"<td>{delta_val}</td><td><span style='color:{color};font-weight:700'>{status_text}</span></td></tr>"
            )
    else:
        parts.append("<tr><td colspan='5'>暂无可用的对比指标数据</td></tr>")
    parts.append("</table>")
    if comparison.metrics:
        parts.append("<h3>差异解读</h3><ul>")
        for metric in comparison.metrics[:8]:
            parts.append(
                f"<li><b>{escape(metric.label)}</b>：{escape(_describe_metric_detail(metric))}"
                f"{'<br>说明：' + escape(metric.note) if getattr(metric, 'note', '') else ''}</li>"
            )
        parts.append("</ul>")
    if hasattr(comparison, "comparison_issues") and comparison.comparison_issues:
        parts.append("<h3>重点差异说明</h3><ul>")
        for i, issue in enumerate(comparison.comparison_issues[:5], 1):
            severity_text = {"critical": "重点关注", "major": "建议改进", "minor": "细节优化"}.get(issue.severity, "待判定")
            manifestation = _describe_comparison_issue(issue)
            cause = _infer_possible_cause(issue.title, issue.template_value, issue.test_value, getattr(issue, "description", ""))
            phase_text = _infer_phase_text(issue.title, getattr(issue, "description", ""), getattr(issue, "time_range", None))
            method = _infer_adjustment_method(issue.title, getattr(issue, "suggestion", ""), getattr(issue, "description", ""))
            focus = _infer_training_focus(issue.title, getattr(issue, "description", ""))
            manifestation, cause, method, focus = _merge_with_category_profile(
                category=getattr(result.rules, "action_category", "通用动作"),
                raw_text=getattr(result.rules, "raw_text", ""),
                manifestation=manifestation,
                cause=cause,
                method=method,
                focus=focus,
            )
            parts.append(f"<li><b>{i}. {issue.title}</b> <span style='color:#D97706'>[{severity_text}]</span>")
            if issue.template_value is not None and issue.test_value is not None:
                parts.append(f"<br>参考值：{issue.template_value:.2f} / 当前值：{issue.test_value:.2f}")
            if getattr(issue, "time_range", None):
                parts.append(f"<br>时间段：{_fmt_time_range(issue.time_range)}")
            if getattr(issue, "description", ""):
                parts.append(f"<br>说明：{escape(issue.description)}")
            parts.append(f"<br>判定：{escape(manifestation)}")
            if getattr(issue, "suggestion", ""):
                parts.append(f"<br>调整建议：{issue.suggestion}")
            parts.append(_render_guidance_block(
                manifestation=manifestation,
                cause=cause,
                phase_text=phase_text,
                method=method,
                focus=focus,
            ))
            parts.append("</li>")
        parts.append("</ul>")
    return "".join(parts)


def build_error_html(result) -> str:
    """构建错误定位 HTML。"""
    if not result.issues:
        return "<p>当前未识别到明显的问题定位点。<br>动作整体完成情况较好，可继续结合模板进行细节复核。</p>"

    critical_issues = []
    major_issues = []
    minor_issues = []

    for issue in result.issues[:10]:
        severity = getattr(issue, "severity", "medium")
        if severity == "high" or severity == "critical":
            critical_issues.append(issue)
        elif severity == "medium" or severity == "major":
            major_issues.append(issue)
        else:
            minor_issues.append(issue)

    parts = ["<h3>问题定位与时间段</h3>"]
    if critical_issues:
        parts.append("<h4 style='color:#DC2626'>重点问题（建议优先纠正）</h4><ul>")
        for issue in critical_issues:
            time_text = f" [{issue.time_range[0]:.1f}s - {issue.time_range[1]:.1f}s]" if issue.time_range else ""
            manifestation = _describe_issue_direction(issue.title, issue.expected_value, issue.actual_value)
            cause = _infer_possible_cause(issue.title, issue.expected_value, issue.actual_value, issue.detail)
            phase_text = _infer_phase_text(issue.title, issue.detail, issue.time_range)
            method = _infer_adjustment_method(issue.title, issue.suggestion, issue.detail)
            focus = _infer_training_focus(issue.title, issue.detail)
            manifestation, cause, method, focus = _merge_with_category_profile(
                category=getattr(result.rules, "action_category", "通用动作"),
                raw_text=getattr(result.rules, "raw_text", ""),
                manifestation=manifestation,
                cause=cause,
                method=method,
                focus=focus,
            )
            parts.append(f"<li><b>{issue.title}</b>{time_text}<br>说明：{issue.detail}")
            if issue.actual_value is not None and issue.expected_value is not None:
                parts.append(f"<br>当前值：{issue.actual_value:.2f} / 参考值：{issue.expected_value:.2f}")
            parts.append(f"<br>详细判定：{escape(manifestation)}")
            parts.append(_render_guidance_block(
                manifestation=manifestation,
                cause=cause,
                phase_text=phase_text,
                method=method,
                focus=focus,
            ))
            parts.append("</li>")
        parts.append("</ul>")
    if major_issues:
        parts.append("<h4 style='color:#D97706'>一般问题（建议持续校正）</h4><ul>")
        for issue in major_issues:
            time_text = f" [{issue.time_range[0]:.1f}s - {issue.time_range[1]:.1f}s]" if issue.time_range else ""
            manifestation = _describe_issue_direction(issue.title, issue.expected_value, issue.actual_value)
            cause = _infer_possible_cause(issue.title, issue.expected_value, issue.actual_value, issue.detail)
            phase_text = _infer_phase_text(issue.title, issue.detail, issue.time_range)
            method = _infer_adjustment_method(issue.title, issue.suggestion, issue.detail)
            focus = _infer_training_focus(issue.title, issue.detail)
            manifestation, cause, method, focus = _merge_with_category_profile(
                category=getattr(result.rules, "action_category", "通用动作"),
                raw_text=getattr(result.rules, "raw_text", ""),
                manifestation=manifestation,
                cause=cause,
                method=method,
                focus=focus,
            )
            parts.append(
                f"<li><b>{issue.title}</b>{time_text}<br>说明：{issue.detail}"
                f"<br>详细判定：{escape(manifestation)}"
                f"{_render_guidance_block(manifestation=manifestation, cause=cause, phase_text=phase_text, method=method, focus=focus)}</li>"
            )
        parts.append("</ul>")
    if minor_issues:
        parts.append("<h4 style='color:#2563EB'>细节问题（可继续优化）</h4><ul>")
        for issue in minor_issues:
            time_text = f" [{issue.time_range[0]:.1f}s - {issue.time_range[1]:.1f}s]" if issue.time_range else ""
            manifestation = _describe_issue_direction(issue.title, issue.expected_value, issue.actual_value)
            cause = _infer_possible_cause(issue.title, issue.expected_value, issue.actual_value, issue.detail)
            phase_text = _infer_phase_text(issue.title, issue.detail, issue.time_range)
            method = _infer_adjustment_method(issue.title, issue.suggestion, issue.detail)
            focus = _infer_training_focus(issue.title, issue.detail)
            manifestation, cause, method, focus = _merge_with_category_profile(
                category=getattr(result.rules, "action_category", "通用动作"),
                raw_text=getattr(result.rules, "raw_text", ""),
                manifestation=manifestation,
                cause=cause,
                method=method,
                focus=focus,
            )
            parts.append(
                f"<li><b>{issue.title}</b>{time_text}"
                f"<br>详细判定：{escape(manifestation)}"
                f"{_render_guidance_block(manifestation=manifestation, cause=cause, phase_text=phase_text, method=method, focus=focus)}</li>"
            )
        parts.append("</ul>")
    return "".join(parts)


def build_suggestion_html(result) -> str:
    """构建修改建议 HTML。"""
    parts = ["<h3>规范性调整建议</h3>"]
    comparison = getattr(result, "comparison", None)
    if comparison and hasattr(comparison, "comparison_issues") and comparison.comparison_issues:
        parts.append("<h4>基于模板对照的调整建议</h4><ol>")
        for issue in comparison.comparison_issues[:6]:
            priority_label = {1: "优先整改", 2: "持续优化", 3: "细节完善"}.get(issue.priority, "待调整")
            manifestation = _describe_comparison_issue(issue)
            cause = _infer_possible_cause(issue.title, issue.template_value, issue.test_value, getattr(issue, "description", ""))
            phase_text = _infer_phase_text(issue.title, getattr(issue, "description", ""), getattr(issue, "time_range", None))
            method = _infer_adjustment_method(issue.title, getattr(issue, "suggestion", ""), getattr(issue, "description", ""))
            focus = _infer_training_focus(issue.title, getattr(issue, "description", ""))
            manifestation, cause, method, focus = _merge_with_category_profile(
                category=getattr(result.rules, "action_category", "通用动作"),
                raw_text=getattr(result.rules, "raw_text", ""),
                manifestation=manifestation,
                cause=cause,
                method=method,
                focus=focus,
            )
            parts.append(
                f"<li><b>{issue.title}</b> <span style='color:#165DFF'>[{priority_label}]</span><br>"
                f"{_render_guidance_block(manifestation=manifestation, cause=cause, phase_text=phase_text, method=method, focus=focus)}</li>"
            )
        parts.append("</ol>")
    if result.issues:
        parts.append("<h4>基于动作识别的纠正建议</h4><ol>")
        for issue in result.issues[:6]:
            manifestation = _describe_issue_direction(issue.title, issue.expected_value, issue.actual_value)
            cause = _infer_possible_cause(issue.title, issue.expected_value, issue.actual_value, issue.detail)
            phase_text = _infer_phase_text(issue.title, issue.detail, issue.time_range)
            method = _infer_adjustment_method(issue.title, issue.suggestion, issue.detail)
            focus = _infer_training_focus(issue.title, issue.detail)
            manifestation, cause, method, focus = _merge_with_category_profile(
                category=getattr(result.rules, "action_category", "通用动作"),
                raw_text=getattr(result.rules, "raw_text", ""),
                manifestation=manifestation,
                cause=cause,
                method=method,
                focus=focus,
            )
            parts.append(
                f"<li><b>{issue.title}</b><br>"
                f"{_render_guidance_block(manifestation=manifestation, cause=cause, phase_text=phase_text, method=method, focus=focus)}</li>"
            )
        parts.append("</ol>")
    if result.summary.improvement_focus:
        parts.append("<h4>后续训练重点</h4><ul>")
        for item in result.summary.improvement_focus:
            parts.append(f"<li>{item}</li>")
        parts.append("</ul>")
    phases = getattr(result.summary, "phases", None) or []
    if phases:
        parts.append("<h4>分阶段优化建议</h4><ul>")
        for phase in phases[:8]:
            phase_name = str(phase.get("name") or "未命名阶段")
            start_sec = float(phase.get("start_sec") or 0.0)
            end_sec = float(phase.get("end_sec") or start_sec)
            guidance = str(phase.get("guidance") or "").strip() or "保持动作中正、路线清楚、节奏均匀。"
            parts.append(
                f"<li><b>{escape(phase_name)}</b> [{start_sec:.1f}s - {end_sec:.1f}s]"
                f"<br>{escape(guidance)}</li>"
            )
        parts.append("</ul>")
    if result.summary.advice:
        parts.append("<h4>综合指导意见</h4><ul>")
        for item in result.summary.advice:
            parts.append(f"<li>{item}</li>")
        parts.append("</ul>")
    if len(parts) <= 1:
        parts.append("<p>当前动作整体较为规范，可继续结合模板进行细节打磨与稳定性复核。</p>")
    return "".join(parts)


__all__ = [
    "build_diff_html",
    "build_error_html",
    "build_suggestion_html",
    "build_summary_html",
    "clear_result_widgets",
    "populate_overview_cards",
    "populate_result_widgets",
]
