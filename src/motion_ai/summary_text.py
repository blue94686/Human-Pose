from __future__ import annotations

from .models import AnalysisResult


def render_preview_text(result: AnalysisResult) -> str:
    severity_labels = {"high": "高", "medium": "中", "low": "低"}
    issue_lines = []
    for issue in result.issues[:5]:
        if issue.time_range:
            issue_lines.append(
                f"[{severity_labels.get(issue.severity, issue.severity)}] {issue.title}（{issue.time_range[0]:.1f}s-{issue.time_range[1]:.1f}s）"
            )
        else:
            issue_lines.append(f"[{severity_labels.get(issue.severity, issue.severity)}] {issue.title}")

    advice_lines = [f"- {item}" for item in result.summary.advice[:5]]
    phases = [
        f"{phase['name']} {phase['start_sec']:.1f}s-{phase['end_sec']:.1f}s"
        for phase in result.summary.phases
    ]

    return "\n".join(
        [
            "验证结果摘要",
            f"分析模式：{'模板原始数据生成' if result.analysis_mode == 'template' else '测试者分析'}",
            f"模板：{result.rules.template_name or '未匹配模板'}",
            f"评价标准：{result.summary.evaluator_level}",
            f"测试者水平：{result.summary.learner_level}",
            f"总评分：{result.summary.total_score:.1f}",
            f"姿态标准度：{result.summary.posture_score:.1f}",
            f"动作连续性：{result.summary.continuity_score:.1f}",
            f"动作稳定性：{result.summary.stability_score:.1f}",
            (
                f"关键点覆盖率：{result.summary.avg_keypoint_coverage * 100:.1f}%"
                if result.summary.avg_keypoint_coverage is not None
                else "关键点覆盖率：暂无"
            ),
            (
                f"姿态质量：{result.summary.avg_pose_quality_score:.1f}（{result.summary.pose_quality_level or '未评级'}）"
                if result.summary.avg_pose_quality_score is not None
                else "姿态质量：暂无"
            ),
            "",
            "阶段识别：",
            *(phases or ["- 未识别出明显阶段"]),
            "",
            "重点问题：",
            *(issue_lines or ["- 未发现明显异常"]),
            "",
            "纠正建议：",
            *(advice_lines or ["- 当前动作整体较稳定，可继续补充标准模板对比。"]),
            "",
            "优化方向：",
            *([f"- {item}" for item in result.summary.improvement_focus[:5]] or ["- 暂无额外优化方向。"]),
        ]
    )
