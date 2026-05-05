from __future__ import annotations

import os
from pathlib import Path

from .config import PROJECT_ROOT

_mpl_config_dir = PROJECT_ROOT / ".mplconfig"
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))

_cache_root = PROJECT_ROOT / ".cache"
_cache_root.mkdir(parents=True, exist_ok=True)
(_cache_root / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .models import AnalysisResult, FrameMetrics

plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Heiti SC",
    "STHeiti",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def build_metrics_plot(result: AnalysisResult, output_path: Path) -> None:
    timestamps = [item.timestamp_sec for item in result.frame_metrics]
    motion = [item.motion_mean for item in result.frame_metrics]
    balance = [item.left_right_balance for item in result.frame_metrics]
    torso = [item.torso_tilt_deg if item.torso_tilt_deg is not None else np.nan for item in result.frame_metrics]
    symmetry = [
        item.arm_symmetry_error if item.arm_symmetry_error is not None else np.nan
        for item in result.frame_metrics
    ]
    jitter = [item.joint_jitter if item.joint_jitter is not None else np.nan for item in result.frame_metrics]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(timestamps, motion, color="#1f77b4", linewidth=2)
    axes[0].set_ylabel("运动强度")
    axes[0].grid(alpha=0.25)

    axes[1].plot(timestamps, balance, color="#ff7f0e", linewidth=2)
    if not np.all(np.isnan(jitter)):
        axes[1].plot(timestamps, jitter, color="#9467bd", linewidth=1.6, label="骨架抖动")
    axes[1].set_ylabel("左右平衡差")
    axes[1].grid(alpha=0.25)
    if not np.all(np.isnan(jitter)):
        axes[1].legend(loc="upper right")

    axes[2].plot(timestamps, torso, color="#2ca02c", linewidth=2, label="躯干倾斜")
    if not np.all(np.isnan(symmetry)):
        axes[2].plot(timestamps, symmetry, color="#d62728", linewidth=2, label="双臂对称误差")
    axes[2].set_ylabel("姿态偏差")
    axes[2].set_xlabel("时间（秒）")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="upper right")

    title = "动作分析指标曲线"
    if result.rules.template_name:
        title += f" - 模板：{result.rules.template_name}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def create_summary_card(result: AnalysisResult, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    lines = [
        "AI 动作分析结果摘要",
        f"总评分：{result.summary.total_score:.1f}",
        f"姿态标准度：{result.summary.posture_score:.1f}",
        f"动作连续性：{result.summary.continuity_score:.1f}",
        f"动作稳定性：{result.summary.stability_score:.1f}",
        f"节奏控制：{result.summary.rhythm_score:.1f}",
        f"动作完整性：{result.summary.completeness_score:.1f}",
        "",
        f"平均运动强度：{result.summary.avg_motion:.3f}",
        f"平均左右平衡差：{result.summary.avg_left_right_balance:.3f}",
    ]

    if result.summary.avg_torso_tilt is not None:
        lines.append(f"平均躯干倾斜：{result.summary.avg_torso_tilt:.2f}°")
    if result.summary.avg_arm_symmetry_error is not None:
        lines.append(f"平均双臂对称误差：{result.summary.avg_arm_symmetry_error:.2f}°")
    if result.summary.phases:
        lines.extend(["", "阶段识别："])
        for phase in result.summary.phases[:6]:
            lines.append(f"- {phase['name']} {phase['start_sec']:.1f}s-{phase['end_sec']:.1f}s")

    if result.issues:
        lines.extend(["", "主要问题："])
        for issue in result.issues[:5]:
            lines.append(f"- {issue.title}：{issue.detail}")

    if result.summary.advice:
        lines.extend(["", "建议："])
        for advice in result.summary.advice[:5]:
            lines.append(f"- {advice}")

    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=12,
        family="sans-serif",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def create_contact_sheet(frames: list[np.ndarray], output_path: Path, columns: int = 3) -> None:
    if not frames:
        return

    thumbnails = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    count = len(thumbnails)
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for axis, image in zip(axes, thumbnails):
        axis.imshow(image)
        axis.axis("off")

    for axis in axes[count:]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


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
