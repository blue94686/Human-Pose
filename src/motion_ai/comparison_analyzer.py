"""对比分析器 - 实现模板与测试动作的对比分析"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .analyzer import ActionAnalyzer
from .models import (
    AnalysisResult,
    ComparisonIssue,
    ComparisonResult,
    DifferenceMetrics,
    PhaseAlignment,
    TemplateBaseline,
)


class ComparisonAnalyzer:
    """对比分析器 - 负责模板分析和测试对比"""

    def __init__(self, weights_path: str | None = None):
        self.analyzer = ActionAnalyzer(weights_path=weights_path)

    def analyze_template(
        self,
        source: str | int,
        description: str,
        template_name: str,
        output_dir: Path,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> TemplateBaseline:
        """
        分析模板动作，生成基线数据

        Args:
            source: 视频路径或摄像头ID
            description: 动作描述
            template_name: 模板名称
            output_dir: 输出目录
            progress_callback: 进度回调

        Returns:
            TemplateBaseline: 模板基线数据
        """
        # 使用标准分析器分析模板
        result = self.analyzer.analyze_video(
            source=source,
            description_text=description,
            output_dir=output_dir,
            progress_callback=progress_callback,
            analysis_mode="template",
        )

        # 转换为模板基线格式
        baseline = TemplateBaseline(
            template_name=template_name,
            source_path=str(source),
            source_type=self._detect_source_type(source),
            created_at=datetime.now().isoformat(),
            fps=result.fps,
            frame_count=result.frame_count,
            duration_sec=result.frame_count / result.fps if result.fps > 0 else 0,
            frames=result.pose_frames if hasattr(result, 'pose_frames') else [],
            frame_metrics=result.frame_metrics,
            phases=result.summary.phases,
            summary=result.summary,
            description=description,
        )

        # 保存模板基线
        baseline_path = output_dir / "template_baseline.json"
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline.to_dict(), f, ensure_ascii=False, indent=2)

        return baseline

    def analyze_test(
        self,
        source: str | int,
        template_baseline: TemplateBaseline,
        user_level: str,
        output_dir: Path,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> ComparisonResult:
        """
        分析测试动作并与模板对比

        Args:
            source: 测试视频路径或摄像头ID
            template_baseline: 模板基线数据
            user_level: 用户等级 (初学者/进阶者/熟练者)
            output_dir: 输出目录
            progress_callback: 进度回调

        Returns:
            ComparisonResult: 对比结果
        """
        # 分析测试视频
        test_result = self.analyzer.analyze_video(
            source=source,
            description_text=template_baseline.description,
            output_dir=output_dir,
            progress_callback=progress_callback,
            analysis_mode="test",
        )

        # 阶段对齐
        phase_alignments = self.align_phases(
            template_baseline.phases, test_result.summary.phases
        )

        # 计算差异
        difference_metrics = self.calculate_differences(
            template_baseline, test_result
        )

        # 生成对比问题
        comparison_issues = self._generate_comparison_issues(
            template_baseline, test_result, difference_metrics, user_level
        )

        # 计算对比评分
        comparison_score = self._calculate_comparison_score(
            template_baseline, test_result, difference_metrics
        )

        # 生成分级建议
        suggestions_by_priority = self._generate_suggestions_by_priority(
            comparison_issues, user_level
        )

        # 构建对比结果
        comparison_result = ComparisonResult(
            baseline_name=template_baseline.template_name,
            baseline_source=template_baseline.source_path,
            baseline_kind=template_baseline.source_type,
            alignment_mode="phase_based",
            template_baseline=template_baseline,
            test_result=test_result,
            phase_alignments=phase_alignments,
            difference_metrics=difference_metrics,
            comparison_score=comparison_score,
            score_breakdown=self._calculate_score_breakdown(
                template_baseline, test_result
            ),
            comparison_issues=comparison_issues,
            suggestions_by_priority=suggestions_by_priority,
            overall_assessment=self._generate_overall_assessment(
                comparison_score, comparison_issues
            ),
        )

        # 保存对比结果
        comparison_path = output_dir / "comparison_result.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison_result.to_dict(), f, ensure_ascii=False, indent=2)

        return comparison_result

    def align_phases(
        self, template_phases: list[dict], test_phases: list[dict]
    ) -> list[PhaseAlignment]:
        """
        对齐模板和测试的动作阶段

        Args:
            template_phases: 模板阶段列表
            test_phases: 测试阶段列表

        Returns:
            list[PhaseAlignment]: 阶段对齐结果
        """
        alignments = []

        # 简单的顺序对齐策略
        for i, (t_phase, test_phase) in enumerate(
            zip(template_phases, test_phases)
        ):
            t_start = t_phase.get("start_frame", 0)
            t_end = t_phase.get("end_frame", 0)
            t_duration = t_phase.get("duration_sec", 0)

            test_start = test_phase.get("start_frame", 0)
            test_end = test_phase.get("end_frame", 0)
            test_duration = test_phase.get("duration_sec", 0)

            duration_diff = test_duration - t_duration

            # 计算相似度（基于时长差异）
            if t_duration > 0:
                similarity = max(
                    0, 1 - abs(duration_diff) / t_duration
                )
            else:
                similarity = 0.5

            # 评估对齐质量
            if similarity >= 0.9:
                quality = "excellent"
            elif similarity >= 0.7:
                quality = "good"
            elif similarity >= 0.5:
                quality = "fair"
            else:
                quality = "poor"

            alignment = PhaseAlignment(
                template_phase=t_phase.get("name", f"阶段{i+1}"),
                test_phase=test_phase.get("name", f"阶段{i+1}"),
                template_start_frame=t_start,
                template_end_frame=t_end,
                test_start_frame=test_start,
                test_end_frame=test_end,
                template_duration=t_duration,
                test_duration=test_duration,
                duration_diff=duration_diff,
                similarity_score=similarity,
                alignment_quality=quality,
            )
            alignments.append(alignment)

        return alignments

    def calculate_differences(
        self, template_baseline: TemplateBaseline, test_result: AnalysisResult
    ) -> DifferenceMetrics:
        """
        计算模板和测试之间的差异指标

        Args:
            template_baseline: 模板基线
            test_result: 测试结果

        Returns:
            DifferenceMetrics: 差异指标
        """
        # 计算平均指标差异
        template_metrics = template_baseline.frame_metrics
        test_metrics = test_result.frame_metrics

        # 躯干倾斜差异
        template_torso = [
            m.torso_tilt_deg for m in template_metrics if m.torso_tilt_deg is not None
        ]
        test_torso = [
            m.torso_tilt_deg for m in test_metrics if m.torso_tilt_deg is not None
        ]
        torso_diff = None
        if template_torso and test_torso:
            torso_diff = abs(np.mean(test_torso) - np.mean(template_torso))

        # 双臂对称性差异
        template_arm = [
            m.arm_symmetry_error
            for m in template_metrics
            if m.arm_symmetry_error is not None
        ]
        test_arm = [
            m.arm_symmetry_error for m in test_metrics if m.arm_symmetry_error is not None
        ]
        arm_diff = None
        if template_arm and test_arm:
            arm_diff = abs(np.mean(test_arm) - np.mean(template_arm))

        # 左右平衡差异
        template_balance = [m.left_right_balance for m in template_metrics]
        test_balance = [m.left_right_balance for m in test_metrics]
        balance_diff = None
        if template_balance and test_balance:
            balance_diff = abs(np.mean(test_balance) - np.mean(template_balance))

        # 节奏差异（基于运动强度）
        template_motion = [m.motion_mean for m in template_metrics]
        test_motion = [m.motion_mean for m in test_metrics]
        rhythm_diff = None
        if template_motion and test_motion:
            rhythm_diff = abs(np.mean(test_motion) - np.mean(template_motion))

        return DifferenceMetrics(
            avg_keypoint_distance=0.0,  # 需要关键点数据计算
            max_keypoint_distance=0.0,
            torso_tilt_diff=torso_diff,
            arm_symmetry_diff=arm_diff,
            balance_diff=balance_diff,
            rhythm_diff=rhythm_diff,
            keypoint_differences={},
            phase_differences=[],
        )

    def _detect_source_type(self, source: str | int) -> str:
        """检测输入源类型"""
        if isinstance(source, int):
            return "camera"
        source_path = Path(source)
        suffix = source_path.suffix.lower()
        if suffix in [".mp4", ".avi", ".mov", ".mkv"]:
            return "video"
        elif suffix in [".jpg", ".jpeg", ".png"]:
            return "image"
        elif suffix in [".xlsx", ".xls"]:
            return "excel"
        else:
            return "unknown"

    def _generate_comparison_issues(
        self,
        template: TemplateBaseline,
        test: AnalysisResult,
        diff: DifferenceMetrics,
        user_level: str,
    ) -> list[ComparisonIssue]:
        """生成对比问题列表"""
        issues = []

        # 躯干倾斜问题
        if diff.torso_tilt_diff and diff.torso_tilt_diff > 10:
            severity = "critical" if diff.torso_tilt_diff > 20 else "major"
            template_avg = np.mean(
                [m.torso_tilt_deg for m in template.frame_metrics if m.torso_tilt_deg is not None]
            ) if template.frame_metrics else None
            test_avg = np.mean(
                [m.torso_tilt_deg for m in test.frame_metrics if m.torso_tilt_deg is not None]
            ) if test.frame_metrics else None

            # 构建描述
            if template_avg is not None and test_avg is not None:
                description = f"测试者躯干倾斜与标准动作相差 {diff.torso_tilt_diff:.1f}°，标准值 {template_avg:.1f}°，当前值 {test_avg:.1f}°"
            else:
                description = f"测试者躯干倾斜与标准动作相差 {diff.torso_tilt_diff:.1f}°"

            issues.append(
                ComparisonIssue(
                    issue_type="posture",
                    severity=severity,
                    title="躯干倾斜角度偏差较大",
                    description=description,
                    template_value=template_avg,
                    test_value=test_avg,
                    difference=diff.torso_tilt_diff,
                    time_range=None,
                    suggestion="保持躯干垂直，避免前倾或后仰。练习时可对着镜子或录像检查躯干姿态，确保肩髋在同一垂直线上",
                    priority=1,
                )
            )

        # 双臂对称性问题
        if diff.arm_symmetry_diff and diff.arm_symmetry_diff > 15:
            template_avg = np.mean(
                [m.arm_symmetry_error for m in template.frame_metrics if m.arm_symmetry_error is not None]
            ) if template.frame_metrics else None
            test_avg = np.mean(
                [m.arm_symmetry_error for m in test.frame_metrics if m.arm_symmetry_error is not None]
            ) if test.frame_metrics else None

            # 构建描述
            if template_avg is not None and test_avg is not None:
                description = f"左右臂动作不对称，误差 {diff.arm_symmetry_diff:.1f}°，标准值 {template_avg:.1f}°，当前值 {test_avg:.1f}°"
            else:
                description = f"左右臂动作不对称，误差 {diff.arm_symmetry_diff:.1f}°"

            issues.append(
                ComparisonIssue(
                    issue_type="symmetry",
                    severity="major",
                    title="双臂对称性不足",
                    description=description,
                    template_value=template_avg,
                    test_value=test_avg,
                    difference=diff.arm_symmetry_diff,
                    time_range=None,
                    suggestion="注意左右臂同步，保持对称。建议分解练习：先单独练习左臂，再练习右臂，最后合并同步练习",
                    priority=1,
                )
            )

        # 左右平衡问题
        if diff.balance_diff and diff.balance_diff > 0.1:
            template_avg = np.mean([m.left_right_balance for m in template.frame_metrics])
            test_avg = np.mean([m.left_right_balance for m in test.frame_metrics])

            issues.append(
                ComparisonIssue(
                    issue_type="balance",
                    severity="major" if diff.balance_diff > 0.15 else "minor",
                    title="左右平衡偏差",
                    description=f"身体重心偏向一侧，偏差 {diff.balance_diff:.3f}，标准值 {template_avg:.3f}，当前值 {test_avg:.3f}",
                    template_value=template_avg,
                    test_value=test_avg,
                    difference=diff.balance_diff,
                    time_range=None,
                    suggestion="调整重心分布，保持身体中线稳定。练习时注意双脚均匀受力，避免身体偏向一侧",
                    priority=1,
                )
            )

        # 节奏问题
        if diff.rhythm_diff and diff.rhythm_diff > 5:
            template_avg = np.mean([m.motion_mean for m in template.frame_metrics])
            test_avg = np.mean([m.motion_mean for m in test.frame_metrics])

            rhythm_status = "偏快" if test_avg > template_avg else "偏慢"
            issues.append(
                ComparisonIssue(
                    issue_type="rhythm",
                    severity="minor",
                    title=f"动作节奏{rhythm_status}",
                    description=f"动作速度与标准不一致，运动强度差异 {diff.rhythm_diff:.3f}，标准值 {template_avg:.3f}，当前值 {test_avg:.3f}",
                    template_value=template_avg,
                    test_value=test_avg,
                    difference=diff.rhythm_diff,
                    time_range=None,
                    suggestion=f"调整动作速度，与标准节奏保持一致。建议{'放慢动作，增加每个动作的停留时间' if rhythm_status == '偏快' else '适当加快动作转换，保持动作连贯性'}",
                    priority=2,
                )
            )

        # 动作稳定性问题
        template_stability = np.mean([m.skeleton_stability for m in template.frame_metrics if m.skeleton_stability])
        test_stability = np.mean([m.skeleton_stability for m in test.frame_metrics if m.skeleton_stability])

        if test_stability and template_stability and test_stability < template_stability * 0.8:
            issues.append(
                ComparisonIssue(
                    issue_type="stability",
                    severity="major",
                    title="动作稳定性不足",
                    description=f"骨架稳定性低于标准，标准值 {template_stability:.3f}，当前值 {test_stability:.3f}",
                    template_value=template_stability,
                    test_value=test_stability,
                    difference=template_stability - test_stability,
                    time_range=None,
                    suggestion="加强核心力量训练，提高动作控制能力。练习时放慢速度，确保每个动作都稳定到位",
                    priority=1,
                )
            )

        # 关键点覆盖率问题
        template_coverage = np.mean([m.keypoint_coverage for m in template.frame_metrics if m.keypoint_coverage])
        test_coverage = np.mean([m.keypoint_coverage for m in test.frame_metrics if m.keypoint_coverage])

        if test_coverage and template_coverage and test_coverage < 0.9:
            issues.append(
                ComparisonIssue(
                    issue_type="detection",
                    severity="minor",
                    title="关键点识别不完整",
                    description=f"部分关键点未被识别，覆盖率 {test_coverage:.1%}，标准值 {template_coverage:.1%}",
                    template_value=template_coverage,
                    test_value=test_coverage,
                    difference=template_coverage - test_coverage,
                    time_range=None,
                    suggestion="调整拍摄角度和距离，确保全身完整入镜。避免遮挡，保持良好的光照条件",
                    priority=3,
                )
            )

        return issues

    def _calculate_comparison_score(
        self, template: TemplateBaseline, test: AnalysisResult, diff: DifferenceMetrics
    ) -> float:
        """计算对比相似度评分"""
        score = 100.0

        # 根据差异扣分
        if diff.torso_tilt_diff:
            score -= min(diff.torso_tilt_diff * 0.5, 20)

        if diff.arm_symmetry_diff:
            score -= min(diff.arm_symmetry_diff * 0.3, 15)

        if diff.balance_diff:
            score -= min(diff.balance_diff * 2, 10)

        if diff.rhythm_diff:
            score -= min(diff.rhythm_diff, 10)

        return max(0, score)

    def _calculate_score_breakdown(
        self, template: TemplateBaseline, test: AnalysisResult
    ) -> dict[str, float]:
        """计算分项评分对比"""
        return {
            "姿态标准度": test.summary.posture_score,
            "动作连贯性": test.summary.continuity_score,
            "节奏控制": test.summary.rhythm_score,
            "左右对称性": 100 - (test.summary.avg_arm_symmetry_error or 0),
        }

    def _generate_suggestions_by_priority(
        self, issues: list[ComparisonIssue], user_level: str
    ) -> dict[int, list[str]]:
        """生成分级建议"""
        suggestions = {1: [], 2: [], 3: []}

        for issue in issues:
            suggestions[issue.priority].append(issue.suggestion)

        return suggestions

    def _generate_overall_assessment(
        self, score: float, issues: list[ComparisonIssue]
    ) -> str:
        """生成总体评估"""
        if score >= 90:
            return "优秀：动作与标准高度一致"
        elif score >= 80:
            return "良好：动作基本符合标准，有少量偏差"
        elif score >= 70:
            return "中等：动作存在明显偏差，需要改进"
        else:
            return "需加强：动作与标准差异较大，需要重点练习"
