"""
详细对比系统模块
实现标准动作与测试动作的详细对比分析
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from .models import FrameMetrics, AnalysisSummary


@dataclass
class PhaseMatch:
    baseline_phase: str
    test_phase: str
    similarity: float
    time_offset: float


@dataclass
class FrameDiff:
    frame_index: int
    timestamp: float
    motion_diff: float
    balance_diff: float
    pose_quality_diff: float
    keypoint_distances: Dict[str, float]


@dataclass
class MetricDiff:
    metric_name: str
    baseline_value: float
    test_value: float
    difference: float
    percentage: float
    status: str


@dataclass
class DetailedComparison:
    baseline_name: str
    test_name: str
    phase_alignment: List[PhaseMatch] = field(default_factory=list)
    frame_differences: List[FrameDiff] = field(default_factory=list)
    keypoint_deviations: Dict[str, float] = field(default_factory=dict)
    metric_comparison: Dict[str, MetricDiff] = field(default_factory=dict)
    overall_similarity: float = 0.0
    major_issues: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)


def align_sequences_dtw(baseline_seq: np.ndarray, test_seq: np.ndarray) -> Tuple[List[Tuple[int, int]], float]:
    n, m = len(baseline_seq), len(test_seq)
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(baseline_seq[i-1] - test_seq[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )
    
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        candidates = [
            (i-1, j, dtw_matrix[i-1, j]),
            (i, j-1, dtw_matrix[i, j-1]),
            (i-1, j-1, dtw_matrix[i-1, j-1])
        ]
        i, j, _ = min(candidates, key=lambda x: x[2])
    
    return list(reversed(path)), dtw_matrix[n, m]


def compare_frame_metrics(baseline: List[FrameMetrics], test: List[FrameMetrics]) -> DetailedComparison:
    baseline_features = np.array([[m.motion_mean, m.left_right_balance, m.pose_quality_score or 50.0] for m in baseline])
    test_features = np.array([[m.motion_mean, m.left_right_balance, m.pose_quality_score or 50.0] for m in test])
    
    alignment_path, dtw_distance = align_sequences_dtw(baseline_features, test_features)
    
    frame_diffs = []
    for baseline_idx, test_idx in alignment_path:
        b_metric = baseline[baseline_idx]
        t_metric = test[test_idx]
        
        frame_diff = FrameDiff(
            frame_index=test_idx,
            timestamp=t_metric.timestamp_sec,
            motion_diff=abs(b_metric.motion_mean - t_metric.motion_mean),
            balance_diff=abs(b_metric.left_right_balance - t_metric.left_right_balance),
            pose_quality_diff=abs((b_metric.pose_quality_score or 50.0) - (t_metric.pose_quality_score or 50.0)),
            keypoint_distances={}
        )
        frame_diffs.append(frame_diff)
    
    avg_motion_diff = np.mean([f.motion_diff for f in frame_diffs])
    avg_balance_diff = np.mean([f.balance_diff for f in frame_diffs])
    avg_quality_diff = np.mean([f.pose_quality_diff for f in frame_diffs])
    
    overall_similarity = max(0, 100 - (avg_motion_diff + avg_balance_diff + avg_quality_diff) / 3)
    
    major_issues = []
    if avg_motion_diff > 10:
        major_issues.append(f"运动强度差异较大（平均差异: {avg_motion_diff:.1f}）")
    if avg_balance_diff > 15:
        major_issues.append(f"左右平衡差异明显（平均差异: {avg_balance_diff:.1f}）")
    if avg_quality_diff > 20:
        major_issues.append(f"姿态质量差距显著（平均差异: {avg_quality_diff:.1f}）")
    
    return DetailedComparison(
        baseline_name="标准动作",
        test_name="测试动作",
        frame_differences=frame_diffs,
        overall_similarity=overall_similarity,
        major_issues=major_issues
    )


def compare_summaries(baseline: AnalysisSummary, test: AnalysisSummary) -> Dict[str, MetricDiff]:
    metrics = {}
    
    def add_metric(name, baseline_val, test_val):
        if baseline_val is not None and test_val is not None:
            diff = test_val - baseline_val
            pct = (diff / baseline_val * 100) if baseline_val != 0 else 0
            status = "接近" if abs(pct) < 10 else ("偏高" if pct > 0 else "偏低")
            metrics[name] = MetricDiff(name, baseline_val, test_val, diff, pct, status)
    
    add_metric("总分", baseline.total_score, test.total_score)
    add_metric("姿态标准度", baseline.posture_score, test.posture_score)
    add_metric("动作连续性", baseline.continuity_score, test.continuity_score)
    add_metric("动作稳定性", baseline.stability_score, test.stability_score)
    add_metric("节奏控制", baseline.rhythm_score, test.rhythm_score)
    add_metric("平均运动强度", baseline.avg_motion, test.avg_motion)
    add_metric("平均左右平衡差", baseline.avg_left_right_balance, test.avg_left_right_balance)
    
    return metrics
