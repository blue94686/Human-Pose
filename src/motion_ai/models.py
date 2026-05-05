from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PoseFrame:
    keypoints: list[list[float]]
    bbox: list[float]
    confidence: float
    keypoint_confidences: list[float] = field(default_factory=list)
    tracking_score: float | None = None
    tracking_mode: str = "detector"
    tracked_keypoints: int = 0
    track_id: int | None = None


@dataclass
class RuleSet:
    raw_text: str
    template_name: str | None = None
    expected_tempo: str = "medium"
    requires_symmetry: bool = False
    requires_upright_torso: bool = False
    requires_hold: bool = False
    focus_body_parts: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)
    action_category: str = "通用动作"


@dataclass
class FrameMetrics:
    frame_index: int
    timestamp_sec: float
    motion_mean: float
    motion_std: float
    left_right_balance: float
    direction_deg: float
    active_region_ratio: float = 0.0
    motion_focus_bbox: list[int] | None = None
    track_id: int | None = None
    trunk_center: list[float] | None = None
    torso_tilt_deg: float | None = None
    shoulder_balance: float | None = None
    arm_raise_left_deg: float | None = None
    arm_raise_right_deg: float | None = None
    arm_symmetry_error: float | None = None
    left_shoulder_angle: float | None = None
    right_shoulder_angle: float | None = None
    left_elbow_angle: float | None = None
    right_elbow_angle: float | None = None
    left_hip_angle: float | None = None
    right_hip_angle: float | None = None
    left_knee_angle: float | None = None
    right_knee_angle: float | None = None
    wrist_speed_left: float | None = None
    wrist_speed_right: float | None = None
    pose_motion: float | None = None
    joint_jitter: float | None = None
    skeleton_stability: float | None = None
    pose_confidence: float | None = None
    keypoint_coverage: float | None = None
    visible_keypoint_count: int | None = None
    pose_quality_score: float | None = None
    pose_quality_label: str | None = None
    keypoint_positions: dict[str, list[float]] = field(default_factory=dict)
    tracking_mode: str | None = None
    phase_label: str | None = None
    transition_score: float | None = None
    template_distance: float | None = None
    sequence_state: str | None = None


@dataclass
class Issue:
    code: str
    severity: str
    title: str
    detail: str
    suggestion: str
    time_range: tuple[float, float] | None = None
    error_type: str = ""
    highlight_metric: str = ""
    expected_value: float | None = None
    actual_value: float | None = None
    overlay_label: str = ""


@dataclass
class AnalysisSummary:
    total_score: float
    posture_score: float
    continuity_score: float
    stability_score: float
    rhythm_score: float
    completeness_score: float
    avg_motion: float
    avg_left_right_balance: float
    avg_torso_tilt: float | None
    avg_arm_symmetry_error: float | None
    avg_arm_raise: float | None = None
    avg_keypoint_coverage: float | None = None
    avg_pose_quality_score: float | None = None
    pose_quality_level: str | None = None
    phases: list[dict[str, Any]] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)
    mode_label: str = "测试者分析"
    pose_backend_reason: str = ""
    evaluator_level: str = "熟练者"
    learner_level: str = "未评定"
    improvement_focus: list[str] = field(default_factory=list)


@dataclass
class ComparisonMetric:
    label: str
    baseline_value: float | None
    current_value: float | None
    delta_value: float | None
    unit: str = ""
    status: str = "未知"
    note: str = ""


@dataclass
class ComparisonPhase:
    baseline_name: str | None
    current_name: str | None
    baseline_duration_sec: float | None
    current_duration_sec: float | None
    delta_duration_sec: float | None
    note: str = ""


@dataclass
class TemplateBaseline:
    """模板基线数据，存储标准动作的完整分析结果。"""

    template_name: str
    source_path: str
    source_type: str
    created_at: str
    fps: float
    frame_count: int
    duration_sec: float
    frames: list[PoseFrame] = field(default_factory=list)
    frame_metrics: list[FrameMetrics] = field(default_factory=list)
    phases: list[dict[str, Any]] = field(default_factory=list)
    summary: AnalysisSummary | None = None
    description: str = ""
    thumbnail_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_python(asdict(self))


@dataclass
class PhaseAlignment:
    """阶段对齐结果。"""

    template_phase: str
    test_phase: str
    template_start_frame: int
    template_end_frame: int
    test_start_frame: int
    test_end_frame: int
    template_duration: float
    test_duration: float
    duration_diff: float
    similarity_score: float
    alignment_quality: str


@dataclass
class DifferenceMetrics:
    """差异指标。"""

    avg_keypoint_distance: float
    max_keypoint_distance: float
    torso_tilt_diff: float | None
    arm_symmetry_diff: float | None
    balance_diff: float | None
    rhythm_diff: float | None
    keypoint_differences: dict[str, float] = field(default_factory=dict)
    phase_differences: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ComparisonIssue:
    """对比分析发现的问题。"""

    issue_type: str
    severity: str
    title: str
    description: str
    template_value: float | None
    test_value: float | None
    difference: float | None
    time_range: tuple[float, float] | None
    affected_keypoints: list[str] = field(default_factory=list)
    suggestion: str = ""
    priority: int = 0


@dataclass
class ComparisonResult:
    """完整对比结果。"""

    baseline_name: str
    baseline_source: str
    baseline_kind: str
    alignment_mode: str
    template_baseline: TemplateBaseline | None = None
    test_result: AnalysisResult | None = None
    phase_alignments: list[PhaseAlignment] = field(default_factory=list)
    difference_metrics: DifferenceMetrics | None = None
    comparison_score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    comparison_issues: list[ComparisonIssue] = field(default_factory=list)
    suggestions_by_priority: dict[int, list[str]] = field(default_factory=dict)
    metrics: list[ComparisonMetric] = field(default_factory=list)
    phases: list[ComparisonPhase] = field(default_factory=list)
    overall_assessment: str = ""
    missing_items: list[str] = field(default_factory=list)
    baseline_summary: dict[str, Any] = field(default_factory=dict)
    sequence_similarity_score: float | None = None
    sequence_similarity_label: str = ""
    snapshot_pairs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _to_python(asdict(self))


@dataclass
class InhibitionMetrics:
    """身体动作抑制指标。"""

    keypoint_std: dict[str, float] = field(default_factory=dict)
    keypoint_fluctuation: dict[str, float] = field(default_factory=dict)
    torso_center_drift: float = 0.0
    limb_dispersion: dict[str, float] = field(default_factory=dict)
    template_euclidean_distance: float = 0.0
    joint_angle_stability: dict[str, float] = field(default_factory=dict)
    limb_sway_degree: float = 0.0
    torso_stability_degree: float = 0.0
    posture_deviation: float = 0.0
    joint_angle_variation: dict[str, float] = field(default_factory=dict)
    invalid_motion_count: int = 0
    frame_count: int = 0
    valid_frame_count: int = 0
    confidence_threshold: float = 0.5
    limb_sway_detail: dict[str, dict[str, float]] = field(default_factory=dict)
    torso_stability_detail: dict[str, float] = field(default_factory=dict)
    body_stability_cv: float = 0.0
    transition_acc_peak: float = 0.0
    transition_jerk_mean: float = 0.0
    body_control_ratio: float = 0.0
    research_metric_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    source: str
    fps: float
    frame_count: int
    processed_frames: int
    used_pose_estimator: bool
    rules: RuleSet
    frame_metrics: list[FrameMetrics]
    issues: list[Issue]
    summary: AnalysisSummary
    output_dir: str
    artifacts: dict[str, str] = field(default_factory=dict)
    comparison: ComparisonResult | None = None
    analysis_mode: str = "test"
    pose_frames: list[PoseFrame] = field(default_factory=list)
    raw_pose_frames: list[PoseFrame] = field(default_factory=list)
    inhibition_metrics: InhibitionMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_python(asdict(self))


def _to_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_python(item) for item in value]
    if isinstance(value, tuple):
        return [_to_python(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value
