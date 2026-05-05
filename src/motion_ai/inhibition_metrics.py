"""躯体抑制指标计算模块 - 用于八段锦等健身气功动作分析。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import FrameMetrics, PoseFrame
from .pose import COCO_KEYPOINTS


LIMB_ENDPOINTS = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]
TORSO_POINTS = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
ANGLE_CONFIGS = {
    "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
}


@dataclass
class InhibitionMetrics:
    """躯体抑制指标。"""

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
    research_metric_details: dict[str, dict | float | int | list] = field(default_factory=dict)


def calculate_inhibition_metrics(
    frame_metrics: list[FrameMetrics],
    pose_frames: list[PoseFrame],
    template_baseline: dict | None = None,
    confidence_threshold: float = 0.5,
) -> InhibitionMetrics:
    """基于平滑后的关键点序列计算躯体动作抑制指标。"""
    if not frame_metrics or not pose_frames:
        return InhibitionMetrics(frame_count=len(frame_metrics), confidence_threshold=confidence_threshold)

    coords, confs = _pose_frames_to_arrays(pose_frames)
    coords = _mask_low_confidence(coords, confs, confidence_threshold)
    coords = np.nan_to_num(coords, nan=np.nan, posinf=np.nan, neginf=np.nan)
    valid_frame_mask = np.sum(np.isfinite(coords).all(axis=2), axis=1) >= 10
    valid_frame_count = int(np.sum(valid_frame_mask))
    if valid_frame_count == 0:
        return InhibitionMetrics(
            frame_count=len(frame_metrics),
            valid_frame_count=0,
            confidence_threshold=confidence_threshold,
        )

    keypoint_std = _calculate_keypoint_std(coords)
    keypoint_fluctuation = _calculate_keypoint_fluctuation(coords)
    limb_sway_degree, limb_sway_detail = _calculate_limb_sway(coords)
    torso_center_drift, torso_detail = _calculate_torso_stability(coords)
    limb_dispersion = _calculate_limb_dispersion(coords)
    joint_angle_stability, joint_angle_variation = _calculate_joint_angle_velocity_stats(coords)
    posture_deviation = _calculate_template_distance(coords, template_baseline or {})
    invalid_motion_count = _count_invalid_motions(coords, frame_metrics)
    body_stability_cv, stability_detail = _calculate_body_stability_cv(coords)
    transition_acc_peak, transition_jerk_mean, fluency_detail = _calculate_action_fluency(coords, frame_metrics)
    body_control_ratio, control_detail = _calculate_body_control_ratio(coords, frame_metrics)

    return InhibitionMetrics(
        keypoint_std=keypoint_std,
        keypoint_fluctuation=keypoint_fluctuation,
        torso_center_drift=torso_center_drift,
        limb_dispersion=limb_dispersion,
        template_euclidean_distance=posture_deviation,
        joint_angle_stability=joint_angle_stability,
        limb_sway_degree=limb_sway_degree,
        torso_stability_degree=torso_center_drift,
        posture_deviation=posture_deviation,
        joint_angle_variation=joint_angle_variation,
        invalid_motion_count=invalid_motion_count,
        frame_count=len(frame_metrics),
        valid_frame_count=valid_frame_count,
        confidence_threshold=confidence_threshold,
        limb_sway_detail=limb_sway_detail,
        torso_stability_detail=torso_detail,
        body_stability_cv=body_stability_cv,
        transition_acc_peak=transition_acc_peak,
        transition_jerk_mean=transition_jerk_mean,
        body_control_ratio=body_control_ratio,
        research_metric_details={
            "body_stability": stability_detail,
            "action_fluency": fluency_detail,
            "body_control": control_detail,
        },
    )


def _pose_frames_to_arrays(pose_frames: list[PoseFrame]) -> tuple[np.ndarray, np.ndarray]:
    frame_count = len(pose_frames)
    keypoint_count = len(COCO_KEYPOINTS)
    coords = np.full((frame_count, keypoint_count, 2), np.nan, dtype=np.float64)
    confs = np.zeros((frame_count, keypoint_count), dtype=np.float64)
    for frame_idx, pose in enumerate(pose_frames):
        for kp_idx in range(min(keypoint_count, len(pose.keypoints))):
            point = np.array(pose.keypoints[kp_idx][:2], dtype=np.float64)
            if np.isfinite(point).all():
                coords[frame_idx, kp_idx] = point
            if kp_idx < len(pose.keypoint_confidences):
                confs[frame_idx, kp_idx] = float(pose.keypoint_confidences[kp_idx])
            elif np.isfinite(coords[frame_idx, kp_idx]).all():
                confs[frame_idx, kp_idx] = float(pose.confidence)
    return coords, confs


def _mask_low_confidence(coords: np.ndarray, confs: np.ndarray, threshold: float) -> np.ndarray:
    cleaned = coords.copy()
    cleaned[confs < threshold] = np.nan
    return cleaned


def _calculate_keypoint_std(coords: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for kp_idx, name in enumerate(COCO_KEYPOINTS):
        points = np.nan_to_num(coords[:, kp_idx, :], nan=np.nan, posinf=np.nan, neginf=np.nan)
        if np.sum(np.isfinite(points).all(axis=1)) >= 2:
            std_xy = np.nan_to_num(np.nanstd(points, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
            result[name] = float(np.linalg.norm(std_xy))
    return result


def _calculate_keypoint_fluctuation(coords: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    displacement = _frame_displacement(coords)
    for kp_idx, name in enumerate(COCO_KEYPOINTS):
        values = np.nan_to_num(displacement[:, kp_idx], nan=np.nan, posinf=np.nan, neginf=np.nan)
        if np.sum(np.isfinite(values)) >= 1:
            result[name] = float(np.nan_to_num(np.nanmean(values), nan=0.0, posinf=0.0, neginf=0.0))
    return result


def _calculate_limb_sway(coords: np.ndarray) -> tuple[float, dict[str, dict[str, float]]]:
    """[核心指标] 四肢末端帧间位移的标准差和 RMS。"""
    displacement = _frame_displacement(coords)
    details: dict[str, dict[str, float]] = {}
    all_values: list[np.ndarray] = []
    for name in LIMB_ENDPOINTS:
        values = displacement[:, COCO_KEYPOINTS.index(name)]
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        safe_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        rms = float(np.sqrt(np.mean(np.square(safe_values))))
        std = float(np.std(safe_values))
        details[name] = {"std": std, "rms": rms}
        all_values.append(safe_values)
    if not all_values:
        return 0.0, details
    merged = np.nan_to_num(np.concatenate(all_values), nan=0.0, posinf=0.0, neginf=0.0)
    limb_sway_degree = float(np.std(merged))
    return limb_sway_degree, details


def _calculate_torso_stability(coords: np.ndarray) -> tuple[float, dict[str, float]]:
    """[核心指标] 左右肩髋构成的躯干中心整体方差和最大位移范围。"""
    indices = [COCO_KEYPOINTS.index(name) for name in TORSO_POINTS]
    torso_points = coords[:, indices, :]
    centers = _nanmean_points(torso_points, axis=1)
    valid = np.isfinite(centers).all(axis=1)
    if np.sum(valid) < 2:
        return 0.0, {"variance": 0.0, "max_range": 0.0, "center_speed_var": 0.0}
    valid_centers = np.nan_to_num(centers[valid], nan=0.0, posinf=0.0, neginf=0.0)
    center_delta = np.diff(valid_centers, axis=0)
    center_speed = np.nan_to_num(np.linalg.norm(center_delta, axis=1), nan=0.0, posinf=0.0, neginf=0.0)
    variance = float(np.var(center_speed)) if len(center_speed) else 0.0
    x_range = float(np.nanmax(valid_centers[:, 0]) - np.nanmin(valid_centers[:, 0]))
    y_range = float(np.nanmax(valid_centers[:, 1]) - np.nanmin(valid_centers[:, 1]))
    max_range = float(np.hypot(x_range, y_range))
    return variance, {"variance": variance, "max_range": max_range, "center_speed_var": variance}


def _calculate_body_stability_cv(coords: np.ndarray) -> tuple[float, dict[str, float]]:
    neck_points, hip_centers = _extract_neck_and_hip_centers(coords)
    spine_vectors = np.nan_to_num(neck_points - hip_centers, nan=np.nan, posinf=np.nan, neginf=np.nan)
    valid = np.isfinite(spine_vectors).all(axis=1)
    if np.sum(valid) < 2:
        return 0.0, {"valid_frames": int(np.sum(valid)), "mean_angle_deg": 0.0, "std_angle_deg": 0.0}

    valid_vectors = np.nan_to_num(spine_vectors[valid], nan=0.0, posinf=0.0, neginf=0.0)
    angles_deg = np.degrees(np.arctan2(valid_vectors[:, 0], -valid_vectors[:, 1]))
    angles_deg = np.nan_to_num(angles_deg, nan=0.0, posinf=0.0, neginf=0.0)
    mean_angle = float(np.mean(np.abs(angles_deg))) if len(angles_deg) else 0.0
    std_angle = float(np.std(angles_deg)) if len(angles_deg) else 0.0
    epsilon = 1e-6
    cv_value = float(std_angle / max(abs(mean_angle), epsilon))
    return cv_value, {
        "valid_frames": int(len(angles_deg)),
        "mean_angle_deg": mean_angle,
        "std_angle_deg": std_angle,
    }


def _calculate_action_fluency(coords: np.ndarray, frame_metrics: list[FrameMetrics]) -> tuple[float, float, dict[str, float | int]]:
    hip_centers = _extract_hip_center_series(coords)
    if len(hip_centers) < 4 or not frame_metrics:
        return 0.0, 0.0, {"transition_frames": 0, "acc_samples": 0, "jerk_samples": 0}

    hip_x = _interpolate_nan_series(hip_centers[:, 0])
    hip_y = _interpolate_nan_series(hip_centers[:, 1])
    hip_x = np.nan_to_num(hip_x, nan=0.0, posinf=0.0, neginf=0.0)
    hip_y = np.nan_to_num(hip_y, nan=0.0, posinf=0.0, neginf=0.0)

    velocity = np.column_stack((np.gradient(hip_x), np.gradient(hip_y)))
    acceleration = np.gradient(velocity, axis=0)
    jerk = np.gradient(acceleration, axis=0)

    acc_mag = np.nan_to_num(np.linalg.norm(acceleration, axis=1), nan=0.0, posinf=0.0, neginf=0.0)
    jerk_mag = np.nan_to_num(np.linalg.norm(jerk, axis=1), nan=0.0, posinf=0.0, neginf=0.0)

    transition_mask = _build_stage_mask(frame_metrics, primary_label="过渡", fallback_label="过渡候选")
    if not np.any(transition_mask):
        return 0.0, 0.0, {"transition_frames": 0, "acc_samples": 0, "jerk_samples": 0}

    transition_acc = acc_mag[transition_mask]
    transition_jerk = jerk_mag[transition_mask]
    if transition_acc.size == 0 or transition_jerk.size == 0:
        return 0.0, 0.0, {"transition_frames": int(np.sum(transition_mask)), "acc_samples": int(transition_acc.size), "jerk_samples": int(transition_jerk.size)}

    acc_peak = float(np.max(np.nan_to_num(transition_acc, nan=0.0, posinf=0.0, neginf=0.0)))
    jerk_mean = float(np.mean(np.nan_to_num(transition_jerk, nan=0.0, posinf=0.0, neginf=0.0)))
    return acc_peak, jerk_mean, {
        "transition_frames": int(np.sum(transition_mask)),
        "acc_samples": int(transition_acc.size),
        "jerk_samples": int(transition_jerk.size),
    }


def _calculate_body_control_ratio(coords: np.ndarray, frame_metrics: list[FrameMetrics]) -> tuple[float, dict[str, float | int]]:
    hip_centers = _extract_hip_center_series(coords)
    if len(hip_centers) < 2 or not frame_metrics:
        return 0.0, {"hold_segments": 0, "transition_segments": 0, "hold_amplitude": 0.0, "transition_amplitude": 0.0}

    hold_segments = _collect_stage_segments(frame_metrics, primary_label="定势", fallback_label="定势候选")
    transition_segments = _collect_stage_segments(frame_metrics, primary_label="过渡", fallback_label="过渡候选")
    hold_amplitude = 0.0
    transition_amplitude = 0.0

    for start, end in hold_segments:
        indices = _select_hold_indices(start, end)
        hold_amplitude += _trajectory_amplitude(hip_centers, indices)

    for start, end in transition_segments:
        indices = _select_transition_indices(start, end)
        transition_amplitude += _trajectory_amplitude(hip_centers, indices)

    if hold_amplitude <= 1e-6:
        ratio = 0.0 if transition_amplitude <= 1e-6 else float(np.nan_to_num(transition_amplitude, nan=0.0, posinf=0.0, neginf=0.0))
    else:
        ratio = float(np.nan_to_num(transition_amplitude / max(hold_amplitude, 1e-6), nan=0.0, posinf=0.0, neginf=0.0))
    return ratio, {
        "hold_segments": int(len(hold_segments)),
        "transition_segments": int(len(transition_segments)),
        "hold_amplitude": float(hold_amplitude),
        "transition_amplitude": float(transition_amplitude),
    }


def _calculate_limb_dispersion(coords: np.ndarray) -> dict[str, float]:
    displacement = _frame_displacement(coords)
    limb_groups = {
        "left_arm": ["left_shoulder", "left_elbow", "left_wrist"],
        "right_arm": ["right_shoulder", "right_elbow", "right_wrist"],
        "left_leg": ["left_hip", "left_knee", "left_ankle"],
        "right_leg": ["right_hip", "right_knee", "right_ankle"],
    }
    result: dict[str, float] = {}
    for limb_name, names in limb_groups.items():
        indices = [COCO_KEYPOINTS.index(name) for name in names]
        values = np.nan_to_num(displacement[:, indices].reshape(-1), nan=np.nan, posinf=np.nan, neginf=np.nan)
        values = values[np.isfinite(values)]
        if len(values) > 0:
            result[limb_name] = float(np.std(values))
    return result


def _extract_neck_and_hip_centers(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_shoulder_idx = COCO_KEYPOINTS.index("left_shoulder")
    right_shoulder_idx = COCO_KEYPOINTS.index("right_shoulder")
    left_hip_idx = COCO_KEYPOINTS.index("left_hip")
    right_hip_idx = COCO_KEYPOINTS.index("right_hip")
    shoulder_points = coords[:, [left_shoulder_idx, right_shoulder_idx], :]
    hip_points = coords[:, [left_hip_idx, right_hip_idx], :]
    neck_proxy = _nanmean_points(shoulder_points, axis=1)
    hip_center = _nanmean_points(hip_points, axis=1)
    return neck_proxy, hip_center


def _extract_hip_center_series(coords: np.ndarray) -> np.ndarray:
    _, hip_center = _extract_neck_and_hip_centers(coords)
    return np.nan_to_num(hip_center, nan=np.nan, posinf=np.nan, neginf=np.nan)


def _build_stage_mask(frame_metrics: list[FrameMetrics], primary_label: str, fallback_label: str) -> np.ndarray:
    primary_mask = np.array([_matches_stage(item, primary_label) for item in frame_metrics], dtype=bool)
    if np.any(primary_mask):
        return primary_mask
    return np.array([_matches_stage(item, fallback_label) for item in frame_metrics], dtype=bool)


def _collect_stage_segments(frame_metrics: list[FrameMetrics], primary_label: str, fallback_label: str) -> list[tuple[int, int]]:
    mask = _build_stage_mask(frame_metrics, primary_label=primary_label, fallback_label=fallback_label)
    if mask.size == 0 or not np.any(mask):
        return []
    segments: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            segments.append((start, index - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def _matches_stage(item: FrameMetrics, label: str) -> bool:
    phase_label = str(item.phase_label or "")
    sequence_state = str(item.sequence_state or "")
    return label in phase_label or label in sequence_state


def _select_hold_indices(start: int, end: int, window: int = 10) -> np.ndarray:
    if end < start:
        return np.array([], dtype=int)
    segment = np.arange(start, end + 1, dtype=int)
    if segment.size <= window:
        return segment
    return segment[-window:]


def _select_transition_indices(start: int, end: int, window: int = 10) -> np.ndarray:
    if end < start:
        return np.array([], dtype=int)
    segment = np.arange(start, end + 1, dtype=int)
    if segment.size <= window:
        return segment
    center = segment.size // 2
    half = window // 2
    left = max(0, center - half)
    right = min(segment.size, left + window)
    left = max(0, right - window)
    return segment[left:right]


def _trajectory_amplitude(points: np.ndarray, indices: np.ndarray) -> float:
    if indices.size < 2:
        return 0.0
    selected = np.nan_to_num(points[indices], nan=np.nan, posinf=np.nan, neginf=np.nan)
    valid = np.isfinite(selected).all(axis=1)
    selected = selected[valid]
    if len(selected) < 2:
        return 0.0
    delta = np.diff(selected, axis=0)
    distances = np.nan_to_num(np.linalg.norm(delta, axis=1), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sum(distances))


def _calculate_template_distance(coords: np.ndarray, template_baseline: dict) -> float:
    """[核心指标] 先做时序重采样，再做逐帧 Procrustes 对齐后的测试/模板平均欧氏距离。"""
    template_rows = template_baseline.get("原始逐帧数据") or []
    if len(template_rows) == 0:
        return 0.0
    template_coords = _template_rows_to_array(template_rows)
    if template_coords.size == 0:
        return 0.0
    test_aligned = _resample_sequence(coords, max(len(coords), 2))
    template_aligned = _resample_sequence(template_coords, len(test_aligned))
    frame_distances: list[float] = []
    for test_frame, template_frame in zip(test_aligned, template_aligned):
        distance = _procrustes_frame_distance(test_frame, template_frame)
        if distance is not None:
            frame_distances.append(distance)
    if not frame_distances:
        return 0.0
    return float(np.mean(np.nan_to_num(frame_distances, nan=0.0, posinf=0.0, neginf=0.0)))


def _template_rows_to_array(rows: list[dict]) -> np.ndarray:
    coords = np.full((len(rows), len(COCO_KEYPOINTS), 2), np.nan, dtype=np.float64)
    for row_idx, row in enumerate(rows):
        positions = row.get("keypoint_positions") or row.get("关键点坐标") or {}
        for kp_idx, name in enumerate(COCO_KEYPOINTS):
            value = positions.get(name)
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                point = np.array(value[:2], dtype=np.float64)
                if np.isfinite(point).all():
                    coords[row_idx, kp_idx] = point
    return coords


def _normalize_pose_sequence(coords: np.ndarray) -> np.ndarray:
    normalized = coords.copy()
    left_shoulder_idx = COCO_KEYPOINTS.index("left_shoulder")
    right_shoulder_idx = COCO_KEYPOINTS.index("right_shoulder")
    center_indices = [COCO_KEYPOINTS.index(name) for name in TORSO_POINTS]
    centers = _nanmean_points(normalized[:, center_indices, :], axis=1)
    shoulder_width = np.linalg.norm(normalized[:, left_shoulder_idx, :] - normalized[:, right_shoulder_idx, :], axis=1)
    shoulder_width[~np.isfinite(shoulder_width) | (shoulder_width < 1e-6)] = np.nan
    normalized = normalized - centers[:, None, :]
    normalized = normalized / shoulder_width[:, None, None]
    return normalized


def _resample_sequence(coords: np.ndarray, target_len: int) -> np.ndarray:
    if len(coords) == target_len:
        return coords.copy()
    target_len = max(int(target_len), 1)
    source_x = np.linspace(0.0, 1.0, len(coords))
    target_x = np.linspace(0.0, 1.0, target_len)
    result = np.full((target_len, coords.shape[1], coords.shape[2]), np.nan, dtype=np.float64)
    for kp_idx in range(coords.shape[1]):
        for axis in range(coords.shape[2]):
            series = coords[:, kp_idx, axis]
            valid = np.isfinite(series)
            if np.sum(valid) == 0:
                continue
            if np.sum(valid) == 1:
                result[:, kp_idx, axis] = float(series[valid][0])
            else:
                result[:, kp_idx, axis] = np.interp(target_x, source_x[valid], series[valid])
    return result


def _calculate_joint_angle_velocity_stats(coords: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    """[核心指标] 计算肩、肘、髋、膝角速度方差，并保留角速度 RMS。"""
    stability: dict[str, float] = {}
    variation: dict[str, float] = {}
    for joint_name, names in ANGLE_CONFIGS.items():
        angles = _joint_angle_series(coords, names)
        valid = np.isfinite(angles)
        if np.sum(valid) < 3:
            continue
        filled = np.nan_to_num(_interpolate_nan_series(angles), nan=0.0, posinf=0.0, neginf=0.0)
        velocity = np.nan_to_num(np.diff(filled), nan=0.0, posinf=0.0, neginf=0.0)
        safe_velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
        stability[joint_name] = float(np.var(safe_velocity)) if len(safe_velocity) else 0.0
        variation[joint_name] = float(np.sqrt(np.mean(np.square(safe_velocity)))) if len(safe_velocity) else 0.0
    return stability, variation


def _joint_angle_series(coords: np.ndarray, names: tuple[str, str, str]) -> np.ndarray:
    p1 = coords[:, COCO_KEYPOINTS.index(names[0]), :]
    p2 = coords[:, COCO_KEYPOINTS.index(names[1]), :]
    p3 = coords[:, COCO_KEYPOINTS.index(names[2]), :]
    v1 = p1 - p2
    v2 = p3 - p2
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    denom = norm1 * norm2
    cosine = np.sum(v1 * v2, axis=1) / np.maximum(denom, 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    invalid = (~np.isfinite(denom)) | (denom <= 1e-6)
    angles[invalid] = np.nan
    return angles


def _count_invalid_motions(coords: np.ndarray, frame_metrics: list[FrameMetrics]) -> int:
    """[核心指标] 预期静止段内四肢末端超过速度阈值的异常抖动次数。"""
    displacement = _frame_displacement(coords)
    endpoint_indices = [COCO_KEYPOINTS.index(name) for name in LIMB_ENDPOINTS]
    endpoint_speed = np.nan_to_num(displacement[:, endpoint_indices], nan=0.0, posinf=0.0, neginf=0.0)
    if endpoint_speed.size == 0:
        return 0
    mean_speed = np.nan_to_num(np.mean(endpoint_speed, axis=1), nan=0.0, posinf=0.0, neginf=0.0)
    if len(mean_speed) == 0:
        return 0
    threshold = 5.0
    stationary_mask = np.array(
        [
            (item.sequence_state or "").find("定势") >= 0
            or (item.phase_label or "").find("定势") >= 0
            or float(item.motion_mean or 0.0) < 0.65
            for item in frame_metrics[1:]
        ],
        dtype=bool,
    )
    if len(stationary_mask) != mean_speed.shape[0]:
        stationary_mask = np.ones(mean_speed.shape[0], dtype=bool)
    abnormal = mean_speed > threshold
    abnormal[~stationary_mask] = False
    return int(np.sum(abnormal))


def _procrustes_frame_distance(test_frame: np.ndarray, template_frame: np.ndarray) -> float | None:
    valid = np.isfinite(test_frame).all(axis=1) & np.isfinite(template_frame).all(axis=1)
    if np.sum(valid) < 3:
        return None
    test_points = np.nan_to_num(test_frame[valid], nan=0.0, posinf=0.0, neginf=0.0)
    template_points = np.nan_to_num(template_frame[valid], nan=0.0, posinf=0.0, neginf=0.0)
    test_centered = test_points - np.mean(test_points, axis=0, keepdims=True)
    template_centered = template_points - np.mean(template_points, axis=0, keepdims=True)
    test_norm = np.linalg.norm(test_centered)
    template_norm = np.linalg.norm(template_centered)
    if test_norm <= 1e-6 or template_norm <= 1e-6:
        return None
    test_scaled = test_centered / test_norm
    template_scaled = template_centered / template_norm
    cross_cov = template_scaled.T @ test_scaled
    u, _, vt = np.linalg.svd(cross_cov, full_matrices=False)
    rotation = u @ vt
    aligned_template = template_scaled @ rotation
    distances = np.linalg.norm(test_scaled - aligned_template, axis=1)
    return float(np.mean(np.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=0.0)))


def _frame_displacement(coords: np.ndarray) -> np.ndarray:
    delta = np.nan_to_num(np.diff(coords, axis=0), nan=np.nan, posinf=np.nan, neginf=np.nan)
    valid = np.isfinite(delta).all(axis=2)
    displacement = np.linalg.norm(delta, axis=2)
    displacement[~valid] = np.nan
    return displacement


def _safe_nanvar(values: np.ndarray) -> float:
    """安全方差：过滤 NaN，避免 Mean of empty slice。"""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 0.0
    return float(np.var(finite))


def _nanmean_points(values: np.ndarray, axis: int) -> np.ndarray:
    """安全均值：逐坐标计算有效点均值，全空切片保持 NaN，不触发运行时 warning。"""
    valid = np.isfinite(values)
    counts = np.sum(valid, axis=axis)
    sums = np.nansum(values, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    means[counts == 0] = np.nan
    return means


def _interpolate_nan_series(values: np.ndarray) -> np.ndarray:
    series = values.astype(np.float64, copy=True)
    valid = np.isfinite(series)
    if valid.all():
        return series
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        return np.zeros_like(series)
    if len(valid_indices) == 1:
        return np.full_like(series, float(series[valid_indices[0]]))
    all_indices = np.arange(len(series), dtype=np.float64)
    return np.interp(all_indices, valid_indices.astype(np.float64), series[valid_indices])
