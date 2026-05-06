from __future__ import annotations

import json
import math
import time
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover
    savgol_filter = None

from .config import DEFAULT_CHINESE_FONT, DEFAULT_FRAME_WIDTH, DEFAULT_RULES_FILE, FLOW_WIDTH
from .description import DescriptionParser
from .exporter import export_frame_metrics_csv, export_result_excel, export_result_json, export_summary_text
from .exporter import (
    build_prepost_metric_row,
    export_analysis_report_markdown,
    export_inhibition_metrics_xlsx,
    export_pose_keypoints_csv,
    export_prepost_summary_xlsx,
    export_summary_metrics_json,
)
from .inhibition_metrics import calculate_inhibition_metrics
from .models import (
    AnalysisResult,
    AnalysisSummary,
    ComparisonMetric,
    ComparisonPhase,
    ComparisonResult,
    FrameMetrics,
    Issue,
    PoseFrame,
    RuleSet,
)
from .pose import (
    COCO_KEYPOINTS,
    PoseEstimator,
    apply_pose_constraints,
    draw_pose,
    recover_pose_with_flow,
    pose_joint_jitter,
    pose_mean_displacement,
    smooth_pose,
    temporal_smooth_pose,
)
from .summary_text import render_preview_text

ANGLE_CONFIGS = {
    "left_shoulder_angle": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder_angle": ("right_elbow", "right_shoulder", "right_hip"),
    "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_hip_angle": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
}

FrameCallback = Callable[[np.ndarray, dict], None]
StopCallback = Callable[[], bool]
ProgressCallback = Callable[[dict], None]

_TRACKING_MODE_LABELS = {
    "detector": "模型识别",
    "detector+flow": "模型加时序跟踪",
    "flow_fallback": "时序补偿",
    "flow_only": "光流跟踪",
    "hold_last": "沿用上一帧",
    "mediapipe": "MediaPipe 实时识别",
}

_EVALUATOR_LEVELS = {
    "初习者": {
        "label": "初习者",
        "tolerance": 1.35,
        "min_factor": 0.82,
        "transition_min": 48.0,
        "coverage_min": 0.48,
        "stability_min": 0.36,
    },
    "进阶者": {
        "label": "进阶者",
        "tolerance": 1.18,
        "min_factor": 0.90,
        "transition_min": 55.0,
        "coverage_min": 0.56,
        "stability_min": 0.42,
    },
    "熟练者": {
        "label": "熟练者",
        "tolerance": 1.0,
        "min_factor": 1.0,
        "transition_min": 58.0,
        "coverage_min": 0.58,
        "stability_min": 0.46,
    },
}

_TRACKING_MODE_LABELS = {
    "detector": "模型识别",
    "detector+flow": "模型识别+时序跟踪",
    "flow_fallback": "光流兜底",
    "flow_only": "纯光流跟踪",
    "hold_last": "沿用上一帧",
    "mediapipe": "MediaPipe 实时识别",
}

_EVALUATOR_LEVELS = {
    "初习者": {
        "label": "初习者",
        "tolerance": 1.35,
        "min_factor": 0.82,
        "transition_min": 48.0,
        "coverage_min": 0.48,
        "stability_min": 0.36,
    },
    "进阶者": {
        "label": "进阶者",
        "tolerance": 1.18,
        "min_factor": 0.90,
        "transition_min": 55.0,
        "coverage_min": 0.56,
        "stability_min": 0.42,
    },
    "熟练者": {
        "label": "熟练者",
        "tolerance": 1.0,
        "min_factor": 1.0,
        "transition_min": 58.0,
        "coverage_min": 0.58,
        "stability_min": 0.46,
    },
}


class ActionAnalyzer:
    def __init__(
        self,
        weights_path: str | Path | None = None,
        template_file: str | Path | None = None,
        rules_file: str | Path | None = None,
    ) -> None:
        self.description_parser = DescriptionParser(Path(template_file) if template_file else None)
        self.pose_estimator = PoseEstimator(weights_path)
        self.rules_catalog = self._load_rules(Path(rules_file) if rules_file else DEFAULT_RULES_FILE)
        self._overlay_font = self._load_overlay_font()

    def create_realtime_session(self, description_text: str, evaluator_level: str = "熟练者") -> dict:
        """创建摄像头实时分析状态；只保留内存数据，不生成导出文件。"""
        rules = self.description_parser.parse(description_text)
        evaluator_level = self._normalize_evaluator_level(evaluator_level)
        rules = self._rules_for_evaluator_level(rules, evaluator_level)
        return {
            "rules": rules,
            "fps": 20.0,
            "started_at": time.perf_counter(),
            "last_frame_at": None,
            "previous_gray": None,
            "previous_frame": None,
            "previous_pose": None,
            "earlier_pose": None,
            "previous_raw_pose": None,
            "last_pose_frame_index": None,
            "consecutive_pose_loss": 0,
            "motion_history": deque(maxlen=28),
            "jitter_history": deque(maxlen=28),
            "pose_motion_history": deque(maxlen=28),
            "balance_history": deque(maxlen=28),
            "processed_frames": 0,
            "frame_index": -1,
        }

    def analyze_realtime_frame(self, frame: np.ndarray, session: dict) -> tuple[np.ndarray, dict]:
        """分析单帧摄像头画面，返回带中文叠字和骨架的预览帧及实时指标。"""
        rules = session["rules"]
        now = time.perf_counter()
        last_frame_at = session.get("last_frame_at")
        if last_frame_at is not None:
            instant_fps = 1.0 / max(now - float(last_frame_at), 1e-6)
            session["fps"] = float(np.clip(0.82 * float(session.get("fps") or 20.0) + 0.18 * instant_fps, 15.0, 25.0))
        session["last_frame_at"] = now
        fps = float(session.get("fps") or 20.0)

        frame_index = int(session.get("frame_index", -1)) + 1
        session["frame_index"] = frame_index
        frame = self._resize_frame(frame)
        gray_small = self._prepare_gray(frame)
        flow_data = self._compute_flow(session.get("previous_gray"), gray_small)
        motion_history = session["motion_history"]
        jitter_history = session["jitter_history"]
        pose_motion_history = session["pose_motion_history"]
        balance_history = session["balance_history"]
        motion_history.append(flow_data["motion_mean"])

        previous_pose = session.get("previous_pose")
        raw_pose = self.pose_estimator.estimate_with_tracking(frame, previous_pose=previous_pose)
        pose = self._stabilize_pose(
            raw_pose=raw_pose,
            previous_pose=previous_pose,
            previous_raw_pose=session.get("previous_raw_pose"),
            earlier_pose=session.get("earlier_pose"),
            previous_frame=session.get("previous_frame"),
            current_frame=frame,
            frame_index=frame_index,
            last_pose_frame_index=session.get("last_pose_frame_index"),
            frame_stride=1,
            lost_frame_count=int(session.get("consecutive_pose_loss") or 0),
        )
        if raw_pose is not None:
            session["last_pose_frame_index"] = frame_index
            session["consecutive_pose_loss"] = 0
        else:
            session["consecutive_pose_loss"] = int(session.get("consecutive_pose_loss") or 0) + 1

        pose_features = self._extract_pose_features(pose, previous_pose, fps)
        pose_motion = pose_mean_displacement(pose, previous_pose)
        pose_motion_history.append(float(pose_motion or 0.0))
        balance_history.append(float(flow_data["left_right_balance"]))
        joint_jitter = pose_joint_jitter(raw_pose or pose, previous_pose)
        if joint_jitter is not None:
            jitter_history.append(joint_jitter)
        skeleton_stability = self._estimate_skeleton_stability(joint_jitter, pose)
        keypoint_coverage = self._estimate_keypoint_coverage(pose)
        visible_keypoint_count = self._count_visible_keypoints(pose)
        pose_quality_score = self._estimate_pose_quality_score(pose, skeleton_stability, keypoint_coverage)
        pose_quality_label = self._classify_pose_quality(pose_quality_score)
        transition_score = self._estimate_transition_score(
            motion_history,
            pose_motion_history,
            jitter_history,
            balance_history,
            pose,
        )
        sequence_state = self._classify_sequence_state(transition_score, motion_history, pose_motion_history)

        metrics = FrameMetrics(
            frame_index=frame_index,
            timestamp_sec=now - float(session.get("started_at") or now),
            motion_mean=float(flow_data["motion_mean"]),
            motion_std=float(flow_data["motion_std"]),
            left_right_balance=float(flow_data["left_right_balance"]),
            direction_deg=float(flow_data["direction_deg"]),
            active_region_ratio=float(flow_data.get("active_region_ratio") or 0.0),
            motion_focus_bbox=flow_data.get("motion_focus_bbox"),
            torso_tilt_deg=pose_features["torso_tilt_deg"],
            shoulder_balance=pose_features["shoulder_balance"],
            arm_raise_left_deg=pose_features["arm_raise_left_deg"],
            arm_raise_right_deg=pose_features["arm_raise_right_deg"],
            arm_symmetry_error=pose_features["arm_symmetry_error"],
            wrist_speed_left=pose_features["wrist_speed_left"],
            wrist_speed_right=pose_features["wrist_speed_right"],
            pose_motion=pose_motion,
            joint_jitter=joint_jitter,
            skeleton_stability=skeleton_stability,
            pose_confidence=pose.confidence if pose else None,
            keypoint_coverage=keypoint_coverage,
            visible_keypoint_count=visible_keypoint_count,
            pose_quality_score=pose_quality_score,
            pose_quality_label=pose_quality_label,
            keypoint_positions=self._collect_keypoint_positions(pose),
            tracking_mode=pose.tracking_mode if pose else None,
            phase_label=self._infer_live_phase(flow_data["motion_mean"], pose_motion, motion_history),
            transition_score=transition_score,
            sequence_state=sequence_state,
        )
        overlay = self._draw_overlay(frame.copy(), flow_data, pose, metrics, rules, motion_history, jitter_history)
        if self._looks_like_placeholder_frame(frame, pose, flow_data):
            overlay = self._draw_multiline_text(
                overlay,
                [
                    "摄像头未返回真实人物画面",
                    "请检查是否选中了虚拟摄像头、权限是否允许，或人物是否完整入镜。",
                ],
                origin=(max(18, overlay.shape[1] // 2 - 260), max(80, overlay.shape[0] // 2 - 34)),
                color=(255, 255, 255),
                background=(32, 46, 72, 220),
                line_spacing=8,
            )

        processed_frames = int(session.get("processed_frames") or 0) + 1
        session["processed_frames"] = processed_frames
        session["previous_gray"] = gray_small
        session["previous_frame"] = frame.copy()
        session["previous_raw_pose"] = raw_pose
        session["earlier_pose"] = previous_pose
        session["previous_pose"] = pose
        live_issues = self._detect_live_issue_hints(metrics, rules)
        live_suggestions = self._build_live_suggestions(metrics, template_distance=None)
        phase_guidance = self._phase_guidance_for_name(str(metrics.phase_label or ""), rules)
        live_feedback = self._build_live_feedback_card(
            metrics,
            rules=rules,
            live_issues=live_issues,
            live_suggestions=live_suggestions,
            phase_guidance=phase_guidance,
            template_distance=None,
        )

        status = {
            "processed_frames": processed_frames,
            "frame_index": frame_index,
            "total_frames": 0,
            "fps": fps,
            "camera_fps": fps,
            "ai_enabled": self.pose_estimator.available,
            "pose_backend": self.pose_estimator.status.reason,
            "keypoint_positions": metrics.keypoint_positions,
            "visible_keypoint_count": metrics.visible_keypoint_count,
            "pose_quality_label": metrics.pose_quality_label,
            "motion_mean": metrics.motion_mean,
            "left_right_balance": metrics.left_right_balance,
            "phase_label": metrics.phase_label,
            "active_region_ratio": metrics.active_region_ratio,
            "tracking_mode": metrics.tracking_mode,
            "transition_score": metrics.transition_score,
            "sequence_state": metrics.sequence_state,
            "issues": live_issues,
            "suggestions": live_suggestions,
            "phase_guidance": phase_guidance,
            "live_feedback": live_feedback,
            "placeholder_frame": self._looks_like_placeholder_frame(frame, pose, flow_data),
        }
        return overlay, status

    def _looks_like_placeholder_frame(self, frame: np.ndarray, pose: PoseFrame | None, flow_data: dict) -> bool:
        """识别虚拟摄像头/占位图标画面，避免用户误以为真实摄像头已正常工作。"""
        if pose is not None:
            return False
        motion_mean = float(flow_data.get("motion_mean") or 0.0)
        active_ratio = float(flow_data.get("active_region_ratio") or 0.0)
        if motion_mean > 0.02 or active_ratio > 0.002:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dark_ratio = float(np.mean(gray < 35))
        edge_ratio = float(np.mean(cv2.Canny(gray, 60, 140) > 0))
        return dark_ratio > 0.55 and edge_ratio < 0.035

    def analyze_video(
        self,
        source: str | int,
        description_text: str,
        output_dir: str | Path,
        frame_stride: int = 1,
        max_frames: int | None = None,
        preview_limit: int = 12,
        frame_callback: FrameCallback | None = None,
        progress_callback: ProgressCallback | None = None,
        should_stop: StopCallback | None = None,
        analysis_mode: str = "test",
        baseline_payload: dict | None = None,
        export_overlay_video: bool = True,
        evaluator_level: str = "熟练者",
    ) -> AnalysisResult:
        if analysis_mode not in {"template", "test"}:
            raise ValueError(f"analysis_mode 只能为 template 或 test，当前为：{analysis_mode!r}")
        if isinstance(source, str):
            source_path_for_check = Path(source).expanduser()
            if not source_path_for_check.exists():
                raise FileNotFoundError(f"视频文件不存在：{source_path_for_check}")
            if not source_path_for_check.is_file():
                raise RuntimeError(f"视频源不是文件：{source_path_for_check}")
        if not self.pose_estimator.weights_path.exists():
            raise FileNotFoundError(f"姿态权重文件不存在：{self.pose_estimator.weights_path}")
        if analysis_mode == "test" and baseline_payload is None:
            raise ValueError("测试分析缺少模板基线 JSON，请先加载标准动作模板。")
        if analysis_mode == "template":
            # [修复状态污染] 模板生成不继承上一轮测试基线，避免模式间状态串扰。
            baseline_payload = None

        baseline_rows = self._extract_baseline_rows(baseline_payload)

        rules = self.description_parser.parse(description_text)
        evaluator_level = self._normalize_evaluator_level(evaluator_level)
        if analysis_mode == "test":
            rules = self._rules_for_evaluator_level(rules, evaluator_level)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        capture = self._open_capture(source)
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频源：{source}")

        # 验证视频是否可读
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # 尝试读取第一帧验证
        test_ok, test_frame = capture.read()
        if not test_ok or test_frame is None or test_frame.size == 0:
            capture.release()
            raise RuntimeError(f"视频已打开但无法读取帧，可能是编码格式不支持或文件损坏：{source}")

        # 重置到开头
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        expected_processed_frames = self._estimate_processed_frame_count(total_frames, frame_stride, max_frames)
        aligned_baseline_rows = self._align_baseline_rows(baseline_rows, expected_processed_frames)

        overlay_path = output_path / "overlay.mp4"

        writer = None
        previous_gray = None
        previous_frame = None
        previous_pose = None
        earlier_pose = None
        previous_raw_pose = None
        last_pose_frame_index: int | None = None
        consecutive_pose_loss = 0
        frame_metrics: list[FrameMetrics] = []
        pose_frames: list[PoseFrame] = []
        raw_pose_frames: list[PoseFrame] = []
        motion_history: deque[float] = deque(maxlen=28)
        jitter_history: deque[float] = deque(maxlen=28)
        pose_motion_history: deque[float] = deque(maxlen=28)
        balance_history: deque[float] = deque(maxlen=28)

        processed_frames = 0
        frame_index = -1

        try:
            while True:
                if should_stop and should_stop():
                    break

                ok, frame = capture.read()
                if not ok:
                    break

                frame_index += 1
                if frame_index % max(1, frame_stride) != 0:
                    continue
                if max_frames is not None and processed_frames >= max_frames:
                    break

                frame = self._resize_frame(frame)
                gray_small = self._prepare_gray(frame)
                flow_data = self._compute_flow(previous_gray, gray_small)
                motion_history.append(flow_data["motion_mean"])

                raw_pose = self.pose_estimator.estimate_with_tracking(frame, previous_pose=previous_pose)
                pose = self._stabilize_pose(
                    raw_pose=raw_pose,
                    previous_pose=previous_pose,
                    previous_raw_pose=previous_raw_pose,
                    earlier_pose=earlier_pose,
                    previous_frame=previous_frame,
                    current_frame=frame,
                    frame_index=frame_index,
                    last_pose_frame_index=last_pose_frame_index,
                    frame_stride=frame_stride,
                    lost_frame_count=consecutive_pose_loss,
                )
                if raw_pose is not None:
                    last_pose_frame_index = frame_index
                    consecutive_pose_loss = 0
                else:
                    consecutive_pose_loss += 1

                pose_features = self._extract_pose_features(pose, previous_pose, fps)
                pose_motion = pose_mean_displacement(pose, previous_pose)
                pose_motion_history.append(float(pose_motion or 0.0))
                balance_history.append(float(flow_data["left_right_balance"]))
                joint_jitter = pose_joint_jitter(raw_pose or pose, previous_pose)
                if joint_jitter is not None:
                    jitter_history.append(joint_jitter)
                skeleton_stability = self._estimate_skeleton_stability(joint_jitter, pose)
                keypoint_coverage = self._estimate_keypoint_coverage(pose)
                visible_keypoint_count = self._count_visible_keypoints(pose)
                pose_quality_score = self._estimate_pose_quality_score(pose, skeleton_stability, keypoint_coverage)
                pose_quality_label = self._classify_pose_quality(pose_quality_score)
                transition_score = self._estimate_transition_score(
                    motion_history,
                    pose_motion_history,
                    jitter_history,
                    balance_history,
                    pose,
                )
                sequence_state = self._classify_sequence_state(transition_score, motion_history, pose_motion_history)

                metrics = FrameMetrics(
                    frame_index=frame_index,
                    timestamp_sec=frame_index / fps if fps else 0.0,
                    motion_mean=float(flow_data["motion_mean"]),
                    motion_std=float(flow_data["motion_std"]),
                    left_right_balance=float(flow_data["left_right_balance"]),
                    direction_deg=float(flow_data["direction_deg"]),
                    active_region_ratio=float(flow_data.get("active_region_ratio") or 0.0),
                    motion_focus_bbox=flow_data.get("motion_focus_bbox"),
                    track_id=pose.track_id if pose else (raw_pose.track_id if raw_pose else None),
                    trunk_center=pose_features["trunk_center"],
                    torso_tilt_deg=pose_features["torso_tilt_deg"],
                    shoulder_balance=pose_features["shoulder_balance"],
                    arm_raise_left_deg=pose_features["arm_raise_left_deg"],
                    arm_raise_right_deg=pose_features["arm_raise_right_deg"],
                    arm_symmetry_error=pose_features["arm_symmetry_error"],
                    left_shoulder_angle=pose_features["left_shoulder_angle"],
                    right_shoulder_angle=pose_features["right_shoulder_angle"],
                    left_elbow_angle=pose_features["left_elbow_angle"],
                    right_elbow_angle=pose_features["right_elbow_angle"],
                    left_hip_angle=pose_features["left_hip_angle"],
                    right_hip_angle=pose_features["right_hip_angle"],
                    left_knee_angle=pose_features["left_knee_angle"],
                    right_knee_angle=pose_features["right_knee_angle"],
                    wrist_speed_left=pose_features["wrist_speed_left"],
                    wrist_speed_right=pose_features["wrist_speed_right"],
                    pose_motion=pose_motion,
                    joint_jitter=joint_jitter,
                    skeleton_stability=skeleton_stability,
                    pose_confidence=pose.confidence if pose else None,
                    keypoint_coverage=keypoint_coverage,
                    visible_keypoint_count=visible_keypoint_count,
                    pose_quality_score=pose_quality_score,
                    pose_quality_label=pose_quality_label,
                    keypoint_positions=self._collect_keypoint_positions(pose),
                    tracking_mode=pose.tracking_mode if pose else None,
                    phase_label=self._infer_live_phase(flow_data["motion_mean"], pose_motion, motion_history),
                    transition_score=transition_score,
                    template_distance=None,
                    sequence_state=sequence_state,
                )
                baseline_row = None
                baseline_keypoint_positions: dict[str, list[float]] = {}
                template_distance = None
                if analysis_mode == "test":
                    # [修复“只分析模板”]
                    # 测试模式必须以测试视频帧为主循环，每处理一帧测试视频，就取时序上对应的一帧模板基线。
                    # 这样模板 JSON 只作为对照，不再反客为主抢占分析入口。
                    baseline_row = self._baseline_row_at_index(aligned_baseline_rows, processed_frames)
                    baseline_keypoint_positions = self._extract_keypoint_positions_from_row(baseline_row)
                    template_distance = self._frame_template_distance(
                        metrics.keypoint_positions,
                        baseline_keypoint_positions,
                    )
                    metrics.template_distance = template_distance
                frame_metrics.append(metrics)
                raw_pose_frames.append(
                    PoseFrame(
                        keypoints=[list(point) for point in raw_pose.keypoints],
                        bbox=list(raw_pose.bbox),
                        confidence=raw_pose.confidence,
                        keypoint_confidences=list(raw_pose.keypoint_confidences),
                        tracking_score=raw_pose.tracking_score,
                        tracking_mode=raw_pose.tracking_mode,
                        tracked_keypoints=raw_pose.tracked_keypoints,
                        track_id=raw_pose.track_id,
                    )
                    if raw_pose is not None
                    else PoseFrame(
                        keypoints=[],
                        bbox=[],
                        confidence=0.0,
                        keypoint_confidences=[],
                        tracking_score=None,
                        tracking_mode="missing",
                        tracked_keypoints=0,
                        track_id=None,
                    )
                )
                pose_frames.append(
                    PoseFrame(
                        keypoints=[list(point) for point in pose.keypoints],
                        bbox=list(pose.bbox),
                        confidence=pose.confidence,
                        keypoint_confidences=list(pose.keypoint_confidences),
                        tracking_score=pose.tracking_score,
                        tracking_mode=pose.tracking_mode,
                        tracked_keypoints=pose.tracked_keypoints,
                        track_id=pose.track_id,
                    )
                    if pose is not None
                    else PoseFrame(
                        keypoints=[],
                        bbox=[],
                        confidence=0.0,
                        keypoint_confidences=[],
                        tracking_score=None,
                        tracking_mode="missing",
                        tracked_keypoints=0,
                        track_id=None,
                    )
                )

                overlay = self._draw_overlay(
                    frame.copy(),
                    flow_data,
                    pose,
                    metrics,
                    rules,
                    motion_history,
                    jitter_history,
                )

                if export_overlay_video and writer is None:
                    writer = self._build_writer(overlay_path, fps, overlay.shape[1], overlay.shape[0])
                if export_overlay_video and writer is not None:
                    writer.write(overlay)

                processed_frames += 1
                previous_gray = gray_small
                previous_frame = frame.copy()
                previous_raw_pose = raw_pose
                earlier_pose = previous_pose
                previous_pose = pose

                live_issues = self._detect_live_issue_hints(metrics, rules)
                live_suggestions = self._build_live_suggestions(metrics, template_distance)
                phase_guidance = self._phase_guidance_for_name(str(metrics.phase_label or ""), rules)
                live_feedback = self._build_live_feedback_card(
                    metrics,
                    rules=rules,
                    live_issues=live_issues,
                    live_suggestions=live_suggestions,
                    phase_guidance=phase_guidance,
                    template_distance=template_distance,
                )
                progress_value = int(round(processed_frames * 100 / total_frames)) if total_frames > 0 else 0
                status_payload = {
                    "processed_frames": processed_frames,
                    "progress": progress_value,
                    "frame_index": frame_index,
                    "total_frames": total_frames,
                    "fps": fps,
                    "keypoint_positions": metrics.keypoint_positions,
                    "visible_keypoint_count": metrics.visible_keypoint_count,
                    "pose_quality_label": metrics.pose_quality_label,
                    "motion_mean": metrics.motion_mean,
                    "left_right_balance": metrics.left_right_balance,
                    "phase_label": metrics.phase_label,
                    "active_region_ratio": metrics.active_region_ratio,
                    "tracking_mode": metrics.tracking_mode,
                    "transition_score": metrics.transition_score,
                    "sequence_state": metrics.sequence_state,
                    "issues": live_issues,
                    "suggestions": live_suggestions,
                    "phase_guidance": phase_guidance,
                    "live_feedback": live_feedback,
                    "baseline_keypoint_positions": baseline_keypoint_positions,
                    "template_distance": template_distance,
                    "template_frame_index": int(baseline_row.get("frame_index") or 0) if baseline_row else None,
                }
                if frame_callback:
                    frame_callback(overlay, status_payload)
                if progress_callback:
                    progress_callback(status_payload)
        finally:
            # [修复分析完成后自动退出]
            # 这里只释放当前分析任务自己的 capture / writer。
            # 所有画面已经统一回流到 Qt 的 QLabel；如果在这里执行 destroyAllWindows，
            # 会把 OpenCV 的全局窗口生命周期重新带进 Qt 主线程，部分环境下会触发异常收尾或主窗体连带退出。
            capture.release()
            if writer is not None:
                writer.release()

        if not frame_metrics:
            raise RuntimeError("视频已打开，但没有成功读取到可分析的帧。")

        # [S-G滤波平滑] 主循环结束后对完整关键点序列做线性插值 + Savitzky-Golay 平滑，
        # 后续阶段划分、指标计算和 smoothed_keypoints.csv 均使用清洗后的轨迹。
        pose_frames = self._smooth_pose_sequence(pose_frames)
        self._refresh_metrics_from_smoothed_poses(frame_metrics, pose_frames, fps)

        phases = self._segment_phases(frame_metrics, rules)
        self._apply_phase_labels(frame_metrics, phases)
        for phase in phases:
            phase_name = str(phase.get("name") or "")
            phase["guidance"] = self._phase_guidance_for_name(phase_name, rules)
        perfect_match_override_reason = self._perfect_match_override_reason(
            frame_metrics,
            source=source,
            baseline_payload=baseline_payload,
            analysis_mode=analysis_mode,
        )
        issues = [] if perfect_match_override_reason else self._evaluate_issues(frame_metrics, rules, fps, phases)
        summary = self._build_summary(
            frame_metrics,
            issues,
            rules,
            phases,
            analysis_mode=analysis_mode,
            evaluator_level=evaluator_level,
            perfect_match_override_reason=perfect_match_override_reason,
        )
        phase_snapshots = self._generate_phase_snapshots(
            source=source,
            metrics=frame_metrics,
            phases=phases,
            issues=issues,
            output_dir=output_path / "phase_snapshots",
        )
        comparison = self._build_comparison_result(
            metrics=frame_metrics,
            summary=summary,
            rules=rules,
            phases=phases,
            source=str(source),
            fps=fps,
            analysis_mode=analysis_mode,
            baseline_payload=baseline_payload,
            perfect_match_override_reason=perfect_match_override_reason,
            phase_snapshots=phase_snapshots,
        )

        # 计算躯体抑制指标
        inhibition_metrics = None
        if self.pose_estimator.available and pose_frames:
            inhibition_metrics = calculate_inhibition_metrics(
                frame_metrics=frame_metrics,
                pose_frames=pose_frames,
                template_baseline=baseline_payload,
                confidence_threshold=0.5,
            )

        result = AnalysisResult(
            source=str(source),
            fps=float(fps),
            frame_count=total_frames,
            processed_frames=processed_frames,
            used_pose_estimator=self.pose_estimator.available,
            rules=rules,
            frame_metrics=frame_metrics,
            issues=issues,
            summary=summary,
            output_dir=str(output_path),
            comparison=comparison,
            analysis_mode=analysis_mode,
            pose_frames=pose_frames,
            raw_pose_frames=raw_pose_frames,
            inhibition_metrics=inhibition_metrics,
        )

        if analysis_mode == "template":
            excel_report = output_path / "01_模板基础报表.xlsx"
            compact_data_json = output_path / "02_模板综合数据.json"
            summary_txt = output_path / "03_模板结果摘要.txt"
            action_guidance_txt = output_path / "04_模板动作提示与优化方向.txt"
            template_baseline_json = output_path / "05_模板基线数据.json"
            frame_metrics_csv = output_path / "06_模板逐帧关键点数据.csv"
        else:
            excel_report = output_path / "01_测试基础报表.xlsx"
            compact_data_json = output_path / "02_测试综合数据.json"
            summary_txt = output_path / "03_测试结果摘要.txt"
            action_guidance_txt = output_path / "04_测试动作提示与优化方向.txt"
            template_baseline_json = output_path / "05_模板基线数据.json"
            frame_metrics_csv = output_path / "06_测试逐帧关键点数据.csv"

        raw_keypoints_csv = output_path / "07_原始关键点.csv"
        smoothed_keypoints_csv = output_path / "08_平滑关键点.csv"
        inhibition_metrics_xlsx = output_path / "09_抑制指标明细.xlsx"
        summary_metrics_json = output_path / "10_摘要指标.json"
        analysis_report_md = output_path / "11_分析报告.md"
        analysis_result_xlsx = output_path / "12_完整分析结果.xlsx"
        analysis_summary_xlsx = output_path / "13_分析摘要.xlsx"
        experiment_summary_xlsx = output_path / "14_科研实验总表.xlsx"
        prepost_summary_xlsx = output_path / "15_前后测改善量总表.xlsx"

        result.artifacts = {
            "excel_report": str(excel_report),
            "compact_data_json": str(compact_data_json),
            "analysis_summary_txt": str(summary_txt),
            "action_guidance_txt": str(action_guidance_txt),
            "frame_metrics_csv": str(frame_metrics_csv),
            "raw_keypoints_csv": str(raw_keypoints_csv),
            "smoothed_keypoints_csv": str(smoothed_keypoints_csv),
            "inhibition_metrics_xlsx": str(inhibition_metrics_xlsx),
            "summary_metrics_json": str(summary_metrics_json),
            "analysis_report_md": str(analysis_report_md),
            "analysis_result_xlsx": str(analysis_result_xlsx),
            "analysis_summary_xlsx": str(analysis_summary_xlsx),
            "experiment_summary_xlsx": str(experiment_summary_xlsx),
            "prepost_summary_xlsx": str(prepost_summary_xlsx),
        }
        if export_overlay_video and overlay_path.exists():
            result.artifacts["overlay_video"] = str(overlay_path)
        if analysis_mode == "template":
            result.artifacts["template_baseline_json"] = str(template_baseline_json)
        summary_text = render_preview_text(result)
        export_result_json(result, compact_data_json, include_frame_rows=True)
        export_result_excel(result, excel_report)
        export_result_excel(result, analysis_result_xlsx)
        export_result_excel(result, analysis_summary_xlsx)
        export_result_excel(result, experiment_summary_xlsx)
        export_summary_text(summary_text, summary_txt)
        export_summary_text(self._render_action_guidance_text(result), action_guidance_txt)
        export_frame_metrics_csv(result.frame_metrics, frame_metrics_csv)
        export_pose_keypoints_csv(result.raw_pose_frames, result.frame_metrics, raw_keypoints_csv, is_smoothed=False)
        export_pose_keypoints_csv(result.pose_frames, result.frame_metrics, smoothed_keypoints_csv, is_smoothed=True)
        export_inhibition_metrics_xlsx(result, inhibition_metrics_xlsx)
        export_summary_metrics_json(result, summary_metrics_json)
        export_analysis_report_markdown(result, analysis_report_md)
        if analysis_mode == "test":
            prepost_rows = [
                build_prepost_metric_row(
                    result,
                    subject_id=Path(result.source).stem,
                    test_type="后测",
                    baseline_name=rules.template_name or "",
                )
            ]
            export_prepost_summary_xlsx(prepost_rows, prepost_summary_xlsx)
        if analysis_mode == "template":
            template_baseline_json.write_text(
                json.dumps(
                    self.build_template_payload(
                        result,
                        template_name=rules.template_name,
                        phase_snapshots=phase_snapshots,
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        return result

    def _render_action_guidance_text(self, result: AnalysisResult) -> str:
        lines = [
            "动作提示与优化方向",
            "",
            f"分析模式：{'模板数据生成' if result.analysis_mode == 'template' else '测试者动作评估'}",
            f"模板名称：{result.rules.template_name or '未匹配模板'}",
            f"动作类别：{result.rules.action_category}",
            f"评价标准：{result.summary.evaluator_level}",
            f"测试者水平：{result.summary.learner_level}",
            f"综合评分：{result.summary.total_score:.1f}",
            "",
        ]
        if result.analysis_mode == "template":
            lines.extend(
                [
                    "模板提示：",
                    "- 当前文件为标准模板数据生成结果。",
                    "- 后续请在测试者动作评估中加载模板基线，进行左右对比。",
                    "",
                    "模板阶段：",
                ]
            )
            if result.summary.phases:
                for phase in result.summary.phases:
                    lines.append(
                        f"- {phase.get('name') or '未命名阶段'}：{phase.get('start_sec', 0):.1f}s - {phase.get('end_sec', 0):.1f}s"
                    )
                lines.extend(["", "阶段录入建议："])
                lines.extend(self._phase_guidance_lines(result))
            else:
                lines.append("- 未识别到清晰阶段。")
            return "\n".join(lines)

        lines.append("优先优化方向：")
        if result.summary.improvement_focus:
            lines.extend([f"- {item}" for item in result.summary.improvement_focus])
        else:
            lines.append("- 当前未发现明显问题，可继续提高动作完整度和模板相似度。")
        lines.extend(["", "问题与修改建议："])
        if result.issues:
            for index, issue in enumerate(result.issues, start=1):
                time_text = ""
                if issue.time_range:
                    time_text = f"（{issue.time_range[0]:.1f}s - {issue.time_range[1]:.1f}s）"
                metric_text = ""
                if issue.highlight_metric:
                    current = "未记录" if issue.actual_value is None else f"{issue.actual_value:.2f}"
                    expected = "未设定" if issue.expected_value is None else f"{issue.expected_value:.2f}"
                    metric_text = f"；指标 {issue.highlight_metric}：当前 {current}，目标 {expected}"
                lines.append(f"{index}. {issue.title}{time_text}")
                lines.append(f"   问题：{issue.detail}{metric_text}")
                lines.append(f"   修改：{issue.suggestion}")
                lines.append(f"   练习目标：{self._issue_practice_goal(issue, result.rules.action_category)}")
        else:
            lines.append("- 当前未发现明显问题。")
        lines.extend(["", "按阶段修改提示："])
        lines.extend(self._phase_guidance_lines(result))
        lines.extend(["", "综合建议："])
        lines.extend([f"- {item}" for item in result.summary.advice] or ["- 保持当前动作节奏和稳定性。"])
        return "\n".join(lines)

    def _phase_guidance_for_name(self, phase_name: str, rules: RuleSet) -> str:
        if phase_name == "准备":
            return "准备阶段：沉肩垂肘，呼吸调匀，立身中正，为起势蓄势。"
        if phase_name == "起势":
            return "起势阶段：动作缓起缓落，意领形随，保持脊柱拔伸与重心稳定。"
        if phase_name == "过渡":
            return "过渡阶段：衔接需圆活不断，避免抢节奏，做到上下相随、内外合一。"
        if phase_name == "定势":
            return "定势阶段：体会气沉丹田，微微沉肩坠肘，减少无效小动作。"
        if phase_name == "发力":
            return "发力阶段：在稳定中完成舒展，注意匀称展开，不可僵硬突冲。"
        if phase_name == "收势":
            return "收势阶段：缓缓回落，神形合一，保持动作收束后的稳定与连贯。"
        return f"{rules.action_category or '动作'}阶段：保持中正、松沉、匀缓，注意细节控制。"

    def _issue_practice_goal(self, issue: Issue, category: str) -> str:
        title = f"{issue.error_type or ''}{issue.title}"
        if "手臂" in title or "抬" in title:
            return "每次抬臂先做到左右同时启动、同时到位，再检查肩部是否放松。"
        if "重心" in title or "平衡" in title:
            return "完成动作时让头、躯干和支撑面保持在同一中线，避免一侧先抢动作。"
        if "躯干" in title or "转体" in title:
            return "先固定肩髋连线，再用腰胯带动动作，减少上身前冲或侧倒。"
        if "节奏" in title or "衔接" in title or "收势" in title:
            return "把动作拆成启动、过渡、到位、收束四段，每段速度均匀，不突然加速。"
        if "骨架" in title or "识别" in title:
            return "复测前保持全身入镜、单人画面、固定机位，先保证识别稳定。"
        if category == "健身气功":
            return "先保证身形中正、呼吸自然、慢而不断，再提高动作幅度。"
        return "按模板视频分段对照，先修正最集中的问题时间段，再整套连贯完成。"

    def _phase_guidance_lines(self, result: AnalysisResult) -> list[str]:
        phases = result.summary.phases or []
        if not phases:
            return ["- 未形成稳定阶段，请录入完整动作，包含准备、起势、过渡、定势/发力和收势。"]

        phase_targets = {
            "准备": "站稳并完整入镜，肩髋放正，重心落在两脚之间。",
            "起势": "双臂同时启动，速度缓慢均匀，躯干不要前冲。",
            "过渡": "手臂、躯干和重心同步移动，连接处不断劲、不抢拍。",
            "定势": "关键姿态短暂停留，肩放松、肘不过度外翻，身体中线稳定。",
            "发力": "发力方向清楚，先稳定支撑脚和躯干，再完成手臂或转体动作。",
            "收势": "速度逐步放慢，动作收完整，避免最后几帧突然下落或偏向一侧。",
        }
        lines: list[str] = []
        for phase in phases:
            name = str(phase.get("name") or "未命名阶段")
            start = float(phase.get("start_sec") or 0.0)
            end = float(phase.get("end_sec") or start)
            target = phase_targets.get(name, "保持路线清楚、速度均匀，并与模板动作节奏对齐。")
            lines.append(f"- {name}（{start:.1f}s - {end:.1f}s）：{target}")
        return lines

    def build_template_payload(
        self,
        result: AnalysisResult,
        template_name: str | None = None,
        phase_snapshots: list[dict] | None = None,
    ) -> dict:
        summary = result.summary
        return {
            "模板名称": template_name or result.rules.template_name or "未命名模板",
            "来源视频": result.source,
            "分析模式": "template",
            "姿态识别已启用": result.used_pose_estimator,
            "姿态后端状态": self.pose_estimator.status.reason,
            "规则信息": {
                "原始描述": result.rules.raw_text,
                "节奏要求": result.rules.expected_tempo,
                "需要对称": result.rules.requires_symmetry,
                "需要躯干中正": result.rules.requires_upright_torso,
                "需要定势停顿": result.rules.requires_hold,
                "关注部位": result.rules.focus_body_parts,
                "关键词": result.rules.keywords,
                "阈值": result.rules.thresholds,
            },
            "模板摘要": {
                "总分": summary.total_score,
                "姿态标准度": summary.posture_score,
                "动作连续性": summary.continuity_score,
                "动作稳定性": summary.stability_score,
                "节奏控制": summary.rhythm_score,
                "动作完整性": summary.completeness_score,
                "平均运动强度": summary.avg_motion,
                "平均左右平衡差": summary.avg_left_right_balance,
                "平均躯干倾斜角度": summary.avg_torso_tilt,
                "平均双臂对称误差": summary.avg_arm_symmetry_error,
                "平均上举角度": summary.avg_arm_raise,
                "平均关键点覆盖率": summary.avg_keypoint_coverage,
                "平均姿态质量分": summary.avg_pose_quality_score,
                "平均衔接评分": self._average_transition_score(result.frame_metrics),
                "姿态质量等级": summary.pose_quality_level,
            },
            "模板阶段": summary.phases,
            "阶段快照": phase_snapshots or [],
            "模板建议": summary.advice,
            "模板问题": [
                {
                    "问题编码": issue.code,
                    "严重程度": issue.severity,
                    "错误类型": issue.error_type,
                    "问题标题": issue.title,
                    "问题说明": issue.detail,
                    "关键指标": issue.highlight_metric,
                    "预期值": issue.expected_value,
                    "实际值": issue.actual_value,
                    "问题时间区间": list(issue.time_range) if issue.time_range else None,
                }
                for issue in result.issues
            ],
            "原始逐帧数据": [asdict(item) for item in result.frame_metrics],
        }

    def _open_capture(self, source: str | int) -> cv2.VideoCapture:
        """打开视频捕获，支持多种后端和错误恢复"""
        if not isinstance(source, int):
            # 对于视频文件，尝试多种方式打开
            capture = cv2.VideoCapture(source)
            if capture.isOpened():
                # 验证是否能读取帧
                ok, test_frame = capture.read()
                if ok and test_frame is not None and test_frame.size > 0:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开头
                    return capture
                capture.release()

            # 尝试使用FFMPEG后端
            capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if capture.isOpened():
                ok, test_frame = capture.read()
                if ok and test_frame is not None and test_frame.size > 0:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    return capture
                capture.release()

            # 最后尝试默认方式
            return cv2.VideoCapture(source)

        # 摄像头设备
        backends: list[int | None] = []
        avfoundation = getattr(cv2, "CAP_AVFOUNDATION", None)
        msmf = getattr(cv2, "CAP_MSMF", None)
        dshow = getattr(cv2, "CAP_DSHOW", None)
        for backend in (avfoundation, msmf, dshow, None):
            if backend in backends:
                continue
            backends.append(backend)

        for backend in backends:
            capture = cv2.VideoCapture(source) if backend is None else cv2.VideoCapture(source, backend)
            if capture.isOpened():
                return capture
            capture.release()
        return cv2.VideoCapture(source)

    def _load_rules(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _normalize_evaluator_level(self, value: str | None) -> str:
        text = (value or "").strip()
        if "初" in text:
            return "初习者"
        if "进" in text or "基础" in text:
            return "进阶者"
        return "熟练者"

    def _level_profile(self, level: str) -> dict:
        return _EVALUATOR_LEVELS.get(self._normalize_evaluator_level(level), _EVALUATOR_LEVELS["熟练者"])

    def _rules_for_evaluator_level(self, rules: RuleSet, evaluator_level: str) -> RuleSet:
        profile = self._level_profile(evaluator_level)
        tolerance = float(profile["tolerance"])
        min_factor = float(profile["min_factor"])
        relaxed_thresholds = dict(rules.thresholds)

        for key in [
            "torso_tilt_max_deg",
            "arm_symmetry_error_max",
            "left_right_balance_max",
            "motion_mean_max_slow",
            "motion_std_max",
            "joint_jitter_max",
        ]:
            if key in relaxed_thresholds:
                relaxed_thresholds[key] = float(relaxed_thresholds[key]) * tolerance

        if "arm_raise_min_deg" in relaxed_thresholds:
            relaxed_thresholds["arm_raise_min_deg"] = float(relaxed_thresholds["arm_raise_min_deg"]) * min_factor
        if "motion_mean_min_fast" in relaxed_thresholds:
            relaxed_thresholds["motion_mean_min_fast"] = float(relaxed_thresholds["motion_mean_min_fast"]) * min_factor

        relaxed_thresholds["transition_score_min"] = float(profile["transition_min"])
        relaxed_thresholds["keypoint_coverage_min"] = float(profile["coverage_min"])
        relaxed_thresholds["skeleton_stability_min"] = float(profile["stability_min"])

        return RuleSet(
            raw_text=rules.raw_text,
            template_name=rules.template_name,
            expected_tempo=rules.expected_tempo,
            requires_symmetry=rules.requires_symmetry,
            requires_upright_torso=rules.requires_upright_torso,
            requires_hold=rules.requires_hold,
            focus_body_parts=list(rules.focus_body_parts),
            thresholds=relaxed_thresholds,
            keywords=list(rules.keywords),
            action_category=rules.action_category,
        )

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= DEFAULT_FRAME_WIDTH:
            return frame
        target_height = int(height * (DEFAULT_FRAME_WIDTH / width))
        return cv2.resize(frame, (DEFAULT_FRAME_WIDTH, target_height), interpolation=cv2.INTER_AREA)

    def _prepare_gray(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        target_width = min(FLOW_WIDTH, width)
        target_height = max(1, int(height * (target_width / width)))
        resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    def _compute_flow(self, previous_gray: np.ndarray | None, gray: np.ndarray) -> dict:
        if previous_gray is None or previous_gray.shape != gray.shape:
            h, w = gray.shape
            return {
                "flow": np.zeros((h, w, 2), dtype=np.float32),
                "motion_mean": 0.0,
                "motion_std": 0.0,
                "left_right_balance": 0.0,
                "direction_deg": 0.0,
                "active_region_ratio": 0.0,
                "active_mask": np.zeros((h, w), dtype=np.uint8),
                "motion_focus_bbox": None,
            }

        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        active_mask = self._build_active_motion_mask(magnitude)
        active_region_ratio = float(np.count_nonzero(active_mask)) / max(active_mask.size, 1)
        center_weight = self._build_center_weight(magnitude.shape)
        effective_weights = center_weight * (0.25 + 0.75 * (active_mask > 0).astype(np.float32))
        motion_mean = self._weighted_mean(magnitude, effective_weights)
        motion_std = self._weighted_std(magnitude, effective_weights, motion_mean)

        midpoint = magnitude.shape[1] // 2
        left_energy = (
            self._weighted_mean(magnitude[:, :midpoint], effective_weights[:, :midpoint]) if midpoint > 0 else 0.0
        )
        right_energy = (
            self._weighted_mean(magnitude[:, midpoint:], effective_weights[:, midpoint:])
            if midpoint < magnitude.shape[1]
            else 0.0
        )
        denominator = max(left_energy + right_energy, 1e-6)
        left_right_balance = abs(left_energy - right_energy) / denominator

        mean_vector = np.array(
            [
                self._weighted_mean(flow[..., 0], effective_weights),
                self._weighted_mean(flow[..., 1], effective_weights),
            ],
            dtype=np.float32,
        )
        direction_deg = float(math.degrees(math.atan2(mean_vector[1], mean_vector[0])))
        if math.isnan(direction_deg):
            direction_deg = 0.0

        return {
            "flow": flow,
            "motion_mean": motion_mean,
            "motion_std": motion_std,
            "left_right_balance": left_right_balance,
            "direction_deg": direction_deg,
            "active_region_ratio": active_region_ratio,
            "active_mask": active_mask,
            "motion_focus_bbox": self._mask_to_bbox(active_mask),
        }

    def _build_active_motion_mask(self, magnitude: np.ndarray) -> np.ndarray:
        if magnitude.size == 0:
            return np.zeros_like(magnitude, dtype=np.uint8)
        mean_value = float(np.mean(magnitude))
        std_value = float(np.std(magnitude))
        percentile_70 = float(np.percentile(magnitude, 70))
        threshold = max(0.12, percentile_70, mean_value + std_value * 0.35)
        mask = (magnitude >= threshold).astype(np.uint8) * 255
        kernel3 = np.ones((3, 3), dtype=np.uint8)
        kernel5 = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5)

        min_pixels = max(12, int(mask.size * 0.012))
        if int(np.count_nonzero(mask)) < min_pixels:
            fallback_threshold = max(0.08, float(np.percentile(magnitude, 84)))
            mask = (magnitude >= fallback_threshold).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel3)

        if np.count_nonzero(mask):
            mask = cv2.dilate(mask, kernel3, iterations=1)
        return mask

    def _build_center_weight(self, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        center_x = max((width - 1) / 2.0, 1.0)
        center_y = max((height - 1) / 2.0, 1.0)
        norm_x = (xx - center_x) / max(width * 0.46, 1.0)
        norm_y = (yy - center_y) / max(height * 0.62, 1.0)
        radial = norm_x * norm_x + norm_y * norm_y
        return (0.35 + 0.65 * np.exp(-radial)).astype(np.float32)

    def _mask_to_bbox(self, mask: np.ndarray) -> list[int] | None:
        points = cv2.findNonZero(mask)
        if points is None:
            return None
        x, y, w, h = cv2.boundingRect(points)
        return [int(x), int(y), int(w), int(h)]

    def _stabilize_pose(
        self,
        raw_pose: PoseFrame | None,
        previous_pose: PoseFrame | None,
        previous_raw_pose: PoseFrame | None,
        earlier_pose: PoseFrame | None,
        previous_frame: np.ndarray | None,
        current_frame: np.ndarray,
        frame_index: int,
        last_pose_frame_index: int | None,
        frame_stride: int,
        lost_frame_count: int,
    ) -> PoseFrame | None:
        recovered = recover_pose_with_flow(
            raw_pose,
            previous_pose=previous_pose,
            previous_frame=previous_frame,
            current_frame=current_frame,
            lost_frame_count=lost_frame_count,
        )
        if recovered is None:
            if previous_pose is not None and last_pose_frame_index is not None:
                if frame_index - last_pose_frame_index <= max(2, frame_stride * 2):
                    fallback_pose = PoseFrame(
                        keypoints=[list(point) for point in previous_pose.keypoints],
                        bbox=list(previous_pose.bbox),
                        confidence=previous_pose.confidence,
                        keypoint_confidences=list(previous_pose.keypoint_confidences),
                        tracking_score=previous_pose.tracking_score,
                        tracking_mode="hold_last",
                        tracked_keypoints=previous_pose.tracked_keypoints,
                    )
                    return fallback_pose
            return None

        smoothed = temporal_smooth_pose(recovered, previous_pose, earlier_pose, alpha=0.64)
        constrained = apply_pose_constraints(smoothed, previous_pose or previous_raw_pose)
        if constrained is None:
            return None
        constrained.tracking_mode = recovered.tracking_mode
        constrained.tracked_keypoints = max(recovered.tracked_keypoints, constrained.tracked_keypoints)
        return constrained

    def _smooth_pose_sequence(self, pose_frames: list[PoseFrame]) -> list[PoseFrame]:
        """[S-G滤波平滑] 对完整 17 点轨迹执行线性插值和 Savitzky-Golay 平滑。"""
        if not pose_frames:
            return []
        frame_count = len(pose_frames)
        keypoint_count = len(COCO_KEYPOINTS)
        coords = np.full((frame_count, keypoint_count, 2), np.nan, dtype=np.float64)
        confidences = np.full((frame_count, keypoint_count), np.nan, dtype=np.float64)

        for frame_idx, pose in enumerate(pose_frames):
            for kp_idx in range(min(keypoint_count, len(pose.keypoints))):
                point = np.array(pose.keypoints[kp_idx][:2], dtype=np.float64)
                if np.isfinite(point).all():
                    coords[frame_idx, kp_idx] = point
                if kp_idx < len(pose.keypoint_confidences):
                    confidences[frame_idx, kp_idx] = float(pose.keypoint_confidences[kp_idx])

        coords = self._filter_skeletal_outliers(coords, confidences)
        for kp_idx in range(keypoint_count):
            for axis in range(2):
                coords[:, kp_idx, axis] = self._interpolate_series(coords[:, kp_idx, axis])
                coords[:, kp_idx, axis] = self._savgol_smooth_series(coords[:, kp_idx, axis])

        smoothed_frames: list[PoseFrame] = []
        for frame_idx, pose in enumerate(pose_frames):
            keypoints = coords[frame_idx].tolist()
            conf_row = confidences[frame_idx]
            confidence_values = [
                float(value) if np.isfinite(value) else 0.0
                for value in conf_row.tolist()
            ]
            finite_points = [point for point in keypoints if np.isfinite(np.array(point, dtype=np.float64)).all()]
            bbox = self._bbox_from_finite_points(finite_points, pose.bbox)
            smoothed_frames.append(
                PoseFrame(
                    keypoints=[[float(point[0]), float(point[1])] for point in keypoints],
                    bbox=bbox,
                    confidence=pose.confidence,
                    keypoint_confidences=confidence_values,
                    tracking_score=pose.tracking_score,
                    tracking_mode=pose.tracking_mode,
                    tracked_keypoints=sum(1 for value in confidence_values if value >= 0.5),
                    track_id=pose.track_id,
                )
            )
        return smoothed_frames

    def _filter_skeletal_outliers(self, coords: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """
        [异常点过滤与骨骼约束] 将明显违背人体骨段长度连续性的点置为 NaN。
        数学逻辑：慢动作中相邻帧骨段长度应平滑变化；若当前腕-肘/踝-膝等长度
        相比前后有效长度中位数暴增，通常是 YOLO 单帧跳点，后续用线性插值修补。
        """
        cleaned = coords.copy()
        cleaned[confidences < 0.5] = np.nan
        bone_pairs = [
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
        ]
        for first_name, second_name in bone_pairs:
            first_idx = COCO_KEYPOINTS.index(first_name)
            second_idx = COCO_KEYPOINTS.index(second_name)
            first_points = cleaned[:, first_idx, :]
            second_points = cleaned[:, second_idx, :]
            valid = np.isfinite(first_points).all(axis=1) & np.isfinite(second_points).all(axis=1)
            if np.sum(valid) < 5:
                continue
            lengths = np.linalg.norm(first_points - second_points, axis=1)
            reference = float(np.nanmedian(lengths[valid]))
            if reference <= 1.0:
                continue
            # 允许真实动作幅度，但超过骨段中位数 2.2 倍或短到 0.35 倍时判定为跳点。
            outlier = valid & ((lengths > reference * 2.2) | (lengths < reference * 0.35))
            cleaned[outlier, second_idx, :] = np.nan
        return cleaned

    def _interpolate_series(self, values: np.ndarray) -> np.ndarray:
        """[线性补帧] 优先使用 pandas.interpolate，对低置信关键点导致的时序空洞做补齐。"""
        series = values.astype(np.float64, copy=True)
        valid_mask = np.isfinite(series)
        if valid_mask.all():
            return series
        valid_indices = np.flatnonzero(valid_mask)
        if len(valid_indices) == 0:
            return np.zeros_like(series, dtype=np.float64)
        if len(valid_indices) == 1:
            return np.full_like(series, float(series[valid_indices[0]]), dtype=np.float64)
        if pd is not None:
            return pd.Series(series).interpolate(method="linear", limit_direction="both").to_numpy(dtype=np.float64)
        all_indices = np.arange(len(series), dtype=np.float64)
        return np.interp(all_indices, valid_indices.astype(np.float64), series[valid_indices])

    def _savgol_smooth_series(self, values: np.ndarray, window_length: int = 13, polyorder: int = 2) -> np.ndarray:
        """[S-G 滤波平滑] 优先使用 scipy.signal.savgol_filter，缺依赖时回退到本地实现。"""
        series = values.astype(np.float64, copy=True)
        if len(series) < 5:
            return series
        window = min(window_length, len(series) if len(series) % 2 == 1 else len(series) - 1)
        if window <= polyorder + 2:
            window = polyorder + 3
            if window % 2 == 0:
                window += 1
        if window > len(series):
            window = len(series) if len(series) % 2 == 1 else len(series) - 1
        if window < 5:
            return series
        if savgol_filter is not None:
            return savgol_filter(series, window_length=window, polyorder=min(polyorder, window - 2), mode="interp")
        half = window // 2
        x = np.arange(-half, half + 1, dtype=np.float64)
        vandermonde = np.vander(x, polyorder + 1, increasing=True)
        coeffs = np.linalg.pinv(vandermonde)[0]
        padded = np.pad(series, (half, half), mode="edge")
        return np.convolve(padded, coeffs[::-1], mode="valid")[: len(series)]

    def _extract_baseline_rows(self, baseline_payload: dict | None) -> list[dict]:
        if not isinstance(baseline_payload, dict):
            return []
        rows = baseline_payload.get("原始逐帧数据") or []
        return [row for row in rows if isinstance(row, dict)]

    def _estimate_processed_frame_count(
        self,
        total_frames: int,
        frame_stride: int,
        max_frames: int | None,
    ) -> int:
        estimated = max(1, math.ceil(max(total_frames, 1) / max(1, frame_stride)))
        if max_frames is not None:
            estimated = min(estimated, max(1, int(max_frames)))
        return estimated

    def _align_baseline_rows(self, baseline_rows: list[dict], target_length: int) -> list[dict]:
        if not baseline_rows:
            return []
        target_length = max(1, int(target_length))
        if len(baseline_rows) == target_length:
            return list(baseline_rows)
        if len(baseline_rows) == 1:
            return [baseline_rows[0]] * target_length
        sample_positions = np.linspace(0.0, len(baseline_rows) - 1, target_length)
        return [baseline_rows[min(len(baseline_rows) - 1, max(0, int(round(pos))))] for pos in sample_positions]

    def _baseline_row_at_index(self, aligned_rows: list[dict], current_index: int) -> dict | None:
        if not aligned_rows:
            return None
        return aligned_rows[min(len(aligned_rows) - 1, max(0, int(current_index)))]

    def _extract_keypoint_positions_from_row(self, row: dict | None) -> dict[str, list[float]]:
        if not isinstance(row, dict):
            return {}
        positions = row.get("keypoint_positions") or row.get("关键点坐标") or {}
        if not isinstance(positions, dict):
            return {}
        normalized: dict[str, list[float]] = {}
        for name, value in positions.items():
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                continue
            point = np.array(value[:2], dtype=np.float64)
            if not np.isfinite(point).all():
                continue
            normalized[str(name)] = [float(point[0]), float(point[1])]
        return normalized

    def _frame_template_distance(
        self,
        current_points: dict[str, list[float]],
        baseline_points: dict[str, list[float]],
    ) -> float | None:
        if not current_points or not baseline_points:
            return None
        distances: list[float] = []
        for name in COCO_KEYPOINTS:
            if name not in current_points or name not in baseline_points:
                continue
            current = np.array(current_points[name][:2], dtype=np.float64)
            baseline = np.array(baseline_points[name][:2], dtype=np.float64)
            if not np.isfinite(current).all() or not np.isfinite(baseline).all():
                continue
            distances.append(float(np.linalg.norm(current - baseline)))
        if not distances:
            return None
        return float(np.mean(distances))

    def _build_live_suggestions(self, metrics: FrameMetrics, template_distance: float | None) -> list[str]:
        suggestions: list[str] = []
        trunk_angle = float(metrics.torso_tilt_deg) if metrics.torso_tilt_deg is not None else 0.0
        arm_symmetry_error = float(metrics.arm_symmetry_error) if metrics.arm_symmetry_error is not None else 0.0
        stability_score = float(metrics.skeleton_stability) if metrics.skeleton_stability is not None else 1.0
        joint_jitter = float(metrics.joint_jitter) if metrics.joint_jitter is not None else 0.0
        template_distance_value = None
        if template_distance is not None:
            try:
                template_distance_value = float(template_distance)
            except (TypeError, ValueError):
                template_distance_value = None
        if template_distance_value is not None and not math.isfinite(template_distance_value):
            template_distance_value = None

        if metrics.torso_tilt_deg is not None and trunk_angle > 15.0:
            suggestions.append(
                "提示：姿态偏差。躯干明显倾斜。建议百会穴上领，收腹提肛，保持脊柱中正。"
            )
        if metrics.arm_symmetry_error is not None and arm_symmetry_error > 10.0:
            suggestions.append(
                "提示：动作非对称。双臂高度不一。建议平衡双肩发力，动作尽量匀称舒展。"
            )
        if template_distance_value is not None and template_distance_value > 0.5:
            suggestions.append(
                "提示：框架偏差。肢体伸展不足。建议拉伸四肢，定势时注意沉肩坠肘。"
            )
        if (
            metrics.skeleton_stability is not None
            and stability_score < 0.45
        ) or joint_jitter > 0.035:
            suggestions.append(
                "提示：控制力不足。检测到肢体细微抖动。建议减慢速度，加强核心支撑，体会气沉丹田。"
            )
        if (
            not suggestions
            and (metrics.pose_quality_score or 0.0) >= 80.0
            and int(metrics.visible_keypoint_count or 0) >= 8
            and metrics.frame_index % 45 == 0
        ):
            suggestions.append("提示：当前动作框架保持良好。请继续保持中正、松沉、匀缓的动作节奏。")
        return suggestions

    def _build_live_feedback_card(
        self,
        metrics: FrameMetrics,
        *,
        rules: RuleSet,
        live_issues: list[str],
        live_suggestions: list[str],
        phase_guidance: str,
        template_distance: float | None,
    ) -> dict[str, object]:
        phase_label = str(metrics.phase_label or "未识别阶段")
        category = str(rules.action_category or "通用动作")
        action_name = str(rules.template_name or category or "当前动作")
        focus_summary = self._live_focus_summary(metrics, rules, template_distance)
        manifestation = self._live_manifestation(metrics, live_issues, template_distance)
        cause = self._live_possible_cause(metrics, rules, template_distance)
        method = self._live_adjustment_method(metrics, rules, phase_guidance, template_distance)
        focus = self._live_training_focus(metrics, rules, template_distance)
        metrics_lines = self._live_metric_lines(metrics, template_distance)
        detail_items: list[str] = []
        detail_items.extend(live_suggestions[:3])
        if phase_guidance and phase_guidance not in detail_items:
            detail_items.append(phase_guidance)
        return {
            "action_name": action_name,
            "category": category,
            "phase_label": phase_label,
            "focus_summary": focus_summary,
            "manifestation": manifestation,
            "possible_cause": cause,
            "adjustment_method": method,
            "training_focus": focus,
            "detail_items": detail_items,
            "metric_lines": metrics_lines,
        }

    def _live_focus_summary(self, metrics: FrameMetrics, rules: RuleSet, template_distance: float | None) -> str:
        phase_label = str(metrics.phase_label or "当前阶段")
        category = str(rules.action_category or "通用动作")
        if category == "健身气功":
            return f"{phase_label}重点查看身形中正、双臂匀称舒展与定势稳定。"
        if category == "太极":
            return f"{phase_label}重点查看立身中正、虚实转换和上下相随。"
        if category == "武术基本功":
            return f"{phase_label}重点查看步型规格、出手路线和发力定型。"
        if category == "民族舞":
            return f"{phase_label}重点查看节拍、摆臂路线和轴心稳定。"
        if template_distance is not None and float(template_distance) > 0.5:
            return f"{phase_label}与模板框架存在偏差，建议优先校正动作路线。"
        return f"{phase_label}请继续观察躯干、双臂、重心与动作节奏的一致性。"

    def _live_manifestation(self, metrics: FrameMetrics, live_issues: list[str], template_distance: float | None) -> str:
        details: list[str] = []
        if metrics.torso_tilt_deg is not None and float(metrics.torso_tilt_deg) > 15.0:
            details.append("躯干有明显倾斜，身形中线不够稳定")
        if metrics.arm_symmetry_error is not None and float(metrics.arm_symmetry_error) > 10.0:
            details.append("双臂高度与到位时机不一致")
        if template_distance is not None:
            try:
                distance_value = float(template_distance)
            except (TypeError, ValueError):
                distance_value = 0.0
            if distance_value > 0.5:
                details.append("当前动作框架与标准模板存在较明显偏差")
        if metrics.joint_jitter is not None and float(metrics.joint_jitter) > 0.035:
            details.append("肢体末端存在细微抖动，定势控制不足")
        if not details and live_issues:
            details.append("、".join(str(item) for item in live_issues[:2]))
        if not details:
            return "当前动作整体完成较稳定，未发现突出的即时偏差。"
        return "；".join(details) + "。"

    def _live_possible_cause(self, metrics: FrameMetrics, rules: RuleSet, template_distance: float | None) -> str:
        if metrics.torso_tilt_deg is not None and float(metrics.torso_tilt_deg) > 15.0:
            return "多与核心支撑不足、肩髋中线没有锁住，或转接时上身先行有关。"
        if metrics.arm_symmetry_error is not None and float(metrics.arm_symmetry_error) > 10.0:
            return "多与左右发力时机不一致、一侧抢动作，或肩肘联动路线不同步有关。"
        if template_distance is not None:
            try:
                distance_value = float(template_distance)
            except (TypeError, ValueError):
                distance_value = 0.0
            if distance_value > 0.5:
                return "多与动作路线不清、伸展不足，或模板关键姿态没有完全到位有关。"
        if metrics.joint_jitter is not None and float(metrics.joint_jitter) > 0.035:
            return "多与速度偏快、定势停留不足，或呼吸与动作节拍没有稳定配合有关。"
        if rules.action_category == "健身气功":
            return "建议继续关注呼吸、节奏和身形中正的统一。"
        return "当前未出现集中性错误，可继续细查动作路线、节奏与身体控制。"

    def _live_adjustment_method(
        self,
        metrics: FrameMetrics,
        rules: RuleSet,
        phase_guidance: str,
        template_distance: float | None,
    ) -> str:
        methods: list[str] = []
        if metrics.torso_tilt_deg is not None and float(metrics.torso_tilt_deg) > 15.0:
            methods.append("先做百会穴上领、收腹提肛，保持脊柱拔伸后再进入下一段动作")
        if metrics.arm_symmetry_error is not None and float(metrics.arm_symmetry_error) > 10.0:
            methods.append("用镜像方式检查双臂是否同时启动、同时到位，强调双肩平衡发力")
        if template_distance is not None:
            try:
                distance_value = float(template_distance)
            except (TypeError, ValueError):
                distance_value = 0.0
            if distance_value > 0.5:
                methods.append("先分解手臂与躯干路线，再按模板关键姿态逐段校正动作框架")
        if metrics.joint_jitter is not None and float(metrics.joint_jitter) > 0.035:
            methods.append("把速度降下来，在定势点多停半拍，体会气沉丹田后的稳定支撑")
        if phase_guidance:
            methods.append(phase_guidance)
        if not methods:
            methods.append("当前阶段以保持中正、匀缓、稳定为主，再逐步放大动作幅度。")
        return "；".join(methods) + "。"

    def _live_training_focus(self, metrics: FrameMetrics, rules: RuleSet, template_distance: float | None) -> str:
        if metrics.torso_tilt_deg is not None and float(metrics.torso_tilt_deg) > 15.0:
            return "训练重点：核心稳定、肩髋中线控制和转接时的躯干保持。"
        if metrics.arm_symmetry_error is not None and float(metrics.arm_symmetry_error) > 10.0:
            return "训练重点：左右同步性、双侧幅度一致性和肩肘联动顺序。"
        if template_distance is not None:
            try:
                distance_value = float(template_distance)
            except (TypeError, ValueError):
                distance_value = 0.0
            if distance_value > 0.5:
                return "训练重点：模板关键姿态复现、动作路线清晰度和到位质量。"
        if metrics.joint_jitter is not None and float(metrics.joint_jitter) > 0.035:
            return "训练重点：定势停留、末端控制和慢节奏下的整体稳定。"
        if rules.action_category == "健身气功":
            return "训练重点：沉肩坠肘、立身中正、呼吸配合与动作连贯。"
        return "训练重点：动作路线、节奏均匀性和关键姿态稳定复现。"

    def _live_metric_lines(self, metrics: FrameMetrics, template_distance: float | None) -> list[str]:
        lines: list[str] = []
        if metrics.torso_tilt_deg is not None:
            lines.append(f"躯干倾斜：{float(metrics.torso_tilt_deg):.1f}°")
        if metrics.arm_symmetry_error is not None:
            lines.append(f"双臂对称差：{float(metrics.arm_symmetry_error):.1f}px")
        if metrics.skeleton_stability is not None:
            lines.append(f"稳定度：{float(metrics.skeleton_stability):.2f}")
        if metrics.transition_score is not None:
            lines.append(f"衔接评分：{float(metrics.transition_score):.1f}")
        if template_distance is not None:
            try:
                lines.append(f"模板距离：{float(template_distance):.3f}")
            except (TypeError, ValueError):
                pass
        return lines

    def _perfect_match_override_reason(
        self,
        metrics: list[FrameMetrics],
        *,
        source: str | int,
        baseline_payload: dict | None,
        analysis_mode: str,
    ) -> str | None:
        if analysis_mode != "test":
            return None

        same_source = False
        if isinstance(source, str):
            baseline_source = self._resolve_baseline_source(baseline_payload)
            if baseline_source:
                same_source = self._same_source_path(source, baseline_source)

        template_distances = [
            float(item.template_distance)
            for item in metrics
            if item.template_distance is not None and math.isfinite(float(item.template_distance))
        ]
        avg_template_distance = mean(template_distances) if template_distances else None
        if same_source or (avg_template_distance is not None and avg_template_distance < 1e-3):
            return "检测到基准视频自校准，动作完美匹配。"
        if same_source or (avg_template_distance is not None and avg_template_distance < 1e-2):
            return "检测到当前为极高匹配度或同源基准比对，动作完美符合。"
        return None

    def _resolve_baseline_source(self, baseline_payload: dict | None) -> str | None:
        if not isinstance(baseline_payload, dict):
            return None
        for key in ("来源视频", "输入源", "source", "source_path", "video_path"):
            value = baseline_payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _same_source_path(self, current_source: str, baseline_source: str) -> bool:
        try:
            current_path = Path(str(current_source)).expanduser().resolve(strict=False)
        except Exception:
            current_path = Path(str(current_source)).expanduser()
        try:
            baseline_path = Path(str(baseline_source)).expanduser().resolve(strict=False)
        except Exception:
            baseline_path = Path(str(baseline_source)).expanduser()
        return str(current_path).lower() == str(baseline_path).lower()

    def _refresh_metrics_from_smoothed_poses(
        self,
        frame_metrics: list[FrameMetrics],
        pose_frames: list[PoseFrame],
        fps: float,
    ) -> None:
        """用全序列平滑后的骨架刷新关键点、角度和速度指标。"""
        previous_pose: PoseFrame | None = None
        for metrics, pose in zip(frame_metrics, pose_frames):
            features = self._extract_pose_features(pose, previous_pose, fps)
            metrics.trunk_center = features["trunk_center"]
            metrics.torso_tilt_deg = features["torso_tilt_deg"]
            metrics.shoulder_balance = features["shoulder_balance"]
            metrics.arm_raise_left_deg = features["arm_raise_left_deg"]
            metrics.arm_raise_right_deg = features["arm_raise_right_deg"]
            metrics.arm_symmetry_error = features["arm_symmetry_error"]
            metrics.left_shoulder_angle = features["left_shoulder_angle"]
            metrics.right_shoulder_angle = features["right_shoulder_angle"]
            metrics.left_elbow_angle = features["left_elbow_angle"]
            metrics.right_elbow_angle = features["right_elbow_angle"]
            metrics.left_hip_angle = features["left_hip_angle"]
            metrics.right_hip_angle = features["right_hip_angle"]
            metrics.left_knee_angle = features["left_knee_angle"]
            metrics.right_knee_angle = features["right_knee_angle"]
            metrics.wrist_speed_left = features["wrist_speed_left"]
            metrics.wrist_speed_right = features["wrist_speed_right"]
            metrics.pose_motion = pose_mean_displacement(pose, previous_pose)
            metrics.keypoint_positions = self._collect_keypoint_positions(pose, threshold=0.5)
            previous_pose = pose

    def _bbox_from_finite_points(self, points: list[list[float]], fallback_bbox: list[float] | None = None) -> list[float]:
        if not points:
            return list(fallback_bbox or [0.0, 0.0, 1.0, 1.0])
        arr = np.array(points, dtype=np.float64)
        return [
            float(np.min(arr[:, 0])),
            float(np.min(arr[:, 1])),
            float(np.max(arr[:, 0])),
            float(np.max(arr[:, 1])),
        ]

    def _extract_pose_features(
        self,
        pose: PoseFrame | None,
        previous_pose: PoseFrame | None,
        fps: float,
    ) -> dict:
        default = {
            "trunk_center": None,
            "torso_tilt_deg": None,
            "shoulder_balance": None,
            "arm_raise_left_deg": None,
            "arm_raise_right_deg": None,
            "arm_symmetry_error": None,
            "left_shoulder_angle": None,
            "right_shoulder_angle": None,
            "left_elbow_angle": None,
            "right_elbow_angle": None,
            "left_hip_angle": None,
            "right_hip_angle": None,
            "left_knee_angle": None,
            "right_knee_angle": None,
            "wrist_speed_left": None,
            "wrist_speed_right": None,
        }
        if pose is None:
            return default

        if len(pose.keypoints) < len(COCO_KEYPOINTS):
            return default
        points = {name: np.array(pose.keypoints[idx], dtype=np.float32) for idx, name in enumerate(COCO_KEYPOINTS)}
        required_points = [
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
        ]
        if any(not np.isfinite(points[name]).all() for name in required_points):
            return default

        left_shoulder = points["left_shoulder"]
        right_shoulder = points["right_shoulder"]
        left_elbow = points["left_elbow"]
        right_elbow = points["right_elbow"]
        left_wrist = points["left_wrist"]
        right_wrist = points["right_wrist"]
        left_hip = points["left_hip"]
        right_hip = points["right_hip"]

        shoulder_mid = (left_shoulder + right_shoulder) / 2.0
        hip_mid = (left_hip + right_hip) / 2.0
        trunk_center = ((shoulder_mid + hip_mid) / 2.0).tolist()
        torso_vector = shoulder_mid - hip_mid
        torso_tilt_deg = abs(float(math.degrees(math.atan2(torso_vector[0], -torso_vector[1] + 1e-6))))

        shoulder_balance = abs(float(left_shoulder[1] - right_shoulder[1]))
        arm_raise_left_deg = self._calculate_arm_raise(left_shoulder, left_elbow, left_wrist, left_hip, right_hip)
        arm_raise_right_deg = self._calculate_arm_raise(
            right_shoulder,
            right_elbow,
            right_wrist,
            left_hip,
            right_hip,
        )
        arm_symmetry_error = abs(arm_raise_left_deg - arm_raise_right_deg)

        wrist_speed_left = None
        wrist_speed_right = None
        if previous_pose is not None and fps > 0:
            prev_points = {
                name: np.array(previous_pose.keypoints[idx], dtype=np.float32)
                for idx, name in enumerate(COCO_KEYPOINTS)
            }
            if np.isfinite(prev_points["left_wrist"]).all():
                wrist_speed_left = float(np.linalg.norm(left_wrist - prev_points["left_wrist"]) * fps)
            if np.isfinite(prev_points["right_wrist"]).all():
                wrist_speed_right = float(np.linalg.norm(right_wrist - prev_points["right_wrist"]) * fps)

        joint_angles = {}
        for angle_name, (kp1, kp2, kp3) in ANGLE_CONFIGS.items():
            joint_angles[angle_name] = self._calculate_joint_angle(points, kp1, kp2, kp3)

        return {
            "trunk_center": [float(trunk_center[0]), float(trunk_center[1])],
            "torso_tilt_deg": torso_tilt_deg,
            "shoulder_balance": shoulder_balance,
            "arm_raise_left_deg": arm_raise_left_deg,
            "arm_raise_right_deg": arm_raise_right_deg,
            "arm_symmetry_error": arm_symmetry_error,
            "left_shoulder_angle": joint_angles["left_shoulder_angle"],
            "right_shoulder_angle": joint_angles["right_shoulder_angle"],
            "left_elbow_angle": joint_angles["left_elbow_angle"],
            "right_elbow_angle": joint_angles["right_elbow_angle"],
            "left_hip_angle": joint_angles["left_hip_angle"],
            "right_hip_angle": joint_angles["right_hip_angle"],
            "left_knee_angle": joint_angles["left_knee_angle"],
            "right_knee_angle": joint_angles["right_knee_angle"],
            "wrist_speed_left": wrist_speed_left,
            "wrist_speed_right": wrist_speed_right,
        }

    def _calculate_arm_raise(
        self,
        shoulder: np.ndarray,
        elbow: np.ndarray,
        wrist: np.ndarray,
        left_hip: np.ndarray,
        right_hip: np.ndarray,
    ) -> float:
        shoulder_to_wrist = wrist - shoulder
        torso_axis = ((left_hip + right_hip) / 2.0) - shoulder
        torso_norm = float(np.linalg.norm(torso_axis))
        if torso_norm <= 1e-6:
            torso_axis = np.array([0.0, 1.0], dtype=np.float32)
        else:
            torso_axis = torso_axis / torso_norm

        wrist_norm = float(np.linalg.norm(shoulder_to_wrist))
        if wrist_norm <= 1e-6:
            elbow_vector = elbow - shoulder
            wrist_norm = float(np.linalg.norm(elbow_vector))
            if wrist_norm <= 1e-6:
                return 0.0
            shoulder_to_wrist = elbow_vector

        arm_axis = shoulder_to_wrist / max(float(np.linalg.norm(shoulder_to_wrist)), 1e-6)
        cosine = float(np.dot(arm_axis, torso_axis))
        cosine = max(-1.0, min(1.0, cosine))
        torso_relative = float(np.degrees(np.arccos(cosine)))
        # 与身体主轴反向越明显，表示上举越充分：垂落约 0 度，上举接近 180 度。
        return float(max(0.0, min(180.0, 180.0 - torso_relative)))

    def _calculate_joint_angle(
        self,
        points: dict[str, np.ndarray],
        point1: str,
        pivot: str,
        point3: str,
    ) -> float | None:
        if point1 not in points or pivot not in points or point3 not in points:
            return None
        p1 = points[point1]
        p2 = points[pivot]
        p3 = points[point3]
        if not np.isfinite(p1).all() or not np.isfinite(p2).all() or not np.isfinite(p3).all():
            return None
        v1 = p1 - p2
        v2 = p3 - p2
        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))
        if norm1 <= 1e-6 or norm2 <= 1e-6:
            return None
        cos_angle = float(np.dot(v1, v2) / max(norm1 * norm2, 1e-6))
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return float(np.degrees(np.arccos(cos_angle)))

    def _estimate_skeleton_stability(self, joint_jitter: float | None, pose: PoseFrame | None) -> float | None:
        if pose is None:
            return None
        confidence = pose.confidence
        coverage = self._estimate_keypoint_coverage(pose) or 0.0
        jitter_penalty = min(1.0, (joint_jitter or 0.0) * 8.0)
        tracked_ratio = float(pose.tracked_keypoints / max(len(pose.keypoints), 1))
        tracking_bonus = 0.06 if pose.tracking_mode in {"detector+flow", "flow_fallback"} else 0.02
        stability = confidence * 0.56 + coverage * 0.28 + tracked_ratio * 0.16 + tracking_bonus
        return float(max(0.0, min(1.0, stability * (1.0 - jitter_penalty))))

    def _estimate_keypoint_coverage(self, pose: PoseFrame | None, threshold: float = 0.25) -> float | None:
        if pose is None:
            return None
        confidences = pose.keypoint_confidences or []
        if not confidences:
            return 1.0 if pose.keypoints else 0.0
        visible_count = sum(1 for value in confidences if float(value) >= threshold)
        return float(visible_count / max(len(confidences), 1))

    def _count_visible_keypoints(self, pose: PoseFrame | None, threshold: float = 0.25) -> int | None:
        if pose is None:
            return None
        confidences = pose.keypoint_confidences or []
        if not confidences:
            return len(pose.keypoints)
        return int(sum(1 for value in confidences if float(value) >= threshold))

    def _estimate_pose_quality_score(
        self,
        pose: PoseFrame | None,
        skeleton_stability: float | None,
        keypoint_coverage: float | None,
    ) -> float | None:
        if pose is None:
            return None
        confidence = pose.confidence or 0.0
        stability = skeleton_stability or 0.0
        coverage = keypoint_coverage or 0.0
        tracked_ratio = float(pose.tracked_keypoints / max(len(pose.keypoints), 1))
        tracking_bonus = 0.07 if pose.tracking_mode in {"detector", "detector+flow"} else 0.03
        return self._clamp_score(
            (confidence * 0.35 + stability * 0.32 + coverage * 0.21 + tracked_ratio * 0.12 + tracking_bonus) * 100.0
        )

    def _classify_pose_quality(self, score: float | None) -> str | None:
        if score is None:
            return None
        if score >= 85:
            return "优"
        if score >= 70:
            return "良"
        if score >= 55:
            return "中"
        return "弱"

    def _collect_keypoint_positions(self, pose: PoseFrame | None, threshold: float = 0.18) -> dict[str, list[float]]:
        if pose is None:
            return {}
        positions: dict[str, list[float]] = {}
        confidences = pose.keypoint_confidences or []
        for index, name in enumerate(COCO_KEYPOINTS):
            if index >= len(pose.keypoints):
                continue
            if confidences and index < len(confidences) and float(confidences[index]) < threshold:
                continue
            point = pose.keypoints[index]
            point_arr = np.array(point[:2], dtype=np.float32)
            if not np.isfinite(point_arr).all():
                continue
            positions[name] = [float(point[0]), float(point[1])]
        return positions

    def _infer_live_phase(
        self,
        motion_mean: float,
        pose_motion: float | None,
        motion_history: deque[float],
    ) -> str:
        recent_motion = mean(motion_history) if motion_history else motion_mean
        pose_signal = pose_motion or 0.0
        if recent_motion < 0.45 and pose_signal < 4.0:
            return "定势候选"
        if recent_motion > 1.8 or pose_signal > 18.0:
            return "发力候选"
        return "过渡候选"

    def _estimate_transition_score(
        self,
        motion_history: deque[float],
        pose_motion_history: deque[float],
        jitter_history: deque[float],
        balance_history: deque[float],
        pose: PoseFrame | None,
    ) -> float | None:
        if len(motion_history) < 4:
            return None
        motion_values = np.array(list(motion_history), dtype=np.float32)
        pose_values = np.array(list(pose_motion_history), dtype=np.float32) if pose_motion_history else np.zeros(1, dtype=np.float32)
        balance_values = np.array(list(balance_history), dtype=np.float32) if balance_history else np.zeros(1, dtype=np.float32)
        jitter_values = np.array(list(jitter_history), dtype=np.float32) if jitter_history else np.zeros(1, dtype=np.float32)

        motion_smoothness = 1.0 - min(1.0, float(np.std(np.diff(motion_values))) / 1.35) if len(motion_values) >= 3 else 0.6
        pose_smoothness = 1.0 - min(1.0, float(np.std(np.diff(pose_values))) / 18.0) if len(pose_values) >= 3 else 0.6
        balance_stability = 1.0 - min(1.0, float(np.mean(balance_values)) / 0.28)
        jitter_stability = 1.0 - min(1.0, float(np.mean(jitter_values)) / 0.075)
        tracking_bonus = 0.08 if pose is not None and pose.tracking_mode in {"detector+flow", "flow_fallback", "flow_only"} else 0.0
        score = (
            motion_smoothness * 0.34
            + pose_smoothness * 0.28
            + balance_stability * 0.18
            + jitter_stability * 0.20
            + tracking_bonus
        ) * 100.0
        return self._clamp_score(score)

    def _classify_sequence_state(
        self,
        transition_score: float | None,
        motion_history: deque[float],
        pose_motion_history: deque[float],
    ) -> str | None:
        if transition_score is None:
            return None
        recent_motion = mean(motion_history) if motion_history else 0.0
        recent_pose_motion = mean(pose_motion_history) if pose_motion_history else 0.0
        if recent_motion < 0.45 and recent_pose_motion < 4.0 and transition_score >= 70:
            return "稳定定势"
        if transition_score >= 78:
            return "衔接平顺"
        if transition_score >= 58:
            return "衔接一般"
        return "衔接突兀"

    def _draw_overlay(
        self,
        frame: np.ndarray,
        flow_data: dict,
        pose: PoseFrame | None,
        metrics: FrameMetrics,
        rules: RuleSet,
        motion_history: deque[float],
        jitter_history: deque[float],
    ) -> np.ndarray:
        flow = flow_data["flow"]
        canvas = draw_pose(frame, pose)
        self._draw_flow_arrows(canvas, flow, flow_data.get("active_mask"))
        self._draw_motion_focus_box(canvas, flow.shape[:2], metrics.motion_focus_bbox)

        info_lines = [
            f"节奏 {self._tempo_label(rules.expected_tempo)}",
            f"阶段 {metrics.phase_label or '分析中'}",
            f"运动强度 {metrics.motion_mean:.3f}",
            f"左右平衡差 {metrics.left_right_balance:.3f}",
            f"活动区域占比 {metrics.active_region_ratio:.1%}",
        ]
        if metrics.pose_motion is not None:
            info_lines.append(f"骨架位移 {metrics.pose_motion:.1f}")
        if metrics.joint_jitter is not None:
            info_lines.append(f"骨架抖动 {metrics.joint_jitter:.3f}")
        if metrics.keypoint_coverage is not None:
            info_lines.append(f"关键点覆盖 {metrics.keypoint_coverage:.0%}")
        if metrics.pose_quality_score is not None:
            info_lines.append(f"姿态质量 {metrics.pose_quality_score:.0f}")
        if metrics.transition_score is not None:
            info_lines.append(f"衔接评分 {metrics.transition_score:.0f}")
        if metrics.sequence_state:
            info_lines.append(f"时序状态 {metrics.sequence_state}")
        if metrics.torso_tilt_deg is not None:
            info_lines.append(f"躯干倾斜 {metrics.torso_tilt_deg:.1f} 度")
        if metrics.arm_symmetry_error is not None:
            info_lines.append(f"双臂对称误差 {metrics.arm_symmetry_error:.1f}")
        if metrics.tracking_mode:
            info_lines.append(f"跟踪方式 {_TRACKING_MODE_LABELS.get(metrics.tracking_mode, metrics.tracking_mode)}")

        canvas = self._draw_multiline_text(
            canvas,
            info_lines,
            origin=(18, 18),
            color=(255, 255, 255),
            background=(15, 26, 37, 165),
            line_spacing=8,
        )

        if motion_history:
            sparkline = self._sparkline(list(motion_history), width=220, height=66, line_color=(0, 220, 255))
            self._place_corner_card(canvas, sparkline, top=15, right=15)

        if jitter_history:
            jitter_card = self._sparkline(list(jitter_history), width=220, height=66, line_color=(255, 206, 84))
            self._place_corner_card(canvas, jitter_card, top=90, right=15)

        live_hints = self._detect_live_issue_hints(metrics, rules)
        if live_hints:
            canvas = self._draw_issue_hints(canvas, live_hints[:3], pose, metrics)

        if not self.pose_estimator.available:
            canvas = self._draw_multiline_text(
                canvas,
                ["未检测到姿态权重，当前使用运动趋势兜底分析"],
                origin=(18, canvas.shape[0] - 42),
                color=(80, 220, 255),
                background=(10, 25, 35, 170),
                line_spacing=6,
            )
        return canvas

    def _detect_live_issue_hints(self, metrics: FrameMetrics, rules: RuleSet) -> list[str]:
        hints: list[str] = []
        threshold_speed = rules.thresholds.get("motion_mean_max_slow", 2.5)
        threshold_balance = rules.thresholds.get("left_right_balance_max", 0.18)
        threshold_torso = rules.thresholds.get("torso_tilt_max_deg", 15.0)
        threshold_symmetry = rules.thresholds.get("arm_symmetry_error_max", 18.0)
        arm_raise_floor = float(rules.thresholds.get("arm_raise_min_deg", 135.0 if rules.expected_tempo == "slow" else 120.0))
        pose_ready = (metrics.visible_keypoint_count or 0) >= 8 and (metrics.keypoint_coverage or 0.0) >= 0.45

        arm_values = [value for value in [metrics.arm_raise_left_deg, metrics.arm_raise_right_deg] if value is not None]
        if pose_ready and arm_values and mean(arm_values) < arm_raise_floor:
            hints.append("手臂抬高不足")
        elif pose_ready and rules.requires_symmetry and (metrics.arm_symmetry_error or 0.0) > threshold_symmetry:
            hints.append("手臂抬高不足")

        if rules.requires_symmetry and metrics.left_right_balance > threshold_balance:
            hints.append("重心偏移")

        if pose_ready and rules.requires_upright_torso and (metrics.torso_tilt_deg or 0.0) > threshold_torso:
            hints.append("转体角度不足")

        if rules.expected_tempo == "slow" and metrics.motion_mean > threshold_speed:
            hints.append("收势过快")

        return list(dict.fromkeys(hints))

    def _draw_issue_hints(
        self,
        frame: np.ndarray,
        hint_labels: list[str],
        pose: PoseFrame | None,
        metrics: FrameMetrics,
    ) -> np.ndarray:
        canvas = frame
        for index, label in enumerate(hint_labels[:3]):
            color = self._issue_color(label)
            bbox = self._resolve_issue_bbox(label, pose, metrics, canvas.shape)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
                text_x = max(16, min(x1 + 4, canvas.shape[1] - 180))
                text_y = max(16, y1 - 38 - index * 4)
            else:
                text_x = 18
                text_y = 152 + index * 56
            canvas = self._draw_multiline_text(
                canvas,
                [label],
                origin=(text_x, text_y),
                color=(255, 255, 255),
                background=self._issue_background(label),
                line_spacing=4,
            )
        return canvas

    def _resolve_issue_bbox(
        self,
        label: str,
        pose: PoseFrame | None,
        metrics: FrameMetrics,
        frame_shape: tuple[int, int, int],
    ) -> tuple[int, int, int, int] | None:
        positions = metrics.keypoint_positions or {}
        if any(keyword in label for keyword in ["手臂", "抬高"]):
            bbox = self._bbox_from_named_positions(
                positions,
                ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"],
                frame_shape,
                padding=24,
            )
            if bbox is not None:
                return bbox
        if any(keyword in label for keyword in ["重心", "转体", "躯干"]):
            bbox = self._bbox_from_named_positions(
                positions,
                ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
                frame_shape,
                padding=28,
            )
            if bbox is not None:
                return bbox
        if pose is not None and pose.bbox:
            return self._normalize_bbox(pose.bbox, frame_shape)
        return None

    def _bbox_from_named_positions(
        self,
        positions: dict[str, list[float]],
        names: list[str],
        frame_shape: tuple[int, int, int],
        padding: int = 20,
    ) -> tuple[int, int, int, int] | None:
        selected = [positions.get(name) for name in names]
        points = [
            point
            for point in selected
            if point is not None and len(point) >= 2 and np.isfinite(np.array(point[:2], dtype=np.float32)).all()
        ]
        if not points:
            return None
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        bbox = [min(xs) - padding, min(ys) - padding, max(xs) + padding, max(ys) + padding]
        return self._normalize_bbox(bbox, frame_shape)

    def _normalize_bbox(
        self,
        bbox: list[float],
        frame_shape: tuple[int, int, int],
    ) -> tuple[int, int, int, int] | None:
        frame_h, frame_w = frame_shape[:2]
        if len(bbox) < 4:
            return None
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(x1 + 1, min(frame_w - 1, x2))
        y2 = max(y1 + 1, min(frame_h - 1, y2))
        return (x1, y1, x2, y2)

    def _issue_color(self, label: str) -> tuple[int, int, int]:
        if "手臂" in label:
            return (44, 132, 255)
        if "重心" in label:
            return (43, 182, 255)
        if "转体" in label or "躯干" in label:
            return (92, 88, 255)
        if "收势" in label or "速度" in label:
            return (78, 210, 255)
        return (86, 214, 112)

    def _issue_background(self, label: str) -> tuple[int, int, int, int]:
        if "手臂" in label:
            return (40, 98, 176, 208)
        if "重心" in label:
            return (24, 108, 168, 208)
        if "转体" in label or "躯干" in label:
            return (76, 65, 176, 208)
        if "收势" in label or "速度" in label:
            return (127, 88, 21, 208)
        return (28, 92, 54, 208)

    def _load_overlay_font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        font_candidates = [
            str(DEFAULT_CHINESE_FONT),
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
        for candidate in font_candidates:
            path = Path(candidate)
            if not path.exists():
                continue
            try:
                return ImageFont.truetype(str(path), 22)
            except Exception:
                continue
        return ImageFont.load_default()

    def _draw_multiline_text(
        self,
        frame: np.ndarray,
        lines: list[str],
        origin: tuple[int, int],
        color: tuple[int, int, int],
        background: tuple[int, int, int, int] | None = None,
        line_spacing: int = 6,
    ) -> np.ndarray:
        if not lines:
            return frame
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        draw = ImageDraw.Draw(pil_image)
        x, y = origin
        line_boxes = [draw.textbbox((0, 0), line, font=self._overlay_font) for line in lines]
        widths = [box[2] - box[0] for box in line_boxes]
        heights = [box[3] - box[1] for box in line_boxes]
        total_height = sum(heights) + max(0, len(lines) - 1) * line_spacing
        max_width = max(widths) if widths else 0

        if background is not None:
            pad_x = 12
            pad_y = 10
            overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rounded_rectangle(
                (
                    x - pad_x,
                    y - pad_y,
                    x + max_width + pad_x,
                    y + total_height + pad_y,
                ),
                radius=12,
                fill=background,
            )
            pil_image = Image.alpha_composite(pil_image, overlay)
            draw = ImageDraw.Draw(pil_image)

        current_y = y
        fill = (color[2], color[1], color[0], 255)
        for line, height in zip(lines, heights):
            draw.text((x, current_y), line, font=self._overlay_font, fill=fill)
            current_y += height + line_spacing

        rgb = pil_image.convert("RGB")
        return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

    def _tempo_label(self, value: str) -> str:
        return {"slow": "缓慢", "medium": "平稳", "fast": "较快"}.get(value, value)

    def _draw_flow_arrows(
        self,
        frame: np.ndarray,
        flow: np.ndarray,
        active_mask: np.ndarray | None = None,
        step: int = 24,
    ) -> None:
        if flow.size == 0:
            return
        h_small, w_small = flow.shape[:2]
        scale_x = frame.shape[1] / max(w_small, 1)
        scale_y = frame.shape[0] / max(h_small, 1)

        for y in range(step // 2, h_small, step):
            for x in range(step // 2, w_small, step):
                if active_mask is not None and active_mask.size and active_mask[y, x] == 0:
                    continue
                dx, dy = flow[y, x]
                start = (int(x * scale_x), int(y * scale_y))
                end = (int((x + dx * 2.0) * scale_x), int((y + dy * 2.0) * scale_y))
                cv2.arrowedLine(frame, start, end, (255, 196, 0), 1, cv2.LINE_AA, tipLength=0.3)

    def _draw_motion_focus_box(
        self,
        frame: np.ndarray,
        flow_shape: tuple[int, int],
        bbox: list[int] | None,
    ) -> None:
        if bbox is None:
            return
        flow_h, flow_w = flow_shape
        x, y, w, h = bbox
        scale_x = frame.shape[1] / max(flow_w, 1)
        scale_y = frame.shape[0] / max(flow_h, 1)
        start = (int(x * scale_x), int(y * scale_y))
        end = (int((x + w) * scale_x), int((y + h) * scale_y))
        cv2.rectangle(frame, start, end, (86, 214, 112), 2, cv2.LINE_AA)
        # 视频叠字统一走 Pillow，避免 OpenCV 字体接口在中文场景下出现乱码/方框。
        text_frame = self._draw_multiline_text(
            frame,
            ["活动区域"],
            origin=(start[0], max(24, start[1] - 30)),
            color=(86, 214, 112),
            background=(255, 255, 255, 180),
            line_spacing=4,
        )
        frame[:] = text_frame

    def _sparkline(
        self,
        values: list[float],
        width: int,
        height: int,
        line_color: tuple[int, int, int],
    ) -> np.ndarray:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        if len(values) < 2:
            return canvas
        values_arr = np.array(values, dtype=np.float32)
        min_value = float(np.min(values_arr))
        max_value = float(np.max(values_arr))
        span = max(max_value - min_value, 1e-6)
        points = []
        for index, value in enumerate(values_arr):
            x = int(index * (width - 1) / (len(values_arr) - 1))
            y = int(height - 1 - ((value - min_value) / span) * (height - 12))
            points.append((x, y))
        cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, line_color, 2)
        cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (86, 86, 86), 1)
        return canvas

    def _place_corner_card(self, canvas: np.ndarray, card: np.ndarray, top: int, right: int) -> None:
        h, w = card.shape[:2]
        x1 = max(0, canvas.shape[1] - w - right)
        y1 = max(0, top)
        canvas[y1 : y1 + h, x1 : x1 + w] = card

    def _build_writer(self, output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter | None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, max(fps, 1.0), (width, height))
        if writer.isOpened():
            return writer
        return None

    def _evaluate_issues(
        self,
        metrics: list[FrameMetrics],
        rules: RuleSet,
        fps: float,
        phases: list[dict],
    ) -> list[Issue]:
        issues: list[Issue] = []
        avg_motion = mean(item.motion_mean for item in metrics)
        motion_std = mean(item.motion_std for item in metrics)
        avg_balance = mean(item.left_right_balance for item in metrics)

        torso_values = [item.torso_tilt_deg for item in metrics if item.torso_tilt_deg is not None]
        symmetry_values = [item.arm_symmetry_error for item in metrics if item.arm_symmetry_error is not None]
        jitter_values = [item.joint_jitter for item in metrics if item.joint_jitter is not None]
        stability_values = [item.skeleton_stability for item in metrics if item.skeleton_stability is not None]
        transition_values = [item.transition_score for item in metrics if item.transition_score is not None]
        pose_conf_values = [item.pose_confidence for item in metrics if item.pose_confidence is not None]
        coverage_values = [item.keypoint_coverage for item in metrics if item.keypoint_coverage is not None]

        threshold_torso = rules.thresholds.get("torso_tilt_max_deg", 15.0)
        threshold_symmetry = rules.thresholds.get("arm_symmetry_error_max", 18.0)
        threshold_balance = rules.thresholds.get("left_right_balance_max", 0.18)
        threshold_jitter = rules.thresholds.get("joint_jitter_max", 0.055)
        arm_raise_floor = float(rules.thresholds.get("arm_raise_min_deg", 135.0 if rules.expected_tempo == "slow" else 120.0))
        arm_raise_metrics = self._select_arm_raise_metrics(metrics)
        arm_raise_values = [
            min(float(item.arm_raise_left_deg), float(item.arm_raise_right_deg))
            for item in arm_raise_metrics
            if item.arm_raise_left_deg is not None and item.arm_raise_right_deg is not None
        ]

        if rules.expected_tempo == "slow" and avg_motion > rules.thresholds.get("motion_mean_max_slow", 2.5):
            issues.append(
                self._make_issue(
                    "tempo_too_fast",
                    "medium",
                    "动作节奏偏快",
                    f"当前平均运动强度为 {avg_motion:.2f}，高于舒缓动作推荐范围。",
                    "放慢抬手和过渡速度，给定势阶段留出更明显的停顿。",
                    metrics,
                    predicate=lambda item: item.motion_mean > rules.thresholds.get("motion_mean_max_slow", 2.5),
                    error_type="收势过快",
                    highlight_metric="motion_mean",
                    expected_value=rules.thresholds.get("motion_mean_max_slow", 2.5),
                    actual_value=avg_motion,
                    overlay_label="速度偏快",
                )
            )

        if rules.expected_tempo == "fast" and avg_motion < rules.thresholds.get("motion_mean_min_fast", 1.0):
            issues.append(
                self._make_issue(
                    "tempo_too_slow",
                    "low",
                    "动作力度偏弱",
                    f"当前平均运动强度为 {avg_motion:.2f}，低于快速动作的参考下限。",
                    "适当提高发力速度，保证关键动作段有足够的位移和节奏变化。",
                    metrics,
                    predicate=lambda item: item.motion_mean < rules.thresholds.get("motion_mean_min_fast", 1.0),
                    error_type="发力不足",
                    highlight_metric="motion_mean",
                    expected_value=rules.thresholds.get("motion_mean_min_fast", 1.0),
                    actual_value=avg_motion,
                    overlay_label="动作力度不足",
                )
            )

        if arm_raise_values and mean(arm_raise_values) < arm_raise_floor:
            arm_raise_average = float(mean(arm_raise_values))
            issues.append(
                self._make_issue(
                    "arm_raise_low",
                    "high",
                    "手臂抬高不足",
                    f"关键抬臂阶段双臂平均上举角度约 {arm_raise_average:.1f}°，低于标准动作建议角度。",
                    "提高双臂上举高度，保持肩部打开，抬臂时不要提前塌肩或收肘。",
                    arm_raise_metrics,
                    predicate=lambda item: (
                        item.arm_raise_left_deg is not None
                        and item.arm_raise_right_deg is not None
                        and self._is_arm_raise_focus_phase(item.phase_label)
                        and min(item.arm_raise_left_deg, item.arm_raise_right_deg) < arm_raise_floor
                    ),
                    error_type="手臂抬高不足",
                    highlight_metric="arm_raise_focus_deg",
                    expected_value=arm_raise_floor,
                    actual_value=arm_raise_average,
                    overlay_label="抬臂不足",
                )
            )

        if rules.requires_symmetry and symmetry_values:
            avg_symmetry = mean(symmetry_values)
            if avg_symmetry > threshold_symmetry:
                issues.append(
                    self._make_issue(
                        "arm_symmetry",
                        "high",
                        "双臂轨迹不够对称",
                        f"双臂平均夹角差约 {avg_symmetry:.1f}°，左右动作不一致。",
                        "对照标准动作检查左右手抬起高度和肘部打开角度，避免一侧先发力。",
                        metrics,
                        predicate=lambda item: (item.arm_symmetry_error or 0.0) > threshold_symmetry,
                        error_type="手臂抬高不足",
                        highlight_metric="arm_symmetry_error",
                        expected_value=threshold_symmetry,
                        actual_value=avg_symmetry,
                        overlay_label="双臂不对称",
                    )
                )
        elif rules.requires_symmetry and avg_balance > threshold_balance:
            issues.append(
                self._make_issue(
                    "frame_balance",
                    "medium",
                    "左右运动分布不均",
                    f"画面左右运动能量差为 {avg_balance:.2f}，说明动作重心偏向单侧。",
                    "检查左右手或左右腿是否同步发力，保持重心与动作路线平衡。",
                    metrics,
                    predicate=lambda item: item.left_right_balance > threshold_balance,
                    error_type="重心偏移",
                    highlight_metric="left_right_balance",
                    expected_value=threshold_balance,
                    actual_value=avg_balance,
                    overlay_label="重心偏移",
                )
            )

        if rules.requires_upright_torso and torso_values:
            avg_torso = mean(torso_values)
            if avg_torso > threshold_torso:
                issues.append(
                    self._make_issue(
                        "torso_tilt",
                        "high",
                        "躯干前倾或侧倾偏大",
                        f"平均躯干偏离垂直线约 {avg_torso:.1f}°，不符合直立要求。",
                        "收紧核心，保持肩髋连线稳定，抬手时避免上身跟着前冲。",
                        metrics,
                        predicate=lambda item: (item.torso_tilt_deg or 0.0) > threshold_torso,
                        error_type="转体角度不足",
                        highlight_metric="torso_tilt_deg",
                        expected_value=threshold_torso,
                        actual_value=avg_torso,
                        overlay_label="躯干偏移",
                    )
                )

        if motion_std > rules.thresholds.get("motion_std_max", 2.0):
            issues.append(
                self._make_issue(
                    "unstable_motion",
                    "medium",
                    "动作稳定性不足",
                    f"帧间运动波动较大，平均波动值为 {motion_std:.2f}。",
                    "减小多余抖动，确保动作过程平滑衔接，不要突然加速或停顿。",
                    metrics,
                    predicate=lambda item: item.motion_std > rules.thresholds.get("motion_std_max", 2.0),
                )
            )

        if jitter_values and mean(jitter_values) > threshold_jitter:
            issues.append(
                self._make_issue(
                    "joint_jitter",
                    "medium",
                    "骨架抖动偏大",
                    f"关键点平均抖动为 {mean(jitter_values):.3f}，连续动作衔接不够稳定。",
                    "提高拍摄稳定性，并通过更多标准视频完善模板，让动作过渡更平滑。",
                    metrics,
                    predicate=lambda item: (item.joint_jitter or 0.0) > threshold_jitter,
                )
            )

        transition_min = float(rules.thresholds.get("transition_score_min", 58.0))
        if transition_values and mean(transition_values) < transition_min:
            avg_transition = float(mean(transition_values))
            issues.append(
                self._make_issue(
                    "transition_not_smooth",
                    "medium",
                    "动作衔接不够顺畅",
                    f"连续多帧衔接评分约 {avg_transition:.1f}，说明慢动作过渡中存在突然加速、停顿或骨架跳变。",
                    "先把连接段做慢做稳，再逐步提高动作完整度；保持手臂、躯干和重心同步移动。",
                    metrics,
                    predicate=lambda item: (item.transition_score or 100.0) < transition_min,
                    error_type="动作衔接不顺",
                    highlight_metric="transition_score",
                    expected_value=transition_min,
                    actual_value=avg_transition,
                    overlay_label="衔接不顺",
                )
            )

        if pose_conf_values and stability_values:
            pose_coverage = len(pose_conf_values) / max(len(metrics), 1)
            avg_stability = mean(stability_values)
            avg_keypoint_coverage = mean(coverage_values) if coverage_values else 0.0
            stability_min = float(rules.thresholds.get("skeleton_stability_min", 0.46))
            coverage_min = float(rules.thresholds.get("keypoint_coverage_min", 0.58))
            pose_coverage_min = max(0.50, coverage_min)
            if pose_coverage < pose_coverage_min or avg_stability < stability_min or avg_keypoint_coverage < coverage_min:
                issues.append(
                    self._make_issue(
                        "tracking_loss",
                        "low",
                        "连续识别稳定性一般",
                        (
                            f"姿态识别覆盖率约 {pose_coverage:.0%}，"
                            f"关键点覆盖率约 {avg_keypoint_coverage:.0%}，骨架稳定度约 {avg_stability:.2f}。"
                        ),
                        "保持单人居中、减少遮挡，并优先使用正面固定机位拍摄；必要时补充标准动作视频提升连续跟踪。",
                        metrics,
                        predicate=lambda item: (item.skeleton_stability or 0.0) < stability_min or (item.keypoint_coverage or 0.0) < coverage_min,
                    )
                )

        if rules.requires_hold and not self._has_hold_phase(phases, fps):
            issues.append(
                self._make_issue(
                    "missing_hold",
                    "medium",
                    "定势停顿不明显",
                    "动作序列中没有检测到足够稳定的停顿阶段。",
                    "在关键姿态形成后稍作停留，让动作起承转合更清楚。",
                    metrics,
                    predicate=lambda item: item.phase_label == "发力",
                )
            )

        if len(phases) < 3:
            issues.append(
                self._make_issue(
                    "phase_sparse",
                    "low",
                    "动作阶段划分不够清楚",
                    "当前动作序列的起势、过渡和收势区分不明显。",
                    "重新录制完整动作过程，避免起始或结束段过短。",
                    metrics,
                )
            )

        return issues

    def _build_summary(
        self,
        metrics: list[FrameMetrics],
        issues: list[Issue],
        rules: RuleSet,
        phases: list[dict],
        analysis_mode: str = "test",
        evaluator_level: str = "熟练者",
        perfect_match_override_reason: str | None = None,
    ) -> AnalysisSummary:
        avg_motion = mean(item.motion_mean for item in metrics)
        avg_balance = mean(item.left_right_balance for item in metrics)
        torso_values = [item.torso_tilt_deg for item in metrics if item.torso_tilt_deg is not None]
        symmetry_values = [item.arm_symmetry_error for item in metrics if item.arm_symmetry_error is not None]
        arm_raise_values = [
            min(float(item.arm_raise_left_deg), float(item.arm_raise_right_deg))
            for item in self._select_arm_raise_metrics(metrics)
            if item.arm_raise_left_deg is not None and item.arm_raise_right_deg is not None
        ]
        jitter_values = [item.joint_jitter for item in metrics if item.joint_jitter is not None]
        stability_values = [item.skeleton_stability for item in metrics if item.skeleton_stability is not None]
        coverage_values = [item.keypoint_coverage for item in metrics if item.keypoint_coverage is not None]
        quality_values = [item.pose_quality_score for item in metrics if item.pose_quality_score is not None]
        transition_values = [item.transition_score for item in metrics if item.transition_score is not None]
        avg_torso_tilt = mean(torso_values) if torso_values else None
        avg_arm_symmetry_error = mean(symmetry_values) if symmetry_values else None
        avg_arm_raise = mean(arm_raise_values) if arm_raise_values else None
        avg_keypoint_coverage = mean(coverage_values) if coverage_values else None
        avg_pose_quality_score = mean(quality_values) if quality_values else None

        posture_score = self._clamp_score(
            100.0
            - (mean(torso_values) * 1.35 if torso_values else 0.0)
            - (mean(symmetry_values) * 0.85 if symmetry_values else avg_balance * 120.0)
        )
        continuity_score = self._clamp_score(
            100.0
            - mean(item.motion_std for item in metrics) * 16.0
            - (mean(jitter_values) * 820.0 if jitter_values else 0.0)
            - ((70.0 - mean(transition_values)) * 0.45 if transition_values and mean(transition_values) < 70.0 else 0.0)
        )
        stability_score = self._clamp_score(
            100.0
            - avg_balance * 135.0
            - ((1.0 - mean(stability_values)) * 52.0 if stability_values else 0.0)
            - ((1.0 - mean(coverage_values)) * 24.0 if coverage_values else 0.0)
        )
        rhythm_score = self._score_rhythm(avg_motion, metrics, rules, phases)
        completeness_score = self._score_completeness(phases, rules)

        penalty = sum({"low": 4, "medium": 9, "high": 14}.get(issue.severity, 6) for issue in issues)
        if not self.pose_estimator.available:
            penalty += 4
        raw_total = mean(
            [
                posture_score,
                continuity_score,
                stability_score,
                rhythm_score,
                completeness_score,
            ]
        )
        total_score = self._clamp_score(raw_total - penalty * 0.35)

        learner_level = self._classify_learner_level(total_score, issues, transition_values, coverage_values)
        improvement_focus = self._build_improvement_focus(
            issues,
            rules=rules,
            summary_score=total_score,
            learner_level=learner_level,
        )
        advice = [issue.suggestion for issue in issues[:5]]
        if analysis_mode == "test":
            advice.extend(improvement_focus[:3])
        if not advice:
            advice = [
                "当前动作连续性较稳定，可继续导入标准模板视频，细化阶段对比。",
                "如果要提高动作评分精度，建议补充更多标准动作视频和错误示例视频。",
            ]
        if not self.pose_estimator.available:
            advice.append("当前未接入 YOLOv8-Pose 权重，已使用光流兜底分析，建议尽快补权重文件。")

        if perfect_match_override_reason:
            posture_score = 100.0
            continuity_score = 100.0
            stability_score = 100.0
            rhythm_score = 100.0
            completeness_score = 100.0
            total_score = 100.0
            avg_balance = 0.0
            avg_torso_tilt = 0.0
            avg_arm_symmetry_error = 0.0
            advice = [perfect_match_override_reason]
            improvement_focus = []

        return AnalysisSummary(
            total_score=total_score,
            posture_score=posture_score,
            continuity_score=continuity_score,
            stability_score=stability_score,
            rhythm_score=rhythm_score,
            completeness_score=completeness_score,
            avg_motion=avg_motion,
            avg_left_right_balance=avg_balance,
            avg_torso_tilt=avg_torso_tilt,
            avg_arm_symmetry_error=avg_arm_symmetry_error,
            avg_arm_raise=avg_arm_raise,
            avg_keypoint_coverage=avg_keypoint_coverage,
            avg_pose_quality_score=avg_pose_quality_score,
            pose_quality_level=self._classify_pose_quality(mean(quality_values)) if quality_values else None,
            phases=phases,
            advice=advice[:6],
            mode_label="模板原始数据生成" if analysis_mode == "template" else "测试者分析",
            pose_backend_reason=self.pose_estimator.status.reason,
            evaluator_level=evaluator_level if analysis_mode == "test" else "模板标准",
            learner_level=learner_level if analysis_mode == "test" else "模板基线",
            improvement_focus=improvement_focus if analysis_mode == "test" else [],
        )

    def _classify_learner_level(
        self,
        total_score: float,
        issues: list[Issue],
        transition_values: list[float | None],
        coverage_values: list[float | None],
    ) -> str:
        high_issue_count = sum(1 for issue in issues if issue.severity == "high")
        avg_transition = mean([float(v) for v in transition_values if v is not None]) if transition_values else 0.0
        avg_coverage = mean([float(v) for v in coverage_values if v is not None]) if coverage_values else 0.0
        if total_score >= 82 and high_issue_count == 0 and (not transition_values or avg_transition >= 68):
            return "熟练者"
        if total_score >= 65 and high_issue_count <= 1 and (not coverage_values or avg_coverage >= 0.55):
            return "进阶者"
        return "初习者"

    def _build_improvement_focus(
        self,
        issues: list[Issue],
        *,
        rules: RuleSet,
        summary_score: float,
        learner_level: str,
    ) -> list[str]:
        focus: list[str] = []
        issue_titles = [issue.error_type or issue.title for issue in issues]
        category = rules.action_category or "通用动作"
        category_plan = self._category_improvement_plan(category)
        if any("手臂" in title or "抬" in title for title in issue_titles):
            focus.append(category_plan["arms"])
        if any("重心" in title or "平衡" in title for title in issue_titles):
            focus.append(category_plan["balance"])
        if any("躯干" in title or "转体" in title for title in issue_titles):
            focus.append(category_plan["torso"])
        if any("节奏" in title or "收势" in title or "衔接" in title for title in issue_titles):
            focus.append(category_plan["tempo"])
        if any("识别" in title or "骨架" in title for title in issue_titles):
            focus.append("优化方向：固定机位，全身入镜，减少遮挡后再做动作质量判断。")
        if not focus and category_plan.get("default"):
            focus.append(category_plan["default"])
        if learner_level == "初习者":
            focus.append(category_plan["beginner"])
        elif learner_level == "进阶者":
            focus.append(category_plan["intermediate"])
        else:
            focus.append(category_plan["skilled"])
        if summary_score < 60:
            focus.append(category_plan["low_score"])
        return list(dict.fromkeys(focus))[:6]

    def _category_improvement_plan(self, category: str) -> dict[str, str]:
        if category == "健身气功":
            return {
                "arms": "健身气功优化：先统一双手启动和到位高度，再配合沉肩坠肘，避免只追求抬高。",
                "balance": "健身气功优化：先把重心放回两脚之间，保持身体中线，再练习手臂路线。",
                "torso": "健身气功优化：保持肩髋连线中正，动作慢而不断，不要用躯干前冲带动手臂。",
                "tempo": "健身气功优化：把起势、过渡、定势、收势分清楚，关键定势短暂停留。",
                "default": "健身气功优化：优先保证身形中正、呼吸自然、动作连贯，再提高幅度。",
                "beginner": "阶段目标：初习者先完成完整动作和稳定站姿，不急于追求大幅度。",
                "intermediate": "阶段目标：进阶者重点提高左右对称、定势停顿和节奏均匀。",
                "skilled": "阶段目标：熟练者重点细化呼吸配合、劲力连贯和模板相似度。",
                "low_score": "训练建议：建议按准备、起势、定势、收势分段练习后再整套复测。",
            }
        if category == "太极":
            return {
                "arms": "太极优化：手臂路线要圆活连贯，先找棚劲支撑，再调整手高和手型。",
                "balance": "太极优化：重心转换要先稳后移，避免上身先走、脚下滞后。",
                "torso": "太极优化：保持立身中正，以腰胯带动上肢，不要单靠手臂摆动。",
                "tempo": "太极优化：动作速度保持匀缓，转接处不断劲，避免突然停顿或抢拍。",
                "default": "太极优化：优先练习重心转换、腰胯带动和上下相随。",
                "beginner": "阶段目标：初习者先把步法和重心路线走稳，手型可先降低要求。",
                "intermediate": "阶段目标：进阶者重点提高上下相随、虚实转换和路线圆活。",
                "skilled": "阶段目标：熟练者重点细化腰胯发力、连贯劲路和节奏层次。",
                "low_score": "训练建议：建议先拆成步法、转腰、手臂三段练习，再合成完整动作。",
            }
        if category == "武术基本功":
            return {
                "arms": "武术基本功优化：上肢动作先保证出手路线、力点和收放清楚，再提高速度。",
                "balance": "武术基本功优化：先稳定步型和支撑脚，再做冲拳、踢腿或转体动作。",
                "torso": "武术基本功优化：躯干要保持发力方向明确，避免塌腰、晃肩或转体不足。",
                "tempo": "武术基本功优化：发力动作要有启动、加速、定型，不要全程平均用力。",
                "default": "武术基本功优化：优先保证步型、力点、路线，再追求速度和爆发。",
                "beginner": "阶段目标：初习者先稳定马步/弓步等基础步型，降低速度要求。",
                "intermediate": "阶段目标：进阶者重点提高发力清晰度、手脚配合和定型稳定。",
                "skilled": "阶段目标：熟练者重点提高爆发力、节奏变化和动作规格。",
                "low_score": "训练建议：建议先做静态步型和慢速路线，再加入速度与发力。",
            }
        if category == "民族舞":
            return {
                "arms": "民族舞优化：手臂先跟准节拍和路线，再处理手腕、手型和上身韵律。",
                "balance": "民族舞优化：转身或摆臂时保持轴心稳定，避免上身抢节拍导致重心偏移。",
                "torso": "民族舞优化：躯干要服务于韵律表达，先稳定胸腰方向，再加强幅度。",
                "tempo": "民族舞优化：优先跟准节拍点，动作之间要有连贯过渡和清晰收束。",
                "default": "民族舞优化：优先保证节拍、路线和身体韵律，再提高表现幅度。",
                "beginner": "阶段目标：初习者先跟准节拍和方向，不急于加大摆幅。",
                "intermediate": "阶段目标：进阶者重点提高上肢路线、身体韵律和重心控制。",
                "skilled": "阶段目标：熟练者重点细化节奏层次、姿态美感和动作表现力。",
                "low_score": "训练建议：建议先按节拍分段练习，再加入转身、摆臂和表情处理。",
            }
        if category == "站姿稳定":
            return {
                "arms": "站姿优化：手臂先保持自然放松，不要因手部动作破坏身体中线。",
                "balance": "站姿优化：先调整两脚受力和骨盆位置，让重心稳定落在支撑面内。",
                "torso": "站姿优化：保持头、胸、骨盆纵向对齐，减少肩部高低差。",
                "tempo": "站姿优化：定势保持要稳定，不要在停顿阶段持续晃动。",
                "default": "站姿优化：优先保证头肩髋脚的垂直关系和静态稳定。",
                "beginner": "阶段目标：初习者先做到全身入镜、站稳和不明显晃动。",
                "intermediate": "阶段目标：进阶者重点减少肩髋偏斜和左右重心漂移。",
                "skilled": "阶段目标：熟练者重点保持定势稳定和细微姿态控制。",
                "low_score": "训练建议：建议先做静态站姿保持，再加入上肢或步法动作。",
            }
        return {
            "arms": "通用优化：先统一左右手启动和到位高度，再逐步提高动作幅度。",
            "balance": "通用优化：先稳定两脚支撑和身体中线，避免动作长期偏向单侧。",
            "torso": "通用优化：收紧核心，保持肩髋连线稳定，再做上肢或步法调整。",
            "tempo": "通用优化：降低动作速度，把准备、发力、定型和收势分清楚。",
            "default": "通用优化：优先保证动作完整、身体稳定和路线清楚。",
            "beginner": "阶段目标：初习者先达成动作完整和身体稳定，不急于追求标准幅度。",
            "intermediate": "阶段目标：进阶者重点提高左右对称、节奏停顿和关键姿态到位程度。",
            "skilled": "阶段目标：熟练者重点细化节奏、力度控制和模板相似度。",
            "low_score": "训练建议：建议先分段练习，再录制完整动作进行复测。",
        }

    def _score_rhythm(
        self,
        avg_motion: float,
        metrics: list[FrameMetrics],
        rules: RuleSet,
        phases: list[dict],
    ) -> float:
        variability = mean(item.motion_std for item in metrics)
        if rules.expected_tempo == "slow":
            base = 100.0 - abs(avg_motion - 1.0) * 22.0
        elif rules.expected_tempo == "fast":
            base = 100.0 - abs(avg_motion - 2.6) * 18.0
        else:
            base = 100.0 - abs(avg_motion - 1.6) * 19.0
        base -= variability * 10.0
        if len(phases) < 3:
            base -= 8.0
        if phases and phases[-1]["name"] != "收势":
            base -= 4.0
        return self._clamp_score(base)

    def _score_completeness(self, phases: list[dict], rules: RuleSet) -> float:
        score = 60.0
        names = [phase["name"] for phase in phases]
        if len(phases) >= 3:
            score += 12.0
        if "起势" in names:
            score += 8.0
        if "过渡" in names:
            score += 8.0
        if "收势" in names:
            score += 6.0
        if rules.requires_hold and "定势" in names:
            score += 10.0
        if phases and phases[0]["name"] in {"准备", "起势"}:
            score += 4.0
        if phases and phases[-1]["name"] == "收势":
            score += 4.0
        return self._clamp_score(score)

    def _build_comparison_result(
        self,
        metrics: list[FrameMetrics],
        summary: AnalysisSummary,
        rules: RuleSet,
        phases: list[dict],
        source: str,
        fps: float,
        analysis_mode: str = "test",
        baseline_payload: dict | None = None,
        perfect_match_override_reason: str | None = None,
        phase_snapshots: list[dict] | None = None,
    ) -> ComparisonResult:
        baseline_name = rules.template_name or "标准模板"
        baseline_source = source
        baseline_kind = "规则模板基线"
        alignment_mode = "规则基线 + 阶段对齐预备"
        baseline_summary: dict = {}

        if baseline_payload:
            baseline_name = str(baseline_payload.get("模板名称") or baseline_name)
            baseline_source = str(baseline_payload.get("来源视频") or baseline_payload.get("输入源") or baseline_source)
            baseline_kind = "模板原始数据基线"
            alignment_mode = "模板原始数据 + 阶段对齐"
            baseline_summary = dict(baseline_payload.get("模板摘要") or {})

        motion_baseline = self._expected_motion_baseline(rules.expected_tempo)
        balance_baseline = rules.thresholds.get("left_right_balance_max", 0.18)
        torso_baseline = rules.thresholds.get("torso_tilt_max_deg") if rules.requires_upright_torso else None
        symmetry_baseline = rules.thresholds.get("arm_symmetry_error_max") if rules.requires_symmetry else None
        arm_raise_baseline = rules.thresholds.get("arm_raise_min_deg", 135.0 if rules.expected_tempo == "slow" else 120.0)
        hold_baseline = 1.0 if rules.requires_hold else None
        coverage_baseline = 0.75 if self.pose_estimator.available else None
        transition_baseline = 70.0

        if baseline_summary:
            motion_baseline = baseline_summary.get("平均运动强度", motion_baseline)
            balance_baseline = baseline_summary.get("平均左右平衡差", balance_baseline)
            torso_baseline = baseline_summary.get("平均躯干倾斜角度", torso_baseline)
            symmetry_baseline = baseline_summary.get("平均双臂对称误差", symmetry_baseline)
            arm_raise_baseline = baseline_summary.get("平均抬臂角度", baseline_summary.get("平均上举角度", arm_raise_baseline))
            coverage_baseline = baseline_summary.get("平均关键点覆盖率", coverage_baseline)
            transition_baseline = baseline_summary.get("平均衔接评分", transition_baseline)
            hold_baseline = 1.0 if hold_baseline is not None else None

        comparison_metrics = [
            self._build_metric_comparison(
                "节奏强度",
                motion_baseline,
                summary.avg_motion,
                "motion",
                "越接近标准节奏越好",
            ),
            self._build_metric_comparison(
                "左右平衡差",
                balance_baseline,
                summary.avg_left_right_balance,
                "balance",
                "数值越低越接近左右均衡",
            ),
            self._build_metric_comparison(
                "躯干倾斜",
                torso_baseline,
                summary.avg_torso_tilt,
                "max",
                "要求躯干保持中正时才作为强约束",
                unit="度",
            ),
            self._build_metric_comparison(
                "双臂对称误差",
                symmetry_baseline,
                summary.avg_arm_symmetry_error,
                "max",
                "要求双臂对称时才作为强约束",
                unit="度",
            ),
            self._build_metric_comparison(
                "抬臂角度",
                arm_raise_baseline,
                self._average_arm_raise(metrics),
                "min",
                "数值越高越接近标准抬臂高度",
                unit="度",
            ),
            self._build_metric_comparison(
                "定势停顿",
                hold_baseline,
                1.0 if self._has_hold_phase(phases, fps) else 0.0,
                "hold",
                "有定势要求时，应至少识别到一次稳定停顿",
            ),
            self._build_metric_comparison(
                "关键点覆盖率",
                coverage_baseline,
                summary.avg_keypoint_coverage,
                "coverage",
                "仅在启用姿态模型时有意义",
            ),
            self._build_metric_comparison(
                "动作衔接评分",
                transition_baseline,
                self._average_transition_score(metrics),
                "min",
                "连续多帧越平顺，越适合健身气功慢动作分析",
            ),
        ]
        comparison_phases = self._build_phase_comparison(rules, phases, baseline_payload=baseline_payload)
        sequence_similarity_score = self._estimate_sequence_similarity(metrics, baseline_payload=baseline_payload)
        snapshot_pairs = []
        if analysis_mode != "template":
            snapshot_pairs = self._build_snapshot_pairs(
                metrics,
                phases,
                baseline_payload=baseline_payload,
                phase_snapshots=phase_snapshots,
            )
        missing_items = []
        if not self.pose_estimator.available:
            missing_items.append("当前未加载姿态模型，标准骨架与测试骨架无法形成完整关键点对比。")
        if not phases:
            missing_items.append("当前分析未形成稳定阶段，阶段级对比可信度有限。")
        if analysis_mode == "template":
            missing_items.append("当前结果为模板原始数据生成结果，可用于后续测试者动作比对。")
        elif not baseline_payload:
            missing_items.append("当前未加载模板基线，已按动作描述和通用规则进行实时分析；加载模板后可生成左右阶段快照对比。")
        if perfect_match_override_reason:
            for item in comparison_metrics:
                if item.baseline_value is not None:
                    item.current_value = item.baseline_value
                item.delta_value = 0.0
                item.status = "完美"
                item.note = perfect_match_override_reason
            sequence_similarity_score = 100.0
        overall_assessment = self._build_comparison_assessment(comparison_metrics, missing_items)
        if perfect_match_override_reason:
            overall_assessment = (
                f"{overall_assessment}\n{perfect_match_override_reason}"
                if overall_assessment else perfect_match_override_reason
            )
        return ComparisonResult(
            baseline_name=baseline_name,
            baseline_source=baseline_source,
            baseline_kind=baseline_kind,
            alignment_mode=alignment_mode,
            metrics=comparison_metrics,
            phases=comparison_phases,
            overall_assessment=overall_assessment,
            missing_items=missing_items,
            baseline_summary=baseline_summary,
            sequence_similarity_score=sequence_similarity_score,
            sequence_similarity_label=self._sequence_similarity_label(sequence_similarity_score),
            snapshot_pairs=snapshot_pairs,
        )

    def _estimate_sequence_similarity(
        self,
        metrics: list[FrameMetrics],
        baseline_payload: dict | None = None,
    ) -> float | None:
        if not metrics:
            return None
        baseline_rows = (baseline_payload or {}).get("原始逐帧数据") or []
        if not baseline_rows:
            return None
        current_signal = np.array([float(item.motion_mean or 0.0) for item in metrics], dtype=np.float32)
        baseline_signal = np.array([float((row.get("motion_mean") or row.get("运动强度均值") or 0.0)) for row in baseline_rows], dtype=np.float32)
        if len(current_signal) < 3 or len(baseline_signal) < 3:
            return None
        current_norm = self._normalize_sequence(current_signal)
        baseline_norm = self._normalize_sequence(baseline_signal)
        distance = self._dtw_distance(current_norm, baseline_norm)
        normalized_distance = distance / max(len(current_norm) + len(baseline_norm), 1)
        return float(max(0.0, min(100.0, 100.0 - normalized_distance * 120.0)))

    def _normalize_sequence(self, values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        span = max(max_value - min_value, 1e-6)
        return (values - min_value) / span

    def _dtw_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        rows = len(a)
        cols = len(b)
        dp = np.full((rows + 1, cols + 1), np.inf, dtype=np.float32)
        dp[0, 0] = 0.0
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                cost = abs(float(a[i - 1] - b[j - 1]))
                dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
        return float(dp[rows, cols])

    def _sequence_similarity_label(self, score: float | None) -> str:
        if score is None:
            return "未计算"
        if score >= 85:
            return "高度接近模板"
        if score >= 70:
            return "基本接近模板"
        if score >= 55:
            return "存在中等偏差"
        return "与模板差异较大"

    def _build_snapshot_pairs(
        self,
        metrics: list[FrameMetrics],
        phases: list[dict],
        baseline_payload: dict | None = None,
        phase_snapshots: list[dict] | None = None,
    ) -> list[dict]:
        if not metrics:
            return []
        baseline_phases = (baseline_payload or {}).get("模板阶段") or []
        baseline_snapshots = (baseline_payload or {}).get("阶段快照") or []
        baseline_rows = (baseline_payload or {}).get("原始逐帧数据") or []
        current_lookup = {phase.get("name"): phase for phase in phases if phase.get("name")}
        baseline_lookup = {phase.get("name"): phase for phase in baseline_phases if phase.get("name")}
        current_snapshot_lookup = {
            snapshot.get("phase_name"): snapshot for snapshot in (phase_snapshots or []) if snapshot.get("phase_name")
        }
        baseline_snapshot_lookup = {
            snapshot.get("phase_name"): snapshot for snapshot in baseline_snapshots if snapshot.get("phase_name")
        }
        candidates = []
        for name in ["起势", "过渡", "定势", "发力", "收势"]:
            current_phase = current_lookup.get(name)
            if current_phase is None:
                continue
            start_sec = float(current_phase.get("start_sec") or 0.0)
            end_sec = float(current_phase.get("end_sec") or start_sec)
            middle_sec = (start_sec + end_sec) / 2.0
            nearest = min(metrics, key=lambda item: abs(item.timestamp_sec - middle_sec))
            baseline_phase = baseline_lookup.get(name)
            baseline_middle = None
            if baseline_phase is not None:
                baseline_middle = (
                    float(baseline_phase.get("start_sec") or 0.0) + float(baseline_phase.get("end_sec") or 0.0)
                ) / 2.0
            current_snapshot = current_snapshot_lookup.get(name) or {}
            baseline_snapshot = baseline_snapshot_lookup.get(name) or {}
            baseline_keypoint_positions = baseline_snapshot.get("keypoint_positions") or {}
            if not baseline_keypoint_positions and baseline_middle is not None:
                baseline_keypoint_positions = self._find_baseline_keypoints_at_time(baseline_rows, baseline_middle)
            candidates.append(
                {
                    "phase_name": name,
                    "current_time_sec": round(float(nearest.timestamp_sec), 2),
                    "baseline_time_sec": round(float(baseline_middle), 2) if baseline_middle is not None else None,
                    "baseline_motion": baseline_snapshot.get("motion"),
                    "current_motion": round(float(nearest.motion_mean), 3),
                    "current_balance": round(float(nearest.left_right_balance), 3),
                    "current_pose_quality": round(float(nearest.pose_quality_score or 0.0), 1) if nearest.pose_quality_score is not None else None,
                    "current_image_path": current_snapshot.get("image_path"),
                    "baseline_image_path": baseline_snapshot.get("image_path"),
                    "baseline_keypoint_positions": baseline_keypoint_positions,
                    "current_keypoint_positions": current_snapshot.get("keypoint_positions") or {},
                    "current_issue_labels": current_snapshot.get("issue_labels") or [],
                }
            )
        return candidates[:4]

    def _generate_phase_snapshots(
        self,
        source: str | int,
        metrics: list[FrameMetrics],
        phases: list[dict],
        issues: list[Issue],
        output_dir: Path,
    ) -> list[dict]:
        if not phases or isinstance(source, int):
            return []
        source_path = Path(str(source)).expanduser()
        if not source_path.exists():
            return []
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            return []
        output_dir.mkdir(parents=True, exist_ok=True)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        snapshots: list[dict] = []
        try:
            for index, phase in enumerate(phases[:6], start=1):
                phase_name = str(phase.get("name") or f"phase_{index}")
                start_sec = float(phase.get("start_sec") or 0.0)
                end_sec = float(phase.get("end_sec") or start_sec)
                middle_sec = (start_sec + end_sec) / 2.0
                frame_number = max(0, int(round(middle_sec * fps)))
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                frame = self._resize_frame(frame)
                pose = self.pose_estimator.estimate_with_tracking(frame, previous_pose=None)
                nearest = min(metrics, key=lambda item: abs(item.timestamp_sec - middle_sec))
                issue_labels = [
                    issue.overlay_label or issue.error_type or issue.title
                    for issue in issues
                    if issue.time_range and issue.time_range[0] <= middle_sec <= issue.time_range[1]
                ]
                snapshot = draw_pose(frame.copy(), pose)
                if issue_labels:
                    snapshot = self._draw_issue_hints(snapshot, issue_labels[:2], pose, nearest)
                snapshot = self._draw_multiline_text(
                    snapshot,
                    [
                        f"{phase_name} · {middle_sec:.2f}s",
                        f"运动强度 {nearest.motion_mean:.3f}",
                        f"左右平衡差 {nearest.left_right_balance:.3f}",
                        (
                            f"姿态质量 {nearest.pose_quality_score:.1f}"
                            if nearest.pose_quality_score is not None
                            else "姿态质量 暂无"
                        ),
                    ],
                    origin=(18, 18),
                    color=(255, 255, 255),
                    background=(20, 39, 59, 188),
                    line_spacing=5,
                )
                filename = f"{index:02d}_{self._safe_snapshot_name(phase_name)}.jpg"
                image_path = output_dir / filename
                cv2.imwrite(str(image_path), snapshot)
                snapshots.append(
                    {
                        "phase_name": phase_name,
                        "time_sec": round(middle_sec, 2),
                        "frame_index": frame_number,
                        "image_path": str(image_path),
                        "issue_labels": issue_labels[:3],
                        "motion": round(float(nearest.motion_mean), 3),
                        "balance": round(float(nearest.left_right_balance), 3),
                        "pose_quality": round(float(nearest.pose_quality_score), 1) if nearest.pose_quality_score is not None else None,
                        "keypoint_positions": nearest.keypoint_positions,
                    }
                )
        finally:
            capture.release()
        return snapshots

    def _find_baseline_keypoints_at_time(
        self,
        baseline_rows: list[dict],
        timestamp_sec: float,
    ) -> dict[str, list[float]]:
        if not baseline_rows:
            return {}
        matched_row = None
        smallest_delta = float("inf")
        for row in baseline_rows:
            row_time = row.get("timestamp_sec")
            if row_time is None:
                row_time = row.get("时间秒")
            if row_time is None:
                continue
            delta = abs(float(row_time) - float(timestamp_sec))
            if delta < smallest_delta:
                smallest_delta = delta
                matched_row = row
        if matched_row is None:
            return {}
        positions = matched_row.get("keypoint_positions") or matched_row.get("关键点坐标") or {}
        return self._normalize_keypoint_positions(positions)

    def _normalize_keypoint_positions(self, positions: dict) -> dict[str, list[float]]:
        if not isinstance(positions, dict):
            return {}
        normalized: dict[str, list[float]] = {}
        chinese_alias = {
            "鼻尖": "nose",
            "左眼": "left_eye",
            "右眼": "right_eye",
            "左耳": "left_ear",
            "右耳": "right_ear",
            "左肩": "left_shoulder",
            "右肩": "right_shoulder",
            "左肘": "left_elbow",
            "右肘": "right_elbow",
            "左腕": "left_wrist",
            "右腕": "right_wrist",
            "左髋": "left_hip",
            "右髋": "right_hip",
            "左膝": "left_knee",
            "右膝": "right_knee",
            "左踝": "left_ankle",
            "右踝": "right_ankle",
        }
        for key, value in positions.items():
            mapped_key = chinese_alias.get(str(key), str(key))
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                continue
            normalized[mapped_key] = [float(value[0]), float(value[1])]
        return normalized

    def _safe_snapshot_name(self, value: str) -> str:
        safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value.strip())
        return safe or "phase"

    def _expected_motion_baseline(self, tempo: str) -> float | None:
        if tempo == "slow":
            return 1.0
        if tempo == "fast":
            return 2.6
        if tempo == "medium":
            return 1.6
        return None

    def _build_metric_comparison(
        self,
        label: str,
        baseline_value: float | None,
        current_value: float | None,
        mode: str,
        note: str,
        unit: str = "",
    ) -> ComparisonMetric:
        delta_value = None
        status = "未知"
        if baseline_value is not None and current_value is not None:
            delta_value = current_value - baseline_value
            if mode in {"max", "balance", "coverage", "min"}:
                if mode == "coverage":
                    status = "良好" if current_value >= baseline_value else "注意"
                elif mode == "min":
                    status = "良好" if current_value >= baseline_value else "注意"
                else:
                    status = "良好" if current_value <= baseline_value else "注意"
            elif mode == "hold":
                status = "良好" if current_value >= baseline_value else "注意"
            else:
                status = "良好" if abs(delta_value) <= max(0.15 * max(abs(baseline_value), 1.0), 0.18) else "注意"
        elif current_value is None:
            status = "缺失"
        return ComparisonMetric(
            label=label,
            baseline_value=baseline_value,
            current_value=current_value,
            delta_value=delta_value,
            unit=unit,
            status=status,
            note=note,
        )

    def _average_arm_raise(self, metrics: list[FrameMetrics]) -> float | None:
        values = [
            min(float(item.arm_raise_left_deg), float(item.arm_raise_right_deg))
            for item in self._select_arm_raise_metrics(metrics)
            if item.arm_raise_left_deg is not None and item.arm_raise_right_deg is not None
        ]
        if not values:
            return None
        return float(mean(values))

    def _average_transition_score(self, metrics: list[FrameMetrics]) -> float | None:
        values = [float(item.transition_score) for item in metrics if item.transition_score is not None]
        if not values:
            return None
        return float(mean(values))

    def _select_arm_raise_metrics(self, metrics: list[FrameMetrics]) -> list[FrameMetrics]:
        focused = [item for item in metrics if self._is_arm_raise_focus_phase(item.phase_label)]
        source = focused or metrics
        values = [
            min(float(item.arm_raise_left_deg), float(item.arm_raise_right_deg))
            for item in source
            if item.arm_raise_left_deg is not None and item.arm_raise_right_deg is not None
        ]
        if not values:
            return source
        pivot = float(np.percentile(np.array(values, dtype=np.float32), 70))
        selected = [
            item
            for item in source
            if item.arm_raise_left_deg is not None
            and item.arm_raise_right_deg is not None
            and min(float(item.arm_raise_left_deg), float(item.arm_raise_right_deg)) >= pivot
        ]
        return selected or source

    def _is_arm_raise_focus_phase(self, phase_label: str | None) -> bool:
        return phase_label in {"起势", "发力", "定势"}

    def _build_phase_comparison(
        self,
        rules: RuleSet,
        phases: list[dict],
        baseline_payload: dict | None = None,
    ) -> list[ComparisonPhase]:
        if baseline_payload and baseline_payload.get("模板阶段"):
            baseline_phases = baseline_payload.get("模板阶段") or []
            normalized_expected = []
            for phase in baseline_phases:
                name = phase.get("name")
                if name and name not in normalized_expected:
                    normalized_expected.append(name)
        else:
            expected_order = ["准备", "起势", "过渡"]
            if rules.requires_hold:
                expected_order.append("定势")
            expected_order.extend(["发力", "收势"])
            normalized_expected = []
            for name in expected_order:
                if name not in normalized_expected:
                    normalized_expected.append(name)

        phase_lookup = {phase["name"]: phase for phase in phases}
        baseline_lookup = {
            phase.get("name"): phase for phase in (baseline_payload.get("模板阶段") or [])
        } if baseline_payload else {}
        comparisons: list[ComparisonPhase] = []
        for name in normalized_expected:
            current = phase_lookup.get(name)
            current_duration = float(current["duration_sec"]) if current else None
            expected_duration = None
            if baseline_lookup.get(name):
                expected_duration = float(baseline_lookup[name].get("duration_sec") or 0.0)
            if expected_duration in {0.0, None}:
                expected_duration = self._expected_phase_duration(name, rules)
            delta = None if current_duration is None or expected_duration is None else current_duration - expected_duration
            note = self._build_phase_note(name, current_duration, expected_duration)
            comparisons.append(
                ComparisonPhase(
                    baseline_name=name,
                    current_name=current["name"] if current else None,
                    baseline_duration_sec=expected_duration,
                    current_duration_sec=current_duration,
                    delta_duration_sec=delta,
                    note=note,
                )
            )
        return comparisons

    def _expected_phase_duration(self, name: str, rules: RuleSet) -> float | None:
        if name == "准备":
            return 0.6
        if name == "起势":
            return 1.6 if rules.expected_tempo == "slow" else 1.1
        if name == "过渡":
            return 0.9
        if name == "定势":
            return 0.8 if rules.requires_hold else None
        if name == "发力":
            return 1.0 if rules.expected_tempo != "slow" else 1.4
        if name == "收势":
            return 0.8
        return None

    def _build_phase_note(self, name: str, current_duration: float | None, expected_duration: float | None) -> str:
        if current_duration is None:
            return f"当前未识别到{name}阶段。"
        if expected_duration is None:
            return f"{name}阶段已识别。"
        delta = current_duration - expected_duration
        if abs(delta) <= 0.35:
            return f"{name}阶段时长基本接近参考范围。"
        if delta > 0:
            return f"{name}阶段偏长，动作节奏可能偏慢或停顿过久。"
        return f"{name}阶段偏短，动作衔接可能过快。"

    def _build_comparison_assessment(
        self,
        metrics: list[ComparisonMetric],
        missing_items: list[str],
    ) -> str:
        matched = sum(1 for item in metrics if item.status in {"完美", "极佳", "良好"})
        warned = sum(1 for item in metrics if item.status in {"一般", "注意", "偏差", "较差"})
        if missing_items:
            return (
                f"当前对比以规则基线为主，已有 {matched} 项接近标准，"
                f"{warned} 项存在偏差，但仍缺少完整标准骨架样本对照。"
            )
        if warned == 0:
            return "当前结果整体接近标准基线，可继续补充标准视频做逐阶段和逐骨架对比。"
        return f"当前结果与标准基线相比存在 {warned} 项明显偏差，建议优先查看差异表和问题时间段。"

    def _render_comparison_text(self, comparison: ComparisonResult | None) -> str:
        if comparison is None:
            return "暂无详细对比结果。"
        lines = [
            "详细对比摘要",
            f"基线名称：{comparison.baseline_name}",
            f"基线类型：{comparison.baseline_kind}",
            f"对齐方式：{comparison.alignment_mode}",
            "",
            "指标对比：",
        ]
        for metric in comparison.metrics:
            baseline = "暂无" if metric.baseline_value is None else f"{metric.baseline_value:.3f}{metric.unit}"
            current = "暂无" if metric.current_value is None else f"{metric.current_value:.3f}{metric.unit}"
            delta = "暂无" if metric.delta_value is None else f"{metric.delta_value:+.3f}{metric.unit}"
            lines.append(f"- {metric.label}：标准 {baseline} / 当前 {current} / 差值 {delta}")
            if metric.note:
                lines.append(f"  说明：{metric.note}")
        lines.extend(["", "阶段对比："])
        for phase in comparison.phases:
            baseline = "暂无" if phase.baseline_duration_sec is None else f"{phase.baseline_duration_sec:.2f}s"
            current = "未识别" if phase.current_duration_sec is None else f"{phase.current_duration_sec:.2f}s"
            lines.append(f"- {phase.baseline_name or '未定义'}：标准 {baseline} / 当前 {current}")
            if phase.note:
                lines.append(f"  说明：{phase.note}")
        if comparison.missing_items:
            lines.extend(["", "当前缺口："])
            lines.extend([f"- {item}" for item in comparison.missing_items])
        lines.extend(["", "总体判断：", comparison.overall_assessment])
        return "\n".join(lines)

    def _segment_phases(self, metrics: list[FrameMetrics], rules: RuleSet) -> list[dict]:
        if not metrics:
            return []

        smooth_window = 9 if rules.expected_tempo == "slow" else 7
        motion_signal = self._smooth_numeric_signal(
            self._normalize_values([item.motion_mean for item in metrics]),
            window=smooth_window,
        )
        pose_signal = self._smooth_numeric_signal(
            self._normalize_values([item.pose_motion for item in metrics]),
            window=7,
        )
        balance_signal = self._smooth_numeric_signal(
            self._normalize_values([item.left_right_balance for item in metrics]),
            window=7,
        )
        active_signal = self._smooth_numeric_signal(
            self._normalize_values([item.active_region_ratio for item in metrics]),
            window=7,
        )
        stability_signal = 1.0 - self._smooth_numeric_signal(
            self._normalize_values([item.joint_jitter for item in metrics]),
            window=7,
        )
        blended_signal = np.clip(
            0.44 * motion_signal + 0.22 * pose_signal + 0.20 * active_signal + 0.14 * (1.0 - balance_signal),
            0.0,
            1.0,
        )
        hold_signal = np.clip(
            0.38 * (1.0 - motion_signal)
            + 0.18 * (1.0 - pose_signal)
            + 0.26 * stability_signal
            + 0.18 * (1.0 - balance_signal),
            0.0,
            1.0,
        )
        low_threshold = float(np.percentile(blended_signal, 30 if rules.expected_tempo == "slow" else 35))
        high_threshold = float(np.percentile(blended_signal, 74 if rules.expected_tempo == "slow" else 70))
        hold_threshold = float(np.percentile(hold_signal, 72 if rules.requires_hold else 82))

        raw_labels: list[str] = []
        total = max(len(metrics) - 1, 1)
        for index, signal_value in enumerate(blended_signal):
            relative = index / total
            hold_ready = (
                rules.requires_hold
                and 0.22 <= relative <= 0.82
                and hold_signal[index] >= hold_threshold
                and signal_value <= max(low_threshold + 0.08, 0.58)
            )
            if hold_ready:
                raw_labels.append("定势")
            elif signal_value <= low_threshold:
                if relative < 0.16:
                    raw_labels.append("准备")
                elif relative > 0.84:
                    raw_labels.append("收势")
                else:
                    raw_labels.append("过渡")
            elif signal_value >= high_threshold:
                if relative < 0.36:
                    raw_labels.append("起势")
                elif relative > 0.82:
                    raw_labels.append("收势")
                else:
                    raw_labels.append("发力")
            else:
                raw_labels.append("过渡")

        labels = self._smooth_labels(raw_labels, window=7 if rules.expected_tempo == "slow" else 5)
        phases: list[dict] = []
        current_label = labels[0]
        start_idx = 0

        for index in range(1, len(labels)):
            if labels[index] != current_label:
                phases.append(
                    {
                        "name": current_label,
                        "start_sec": metrics[start_idx].timestamp_sec,
                        "end_sec": metrics[index - 1].timestamp_sec,
                    }
                )
                current_label = labels[index]
                start_idx = index

        phases.append(
            {
                "name": current_label,
                "start_sec": metrics[start_idx].timestamp_sec,
                "end_sec": metrics[-1].timestamp_sec,
                }
            )

        compact_phases = self._merge_short_phases(phases, min_duration=0.28)
        compact_phases = self._promote_hold_phase(compact_phases, metrics, rules, hold_signal)
        compact_phases = self._stabilize_phase_sequence(compact_phases, rules)
        compact_phases = self._ensure_complete_phase_sequence(compact_phases, metrics, rules, blended_signal, hold_signal)
        for phase in compact_phases:
            phase["duration_sec"] = max(0.0, float(phase["end_sec"] - phase["start_sec"]))
        return compact_phases[:8]

    def _ensure_complete_phase_sequence(
        self,
        phases: list[dict],
        metrics: list[FrameMetrics],
        rules: RuleSet,
        blended_signal: np.ndarray,
        hold_signal: np.ndarray,
    ) -> list[dict]:
        if not metrics:
            return phases

        duration = max(0.0, float(metrics[-1].timestamp_sec - metrics[0].timestamp_sec))
        unique_names = {str(phase.get("name") or "") for phase in phases}
        expected_middle = "定势" if rules.requires_hold else "发力"
        required_names = {"准备", "起势", "过渡", expected_middle, "收势"}
        if len(phases) >= 4 and len(unique_names & required_names) >= 4:
            return phases

        start_time = float(metrics[0].timestamp_sec)
        end_time = float(metrics[-1].timestamp_sec)
        if duration <= 0.0:
            return [{"name": "准备", "start_sec": start_time, "end_sec": end_time}]

        fallback = self._build_fallback_phases(metrics, rules, blended_signal, hold_signal)
        if len(phases) <= 2:
            return fallback

        existing_by_name = {str(phase.get("name") or ""): phase for phase in phases}
        completed: list[dict] = []
        for phase in fallback:
            name = str(phase.get("name") or "")
            existing = existing_by_name.get(name)
            if existing is not None:
                completed.append(existing.copy())
            else:
                completed.append(phase)
        completed = sorted(completed, key=lambda item: float(item.get("start_sec") or 0.0))
        return self._merge_overlapping_phase_bounds(completed, start_time, end_time)

    def _build_fallback_phases(
        self,
        metrics: list[FrameMetrics],
        rules: RuleSet,
        blended_signal: np.ndarray,
        hold_signal: np.ndarray,
    ) -> list[dict]:
        start_time = float(metrics[0].timestamp_sec)
        end_time = float(metrics[-1].timestamp_sec)
        duration = max(0.0, end_time - start_time)
        if duration < 1.2:
            return [{"name": "起势", "start_sec": start_time, "end_sec": end_time}]

        peak_index = int(np.argmax(blended_signal)) if len(blended_signal) else max(1, len(metrics) // 2)
        peak_time = float(metrics[min(peak_index, len(metrics) - 1)].timestamp_sec)
        hold_index = int(np.argmax(hold_signal)) if len(hold_signal) else max(1, len(metrics) // 2)
        hold_time = float(metrics[min(hold_index, len(metrics) - 1)].timestamp_sec)
        middle_time = hold_time if rules.requires_hold else peak_time
        middle_time = max(start_time + duration * 0.34, min(middle_time, start_time + duration * 0.72))

        prepare_end = start_time + duration * 0.14
        rise_end = max(prepare_end + duration * 0.18, min(middle_time - duration * 0.08, start_time + duration * 0.38))
        middle_end = max(middle_time + duration * 0.12, start_time + duration * 0.62)
        close_start = start_time + duration * 0.84
        if middle_end >= close_start:
            middle_end = start_time + duration * 0.74
        phases = [
            {"name": "准备", "start_sec": start_time, "end_sec": prepare_end},
            {"name": "起势", "start_sec": prepare_end, "end_sec": rise_end},
            {"name": "过渡", "start_sec": rise_end, "end_sec": middle_time},
            {"name": "定势" if rules.requires_hold else "发力", "start_sec": middle_time, "end_sec": middle_end},
            {"name": "收势", "start_sec": middle_end, "end_sec": end_time},
        ]
        return self._merge_overlapping_phase_bounds(phases, start_time, end_time)

    def _merge_overlapping_phase_bounds(self, phases: list[dict], start_time: float, end_time: float) -> list[dict]:
        if not phases:
            return []
        if end_time <= start_time:
            return [{"name": phases[0].get("name") or "准备", "start_sec": start_time, "end_sec": end_time}]
        normalized: list[dict] = []
        total = end_time - start_time
        boundaries = [start_time]
        for phase in phases[:-1]:
            raw_end = float(phase.get("end_sec") or boundaries[-1])
            boundaries.append(max(start_time, min(raw_end, end_time)))
        boundaries.append(end_time)

        min_span = max(0.04, total / max(len(phases), 1) * 0.12)
        for index in range(1, len(boundaries)):
            floor = boundaries[index - 1] + min_span
            remaining = len(boundaries) - index - 1
            ceiling = end_time - remaining * min_span
            boundaries[index] = max(floor, min(boundaries[index], ceiling))
        boundaries[-1] = end_time

        for index, phase in enumerate(phases):
            phase_start = boundaries[index]
            phase_end = boundaries[index + 1]
            if phase_end <= phase_start:
                continue
            normalized.append({"name": phase.get("name") or "过渡", "start_sec": phase_start, "end_sec": phase_end})
        return normalized

    def _stabilize_phase_sequence(self, phases: list[dict], rules: RuleSet) -> list[dict]:
        if len(phases) <= 2:
            return phases

        min_duration_map = {
            "准备": 0.8,
            "起势": 1.0 if rules.expected_tempo == "slow" else 0.6,
            "过渡": 0.7,
            "定势": 0.9 if rules.requires_hold else 0.0,
            "发力": 0.8,
            "收势": 0.8,
        }

        stabilized: list[dict] = []
        for phase in phases:
            current = phase.copy()
            duration = float(current["end_sec"] - current["start_sec"])
            min_duration = min_duration_map.get(current["name"], 0.45)
            if stabilized and duration < min_duration:
                prev = stabilized[-1]
                prev_duration = float(prev["end_sec"] - prev["start_sec"])
                if prev["name"] == current["name"]:
                    prev["end_sec"] = current["end_sec"]
                    continue
                if (
                    len(stabilized) >= 2
                    and stabilized[-2]["name"] == current["name"]
                    and prev_duration < min_duration_map.get(prev["name"], 0.45) * 1.1
                ):
                    stabilized[-2]["end_sec"] = current["end_sec"]
                    stabilized.pop()
                    continue
                prev["end_sec"] = current["end_sec"]
                continue
            stabilized.append(current)

        merged: list[dict] = []
        for phase in stabilized:
            if merged and merged[-1]["name"] == phase["name"]:
                merged[-1]["end_sec"] = phase["end_sec"]
            else:
                merged.append(phase)

        if len(merged) >= 3:
            tail_names = [item["name"] for item in merged[-3:]]
            if tail_names[-1] == "收势" and tail_names[-2] == "过渡":
                merged[-2]["name"] = "收势"
                merged[-2]["end_sec"] = merged[-1]["end_sec"]
                merged.pop()

        bridge_merged: list[dict] = []
        index = 0
        while index < len(merged):
            current = merged[index].copy()
            if index + 2 < len(merged):
                middle = merged[index + 1]
                next_phase = merged[index + 2]
                middle_duration = float(middle["end_sec"] - middle["start_sec"])
                if (
                    current["name"] == next_phase["name"]
                    and middle["name"] == "过渡"
                    and middle_duration <= (0.9 if rules.expected_tempo == "slow" else 0.6)
                ):
                    current["end_sec"] = next_phase["end_sec"]
                    bridge_merged.append(current)
                    index += 3
                    continue
                if (
                    current["name"] == "定势"
                    and middle["name"] == "过渡"
                    and next_phase["name"] == "收势"
                    and middle_duration <= 0.75
                ):
                    current["name"] = "收势"
                    current["end_sec"] = next_phase["end_sec"]
                    bridge_merged.append(current)
                    index += 3
                    continue
            bridge_merged.append(current)
            index += 1

        return bridge_merged

    def _promote_hold_phase(
        self,
        phases: list[dict],
        metrics: list[FrameMetrics],
        rules: RuleSet,
        hold_signal: np.ndarray,
    ) -> list[dict]:
        if not rules.requires_hold or self._has_hold_phase(phases, 25.0) or not len(metrics):
            return phases
        candidate_index = None
        candidate_score = -1.0
        timestamps = np.array([item.timestamp_sec for item in metrics], dtype=np.float32)
        for index, phase in enumerate(phases):
            if phase["name"] in {"准备", "起势", "收势"}:
                continue
            start = float(phase["start_sec"])
            end = float(phase["end_sec"])
            if end - start < 0.38:
                continue
            mask = (timestamps >= start) & (timestamps <= end)
            if not np.any(mask):
                continue
            score = float(np.mean(hold_signal[mask]))
            if score > candidate_score:
                candidate_score = score
                candidate_index = index
        if candidate_index is not None and candidate_score >= 0.56:
            phases[candidate_index]["name"] = "定势"
        return phases

    def _smooth_labels(self, labels: list[str], window: int = 5) -> list[str]:
        if len(labels) <= 2:
            return labels
        radius = max(1, window // 2)
        smoothed: list[str] = []
        for index in range(len(labels)):
            start = max(0, index - radius)
            end = min(len(labels), index + radius + 1)
            dominant = Counter(labels[start:end]).most_common(1)[0][0]
            smoothed.append(dominant)
        return smoothed

    def _merge_short_phases(self, phases: list[dict], min_duration: float) -> list[dict]:
        if len(phases) <= 1:
            return phases
        merged: list[dict] = [phases[0].copy()]
        for phase in phases[1:]:
            duration = phase["end_sec"] - phase["start_sec"]
            if duration < min_duration:
                merged[-1]["end_sec"] = phase["end_sec"]
                continue
            if merged[-1]["name"] == phase["name"]:
                merged[-1]["end_sec"] = phase["end_sec"]
            else:
                merged.append(phase.copy())
        return merged

    def _apply_phase_labels(self, metrics: list[FrameMetrics], phases: list[dict]) -> None:
        for item in metrics:
            for phase in phases:
                if phase["start_sec"] <= item.timestamp_sec <= phase["end_sec"]:
                    item.phase_label = phase["name"]
                    break

    def _has_hold_phase(self, phases: list[dict], fps: float) -> bool:
        min_duration = max(0.35, 8.0 / max(fps, 1.0))
        for phase in phases:
            if phase["name"] == "定势" and (phase["end_sec"] - phase["start_sec"]) >= min_duration:
                return True
        return False

    def _make_issue(
        self,
        code: str,
        severity: str,
        title: str,
        detail: str,
        suggestion: str,
        metrics: list[FrameMetrics],
        predicate=None,
        error_type: str = "",
        highlight_metric: str = "",
        expected_value: float | None = None,
        actual_value: float | None = None,
        overlay_label: str = "",
    ) -> Issue:
        rule = self.rules_catalog.get(code, {})
        time_range = self._locate_issue_range(metrics, predicate)
        return Issue(
            code=code,
            severity=severity,
            title=rule.get("title", title),
            detail=rule.get("detail_prefix", "") + detail,
            suggestion=rule.get("suggestion", suggestion),
            time_range=time_range,
            error_type=error_type,
            highlight_metric=highlight_metric,
            expected_value=expected_value,
            actual_value=actual_value,
            overlay_label=overlay_label,
        )

    def _locate_issue_range(self, metrics: list[FrameMetrics], predicate) -> tuple[float, float] | None:
        if not metrics:
            return None
        if predicate is None:
            start = metrics[max(0, len(metrics) // 4)].timestamp_sec
            end = metrics[min(len(metrics) - 1, len(metrics) * 3 // 4)].timestamp_sec
            return (start, end)

        longest_start = None
        longest_end = None
        current_start = None
        previous_time = None

        for item in metrics:
            matched = bool(predicate(item))
            if matched and current_start is None:
                current_start = item.timestamp_sec
            if not matched and current_start is not None:
                current_end = previous_time if previous_time is not None else item.timestamp_sec
                if longest_start is None or (current_end - current_start) > (longest_end - longest_start):
                    longest_start, longest_end = current_start, current_end
                current_start = None
            previous_time = item.timestamp_sec

        if current_start is not None:
            current_end = metrics[-1].timestamp_sec
            if longest_start is None or (current_end - current_start) > (longest_end - longest_start):
                longest_start, longest_end = current_start, current_end

        if longest_start is None or longest_end is None:
            return None
        return (longest_start, longest_end)

    def _normalize_values(self, values: list[float | None]) -> np.ndarray:
        array = np.array([np.nan if value is None else float(value) for value in values], dtype=np.float32)
        if np.all(np.isnan(array)):
            return np.zeros(len(values), dtype=np.float32)
        if np.any(np.isnan(array)):
            fill_value = float(np.nanmedian(array))
            array = np.where(np.isnan(array), fill_value, array)
        min_value = float(np.min(array))
        max_value = float(np.max(array))
        span = max(max_value - min_value, 1e-6)
        return (array - min_value) / span

    def _smooth_numeric_signal(self, values: np.ndarray, window: int = 5) -> np.ndarray:
        if len(values) <= 2 or window <= 1:
            return values
        if window % 2 == 0:
            window += 1
        radius = window // 2
        padded = np.pad(values, (radius, radius), mode="edge")
        kernel = np.ones(window, dtype=np.float32) / float(window)
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed[: len(values)]

    def _weighted_mean(self, values: np.ndarray, weights: np.ndarray) -> float:
        total_weight = float(np.sum(weights))
        if total_weight <= 1e-6:
            return float(np.mean(values)) if values.size else 0.0
        return float(np.sum(values * weights) / total_weight)

    def _weighted_std(self, values: np.ndarray, weights: np.ndarray, mean_value: float) -> float:
        total_weight = float(np.sum(weights))
        if total_weight <= 1e-6:
            return float(np.std(values)) if values.size else 0.0
        variance = float(np.sum(((values - mean_value) ** 2) * weights) / total_weight)
        return float(math.sqrt(max(variance, 0.0)))

    def _clamp_score(self, value: float) -> float:
        return float(max(0.0, min(100.0, value)))


def _normalize_output_artifact_names(result: AnalysisResult) -> None:
    output_dir = Path(result.output_dir)
    mapping = {
        "excel_report": "01_模板基础报告.xlsx" if result.analysis_mode == "template" else "01_测试基础报告.xlsx",
        "compact_data_json": "02_模板综合数据.json" if result.analysis_mode == "template" else "02_测试综合数据.json",
        "analysis_summary_txt": "03_模板结果摘要.txt" if result.analysis_mode == "template" else "03_测试结果摘要.txt",
        "action_guidance_txt": "04_模板动作提示与优化方向.txt" if result.analysis_mode == "template" else "04_测试动作提示与优化方向.txt",
        "template_baseline_json": "05_模板基线数据.json",
        "frame_metrics_csv": "06_模板逐帧关键点数据.csv" if result.analysis_mode == "template" else "06_测试逐帧关键点数据.csv",
        "raw_keypoints_csv": "07_原始关键点.csv",
        "smoothed_keypoints_csv": "08_平滑关键点.csv",
        "inhibition_metrics_xlsx": "09_科研指标明细.xlsx",
        "summary_metrics_json": "10_摘要指标.json",
        "analysis_report_md": "11_分析报告.md",
        "analysis_result_xlsx": "12_完整分析结果.xlsx",
        "analysis_summary_xlsx": "13_分析摘要.xlsx",
        "experiment_summary_xlsx": "14_科研实验总表.xlsx",
        "prepost_summary_xlsx": "15_前后测改善量总表.xlsx",
    }
    for key, filename in mapping.items():
        current = result.artifacts.get(key)
        if not current:
            continue
        result.artifacts[key] = str(output_dir / filename)


def _render_action_guidance_text_cn(self: ActionAnalyzer, result: AnalysisResult) -> str:
    lines = [
        "动作提示与优化方向",
        "",
        f"分析模式：{'模板数据生成' if result.analysis_mode == 'template' else '测试者动作评估'}",
        f"模板名称：{result.rules.template_name or '未匹配模板'}",
        f"动作类别：{result.rules.action_category}",
        f"评价标准：{result.summary.evaluator_level}",
        f"测试者水平：{result.summary.learner_level}",
        f"综合评分：{result.summary.total_score:.1f}",
        "",
    ]
    if result.analysis_mode == "template":
        lines.extend(
            [
                "模板提示：",
                "- 当前文件为标准模板数据生成结果。",
                "- 后续请在测试者动作评估中加载模板基线，进行逐段对比。",
                "",
                "模板阶段：",
            ]
        )
        if result.summary.phases:
            for phase in result.summary.phases:
                lines.append(
                    f"- {phase.get('name') or '未命名阶段'}：{phase.get('start_sec', 0):.1f}s - {phase.get('end_sec', 0):.1f}s"
                )
            lines.extend(["", "分阶段录入建议："])
            lines.extend(self._phase_guidance_lines(result))
        else:
            lines.append("- 未识别到清晰阶段。")
        return "\n".join(lines)

    lines.append("优先优化方向：")
    if result.summary.improvement_focus:
        lines.extend([f"- {item}" for item in result.summary.improvement_focus])
    else:
        lines.append("- 当前未发现明显问题，可继续提高动作完整度和模板相似度。")
    lines.extend(["", "问题与修改建议："])
    if result.issues:
        for index, issue in enumerate(result.issues, start=1):
            time_text = ""
            if issue.time_range:
                time_text = f"（{issue.time_range[0]:.1f}s - {issue.time_range[1]:.1f}s）"
            metric_text = ""
            if issue.highlight_metric:
                current = "未记录" if issue.actual_value is None else f"{issue.actual_value:.2f}"
                expected = "未设定" if issue.expected_value is None else f"{issue.expected_value:.2f}"
                metric_text = f"；指标 {issue.highlight_metric}：当前 {current}，目标 {expected}"
            lines.append(f"{index}. {issue.title}{time_text}")
            lines.append(f"   问题：{issue.detail}{metric_text}")
            lines.append(f"   修改：{issue.suggestion}")
            lines.append(f"   练习目标：{self._issue_practice_goal(issue, result.rules.action_category)}")
    else:
        lines.append("- 当前未发现明显问题。")
    lines.extend(["", "按阶段修改提示："])
    lines.extend(self._phase_guidance_lines(result))
    lines.extend(["", "综合建议："])
    lines.extend([f"- {item}" for item in result.summary.advice] or ["- 保持当前动作节奏和稳定性。"])
    return "\n".join(lines)


def _phase_guidance_for_name_cn(self: ActionAnalyzer, phase_name: str, rules: RuleSet) -> str:
    if phase_name == "准备":
        return "准备阶段：沉肩垂肘，呼吸调匀，立身中正，为起势蓄力。"
    if phase_name == "起势":
        return "起势阶段：动作缓起缓落，意领形随，保持脊柱拔伸与重心稳定。"
    if phase_name == "过渡":
        return "过渡阶段：衔接需圆活不断，避免抢节奏，做到上下相随、内外合一。"
    if phase_name == "定势":
        return "定势阶段：体会气沉丹田，微微沉肩坠肘，减少无效小动作。"
    if phase_name == "发力":
        return "发力阶段：在稳定中完成舒展，注意开合展开，不可僵硬突冲。"
    if phase_name == "收势":
        return "收势阶段：缓缓回落，神形合一，保持动作收束后的稳定与连贯。"
    return f"{rules.action_category or '动作'}阶段：保持中正、松沉、匀缓，注意细节控制。"


def _build_template_payload_cn(
    self: ActionAnalyzer,
    result: AnalysisResult,
    template_name: str | None = None,
    phase_snapshots: list[dict] | None = None,
) -> dict:
    summary = result.summary
    return {
        "模板名称": template_name or result.rules.template_name or "未命名模板",
        "来源视频": result.source,
        "分析模式": "template",
        "姿态识别已启用": result.used_pose_estimator,
        "姿态后端状态": self.pose_estimator.status.reason,
        "规则信息": {
            "原始描述": result.rules.raw_text,
            "节奏要求": result.rules.expected_tempo,
            "需要对称": result.rules.requires_symmetry,
            "需要躯干中正": result.rules.requires_upright_torso,
            "需要定势停顿": result.rules.requires_hold,
            "关注部位": result.rules.focus_body_parts,
            "关键词": result.rules.keywords,
            "阈值": result.rules.thresholds,
        },
        "模板摘要": {
            "总分": summary.total_score,
            "姿态标准度": summary.posture_score,
            "动作连续性": summary.continuity_score,
            "动作稳定性": summary.stability_score,
            "节奏控制": summary.rhythm_score,
            "动作完整性": summary.completeness_score,
            "平均运动强度": summary.avg_motion,
            "平均左右平衡差": summary.avg_left_right_balance,
            "平均躯干倾斜角度": summary.avg_torso_tilt,
            "平均双臂对称误差": summary.avg_arm_symmetry_error,
            "平均抬臂角度": summary.avg_arm_raise,
            "平均关键点覆盖率": summary.avg_keypoint_coverage,
            "平均姿态质量分": summary.avg_pose_quality_score,
            "平均衔接评分": self._average_transition_score(result.frame_metrics),
            "姿态质量等级": summary.pose_quality_level,
        },
        "模板阶段": summary.phases,
        "阶段快照": phase_snapshots or [],
        "模板建议": summary.advice,
        "模板问题": [
            {
                "问题编码": issue.code,
                "严重程度": issue.severity,
                "错误类型": issue.error_type,
                "问题标题": issue.title,
                "问题说明": issue.detail,
                "关键指标": issue.highlight_metric,
                "预期值": issue.expected_value,
                "实际值": issue.actual_value,
                "问题时间区间": list(issue.time_range) if issue.time_range else None,
            }
            for issue in result.issues
        ],
        "原始逐帧数据": [asdict(item) for item in result.frame_metrics],
    }


def _normalize_evaluator_level_cn(self: ActionAnalyzer, value: str | None) -> str:
    text = (value or "").strip()
    if "初" in text:
        return "初习者"
    if "进" in text or "基础" in text:
        return "进阶者"
    return "熟练者"


def _level_profile_cn(self: ActionAnalyzer, level: str) -> dict:
    return _EVALUATOR_LEVELS.get(self._normalize_evaluator_level(level), _EVALUATOR_LEVELS["熟练者"])


def _sequence_similarity_label_cn(self: ActionAnalyzer, score: float | None) -> str:
    if score is None:
        return "未计算"
    if score >= 85:
        return "高度接近模板"
    if score >= 70:
        return "基本接近模板"
    if score >= 55:
        return "存在中等偏差"
    return "与模板差异较大"


def _normalize_keypoint_positions_cn(self: ActionAnalyzer, positions: dict) -> dict[str, list[float]]:
    if not isinstance(positions, dict):
        return {}
    normalized: dict[str, list[float]] = {}
    chinese_alias = {
        "鼻尖": "nose",
        "左眼": "left_eye",
        "右眼": "right_eye",
        "左耳": "left_ear",
        "右耳": "right_ear",
        "左肩": "left_shoulder",
        "右肩": "right_shoulder",
        "左肘": "left_elbow",
        "右肘": "right_elbow",
        "左腕": "left_wrist",
        "右腕": "right_wrist",
        "左髋": "left_hip",
        "右髋": "right_hip",
        "左膝": "left_knee",
        "右膝": "right_knee",
        "左踝": "left_ankle",
        "右踝": "right_ankle",
    }
    for key, value in positions.items():
        mapped_key = chinese_alias.get(str(key), str(key))
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        normalized[mapped_key] = [float(value[0]), float(value[1])]
    return normalized


def _build_metric_comparison_cn(
    self: ActionAnalyzer,
    label: str,
    baseline_value: float | None,
    current_value: float | None,
    mode: str,
    note: str,
    unit: str = "",
) -> ComparisonMetric:
    delta_value = None
    status = "未知"
    if baseline_value is not None and current_value is not None:
        delta_value = current_value - baseline_value
        if mode in {"max", "balance", "coverage", "min"}:
            if mode == "coverage":
                status = "良好" if current_value >= baseline_value else "注意"
            elif mode == "min":
                status = "良好" if current_value >= baseline_value else "注意"
            else:
                status = "良好" if current_value <= baseline_value else "注意"
        elif mode == "hold":
            status = "良好" if current_value >= baseline_value else "注意"
        else:
            status = "良好" if abs(delta_value) <= max(0.15 * max(abs(baseline_value), 1.0), 0.18) else "注意"
    elif current_value is None:
        status = "缺失"
    return ComparisonMetric(
        label=label,
        baseline_value=baseline_value,
        current_value=current_value,
        delta_value=delta_value,
        unit=unit,
        status=status,
        note=note,
    )


ActionAnalyzer._render_action_guidance_text = _render_action_guidance_text_cn
ActionAnalyzer._phase_guidance_for_name = _phase_guidance_for_name_cn
ActionAnalyzer.build_template_payload = _build_template_payload_cn
ActionAnalyzer._normalize_evaluator_level = _normalize_evaluator_level_cn
ActionAnalyzer._level_profile = _level_profile_cn
ActionAnalyzer._sequence_similarity_label = _sequence_similarity_label_cn
ActionAnalyzer._normalize_keypoint_positions = _normalize_keypoint_positions_cn
ActionAnalyzer._build_metric_comparison = _build_metric_comparison_cn
