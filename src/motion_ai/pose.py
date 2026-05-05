from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import DEFAULT_WEIGHTS
from .models import PoseFrame

try:
    from ultralytics import YOLO
    _ULTRALYTICS_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    YOLO = None
    _ULTRALYTICS_IMPORT_ERROR = str(exc).strip()


COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

SKELETON_EDGES = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]


@dataclass
class PoseBackendStatus:
    available: bool
    reason: str
    model_path: str = ""


class PoseEstimator:
    def __init__(
        self,
        weights_path: str | Path | None = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.weights_path = Path(weights_path or DEFAULT_WEIGHTS).expanduser()
        self.confidence_threshold = float(confidence_threshold)
        self.model = None
        self.status = PoseBackendStatus(
            available=False,
            reason="当前为光流兜底模式，无骨架可视化",
            model_path=str(self.weights_path),
        )
        self._load_model()

    @property
    def available(self) -> bool:
        return bool(self.status.available and self.model is not None)

    def _load_model(self) -> None:
        if YOLO is None:
            import_reason = _ULTRALYTICS_IMPORT_ERROR or "未知导入错误"
            self.status = PoseBackendStatus(
                available=False,
                reason=f"Ultralytics 导入失败：{import_reason}，当前为光流兜底模式，无骨架可视化",
                model_path=str(self.weights_path),
            )
            return
        if not self.weights_path.exists():
            self.status = PoseBackendStatus(
                available=False,
                reason=f"姿态权重文件不存在：{self.weights_path}",
                model_path=str(self.weights_path),
            )
            return
        try:
            self.model = YOLO(str(self.weights_path))
            self.status = PoseBackendStatus(
                available=True,
                reason=f"YOLOv8-Pose 已启用：{self.weights_path.name}",
                model_path=str(self.weights_path),
            )
        except Exception as exc:
            self.model = None
            self.status = PoseBackendStatus(
                available=False,
                reason=f"姿态模型加载失败：{exc}",
                model_path=str(self.weights_path),
            )

    def estimate(self, frame: np.ndarray) -> PoseFrame | None:
        if not self.available:
            return None
        try:
            results = self.model.predict(
                source=frame,
                conf=max(0.05, min(self.confidence_threshold, 0.95)),
                verbose=False,
                save=False,
                imgsz=640,
                max_det=5,
            )
        except Exception as exc:
            self.status = PoseBackendStatus(
                available=False,
                reason=f"姿态推理失败：{exc}",
                model_path=str(self.weights_path),
            )
            self.model = None
            return None
        if not results:
            return None
        return self._result_to_pose_frame(results[0], frame.shape)

    def estimate_with_tracking(self, frame: np.ndarray, previous_pose: PoseFrame | None = None) -> PoseFrame | None:
        if not self.available:
            return None
        try:
            results = self.model.predict(
                source=frame,
                conf=max(0.05, min(self.confidence_threshold, 0.95)),
                verbose=False,
                save=False,
                imgsz=640,
                max_det=5,
            )
        except Exception as exc:
            self.status = PoseBackendStatus(
                available=False,
                reason=f"姿态推理失败：{exc}",
                model_path=str(self.weights_path),
            )
            self.model = None
            return None
        if not results:
            return None
        pose = self._result_to_pose_frame(results[0], frame.shape, previous_pose=previous_pose)
        if pose is None:
            return None
        if previous_pose is not None:
            pose.tracking_mode = "detector+flow"
            pose.tracking_score = _tracking_match_score(previous_pose, pose)
            pose.track_id = previous_pose.track_id if previous_pose.track_id is not None else 1
        else:
            pose.tracking_mode = "detector"
            pose.tracking_score = 1.0
            pose.track_id = 1
        pose.tracked_keypoints = sum(
            1 for value in pose.keypoint_confidences if float(value) >= self.confidence_threshold
        )
        return pose

    def _result_to_pose_frame(
        self,
        result,
        frame_shape: tuple[int, ...],
        previous_pose: PoseFrame | None = None,
    ) -> PoseFrame | None:
        keypoints = getattr(result, "keypoints", None)
        boxes = getattr(result, "boxes", None)
        if keypoints is None or boxes is None or len(boxes) == 0:
            return None
        try:
            xy = keypoints.xy.cpu().numpy()
            conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None
            box_xyxy = boxes.xyxy.cpu().numpy()
            box_conf = boxes.conf.cpu().numpy() if boxes.conf is not None else None
        except Exception:
            return None
        if len(xy) == 0 or len(box_xyxy) == 0:
            return None

        candidates: list[PoseFrame] = []
        for person_idx in range(min(len(xy), len(box_xyxy), 5)):
            candidate = self._build_pose_candidate(
                person_idx=person_idx,
                xy=xy,
                conf=conf,
                box_xyxy=box_xyxy,
                box_conf=box_conf,
                frame_shape=frame_shape,
            )
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            return None
        if previous_pose is None:
            return max(candidates, key=lambda item: item.confidence)
        return max(candidates, key=lambda item: _tracking_match_score(previous_pose, item))

    def _build_pose_candidate(
        self,
        *,
        person_idx: int,
        xy: np.ndarray,
        conf: np.ndarray | None,
        box_xyxy: np.ndarray,
        box_conf: np.ndarray | None,
        frame_shape: tuple[int, ...],
    ) -> PoseFrame | None:
        points = xy[person_idx]
        point_conf = conf[person_idx] if conf is not None and len(conf) > person_idx else np.ones(len(points), dtype=np.float32)
        valid_pairs: list[list[float]] = []
        visible_count = 0
        frame_h, frame_w = frame_shape[:2]
        for idx in range(min(len(COCO_KEYPOINTS), len(points))):
            x = float(np.clip(points[idx][0], 0, frame_w - 1))
            y = float(np.clip(points[idx][1], 0, frame_h - 1))
            score = float(point_conf[idx]) if idx < len(point_conf) else 1.0
            if score < self.confidence_threshold:
                valid_pairs.append([float("nan"), float("nan")])
                continue
            visible_count += 1
            valid_pairs.append([x, y])
        if visible_count < 4:
            return None
        bbox_xyxy = [float(value) for value in box_xyxy[person_idx].tolist()]
        confidence = float(box_conf[person_idx]) if box_conf is not None and len(box_conf) > person_idx else float(np.mean(point_conf))
        return PoseFrame(
            keypoints=valid_pairs,
            bbox=bbox_xyxy,
            confidence=confidence,
            keypoint_confidences=[float(value) for value in point_conf[: len(valid_pairs)]],
            tracking_score=confidence,
            tracking_mode="detector",
            tracked_keypoints=visible_count,
        )


def _tracking_match_score(previous_pose: PoseFrame, current_pose: PoseFrame) -> float:
    prev_center = _pose_center(previous_pose)
    curr_center = _pose_center(current_pose)
    distance_score = 0.0
    if prev_center is not None and curr_center is not None:
        distance = float(np.linalg.norm(np.array(prev_center) - np.array(curr_center)))
        distance_score = 1.0 / (1.0 + distance / 120.0)
    iou_score = _bbox_iou(previous_pose.bbox, current_pose.bbox)
    confidence_score = float(current_pose.confidence)
    return 0.45 * distance_score + 0.35 * iou_score + 0.20 * confidence_score


def _pose_center(pose: PoseFrame) -> tuple[float, float] | None:
    finite_points = [
        point for point in pose.keypoints
        if isinstance(point, (list, tuple)) and len(point) >= 2 and np.isfinite(point[0]) and np.isfinite(point[1])
    ]
    if not finite_points:
        return None
    xs = [float(point[0]) for point in finite_points]
    ys = [float(point[1]) for point in finite_points]
    return (float(np.mean(xs)), float(np.mean(ys)))


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union_area = max(area_a + area_b - inter_area, 1e-6)
    return float(inter_area / union_area)


def _clone_pose_frame(pose: PoseFrame) -> PoseFrame:
    return PoseFrame(
        keypoints=[list(point) for point in pose.keypoints],
        bbox=list(pose.bbox),
        confidence=float(pose.confidence),
        keypoint_confidences=[float(value) for value in pose.keypoint_confidences],
        tracking_score=pose.tracking_score,
        tracking_mode=pose.tracking_mode,
        tracked_keypoints=int(pose.tracked_keypoints),
        track_id=pose.track_id,
    )


def _is_valid_keypoint(point: list[float] | tuple[float, ...]) -> bool:
    return len(point) >= 2 and np.isfinite(point[0]) and np.isfinite(point[1])


def _count_visible_keypoints(points: list[list[float]]) -> int:
    return sum(1 for point in points if _is_valid_keypoint(point))


def pose_mean_displacement(current_pose: PoseFrame | None, previous_pose: PoseFrame | None) -> float:
    if current_pose is None or previous_pose is None:
        return 0.0
    distances: list[float] = []
    for curr, prev in zip(current_pose.keypoints, previous_pose.keypoints):
        if len(curr) < 2 or len(prev) < 2:
            continue
        if not (np.isfinite(curr[0]) and np.isfinite(curr[1]) and np.isfinite(prev[0]) and np.isfinite(prev[1])):
            continue
        distances.append(float(np.linalg.norm(np.array(curr[:2]) - np.array(prev[:2]))))
    return float(np.mean(distances)) if distances else 0.0


def pose_joint_jitter(current_pose: PoseFrame | None, previous_pose: PoseFrame | None) -> float:
    return pose_mean_displacement(current_pose, previous_pose)


def smooth_pose(pose: PoseFrame | None, previous_pose: PoseFrame | None = None, alpha: float = 0.6) -> PoseFrame | None:
    if pose is None or previous_pose is None:
        return pose
    smoothed = []
    for curr, prev in zip(pose.keypoints, previous_pose.keypoints):
        if len(curr) < 2 or len(prev) < 2:
            smoothed.append(curr)
            continue
        if not (np.isfinite(curr[0]) and np.isfinite(curr[1]) and np.isfinite(prev[0]) and np.isfinite(prev[1])):
            smoothed.append(curr)
            continue
        smoothed.append([
            float(alpha * curr[0] + (1.0 - alpha) * prev[0]),
            float(alpha * curr[1] + (1.0 - alpha) * prev[1]),
        ])
    pose.keypoints = smoothed
    return pose


def temporal_smooth_pose(
    pose: PoseFrame | None,
    previous_pose: PoseFrame | None = None,
    earlier_pose: PoseFrame | None = None,
    alpha: float = 0.64,
) -> PoseFrame | None:
    pose = smooth_pose(pose, previous_pose, alpha=alpha)
    if pose is None or earlier_pose is None:
        return pose
    return smooth_pose(pose, earlier_pose, alpha=min(0.85, alpha + 0.12))


def apply_pose_constraints(pose: PoseFrame | None, previous_pose: PoseFrame | None = None) -> PoseFrame | None:
    if pose is None:
        return None
    frame = smooth_pose(pose, previous_pose, alpha=0.78)
    return frame


def recover_pose_with_flow(
    pose: PoseFrame | None,
    previous_pose: PoseFrame | None,
    previous_frame: np.ndarray | None = None,
    current_frame: np.ndarray | None = None,
    lost_frame_count: int = 0,
) -> PoseFrame | None:
    if pose is None:
        if previous_pose is None:
            return None
        if previous_frame is None or current_frame is None or lost_frame_count > 1:
            return None
        fallback = _clone_pose_frame(previous_pose)
        fallback.tracking_mode = "flow_fallback"
        fallback.tracking_score = max(float(fallback.tracking_score or fallback.confidence or 0.0) * 0.92, 0.0)
        fallback.tracked_keypoints = _count_visible_keypoints(fallback.keypoints)
        return fallback

    _ = previous_frame, current_frame, lost_frame_count
    if pose is None:
        return None

    recovered = _clone_pose_frame(pose)
    if previous_pose is None:
        recovered.tracked_keypoints = max(recovered.tracked_keypoints, _count_visible_keypoints(recovered.keypoints))
        recovered.tracking_mode = recovered.tracking_mode or "detector"
        return recovered

    merged_keypoints: list[list[float]] = []
    keypoint_confidences = list(recovered.keypoint_confidences)
    filled_keypoints = 0
    max_points = max(len(recovered.keypoints), len(previous_pose.keypoints))
    for idx in range(max_points):
        current_point = recovered.keypoints[idx] if idx < len(recovered.keypoints) else [float("nan"), float("nan")]
        previous_point = previous_pose.keypoints[idx] if idx < len(previous_pose.keypoints) else [float("nan"), float("nan")]
        if _is_valid_keypoint(current_point):
            merged_keypoints.append(list(current_point))
            continue
        if _is_valid_keypoint(previous_point):
            merged_keypoints.append(list(previous_point))
            filled_keypoints += 1
            previous_confidence = (
                float(previous_pose.keypoint_confidences[idx])
                if idx < len(previous_pose.keypoint_confidences)
                else 0.0
            )
            if idx < len(keypoint_confidences):
                keypoint_confidences[idx] = max(float(keypoint_confidences[idx]), previous_confidence * 0.85)
            else:
                keypoint_confidences.append(previous_confidence * 0.85)
            continue
        merged_keypoints.append(list(current_point))
        if idx >= len(keypoint_confidences):
            keypoint_confidences.append(0.0)

    recovered.keypoints = merged_keypoints
    recovered.keypoint_confidences = keypoint_confidences[: len(merged_keypoints)]
    recovered.tracked_keypoints = _count_visible_keypoints(merged_keypoints)
    recovered.tracking_mode = "detector+flow" if filled_keypoints > 0 else (recovered.tracking_mode or "detector")
    if recovered.track_id is None:
        recovered.track_id = previous_pose.track_id
    if recovered.tracking_score is None and previous_pose.tracking_score is not None:
        recovered.tracking_score = previous_pose.tracking_score
    return recovered


def draw_pose(frame: np.ndarray, pose: PoseFrame | None) -> np.ndarray:
    if pose is None:
        return frame
    canvas = frame.copy()
    for start_idx, end_idx in SKELETON_EDGES:
        if start_idx >= len(pose.keypoints) or end_idx >= len(pose.keypoints):
            continue
        start = pose.keypoints[start_idx]
        end = pose.keypoints[end_idx]
        if len(start) < 2 or len(end) < 2:
            continue
        if not (np.isfinite(start[0]) and np.isfinite(start[1]) and np.isfinite(end[0]) and np.isfinite(end[1])):
            continue
        cv2.line(
            canvas,
            (int(start[0]), int(start[1])),
            (int(end[0]), int(end[1])),
            (0, 220, 140),
            2,
            cv2.LINE_AA,
        )
    for point in pose.keypoints:
        if len(point) < 2:
            continue
        if not (np.isfinite(point[0]) and np.isfinite(point[1])):
            continue
        cv2.circle(canvas, (int(point[0]), int(point[1])), 3, (30, 144, 255), -1, cv2.LINE_AA)
    return canvas
