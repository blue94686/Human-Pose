"""
增强的姿态跟踪和平滑算法模块

提供以下增强功能：
1. 卡尔曼滤波平滑
2. 增强的IOU跟踪
3. 关键点距离跟踪
4. 运动预测
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .models import PoseFrame


class KalmanFilter:
    """卡尔曼滤波器用于关键点平滑"""

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        """
        初始化卡尔曼滤波器

        Args:
            process_noise: 过程噪声（运动模型的不确定性）
            measurement_noise: 测量噪声（观测的不确定性）
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        更新滤波器状态

        Args:
            measurement: 当前测量值（关键点坐标）

        Returns:
            平滑后的坐标
        """
        if self.state is None:
            # 初始化
            self.state = measurement.copy()
            self.covariance = np.eye(len(measurement)) * 0.1
            return measurement

        # 预测步骤
        predicted_state = self.state
        predicted_covariance = self.covariance + self.process_noise

        # 更新步骤
        kalman_gain = predicted_covariance / (predicted_covariance + self.measurement_noise)
        self.state = predicted_state + kalman_gain * (measurement - predicted_state)
        self.covariance = (1 - kalman_gain) * predicted_covariance

        return self.state.copy()

    def reset(self):
        """重置滤波器状态"""
        self.state = None
        self.covariance = None


class EnhancedPoseTracker:
    """增强的姿态跟踪器"""

    def __init__(self, history_size: int = 10):
        """
        初始化跟踪器

        Args:
            history_size: 历史帧数量
        """
        self.history: deque[PoseFrame] = deque(maxlen=history_size)
        self.kalman_filters: dict[int, KalmanFilter] = {}
        self.lost_count = 0
        self.max_lost_frames = 5

    def compute_iou(self, bbox1: list[float], bbox2: list[float]) -> float:
        """
        计算两个边界框的IOU

        Args:
            bbox1: 第一个边界框 [x1, y1, x2, y2]
            bbox2: 第二个边界框 [x1, y1, x2, y2]

        Returns:
            IOU值 (0-1)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)

    def compute_keypoint_distance(self, pose1: PoseFrame, pose2: PoseFrame) -> float:
        """
        计算两个姿态的关键点平均距离

        Args:
            pose1: 第一个姿态
            pose2: 第二个姿态

        Returns:
            平均距离
        """
        if len(pose1.keypoints) != len(pose2.keypoints):
            return float('inf')

        distances = []
        for kp1, kp2 in zip(pose1.keypoints, pose2.keypoints):
            dist = np.linalg.norm(np.array(kp1) - np.array(kp2))
            distances.append(dist)

        return np.mean(distances)

    def predict_next_position(self, pose: PoseFrame) -> Optional[PoseFrame]:
        """
        基于历史轨迹预测下一帧位置

        Args:
            pose: 当前姿态

        Returns:
            预测的姿态
        """
        if len(self.history) < 2:
            return None

        # 使用最近两帧计算速度
        prev_pose = self.history[-1]
        prev_prev_pose = self.history[-2]

        # 计算关键点速度
        velocities = []
        for i in range(len(prev_pose.keypoints)):
            v = np.array(prev_pose.keypoints[i]) - np.array(prev_prev_pose.keypoints[i])
            velocities.append(v)

        # 预测下一帧位置
        predicted_keypoints = []
        for i, kp in enumerate(prev_pose.keypoints):
            predicted = np.array(kp) + velocities[i] * 0.8  # 衰减系数
            predicted_keypoints.append(predicted.tolist())

        # 预测边界框
        bbox_velocity = np.array(prev_pose.bbox) - np.array(prev_prev_pose.bbox)
        predicted_bbox = (np.array(prev_pose.bbox) + bbox_velocity * 0.8).tolist()

        return PoseFrame(
            keypoints=predicted_keypoints,
            bbox=predicted_bbox,
            confidence=prev_pose.confidence,
            keypoint_confidences=prev_pose.keypoint_confidences,
        )

    def select_best_detection(
        self,
        detections: list[PoseFrame],
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, float]:
        """
        从多个检测结果中选择最佳目标

        Args:
            detections: 检测结果列表
            frame_width: 帧宽度
            frame_height: 帧高度

        Returns:
            (最佳索引, 跟踪分数)
        """
        if not detections:
            return -1, 0.0

        if len(self.history) == 0:
            # 第一帧，选择最中心、最大的目标
            frame_center = np.array([frame_width / 2.0, frame_height / 2.0])
            best_idx = 0
            best_score = float('-inf')

            for idx, det in enumerate(detections):
                x1, y1, x2, y2 = det.bbox
                area = (x2 - x1) * (y2 - y1)
                center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
                center_dist = np.linalg.norm(center - frame_center)

                score = area * det.confidence - center_dist * 0.5
                if score > best_score:
                    best_score = score
                    best_idx = idx

            return best_idx, 1.0

        # 有历史帧，使用增强跟踪
        previous_pose = self.history[-1]
        predicted_pose = self.predict_next_position(previous_pose)

        best_idx = 0
        best_score = float('-inf')

        for idx, det in enumerate(detections):
            # 1. IOU分数
            iou_score = self.compute_iou(det.bbox, previous_pose.bbox)

            # 2. 关键点距离分数
            kp_dist = self.compute_keypoint_distance(det, previous_pose)
            body_height = self._estimate_body_height(previous_pose)
            kp_score = max(0, 1.0 - kp_dist / (body_height + 1e-6))

            # 3. 运动预测分数
            motion_score = 0.0
            if predicted_pose:
                pred_dist = self.compute_keypoint_distance(det, predicted_pose)
                motion_score = max(0, 1.0 - pred_dist / (body_height + 1e-6))

            # 4. 置信度分数
            conf_score = det.confidence

            # 综合评分
            composite_score = (
                0.35 * iou_score +
                0.30 * kp_score +
                0.20 * motion_score +
                0.15 * conf_score
            )

            if composite_score > best_score:
                best_score = composite_score
                best_idx = idx

        return best_idx, best_score

    def smooth_pose_with_kalman(self, pose: PoseFrame) -> PoseFrame:
        """
        使用卡尔曼滤波平滑姿态

        Args:
            pose: 输入姿态

        Returns:
            平滑后的姿态
        """
        smoothed_keypoints = []

        for idx, kp in enumerate(pose.keypoints):
            # 为每个关键点创建独立的卡尔曼滤波器
            if idx not in self.kalman_filters:
                self.kalman_filters[idx] = KalmanFilter(
                    process_noise=0.01,
                    measurement_noise=0.1
                )

            # 应用滤波
            measurement = np.array(kp, dtype=np.float32)
            smoothed = self.kalman_filters[idx].update(measurement)
            smoothed_keypoints.append(smoothed.tolist())

        return PoseFrame(
            keypoints=smoothed_keypoints,
            bbox=pose.bbox,
            confidence=pose.confidence,
            keypoint_confidences=pose.keypoint_confidences,
            tracking_score=pose.tracking_score,
            tracking_mode=pose.tracking_mode,
        )

    def update(self, pose: Optional[PoseFrame]) -> Optional[PoseFrame]:
        """
        更新跟踪器状态

        Args:
            pose: 当前帧的姿态

        Returns:
            平滑后的姿态
        """
        if pose is None:
            self.lost_count += 1
            if self.lost_count > self.max_lost_frames:
                # 丢失太久，重置
                self.reset()
            return None

        # 重置丢失计数
        self.lost_count = 0

        # 应用卡尔曼滤波
        smoothed_pose = self.smooth_pose_with_kalman(pose)

        # 添加到历史
        self.history.append(smoothed_pose)

        return smoothed_pose

    def reset(self):
        """重置跟踪器"""
        self.history.clear()
        self.kalman_filters.clear()
        self.lost_count = 0

    def _estimate_body_height(self, pose: PoseFrame) -> float:
        """估算身体高度"""
        keypoints = np.array(pose.keypoints)

        # 使用鼻子到脚踝的距离作为身体高度
        nose_idx = 0
        left_ankle_idx = 15
        right_ankle_idx = 16

        if len(keypoints) > max(nose_idx, left_ankle_idx, right_ankle_idx):
            nose = keypoints[nose_idx]
            left_ankle = keypoints[left_ankle_idx]
            right_ankle = keypoints[right_ankle_idx]

            height1 = np.linalg.norm(nose - left_ankle)
            height2 = np.linalg.norm(nose - right_ankle)

            return max(height1, height2)

        # 备用方案：使用边界框高度
        return pose.bbox[3] - pose.bbox[1]


def apply_enhanced_bone_constraints(
    pose: PoseFrame,
    reference_pose: Optional[PoseFrame] = None,
) -> PoseFrame:
    """
    应用增强的骨骼约束

    Args:
        pose: 输入姿态
        reference_pose: 参考姿态（可选）

    Returns:
        约束后的姿态
    """
    from .pose import COCO_KEYPOINTS

    keypoints = np.array(pose.keypoints, dtype=np.float32)

    # 骨骼长度比例（相对于身体高度）
    bone_length_ratios = {
        ('left_shoulder', 'left_elbow'): (0.25, 0.35),
        ('left_elbow', 'left_wrist'): (0.20, 0.30),
        ('right_shoulder', 'right_elbow'): (0.25, 0.35),
        ('right_elbow', 'right_wrist'): (0.20, 0.30),
        ('left_hip', 'left_knee'): (0.35, 0.45),
        ('left_knee', 'left_ankle'): (0.35, 0.45),
        ('right_hip', 'right_knee'): (0.35, 0.45),
        ('right_knee', 'right_ankle'): (0.35, 0.45),
    }

    # 估算身体高度
    body_height = pose.bbox[3] - pose.bbox[1]

    # 应用骨骼长度约束
    for (start_name, end_name), (min_ratio, max_ratio) in bone_length_ratios.items():
        try:
            start_idx = COCO_KEYPOINTS.index(start_name)
            end_idx = COCO_KEYPOINTS.index(end_name)
        except ValueError:
            continue

        if start_idx >= len(keypoints) or end_idx >= len(keypoints):
            continue

        start_point = keypoints[start_idx]
        end_point = keypoints[end_idx]

        # 计算当前长度
        current_length = np.linalg.norm(end_point - start_point)

        # 期望长度范围
        expected_min = body_height * min_ratio
        expected_max = body_height * max_ratio

        # 如果超出范围，修正
        if current_length < expected_min:
            # 太短，拉长
            direction = (end_point - start_point) / (current_length + 1e-6)
            keypoints[end_idx] = start_point + direction * expected_min
        elif current_length > expected_max:
            # 太长，缩短
            direction = (end_point - start_point) / (current_length + 1e-6)
            keypoints[end_idx] = start_point + direction * expected_max

    # 应用对称性约束
    try:
        left_shoulder_idx = COCO_KEYPOINTS.index('left_shoulder')
        right_shoulder_idx = COCO_KEYPOINTS.index('right_shoulder')

        if left_shoulder_idx < len(keypoints) and right_shoulder_idx < len(keypoints):
            center = (keypoints[left_shoulder_idx] + keypoints[right_shoulder_idx]) / 2

            # 确保左右对称部位相对中心的距离相近
            symmetric_pairs = [
                ('left_hip', 'right_hip'),
                ('left_knee', 'right_knee'),
                ('left_ankle', 'right_ankle'),
            ]

            for left_name, right_name in symmetric_pairs:
                try:
                    left_idx = COCO_KEYPOINTS.index(left_name)
                    right_idx = COCO_KEYPOINTS.index(right_name)
                except ValueError:
                    continue

                if left_idx >= len(keypoints) or right_idx >= len(keypoints):
                    continue

                left_dist = np.linalg.norm(keypoints[left_idx] - center)
                right_dist = np.linalg.norm(keypoints[right_idx] - center)

                # 如果差异过大（超过10%身体高度），调整
                if abs(left_dist - right_dist) > body_height * 0.1:
                    avg_dist = (left_dist + right_dist) / 2

                    left_dir = (keypoints[left_idx] - center) / (left_dist + 1e-6)
                    right_dir = (keypoints[right_idx] - center) / (right_dist + 1e-6)

                    keypoints[left_idx] = center + left_dir * avg_dist
                    keypoints[right_idx] = center + right_dir * avg_dist
    except Exception:
        pass

    return PoseFrame(
        keypoints=keypoints.tolist(),
        bbox=pose.bbox,
        confidence=pose.confidence,
        keypoint_confidences=pose.keypoint_confidences,
        tracking_score=pose.tracking_score,
        tracking_mode=pose.tracking_mode,
    )
