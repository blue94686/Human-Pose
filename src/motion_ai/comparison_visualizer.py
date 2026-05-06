"""对比可视化器 - 实现双视频、双骨架、热图等对比显示"""

from __future__ import annotations

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .models import DifferenceMetrics, PoseFrame


class ComparisonVisualizer:
    """对比可视化器"""

    # COCO关键点连接关系
    SKELETON = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
        [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7]
    ]

    def __init__(self):
        self.template_color = (16, 185, 129)  # 绿色 #10b981
        self.test_color = (59, 130, 246)  # 蓝色 #3b82f6
        self.diff_color = (239, 68, 68)  # 红色 #ef4444

    def draw_dual_video(
        self,
        template_frame: np.ndarray,
        test_frame: np.ndarray,
        template_pose: PoseFrame | None = None,
        test_pose: PoseFrame | None = None,
        template_info: dict | None = None,
        test_info: dict | None = None,
    ) -> np.ndarray:
        """
        绘制双视频对比

        Args:
            template_frame: 模板视频帧
            test_frame: 测试视频帧
            template_pose: 模板姿态数据
            test_pose: 测试姿态数据
            template_info: 模板信息（阶段、帧号等）
            test_info: 测试信息

        Returns:
            np.ndarray: 合成的对比图像
        """
        # 确保两帧尺寸一致
        h1, w1 = template_frame.shape[:2]
        h2, w2 = test_frame.shape[:2]
        target_h = max(h1, h2)
        target_w = max(w1, w2)

        # 调整尺寸
        template_resized = cv2.resize(template_frame, (target_w, target_h))
        test_resized = cv2.resize(test_frame, (target_w, target_h))

        # 绘制骨架
        if template_pose:
            template_resized = self._draw_skeleton(
                template_resized, template_pose, self.template_color
            )
        if test_pose:
            test_resized = self._draw_skeleton(
                test_resized, test_pose, self.test_color
            )

        # 添加标签
        if template_info:
            template_resized = self._add_info_overlay(
                template_resized, template_info, "模板动作", self.template_color
            )
        if test_info:
            test_resized = self._add_info_overlay(
                test_resized, test_info, "测试动作", self.test_color
            )

        # 水平拼接
        combined = np.hstack([template_resized, test_resized])

        # 添加中间分隔线
        mid_x = target_w
        cv2.line(combined, (mid_x, 0), (mid_x, target_h), (229, 231, 235), 2)

        return combined

    def draw_dual_skeleton(
        self,
        template_pose: PoseFrame,
        test_pose: PoseFrame,
        canvas_size: tuple[int, int] = (800, 600),
        highlight_differences: bool = True,
    ) -> np.ndarray:
        """
        绘制双骨架对比

        Args:
            template_pose: 模板姿态
            test_pose: 测试姿态
            canvas_size: 画布尺寸
            highlight_differences: 是否高亮差异点

        Returns:
            np.ndarray: 骨架对比图像
        """
        w, h = canvas_size
        canvas = np.ones((h, w * 2, 3), dtype=np.uint8) * 250

        # 绘制模板骨架（左侧）
        template_canvas = canvas[:, :w].copy()
        template_canvas = self._draw_skeleton_on_canvas(
            template_canvas, template_pose, self.template_color, "标准骨架"
        )
        canvas[:, :w] = template_canvas

        # 绘制测试骨架（右侧）
        test_canvas = canvas[:, w:].copy()
        test_canvas = self._draw_skeleton_on_canvas(
            test_canvas, test_pose, self.test_color, "测试骨架"
        )
        canvas[:, w:] = test_canvas

        # 高亮差异点
        if highlight_differences:
            canvas = self._highlight_keypoint_differences(
                canvas, template_pose, test_pose, w
            )

        # 添加中间分隔线
        cv2.line(canvas, (w, 0), (w, h), (229, 231, 235), 2)

        return canvas

    def draw_heatmap(
        self,
        template_pose: PoseFrame,
        test_pose: PoseFrame,
        differences: DifferenceMetrics,
        canvas_size: tuple[int, int] = (800, 600),
    ) -> np.ndarray:
        """
        绘制差异热图

        Args:
            template_pose: 模板姿态
            test_pose: 测试姿态
            differences: 差异指标
            canvas_size: 画布尺寸

        Returns:
            np.ndarray: 热图图像
        """
        w, h = canvas_size
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 250

        # 绘制标准骨架（绿色）
        canvas = self._draw_skeleton_on_canvas(
            canvas, template_pose, self.template_color, None
        )

        # 绘制测试骨架（蓝色）
        canvas = self._draw_skeleton_on_canvas(
            canvas, test_pose, self.test_color, None
        )

        # 绘制差异热力区域
        if template_pose.keypoints and test_pose.keypoints:
            for i, (t_kp, test_kp) in enumerate(
                zip(template_pose.keypoints, test_pose.keypoints)
            ):
                if len(t_kp) >= 2 and len(test_kp) >= 2:
                    # 计算距离
                    dist = np.sqrt(
                        (t_kp[0] - test_kp[0]) ** 2 + (t_kp[1] - test_kp[1]) ** 2
                    )

                    # 根据距离绘制热力圆
                    if dist > 5:  # 阈值
                        # 归一化距离到0-1
                        normalized_dist = min(dist / 100, 1.0)
                        # 红色强度
                        intensity = int(255 * normalized_dist)
                        color = (intensity, 0, 255 - intensity)

                        # 绘制半透明圆
                        overlay = canvas.copy()
                        cv2.circle(
                            overlay,
                            (int(test_kp[0]), int(test_kp[1])),
                            int(20 + dist / 5),
                            color,
                            -1,
                        )
                        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

        # 添加图例
        canvas = self._add_heatmap_legend(canvas)

        # 添加标题
        cv2.putText(
            canvas,
            "Difference Heatmap",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (31, 41, 55),
            2,
        )

        return canvas

    def plot_time_series(
        self,
        template_metrics: list[dict],
        test_metrics: list[dict],
        metric_names: list[str] | None = None,
    ) -> Figure:
        """
        绘制时序曲线对比

        Args:
            template_metrics: 模板指标序列
            test_metrics: 测试指标序列
            metric_names: 要绘制的指标名称列表

        Returns:
            Figure: matplotlib图表对象
        """
        if metric_names is None:
            metric_names = [
                "torso_tilt_deg",
                "arm_symmetry_error",
                "motion_mean",
                "left_right_balance",
            ]

        fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 3 * len(metric_names)))
        if len(metric_names) == 1:
            axes = [axes]

        metric_labels = {
            "torso_tilt_deg": "躯干倾斜角度 (°)",
            "arm_symmetry_error": "双臂对称误差 (°)",
            "motion_mean": "运动强度",
            "left_right_balance": "左右平衡差",
        }

        for ax, metric_name in zip(axes, metric_names):
            # 提取数据
            template_values = [
                m.get(metric_name, 0) for m in template_metrics if m.get(metric_name) is not None
            ]
            test_values = [
                m.get(metric_name, 0) for m in test_metrics if m.get(metric_name) is not None
            ]

            # 绘制曲线
            if template_values:
                ax.plot(
                    template_values,
                    label="标准动作",
                    color="#10b981",
                    linewidth=2,
                    alpha=0.8,
                )
            if test_values:
                ax.plot(
                    test_values,
                    label="测试动作",
                    color="#3b82f6",
                    linewidth=2,
                    alpha=0.8,
                )

            ax.set_ylabel(metric_labels.get(metric_name, metric_name))
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("帧号")
        plt.tight_layout()

        return fig

    def _draw_skeleton(
        self, frame: np.ndarray, pose: PoseFrame, color: tuple[int, int, int]
    ) -> np.ndarray:
        """在视频帧上绘制骨架"""
        if not pose.keypoints:
            return frame

        # 绘制骨骼连接
        for connection in self.SKELETON:
            pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1
            if pt1_idx < len(pose.keypoints) and pt2_idx < len(pose.keypoints):
                pt1 = pose.keypoints[pt1_idx]
                pt2 = pose.keypoints[pt2_idx]
                if len(pt1) >= 2 and len(pt2) >= 2:
                    cv2.line(
                        frame,
                        (int(pt1[0]), int(pt1[1])),
                        (int(pt2[0]), int(pt2[1])),
                        color,
                        2,
                    )

        # 绘制关键点
        for kp in pose.keypoints:
            if len(kp) >= 2:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, color, -1)

        return frame

    def _draw_skeleton_on_canvas(
        self,
        canvas: np.ndarray,
        pose: PoseFrame,
        color: tuple[int, int, int],
        title: str | None,
    ) -> np.ndarray:
        """在画布上绘制骨架"""
        if not pose.keypoints:
            return canvas

        h, w = canvas.shape[:2]

        # 归一化关键点到画布尺寸
        keypoints_normalized = []
        for kp in pose.keypoints:
            if len(kp) >= 2:
                # 假设原始坐标在0-1范围内，缩放到画布
                x = int(kp[0] * w * 0.8 + w * 0.1)
                y = int(kp[1] * h * 0.8 + h * 0.1)
                keypoints_normalized.append([x, y])
            else:
                keypoints_normalized.append([])

        # 绘制骨骼
        for connection in self.SKELETON:
            pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1
            if pt1_idx < len(keypoints_normalized) and pt2_idx < len(
                keypoints_normalized
            ):
                pt1 = keypoints_normalized[pt1_idx]
                pt2 = keypoints_normalized[pt2_idx]
                if len(pt1) >= 2 and len(pt2) >= 2:
                    cv2.line(canvas, tuple(pt1), tuple(pt2), color, 3)

        # 绘制关键点
        for kp in keypoints_normalized:
            if len(kp) >= 2:
                cv2.circle(canvas, tuple(kp), 6, color, -1)
                cv2.circle(canvas, tuple(kp), 6, (255, 255, 255), 1)

        # 添加标题
        if title:
            cv2.putText(
                canvas,
                title,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (31, 41, 55),
                2,
            )

        return canvas

    def _add_info_overlay(
        self,
        frame: np.ndarray,
        info: dict,
        title: str,
        color: tuple[int, int, int],
    ) -> np.ndarray:
        """添加信息叠加层"""
        h, w = frame.shape[:2]

        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 100), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 标题
        cv2.putText(
            frame, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

        # 信息
        y_offset = 70
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(
                frame,
                text,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (31, 41, 55),
                1,
            )
            y_offset += 25

        return frame

    def _highlight_keypoint_differences(
        self,
        canvas: np.ndarray,
        template_pose: PoseFrame,
        test_pose: PoseFrame,
        split_x: int,
    ) -> np.ndarray:
        """高亮关键点差异"""
        if not template_pose.keypoints or not test_pose.keypoints:
            return canvas

        h = canvas.shape[0]

        for i, (t_kp, test_kp) in enumerate(
            zip(template_pose.keypoints, test_pose.keypoints)
        ):
            if len(t_kp) >= 2 and len(test_kp) >= 2:
                # 计算距离
                dist = np.sqrt((t_kp[0] - test_kp[0]) ** 2 + (t_kp[1] - test_kp[1]) ** 2)

                # 如果差异较大，标红
                if dist > 20:
                    # 左侧标记
                    cv2.circle(
                        canvas,
                        (int(t_kp[0]), int(t_kp[1])),
                        8,
                        self.diff_color,
                        2,
                    )
                    # 右侧标记
                    cv2.circle(
                        canvas,
                        (int(test_kp[0]) + split_x, int(test_kp[1])),
                        8,
                        self.diff_color,
                        2,
                    )

        return canvas

    def _add_heatmap_legend(self, canvas: np.ndarray) -> np.ndarray:
        """添加热图图例"""
        h, w = canvas.shape[:2]

        # 图例位置
        legend_x = w - 200
        legend_y = h - 100

        # 绘制图例背景
        cv2.rectangle(
            canvas,
            (legend_x - 10, legend_y - 10),
            (legend_x + 180, legend_y + 80),
            (255, 255, 255),
            -1,
        )
        cv2.rectangle(
            canvas,
            (legend_x - 10, legend_y - 10),
            (legend_x + 180, legend_y + 80),
            (200, 200, 200),
            1,
        )

        # 图例标题
        cv2.putText(
            canvas,
            "Deviation",
            (legend_x, legend_y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (31, 41, 55),
            1,
        )

        # 颜色条
        levels = [
            ("0-5cm", (100, 200, 100)),
            ("5-10cm", (255, 200, 0)),
            ("10cm+", (255, 100, 100)),
        ]

        y_offset = legend_y + 30
        for label, color in levels:
            cv2.circle(canvas, (legend_x + 10, y_offset), 8, color, -1)
            cv2.putText(
                canvas,
                label,
                (legend_x + 30, y_offset + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (31, 41, 55),
                1,
            )
            y_offset += 20

        return canvas
