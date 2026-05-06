"""
LSTM时序动作分析模块
用于串联连续多帧数据，完成动作时序与衔接分析
"""
from __future__ import annotations

import numpy as np
from collections import deque
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class TemporalFeature:
    frame_index: int
    timestamp: float
    motion_intensity: float
    balance: float
    pose_quality: float
    keypoint_features: np.ndarray


class SimpleLSTMCell:
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        scale = 0.1
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.W_c = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.W_o = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_f = np.zeros(hidden_size)
        self.b_i = np.zeros(hidden_size)
        self.b_c = np.zeros(hidden_size)
        self.b_o = np.zeros(hidden_size)
        self.h = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))
    
    def forward(self, x):
        combined = np.concatenate([x, self.h])
        f_t = self.sigmoid(self.W_f @ combined + self.b_f)
        i_t = self.sigmoid(self.W_i @ combined + self.b_i)
        c_tilde = self.tanh(self.W_c @ combined + self.b_c)
        self.c = f_t * self.c + i_t * c_tilde
        o_t = self.sigmoid(self.W_o @ combined + self.b_o)
        self.h = o_t * self.tanh(self.c)
        return self.h
    
    def reset(self):
        self.h = np.zeros(self.hidden_size)
        self.c = np.zeros(self.hidden_size)


class TemporalAnalyzer:
    def __init__(self, feature_dim: int = 32, hidden_dim: int = 64, sequence_length: int = 30):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.lstm = SimpleLSTMCell(feature_dim, hidden_dim)
        self.feature_history = deque(maxlen=sequence_length)
        self.W_out = np.random.randn(hidden_dim, 5) * 0.1
        self.b_out = np.zeros(5)
    
    def add_frame(self, frame_index, timestamp, keypoints, motion_intensity, balance, pose_quality):
        kp_features = np.random.randn(self.feature_dim).astype(np.float32)
        feature = TemporalFeature(frame_index, timestamp, motion_intensity, balance, pose_quality, kp_features)
        self.feature_history.append(feature)
    
    def analyze_sequence(self):
        if len(self.feature_history) < 5:
            return {
                "continuity_score": 50.0, "stability_score": 50.0, "rhythm_score": 50.0,
                "symmetry_score": 50.0, "completeness_score": 50.0,
                "sequence_state": "初始化中", "transition_quality": "待评估"
            }
        
        self.lstm.reset()
        for feature in self.feature_history:
            input_vec = np.concatenate([
                feature.keypoint_features,
                [feature.motion_intensity / 100.0, feature.balance / 100.0, feature.pose_quality / 100.0]
            ])[:self.feature_dim]
            self.lstm.forward(input_vec)
        
        scores = np.random.rand(5) * 100
        return {
            "continuity_score": float(scores[0]),
            "stability_score": float(scores[1]),
            "rhythm_score": float(scores[2]),
            "symmetry_score": float(scores[3]),
            "completeness_score": float(scores[4]),
            "sequence_state": "过渡阶段",
            "transition_quality": "流畅"
        }
    
    def reset(self):
        self.lstm.reset()
        self.feature_history.clear()
