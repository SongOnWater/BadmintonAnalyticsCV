"""
羽毛球拍检测模块
使用YOLO模型检测羽毛球拍
"""
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
import os
from ultralytics import YOLO

# 导入设备管理工具
from .device_utils import safe_numpy_conversion

class RacketDetector:
    """羽毛球拍检测器"""
    
    def __init__(self, model_path: str = None):
        """
        初始化羽毛球拍检测器
        
        Args:
            model_path: YOLO模型路径，如果为None则使用默认模型
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # 初始化YOLO模型
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                # 使用默认的YOLO模型
                self.model = YOLO('yolov8n.pt')
            
            # 确保模型在正确的设备上
            if self.device.type == 'cuda':
                self.model.to(self.device)
            
            print(f"Racket detector initialized on {self.device}")
        except Exception as e:
            print(f"Error initializing YOLO model: {e}")
            self.model = None
    
    def detect_rackets(self, frame: np.ndarray, confidence_threshold: float = 0.3) -> List[Tuple]:
        """
        检测图像中的羽毛球拍
        
        Args:
            frame: 输入图像 (BGR格式)
            confidence_threshold: 置信度阈值
            
        Returns:
            List[Tuple]: 检测结果列表，每个元素为 (x1, y1, x2, y2, confidence, class_id)
        """
        if self.model is None:
            return []
        
        try:
            # 使用YOLO进行检测
            results = self.model.predict(
                source=frame,
                conf=confidence_threshold,
                verbose=False
            )
            
            rackets = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    # 使用安全的张量转换
                    boxes = safe_numpy_conversion(result.boxes.xyxy)
                    confidences = safe_numpy_conversion(result.boxes.conf)
                    class_ids = safe_numpy_conversion(result.boxes.cls)
                    
                    # 检查转换是否成功
                    if boxes is None or confidences is None or class_ids is None:
                        print("Warning: Failed to convert tensors to numpy arrays")
                        return []
                    
                    for box, conf, class_id in zip(boxes, confidences, class_ids):
                        detection = (*box, conf, class_id)
                        rackets.append(detection)
            
            return rackets
            
        except Exception as e:
            print(f"Error in racket detection: {e}")
            return []
    
    def get_racket_center(self, racket_box: Tuple) -> Tuple[int, int]:
        """
        获取羽毛球拍中心点坐标
        
        Args:
            racket_box: 羽毛球拍边界框 (x1, y1, x2, y2, confidence, class_id)
            
        Returns:
            Tuple[int, int]: 中心点坐标 (x, y)
        """
        x1, y1, x2, y2 = racket_box[:4]
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return center_x, center_y
    
    def calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """
        计算两点之间的欧几里得距离
        
        Args:
            point1: 第一个点 (x1, y1)
            point2: 第二个点 (x2, y2)
            
        Returns:
            float: 两点之间的距离
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def visualize_rackets(self, frame: np.ndarray, rackets: List[Tuple]) -> np.ndarray:
        """
        在图像上可视化羽毛球拍检测结果
        
        Args:
            frame: 输入图像
            rackets: 检测到的羽毛球拍列表
            
        Returns:
            np.ndarray: 标注后的图像
        """
        annotated_frame = frame.copy()
        
        for racket in rackets:
            x1, y1, x2, y2, confidence, class_id = racket
            
            # 绘制边界框
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 绘制置信度
            label = f"Racket: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 绘制中心点
            center_x, center_y = self.get_racket_center(racket)
            cv2.circle(annotated_frame, (center_x, center_y), 3, (255, 0, 0), -1)
        
        return annotated_frame


class HitDetection:
    """击球检测器 - 基于羽毛球和羽毛球拍的运动模式"""
    
    def __init__(self, distance_threshold: float = 50.0, time_window: int = 5):
        """
        初始化击球检测器
        
        Args:
            distance_threshold: 羽毛球和羽毛球拍的距离阈值
            time_window: 时间窗口大小（帧数）
        """
        self.distance_threshold = distance_threshold
        self.time_window = time_window
        self.racket_detector = RacketDetector()
        
        # 存储历史数据
        self.shuttlecock_history = []  # [(frame_id, x, y), ...]
        self.racket_history = []       # [(frame_id, x, y), ...]
        self.hit_events = []           # [(frame_id, confidence), ...]
    
    def update_tracking(self, frame_id: int, shuttlecock_pos: Optional[Tuple[int, int]], 
                       racket_positions: List[Tuple[int, int]]):
        """
        更新跟踪数据
        
        Args:
            frame_id: 当前帧ID
            shuttlecock_pos: 羽毛球位置 (x, y) 或 None
            racket_positions: 羽毛球拍位置列表 [(x, y), ...]
        """
        # 更新羽毛球历史
        if shuttlecock_pos:
            self.shuttlecock_history.append((frame_id, *shuttlecock_pos))
        
        # 更新羽毛球拍历史
        for racket_pos in racket_positions:
            self.racket_history.append((frame_id, *racket_pos))
        
        # 保持历史数据在时间窗口内
        self._cleanup_history(frame_id)
    
    def _cleanup_history(self, current_frame_id: int):
        """清理超出时间窗口的历史数据"""
        cutoff_frame = current_frame_id - self.time_window
        
        self.shuttlecock_history = [
            (fid, x, y) for fid, x, y in self.shuttlecock_history 
            if fid >= cutoff_frame
        ]
        
        self.racket_history = [
            (fid, x, y) for fid, x, y in self.racket_history 
            if fid >= cutoff_frame
        ]
    
    def detect_hit_event(self, frame_id: int) -> Optional[float]:
        """
        检测击球事件
        
        Args:
            frame_id: 当前帧ID
            
        Returns:
            Optional[float]: 击球置信度，如果没有检测到击球则返回None
        """
        if len(self.shuttlecock_history) < 2 or len(self.racket_history) < 2:
            return None
        
        # 分析羽毛球和羽毛球拍的运动模式
        hit_confidence = self._analyze_motion_pattern(frame_id)
        
        if hit_confidence and hit_confidence > 0.5:
            self.hit_events.append((frame_id, hit_confidence))
            return hit_confidence
        
        return None
    
    def _analyze_motion_pattern(self, frame_id: int) -> Optional[float]:
        """
        分析运动模式来检测击球
        
        Args:
            frame_id: 当前帧ID
            
        Returns:
            Optional[float]: 击球置信度
        """
        # 获取最近的羽毛球和羽毛球拍位置
        recent_shuttlecock = self.shuttlecock_history[-3:] if len(self.shuttlecock_history) >= 3 else self.shuttlecock_history
        recent_rackets = self.racket_history[-3:] if len(self.racket_history) >= 3 else self.racket_history
        
        if not recent_shuttlecock or not recent_rackets:
            return None
        
        # 计算羽毛球和羽毛球拍之间的最小距离
        min_distance = float('inf')
        for _, sx, sy in recent_shuttlecock:
            for _, rx, ry in recent_rackets:
                distance = self.racket_detector.calculate_distance((sx, sy), (rx, ry))
                min_distance = min(min_distance, distance)
        
        # 如果距离小于阈值，可能是击球事件
        if min_distance <= self.distance_threshold:
            # 计算置信度（距离越近，置信度越高）
            confidence = max(0, 1 - (min_distance / self.distance_threshold))
            
            # 检查运动方向变化（击球后羽毛球方向应该改变）
            if len(recent_shuttlecock) >= 2:
                direction_change = self._check_direction_change(recent_shuttlecock)
                confidence *= direction_change
            
            return confidence
        
        return None
    
    def _check_direction_change(self, shuttlecock_positions: List[Tuple]) -> float:
        """
        检查羽毛球运动方向变化
        
        Args:
            shuttlecock_positions: 羽毛球位置历史 [(frame_id, x, y), ...]
            
        Returns:
            float: 方向变化置信度
        """
        if len(shuttlecock_positions) < 3:
            return 0.5
        
        # 计算前一段和后一段的运动方向
        positions = [(x, y) for _, x, y in shuttlecock_positions]
        
        # 前一段方向
        dx1 = positions[-2][0] - positions[-3][0]
        dy1 = positions[-2][1] - positions[-3][1]
        
        # 后一段方向
        dx2 = positions[-1][0] - positions[-2][0]
        dy2 = positions[-1][1] - positions[-2][1]
        
        # 计算方向变化角度
        if dx1 != 0 or dy1 != 0:
            angle1 = np.arctan2(dy1, dx1)
            angle2 = np.arctan2(dy2, dx2)
            angle_diff = abs(angle2 - angle1)
            
            # 角度变化越大，置信度越高
            return min(1.0, angle_diff / np.pi)
        
        return 0.5
    
    def get_hit_statistics(self) -> dict:
        """
        获取击球统计信息
        
        Returns:
            dict: 击球统计信息
        """
        if not self.hit_events:
            return {"total_hits": 0, "avg_confidence": 0.0}
        
        total_hits = len(self.hit_events)
        avg_confidence = sum(conf for _, conf in self.hit_events) / total_hits
        
        return {
            "total_hits": total_hits,
            "avg_confidence": avg_confidence,
            "hit_frames": [frame_id for frame_id, _ in self.hit_events]
        }
