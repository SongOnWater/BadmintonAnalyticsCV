"""
羽毛球检测模块
使用YOLO模型检测羽毛球
"""
import cv2
import numpy as np
import torch
import os
from typing import List, Tuple, Optional
from ultralytics import YOLO

# 导入设备管理工具
from .device_utils import safe_numpy_conversion

class ShuttlecockDetector:
    """羽毛球检测器"""
    
    def __init__(self, model_path: str = None):
        """
        初始化羽毛球检测器
        
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
            
            print(f"Shuttlecock detector initialized on {self.device}")
        except Exception as e:
            print(f"Error initializing YOLO model: {e}")
            self.model = None
    
    def detect_shuttlecocks(self, frame: np.ndarray, confidence_threshold: float = 0.3) -> List[Tuple]:
        """
        检测图像中的羽毛球
        
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
            
            shuttlecocks = []
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
                        shuttlecocks.append(detection)
            
            return shuttlecocks
            
        except Exception as e:
            print(f"Error in shuttlecock detection: {e}")
            return []
    
    def get_shuttlecock_center(self, shuttlecock_box: Tuple) -> Tuple[int, int]:
        """
        获取羽毛球中心点坐标
        
        Args:
            shuttlecock_box: 羽毛球边界框 (x1, y1, x2, y2, confidence, class_id)
            
        Returns:
            Tuple[int, int]: 中心点坐标 (x, y)
        """
        x1, y1, x2, y2 = shuttlecock_box[:4]
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
    
    def visualize_shuttlecocks(self, frame: np.ndarray, shuttlecocks: List[Tuple]) -> np.ndarray:
        """
        在图像上可视化羽毛球检测结果
        
        Args:
            frame: 输入图像
            shuttlecocks: 检测到的羽毛球列表
            
        Returns:
            np.ndarray: 可视化后的图像
        """
        result_frame = frame.copy()
        
        for shuttlecock in shuttlecocks:
            x1, y1, x2, y2, confidence, class_id = shuttlecock
            
            # 绘制边界框
            cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 绘制置信度
            label = f"Shuttlecock: {confidence:.2f}"
            cv2.putText(result_frame, label, (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 绘制中心点
            center = self.get_shuttlecock_center(shuttlecock)
            cv2.circle(result_frame, center, 5, (0, 255, 0), -1)
        
        return result_frame
