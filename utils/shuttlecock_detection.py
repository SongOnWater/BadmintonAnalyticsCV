"""
羽毛球检测器 - 使用YOLO-World开放词汇检测
"""
import cv2
import numpy as np
import torch
import os
from typing import List, Tuple, Optional
from ultralytics import YOLOWorld

class ShuttlecockDetector:
    """羽毛球检测器"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化羽毛球检测器
        
        Args:
            model_path: YOLO-World模型路径，如果为None则使用默认模型
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.shuttlecock_keywords = [
            "badminton shuttlecock", 
            "shuttlecock", 
            "badminton birdie",
            "birdie",
            "羽毛球",
            "羽毛球球",
            "羽毛球毛球"
        ]
        
        try:
            # 初始化YOLO-World模型
            if model_path and os.path.exists(model_path):
                self.model = YOLOWorld(model_path)
            else:
                # 使用本地下载的最强YOLO-World模型
                local_model_path = 'ckpts/yolov8x-worldv2.pt'
                if os.path.exists(local_model_path):
                    self.model = YOLOWorld(local_model_path)
                else:
                    # 备用方案：使用在线模型
                    self.model = YOLOWorld('yolov8x-worldv2.pt')
            
            # 确保模型在正确的设备上
            if self.device.type == 'cuda':
                self.model.to(self.device)
            
            print(f"Shuttlecock detector initialized on {self.device}")
        except Exception as e:
            print(f"Error initializing YOLO-World model for shuttlecock: {e}")
            self.model = None
    
    def detect_shuttlecocks(self, frame: np.ndarray, confidence_threshold: float = 0.3) -> List[Tuple]:
        """
        检测图像中的羽毛球
        
        Args:
            frame: 输入图像 (BGR格式)
            confidence_threshold: 置信度阈值
            
        Returns:
            List[Tuple]: 检测到的羽毛球信息 [(x1, y1, x2, y2, confidence, class_id), ...]
        """
        if self.model is None:
            return []
        
        try:
            # 使用YOLO-World进行检测，指定羽毛球关键词
            self.model.set_classes(self.shuttlecock_keywords)
            
            # YOLO-World可以直接处理numpy数组
            results = self.model.predict(
                source=frame,
                conf=confidence_threshold,
                verbose=False
            )
            
            shuttlecocks = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    
                    for box, conf, class_id in zip(boxes, confidences, class_ids):
                        shuttlecocks.append((*box, conf, class_id))
            
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
