"""
改进的击球识别模块
集成羽毛球拍检测、动作识别和视觉大模型
"""
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
import os
import json
from collections import deque

from .racket_detection import RacketDetector, HitDetection

try:
    from transformers import pipeline, AutoProcessor, AutoModel
    import torch.nn.functional as F
except ImportError as e:
    print(f"Warning: Could not import transformers: {e}")

class ActionRecognition:
    """动作识别模块 - 用于识别击球动作"""
    
    def __init__(self, model_name: str = "microsoft/xclip-base-patch32"):
        """
        初始化动作识别器
        
        Args:
            model_name: 预训练模型名称
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # 使用CLIP模型进行图像理解，因为XCLIP需要特殊的视频分类模型
            from transformers import CLIPProcessor, CLIPModel
            
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            
            # 定义击球相关的动作类别
            self.hit_actions = [
                "badminton hit", "tennis hit", "racket swing", "smash", "serve",
                "羽毛球击球", "网球击球", "挥拍", "扣杀", "发球"
            ]
            
            print(f"Action recognition initialized on {self.device}")
        except Exception as e:
            print(f"Error initializing action recognition: {e}")
            self.processor = None
            self.model = None
    
    def recognize_action(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        识别视频帧中的动作
        
        Args:
            frames: 视频帧列表
            
        Returns:
            Dict[str, float]: 动作类别和置信度
        """
        if self.model is None or len(frames) < 1:
            return {"no_action": 1.0}
        
        try:
            # 使用中间帧进行动作识别
            middle_frame = frames[len(frames) // 2]
            
            # 使用CLIP进行图像理解
            inputs = self.processor(images=middle_frame, return_tensors="pt").to(self.device)
            
            # 获取图像特征
            image_features = self.model.get_image_features(**inputs)
            
            # 计算与击球动作的相似度
            hit_confidence = 0.0
            for action in self.hit_actions:
                text_inputs = self.processor(text=action, return_tensors="pt", padding=True).to(self.device)
                text_features = self.model.get_text_features(**text_inputs)
                
                # 计算相似度
                similarity = torch.cosine_similarity(image_features, text_features).item()
                hit_confidence = max(hit_confidence, similarity)
            
            return {
                "hit_action": max(0, hit_confidence),
                "no_hit": max(0, 1.0 - hit_confidence)
            }
            
        except Exception as e:
            print(f"Error in action recognition: {e}")
            return {"no_action": 1.0}


class VisionLLMIntegration:
    """视觉大模型集成 - 用于高级场景理解"""
    
    def __init__(self, model_name: str = "microsoft/git-base"):
        """
        初始化视觉大模型
        
        Args:
            model_name: 预训练模型名称
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # 初始化视觉-语言模型
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            
            # 定义击球相关的提示词
            self.hit_prompts = [
                "A person hitting a badminton shuttlecock with a racket",
                "Badminton player making a shot",
                "Racket hitting shuttlecock",
                "羽毛球运动员击球",
                "挥拍击打羽毛球"
            ]
            
            print(f"Vision LLM initialized on {self.device}")
        except Exception as e:
            print(f"Error initializing vision LLM: {e}")
            self.processor = None
            self.model = None
    
    def analyze_scene(self, frame: np.ndarray) -> Dict[str, float]:
        """
        分析场景中的击球行为
        
        Args:
            frame: 输入图像
            
        Returns:
            Dict[str, float]: 场景分析结果
        """
        if self.processor is None or self.model is None:
            return {"hit_scene": 0.0}
        
        try:
            # 预处理图像
            inputs = self.processor(images=frame, return_tensors="pt").to(self.device)
            
            # 计算图像特征
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_features = outputs.image_embeds
            
            # 计算与击球提示词的相似度
            hit_scores = []
            for prompt in self.hit_prompts:
                text_inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    text_outputs = self.model(**text_inputs)
                    text_features = text_outputs.text_embeds
                
                # 计算余弦相似度
                similarity = F.cosine_similarity(image_features, text_features, dim=-1)
                hit_scores.append(similarity.item())
            
            # 返回最高相似度
            max_hit_score = max(hit_scores) if hit_scores else 0.0
            
            return {
                "hit_scene": max_hit_score,
                "no_hit_scene": 1.0 - max_hit_score
            }
            
        except Exception as e:
            print(f"Error in scene analysis: {e}")
            return {"hit_scene": 0.0}


"""
改进的击球检测器 - 基于"接近-远离"相对运动模式
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class ImprovedHitDetection:
    """改进的击球检测器 - 基于接近-远离运动模式"""
    
    def __init__(self, 
                 min_approach_distance: float = 50.0,      # 最小接近距离阈值
                 max_separation_distance: float = 200.0,   # 最大分离距离阈值
                 time_window: int = 10,                    # 时间窗口（帧数）
                 min_velocity_change: float = 5.0,         # 最小速度变化阈值
                 confidence_threshold: float = 0.6):        # 置信度阈值
        
        self.min_approach_distance = min_approach_distance
        self.max_separation_distance = max_separation_distance
        self.time_window = time_window
        self.min_velocity_change = min_velocity_change
        self.confidence_threshold = confidence_threshold
        
        # 历史数据存储
        self.shuttlecock_history = []  # [(frame_id, x, y, timestamp), ...]
        self.racket_history = []       # [(frame_id, x, y, timestamp), ...]
        self.hit_events = []           # [(frame_id, confidence, details), ...]
        
        # 击球检测状态
        self.approach_detected = False
        self.approach_frame = None
        self.approach_distance = None
        
        logger.info(f"改进击球检测器初始化完成 - 接近阈值: {min_approach_distance}, 分离阈值: {max_separation_distance}")
    
    def update_tracking(self, frame_id: int, shuttlecock_pos: Optional[Tuple[int, int]], 
                       racket_positions: List[Tuple[int, int]], timestamp: float = None):
        """
        更新跟踪数据
        
        Args:
            frame_id: 当前帧ID
            shuttlecock_pos: 羽毛球位置 (x, y) 或 None
            racket_positions: 羽毛球拍位置列表 [(x, y), ...]
            timestamp: 时间戳（可选）
        """
        if timestamp is None:
            timestamp = frame_id
        
        # 更新羽毛球历史
        if shuttlecock_pos:
            self.shuttlecock_history.append((frame_id, shuttlecock_pos[0], shuttlecock_pos[1], timestamp))
        
        # 更新羽毛球拍历史（使用最近的一个位置）
        if racket_positions:
            # 选择最接近羽毛球位置的球拍
            if shuttlecock_pos:
                closest_racket = min(racket_positions, 
                                   key=lambda pos: self._calculate_distance(shuttlecock_pos, pos))
                self.racket_history.append((frame_id, closest_racket[0], closest_racket[1], timestamp))
            else:
                # 如果没有羽毛球位置，使用第一个球拍位置
                self.racket_history.append((frame_id, racket_positions[0][0], racket_positions[0][1], timestamp))
        
        # 清理历史数据，保持时间窗口
        self._cleanup_history(frame_id)
    
    def detect_hit_event(self, frame_id: int) -> Optional[float]:
        """
        检测击球事件
        
        Args:
            frame_id: 当前帧ID
            
        Returns:
            Optional[float]: 击球置信度，如果没有检测到击球则返回None
        """
        if len(self.shuttlecock_history) < 3 or len(self.racket_history) < 3:
            return None
        
        # 分析运动模式
        hit_confidence = self._analyze_approach_separation_pattern(frame_id)
        
        if hit_confidence and hit_confidence >= self.confidence_threshold:
            # 记录击球事件
            details = self._get_hit_details(frame_id)
            self.hit_events.append((frame_id, hit_confidence, details))
            
            # 重置接近检测状态
            self.approach_detected = False
            self.approach_frame = None
            self.approach_distance = None
            
            logger.info(f"检测到击球事件 - 帧 {frame_id}, 置信度: {hit_confidence:.3f}")
            return hit_confidence
        
        return None
    
    def _analyze_approach_separation_pattern(self, frame_id: int) -> Optional[float]:
        """
        分析"接近-远离"运动模式
        
        Args:
            frame_id: 当前帧ID
            
        Returns:
            Optional[float]: 击球置信度
        """
        # 获取最近的羽毛球和球拍位置
        recent_shuttlecock = self._get_recent_positions(self.shuttlecock_history, frame_id)
        recent_rackets = self._get_recent_positions(self.racket_history, frame_id)
        
        if not recent_shuttlecock or not recent_rackets:
            return None
        
        # 计算当前距离
        current_distance = self._calculate_min_distance(recent_shuttlecock, recent_rackets)
        
        # 检测接近阶段
        if not self.approach_detected and current_distance <= self.min_approach_distance:
            self.approach_detected = True
            self.approach_frame = frame_id
            self.approach_distance = current_distance
            logger.debug(f"检测到接近阶段 - 帧 {frame_id}, 距离: {current_distance:.2f}")
            return None
        
        # 检测分离阶段
        if self.approach_detected and current_distance >= self.max_separation_distance:
            # 计算分离时间
            separation_time = frame_id - self.approach_frame
            
            # 分析运动模式
            pattern_confidence = self._analyze_motion_pattern(recent_shuttlecock, recent_rackets)
            
            # 计算最终置信度
            hit_confidence = self._calculate_hit_confidence(
                self.approach_distance, current_distance, separation_time, pattern_confidence
            )
            
            return hit_confidence
        
        return None
    
    def _analyze_motion_pattern(self, shuttlecock_positions: List[Tuple], 
                               racket_positions: List[Tuple]) -> float:
        """
        分析运动模式
        
        Args:
            shuttlecock_positions: 羽毛球位置历史
            racket_positions: 球拍位置历史
            
        Returns:
            float: 运动模式置信度
        """
        if len(shuttlecock_positions) < 3 or len(racket_positions) < 3:
            return 0.5
        
        # 1. 羽毛球速度变化分析
        shuttlecock_velocity_change = self._analyze_velocity_change(shuttlecock_positions)
        
        # 2. 球拍速度分析
        racket_velocity = self._analyze_racket_velocity(racket_positions)
        
        # 3. 相对运动分析
        relative_motion = self._analyze_relative_motion(shuttlecock_positions, racket_positions)
        
        # 综合置信度
        pattern_confidence = (shuttlecock_velocity_change * 0.4 + 
                            racket_velocity * 0.3 + 
                            relative_motion * 0.3)
        
        return pattern_confidence
    
    def _analyze_velocity_change(self, positions: List[Tuple]) -> float:
        """
        分析羽毛球速度变化
        
        Args:
            positions: 位置历史 [(frame_id, x, y, timestamp), ...]
            
        Returns:
            float: 速度变化置信度
        """
        if len(positions) < 3:
            return 0.5
        
        # 计算前一段和后一段的速度
        positions_sorted = sorted(positions, key=lambda x: x[0])
        
        # 前一段速度
        dx1 = positions_sorted[-2][1] - positions_sorted[-3][1]
        dy1 = positions_sorted[-2][2] - positions_sorted[-3][2]
        v1 = np.sqrt(dx1**2 + dy1**2)
        
        # 后一段速度
        dx2 = positions_sorted[-1][1] - positions_sorted[-2][1]
        dy2 = positions_sorted[-1][2] - positions_sorted[-2][2]
        v2 = np.sqrt(dx2**2 + dy2**2)
        
        # 速度变化
        velocity_change = abs(v2 - v1)
        
        # 速度变化越大，置信度越高
        return min(1.0, velocity_change / self.min_velocity_change)
    
    def _analyze_racket_velocity(self, positions: List[Tuple]) -> float:
        """
        分析球拍速度
        
        Args:
            positions: 位置历史 [(frame_id, x, y, timestamp), ...]
            
        Returns:
            float: 球拍速度置信度
        """
        if len(positions) < 2:
            return 0.5
        
        positions_sorted = sorted(positions, key=lambda x: x[0])
        
        # 计算球拍移动速度
        dx = positions_sorted[-1][1] - positions_sorted[-2][1]
        dy = positions_sorted[-1][2] - positions_sorted[-2][2]
        velocity = np.sqrt(dx**2 + dy**2)
        
        # 球拍移动越快，置信度越高（表示挥拍动作）
        return min(1.0, velocity / 20.0)  # 20像素/帧作为参考
    
    def _analyze_relative_motion(self, shuttlecock_positions: List[Tuple], 
                                racket_positions: List[Tuple]) -> float:
        """
        分析相对运动
        
        Args:
            shuttlecock_positions: 羽毛球位置历史
            racket_positions: 球拍位置历史
            
        Returns:
            float: 相对运动置信度
        """
        if len(shuttlecock_positions) < 2 or len(racket_positions) < 2:
            return 0.5
        
        # 计算羽毛球和球拍的运动方向
        shuttlecock_direction = self._calculate_direction(shuttlecock_positions)
        racket_direction = self._calculate_direction(racket_positions)
        
        # 计算方向差异
        if shuttlecock_direction is not None and racket_direction is not None:
            angle_diff = abs(shuttlecock_direction - racket_direction)
            # 方向差异越大，置信度越高（表示击球后方向改变）
            return min(1.0, angle_diff / np.pi)
        
        return 0.5
    
    def _calculate_direction(self, positions: List[Tuple]) -> Optional[float]:
        """
        计算运动方向
        
        Args:
            positions: 位置历史 [(frame_id, x, y, timestamp), ...]
            
        Returns:
            Optional[float]: 运动方向角度（弧度）
        """
        if len(positions) < 2:
            return None
        
        positions_sorted = sorted(positions, key=lambda x: x[0])
        
        dx = positions_sorted[-1][1] - positions_sorted[-2][1]
        dy = positions_sorted[-1][2] - positions_sorted[-2][2]
        
        if dx == 0 and dy == 0:
            return None
        
        return np.arctan2(dy, dx)
    
    def _calculate_hit_confidence(self, approach_distance: float, separation_distance: float, 
                                 separation_time: int, pattern_confidence: float) -> float:
        """
        计算击球置信度
        
        Args:
            approach_distance: 接近距离
            separation_distance: 分离距离
            separation_time: 分离时间
            pattern_confidence: 运动模式置信度
            
        Returns:
            float: 击球置信度
        """
        # 距离因子：接近距离越小，分离距离越大，置信度越高
        distance_factor = (1.0 - approach_distance / self.min_approach_distance) * 0.5 + \
                         (separation_distance / self.max_separation_distance) * 0.5
        
        # 时间因子：分离时间越短，置信度越高（击球是瞬间事件）
        time_factor = max(0, 1.0 - separation_time / self.time_window)
        
        # 综合置信度
        confidence = (distance_factor * 0.4 + 
                     time_factor * 0.3 + 
                     pattern_confidence * 0.3)
        
        return min(1.0, confidence)
    
    def _get_recent_positions(self, history: List[Tuple], current_frame: int) -> List[Tuple]:
        """
        获取最近的位置历史
        
        Args:
            history: 位置历史
            current_frame: 当前帧ID
            
        Returns:
            List[Tuple]: 最近的位置历史
        """
        cutoff_frame = current_frame - self.time_window
        return [(fid, x, y, timestamp) for fid, x, y, timestamp in history if fid >= cutoff_frame]
    
    def _calculate_min_distance(self, positions1: List[Tuple], positions2: List[Tuple]) -> float:
        """
        计算两组位置之间的最小距离
        
        Args:
            positions1: 第一组位置 [(frame_id, x, y, timestamp), ...]
            positions2: 第二组位置 [(frame_id, x, y, timestamp), ...]
            
        Returns:
            float: 最小距离
        """
        if not positions1 or not positions2:
            return float('inf')
        
        min_distance = float('inf')
        for _, x1, y1, _ in positions1:
            for _, x2, y2, _ in positions2:
                distance = self._calculate_distance((x1, y1), (x2, y2))
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """
        计算两点之间的欧几里得距离
        
        Args:
            point1: 第一个点 (x1, y1)
            point2: 第二个点 (x2, y2)
            
        Returns:
            float: 两点之间的距离
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _cleanup_history(self, current_frame: int):
        """清理历史数据"""
        cutoff_frame = current_frame - self.time_window * 2
        
        self.shuttlecock_history = [
            (fid, x, y, timestamp) for fid, x, y, timestamp in self.shuttlecock_history 
            if fid >= cutoff_frame
        ]
        
        self.racket_history = [
            (fid, x, y, timestamp) for fid, x, y, timestamp in self.racket_history 
            if fid >= cutoff_frame
        ]
    
    def _get_hit_details(self, frame_id: int) -> Dict:
        """
        获取击球事件详细信息
        
        Args:
            frame_id: 击球帧ID
            
        Returns:
            Dict: 击球详细信息
        """
        return {
            'approach_frame': self.approach_frame,
            'approach_distance': self.approach_distance,
            'separation_frame': frame_id,
            'separation_time': frame_id - self.approach_frame if self.approach_frame else 0
        }
    
    def get_hit_statistics(self) -> Dict:
        """
        获取击球统计信息
        
        Returns:
            Dict: 击球统计信息
        """
        if not self.hit_events:
            return {
                "total_hits": 0,
                "avg_confidence": 0.0,
                "hit_frames": [],
                "approach_separation_patterns": []
            }
        
        total_hits = len(self.hit_events)
        avg_confidence = sum(conf for _, conf, _ in self.hit_events) / total_hits
        hit_frames = [frame_id for frame_id, _, _ in self.hit_events]
        
        # 分析接近-分离模式
        approach_separation_patterns = []
        for _, confidence, details in self.hit_events:
            if details:
                pattern = {
                    'confidence': confidence,
                    'approach_distance': details.get('approach_distance', 0),
                    'separation_time': details.get('separation_time', 0)
                }
                approach_separation_patterns.append(pattern)
        
        return {
            "total_hits": total_hits,
            "avg_confidence": avg_confidence,
            "hit_frames": hit_frames,
            "approach_separation_patterns": approach_separation_patterns
        }
    
    def reset(self):
        """重置检测器状态"""
        self.shuttlecock_history.clear()
        self.racket_history.clear()
        self.hit_events.clear()
        self.approach_detected = False
        self.approach_frame = None
        self.approach_distance = None
