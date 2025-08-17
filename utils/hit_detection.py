#!/usr/bin/env python3
"""
羽毛球击球检测和统计模块
包含击球检测算法、统计逻辑和绘制功能
"""
import numpy as np
import cv2
from collections import deque


class HitDetector:
    """击球检测器"""
    
    def __init__(self, frame_height=720, frame_width=1280):
        self.frame_height = frame_height
        self.frame_width = frame_width
        
        # 羽毛球场地参数（相对于视频帧的比例）
        self.court_boundaries = {
            'top': 0.15,      # 场地顶部边界（相对高度）
            'bottom': 0.85,   # 场地底部边界（相对高度）
            'left': 0.1,      # 场地左边界（相对宽度）
            'right': 0.9,     # 场地右边界（相对宽度）
            'net_y': 0.5      # 网的位置（相对高度）
        }
        
    def detect_hit_from_trajectory(self, traj_queue, current_frame, last_hit_frame):
        """改进的击球检测算法
        
        Args:
            traj_queue (deque): 轨迹队列，包含最近的球位置
            current_frame (int): 当前帧数
            last_hit_frame (int): 上次击球的帧数
        
        Returns:
            dict: 击球检测结果 {'hit_detected': bool, 'hit_type': str, 'confidence': float}
        """
        
        # 时间间隔检查：避免过于频繁的击球检测
        if current_frame - last_hit_frame < 12:  # 进一步减少到12帧，提高敏感度
            return {'hit_detected': False, 'hit_type': 'too_soon', 'confidence': 0.0}
        
        # 获取轨迹点（从新到旧）
        valid_points = [point for point in traj_queue if point is not None]
        
        if len(valid_points) < 5:  # 需要更多点来准确分析
            return {'hit_detected': False, 'hit_type': 'insufficient_data', 'confidence': 0.0}
        
        # 计算各种运动特征
        velocities = []
        vertical_velocities = []
        horizontal_velocities = []
        
        for i in range(1, min(5, len(valid_points))):
            dx = valid_points[i-1][0] - valid_points[i][0]
            dy = valid_points[i-1][1] - valid_points[i][1]
            
            velocity = np.sqrt(dx*dx + dy*dy)
            velocities.append(velocity)
            vertical_velocities.append(abs(dy))
            horizontal_velocities.append(abs(dx))
        
        if len(velocities) < 3:
            return {'hit_detected': False, 'hit_type': 'insufficient_velocity_data', 'confidence': 0.0}
        
        # 获取球的当前位置（用于重力判断）
        current_y = valid_points[0][1]
        ball_height_ratio = current_y / self.frame_height  # 0表示顶部，1表示底部
        
        # 检测1：排除重力导致的自然下落
        recent_dy = valid_points[0][1] - valid_points[1][1]  # 正值表示向下
        prev_dy = valid_points[1][1] - valid_points[2][1]
        
        # 如果球在上半部分且主要是向下运动，可能是重力下落
        if ball_height_ratio < 0.6 and recent_dy > 0 and prev_dy > 0:
            # 检查是否是自然的重力加速
            if recent_dy > prev_dy * 0.8:  # 下落加速是正常的
                gravity_confidence = min(0.8, (recent_dy - prev_dy) / 10.0)
                if gravity_confidence > 0.3:
                    return {'hit_detected': False, 'hit_type': 'gravity_fall', 'confidence': gravity_confidence}
        
        # 检测2：真正的击球特征
        hit_confidence = 0.0
        hit_reasons = []
        
        # 特征1：速度突变（击球会导致速度突然增加）
        recent_velocity = velocities[0]
        avg_prev_velocity = np.mean(velocities[1:3])
        
        # 降低速度突变的阈值，提高敏感度
        if recent_velocity > avg_prev_velocity * 1.4 and recent_velocity > 20:
            # 避免除零错误
            if avg_prev_velocity > 0:
                speed_boost = min(1.0, (recent_velocity - avg_prev_velocity) / avg_prev_velocity)
            else:
                speed_boost = 1.0  # 如果之前速度为0，设为最大值
            hit_confidence += speed_boost * 0.5  # 增加权重
            hit_reasons.append(f"speed_boost({speed_boost:.2f})")
        
        # 特征2：水平方向突变（排除纯垂直运动）
        if len(valid_points) >= 4:
            dx1 = valid_points[0][0] - valid_points[1][0]
            dx2 = valid_points[1][0] - valid_points[2][0]
            dx3 = valid_points[2][0] - valid_points[3][0]
            
            # 检测水平方向的突然改变
            if abs(dx1) > 15 and abs(dx2) > 8:  # 降低阈值
                if (dx1 > 0) != (dx2 > 0):  # 方向相反
                    # 确保不是纯垂直的重力运动
                    horizontal_ratio = abs(dx1) / (abs(recent_dy) + 1)
                    if horizontal_ratio > 0.3:  # 降低水平分量要求
                        direction_change = min(1.0, abs(dx1) / 40.0)
                        hit_confidence += direction_change * 0.4  # 增加权重
                        hit_reasons.append(f"direction_change({direction_change:.2f})")
        
        # 特征3：综合运动模式分析
        if len(valid_points) >= 5:
            # 分析最近3个点的运动模式
            recent_pattern = []
            for i in range(3):
                dx = valid_points[i][0] - valid_points[i+1][0]
                dy = valid_points[i][1] - valid_points[i+1][1]
                recent_pattern.append((dx, dy))
            
            # 检测运动模式的突变
            pattern_change = 0.0
            for i in range(1, len(recent_pattern)):
                prev_angle = np.arctan2(recent_pattern[i-1][1], recent_pattern[i-1][0])
                curr_angle = np.arctan2(recent_pattern[i][1], recent_pattern[i][0])
                angle_diff = abs(prev_angle - curr_angle)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                pattern_change = max(pattern_change, angle_diff)
            
            if pattern_change > 1.0:  # 大于约57度的角度变化
                trajectory_change = min(1.0, pattern_change / np.pi)
                hit_confidence += trajectory_change * 0.3
                hit_reasons.append(f"trajectory_change({trajectory_change:.2f})")
        
        # 综合判断 - 进一步降低阈值
        hit_detected = hit_confidence > 0.35  # 进一步降低阈值
        hit_type = "hit_" + "_".join(hit_reasons) if hit_detected else "no_hit"
        
        return {
            'hit_detected': hit_detected,
            'hit_type': hit_type,
            'confidence': hit_confidence,
            'ball_position': ball_height_ratio,
            'reasons': hit_reasons
        }
    
    def detect_ball_landing(self, traj_queue, current_frame, last_landing_frame):
        """检测羽毛球落地
        
        Args:
            traj_queue (deque): 轨迹队列，包含最近的球位置
            current_frame (int): 当前帧数
            last_landing_frame (int): 上次落地的帧数
        
        Returns:
            dict: 落地检测结果 {'landing_detected': bool, 'landing_type': str, 'position': dict}
        """
        
        # 时间间隔检查：避免过于频繁的落地检测
        if current_frame - last_landing_frame < 20:  # 至少间隔20帧
            return {'landing_detected': False, 'landing_type': 'too_soon', 'position': None}
        
        # 获取轨迹点（从新到旧）
        valid_points = [point for point in traj_queue if point is not None]
        
        if len(valid_points) < 4:
            return {'landing_detected': False, 'landing_type': 'insufficient_data', 'position': None}
        
        # 分析垂直运动模式
        recent_y_positions = [point[1] for point in valid_points[:4]]
        recent_x_positions = [point[0] for point in valid_points[:4]]
        
        # 计算垂直速度变化
        dy_recent = recent_y_positions[0] - recent_y_positions[1]  # 最近的垂直变化
        dy_prev = recent_y_positions[1] - recent_y_positions[2]    # 之前的垂直变化
        
        # 检测落地特征
        landing_confidence = 0.0
        landing_reasons = []
        
        # 特征1：球接近场地底部
        current_y = valid_points[0][1]
        current_x = valid_points[0][0]
        y_ratio = current_y / self.frame_height
        x_ratio = current_x / self.frame_width
        
        # 检查球是否在场地范围内且接近底部
        if y_ratio > self.court_boundaries['bottom'] - 0.1:  # 接近场地底部
            landing_confidence += 0.4
            landing_reasons.append("near_ground")
        
        # 特征2：垂直速度突然减小或反向（弹跳效应）
        if dy_recent < 0 and dy_prev > 0:  # 从向下变为向上（弹跳）
            bounce_strength = abs(dy_recent + dy_prev) / max(abs(dy_prev), 1)
            landing_confidence += min(0.5, bounce_strength)
            landing_reasons.append(f"bounce({bounce_strength:.2f})")
        
        # 特征3：球突然消失（可能落地后被遮挡）
        if len(valid_points) < len([p for p in traj_queue]):  # 有None值，表示球消失
            landing_confidence += 0.3
            landing_reasons.append("ball_disappeared")
        
        # 特征4：水平速度保持但垂直速度变化（典型的落地模式）
        if len(valid_points) >= 4:
            dx_recent = abs(recent_x_positions[0] - recent_x_positions[1])
            dx_prev = abs(recent_x_positions[1] - recent_x_positions[2])
            
            if dx_recent > 5 and abs(dy_recent) > abs(dx_recent) * 0.5:  # 垂直变化显著
                landing_confidence += 0.2
                landing_reasons.append("vertical_dominant")
        
        # 判断落地位置
        landing_position = None
        if landing_confidence > 0.5:
            # 判断落地位置
            if (self.court_boundaries['left'] <= x_ratio <= self.court_boundaries['right'] and
                self.court_boundaries['top'] <= y_ratio <= self.court_boundaries['bottom']):
                if x_ratio < 0.5:
                    landing_position = "left_court"
                else:
                    landing_position = "right_court"
            else:
                landing_position = "out_of_bounds"
        
        landing_detected = landing_confidence > 0.5
        landing_type = "landing_" + "_".join(landing_reasons) if landing_detected else "no_landing"
        
        return {
            'landing_detected': landing_detected,
            'landing_type': landing_type,
            'confidence': landing_confidence,
            'position': {
                'x': current_x,
                'y': current_y,
                'x_ratio': x_ratio,
                'y_ratio': y_ratio,
                'court_area': landing_position
            },
            'reasons': landing_reasons
        }


class HitCounter:
    """击球计数器"""
    
    def __init__(self, frame_width=1280):
        self.frame_width = frame_width
        self.hit_counts = {"p1": 0, "p2": 0}
        self.last_hit_frame = -30
        self.last_hit_side = None
        self.hit_buffer = []
        
        # 落地统计
        self.landing_counts = {"total": 0, "in_bounds": 0, "out_of_bounds": 0, "left_court": 0, "right_court": 0}
        self.last_landing_frame = -30
        self.landing_buffer = []
        
    def add_hit_detection(self, frame, confidence, ball_x, ball_position, reasons):
        """添加击球检测结果到缓冲区"""
        self.hit_buffer.append({
            'frame': frame,
            'confidence': confidence,
            'ball_x': ball_x,
            'ball_position': ball_position,
            'reasons': reasons
        })
    
    def process_hit_buffer(self, current_frame):
        """处理击球缓冲区，返回确认的击球"""
        confirmed_hits = []
        for hit in self.hit_buffer[:]:
            if current_frame - hit['frame'] >= 3:  # 3帧延迟确认
                confirmed_hits.append(hit)
                self.hit_buffer.remove(hit)
        
        return confirmed_hits
    
    def assign_hit_to_player(self, hit):
        """将击球分配给对应球员"""
        # 检查是否与最近的击球太接近（避免重复计数）
        if self.last_hit_frame > 0 and abs(hit['frame'] - self.last_hit_frame) < 20:
            return None  # 跳过太接近的击球
        
        # 根据球的位置判断是哪个球员击球
        ball_x = hit['ball_x']
        current_side = 'left' if ball_x < self.frame_width / 2 else 'right'
        
        # 智能分配球员
        if self.last_hit_side is None:
            # 第一次击球，根据位置分配
            if current_side == 'left':
                self.hit_counts["p1"] += 1
                assigned_player = "P1"
            else:
                self.hit_counts["p2"] += 1
                assigned_player = "P2"
        else:
            # 根据场地位置和上次击球位置判断
            if current_side != self.last_hit_side:
                # 球从一边到另一边，正常的回合
                if current_side == 'left':
                    self.hit_counts["p1"] += 1
                    assigned_player = "P1"
                else:
                    self.hit_counts["p2"] += 1
                    assigned_player = "P2"
            else:
                # 球在同一边，可能是连续击球或误检
                if hit['confidence'] > 0.8:  # 高置信度才计数
                    if current_side == 'left':
                        self.hit_counts["p1"] += 1
                        assigned_player = "P1"
                    else:
                        self.hit_counts["p2"] += 1
                        assigned_player = "P2"
                else:
                    return None  # 跳过低置信度的同侧击球
        
        self.last_hit_side = current_side
        self.last_hit_frame = hit['frame']
        
        return {
            'player': assigned_player,
            'side': current_side,
            'frame': hit['frame'],
            'confidence': hit['confidence'],
            'reasons': hit['reasons']
        }
    
    def add_landing_detection(self, frame, confidence, position, reasons):
        """添加落地检测结果到缓冲区"""
        self.landing_buffer.append({
            'frame': frame,
            'confidence': confidence,
            'position': position,
            'reasons': reasons
        })
    
    def process_landing_buffer(self, current_frame):
        """处理落地缓冲区，返回确认的落地"""
        confirmed_landings = []
        for landing in self.landing_buffer[:]:
            if current_frame - landing['frame'] >= 5:  # 5帧延迟确认
                confirmed_landings.append(landing)
                self.landing_buffer.remove(landing)
        
        return confirmed_landings
    
    def record_landing(self, landing):
        """记录落地事件"""
        # 检查是否与最近的落地太接近（避免重复计数）
        if self.last_landing_frame > 0 and abs(landing['frame'] - self.last_landing_frame) < 30:
            return None  # 跳过太接近的落地
        
        position = landing['position']
        court_area = position['court_area']
        
        # 更新落地统计
        self.landing_counts["total"] += 1
        
        if court_area == "out_of_bounds":
            self.landing_counts["out_of_bounds"] += 1
        else:
            self.landing_counts["in_bounds"] += 1
            if court_area == "left_court":
                self.landing_counts["left_court"] += 1
            elif court_area == "right_court":
                self.landing_counts["right_court"] += 1
        
        self.last_landing_frame = landing['frame']
        
        return {
            'frame': landing['frame'],
            'confidence': landing['confidence'],
            'position': position,
            'court_area': court_area,
            'reasons': landing['reasons']
        }
    
    def get_hit_counts(self):
        """获取当前击球计数"""
        return self.hit_counts.copy()
    
    def get_landing_counts(self):
        """获取当前落地计数"""
        return self.landing_counts.copy()


class HitDisplayRenderer:
    """击球数据显示渲染器"""
    
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_thickness = 2
        self.score_font_color = (0, 0, 255)  # 红色用于得分
        self.hit_font_color = (255, 0, 0)    # 蓝色用于击球数据
        self.font_line_type = cv2.LINE_AA
        self.padding = 10
    
    def render_hit_data(self, frame, scores, hit_counts, landing_counts=None):
        """在帧上渲染击球数据、落地数据和得分
        
        Args:
            frame (numpy.ndarray): 视频帧
            scores (dict): 得分数据 {'p1': int, 'p2': int}
            hit_counts (dict): 击球数据 {'p1': int, 'p2': int}
            landing_counts (dict, optional): 落地数据
        
        Returns:
            numpy.ndarray: 渲染后的帧
        """
        frame_copy = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        # 准备文本内容
        score_text_p1 = f"Player 1: {scores['p1']}"
        score_text_p2 = f"Player 2: {scores['p2']}"
        hit_text_p1 = f"P1 Hits: {hit_counts['p1']}"
        hit_text_p2 = f"P2 Hits: {hit_counts['p2']}"
        
        # 落地数据文本（如果提供）
        landing_texts = []
        if landing_counts:
            landing_texts = [
                f"Landings: {landing_counts['total']}",
                f"In: {landing_counts['in_bounds']} Out: {landing_counts['out_of_bounds']}"
            ]
        
        # 计算文本尺寸
        score_size_p1, _ = cv2.getTextSize(score_text_p1, self.font, self.font_scale, self.font_thickness)
        score_size_p2, _ = cv2.getTextSize(score_text_p2, self.font, self.font_scale, self.font_thickness)
        hit_size_p1, _ = cv2.getTextSize(hit_text_p1, self.font, self.font_scale, self.font_thickness)
        hit_size_p2, _ = cv2.getTextSize(hit_text_p2, self.font, self.font_scale, self.font_thickness)
        
        # 计算落地文本尺寸
        landing_sizes = []
        if landing_texts:
            for text in landing_texts:
                size, _ = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)
                landing_sizes.append(size)
        
        text_height = max(score_size_p1[1], score_size_p2[1], hit_size_p1[1], hit_size_p2[1])
        max_score_width = max(score_size_p1[0], score_size_p2[0])
        max_hit_width = max(hit_size_p1[0], hit_size_p2[0])
        
        # 右上角位置（得分）
        score_pos_p1 = (frame_width - max_score_width - self.padding, text_height + self.padding)
        score_pos_p2 = (frame_width - max_score_width - self.padding, 2 * text_height + 2 * self.padding)
        
        # 左上角位置（击球数据）
        hit_pos_p1 = (self.padding, text_height + self.padding)
        hit_pos_p2 = (self.padding, 2 * text_height + 2 * self.padding)
        
        # 绘制得分文本（右上角，红色）
        cv2.putText(frame_copy, score_text_p1, score_pos_p1, self.font, self.font_scale, 
                   self.score_font_color, self.font_thickness, self.font_line_type)
        cv2.putText(frame_copy, score_text_p2, score_pos_p2, self.font, self.font_scale, 
                   self.score_font_color, self.font_thickness, self.font_line_type)
        
        # 绘制击球数据文本（左上角，蓝色）
        cv2.putText(frame_copy, hit_text_p1, hit_pos_p1, self.font, self.font_scale, 
                   self.hit_font_color, self.font_thickness, self.font_line_type)
        cv2.putText(frame_copy, hit_text_p2, hit_pos_p2, self.font, self.font_scale, 
                   self.hit_font_color, self.font_thickness, self.font_line_type)
        
        # 绘制落地数据文本（左下角，绿色）
        if landing_texts:
            landing_font_color = (0, 255, 0)  # 绿色用于落地数据
            for i, text in enumerate(landing_texts):
                landing_pos = (self.padding, frame_height - (len(landing_texts) - i) * (text_height + self.padding))
                cv2.putText(frame_copy, text, landing_pos, self.font, self.font_scale, 
                           landing_font_color, self.font_thickness, self.font_line_type)
        
        return frame_copy


class HitAnalyzer:
    """击球分析器 - 整合所有击球相关功能"""
    
    def __init__(self, frame_height=720, frame_width=1280):
        self.detector = HitDetector(frame_height, frame_width)
        self.counter = HitCounter(frame_width)
        self.renderer = HitDisplayRenderer()
        self.hit_detection_enabled = True
        
    def process_frame(self, frame, traj_queue, current_frame, x_pred, y_pred, vis_pred):
        """处理单帧的击球检测、落地检测和统计
        
        Args:
            frame (numpy.ndarray): 当前帧
            traj_queue (deque): 轨迹队列
            current_frame (int): 当前帧数
            x_pred (list): X坐标预测
            y_pred (list): Y坐标预测
            vis_pred (list): 可见性预测
        
        Returns:
            dict: 处理结果 {'hit_info': dict, 'landing_info': dict}
        """
        hit_info = None
        landing_info = None
        
        # 击球检测
        if len(traj_queue) >= 5 and self.hit_detection_enabled:
            hit_result = self.detector.detect_hit_from_trajectory(
                traj_queue, current_frame, self.counter.last_hit_frame
            )
            
            if hit_result['hit_detected']:
                # 将击球添加到缓冲区
                ball_x = traj_queue[0][0] if traj_queue[0] else 0
                self.counter.add_hit_detection(
                    current_frame,
                    hit_result['confidence'],
                    ball_x,
                    hit_result['ball_position'],
                    hit_result['reasons']
                )
        
        # 落地检测
        if len(traj_queue) >= 4 and self.hit_detection_enabled:
            landing_result = self.detector.detect_ball_landing(
                traj_queue, current_frame, self.counter.last_landing_frame
            )
            
            if landing_result['landing_detected']:
                # 将落地添加到缓冲区
                self.counter.add_landing_detection(
                    current_frame,
                    landing_result['confidence'],
                    landing_result['position'],
                    landing_result['reasons']
                )
        
        # 处理击球缓冲区
        confirmed_hits = self.counter.process_hit_buffer(current_frame)
        
        # 处理确认的击球
        for hit in confirmed_hits:
            hit_assignment = self.counter.assign_hit_to_player(hit)
            if hit_assignment:
                hit_info = hit_assignment
                print(f"帧{hit_assignment['frame']}: {hit_assignment['player']}击球 "
                      f"(置信度:{hit_assignment['confidence']:.2f}, "
                      f"位置:{hit_assignment['side']}, 原因:{hit_assignment['reasons']}) "
                      f"总计: P1={self.counter.hit_counts['p1']}, P2={self.counter.hit_counts['p2']}")
        
        # 处理落地缓冲区
        confirmed_landings = self.counter.process_landing_buffer(current_frame)
        
        # 处理确认的落地
        for landing in confirmed_landings:
            landing_record = self.counter.record_landing(landing)
            if landing_record:
                landing_info = landing_record
                print(f"帧{landing_record['frame']}: 球落地 "
                      f"(置信度:{landing_record['confidence']:.2f}, "
                      f"位置:{landing_record['court_area']}, 原因:{landing_record['reasons']}) "
                      f"落地统计: 总计={self.counter.landing_counts['total']}, "
                      f"界内={self.counter.landing_counts['in_bounds']}, "
                      f"界外={self.counter.landing_counts['out_of_bounds']}")
        
        return {'hit_info': hit_info, 'landing_info': landing_info}
    
    def render_frame_with_hit_data(self, frame, scores):
        """渲染带有击球数据和落地数据的帧"""
        hit_counts = self.counter.get_hit_counts()
        landing_counts = self.counter.get_landing_counts()
        return self.renderer.render_hit_data(frame, scores, hit_counts, landing_counts)
    
    def get_hit_statistics(self):
        """获取击球和落地统计信息"""
        hit_counts = self.counter.get_hit_counts()
        landing_counts = self.counter.get_landing_counts()
        
        return {
            'hit_counts': hit_counts,
            'total_hits': sum(hit_counts.values()),
            'last_hit_frame': self.counter.last_hit_frame,
            'last_hit_side': self.counter.last_hit_side,
            'landing_counts': landing_counts,
            'last_landing_frame': self.counter.last_landing_frame
        }