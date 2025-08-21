"""
分析真实击球帧中羽毛球和球拍的距离分布
用于优化击球检测的距离参数
"""
import cv2
import numpy as np
import pandas as pd
import json
import os
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入模块
from utils.unified_detector import UnifiedDetector

def load_real_hit_frames(file_path: str) -> List[int]:
    """加载真实击球帧列表"""
    with open(file_path, 'r') as f:
        frames = [int(line.strip()) for line in f.readlines()]
    return sorted(frames)

def analyze_hit_frame_distances(video_path: str, real_hit_frames: List[int], 
                              output_dir: str = "analysis") -> Dict:
    """
    分析真实击球帧及其前后4帧中羽毛球和球拍的距离分布
    每个GT帧及其前后4帧构成一个GT组，分析组内的"接近-分离"模式
    
    Args:
        video_path: 视频文件路径
        real_hit_frames: 真实击球帧列表
        output_dir: 输出目录
        
    Returns:
        Dict: 分析结果
    """
    logger.info(f"开始分析 {len(real_hit_frames)} 个真实击球帧及其前后4帧...")
    
    # 初始化检测器
    unified_detector = UnifiedDetector()
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 存储分析结果
    analysis_results = []
    gt_groups = []
    
    # 分析每个真实击球帧及其前后4帧
    for frame_id in tqdm(real_hit_frames, desc="分析GT组"):
        # 构建GT组：前后4帧 + GT帧 = 9帧
        gt_group_frames = []
        for offset in range(-4, 5):  # -4, -3, -2, -1, 0, 1, 2, 3, 4
            target_frame = frame_id + offset
            if target_frame > 0:  # 帧ID必须大于0
                gt_group_frames.append(target_frame)
        
        logger.info(f"GT组 {frame_id}: 分析帧 {gt_group_frames}")
        
        # 分析GT组内的每一帧
        group_results = []
        reference_positions = []  # 用于多目标选择的参考位置
        
        for target_frame in gt_group_frames:
            frame_desc = "GT帧" if target_frame == frame_id else f"GT帧{frame_id}的前{target_frame - frame_id:+d}帧"
            result = analyze_single_frame(cap, target_frame, fps, unified_detector, frame_desc, reference_positions)
            if result:
                result['gt_group'] = frame_id
                result['frame_offset'] = target_frame - frame_id
                result['frame_type'] = 'gt' if target_frame == frame_id else 'context'
                group_results.append(result)
                analysis_results.append(result)
                
                # 更新参考位置（用于后续帧的多目标选择）
                if result['distances']:
                    # 提取当前帧的最佳目标位置作为参考
                    if result['shuttlecocks_detected'] > 0:
                        shuttlecock_center = get_target_center(result, 'shuttlecock')
                        if shuttlecock_center:
                            reference_positions.append(shuttlecock_center)
                    
                    if result['rackets_detected'] > 0:
                        racket_center = get_target_center(result, 'racket')
                        if racket_center:
                            reference_positions.append(racket_center)
        
        # 分析GT组内的"接近-分离"模式
        if len(group_results) > 1:
            group_pattern_analysis = analyze_gt_group_pattern(group_results, frame_id)
            gt_groups.append(group_pattern_analysis)
    
    cap.release()
    
    cap.release()
    
    # 输出检测统计
    total_frames = len(analysis_results)
    frames_with_both = sum(1 for r in analysis_results if r['shuttlecocks_detected'] > 0 and r['rackets_detected'] > 0)
    frames_with_shuttlecock = sum(1 for r in analysis_results if r['shuttlecocks_detected'] > 0)
    frames_with_racket = sum(1 for r in analysis_results if r['rackets_detected'] > 0)
    
        # 按帧类型统计
    gt_frames = [r for r in analysis_results if r.get('frame_type') == 'gt']
    context_frames = [r for r in analysis_results if r.get('frame_type') == 'context']
    
    gt_frames_with_both = sum(1 for r in gt_frames if r['shuttlecocks_detected'] > 0 and r['rackets_detected'] > 0)
    context_frames_with_both = sum(1 for r in context_frames if r['shuttlecocks_detected'] > 0 and r['rackets_detected'] > 0)
    
    logger.info(f"检测统计:")
    logger.info(f"  总分析帧数: {total_frames}")
    logger.info(f"  GT帧数: {len(gt_frames)}")
    logger.info(f"  上下文帧数: {len(context_frames)}")
    logger.info(f"  同时检测到羽毛球和球拍的帧数: {frames_with_both}")
    logger.info(f"  GT帧中同时检测到的帧数: {gt_frames_with_both}")
    logger.info(f"  上下文帧中同时检测到的帧数: {context_frames_with_both}")
    logger.info(f"  检测到羽毛球的帧数: {frames_with_shuttlecock}")
    logger.info(f"  检测到球拍的帧数: {frames_with_racket}")
    
    # 统计分析
    statistics = analyze_distance_statistics(analysis_results)
    
    # 检查统计结果
    if not statistics:
        logger.error("统计分析失败，没有生成统计结果")
        return {
            'frame_analysis': analysis_results,
            'statistics': {},
            'gt_groups': gt_groups
        }
    
    # 保存结果
    save_analysis_results(analysis_results, statistics, output_dir, gt_groups)
    
    # 生成可视化图表
    generate_distance_plots(analysis_results, statistics, output_dir)
    
    return {
        'frame_analysis': analysis_results,
        'statistics': statistics,
        'gt_groups': gt_groups
    }

def analyze_single_frame(cap, frame_id: int, fps: int, unified_detector, frame_desc: str, 
                        reference_positions: List[Tuple] = None) -> Optional[Dict]:
    """分析单个帧"""
    try:
        # 跳转到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"无法读取帧 {frame_id}")
            return None
        
        # 检测羽毛球和球拍
        # 尝试不同的置信度阈值
        detections = unified_detector.detect_all(frame, confidence_threshold=0.1)
        
        # 记录检测结果
        shuttlecocks_detected = len(detections['shuttlecocks'])
        rackets_detected = len(detections['rackets'])
        
        logger.info(f"帧 {frame_id} ({frame_desc}): 检测到 {shuttlecocks_detected} 个羽毛球, {rackets_detected} 个球拍")
        
        # 如果检测失败，尝试更低的阈值
        if shuttlecocks_detected == 0 and rackets_detected == 0:
            logger.info(f"帧 {frame_id} ({frame_desc}): 尝试更低阈值检测...")
            detections = unified_detector.detect_all(frame, confidence_threshold=0.05)
            shuttlecocks_detected = len(detections['shuttlecocks'])
            rackets_detected = len(detections['rackets'])
            logger.info(f"帧 {frame_id} ({frame_desc}): 低阈值检测结果 - 羽毛球: {shuttlecocks_detected}, 球拍: {rackets_detected}")
        
        # 多目标选择：选择最佳目标
        if shuttlecocks_detected > 1 or rackets_detected > 1:
            logger.info(f"帧 {frame_id}: 多目标检测，进行智能选择...")
            selected_detections = select_best_targets(detections, reference_positions)
            detections = selected_detections
            logger.info(f"帧 {frame_id}: 选择后 - 羽毛球: {len(detections['shuttlecocks'])}, 球拍: {len(detections['rackets'])}")
        
        # 计算距离
        distances = calculate_distances(detections)
        
        # 记录结果
        result = {
            'frame_id': frame_id,
            'timestamp': frame_id / fps,
            'distances': distances,
            'min_distance': min(distances) if distances else None,
            'max_distance': max(distances) if distances else None,
            'avg_distance': np.mean(distances) if distances else None,
            'shuttlecocks_detected': len(detections['shuttlecocks']),
            'rackets_detected': len(detections['rackets']),
            'original_detections': {
                'shuttlecocks': shuttlecocks_detected,
                'rackets': rackets_detected
            }
        }
        
        return result
        
    except Exception as e:
        logger.warning(f"分析帧 {frame_id} ({frame_desc}) 失败: {e}")
        return None

def select_best_targets(detections: Dict, reference_positions: List[Tuple] = None) -> Dict:
    """
    从多个检测目标中选择最佳目标
    多拍选离球最近的，多球根据前后帧位置选择最近的
    
    Args:
        detections: 检测结果
        reference_positions: 参考位置（前后帧的目标位置）
    
    Returns:
        选择后的最佳目标
    """
    selected = {'shuttlecocks': [], 'rackets': []}
    
    # 选择最佳羽毛球
    if detections['shuttlecocks']:
        if len(detections['shuttlecocks']) == 1:
            selected['shuttlecocks'] = detections['shuttlecocks']
        else:
            # 多球情况：根据前后帧位置选择最近的
            if reference_positions:
                best_shuttlecock = select_closest_to_reference(
                    detections['shuttlecocks'], reference_positions
                )
                selected['shuttlecocks'] = [best_shuttlecock]
            else:
                # 没有参考位置，选择第一个
                selected['shuttlecocks'] = [detections['shuttlecocks'][0]]
    
    # 选择最佳球拍
    if detections['rackets']:
        if len(detections['rackets']) == 1:
            selected['rackets'] = [detections['rackets'][0]]
        else:
            # 多拍情况：选择离球最近的
            if selected['shuttlecocks']:
                best_racket = select_closest_racket_to_shuttlecock(
                    detections['rackets'], selected['shuttlecocks'][0]
                )
                selected['rackets'] = [best_racket]
            else:
                # 没有球，选择第一个
                selected['rackets'] = [detections['rackets'][0]]
    
    return selected

def select_closest_to_reference(targets: List, reference_positions: List[Tuple]) -> Tuple:
    """选择离参考位置最近的目标"""
    if not reference_positions:
        return targets[0]
    
    min_distance = float('inf')
    best_target = targets[0]
    
    for target in targets:
        target_center = (target[0] + target[2]) / 2, (target[1] + target[3]) / 2
        
        for ref_pos in reference_positions:
            distance = np.sqrt((target_center[0] - ref_pos[0])**2 + 
                             (target_center[1] - ref_pos[1])**2)
            if distance < min_distance:
                min_distance = distance
                best_target = target
    
    return best_target

def select_closest_racket_to_shuttlecock(rackets: List, shuttlecock: Tuple) -> Tuple:
    """选择离羽毛球最近的球拍"""
    shuttlecock_center = (shuttlecock[0] + shuttlecock[2]) / 2, (shuttlecock[1] + shuttlecock[3]) / 2
    
    min_distance = float('inf')
    best_racket = rackets[0]
    
    for racket in rackets:
        racket_center = (racket[0] + racket[2]) / 2, (racket[1] + racket[3]) / 2
        distance = np.sqrt((shuttlecock_center[0] - racket_center[0])**2 + 
                          (shuttlecock_center[1] - racket_center[1])**2)
        if distance < min_distance:
            min_distance = distance
            best_racket = racket
    
    return best_racket

def get_target_center(result: Dict, target_type: str) -> Optional[Tuple[int, int]]:
    """获取目标中心位置"""
    if target_type == 'shuttlecock' and result['shuttlecocks_detected'] > 0:
        # 这里需要从原始检测结果中获取位置
        # 由于我们简化了数据结构，这里返回一个默认位置
        return None
    elif target_type == 'racket' and result['rackets_detected'] > 0:
        # 这里需要从原始检测结果中获取位置
        # 由于我们简化了数据结构，这里返回一个默认位置
        return None
    return None

def calculate_distances(detections: Dict) -> List[float]:
    """计算羽毛球和球拍之间的距离"""
    distances = []
    
    shuttlecocks = detections['shuttlecocks']
    rackets = detections['rackets']
    
    for shuttlecock in shuttlecocks:
        sc_center = (shuttlecock[0] + shuttlecock[2]) / 2, (shuttlecock[1] + shuttlecock[3]) / 2
        
        for racket in rackets:
            racket_center = (racket[0] + racket[2]) / 2, (racket[1] + racket[3]) / 2
            
            # 计算欧几里得距离
            distance = np.sqrt((sc_center[0] - racket_center[0])**2 + 
                             (sc_center[1] - racket_center[1])**2)
            distances.append(distance)
    
    return distances

def analyze_gt_group_pattern(group_results: List[Dict], gt_frame_id: int) -> Dict:
    """
    分析GT组内的"接近-分离"模式
    
    Args:
        group_results: GT组内所有帧的分析结果
        gt_frame_id: GT帧ID
    
    Returns:
        GT组模式分析结果
    """
    # 按帧偏移排序
    sorted_results = sorted(group_results, key=lambda x: x['frame_offset'])
    
    # 提取距离数据
    frame_distances = []
    for result in sorted_results:
        if result['distances']:
            min_dist = min(result['distances'])
            frame_distances.append({
                'frame_id': result['frame_id'],
                'frame_offset': result['frame_offset'],
                'min_distance': min_dist,
                'timestamp': result['timestamp']
            })
    
    if len(frame_distances) < 2:
        return {
            'gt_frame_id': gt_frame_id,
            'pattern_detected': False,
            'reason': '帧数不足',
            'frame_distances': frame_distances
        }
    
    # 分析"接近-分离"模式
    pattern_analysis = analyze_approach_separation_pattern(frame_distances)
    
    return {
        'gt_frame_id': gt_frame_id,
        'pattern_detected': pattern_analysis['pattern_detected'],
        'pattern_type': pattern_analysis['pattern_type'],
        'confidence': pattern_analysis['confidence'],
        'approach_frame': pattern_analysis['approach_frame'],
        'separation_frame': pattern_analysis['separation_frame'],
        'frame_distances': frame_distances,
        'pattern_details': pattern_analysis
    }

def analyze_approach_separation_pattern(frame_distances: List[Dict]) -> Dict:
    """
    分析"接近-分离"模式
    
    Args:
        frame_distances: 帧距离数据列表
    
    Returns:
        模式分析结果
    """
    if len(frame_distances) < 3:
        return {
            'pattern_detected': False,
            'pattern_type': 'insufficient_data',
            'reason': '数据不足',
            'confidence': 0.0,
            'approach_frame': None,
            'separation_frame': None,
            'min_distance': None,
            'min_distance_offset': None,
            'approach_threshold': 150.0,
            'separation_threshold': 300.0
        }
    
    # 寻找距离最小值（接近点）
    min_distance_frame = min(frame_distances, key=lambda x: x['min_distance'])
    min_distance = min_distance_frame['min_distance']
    min_distance_offset = min_distance_frame['frame_offset']
    
    # 定义接近和分离的阈值
    approach_threshold = 150.0  # 像素
    separation_threshold = 300.0  # 像素
    
    # 检查是否在GT帧附近有接近
    approach_detected = min_distance < approach_threshold
    
    # 检查是否有分离
    separation_detected = False
    separation_frame = None
    
    for frame_data in frame_distances:
        if frame_data['min_distance'] > separation_threshold:
            separation_detected = True
            separation_frame = frame_data
            break
    
    # 计算模式置信度
    confidence = 0.0
    pattern_type = "none"
    
    if approach_detected and separation_detected:
        # 完整的"接近-分离"模式
        pattern_type = "approach_separation"
        
        # 计算置信度：基于接近程度和分离程度
        approach_score = max(0, 1 - min_distance / approach_threshold)
        separation_score = min(1, separation_frame['min_distance'] / separation_threshold)
        confidence = (approach_score + separation_score) / 2
        
    elif approach_detected:
        # 只有接近，没有分离
        pattern_type = "approach_only"
        confidence = max(0, 1 - min_distance / approach_threshold)
        
    elif separation_detected:
        # 只有分离，没有接近
        pattern_type = "separation_only"
        confidence = min(1, separation_frame['min_distance'] / separation_threshold)
    
    return {
        'pattern_detected': bool(approach_detected or separation_detected),
        'pattern_type': pattern_type,
        'confidence': float(confidence),
        'approach_frame': min_distance_frame if approach_detected else None,
        'separation_frame': separation_frame,
        'min_distance': float(min_distance) if min_distance is not None else None,
        'min_distance_offset': int(min_distance_offset) if min_distance_offset is not None else None,
        'approach_threshold': float(approach_threshold),
        'separation_threshold': float(separation_threshold)
    }

def filter_outliers(distances: List[float], method: str = 'iqr') -> List[float]:
    """过滤异常值
    
    Args:
        distances: 距离列表
        method: 过滤方法 ('iqr' 或 'zscore')
    
    Returns:
        过滤后的距离列表
    """
    if not distances or len(distances) < 3:
        return distances
    
    if method == 'iqr':
        # IQR方法：移除超出Q1-1.5*IQR和Q3+1.5*IQR范围的值
        q1 = np.percentile(distances, 25)
        q3 = np.percentile(distances, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        filtered_distances = [d for d in distances if lower_bound <= d <= upper_bound]
        
        logger.info(f"IQR过滤: 原始距离数 {len(distances)}, 过滤后 {len(filtered_distances)}")
        logger.info(f"过滤边界: [{lower_bound:.1f}, {upper_bound:.1f}]")
        logger.info(f"被过滤的异常值: {[d for d in distances if d < lower_bound or d > upper_bound]}")
        
        return filtered_distances
    
    elif method == 'zscore':
        # Z-score方法：移除Z-score绝对值大于2的值
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        if std_dist == 0:
            return distances
        
        z_scores = [(d - mean_dist) / std_dist for d in distances]
        filtered_distances = [d for d, z in zip(distances, z_scores) if abs(z) <= 2]
        
        logger.info(f"Z-score过滤: 原始距离数 {len(distances)}, 过滤后 {len(filtered_distances)}")
        logger.info(f"被过滤的异常值: {[d for d, z in zip(distances, z_scores) if abs(z) > 2]}")
        
        return filtered_distances
    
    return distances

def analyze_distance_statistics(analysis_results: List[Dict]) -> Dict:
    """分析距离统计信息"""
    if not analysis_results:
        return {}
    
    # 按帧类型分组
    gt_frames = [r for r in analysis_results if r.get('frame_type') == 'gt']
    context_frames = [r for r in analysis_results if r.get('frame_type') == 'context']
    
    # 提取所有距离数据
    all_distances = []
    min_distances = []
    max_distances = []
    avg_distances = []
    
    # GT帧的距离数据
    gt_distances = []
    gt_min_distances = []
    
    # 上下文帧的距离数据
    context_distances = []
    context_min_distances = []
    
    for result in analysis_results:
        if result['distances']:
            all_distances.extend(result['distances'])
            min_distances.append(result['min_distance'])
            max_distances.append(result['max_distance'])
            avg_distances.append(result['avg_distance'])
            
            # 按帧类型分类
            if result.get('frame_type') == 'gt':
                gt_distances.extend(result['distances'])
                gt_min_distances.append(result['min_distance'])
            elif result.get('frame_type') == 'context':
                context_distances.extend(result['distances'])
                context_min_distances.append(result['min_distance'])
    
    if not all_distances:
        return {}
    
    # 过滤异常值
    logger.info("开始过滤异常距离值...")
    filtered_all_distances = filter_outliers(all_distances, method='iqr')
    filtered_min_distances = filter_outliers(min_distances, method='iqr')
    filtered_gt_distances = filter_outliers(gt_distances, method='iqr') if gt_distances else []
    filtered_context_distances = filter_outliers(context_distances, method='iqr') if context_distances else []
    
    # 计算统计量（使用过滤后的数据）
    statistics = {
        'total_frames_analyzed': len(analysis_results),
        'frames_with_detections': len([r for r in analysis_results if r['distances']]),
        'total_distance_measurements': len(all_distances),
        'filtered_distance_measurements': len(filtered_all_distances),
        'outliers_removed': len(all_distances) - len(filtered_all_distances),
        
        # 总体距离统计（过滤后）
        'distance_stats': {
            'min': float(np.min(filtered_all_distances)) if filtered_all_distances else 0,
            'max': float(np.max(filtered_all_distances)) if filtered_all_distances else 0,
            'mean': float(np.mean(filtered_all_distances)) if filtered_all_distances else 0,
            'median': float(np.median(filtered_all_distances)) if filtered_all_distances else 0,
            'std': float(np.std(filtered_all_distances)) if filtered_all_distances else 0,
            'q25': float(np.percentile(filtered_all_distances, 25)) if filtered_all_distances else 0,
            'q75': float(np.percentile(filtered_all_distances, 75)) if filtered_all_distances else 0
        },
        
        # 总体最小距离统计（过滤后）
        'min_distance_stats': {
            'min': float(np.min(filtered_min_distances)) if filtered_min_distances else 0,
            'max': float(np.max(filtered_min_distances)) if filtered_min_distances else 0,
            'mean': float(np.mean(filtered_min_distances)) if filtered_min_distances else 0,
            'median': float(np.median(filtered_min_distances)) if filtered_min_distances else 0,
            'std': float(np.std(filtered_min_distances)) if filtered_min_distances else 0,
            'q25': float(np.percentile(filtered_min_distances, 25)) if filtered_min_distances else 0,
            'q75': float(np.percentile(filtered_min_distances, 75)) if filtered_min_distances else 0
        },
        
        # 总体平均距离统计
        'avg_distance_stats': {
            'min': float(np.min(avg_distances)) if avg_distances else 0,
            'max': float(np.max(avg_distances)) if avg_distances else 0,
            'mean': float(np.mean(avg_distances)) if avg_distances else 0,
            'median': float(np.median(avg_distances)) if avg_distances else 0,
            'std': float(np.std(avg_distances)) if avg_distances else 0
        },
        
        # GT帧距离统计（过滤后）
        'gt_frame_distance_stats': {
            'min': float(np.min(filtered_gt_distances)) if filtered_gt_distances else 0,
            'max': float(np.max(filtered_gt_distances)) if filtered_gt_distances else 0,
            'mean': float(np.mean(filtered_gt_distances)) if filtered_gt_distances else 0,
            'median': float(np.median(filtered_gt_distances)) if filtered_gt_distances else 0,
            'std': float(np.std(filtered_gt_distances)) if filtered_gt_distances else 0,
            'q25': float(np.percentile(filtered_gt_distances, 25)) if filtered_gt_distances else 0,
            'q75': float(np.percentile(filtered_gt_distances, 75)) if filtered_gt_distances else 0
        },
        
        # 上下文帧距离统计（过滤后）
        'context_frame_distance_stats': {
            'min': float(np.min(filtered_context_distances)) if filtered_context_distances else 0,
            'max': float(np.max(filtered_context_distances)) if filtered_context_distances else 0,
            'mean': float(np.mean(filtered_context_distances)) if filtered_context_distances else 0,
            'median': float(np.median(filtered_context_distances)) if filtered_context_distances else 0,
            'std': float(np.std(filtered_context_distances)) if filtered_context_distances else 0,
            'q25': float(np.percentile(filtered_context_distances, 25)) if filtered_context_distances else 0,
            'q75': float(np.percentile(filtered_context_distances, 75)) if filtered_context_distances else 0
        },
        
        # 检测统计
        'detection_stats': {
            'avg_shuttlecocks_per_frame': float(np.mean([r['shuttlecocks_detected'] for r in analysis_results])),
            'avg_rackets_per_frame': float(np.mean([r['rackets_detected'] for r in analysis_results])),
            'frames_with_shuttlecock': sum(1 for r in analysis_results if r['shuttlecocks_detected'] > 0),
            'frames_with_racket': sum(1 for r in analysis_results if r['rackets_detected'] > 0)
        },
        
        # 帧类型统计
        'frame_type_stats': {
            'gt_frames_total': len(gt_frames),
            'context_frames_total': len(context_frames),
            'gt_frames_with_distances': len([r for r in gt_frames if r['distances']]),
            'context_frames_with_distances': len([r for r in context_frames if r['distances']])
        }
    }
    
    return statistics

def save_analysis_results(analysis_results: List[Dict], statistics: Dict, output_dir: str, gt_groups: List[Dict] = None):
    """保存分析结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细分析结果
    results_file = os.path.join(output_dir, "real_hit_analysis.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'frame_analysis': analysis_results,
            'statistics': statistics,
            'gt_groups': gt_groups or []
        }, f, ensure_ascii=False, indent=2)
    
    # 保存统计摘要
    summary_file = os.path.join(output_dir, "distance_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("真实击球帧距离分析摘要\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"分析帧数: {statistics['total_frames_analyzed']}\n")
        f.write(f"有效检测帧数: {statistics['frames_with_detections']}\n")
        f.write(f"总距离测量数: {statistics['total_distance_measurements']}\n")
        f.write(f"过滤后距离测量数: {statistics['filtered_distance_measurements']}\n")
        f.write(f"异常值移除数: {statistics['outliers_removed']}\n\n")
        
        f.write("距离统计 (像素):\n")
        f.write(f"  最小值: {statistics['distance_stats']['min']:.2f}\n")
        f.write(f"  最大值: {statistics['distance_stats']['max']:.2f}\n")
        f.write(f"  平均值: {statistics['distance_stats']['mean']:.2f}\n")
        f.write(f"  中位数: {statistics['distance_stats']['median']:.2f}\n")
        f.write(f"  标准差: {statistics['distance_stats']['std']:.2f}\n")
        f.write(f"  25%分位数: {statistics['distance_stats']['q25']:.2f}\n")
        f.write(f"  75%分位数: {statistics['distance_stats']['q75']:.2f}\n\n")
        
        f.write("最小距离统计 (每帧最接近的距离):\n")
        f.write(f"  最小值: {statistics['min_distance_stats']['min']:.2f}\n")
        f.write(f"  最大值: {statistics['min_distance_stats']['max']:.2f}\n")
        f.write(f"  平均值: {statistics['min_distance_stats']['mean']:.2f}\n")
        f.write(f"  中位数: {statistics['min_distance_stats']['median']:.2f}\n\n")
        
        f.write("检测统计:\n")
        f.write(f"  平均每帧羽毛球数: {statistics['detection_stats']['avg_shuttlecocks_per_frame']:.2f}\n")
        f.write(f"  平均每帧球拍数: {statistics['detection_stats']['avg_rackets_per_frame']:.2f}\n")
        f.write(f"  检测到羽毛球的帧数: {statistics['detection_stats']['frames_with_shuttlecock']}\n")
        f.write(f"  检测到球拍的帧数: {statistics['detection_stats']['frames_with_racket']}\n\n")
        
        f.write("建议的距离参数:\n")
        f.write(f"  接近距离阈值: {statistics['min_distance_stats']['q75']:.1f} (75%分位数)\n")
        f.write(f"  分离距离阈值: {statistics['distance_stats']['q75']:.1f} (75%分位数)\n\n")
        
        f.write("GT帧距离统计:\n")
        f.write(f"  最小值: {statistics['gt_frame_distance_stats']['min']:.2f}\n")
        f.write(f"  最大值: {statistics['gt_frame_distance_stats']['max']:.2f}\n")
        f.write(f"  平均值: {statistics['gt_frame_distance_stats']['mean']:.2f}\n")
        f.write(f"  中位数: {statistics['gt_frame_distance_stats']['median']:.2f}\n")
        f.write(f"  75%分位数: {statistics['gt_frame_distance_stats']['q75']:.2f}\n\n")
        
        f.write("上下文帧距离统计:\n")
        f.write(f"  最小值: {statistics['context_frame_distance_stats']['min']:.2f}\n")
        f.write(f"  最大值: {statistics['context_frame_distance_stats']['max']:.2f}\n")
        f.write(f"  平均值: {statistics['context_frame_distance_stats']['mean']:.2f}\n")
        f.write(f"  中位数: {statistics['context_frame_distance_stats']['median']:.2f}\n")
        f.write(f"  75%分位数: {statistics['context_frame_distance_stats']['q75']:.2f}\n\n")
        
        f.write("优化建议:\n")
        f.write(f"  基于GT帧数据 - 接近阈值: {statistics['gt_frame_distance_stats']['q75']:.1f}\n")
        f.write(f"  基于上下文帧数据 - 接近阈值: {statistics['context_frame_distance_stats']['q75']:.1f}\n")
        f.write(f"  综合建议 - 接近阈值: {min(statistics['gt_frame_distance_stats']['q75'], statistics['context_frame_distance_stats']['q75']):.1f}\n\n")
        
        # 添加GT组模式分析结果
        if gt_groups:
            f.write("GT组模式分析结果:\n")
            f.write("=" * 30 + "\n")
            for group in gt_groups:
                f.write(f"GT帧 {group['gt_frame_id']}:\n")
                f.write(f"  模式检测: {'是' if group['pattern_detected'] else '否'}\n")
                if group['pattern_detected']:
                    f.write(f"  模式类型: {group['pattern_type']}\n")
                    f.write(f"  置信度: {group['confidence']:.3f}\n")
                    if group['approach_frame']:
                        f.write(f"  接近帧: {group['approach_frame']['frame_id']} (偏移: {group['approach_frame']['frame_offset']:+d})\n")
                    if group['separation_frame']:
                        f.write(f"  分离帧: {group['separation_frame']['frame_id']} (偏移: {group['separation_frame']['frame_offset']:+d})\n")
                else:
                    f.write(f"  原因: {group.get('reason', '未知')}\n")
                f.write("\n")
    
    logger.info(f"分析结果已保存到: {output_dir}")

def generate_distance_plots(analysis_results: List[Dict], statistics: Dict, output_dir: str):
    """生成距离分布图表"""
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 提取距离数据
        all_distances = []
        min_distances = []
        
        for result in analysis_results:
            if result['distances']:
                all_distances.extend(result['distances'])
                min_distances.append(result['min_distance'])
        
        if not all_distances:
            logger.warning("没有距离数据，跳过图表生成")
            return
        
        # 过滤异常值用于图表显示
        filtered_all_distances = filter_outliers(all_distances, method='iqr')
        filtered_min_distances = filter_outliers(min_distances, method='iqr')
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('真实击球帧距离分布分析 (异常值已过滤)', fontsize=16)
        
        # 1. 过滤前后的距离对比直方图
        axes[0, 0].hist(all_distances, bins=30, alpha=0.3, color='red', edgecolor='black', label='原始数据')
        axes[0, 0].hist(filtered_all_distances, bins=30, alpha=0.7, color='skyblue', edgecolor='black', label='过滤后数据')
        axes[0, 0].axvline(statistics['distance_stats']['mean'], color='red', linestyle='--', 
                           label=f'过滤后平均值: {statistics["distance_stats"]["mean"]:.1f}')
        axes[0, 0].axvline(statistics['distance_stats']['median'], color='orange', linestyle='--', 
                           label=f'过滤后中位数: {statistics["distance_stats"]["median"]:.1f}')
        axes[0, 0].set_xlabel('距离 (像素)')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].set_title('距离分布对比 (原始 vs 过滤后)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 过滤前后的最小距离对比直方图
        axes[0, 1].hist(min_distances, bins=20, alpha=0.3, color='red', edgecolor='black', label='原始数据')
        axes[0, 1].hist(filtered_min_distances, bins=20, alpha=0.7, color='lightgreen', edgecolor='black', label='过滤后数据')
        axes[0, 1].axvline(statistics['min_distance_stats']['mean'], color='red', linestyle='--', 
                           label=f'过滤后平均值: {statistics["min_distance_stats"]["mean"]:.1f}')
        axes[0, 1].set_xlabel('距离 (像素)')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].set_title('最小距离分布对比 (原始 vs 过滤后)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 距离箱线图
        axes[1, 0].boxplot([min_distances], labels=['最小距离'])
        axes[1, 0].set_ylabel('距离 (像素)')
        axes[1, 0].set_title('距离箱线图')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 帧ID vs 最小距离的散点图 (显示异常值)
        frame_ids = [r['frame_id'] for r in analysis_results if r['min_distance'] is not None]
        min_dists = [r['min_distance'] for r in analysis_results if r['min_distance'] is not None]
        
        # 分离正常值和异常值
        normal_frame_ids = []
        normal_min_dists = []
        outlier_frame_ids = []
        outlier_min_dists = []
        
        for frame_id, min_dist in zip(frame_ids, min_dists):
            if min_dist in filtered_min_distances:
                normal_frame_ids.append(frame_id)
                normal_min_dists.append(min_dist)
            else:
                outlier_frame_ids.append(frame_id)
                outlier_min_dists.append(min_dist)
        
        # 绘制正常值
        if normal_frame_ids:
            axes[1, 1].scatter(normal_frame_ids, normal_min_dists, alpha=0.7, color='blue', label='正常值')
        
        # 绘制异常值
        if outlier_frame_ids:
            axes[1, 1].scatter(outlier_frame_ids, outlier_min_dists, alpha=0.7, color='red', marker='x', s=100, label='异常值')
        
        axes[1, 1].axhline(statistics['min_distance_stats']['mean'], color='red', linestyle='--', 
                           label=f'过滤后平均值: {statistics["min_distance_stats"]["mean"]:.1f}')
        axes[1, 1].set_xlabel('帧ID')
        axes[1, 1].set_ylabel('最小距离 (像素)')
        axes[1, 1].set_title('帧ID vs 最小距离 (异常值标记)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = os.path.join(output_dir, "distance_analysis_plots.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"距离分布图表已保存: {plot_file}")
        
    except Exception as e:
        logger.warning(f"生成图表失败: {e}")

def generate_gt_group_reports(analysis_results: Dict, output_dir: str = "analysis") -> None:
    """为每个GT组生成详细的测试结果报告"""
    
    gt_groups = analysis_results.get('gt_groups', [])
    if not gt_groups:
        print("没有找到GT组数据，无法生成报告")
        return
    
    # 创建报告文件
    report_file = os.path.join(output_dir, "gt_group_test_reports.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("羽毛球击球检测 - GT组测试结果详细报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 总体统计
        total_gt_groups = len(gt_groups)
        pattern_detected_count = sum(1 for group in gt_groups if group.get('pattern_detected', False))
        pattern_not_detected_count = total_gt_groups - pattern_detected_count
        
        f.write(f"总体统计:\n")
        f.write(f"- 总GT组数量: {total_gt_groups}\n")
        f.write(f"- 检测到模式的组数: {pattern_detected_count}\n")
        f.write(f"- 未检测到模式的组数: {pattern_not_detected_count}\n")
        f.write(f"- 模式检测成功率: {pattern_detected_count/total_gt_groups*100:.1f}%\n\n")
        
        # 模式类型统计
        pattern_types = {}
        for group in gt_groups:
            pattern_type = group.get('pattern_type', 'unknown')
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
        
        f.write("模式类型分布:\n")
        for pattern_type, count in pattern_types.items():
            f.write(f"- {pattern_type}: {count} 组 ({count/total_gt_groups*100:.1f}%)\n")
        f.write("\n")
        
        # 每个GT组的详细报告
        f.write("=" * 80 + "\n")
        f.write("各GT组详细测试结果\n")
        f.write("=" * 80 + "\n\n")
        
        for i, group in enumerate(gt_groups, 1):
            gt_frame_id = group.get('gt_frame_id', 'unknown')
            pattern_detected = group.get('pattern_detected', False)
            pattern_type = group.get('pattern_type', 'unknown')
            confidence = group.get('confidence', 0.0)
            
            f.write(f"GT组 {i}: 帧 {gt_frame_id}\n")
            f.write("-" * 50 + "\n")
            f.write(f"模式检测结果: {'✓ 检测到' if pattern_detected else '✗ 未检测到'}\n")
            f.write(f"模式类型: {pattern_type}\n")
            f.write(f"置信度: {confidence:.3f}\n")
            
            # 接近帧信息
            approach_frame = group.get('approach_frame')
            if approach_frame:
                f.write(f"接近帧: 帧 {approach_frame['frame_id']} (偏移: {approach_frame['frame_offset']:>2}), "
                       f"距离: {approach_frame['min_distance']:.2f}, "
                       f"时间: {approach_frame['timestamp']:.3f}s\n")
            else:
                f.write("接近帧: 无\n")
            
            # 分离帧信息
            separation_frame = group.get('separation_frame')
            if separation_frame:
                f.write(f"分离帧: 帧 {separation_frame['frame_id']} (偏移: {separation_frame['frame_offset']:>2}), "
                       f"距离: {separation_frame['min_distance']:.2f}, "
                       f"时间: {separation_frame['timestamp']:.3f}s\n")
            else:
                f.write("分离帧: 无\n")
            
            # 帧距离详情
            frame_distances = group.get('frame_distances', [])
            if frame_distances:
                f.write("帧距离序列:\n")
                for fd in frame_distances:
                    frame_id = fd['frame_id']
                    frame_offset = fd['frame_offset']
                    min_distance = fd['min_distance']
                    timestamp = fd['timestamp']
                    
                    # 添加标记
                    if frame_offset == 0:
                        marker = " [GT]"
                    elif frame_offset < 0:
                        marker = f" [-{abs(frame_offset)}]"
                    else:
                        marker = f" [+{frame_offset}]"
                    
                    f.write(f"  帧 {frame_id:>3}{marker:>6}: 距离 {min_distance:>8.2f}, "
                           f"时间 {timestamp:>6.3f}s\n")
            
            # 模式详情
            pattern_details = group.get('pattern_details', {})
            if pattern_details:
                reason = pattern_details.get('reason', '')
                if reason:
                    f.write(f"检测原因: {reason}\n")
                
                approach_threshold = pattern_details.get('approach_threshold')
                separation_threshold = pattern_details.get('separation_threshold')
                if approach_threshold and separation_threshold:
                    f.write(f"距离阈值: 接近 {approach_threshold:.1f}, 分离 {separation_threshold:.1f}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"GT组测试结果报告已保存到: {report_file}")
    
    # 生成简化的统计报告
    summary_file = os.path.join(output_dir, "gt_group_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("GT组测试结果统计摘要\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"总GT组数: {total_gt_groups}\n")
        f.write(f"成功检测: {pattern_detected_count}\n")
        f.write(f"检测失败: {pattern_not_detected_count}\n")
        f.write(f"成功率: {pattern_detected_count/total_gt_groups*100:.1f}%\n\n")
        
        f.write("模式类型分布:\n")
        for pattern_type, count in sorted(pattern_types.items()):
            f.write(f"  {pattern_type}: {count}\n")
        
        # 置信度统计
        confidences = [group.get('confidence', 0.0) for group in gt_groups if group.get('pattern_detected', False)]
        if confidences:
            f.write(f"\n检测成功的置信度统计:\n")
            f.write(f"  平均置信度: {np.mean(confidences):.3f}\n")
            f.write(f"  最高置信度: {np.max(confidences):.3f}\n")
            f.write(f"  最低置信度: {np.min(confidences):.3f}\n")
            f.write(f"  标准差: {np.std(confidences):.3f}\n")
    
    print(f"GT组统计摘要已保存到: {summary_file}")

def main():
    """主函数"""
    # 文件路径
    video_path = "input/bd_7.mp4"
    real_hit_frames_file = "input/bd_7.txt"
    output_dir = "analysis"
    
    # 检查文件是否存在
    if not os.path.exists(video_path):
        logger.error(f"视频文件不存在: {video_path}")
        return
    
    if not os.path.exists(real_hit_frames_file):
        logger.error(f"真实击球帧文件不存在: {real_hit_frames_file}")
        return
    
    try:
        # 加载真实击球帧
        real_hit_frames = load_real_hit_frames(real_hit_frames_file)
        logger.info(f"加载了 {len(real_hit_frames)} 个真实击球帧")
        
        # 分析距离分布
        results = analyze_hit_frame_distances(video_path, real_hit_frames, output_dir)
        
        # 输出建议
        stats = results['statistics']
        gt_groups = results.get('gt_groups', [])
        
        if stats:
            logger.info("=" * 60)
            logger.info("距离分析完成！建议的距离参数:")
            logger.info(f"异常值过滤: 移除了 {stats['outliers_removed']} 个异常距离值")
            logger.info(f"过滤后数据: {stats['filtered_distance_measurements']} 个有效距离值")
            logger.info("")
            logger.info("基于过滤后数据的建议:")
            logger.info(f"总体接近距离阈值: {stats['min_distance_stats']['q75']:.1f} 像素")
            logger.info(f"总体分离距离阈值: {stats['distance_stats']['q75']:.1f} 像素")
            logger.info("")
            logger.info("按帧类型分析:")
            logger.info(f"GT帧接近阈值: {stats['gt_frame_distance_stats']['q75']:.1f} 像素")
            logger.info(f"上下文帧接近阈值: {stats['context_frame_distance_stats']['q75']:.1f} 像素")
            logger.info("")
            logger.info("综合优化建议:")
            logger.info(f"推荐接近阈值: {min(stats['gt_frame_distance_stats']['q75'], stats['context_frame_distance_stats']['q75']):.1f} 像素")
            logger.info("=" * 60)
        
        # 输出GT组模式分析结果
        if gt_groups:
            logger.info("=" * 60)
            logger.info("GT组模式分析结果:")
            pattern_detected_count = sum(1 for g in gt_groups if g['pattern_detected'])
            logger.info(f"总GT组数: {len(gt_groups)}")
            logger.info(f"检测到模式组数: {pattern_detected_count}")
            logger.info(f"模式检测率: {pattern_detected_count/len(gt_groups)*100:.1f}%")
            logger.info("=" * 60)
            
            # 生成GT组测试结果报告
            generate_gt_group_reports(results, output_dir)
        
        print("✅ 真实击球帧距离分析完成！")
        print(f"结果保存在: {output_dir}/")
        
    except Exception as e:
        logger.error(f"分析失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
