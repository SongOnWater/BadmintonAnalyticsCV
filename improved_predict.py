"""
优化的改进羽毛球预测脚本
复用原有的羽毛球检测机制，只添加羽毛球拍检测和击球识别
"""
import cv2
import numpy as np
import torch
import pandas as pd
import json
import os
import argparse
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import logging
import time

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入原有模块
from predict import pred_main, get_models, predict_location
from utils.improved_hit_detection import ImprovedHitDetection
from utils.unified_detector import UnifiedDetector
from utils.general import *

# 常量定义
HEIGHT = 360
WIDTH = 640

def generate_frames_from_video(video_path: str) -> Tuple[List[np.ndarray], int, Tuple[int, int]]:
    """从视频生成帧列表"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames, fps, (width, height)

def enhanced_pred_main(video_path: str, model_path: str, output_dir: str = "output") -> Dict[str, Any]:
    """
    增强的预测主函数，复用原有羽毛球检测，添加羽毛球拍检测和击球识别
    """
    logger.info("开始增强的羽毛球预测和击球检测...")
    logger.info(f"视频文件: {video_path}")
    logger.info(f"模型文件: {model_path}")
    
    # 检查文件是否存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 生成帧列表
    logger.info("正在从视频生成帧列表...")
    frame_list, fps, (w, h) = generate_frames_from_video(video_path)
    logger.info(f"视频信息: {w}x{h}, {fps}fps, {len(frame_list)}帧")
    
    # 使用原有的羽毛球检测
    logger.info("开始羽毛球检测...")
    tracknet_pred_dict = pred_main(
        frame_list=frame_list,
        fps=fps,
        w=w,
        h=h,
        tracknet_file=model_path,
        output_video=False  # 不生成视频，只获取预测结果
    )
    
    # 初始化统一检测器
    logger.info("初始化统一检测器...")
    unified_detector = UnifiedDetector()
    hit_detector = ImprovedHitDetection(
        min_approach_distance=105.0,     # 最小接近距离阈值 (基于异常值过滤后的真实数据)
        max_separation_distance=167.0,   # 最大分离距离阈值 (基于异常值过滤后的数据)
        time_window=6,                   # 时间窗口（帧数）(基于前一帧分析优化)
        min_velocity_change=3.0,         # 最小速度变化阈值
        confidence_threshold=0.5         # 置信度阈值
    )
    
    # 处理每一帧，添加羽毛球拍检测和击球识别
    logger.info("开始羽毛球拍检测和击球识别...")
    enhanced_predictions = []
    
    # 创建羽毛球位置查找字典
    shuttlecock_lookup = {}
    for i in range(len(tracknet_pred_dict['Frame'])):
        frame_id = tracknet_pred_dict['Frame'][i]
        x = tracknet_pred_dict['X'][i]
        y = tracknet_pred_dict['Y'][i]
        visibility = tracknet_pred_dict['Visibility'][i]
        if visibility == 1:
            shuttlecock_lookup[frame_id] = (x, y)
    
    # 处理每一帧
    with tqdm(total=len(frame_list), desc="处理视频帧") as pbar:
        for frame_id, frame in enumerate(frame_list, 1):
            # 获取羽毛球位置（原有TrackNet检测）
            tracknet_shuttlecock_pos = shuttlecock_lookup.get(frame_id)
            
            # 统一检测：一次检测羽毛球拍和羽毛球
            racket_positions = []
            racket_details = []
            yolo_shuttlecock_positions = []
            yolo_shuttlecock_details = []
            
            try:
                # 一次检测所有目标
                detections = unified_detector.detect_all(frame, confidence_threshold=0.3)
                
                # 处理羽毛球拍检测结果
                for racket in detections['rackets']:
                    center = unified_detector.get_center(racket)
                    racket_positions.append(center)
                    racket_detail = {
                        'bbox': [float(x) for x in racket[:4]],  # x1, y1, x2, y2
                        'confidence': float(racket[4]),
                        'class_id': int(racket[5]),
                        'center': [int(center[0]), int(center[1])]
                    }
                    racket_details.append(racket_detail)
                
                # 处理羽毛球检测结果
                for shuttlecock in detections['shuttlecocks']:
                    center = unified_detector.get_center(shuttlecock)
                    yolo_shuttlecock_positions.append(center)
                    shuttlecock_detail = {
                        'bbox': [float(x) for x in shuttlecock[:4]],  # x1, y1, x2, y2
                        'confidence': float(shuttlecock[4]),
                        'class_id': int(shuttlecock[5]),
                        'center': [int(center[0]), int(center[1])]
                    }
                    yolo_shuttlecock_details.append(shuttlecock_detail)
                    
            except Exception as e:
                logger.warning(f"统一检测失败 (帧 {frame_id}): {e}")
                racket_positions = []
                racket_details = []
                yolo_shuttlecock_positions = []
                yolo_shuttlecock_details = []
            
            # 羽毛球位置融合：优先使用TrackNet，YOLO-World作为补充
            shuttlecock_pos = tracknet_shuttlecock_pos
            if shuttlecock_pos is None and yolo_shuttlecock_positions:
                # 如果TrackNet没有检测到，使用YOLO-World检测到的第一个羽毛球
                shuttlecock_pos = yolo_shuttlecock_positions[0]
                logger.debug(f"帧 {frame_id}: TrackNet未检测到，使用YOLO-World补充检测")
            
            # 更新击球检测
            hit_detector.update_tracking(frame_id, shuttlecock_pos, racket_positions)
            hit_confidence = hit_detector.detect_hit_event(frame_id)
            
            # 记录增强的预测结果
            enhanced_prediction = {
                'frame': frame_id,
                'shuttlecock_pos': shuttlecock_pos,  # 融合后的羽毛球位置
                'tracknet_shuttlecock_pos': tracknet_shuttlecock_pos,  # TrackNet原始检测结果
                'yolo_shuttlecock_positions': yolo_shuttlecock_positions,  # YOLO-World检测结果
                'yolo_shuttlecock_details': yolo_shuttlecock_details,  # YOLO-World详细结果
                'racket_positions': racket_positions,
                'racket_details': racket_details,  # 详细的羽毛球拍信息
                'hit_confidence': hit_confidence,
                'rackets_detected': len(detections['rackets']) if 'detections' in locals() else 0,
                'yolo_shuttlecocks_detected': len(detections['shuttlecocks']) if 'detections' in locals() else 0,
                'original_x': tracknet_pred_dict['X'][frame_id-1] if frame_id <= len(tracknet_pred_dict['X']) else None,
                'original_y': tracknet_pred_dict['Y'][frame_id-1] if frame_id <= len(tracknet_pred_dict['Y']) else None,
                'original_visibility': tracknet_pred_dict['Visibility'][frame_id-1] if frame_id <= len(tracknet_pred_dict['Visibility']) else None
            }
            enhanced_predictions.append(enhanced_prediction)
            
            pbar.update(1)
    
    # 获取击球统计
    hit_stats = hit_detector.get_hit_statistics()
    
    # 保存结果
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_csv, hit_results_file, racket_results_file = save_enhanced_results(
        enhanced_predictions, hit_stats, output_dir, video_name
    )
    
    # 创建可视化视频
    output_video = create_enhanced_visualization(
        video_path, enhanced_predictions, output_dir, video_name
    )
    
    # 统计信息
    total_frames = len(enhanced_predictions)
    frames_with_rackets = sum(1 for p in enhanced_predictions if p['rackets_detected'] > 0)
    frames_with_tracknet_shuttlecock = sum(1 for p in enhanced_predictions if p['tracknet_shuttlecock_pos'] is not None)
    frames_with_yolo_shuttlecock = sum(1 for p in enhanced_predictions if p['yolo_shuttlecocks_detected'] > 0)
    frames_with_fused_shuttlecock = sum(1 for p in enhanced_predictions if p['shuttlecock_pos'] is not None)
    hit_events = sum(1 for p in enhanced_predictions if p['hit_confidence'] is not None and p['hit_confidence'] > 0.5)
    
    # 羽毛球拍检测统计
    total_rackets_detected = sum(p['rackets_detected'] for p in enhanced_predictions)
    racket_confidence_stats = []
    for p in enhanced_predictions:
        if p['racket_details']:
            for racket in p['racket_details']:
                racket_confidence_stats.append(racket['confidence'])
    
    # YOLO-World羽毛球检测统计
    total_yolo_shuttlecocks_detected = sum(p['yolo_shuttlecocks_detected'] for p in enhanced_predictions)
    yolo_shuttlecock_confidence_stats = []
    for p in enhanced_predictions:
        if p['yolo_shuttlecock_details']:
            for shuttlecock in p['yolo_shuttlecock_details']:
                yolo_shuttlecock_confidence_stats.append(shuttlecock['confidence'])
    
    racket_statistics = {
        'total_rackets_detected': int(total_rackets_detected),
        'frames_with_rackets': int(frames_with_rackets),
        'racket_detection_rate': float(frames_with_rackets / total_frames if total_frames > 0 else 0),
        'avg_rackets_per_frame': float(total_rackets_detected / total_frames if total_frames > 0 else 0),
        'avg_confidence': float(np.mean(racket_confidence_stats) if racket_confidence_stats else 0),
        'max_confidence': float(np.max(racket_confidence_stats) if racket_confidence_stats else 0),
        'min_confidence': float(np.min(racket_confidence_stats) if racket_confidence_stats else 0)
    }
    
    yolo_shuttlecock_statistics = {
        'total_yolo_shuttlecocks_detected': int(total_yolo_shuttlecocks_detected),
        'frames_with_yolo_shuttlecock': int(frames_with_yolo_shuttlecock),
        'yolo_shuttlecock_detection_rate': float(frames_with_yolo_shuttlecock / total_frames if total_frames > 0 else 0),
        'avg_yolo_shuttlecocks_per_frame': float(total_yolo_shuttlecocks_detected / total_frames if total_frames > 0 else 0),
        'avg_confidence': float(np.mean(yolo_shuttlecock_confidence_stats) if yolo_shuttlecock_confidence_stats else 0),
        'max_confidence': float(np.max(yolo_shuttlecock_confidence_stats) if yolo_shuttlecock_confidence_stats else 0),
        'min_confidence': float(np.min(yolo_shuttlecock_confidence_stats) if yolo_shuttlecock_confidence_stats else 0)
    }
    
    statistics = {
        'total_frames': total_frames,
        'frames_with_rackets': frames_with_rackets,
        'frames_with_tracknet_shuttlecock': frames_with_tracknet_shuttlecock,
        'frames_with_yolo_shuttlecock': frames_with_yolo_shuttlecock,
        'frames_with_fused_shuttlecock': frames_with_fused_shuttlecock,
        'racket_detection_rate': frames_with_rackets / total_frames if total_frames > 0 else 0,
        'tracknet_shuttlecock_detection_rate': frames_with_tracknet_shuttlecock / total_frames if total_frames > 0 else 0,
        'yolo_shuttlecock_detection_rate': frames_with_yolo_shuttlecock / total_frames if total_frames > 0 else 0,
        'fused_shuttlecock_detection_rate': frames_with_fused_shuttlecock / total_frames if total_frames > 0 else 0,
        'hit_events_detected': hit_events,
        'hit_detection_rate': hit_events / total_frames if total_frames > 0 else 0,
        'hit_statistics': hit_stats,
        'racket_statistics': racket_statistics,  # 羽毛球拍统计信息
        'yolo_shuttlecock_statistics': yolo_shuttlecock_statistics  # YOLO-World羽毛球统计信息
    }
    
    logger.info("处理完成！")
    logger.info(f"总帧数: {total_frames}")
    logger.info(f"检测到羽毛球拍的帧数: {frames_with_rackets}")
    logger.info(f"TrackNet检测到羽毛球的帧数: {frames_with_tracknet_shuttlecock}")
    logger.info(f"YOLO-World检测到羽毛球的帧数: {frames_with_yolo_shuttlecock}")
    logger.info(f"融合后检测到羽毛球的帧数: {frames_with_fused_shuttlecock}")
    logger.info(f"检测到击球事件: {hit_events}")
    logger.info(f"羽毛球拍检测统计:")
    logger.info(f"  总检测到的羽毛球拍数量: {racket_statistics['total_rackets_detected']}")
    logger.info(f"  平均每帧羽毛球拍数量: {racket_statistics['avg_rackets_per_frame']:.2f}")
    logger.info(f"  平均置信度: {racket_statistics['avg_confidence']:.3f}")
    logger.info(f"  最高置信度: {racket_statistics['max_confidence']:.3f}")
    logger.info(f"  最低置信度: {racket_statistics['min_confidence']:.3f}")
    logger.info(f"YOLO-World羽毛球检测统计:")
    logger.info(f"  总检测到的羽毛球数量: {yolo_shuttlecock_statistics['total_yolo_shuttlecocks_detected']}")
    logger.info(f"  平均每帧羽毛球数量: {yolo_shuttlecock_statistics['avg_yolo_shuttlecocks_per_frame']:.2f}")
    logger.info(f"  平均置信度: {yolo_shuttlecock_statistics['avg_confidence']:.3f}")
    logger.info(f"  最高置信度: {yolo_shuttlecock_statistics['max_confidence']:.3f}")
    logger.info(f"  最低置信度: {yolo_shuttlecock_statistics['min_confidence']:.3f}")
    logger.info(f"融合检测效果:")
    logger.info(f"  TrackNet检测率: {statistics['tracknet_shuttlecock_detection_rate']:.3f}")
    logger.info(f"  YOLO-World检测率: {statistics['yolo_shuttlecock_detection_rate']:.3f}")
    logger.info(f"  融合后检测率: {statistics['fused_shuttlecock_detection_rate']:.3f}")
    logger.info(f"  检测率提升: {statistics['fused_shuttlecock_detection_rate'] - statistics['tracknet_shuttlecock_detection_rate']:.3f}")
    
    return {
        'predictions': enhanced_predictions,
        'statistics': statistics,
        'output_files': {
            'csv': output_csv,
            'hit_results': hit_results_file,
            'racket_results': racket_results_file,  # 新增：羽毛球拍检测结果文件
            'video': output_video
        }
    }

def save_enhanced_results(predictions: List[Dict], hit_stats: Dict, output_dir: str, video_name: str):
    """保存增强的预测结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存预测结果
    predictions_df = pd.DataFrame(predictions)
    output_csv = os.path.join(output_dir, f"{video_name}_enhanced_ball.csv")
    predictions_df.to_csv(output_csv, index=False)
    logger.info(f"预测结果已保存: {output_csv}")
    
    # 保存击球检测结果
    hit_results = {
        'video_name': video_name,
        'total_frames': len(predictions),
        'hit_statistics': hit_stats,
        'hit_events': [
            {
                'frame': pred['frame'],
                'hit_confidence': pred['hit_confidence'],
                'rackets_detected': pred['rackets_detected'],
                'shuttlecock_pos': pred['shuttlecock_pos']
            }
            for pred in predictions if pred['hit_confidence'] is not None
        ]
    }
    
    # 保存羽毛球拍检测结果
    # 计算识别统计数据
    total_frames = len(predictions)
    frames_with_rackets = sum(1 for p in predictions if p['rackets_detected'] > 0)
    total_rackets_detected = sum(p['rackets_detected'] for p in predictions)
    racket_confidences = []
    for p in predictions:
        if p['racket_details']:
            for r in p['racket_details']:
                try:
                    racket_confidences.append(float(r.get('confidence', 0.0)))
                except Exception:
                    pass

    racket_statistics = {
        'total_rackets_detected': int(total_rackets_detected),
        'frames_with_rackets': int(frames_with_rackets),
        'racket_detection_rate': float(frames_with_rackets / total_frames) if total_frames > 0 else 0.0,
        'avg_rackets_per_frame': float(total_rackets_detected / total_frames) if total_frames > 0 else 0.0,
        'avg_confidence': float(np.mean(racket_confidences)) if racket_confidences else 0.0,
        'max_confidence': float(np.max(racket_confidences)) if racket_confidences else 0.0,
        'min_confidence': float(np.min(racket_confidences)) if racket_confidences else 0.0
    }

    racket_results = {
        'video_name': video_name,
        'total_frames': total_frames,
        'statistics': racket_statistics,
        'racket_detections': [
            {
                'frame': pred['frame'],
                'rackets_detected': pred['rackets_detected'],
                'racket_details': pred['racket_details']
            }
            for pred in predictions if pred['rackets_detected'] > 0
        ]
    }
    
    hit_results_file = os.path.join(output_dir, f"{video_name}_enhanced_hit_detection.json")
    with open(hit_results_file, 'w', encoding='utf-8') as f:
        json.dump(hit_results, f, ensure_ascii=False, indent=2)
    logger.info(f"击球检测结果已保存: {hit_results_file}")
    
    # 保存羽毛球拍检测结果
    racket_results_file = os.path.join(output_dir, f"{video_name}_racket_detection.json")
    with open(racket_results_file, 'w', encoding='utf-8') as f:
        json.dump(racket_results, f, ensure_ascii=False, indent=2)
    logger.info(f"羽毛球拍检测结果已保存: {racket_results_file}")
    
    return output_csv, hit_results_file, racket_results_file

def create_enhanced_visualization(video_path: str, predictions: List[Dict], output_dir: str, video_name: str):
    """创建增强的可视化视频"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频文件进行可视化: {video_path}")
        return None
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建视频写入器
    output_video_path = os.path.join(output_dir, f"{video_name}_enhanced_analysis.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    with tqdm(total=len(predictions), desc="生成可视化视频") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count < len(predictions):
                pred = predictions[frame_count]
                
                # 绘制羽毛球检测结果（使用原有检测结果）
                if pred['shuttlecock_pos']:
                    pos = pred['shuttlecock_pos']
                    cv2.circle(frame, pos, 15, (0, 255, 0), -1)  # 绿色圆圈
                    cv2.putText(frame, "Shuttlecock", (pos[0] + 20, pos[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 绘制羽毛球拍检测结果
                if pred['racket_positions']:
                    for pos in pred['racket_positions']:
                        cv2.circle(frame, pos, 10, (0, 255, 255), -1)  # 黄色圆圈
                        cv2.putText(frame, "Racket", (pos[0] + 15, pos[1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # 绘制击球事件
                if pred['hit_confidence'] is not None and pred['hit_confidence'] > 0.5:
                    cv2.putText(frame, f"HIT! {pred['hit_confidence']:.2f}", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # 绘制统计信息
                cv2.putText(frame, f"Rackets: {pred['rackets_detected']}", 
                           (10, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Shuttlecock: {'Yes' if pred['shuttlecock_pos'] else 'No'}", 
                           (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Frame: {frame_count}", 
                           (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    out.release()
    
    logger.info(f"可视化视频已保存: {output_video_path}")
    return output_video_path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='优化的改进羽毛球预测和击球检测')
    parser.add_argument('--video', type=str, required=True, help='输入视频文件路径')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    
    args = parser.parse_args()
    
    try:
        results = enhanced_pred_main(args.video, args.model, args.output)
        print("✅ 处理成功完成！")
        print(f"输出文件:")
        for key, path in results['output_files'].items():
            if path:
                print(f"  {key}: {path}")
    except Exception as e:
        logger.error(f"处理失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
