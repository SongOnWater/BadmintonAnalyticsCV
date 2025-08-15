"""
仅分析功能 - 不生成视频文件
"""
import numpy as np
from collections import deque

def analyze_active_frames_only(frame_list, video_config, pred_dict, add_frame, traj_len=8):
    """仅分析活动帧，不生成视频文件
    
    Args:
        frame_list: 帧列表
        video_config: 视频配置
        pred_dict: 预测结果
        add_frame: 帧标记
        traj_len: 轨迹长度
    
    Returns:
        tuple: (add_frame, frame_in_csv, active_frame_list, file_name)
    """
    
    print("仅分析活动帧，不生成视频...")
    
    # 读取预测结果
    x_pred, y_pred, vis_pred = pred_dict['X'], pred_dict['Y'], pred_dict['Visibility']
    
    # 初始化分析变量
    active_frame = 0
    active_frame_list = []
    frame_in_csv = []
    last_frame = False
    
    # 分析每一帧，但不生成视频
    for i, frame in enumerate(frame_list):
        
        if add_frame[i] == True:
            active_frame += 1
        
        if (last_frame == True and add_frame[i] == False):
            print(i, "Ending clip")

        if (last_frame == False and add_frame[i] == True):
            print(i, "Starting clip")
            frame_in_csv.append(i)
            active_frame_list.append(active_frame)

        last_frame = add_frame[i]
    
    # 返回分析结果，但不返回实际的视频文件名
    return add_frame, frame_in_csv, active_frame_list, "analysis_only"