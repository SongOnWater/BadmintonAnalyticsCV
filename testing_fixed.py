"""
修复版本的视频生成 - 解决帧剪切和轨迹绘制问题
"""
import os
import time
from utils.general import *
import pickle
import sys
from utils.func_clips_start_end import *
import cv2
import numpy as np

def create_frames_fixed(original_video_file=None):
    """修复的帧创建函数"""
    with open('predicted.bin','rb') as file:
        pred_dict = pickle.load(file)
        out_file = pickle.load(file)
        video_name = pickle.load(file)
    
    # 如果提供了原始视频文件路径，使用它而不是pickle中的路径
    if original_video_file:
        video_name = original_video_file
        print(f"使用提供的视频文件路径: {video_name}")
    else:
        print(f"使用pickle中的视频文件路径: {video_name}")
    
    frame_list, fps, (w, h) = generate_frames(video_name)
    return frame_list, pred_dict, out_file, fps, (w, h), video_name

def write_complete_pred_video(frame_list, video_config, pred_dict, save_file, traj_len=8):
    """写入完整的预测视频 - 不剪切任何帧"""
    
    print(f"生成完整预测视频: {save_file}")
    
    # 读取预测结果
    x_pred, y_pred, vis_pred = pred_dict['X'], pred_dict['Y'], pred_dict['Visibility']
    
    # 视频配置
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_file, fourcc, video_config['fps'], video_config['shape'])
    
    if not out.isOpened():
        raise Exception(f"无法打开视频写入器: {save_file}")
    
    # 创建轨迹队列
    pred_queue = deque()
    
    print(f"处理 {len(frame_list)} 帧...")
    
    for i, frame in enumerate(frame_list):
        # 检查队列容量
        if len(pred_queue) >= traj_len:
            pred_queue.pop()
        
        # 添加球坐标到队列
        if i < len(x_pred) and vis_pred[i]:
            pred_queue.appendleft([x_pred[i], y_pred[i]])
        else:
            pred_queue.appendleft(None)
        
        # 绘制轨迹
        frame_with_traj = draw_traj(frame.copy(), pred_queue, color='yellow')
        
        # 写入帧
        out.write(frame_with_traj)
        
        # 显示进度
        if i % 500 == 0:
            progress = (i / len(frame_list)) * 100
            print(f"预测视频进度: {progress:.1f}% ({i}/{len(frame_list)})")
    
    out.release()
    print(f"完整预测视频已保存: {save_file}")

def write_score_video_with_trajectory(frame_list, video_config, pred_dict, save_file, 
                                    frame_in_csv, active_frame, original_video_path, traj_len=8):
    """写入带轨迹的得分视频"""
    
    print(f"生成带轨迹的得分视频...")
    
    # 读取预测结果
    x_pred, y_pred, vis_pred = pred_dict['X'], pred_dict['Y'], pred_dict['Visibility']
    
    # 视频配置
    temp_output_path = f"{save_file[:-4]}_score_clip.mp4"
    
    # 尝试不同的编码器，优先使用兼容性更好的编码器
    codec_options = [
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # 最兼容
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),  # 备选
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),  # 备选
        ('H264', cv2.VideoWriter_fourcc(*'H264'))   # 最后尝试
    ]
    
    out = None
    for codec_name, fourcc in codec_options:
        out = cv2.VideoWriter(temp_output_path, fourcc, video_config['fps'], video_config['shape'], isColor=True)
        if out.isOpened():
            print(f"使用编码器: {codec_name}")
            break
        out.release()
    
    if not out or not out.isOpened():
        raise Exception(f"无法打开视频写入器")
    
    # 初始化游戏状态
    scores = {"p1": 0, "p2": 0}
    pointers_to_players = {"closer": "p1", "farther": "p2"}
    set_scores = {"p1": 0, "p2": 0}
    first_serve = True
    
    # 文本参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    font_color = (0, 0, 255)
    font_line_type = cv2.LINE_AA
    padding = 10
    
    frame_width = video_config['shape'][0]
    frame_height = video_config['shape'][1]
    
    # 创建轨迹队列
    pred_queue = deque()
    
    print(f"处理 {len(frame_list)} 帧的得分视频...")
    
    for i, frame in enumerate(frame_list):
        # 检查队列容量
        if len(pred_queue) >= traj_len:
            pred_queue.pop()
        
        # 添加球坐标到队列
        if i < len(x_pred) and vis_pred[i]:
            pred_queue.appendleft([x_pred[i], y_pred[i]])
        else:
            pred_queue.appendleft(None)
        
        # 检查是否是活动帧并更新分数
        if i in active_frame:
            scores = clip_start(frame_in_csv[active_frame.index(i)], i, original_video_path, 
                              scores, pointers_to_players, first_serve)
            
            # 场地交换逻辑
            if set_scores["p1"] == 1 and set_scores["p2"] == 1:
                if scores["p1"] == 11 or scores["p2"] == 11:
                    pointers_to_players["closer"], pointers_to_players["farther"] = \
                        pointers_to_players["farther"], pointers_to_players["closer"]

            # 一局结束检查
            if scores["p1"] == 21 or scores["p2"] == 21:
                if scores["p1"] == 21:
                    set_scores["p1"] += 1
                else:
                    set_scores["p2"] += 1
                scores = {"p1": 0, "p2": 0}
                first_serve = True
                pointers_to_players["closer"], pointers_to_players["farther"] = \
                    pointers_to_players["farther"], pointers_to_players["closer"]
            
            first_serve = False
        
        # 确保帧有效
        if frame is None or frame.size == 0:
            continue
        
        # 复制帧以避免修改原始帧
        frame_copy = frame.copy()
        
        # 绘制轨迹
        frame_copy = draw_traj(frame_copy, pred_queue, color='yellow')
        
        # 绘制分数
        score_text_p1 = f"Player 1: {scores['p1']}"
        score_text_p2 = f"Player 2: {scores['p2']}"
        
        # 计算文本位置
        text_size_p1, _ = cv2.getTextSize(score_text_p1, font, font_scale, font_thickness)
        text_size_p2, _ = cv2.getTextSize(score_text_p2, font, font_scale, font_thickness)
        max_width = max(text_size_p1[0], text_size_p2[0])
        text_height = max(text_size_p1[1], text_size_p2[1])
        
        top_right_p1 = (frame_width - max_width - padding, text_height + padding)
        top_right_p2 = (frame_width - max_width - padding, 2 * text_height + 2 * padding)
        
        # 绘制分数文本
        cv2.putText(frame_copy, score_text_p1, top_right_p1, font, font_scale, 
                   font_color, font_thickness, font_line_type)
        cv2.putText(frame_copy, score_text_p2, top_right_p2, font, font_scale, 
                   font_color, font_thickness, font_line_type)
        
        # 写入帧
        out.write(frame_copy)
        
        # 显示进度
        if i % 500 == 0:
            progress = (i / len(frame_list)) * 100
            print(f"得分视频进度: {progress:.1f}% ({i}/{len(frame_list)})")
    
    out.release()
    print(f"带轨迹的得分视频已保存: {temp_output_path}")

def testing_fixed(generate_pred_video=False, original_video_file=None):
    """修复版本的测试函数
    
    Args:
        generate_pred_video (bool): 是否生成完整预测视频，默认False
        original_video_file (str): 原始视频文件路径，用于替代pickle中可能错误的路径
    """
    
    print("=== 🔧 修复版本视频生成开始 ===")
    print(f"生成预测视频: {'是' if generate_pred_video else '否'}")
    overall_start_time = time.time()
    
    # 加载数据
    frame_list, pred_dict, out_file, fps, (w, h), actual_video_path = create_frames_fixed(original_video_file)
    frame_list = frame_list[:len(pred_dict['Frame'])]
    
    print(f"视频信息: {len(frame_list)} 帧, {fps} FPS, {w}x{h}")
    
    # 视频配置
    video_config = dict(fps=30, shape=(w, h))
    
    # 初始化游戏状态
    scores = {"p1": 0, "p2": 0}
    pointers_to_players = {"closer": "p1", "farther": "p2"}
    set_scores = {"p1": 0, "p2": 0}
    first_serve = True
    
    # 步骤1: 可选生成完整的预测视频
    if generate_pred_video:
        print("\n步骤1: 生成完整预测视频...")
        write_complete_pred_video(frame_list, video_config, pred_dict, out_file, traj_len=8)
    else:
        print("\n步骤1: 跳过预测视频生成（节省时间）")
    
    # 步骤2: 处理帧标记（用于得分片段）
    print("\n步骤2: 处理帧标记...")
    add_frame = pred_dict_modify(False, pred_dict, frame_list, video_config)
    
    # 步骤3: 获取活动帧信息（用于得分计算）
    print("\n步骤3: 分析活动帧...")
    if generate_pred_video:
        # 如果需要生成预测视频，使用原始函数
        af2, frame_in_csv, active_frame, save_file = write_pred_video_modified(
            frame_list, video_config, pred_dict,
            prev_last_frame=None, save_file=out_file, add_frame=add_frame, traj_len=8
        )
    else:
        # 如果不需要预测视频，只做分析
        from utils.analysis_only import analyze_active_frames_only
        af2, frame_in_csv, active_frame, save_file = analyze_active_frames_only(
            frame_list, video_config, pred_dict, add_frame, traj_len=8
        )
        save_file = out_file  # 使用原始文件名
    
    # 步骤4: 生成带轨迹的得分视频
    print("\n步骤4: 生成带轨迹的得分视频...")
    write_score_video_with_trajectory(frame_list, video_config, pred_dict, out_file,
                                    frame_in_csv, active_frame, actual_video_path, traj_len=8)
    
    # 记录结束时间
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    
    print(f"\n=== 🔧 修复版本完成 ===")
    print(f"总处理时间: {overall_duration:.2f} 秒")
    print(f"处理速度: {len(frame_list)/overall_duration:.1f} FPS")
    
    # 输出文件信息
    score_clip_file = f"{out_file[:-4]}_score_clip.mp4"
    print(f"\n生成的文件:")
    if generate_pred_video:
        print(f"✅ 完整预测视频: {out_file}")
    else:
        print(f"⏭️  预测视频: 已跳过")
    print(f"✅ 带轨迹得分片段: {score_clip_file}")

def main(generate_pred_video=False, original_video_file=None):
    """主函数
    
    Args:
        generate_pred_video (bool): 是否生成完整预测视频，默认False
        original_video_file (str): 原始视频文件路径，用于替代pickle中可能错误的路径
    """
    testing_fixed(generate_pred_video, original_video_file)

if __name__ == '__main__':
    main()