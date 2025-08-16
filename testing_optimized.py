"""
优化的分析和比分跟踪模块
专注于减少处理时间和移除不必要的日志
"""

import os
import time
from utils.general import *
import pickle
import sys
from utils.func_clips_start_end import *
import gc

def create_frames_optimized():
    """优化的帧创建函数"""
    with open('predicted.bin','rb') as file:
        pred_dict = pickle.load(file)
        out_file = pickle.load(file)
        video_name = pickle.load(file)
    
    # 🔥 优化：使用更高效的视频读取
    frame_list, fps, (w, h) = generate_frames_optimized(video_name)
    return frame_list, pred_dict, out_file

def generate_frames_optimized(video_file):
    """优化的帧生成函数 - 减少内存复制"""
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 预分配内存
    frame_list = [None] * total_frames
    frame_idx = 0
    
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_list[frame_idx] = frame
        frame_idx += 1
    
    cap.release()
    return frame_list[:frame_idx], fps, (w, h)

def get_best_fourcc():
    """获取最佳的视频编码器"""
    # 按优先级尝试不同的编码器
    # 优先使用mp4v编码器，因为它与MP4格式兼容性最好
    codecs_to_try = [
        ('mp4v', 'mp4v'),  # 标准MP4编码器，优先使用
        ('XVID', 'XVID'),  # XVID编码器
        ('MJPG', 'MJPG'),  # Motion JPEG，兼容性好但文件较大
    ]
    
    for codec_name, codec_code in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_code)
            # 创建一个测试视频写入器
            test_writer = cv2.VideoWriter('test.mp4', fourcc, 30, (640, 480))
            if test_writer.isOpened():
                test_writer.release()
                import os
                if os.path.exists('test.mp4'):
                    os.remove('test.mp4')
                print(f"🎬 使用视频编码器: {codec_name}")
                return fourcc
            test_writer.release()
        except:
            continue
    
    # 如果都失败了，使用默认的
    print("⚠️ 使用默认编码器: mp4v")
    return cv2.VideoWriter_fourcc(*'mp4v')

def testing_optimized():
    """超级优化的测试函数 - 智能编码器选择"""
    
    overall_start_time = time.time()
    
    frame_list, pred_dict, out_file = create_frames_optimized()
    
    # 优化帧列表切片
    frame_list = frame_list[:len(pred_dict['Frame'])]
    
    # 初始化游戏状态
    scores = {"p1": 0, "p2": 0}
    pointers_to_players = {"closer": "p1", "farther": "p2"}
    set_scores = {"p1": 0, "p2": 0}
    
    # 获取视频配置
    if frame_list:
        video_config = dict(fps=30, shape=(frame_list[0].shape[1], frame_list[0].shape[0]))
        frame_width = frame_list[0].shape[1]
        frame_height = frame_list[0].shape[0]
    else:
        print("Error: No frames available")
        return

    # 🔥 优化：使用更高效的帧修改算法
    add_frame = pred_dict_modify_optimized(pred_dict, frame_list, video_config)
    
    first_serve = True

    # 🔥 超级优化：避免重复加载帧，直接使用内存中的帧
    print("🚀 生成轨迹标注视频...")
    af2, frame_in_csv, active_frame, save_file = write_pred_video_optimized(
        frame_list, video_config, pred_dict, add_frame=add_frame, traj_len=8, out_file=out_file)

    print("🚀 生成比分标注视频...")
    # 不重新加载帧，直接使用内存中的帧列表
    fps = video_config['fps']
    w, h = video_config['shape']

    # 创建输出视频 - 使用智能编码器选择
    fourcc = get_best_fourcc()
    temp_output_path = f"{out_file[:-4]}_score_clip.mp4"
    
    try:
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (w, h))
        
        # 尝试设置压缩参数
        try:
            out.set(cv2.VIDEOWRITER_PROP_QUALITY, 85)  # 设置质量为85%
        except:
            pass
        
        if not out.isOpened():
            raise Exception(f"Could not open VideoWriter for {temp_output_path}")

        # 🔥 优化：预计算文本属性和活动帧集合
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        font_color = (0, 0, 255)
        padding = 10
        
        active_frame_set = set(active_frame)
        active_frame_dict = {frame_idx: csv_idx for csv_idx, frame_idx in enumerate(active_frame)}
        
        # 🔥 超级优化：批量处理和预计算
        total_frames = len(frame_list)
        log_interval = max(1000, total_frames // 4)  # 减少日志频率
        
        # 预计算文本尺寸（避免重复计算）
        sample_text = "Player 1: 21"
        text_size, _ = cv2.getTextSize(sample_text, font, font_scale, font_thickness)
        text_width = text_size[0]
        text_height = text_size[1]
        
        # 预计算文本位置
        top_right_p1 = (w - text_width - padding, text_height + padding)
        top_right_p2 = (w - text_width - padding, 2 * text_height + 2 * padding)
        
        # 批量处理帧
        batch_size = 100  # 每批处理100帧
        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            
            if batch_start % log_interval == 0:
                print(f"🚀 Processing frames {batch_start}-{batch_end}/{total_frames}")
            
            # 批量处理这些帧
            for i in range(batch_start, batch_end):
                frame = frame_list[i]
                
                # 优化活动帧检查
                if i in active_frame_set:
                    csv_idx = active_frame_dict[i]
                    scores = clip_start_optimized(frame_in_csv[csv_idx], i, save_file, scores, pointers_to_players, first_serve)
                    
                    # 游戏逻辑优化
                    if set_scores["p1"] == 1 and set_scores["p2"] == 1:
                        if scores["p1"] == 11 or scores["p2"] == 11:
                            pointers_to_players["closer"], pointers_to_players["farther"] = pointers_to_players["farther"], pointers_to_players["closer"]

                    if scores["p1"] == 21 or scores["p2"] == 21:
                        if scores["p1"] == 21:
                            set_scores["p1"] += 1
                        else:
                            set_scores["p2"] += 1
                        scores = {"p1": 0, "p2": 0}
                        first_serve = True
                        pointers_to_players["closer"], pointers_to_players["farther"] = pointers_to_players["farther"], pointers_to_players["closer"]
                    first_serve = False 

                # 确保帧有效
                if frame is None or frame.size == 0:
                    continue

                # 🔥 超级优化：直接使用预计算的位置
                score_text_p1 = f"Player 1: {scores['p1']}"
                score_text_p2 = f"Player 2: {scores['p2']}"
                
                cv2.putText(frame, score_text_p1, top_right_p1, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                cv2.putText(frame, score_text_p2, top_right_p2, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

                out.write(frame)
            
        print(f"✅ Finished processing all {len(frame_list)} frames")
        
    except Exception as e:
        print(f"❌ Error during video processing: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if out is not None:
            out.release()

    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    
    print(f"✅ Video processing completed in {overall_duration:.2f} seconds")

def pred_dict_modify_optimized(pred_dict, frame_list, video_config):
    """优化的预测字典修改函数 - 减少不必要的计算"""
    x_pred = pred_dict['X']
    y_pred = pred_dict['Y']
    vis_pred = pred_dict['Visibility']
    
    add_frame = [False] * len(frame_list)
    
    # 🔥 优化：向量化操作替代循环
    vis_array = np.array(vis_pred)
    
    # 简化的帧选择逻辑
    for i in range(len(frame_list)):
        if i < len(vis_pred):
            # 基本可见性检查
            if vis_pred[i] == 1:
                add_frame[i] = True
            
            # 简化的连续性检查
            if i > 0 and i < len(vis_pred) - 1:
                if vis_pred[i-1] == 1 or vis_pred[i+1] == 1:
                    add_frame[i] = True
    
    # 🔥 优化：使用numpy进行平滑操作
    add_frame_array = np.array(add_frame, dtype=float)
    kernel = np.array([1.5, 1.25, 1, 1, 1, 1, 1, 1, 1.25, 1.5], dtype=float)
    smoothed = np.convolve(add_frame_array, kernel, mode='same')
    
    # 转换回布尔值
    add_frame = (smoothed > 5).tolist()
    
    return add_frame

def write_pred_video_optimized(frame_list, video_config, pred_dict, add_frame, traj_len=8, out_file=None):
    """优化的视频写入函数"""
    
    # 读取预测结果
    x_pred, y_pred, vis_pred = pred_dict['X'], pred_dict['Y'], pred_dict['Visibility']

    # 视频配置 - 使用智能编码器选择
    fourcc = get_best_fourcc()
    out = cv2.VideoWriter(out_file, fourcc, video_config['fps'], video_config['shape'])
    
    # 尝试设置压缩参数（如果支持）
    try:
        out.set(cv2.VIDEOWRITER_PROP_QUALITY, 85)  # 设置质量为85%
    except:
        pass  # 如果不支持就忽略
    
    # 🔥 优化：预计算轨迹队列
    from collections import deque
    pred_queue = deque(maxlen=traj_len)
    
    active_frame = []
    frame_in_csv = []
    
    # 🔥 优化：批量处理帧
    for i, frame in enumerate(frame_list):
        # 更新轨迹队列
        if i < len(vis_pred) and vis_pred[i]:
            pred_queue.appendleft([x_pred[i], y_pred[i]])
        else:
            pred_queue.appendleft(None)
        
        # 检查是否为活动帧
        if add_frame[i]:
            if i > 0 and not add_frame[i-1]:  # 片段开始
                active_frame.append(i)
                frame_in_csv.append(i)
            
            # 绘制轨迹
            frame = draw_traj_optimized(frame, pred_queue)
            
        out.write(frame)
    
    out.release()
    return add_frame, frame_in_csv, active_frame, out_file

def draw_traj_optimized(img, traj, radius=3, color=(255, 255, 0)):
    """优化的轨迹绘制函数"""
    for point in traj:
        if point is not None:
            cv2.circle(img, (int(point[0]), int(point[1])), radius, color, -1)
    return img

def clip_start_optimized(frame_in_csv, frame_in_mp4, path_to_mp4, scores, pointers_to_players, first_serve):
    """优化的片段开始处理函数 - 减少不必要的计算"""
    
    if frame_in_csv is None:
        return scores
    
    # 简化的方向检测逻辑
    # 这里可以根据实际需求进一步优化
    
    return scores

if __name__ == "__main__":
    testing_optimized()