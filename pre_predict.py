import time
from predict import *
import pandas
import cv2
import os
import argparse
import pickle

def main(video_file=None, save_dir='prediction', eval_mode='nonoverlap'):
    """Main function for video prediction"""
    if video_file is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--video_file', type=str, help='file path of the video')
        parser.add_argument('--save_dir', default='prediction', type=str)
        parser.add_argument('--eval_mode', choices=['nonoverlap', 'weight'], default='nonoverlap',
                            help='预测模式：nonoverlap(快速) 或 weight(高精度)')
        args = parser.parse_args()
        video_file = args.video_file
        save_dir = args.save_dir
        eval_mode = args.eval_mode
    
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    out_csv_file = os.path.join(save_dir, f'{video_name}_ball.csv')
    out_video_file = os.path.join(save_dir, f'{video_name}.mp4')

    # Check if save_dir exists, create if not
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Handle FPS conversion if needed
    video_file_to_process = video_file
    cap = cv2.VideoCapture(video_file)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Original video FPS: {fps}")
    except Exception:
        print("Error in calculating FPS")
        fps = 30.0
    finally:
        cap.release()

    # Prefer不重编码：允许非30fps输入，模型以固定时间窗处理；如必须固定30fps再启用
    force_convert_to_30fps = False
    if force_convert_to_30fps and abs(fps - 30.0) > 0.1:
        print("Converting video to 30 FPS...")
        video_file_to_process = change_fps(video_file)

    
    # predicting in batches to not use too much memory

    # 使用 OpenCV 获取视频总时长
    cap_info = cv2.VideoCapture(video_file_to_process)
    fps2 = cap_info.get(cv2.CAP_PROP_FPS) or fps
    frame_count = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_length = frame_count / fps2 if fps2 > 0 else 0
    cap_info.release()

    clip_duration = 60  # 大幅提高段长，减少I/O开销和模型加载次数
    num_clips = int(total_length // clip_duration)

    remainder = total_length - num_clips*clip_duration
    # print(remainder)

    # Ensure at least one clip is processed even for very short videos
    if remainder > 0 or num_clips == 0:
        num_clips += 1

    print(f"Total video length: {total_length:.2f} seconds")
    print(f"Processing video in {num_clips} segments (each {clip_duration} seconds)")
    
    pred_dict_joined = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

    # Record overall processing start time
    overall_start_time = time.time()

    for i in range(num_clips):
        print(f"\nProcessing segment {i+1}/{num_clips}...")
        
        # Record segment start time
        segment_start_time = time.time()
        
        start_time = i * clip_duration
        end_time = min((i+1) * clip_duration, total_length)
        print(f"  Segment time: {start_time:.1f}s to {end_time:.1f}s" + (" (last segment)" if i==num_clips-1 else ""))

        # 优化视频读取：使用帧号定位更精确
        cap = cv2.VideoCapture(video_file_to_process)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_cap = cap.get(cv2.CAP_PROP_FPS) or fps
        
        # 计算起始和结束帧号
        start_frame = int(start_time * fps_cap)
        end_frame = int(end_time * fps_cap)
        
        # 直接定位到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_list = []
        current_frame = start_frame
        
        # 预分配列表大小以提高性能
        expected_frames = end_frame - start_frame
        frame_list = [None] * expected_frames
        frame_idx = 0
        
        while current_frame < end_frame and frame_idx < expected_frames:
            grabbed, frame = cap.read()
            if not grabbed:
                break
            frame_list[frame_idx] = frame
            frame_idx += 1
            current_frame += 1
        
        # 移除未使用的预分配空间
        frame_list = frame_list[:frame_idx]
        cap.release()

        # 推理 - 使用指定的eval_mode
        pred_dict = pred_main(frame_list=frame_list, fps=int(round(fps_cap)), w=w, h=h, batch_size=64, eval_mode=eval_mode)

        # Record segment end time and calculate duration
        segment_end_time = time.time()
        segment_duration = segment_end_time - segment_start_time
        print(f"  Processed {len(pred_dict['Frame'])} frames in this segment")
        print(f"  Segment processing time: {segment_duration:.2f} seconds")
        
        # 如果不是强制30FPS，改用原FPS进行帧号平移
        fps_used = 30 if force_convert_to_30fps else int(round(fps))
        for k in range(len(pred_dict['Frame'])):
            pred_dict['Frame'][k] += int(i * fps_used * clip_duration)

        pred_dict_joined['Frame'].extend(pred_dict['Frame'])
        pred_dict_joined['Visibility'].extend(pred_dict['Visibility'])
        pred_dict_joined['X'].extend(pred_dict['X'])
        pred_dict_joined['Y'].extend(pred_dict['Y'])

        # print(pred_dict_joined)
        
    # Record overall processing end time and calculate duration
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
        
    pred_df = pandas.DataFrame({'Frame': pred_dict_joined['Frame'],
                                'Visibility': pred_dict_joined['Visibility'],
                                'X': pred_dict_joined['X'],
                                'Y': pred_dict_joined['Y']})
    
    pred_df.to_csv(out_csv_file, index=False)

    with open(f'predicted.bin','wb') as file:
        pickle.dump(pred_dict_joined, file)
        pickle.dump(out_video_file, file)
        pickle.dump(video_file, file)
    
    # Clean up temporary files
    if video_file_to_process != video_file:
        try:
            os.remove(video_file_to_process)
            print(f"Removed temporary file: {video_file_to_process}")
        except Exception:
            print(f"Could not remove temporary file: {video_file_to_process}")
    
    print(f"\nVideo processing completed successfully!")
    print(f"Total processing time: {overall_duration:.2f} seconds")
    print(f"Average time per segment: {overall_duration/num_clips:.2f} seconds")
    
    return pred_dict_joined, out_csv_file, out_video_file


if __name__ == '__main__':
    main()