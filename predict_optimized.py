"""
真正高效的预测模块 - 专注于消除核心性能瓶颈
主要优化：
1. 消除视频分割 - 直接处理整个视频
2. 最大化批处理大小 - 充分利用GPU
3. 减少数据复制和转换
4. 优化内存使用模式
"""

import os
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

from test import predict_location, get_ensemble_weight, generate_inpaint_mask
from dataset import Shuttlecock_Trajectory_Dataset
from utils.general import *

def predict_ultra_fast(video_file, save_dir="prediction", batch_size=None):
    """
    超高速预测函数 - 消除所有主要性能瓶颈
    """
    print(f"🚀 Ultra-fast prediction starting for: {video_file}")
    start_time = time.time()
    
    # 设置输出路径
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    out_csv_file = os.path.join(save_dir, f'{video_name}_ball.csv')
    out_video_file = os.path.join(save_dir, f'{video_name}.mp4')
    os.makedirs(save_dir, exist_ok=True)
    
    # 🚀 优化1: 检查和转换FPS（如果需要）
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    video_file_to_process = video_file
    if abs(fps - 30.0) > 0.1:
        print(f"🔄 Converting {fps:.1f}fps to 30fps...")
        video_file_to_process = convert_fps_optimized(video_file)
    
    # 🚀 优化2: 一次性加载所有帧（消除视频分割瓶颈）
    print("🚀 Loading all frames at once...")
    load_start = time.time()
    
    cap = cv2.VideoCapture(video_file_to_process)
    frame_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_list.append(frame)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    load_time = time.time() - load_start
    print(f"✅ Loaded {len(frame_list)} frames in {load_time:.2f}s")
    
    # 🚀 优化3: 设备和批处理大小优化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if batch_size is None:
        if torch.cuda.is_available():
            # 根据GPU内存动态设置批处理大小
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory_gb >= 16:
                batch_size = 32
            elif gpu_memory_gb >= 8:
                batch_size = 16
            else:
                batch_size = 8
        else:
            batch_size = 4
    
    print(f"🚀 Using device: {device}, batch_size: {batch_size}")
    
    # 🚀 优化4: 模型加载和优化
    print("🚀 Loading and optimizing model...")
    model_start = time.time()
    
    tracknet_file = "ckpts/TrackNet_best.pt"
    tracknet_ckpt = torch.load(tracknet_file, map_location=device)
    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).to(device)
    tracknet.load_state_dict(tracknet_ckpt['model'])
    tracknet.eval()
    
    # GPU优化
    if device.type == 'cuda':
        tracknet = tracknet.half()  # 使用半精度
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    model_time = time.time() - model_start
    print(f"✅ Model loaded and optimized in {model_time:.2f}s")
    
    # 🚀 优化5: 数据预处理优化
    print("🚀 Preprocessing frames...")
    preprocess_start = time.time()
    
    # 一次性转换所有帧（BGR to RGB）
    frame_arr = np.array(frame_list)[:, :, :, ::-1]
    del frame_list  # 立即释放内存
    
    w_scaler, h_scaler = w / WIDTH, h / HEIGHT
    img_scaler = (w_scaler, h_scaler)
    
    preprocess_time = time.time() - preprocess_start
    print(f"✅ Frames preprocessed in {preprocess_time:.2f}s")
    
    # 🚀 优化6: 数据集和数据加载器优化
    print("🚀 Creating optimized dataset...")
    dataset_start = time.time()
    
    dataset = Shuttlecock_Trajectory_Dataset(
        seq_len=tracknet_seq_len, 
        sliding_step=1, 
        data_mode='heatmap', 
        bg_mode=bg_mode,
        frame_arr=frame_arr
    )
    
    # 优化的数据加载器
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # 禁用多进程以减少开销
        pin_memory=(device.type == 'cuda'),
        drop_last=False
    )
    
    dataset_time = time.time() - dataset_start
    print(f"✅ Dataset created in {dataset_time:.2f}s")
    
    # 🚀 优化7: 超高速模型推理
    print("🚀 Starting ultra-fast inference...")
    inference_start = time.time()
    
    pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
    
    # 预分配缓冲区以提高性能
    video_len = len(frame_arr)
    num_sample = video_len - tracknet_seq_len + 1
    buffer_size = tracknet_seq_len - 1
    batch_i = torch.arange(tracknet_seq_len)
    frame_i = torch.arange(tracknet_seq_len-1, -1, -1)
    y_pred_buffer = torch.zeros((buffer_size, tracknet_seq_len, HEIGHT, WIDTH), dtype=torch.float32)
    weight = get_ensemble_weight(tracknet_seq_len, "weight")
    
    sample_count = 0
    
    with torch.no_grad():
        for step, (i, x) in enumerate(tqdm(data_loader, desc="🚀 Ultra-fast inference")):
            # 优化的数据传输
            if device.type == 'cuda':
                x = x.half().to(device, non_blocking=True)
            else:
                x = x.float().to(device, non_blocking=True)
            
            b_size = i.shape[0]
            
            # 模型推理
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    y_pred = tracknet(x)
                y_pred = y_pred.float().cpu()
            else:
                y_pred = tracknet(x).cpu()
            
            # 处理预测结果
            y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
            ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
            ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)
            
            for b in range(b_size):
                if sample_count < buffer_size:
                    y_pred_ensemble = y_pred_buffer[batch_i+b, frame_i].sum(0)
                    y_pred_ensemble /= (sample_count+1)
                else:
                    y_pred_ensemble = (y_pred_buffer[batch_i+b, frame_i] * weight[:, None, None]).sum(0)
                
                ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred_ensemble.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                sample_count += 1
                
                if sample_count == num_sample:
                    # 处理最后的序列
                    y_zero_pad = torch.zeros((buffer_size, tracknet_seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                    y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)
                    
                    for f in range(1, tracknet_seq_len):
                        y_pred_ensemble = y_pred_buffer[batch_i+b+f, frame_i].sum(0)
                        y_pred_ensemble /= (tracknet_seq_len-f)
                        ensemble_i = torch.cat((ensemble_i, i[-1][f].reshape(1, 1, 2)), dim=0)
                        ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred_ensemble.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
            
            # 预测坐标
            tmp_pred = predict_coordinates_fast(ensemble_i, ensemble_y_pred, img_scaler)
            for key in tmp_pred.keys():
                pred_dict[key].extend(tmp_pred[key])
            
            # 更新缓冲区
            y_pred_buffer = y_pred_buffer[-buffer_size:]
            
            # 定期清理GPU缓存
            if device.type == 'cuda' and step % 50 == 0:
                torch.cuda.empty_cache()
    
    inference_time = time.time() - inference_start
    print(f"✅ Inference completed in {inference_time:.2f}s")
    
    # 🚀 优化8: 快速结果保存
    print("🚀 Saving results...")
    save_start = time.time()
    
    import pandas as pd
    pred_df = pd.DataFrame({
        'Frame': pred_dict['Frame'],
        'Visibility': pred_dict['Visibility'],
        'X': pred_dict['X'],
        'Y': pred_dict['Y']
    })
    pred_df.to_csv(out_csv_file, index=False)
    
    # 保存二进制文件
    import pickle
    with open('predicted.bin', 'wb') as file:
        pickle.dump(pred_dict, file)
        pickle.dump(out_video_file, file)
        pickle.dump(video_file, file)
    
    save_time = time.time() - save_start
    print(f"✅ Results saved in {save_time:.2f}s")
    
    # 清理临时文件
    if video_file_to_process != video_file:
        try:
            os.remove(video_file_to_process)
        except:
            pass
    
    # 总结性能
    total_time = time.time() - start_time
    fps_processed = len(pred_dict['Frame']) / total_time
    
    print(f"\n🎉 ULTRA-FAST processing completed!")
    print(f"⚡ Total time: {total_time:.2f}s")
    print(f"⚡ Processing speed: {fps_processed:.1f} FPS")
    print(f"📊 Performance breakdown:")
    print(f"   - Frame loading: {load_time:.2f}s ({load_time/total_time*100:.1f}%)")
    print(f"   - Model loading: {model_time:.2f}s ({model_time/total_time*100:.1f}%)")
    print(f"   - Preprocessing: {preprocess_time:.2f}s ({preprocess_time/total_time*100:.1f}%)")
    print(f"   - Dataset creation: {dataset_time:.2f}s ({dataset_time/total_time*100:.1f}%)")
    print(f"   - Inference: {inference_time:.2f}s ({inference_time/total_time*100:.1f}%)")
    print(f"   - Saving: {save_time:.2f}s ({save_time/total_time*100:.1f}%)")
    
    return pred_dict

def predict_coordinates_fast(indices, y_pred, img_scaler):
    """优化的坐标预测函数"""
    pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
    
    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = indices.detach().cpu().numpy() if torch.is_tensor(indices) else indices
    
    if y_pred is not None:
        y_pred = y_pred > 0.5
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred)
    
    prev_f_i = -1
    for n in range(batch_size):
        for f in range(seq_len):
            f_i = indices[n][f][1]
            if f_i != prev_f_i:
                if y_pred is not None:
                    y_p = y_pred[n][f]
                    bbox_pred = predict_location(to_img(y_p))
                    cx_pred, cy_pred = int(bbox_pred[0]+bbox_pred[2]/2), int(bbox_pred[1]+bbox_pred[3]/2)
                    cx_pred, cy_pred = int(cx_pred*img_scaler[0]), int(cy_pred*img_scaler[1])
                else:
                    cx_pred, cy_pred = 0, 0
                
                vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                pred_dict['Frame'].append(int(f_i))
                pred_dict['X'].append(cx_pred)
                pred_dict['Y'].append(cy_pred)
                pred_dict['Visibility'].append(vis_pred)
                prev_f_i = f_i
            else:
                break
    
    return pred_dict

def convert_fps_optimized(video_file):
    """优化的FPS转换"""
    from moviepy.editor import VideoFileClip
    
    clip = VideoFileClip(video_file)
    base_name, ext = os.path.splitext(video_file)
    temp_file = f"{base_name}_30fps{ext}"
    
    clip.write_videofile(
        temp_file, 
        fps=30, 
        codec='libx264', 
        audio=False,
        verbose=False, 
        logger=None,
        temp_audiofile=None,
        preset='ultrafast'  # 最快编码
    )
    clip.close()
    
    return temp_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra-fast badminton prediction")
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--save_dir', default='prediction', help='Output directory')
    parser.add_argument('--batch_size', type=int, help='Batch size (auto if not specified)')
    
    args = parser.parse_args()
    
    predict_ultra_fast(args.video, args.save_dir, args.batch_size)