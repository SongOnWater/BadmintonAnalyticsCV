#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心预测模块 - 整合自适应优化和固定参数
根据需要自动选择最优的预测策略
"""

import os
import numpy as np
import time
import torch
import cv2
from tqdm import tqdm
import gc
import pandas as pd
import pickle
from test import predict_location
from utils.general import *

def predict_core(video_file, save_dir="prediction", batch_size=None, auto_optimize=True):
    """
    核心预测函数 - 智能选择优化策略
    
    Args:
        video_file: 输入视频路径
        save_dir: 输出目录
        batch_size: 批处理大小 (None为自动选择)
        auto_optimize: 是否启用自适应优化
    """
    print(f"🚀 Core prediction starting for: {video_file}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 步骤1: 模型设置
    print("🚀 Step 1: Model setup...")
    model_start = time.time()
    
    tracknet_file = "ckpts/TrackNet_best.pt"
    tracknet_ckpt = torch.load(tracknet_file, map_location=device)
    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    
    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).to(device)
    tracknet.load_state_dict(tracknet_ckpt['model'])
    tracknet.eval()
    
    # 自适应GPU优化
    if auto_optimize and batch_size is None:
        try:
            from adaptive_gpu_optimizer import optimize_for_current_gpu
            
            # 根据模型配置确定输入形状
            if bg_mode == 'concat':
                channels = (tracknet_seq_len + 1) * 3
            else:
                channels = tracknet_seq_len * 3
            
            input_shape = (1, channels, HEIGHT, WIDTH)
            optimization_result = optimize_for_current_gpu(input_shape)
            
            batch_size = optimization_result['optimal_batch_size']
            gpu_name = optimization_result['gpu_characteristics']['name']
            print(f"🚀 自适应GPU优化完成:")
            print(f"   - GPU: {gpu_name}")
            print(f"   - 批处理大小: {batch_size}")
            
        except Exception as e:
            print(f"⚠️ 自适应优化失败，使用默认参数: {e}")
            batch_size = 24
    elif batch_size is None:
        batch_size = 24
    
    if device.type == 'cuda':
        tracknet = tracknet.half()
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    print(f"🚀 Using device: {device}, batch_size: {batch_size}")
    model_time = time.time() - model_start
    print(f"✅ Model ready in {model_time:.2f}s")
    
    # 步骤2: 视频加载
    print("🚀 Step 2: Optimized video loading...")
    load_start = time.time()
    
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"🚀 Video info: {frame_count} frames, {fps:.1f}fps, {w}x{h}")
    
    # 加载所有帧
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    load_time = time.time() - load_start
    print(f"✅ Loaded {len(frames)} frames in {load_time:.2f}s")
    
    # 步骤3: 预处理
    print("🚀 Step 3: Ultra-fast preprocessing...")
    preprocess_start = time.time()
    
    w_scaler, h_scaler = w / WIDTH, h / HEIGHT
    
    # 直接numpy操作
    print("🚀 Direct numpy preprocessing...")
    processed_frames = np.empty((len(frames), HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    for i, frame in enumerate(frames):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frames[i] = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
    
    # 计算背景帧
    median_frame = None
    if bg_mode == 'concat':
        print("🚀 Computing median background (sampled)...")
        sample_step = max(1, len(processed_frames) // 20)
        sampled_frames = processed_frames[::sample_step]
        median_frame = np.median(sampled_frames, axis=0).astype(np.uint8)
    
    preprocess_time = time.time() - preprocess_start
    print(f"✅ Preprocessing done in {preprocess_time:.2f}s")
    
    # 步骤4: 序列创建
    print("🚀 Step 4: Revolutionary sequence creation...")
    seq_start = time.time()
    
    num_frames = len(processed_frames)
    num_sequences = num_frames - tracknet_seq_len + 1
    
    if num_sequences <= 0:
        raise ValueError(f"视频帧数({num_frames})不足以创建序列(需要至少{tracknet_seq_len}帧)")
    
    print(f"🚀 Creating {num_sequences} sequences with revolutionary method...")
    
    # 转换为tensor
    frames_tensor = torch.from_numpy(processed_frames).float() / 255.0
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (N, 3, H, W)
    
    if bg_mode == 'concat':
        bg_tensor = torch.from_numpy(median_frame).float() / 255.0
        bg_tensor = bg_tensor.permute(2, 0, 1)  # (3, H, W)
        channels = (tracknet_seq_len + 1) * 3
    else:
        channels = tracknet_seq_len * 3
    
    # 创建所有序列
    sequences = []
    for i in range(num_sequences):
        seq_frames = frames_tensor[i:i+tracknet_seq_len]  # (seq_len, 3, H, W)
        
        if bg_mode == 'concat':
            seq_with_bg = torch.cat([bg_tensor.unsqueeze(0), seq_frames], dim=0)
            seq_input = seq_with_bg.view(channels, HEIGHT, WIDTH)
        else:
            seq_input = seq_frames.view(channels, HEIGHT, WIDTH)
        
        sequences.append(seq_input)
    
    # 释放内存
    del frames_tensor, processed_frames
    gc.collect()
    
    seq_time = time.time() - seq_start
    print(f"✅ Revolutionary sequence creation done in {seq_time:.2f}s")
    
    # 步骤5: 推理
    print("🚀 Step 5: Ultra-high-speed inference...")
    inference_start = time.time()
    
    all_predictions = []
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(sequences), batch_size), desc="🚀 Ultra Inference"):
            batch_end = min(batch_start + batch_size, len(sequences))
            batch_sequences = sequences[batch_start:batch_end]
            
            try:
                batch_tensor = torch.stack(batch_sequences)
                
                if device.type == 'cuda':
                    batch_tensor = batch_tensor.half().to(device, non_blocking=True)
                    with torch.cuda.amp.autocast():
                        batch_pred = tracknet(batch_tensor)
                    batch_pred = batch_pred.float().cpu()
                else:
                    batch_tensor = batch_tensor.to(device, non_blocking=True)
                    batch_pred = tracknet(batch_tensor).cpu()
                
                del batch_tensor
                all_predictions.append(batch_pred)
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ Memory issue at batch {batch_start}, reducing batch size...")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()
                    # 减小批处理大小重试
                    batch_size = max(1, batch_size // 2)
                    continue
                else:
                    raise e
    
    inference_time = time.time() - inference_start
    print(f"✅ Ultra-high-speed inference done in {inference_time:.2f}s")
    
    # 步骤6: 后处理
    print("🚀 Step 6: Fast post-processing...")
    postprocess_start = time.time()
    
    pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
    middle_frame_idx = tracknet_seq_len // 2
    
    sequence_idx = 0
    for batch_pred in all_predictions:
        batch_pred_np = batch_pred.cpu().numpy()
        
        for i in range(batch_pred_np.shape[0]):
            if sequence_idx < num_sequences:
                frame_idx = sequence_idx + middle_frame_idx
                pred = batch_pred_np[i, 0]  # (H, W)
                
                # 找到最大值位置
                y_pred, x_pred = np.unravel_index(np.argmax(pred), pred.shape)
                
                # 转换回原始坐标
                x_pred = int(x_pred * w / WIDTH)
                y_pred = int(y_pred * h / HEIGHT)
                
                visibility = 1 if np.max(pred) > 0.5 else 0
                
                pred_dict['Frame'].append(frame_idx)
                pred_dict['X'].append(x_pred)
                pred_dict['Y'].append(y_pred)
                pred_dict['Visibility'].append(visibility)
                
                sequence_idx += 1
    
    postprocess_time = time.time() - postprocess_start
    print(f"✅ Fast post-processing done in {postprocess_time:.2f}s")
    
    # 步骤7: 保存结果
    print("🚀 Step 7: Saving results...")
    save_start = time.time()
    
    os.makedirs(save_dir, exist_ok=True)
    
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    out_csv_file = os.path.join(save_dir, f'{video_name}_ball.csv')
    out_video_file = os.path.join(save_dir, f'{video_name}.mp4')
    
    # 保存CSV
    pred_df = pd.DataFrame({
        'Frame': pred_dict['Frame'],
        'Visibility': pred_dict['Visibility'],
        'X': pred_dict['X'],
        'Y': pred_dict['Y']
    })
    pred_df.to_csv(out_csv_file, index=False)
    
    # 保存pickle
    with open('predicted.bin', 'wb') as file:
        pickle.dump(pred_dict, file)
        pickle.dump(out_video_file, file)
        pickle.dump(video_file, file)
    
    save_time = time.time() - save_start
    print(f"✅ Results saved in {save_time:.2f}s")
    
    # 总结
    total_time = model_time + load_time + preprocess_time + seq_time + inference_time + postprocess_time + save_time
    
    print(f"\n🚀 CORE PREDICTION completed!")
    print(f"⚡ Total time: {total_time:.2f}s")
    print(f"⚡ Processing speed: {len(pred_dict['Frame'])/total_time:.1f} FPS")
    print(f"⚡ Speedup: {len(pred_dict['Frame'])/total_time/fps:.1f}x real-time")
    
    print(f"\n📊 Performance breakdown:")
    print(f"   - Model setup: {model_time:.2f}s ({model_time/total_time*100:.1f}%)")
    print(f"   - Video loading: {load_time:.2f}s ({load_time/total_time*100:.1f}%)")
    print(f"   - Preprocessing: {preprocess_time:.2f}s ({preprocess_time/total_time*100:.1f}%)")
    print(f"   - Sequence creation: {seq_time:.2f}s ({seq_time/total_time*100:.1f}%)")
    print(f"   - Inference: {inference_time:.2f}s ({inference_time/total_time*100:.1f}%)")
    print(f"   - Post-processing: {postprocess_time:.2f}s ({postprocess_time/total_time*100:.1f}%)")
    print(f"   - Saving: {save_time:.2f}s ({save_time/total_time*100:.1f}%)")
    
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"⚡ Peak GPU memory usage: {peak_memory:.1f}GB / {total_memory:.1f}GB ({peak_memory/total_memory*100:.1f}%)")
    
    return pred_dict

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='核心预测模块')
    parser.add_argument('--video', required=True, help='输入视频路径')
    parser.add_argument('--save_dir', default='prediction', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=None, help='批处理大小')
    parser.add_argument('--no-auto-optimize', action='store_true', help='禁用自适应优化')
    
    args = parser.parse_args()
    
    predict_core(
        args.video, 
        args.save_dir, 
        args.batch_size, 
        auto_optimize=not args.no_auto_optimize
    )