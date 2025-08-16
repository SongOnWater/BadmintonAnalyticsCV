#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
羽毛球分析系统 - 主执行脚本
智能选择最优处理策略，一键完成预测和视频生成
"""

import argparse
import torch
import gc
import psutil
import os
import time
import cv2
import glob
import shutil
from predict_core import predict_core

def get_memory_info():
    """获取内存信息"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024 / 1024  # GB

def cleanup_intermediate_files():
    """清理所有中间产物"""
    print("🧹 清理中间产物...")
    
    cleaned_count = 0
    
    # 清理临时分段目录
    temp_dirs = glob.glob("temp_segment_*")
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                cleaned_count += 1
            except:
                pass
    
    # 清理临时文件
    temp_files = ["predicted.bin", "gpu_optimization_cache.json", "test_sequences.pkl"]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                cleaned_count += 1
            except:
                pass
    
    # 清理输出目录中的临时CSV文件
    temp_csv_patterns = ["prediction/tmp*_ball.csv", "*/tmp*_ball.csv"]
    for pattern in temp_csv_patterns:
        temp_csv_files = glob.glob(pattern)
        for temp_csv in temp_csv_files:
            if os.path.exists(temp_csv):
                try:
                    os.remove(temp_csv)
                    cleaned_count += 1
                    print(f"   🧹 清理临时CSV: {os.path.basename(temp_csv)}")
                except:
                    pass
    
    # 清理Python缓存
    cache_dirs = glob.glob("__pycache__") + glob.glob("*/__pycache__")
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                cleaned_count += 1
            except:
                pass
    
    if cleaned_count > 0:
        print(f"✅ 清理完成: 删除了 {cleaned_count} 个中间文件/目录")
    else:
        print("✅ 目录已经很干净")

def cleanup_intermediate_files_partial():
    """部分清理中间产物 (保留predicted.bin)"""
    print("🧹 部分清理中间产物...")
    
    cleaned_count = 0
    
    # 清理临时分段目录
    temp_dirs = glob.glob("temp_segment_*")
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                cleaned_count += 1
            except:
                pass
    
    # 清理部分临时文件 (保留predicted.bin)
    temp_files = ["gpu_optimization_cache.json", "test_sequences.pkl"]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                cleaned_count += 1
            except:
                pass
    
    # 清理输出目录中的临时CSV文件
    temp_csv_patterns = ["prediction/tmp*_ball.csv", "*/tmp*_ball.csv"]
    for pattern in temp_csv_patterns:
        temp_csv_files = glob.glob(pattern)
        for temp_csv in temp_csv_files:
            if os.path.exists(temp_csv):
                try:
                    os.remove(temp_csv)
                    cleaned_count += 1
                    print(f"   🧹 清理临时CSV: {os.path.basename(temp_csv)}")
                except:
                    pass
    
    # 清理Python缓存
    cache_dirs = glob.glob("__pycache__") + glob.glob("*/__pycache__")
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                cleaned_count += 1
            except:
                pass
    
    if cleaned_count > 0:
        print(f"✅ 部分清理完成: 删除了 {cleaned_count} 个中间文件/目录")
    else:
        print("✅ 目录已经很干净")

def analyze_video_strategy(video_file):
    """分析视频并决定处理策略"""
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_length = frame_count / fps if fps > 0 else 0
    cap.release()
    

    if total_length > 30:  # 30秒以上
        strategy = "segmentation"
        segment_duration = 30  # 20秒分段是最优配置
        reason = f"中长视频({total_length/60:.1f}分钟)，使用30秒分段处理"
    else:
        strategy = "full_memory"
        segment_duration = None
        reason = f"短视频({total_length/60:.1f}分钟)，使用全内存处理"
    
    return {
        'strategy': strategy,
        'segment_duration': segment_duration,
        'total_length': total_length,
        'frame_count': frame_count,
        'fps': fps,
        'reason': reason
    }

def get_gpu_optimization_config(video_file):
    """在外层进行一次GPU优化配置"""
    try:
        from adaptive_gpu_optimizer import optimize_for_current_gpu
        from utils.general import HEIGHT, WIDTH
        
        # 加载模型配置以确定输入形状
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tracknet_file = "ckpts/TrackNet_best.pt"
        tracknet_ckpt = torch.load(tracknet_file, map_location=device)
        tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
        bg_mode = tracknet_ckpt['param_dict']['bg_mode']
        
        # 根据模型配置确定输入形状
        if bg_mode == 'concat':
            channels = (tracknet_seq_len + 1) * 3
        else:
            channels = tracknet_seq_len * 3
        
        input_shape = (1, channels, HEIGHT, WIDTH)
        
        # 创建临时模型进行优化
        from utils.general import get_model
        tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).to(device)
        tracknet.load_state_dict(tracknet_ckpt['model'])
        tracknet.eval()
        
        optimization_result = optimize_for_current_gpu(input_shape)
        
        batch_size = optimization_result['optimal_batch_size']
        gpu_name = optimization_result['gpu_characteristics']['name']
        
        print(f"🚀 GPU优化完成:")
        print(f"   - GPU: {gpu_name}")
        print(f"   - 推荐批处理大小: {batch_size}")
        
        # 清理临时模型
        del tracknet, tracknet_ckpt
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            'batch_size': batch_size,
            'gpu_name': gpu_name,
            'optimization_result': optimization_result
        }
        
    except Exception as e:
        print(f"⚠️ GPU优化失败，使用默认参数: {e}")
        return {
            'batch_size': 24,
            'gpu_name': 'Unknown',
            'optimization_result': None
        }

def process_single_segment(args):
    """处理单个分段的函数 - 用于多进程，优化内存管理"""
    segment_info, video_file, output_dir, batch_size = args
    i, start_time, end_time, fps = segment_info
    
    import tempfile
    import cv2
    import gc
    import os
    
    print(f"\n🚀 进程{os.getpid()}: 处理分段 {i+1}: {start_time:.1f}s - {end_time:.1f}s")
    
    try:
        # 直接按时间提取分段
        cap = cv2.VideoCapture(video_file)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        
        frames = []
        frame_count = 0
        max_frames = int((end_time - start_time) * fps * 1.1)  # 限制最大帧数
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if current_time > end_time + 0.1:
                break
            
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        if not frames:
            print(f"   ⚠️ 分段 {i+1} 没有有效帧")
            return None
        
        print(f"   📊 分段 {i+1} 加载了 {len(frames)} 帧")
        
        # 创建临时视频文件
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_video_path = temp_file.name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, 
                            (frames[0].shape[1], frames[0].shape[0]))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        # 释放帧内存
        del frames
        gc.collect()
        
        # 使用核心预测模块
        from predict_core import predict_core
        segment_pred = predict_core(
            temp_video_path,
            save_dir=output_dir,
            batch_size=batch_size,
            auto_optimize=False
        )
        
        # 清理临时文件
        temp_video_name = os.path.splitext(os.path.basename(temp_video_path))[0]
        temp_csv_file = os.path.join(output_dir, f'{temp_video_name}_ball.csv')
        
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        
        if os.path.exists(temp_csv_file):
            os.unlink(temp_csv_file)
        
        # 调整帧号偏移
        frame_offset = int(start_time * fps)
        for j in range(len(segment_pred['Frame'])):
            segment_pred['Frame'][j] += frame_offset
        
        print(f"   ✅ 进程{os.getpid()}: 分段 {i+1} 完成: {len(segment_pred['Frame'])} 帧")
        
        # 强制内存清理
        gc.collect()
        
        return segment_pred
        
    except Exception as e:
        print(f"   ❌ 进程{os.getpid()}: 分段 {i+1} 处理失败: {e}")
        # 清理可能的临时文件
        try:
            if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
        except:
            pass

def run_segmented_prediction(video_file, video_info, output_dir, gpu_config):
    """分段处理预测 - 支持多进程"""
    print(f"🔥 使用分段处理策略")
    print(f"   分段长度: {video_info['segment_duration']}秒")
    print(f"   批处理大小: {gpu_config['batch_size']} (预优化)")
    print(f"   原因: {video_info['reason']}")
    
    segment_duration = video_info['segment_duration']
    total_length = video_info['total_length']
    fps = video_info['fps']
    batch_size = gpu_config['batch_size']
    
    num_segments = int(total_length // segment_duration)
    if total_length % segment_duration > 0:
        num_segments += 1
    
    print(f"   总共 {num_segments} 个分段")
    
    # 使用单进程处理 - 经过测试证明是最优方案
    print(f"   🚀 使用单进程处理")
    return run_segmented_prediction_sequential(video_file, video_info, output_dir, gpu_config)



def run_segmented_prediction_sequential(video_file, video_info, output_dir, gpu_config):
    """单进程分段处理 - 最优配置"""
    segment_duration = video_info['segment_duration']
    total_length = video_info['total_length']
    fps = video_info['fps']
    batch_size = gpu_config['batch_size']
    
    num_segments = int(total_length // segment_duration)
    if total_length % segment_duration > 0:
        num_segments += 1
    
    all_predictions = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
    
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, total_length)
        
        print(f"\n🚀 处理分段 {i+1}/{num_segments}: {start_time:.1f}s - {end_time:.1f}s")
        
        # 直接按时间提取分段
        cap = cv2.VideoCapture(video_file)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if current_time > end_time + 0.1:
                break
            
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            continue
        
        # 处理分段
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_video_path = temp_file.name
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, 
                                (frames[0].shape[1], frames[0].shape[0]))
            
            for frame in frames:
                out.write(frame)
            out.release()
            
            # 使用核心预测模块
            from predict_core import predict_core
            segment_pred = predict_core(
                temp_video_path,
                save_dir=output_dir,
                batch_size=batch_size,
                auto_optimize=False
            )
            
            # 清理临时文件
            temp_video_name = os.path.splitext(os.path.basename(temp_video_path))[0]
            temp_csv_file = os.path.join(output_dir, f'{temp_video_name}_ball.csv')
            
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            
            if os.path.exists(temp_csv_file):
                os.unlink(temp_csv_file)
                print(f"   🧹 清理临时文件: {os.path.basename(temp_csv_file)}")
            
            # 调整帧号偏移
            frame_offset = int(start_time * fps)
            for j in range(len(segment_pred['Frame'])):
                segment_pred['Frame'][j] += frame_offset
            
            # 合并结果
            all_predictions['Frame'].extend(segment_pred['Frame'])
            all_predictions['X'].extend(segment_pred['X'])
            all_predictions['Y'].extend(segment_pred['Y'])
            all_predictions['Visibility'].extend(segment_pred['Visibility'])
            
            print(f"   ✅ 分段 {i+1} 完成: {len(segment_pred['Frame'])} 帧")
            
        except Exception as e:
            print(f"   ❌ 分段 {i+1} 处理失败: {e}")
            continue
        
        # 每段清理内存
        del frames, segment_pred
        gc.collect()
        torch.cuda.empty_cache()
    
    # 保存合并结果
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    out_csv_file = f"{output_dir}/{video_name}_ball.csv"
    out_video_file = f"{output_dir}/{video_name}.mp4"
    
    os.makedirs(output_dir, exist_ok=True)
    
    import pandas as pd
    pred_df = pd.DataFrame({
        'Frame': all_predictions['Frame'],
        'Visibility': all_predictions['Visibility'],
        'X': all_predictions['X'],
        'Y': all_predictions['Y']
    })
    pred_df.to_csv(out_csv_file, index=False)
    
    import pickle
    with open('predicted.bin', 'wb') as file:
        pickle.dump(all_predictions, file)
        pickle.dump(out_video_file, file)
        pickle.dump(video_file, file)
    
    return all_predictions

def main():
    parser = argparse.ArgumentParser(description='羽毛球分析系统 - 优化版本')
    parser.add_argument('--video', required=True, help='输入视频路径')
    parser.add_argument('--output', default='prediction', help='输出目录')
    parser.add_argument('--predict-only', action='store_true', help='仅执行预测，不生成视频')
    
    args = parser.parse_args()
    
    print("🚀 羽毛球分析系统 - 优化版本")
    print("=" * 50)
    
    # 检查GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("❌ 未检测到CUDA GPU")
        return
    
    total_start = time.time()
    
    try:
        print(f"\n🚀 阶段1: 智能轨迹预测")
        print(f"💾 初始内存使用: {get_memory_info():.2f}GB")
        
        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()
        
        # 分析视频策略
        print("🔍 分析视频特征...")
        video_info = analyze_video_strategy(args.video)
        
        print(f"📊 视频信息: {video_info['frame_count']} 帧, {video_info['fps']:.1f}fps, {video_info['total_length']:.1f}s")
        print(f"🎯 选择策略: {video_info['strategy']}")
        print(f"💡 原因: {video_info['reason']}")
        
        # 在外层进行一次GPU优化
        print("🚀 进行GPU优化分析...")
        gpu_config = get_gpu_optimization_config(args.video)
        
        pred_start = time.time()
        
        if video_info['strategy'] == 'segmentation':
            # 使用分段处理，传递GPU配置
            pred_dict = run_segmented_prediction(args.video, video_info, args.output, gpu_config)
        else:
            # 使用全内存处理
            print("🚀 使用全内存处理策略")
            from predict_core import predict_core
            pred_dict = predict_core(args.video, save_dir=args.output, 
                                   batch_size=gpu_config['batch_size'], 
                                   auto_optimize=False)  # 使用预优化的参数
        
        pred_time = time.time() - pred_start
        
        if not args.predict_only:
            print(f"\n🚀 阶段2: 视频生成")
            video_start = time.time()
            
            # 清理内存
            gc.collect()
            torch.cuda.empty_cache()
            
            from testing_optimized import testing_optimized
            testing_optimized()
            
            video_time = time.time() - video_start
            
            # 视频生成完成后，清理predicted.bin
            # if os.path.exists('predicted.bin'):
            #     try:
            #         os.remove('predicted.bin')
            #         print("🧹 清理predicted.bin")
            #     except:
            #         pass
        else:
            video_time = 0
            print(f"\n✅ 仅预测模式，跳过视频生成")
        
        total_time = time.time() - total_start
        
        # 检查输出文件
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        output_files = {
            'csv': f'{args.output}/{video_name}_ball.csv',
            'video': f'{args.output}/{video_name}.mp4',
            'score_clip': f'{args.output}/{video_name}_score_clip.mp4'
        }
        
        print(f"\n📁 输出文件:")
        for file_type, file_path in output_files.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
                print(f"   ✅ {file_type.upper()}: {os.path.basename(file_path)} ({file_size:.1f} MB)")
            else:
                if args.predict_only and file_type != 'csv':
                    print(f"   ➖ {file_type.upper()}: 跳过 (仅预测模式)")
                else:
                    print(f"   ❌ {file_type.upper()}: 未生成")
        
        # 性能报告
        print(f"\n🎉 处理完成!")
        print("=" * 50)
        print(f"⚡ 总耗时: {total_time:.2f}s ({total_time/60:.1f}分钟)")
        print(f"⚡ 轨迹预测: {pred_time:.2f}s ({pred_time/total_time*100:.1f}%)")
        if not args.predict_only:
            print(f"⚡ 视频生成: {video_time:.2f}s ({video_time/total_time*100:.1f}%)")
        print(f"⚡ 处理速度: {len(pred_dict['Frame'])/total_time:.1f} FPS")
        print(f"⚡ 内存使用: {get_memory_info():.2f}GB")
        
        # 清理中间产物 (但保留predicted.bin直到视频生成完成)
        if args.predict_only:
            cleanup_intermediate_files()
        else:
            # 只清理部分文件，保留predicted.bin给视频生成使用
            cleanup_intermediate_files_partial()
        
        print("✅ 处理成功！")
        
    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        print(f"💾 错误时内存使用: {get_memory_info():.2f}GB")
        import traceback
        traceback.print_exc()
        
        # 即使出错也清理
        try:
            cleanup_intermediate_files()
        except:
            pass

if __name__ == "__main__":
    main()