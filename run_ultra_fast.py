#!/usr/bin/env python3
"""
Ultra-Fast Badminton Analytics CV
超高性能羽毛球分析系统 - 专注于最大化处理速度

主要优化策略：
1. 消除视频分割 - 直接处理整个视频
2. 最大化批处理大小 - 充分利用GPU并行能力
3. 消除重复I/O操作 - 减少磁盘读写
4. 优化内存使用 - 避免不必要的数据复制
5. 使用最新的PyTorch优化技术
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from predict import pred_main, generate_frames_from_cap
from testing import testing
from utils.performance_monitor import monitor

class UltraFastProcessor:
    def __init__(self):
        self.setup_optimizations()
    
    def setup_optimizations(self):
        """设置所有可能的优化"""
        print("🚀 Setting up ultra-fast optimizations...")
        
        # PyTorch优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # 启用所有可用的优化
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
            except:
                pass
            
            # 设置GPU内存分配策略
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # 设置OpenCV优化
        cv2.setUseOptimized(True)
        cv2.setNumThreads(0)  # 使用所有可用线程
        
        print("✅ Optimizations configured")
    
    def get_optimal_batch_size(self):
        """根据GPU内存动态计算最优批处理大小"""
        if not torch.cuda.is_available():
            return 4
        
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if gpu_memory_gb >= 24:  # RTX 4090, A100等
            return 64
        elif gpu_memory_gb >= 16:  # RTX 4080等
            return 48
        elif gpu_memory_gb >= 12:  # RTX 4070Ti等
            return 32
        elif gpu_memory_gb >= 8:   # RTX 4060Ti等
            return 24
        elif gpu_memory_gb >= 6:   # RTX 3060等
            return 16
        else:
            return 8
    
    def process_video_ultra_fast(self, video_file, save_dir="prediction", batch_size=None):
        """超高速视频处理"""
        print(f"🚀 ULTRA-FAST MODE: Processing {video_file}")
        
        # 记录开始时间
        total_start_time = time.time()
        monitor.start_timer("total_processing")
        monitor.log_memory_usage("Start")
        
        # 设置输出路径
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        out_csv_file = os.path.join(save_dir, f'{video_name}_ball.csv')
        out_video_file = os.path.join(save_dir, f'{video_name}.mp4')
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 检查FPS并转换（如果需要）
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        video_file_to_process = video_file
        if abs(fps - 30.0) > 0.1:
            print(f"🔄 Converting {fps}fps to 30fps...")
            monitor.start_timer("fps_conversion")
            video_file_to_process = self.convert_fps_fast(video_file)
            monitor.end_timer("fps_conversion")
        
        # 🚀 核心优化：一次性加载所有帧
        print("🚀 Loading all frames at once (MAJOR SPEEDUP)...")
        monitor.start_timer("frame_loading")
        
        cap = cv2.VideoCapture(video_file_to_process)
        frame_list, fps, (w, h) = generate_frames_from_cap(cap)
        cap.release()
        
        monitor.end_timer("frame_loading")
        print(f"✅ Loaded {len(frame_list)} frames in {monitor.metrics.get('frame_loading', 0):.2f}s")
        
        # 🚀 超高速模型推理
        print("🚀 Starting ultra-fast model inference...")
        monitor.start_timer("model_inference")
        
        optimal_batch_size = batch_size or self.get_optimal_batch_size()
        print(f"🚀 Using optimal batch size: {optimal_batch_size}")
        
        pred_dict = pred_main(
            frame_list=frame_list, 
            fps=fps, 
            w=w, 
            h=h, 
            batch_size=optimal_batch_size
        )
        
        monitor.end_timer("model_inference")
        print(f"✅ Model inference completed in {monitor.metrics.get('model_inference', 0):.2f}s")
        
        # 清理内存
        del frame_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 保存结果
        monitor.start_timer("save_results")
        
        import pandas as pd
        pred_df = pd.DataFrame({
            'Frame': pred_dict['Frame'],
            'Visibility': pred_dict['Visibility'],
            'X': pred_dict['X'],
            'Y': pred_dict['Y']
        })
        pred_df.to_csv(out_csv_file, index=False)
        
        # 保存二进制文件供后续处理
        with open('predicted.bin', 'wb') as file:
            pickle.dump(pred_dict, file)
            pickle.dump(out_video_file, file)
            pickle.dump(video_file, file)
        
        monitor.end_timer("save_results")
        
        # 清理临时文件
        if video_file_to_process != video_file:
            try:
                os.remove(video_file_to_process)
            except:
                pass
        
        # 记录总时间
        total_time = monitor.end_timer("total_processing")
        monitor.log_memory_usage("End")
        
        print(f"🎉 ULTRA-FAST processing completed!")
        print(f"⚡ Total time: {total_time:.2f}s")
        print(f"⚡ Processing speed: {len(pred_dict['Frame'])/total_time:.1f} FPS")
        print(f"📁 Results saved to: {save_dir}")
        
        return pred_dict, out_video_file
    
    def convert_fps_fast(self, video_file):
        """快速FPS转换"""
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
            preset='ultrafast'  # 最快编码预设
        )
        clip.close()
        
        return temp_file
    
    def run_analysis_ultra_fast(self):
        """超高速分析和比分跟踪"""
        print("🚀 Starting ultra-fast analysis...")
        monitor.start_timer("analysis")
        
        # 运行优化后的分析
        testing()
        
        analysis_time = monitor.end_timer("analysis")
        print(f"✅ Analysis completed in {analysis_time:.2f}s")

def main():
    parser = argparse.ArgumentParser(
        description="Ultra-Fast Badminton Analytics CV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀 Ultra-Fast Mode Examples:
  python run_ultra_fast.py --video input/match.mp4
  python run_ultra_fast.py --video input/match.mp4 --batch_size 32
  python run_ultra_fast.py --video input/match.mp4 --skip_analysis
        """
    )
    
    parser.add_argument('--video', required=True, help='Input video file path')
    parser.add_argument('--batch_size', type=int, help='Batch size (auto-detected if not specified)')
    parser.add_argument('--save_dir', default='prediction', help='Output directory')
    parser.add_argument('--skip_analysis', action='store_true', help='Skip analysis step')
    
    args = parser.parse_args()
    
    print("🚀 ULTRA-FAST Badminton Analytics CV")
    print("=" * 60)
    
    # 检查输入文件
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return 1
    
    # 显示系统信息
    system_info = monitor.get_system_info()
    print(f"💻 System: {system_info['cpu_count']} CPU cores, {system_info['memory_total_gb']:.1f}GB RAM")
    if system_info['cuda_available']:
        print(f"🚀 GPU: {system_info['gpu_name']} ({system_info['gpu_memory_gb']:.1f}GB)")
    else:
        print("⚠️ Running on CPU (GPU not available)")
    
    # 创建处理器实例
    processor = UltraFastProcessor()
    
    # 运行超高速处理
    try:
        pred_dict, out_video_file = processor.process_video_ultra_fast(
            args.video, 
            args.save_dir, 
            args.batch_size
        )
        
        # 运行分析（如果需要）
        if not args.skip_analysis:
            processor.run_analysis_ultra_fast()
        
        # 显示结果
        print("\n🎉 ULTRA-FAST processing completed successfully!")
        
        # 显示输出文件
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        output_files = [
            f"{args.save_dir}/{video_name}_ball.csv",
            f"{args.save_dir}/{video_name}.mp4",
        ]
        
        if not args.skip_analysis:
            output_files.append(f"{video_name}_score_clip.mp4")
        
        print("\n📄 Output files:")
        for file in output_files:
            if os.path.exists(file):
                size_mb = os.path.getsize(file) / 1024 / 1024
                print(f"  ✅ {file} ({size_mb:.1f}MB)")
            else:
                print(f"  ❌ {file} (not found)")
        
        # 显示性能总结
        monitor.print_summary()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())