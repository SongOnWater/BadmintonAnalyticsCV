#!/usr/bin/env python3
"""
真正高效的羽毛球分析系统
专注于消除核心性能瓶颈，实现真正的速度提升

使用方法:
python run_fast.py --video input/your_video.mp4
python run_fast.py --video input/your_video.mp4 --batch_size 16
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(
        description="真正高效的羽毛球分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_fast.py --video input/match.mp4
  python run_fast.py --video input/match.mp4 --batch_size 16
  python run_fast.py --video input/match.mp4 --skip_analysis
        """
    )
    
    parser.add_argument('--video', required=True, help='输入视频文件路径')
    parser.add_argument('--batch_size', type=int, help='批处理大小（自动检测如果未指定）')
    parser.add_argument('--save_dir', default='prediction', help='输出目录')
    parser.add_argument('--skip_analysis', action='store_true', help='跳过分析步骤')
    
    args = parser.parse_args()
    
    print("🚀 真正高效的羽毛球分析系统")
    print("=" * 50)
    
    # 检查输入文件
    if not os.path.exists(args.video):
        print(f"❌ 视频文件未找到: {args.video}")
        return 1
    
    # 显示系统信息
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️ 使用CPU运行（GPU不可用）")
    
    total_start_time = time.time()
    
    try:
        # 🚀 步骤1: 超高速预测
        print("\n🎯 步骤1: 超高速轨迹预测")
        from predict_optimized import predict_ultra_fast
        
        pred_dict = predict_ultra_fast(
            args.video, 
            args.save_dir, 
            args.batch_size
        )
        
        # 🚀 步骤2: 分析和比分跟踪（如果需要）
        if not args.skip_analysis:
            print("\n🎯 步骤2: 分析和比分跟踪")
            analysis_start = time.time()
            
            from testing import testing
            testing()
            
            analysis_time = time.time() - analysis_start
            print(f"✅ 分析完成，耗时: {analysis_time:.2f}s")
        
        # 总结
        total_time = time.time() - total_start_time
        print(f"\n🎉 处理完成!")
        print(f"⚡ 总耗时: {total_time:.2f}s")
        print(f"⚡ 处理速度: {len(pred_dict['Frame'])/total_time:.1f} FPS")
        
        # 显示输出文件
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        output_files = [
            f"{args.save_dir}/{video_name}_ball.csv",
            f"{args.save_dir}/{video_name}.mp4",
        ]
        
        if not args.skip_analysis:
            output_files.append(f"{video_name}_score_clip.mp4")
        
        print("\n📄 输出文件:")
        for file in output_files:
            if os.path.exists(file):
                size_mb = os.path.getsize(file) / 1024 / 1024
                print(f"  ✅ {file} ({size_mb:.1f}MB)")
            else:
                print(f"  ❌ {file} (未找到)")
        
        return 0
        
    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())