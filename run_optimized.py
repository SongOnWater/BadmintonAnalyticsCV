#!/usr/bin/env python3
"""
Optimized Badminton Analytics CV - Quick Start Script
使用优化后的羽毛球分析系统快速启动脚本

Usage:
    python run_optimized.py --video input/your_video.mp4
    python run_optimized.py --video input/your_video.mp4 --batch_size 8 --gpu
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from utils.performance_monitor import monitor
import subprocess

def check_requirements():
    """检查系统要求和依赖"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    
    # Check required packages
    required_packages = ['torch', 'cv2', 'numpy', 'pandas', 'moviepy', 'ultralytics']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install with: pip install -r requirements.txt")
        return False
    
    # Check model files
    model_files = [
        "ckpts/TrackNet_best.pt",
        "ckpts/yolov8n.pt"
    ]
    
    missing_models = []
    for model_file in model_files:
        if not os.path.exists(model_file):
            missing_models.append(model_file)
    
    if missing_models:
        print(f"❌ Missing model files: {', '.join(missing_models)}")
        return False
    
    print("✅ All requirements satisfied")
    return True

def setup_environment():
    """设置运行环境"""
    print("⚙️ Setting up environment...")
    
    # Create necessary directories
    os.makedirs("prediction", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    # Setup torch optimizations
    Config.setup_torch_optimizations()
    
    # Log system info
    system_info = monitor.get_system_info()
    print(f"💻 System: {system_info['cpu_count']} CPU cores, {system_info['memory_total_gb']:.1f}GB RAM")
    
    if system_info['cuda_available']:
        print(f"🚀 GPU: {system_info['gpu_name']} ({system_info['gpu_memory_gb']:.1f}GB)")
    else:
        print("⚠️ Running on CPU (GPU not available)")

def run_prediction(video_file, batch_size=None, save_dir="prediction"):
    """运行预测"""
    print(f"🎯 Starting prediction for: {video_file}")
    
    # Build command
    cmd = [
        sys.executable, "pre_predict.py",
        "--video_file", video_file,
        "--save_dir", save_dir
    ]
    
    if batch_size:
        cmd.extend(["--batch_size", str(batch_size)])
    
    # Run prediction
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Prediction completed successfully")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ Prediction failed")
        print(e.stderr)
        return False
    
    duration = time.time() - start_time
    print(f"⏱️ Total time: {duration:.2f} seconds")
    return True

def run_analysis(save_dir="prediction"):
    """运行分析和比分更新"""
    print("📊 Starting analysis and score tracking...")
    
    try:
        # Import and run testing
        from testing import testing
        testing()
        print("✅ Analysis completed successfully")
        return True
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Optimized Badminton Analytics CV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_optimized.py --video input/match.mp4
  python run_optimized.py --video input/match.mp4 --batch_size 8
  python run_optimized.py --video input/match.mp4 --skip_analysis
        """
    )
    
    parser.add_argument('--video', required=True, help='Input video file path')
    parser.add_argument('--batch_size', type=int, help='Batch size for processing')
    parser.add_argument('--save_dir', default='prediction', help='Output directory')
    parser.add_argument('--skip_analysis', action='store_true', help='Skip analysis step')
    parser.add_argument('--force', action='store_true', help='Skip requirement checks')
    
    args = parser.parse_args()
    
    print("🏸 Badminton Analytics CV - Optimized Version")
    print("=" * 50)
    
    # Check requirements
    if not args.force and not check_requirements():
        print("❌ Requirements check failed. Use --force to skip.")
        return 1
    
    # Check input file
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return 1
    
    # Setup environment
    setup_environment()
    
    # Run prediction
    if not run_prediction(args.video, args.batch_size, args.save_dir):
        return 1
    
    # Run analysis
    if not args.skip_analysis:
        if not run_analysis(args.save_dir):
            return 1
    
    print("\n🎉 Processing completed successfully!")
    print(f"📁 Results saved in: {args.save_dir}")
    
    # Show output files
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    output_files = [
        f"{args.save_dir}/{video_name}_ball.csv",
        f"{args.save_dir}/{video_name}.mp4",
        f"{video_name}_score_clip.mp4"
    ]
    
    print("\n📄 Output files:")
    for file in output_files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / 1024 / 1024
            print(f"  ✅ {file} ({size_mb:.1f}MB)")
        else:
            print(f"  ❌ {file} (not found)")
    
    return 0

if __name__ == "__main__":
    exit(main())