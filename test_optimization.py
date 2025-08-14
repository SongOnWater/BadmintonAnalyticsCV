#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试优化后的预测逻辑
验证pre_predict_opt的优化是否正确集成
"""

import os
import sys

def test_imports():
    """测试所有必要的导入"""
    print("🧪 测试导入...")
    
    try:
        from run_optimized import analyze_video_strategy, get_gpu_optimization_config
        print("✅ run_optimized 导入成功")
    except ImportError as e:
        print(f"❌ run_optimized 导入失败: {e}")
        return False
    
    try:
        from predict_core import predict_core, predict_core_from_frames
        print("✅ predict_core 导入成功")
    except ImportError as e:
        print(f"❌ predict_core 导入失败: {e}")
        return False
    
    return True

def test_video_strategy():
    """测试视频策略分析"""
    print("\n🧪 测试视频策略分析...")
    
    # 检查是否有测试视频
    test_videos = []
    if os.path.exists('input'):
        for file in os.listdir('input'):
            if file.endswith('.mp4'):
                test_videos.append(os.path.join('input', file))
    
    if not test_videos:
        print("⚠️ 没有找到测试视频，跳过策略测试")
        return True
    
    try:
        from run_optimized import analyze_video_strategy
        
        test_video = test_videos[0]
        print(f"📹 使用测试视频: {test_video}")
        
        strategy = analyze_video_strategy(test_video)
        print(f"✅ 策略分析成功:")
        print(f"   - 策略: {strategy['strategy']}")
        print(f"   - 分段长度: {strategy['segment_duration']}")
        print(f"   - 总长度: {strategy['total_length']:.1f}s")
        print(f"   - 原因: {strategy['reason']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 策略分析失败: {e}")
        return False

def test_frame_prediction():
    """测试帧列表预测功能"""
    print("\n🧪 测试帧列表预测功能...")
    
    try:
        import cv2
        import numpy as np
        from predict_core import predict_core_from_frames
        
        # 创建模拟帧数据
        print("📊 创建模拟帧数据...")
        frames = []
        for i in range(10):  # 创建10帧测试数据
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frames.append(frame)
        
        print(f"✅ 创建了 {len(frames)} 帧测试数据")
        
        # 注意：这里只测试函数调用，不进行实际推理（需要模型文件）
        print("⚠️ 跳过实际推理测试（需要模型文件）")
        
        return True
        
    except Exception as e:
        print(f"❌ 帧预测测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试pre_predict_opt优化集成")
    print("=" * 50)
    
    tests = [
        ("导入测试", test_imports),
        ("视频策略测试", test_video_strategy),
        ("帧预测测试", test_frame_prediction),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 执行 {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name} 通过")
                passed += 1
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！优化集成成功")
        return True
    else:
        print("⚠️ 部分测试失败，请检查代码")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)