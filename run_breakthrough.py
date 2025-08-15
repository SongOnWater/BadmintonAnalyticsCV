#!/usr/bin/env python3
"""
突破性能版羽毛球分析 - 移除时间集成
"""
import argparse
import os
import sys
import time

def main():
    parser = argparse.ArgumentParser(description='突破性能版羽毛球分析')
    parser.add_argument('--video_file', required=True, type=str, help='输入视频路径')
    parser.add_argument('--save_dir', default='prediction', type=str, help='输出目录')
    parser.add_argument('--generate_pred_video', action='store_true', help='生成完整预测视频（默认不生成，节省时间）')
    parser.add_argument('--eval_mode', choices=['nonoverlap', 'weight'], default='nonoverlap',
                        help='预测模式：nonoverlap(快速，默认) 或 weight(高精度)')
    args = parser.parse_args()

    video_file = os.path.abspath(args.video_file)
    save_dir = os.path.abspath(args.save_dir)

    if not os.path.exists(video_file):
        print(f"输入视频未找到: {video_file}")
        sys.exit(1)

    os.makedirs(save_dir, exist_ok=True)

    print("=== 突破性能版羽毛球分析 ===")
    if args.eval_mode == 'nonoverlap':
        print(f"预测模式: NoOverlap (快速模式，2.3x性能提升)")
    else:
        print(f"预测模式: Weight (高精度模式，时间集成)")
    print(f"视频: {video_file}")
    print(f"输出: {save_dir}")
    
    overall_start = time.time()
    
    try:
        print('\n步骤 1/2: 突破性能预测...')
        from pre_predict import main as pre_predict_main
        
        pred_dict, csv_file, video_file_out = pre_predict_main(video_file, save_dir, eval_mode=args.eval_mode)
        print(f"预测完成: {csv_file}")

        print('\n步骤 2/2: 修复版分析处理...')
        from testing_fixed import main as testing_fixed_main
        testing_fixed_main(generate_pred_video=args.generate_pred_video, original_video_file=video_file)
        print("修复版分析完成")
                
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    overall_end = time.time()
    overall_duration = overall_end - overall_start

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    
    print('\n=== 突破性能结果 ===')
    
    # CSV文件
    csv_file = os.path.join(save_dir, f'{video_name}_ball.csv')
    csv_status = '[OK]' if os.path.exists(csv_file) else '[FAIL]'
    print(f"{csv_status} CSV: {csv_file}")
    
    # 预测视频 - 根据参数显示状态
    pred_video_file = os.path.join(save_dir, f'{video_name}.mp4')
    if args.generate_pred_video:
        pred_status = '[OK]' if os.path.exists(pred_video_file) else '[FAIL]'
        print(f"{pred_status} 预测视频: {pred_video_file}")
    else:
        print(f"[SKIP] 预测视频: 已跳过（使用 --generate_pred_video 生成）")
    
    # 得分片段
    score_clip_file = os.path.join(save_dir, f'{video_name}_score_clip.mp4')
    score_status = '[OK]' if os.path.exists(score_clip_file) else '[FAIL]'
    print(f"{score_status} 得分片段: {score_clip_file}")
    
    # 计算性能指标
    if 'pred_dict' in locals() and len(pred_dict['Frame']) > 0:
        total_frames = len(pred_dict['Frame'])
        fps_processed = total_frames / overall_duration
        
        # 估算视频长度（假设30fps）
        video_length_seconds = total_frames / 30.0
        speed_multiplier = video_length_seconds / overall_duration
        
        print(f"\n=== 突破性能指标 ===")
        print(f"总处理时间: {overall_duration:.2f} 秒")
        print(f"处理帧数: {total_frames:,}")
        print(f"处理速度: {fps_processed:.1f} FPS")
        print(f"实时速度倍数: {speed_multiplier:.1f}x")
        
        # 性能等级
        if speed_multiplier >= 2.0:
            print("性能等级: 突破性 (2x+)")
        elif speed_multiplier >= 1.5:
            print("性能等级: 极致 (1.5x+)")
        elif speed_multiplier >= 1.0:
            print("性能等级: 实时+ (1x+)")
        else:
            print("性能等级: 标准")
            

    else:
        print(f"\n总处理时间: {overall_duration:.2f} 秒")

if __name__ == '__main__':
    main()