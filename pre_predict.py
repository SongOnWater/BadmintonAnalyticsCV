import time
from predict import *
import pandas
from config import Config
from utils.performance_monitor import monitor, performance_timer
import gc

@performance_timer
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, help='file path of the video')
    parser.add_argument('--save_dir', default='prediction', type=str)
    parser.add_argument('--batch_size', type=int, default=None, help='batch size for processing')
    args = parser.parse_args()
    
    # Initialize configuration and performance monitoring
    Config.setup_torch_optimizations()
    Config.create_temp_dir()
    monitor.log_memory_usage("Initialization")
    
    save_dir = args.save_dir
    video_name = os.path.splitext(os.path.basename(args.video_file))[0]
    out_csv_file = os.path.join(save_dir, f'{video_name}_ball.csv')
    out_video_file = os.path.join(save_dir, f'{video_name}.mp4')

    # Check if save_dir exists, create if not
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Handle FPS conversion if needed
    video_file_to_process = args.video_file
    cap = cv2.VideoCapture(args.video_file)
    
    # converting video to 30 fps if it isn't already
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Original video FPS: {fps}")
    except:
        print("Error in calculating FPS")
        fps = 30.0  # Default to 30 if we can't determine FPS

    cap.release()

    if abs(fps - 30.0) > 0.1: 
        print("Converting video to 30 FPS...")
        monitor.start_timer("fps_conversion")
        video_file_to_process = change_fps(args.video_file)
        monitor.end_timer("fps_conversion")

    
    # 🚀 MAJOR OPTIMIZATION: Process entire video at once instead of splitting
    print("🚀 Processing entire video without splitting (MAJOR SPEEDUP)")
    
    # Record overall processing start time
    overall_start_time = time.time()
    
    # Load all frames at once - much faster than splitting video
    print("Loading all frames from video...")
    cap = cv2.VideoCapture(video_file_to_process)
    frame_list, fps, (w, h) = generate_frames_from_cap(cap)
    cap.release()
    
    print(f"Loaded {len(frame_list)} frames from video")
    
    # Process all frames in one go
    pred_dict_joined = pred_main(frame_list=frame_list, fps=fps, w=w, h=h, batch_size=args.batch_size or 8)
    
    # Clear frame_list to free memory
    del frame_list
        
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
        pickle.dump(args.video_file, file)
    
    # Clean up temporary files
    if video_file_to_process != args.video_file:
        try:
            os.remove(video_file_to_process)
            print(f"Removed temporary file: {video_file_to_process}")
        except:
            print(f"Could not remove temporary file: {video_file_to_process}")
    
    try:
        os.remove("temp_clip.mp4")
        print("Removed temporary file: temp_clip.mp4")
    except:
        print("Could not remove temporary file: temp_clip.mp4")
    
    # Log final performance metrics
    monitor.log_memory_usage("Completion")
    monitor.print_summary()
    
    print(f"\nVideo processing completed successfully!")
    print(f"Total processing time: {overall_duration:.2f} seconds")
    print(f"Frames per second: {len(pred_dict_joined['Frame'])/overall_duration:.1f} FPS")
    
    # Cleanup
    Config.cleanup_temp_files()

if __name__ == '__main__':
    main()