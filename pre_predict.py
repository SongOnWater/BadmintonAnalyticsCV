from predict import *
import pandas

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, help='file path of the video')
    parser.add_argument('--save_dir', default = 'prediction', type = str)
    args = parser.parse_args()
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
        video_file_to_process = change_fps(args.video_file)

    
    # predicting in batches to not use too much memory

    vfc = VideoFileClip(video_file_to_process)
    clip_duration = 10
    total_length = vfc.duration
    num_clips = int(total_length // clip_duration)

    remainder = total_length - num_clips*clip_duration
    # print(remainder)

    # Ensure at least one clip is processed even for very short videos
    if remainder > 0 or num_clips == 0:
        num_clips += 1

    print(f"Total video length: {total_length:.2f} seconds")
    print(f"Processing video in {num_clips} segments (each {clip_duration} seconds)")
    
    pred_dict_joined = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

    for i in range(num_clips):
        print(f"\nProcessing segment {i+1}/{num_clips}...")
        
        if (i != num_clips-1):
            start_time = i * clip_duration
            end_time = (i+1) * clip_duration
            print(f"  Segment time: {start_time:.1f}s to {end_time:.1f}s")
            clip = vfc.subclip(start_time, end_time)
            clip.write_videofile("temp_clip.mp4", codec='libx264', audio=False)
            cap = cv2.VideoCapture("temp_clip.mp4")

            frame_list, fps, (w,h) = generate_frames_from_cap(cap)
            pred_dict = pred_main(frame_list=frame_list, fps=fps, w=w, h=h)
        
        else:
            start_time = i * clip_duration
            end_time = total_length
            print(f"  Segment time: {start_time:.1f}s to {end_time:.1f}s (last segment)")
            clip = vfc.subclip(start_time, end_time)
            clip.write_videofile("temp_clip.mp4", codec='libx264', audio=False)
            cap = cv2.VideoCapture("temp_clip.mp4")
            frame_list, fps, (w,h) = generate_frames_from_cap(cap)
            pred_dict = pred_main(frame_list=frame_list, fps=fps, w=w, h=h)

        print(f"  Processed {len(pred_dict['Frame'])} frames in this segment")
        
        for k in range(len(pred_dict['Frame'])):
            pred_dict['Frame'][k] += i*30*clip_duration

        pred_dict_joined['Frame'].extend(pred_dict['Frame'])
        pred_dict_joined['Visibility'].extend(pred_dict['Visibility'])
        pred_dict_joined['X'].extend(pred_dict['X'])
        pred_dict_joined['Y'].extend(pred_dict['Y'])

        # print(pred_dict_joined)
        
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
    
    print("\nVideo processing completed successfully!")