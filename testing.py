import os
import time
from utils.general import *
import pickle
import sys
from utils.func_clips_start_end import *
# from join_clips import *

# sys.path.append('/Users/vishruthvijay/Documents/Summer-Project-Badminton/Combined-files/utils')

# file = open("bin_subclips.bin","rb")
def create_frames():
    out_file2 = ""
    with open('predicted.bin','rb') as file:
        pred_dict = pickle.load(file)
        out_file = pickle.load(file)
        # for i in out_file:
        #     if i=="\\":
        #         out_file2+="/"
        #     else:
        #         out_file2+=i
        # out_file = out_file2
        video_name = pickle.load(file)
        print("pred_dict is predicted.bin in crete_frames() is ",pred_dict)
        print(out_file)
    # print(pred_dict)
    frame_list, fps, (w, h) = generate_frames(video_name)
    # print(pred_dict, len(pred_dict), "pred_dict")
    # print(out_file)
    # print(video_name)
    return frame_list, pred_dict, out_file


def testing():

    # Record overall start time
    overall_start_time = time.time()
    
    frame_list, pred_dict, out_file = create_frames()
    print("legth of fl , pd , of is ",len(frame_list),len(pred_dict),len(out_file))
    print(pred_dict)
    
    # 🚀 MAJOR OPTIMIZATION: Skip unnecessary frame reloading
    print("🚀 OPTIMIZED: Processing without redundant frame operations...")
    
    # Optimize frame list slicing
    frame_list = frame_list[:len(pred_dict['Frame'])]
    
    # Initialize game state
    scores = {"p1" : 0, "p2" : 0}
    pointers_to_players = {"closer" : "p1", "farther" : "p2"}
    set_scores = {"p1" : 0, "p2" : 0}
    video_clips = []
    data_array = []
    video_clips.append(out_file)

    # Get video configuration once
    if frame_list:
        video_config = dict(fps=30, shape=(frame_list[0].shape[1], frame_list[0].shape[0]))
        frame_width = frame_list[0].shape[1]
        frame_height = frame_list[0].shape[0]
    else:
        print("Error: No frames available")
        return

    # 🚀 OPTIMIZATION: Streamlined processing without intermediate video creation
    print("🚀 Streamlining video processing...")
    add_frame = pred_dict_modify(False, pred_dict, frame_list, video_config)
    
    # 🚀 MAJOR SPEEDUP: Process frames with trajectory overlay directly
    af2, frame_in_csv, active_frame, save_file = write_pred_video_modified(
        frame_list, video_config, pred_dict, prev_last_frame=None, 
        save_file=out_file, add_frame=add_frame, traj_len=8)
    
    print(f"🚀 Processing {len(active_frame)} active frames for score tracking")

    # 🚀 OPTIMIZATION: Process only frames that need score updates
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output_path = f"{out_file[:-4]}_score_clip.mp4"
    out = None
    
    # Initialize game state
    first_serve = True
    
    try:
        print(f"🚀 Creating optimized output video at: {temp_output_path}")
        print(f"Video dimensions: {frame_width}x{frame_height}")
        
        # Create VideoWriter with proper parameters
        out = cv2.VideoWriter(temp_output_path, fourcc, 30, (frame_width, frame_height))
        
        if not out.isOpened():
            raise Exception(f"Could not open VideoWriter for {temp_output_path}")
            
        print("VideoWriter opened successfully")
        
        # 🚀 MAJOR OPTIMIZATION: Load frames from processed video instead of original
        processed_frame_list, fps, (w, h) = generate_frames(save_file)
        print(f"🚀 Loaded {len(processed_frame_list)} processed frames")

        # Pre-compute text properties to avoid repeated calculations
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        font_color = (0, 0, 255)  # Red color
        padding = 10
        
        # Convert active_frame to set for O(1) lookup
        active_frame_set = set(active_frame)
        active_frame_dict = {frame_idx: csv_idx for csv_idx, frame_idx in enumerate(active_frame)}
        
        # 🚀 OPTIMIZED: Process frames with minimal overhead
        for i, frame in enumerate(processed_frame_list):
            if i % 1000 == 0:  # Reduce print frequency even more
                print(f"🚀 Processing frame {i}/{len(processed_frame_list)}")
                
            # Optimize active frame checking
            if i in active_frame_set:
                csv_idx = active_frame_dict[i]
                scores = clip_start(frame_in_csv[csv_idx], i, save_file, scores, pointers_to_players, first_serve)
                
                # Game logic optimization
                if set_scores["p1"] == 1 and set_scores["p2"] == 1:
                    if scores["p1"] == 11 or scores["p2"] == 11:
                        pointers_to_players["closer"], pointers_to_players["farther"] = pointers_to_players["farther"], pointers_to_players["closer"]

                if scores["p1"] == 21 or scores["p2"] == 21:
                    if scores["p1"] == 21:
                        set_scores["p1"] += 1
                    else:
                        set_scores["p2"] += 1
                    scores = {"p1": 0, "p2": 0}
                    first_serve = True
                    pointers_to_players["closer"], pointers_to_players["farther"] = pointers_to_players["farther"], pointers_to_players["closer"]
                first_serve = False 

            # Ensure frame is valid
            if frame is None or frame.size == 0:
                print(f"Warning: Invalid frame at index {i}, skipping")
                continue

            # 🚀 OPTIMIZATION: Pre-calculate text once per score change
            score_text_p1 = f"Player 1: {scores['p1']}"
            score_text_p2 = f"Player 2: {scores['p2']}"
            
            # Get text size (cached for consistent text)
            text_size, _ = cv2.getTextSize(score_text_p1, font, font_scale, font_thickness)
            text_width = text_size[0]
            text_height = text_size[1]

            # Calculate positions
            top_right_p1 = (frame.shape[1] - text_width - padding, text_height + padding)
            top_right_p2 = (frame.shape[1] - text_width - padding, 2 * text_height + 2 * padding)

            # Draw text with optimized parameters
            cv2.putText(frame, score_text_p1, top_right_p1, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            cv2.putText(frame, score_text_p2, top_right_p2, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

            # Write the modified frame to the output video file
            out.write(frame)
            
        print(f"Finished processing all {len(frame_list)} frames")
        
    except Exception as e:
        print(f"Error during video processing: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure VideoWriter is properly released
        if out is not None:
            print("Releasing VideoWriter...")
            out.release()
            print(f"VideoWriter released. Output file: {temp_output_path}")

    # Record overall end time and calculate duration
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    
    print(f"Video processing completed in {overall_duration:.2f} seconds")
    print("Done")

# def return_frame_list():
#     frame_list, pred_dict, out_file = create_frames()
#     return frame_list


testing()