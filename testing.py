import os
import time
from utils.general import *
import pickle
import sys
from utils.func_clips_start_end import *
import cv2
import numpy as np
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
    # print(pred_dict)
    # print("frame_list[0] is ",frame_list[0].shape[1])
    # print("Length of frame_list[0]" , len(frame_list[0]))
    # print("len(pred_dict['Frame'] ", len(pred_dict['Frame']))
    frame_list = frame_list[:len(pred_dict['Frame'])]
    # print("After editing fl is ",frame_list)
    # print("After editing fl length ",len(frame_list))
    # print("Length of frame_list[0]" , len(frame_list[0]))
    scores = {"p1" : 0, "p2" : 0}
    pointers_to_players = {"closer" : "p1", "farther" : "p2"}
    # pointers_to_scores = {"p1" : scores["p1"], "p2" : scores["p2"]}
    set_scores = {"p1" : 0, "p2" : 0}
    video_clips = []
    data_array = []
    video_clips.append(out_file)

    # print(frame_list)

    add_frame = pred_dict_modify(False,pred_dict, frame_list, dict(fps=30, shape=(frame_list[0].shape[1], frame_list[0].shape[0])))
    

    last_frame_of_prev_vid = None
    first_serve = True


    # add_frame = pred_dict_modify(add_frame, args2[1], args2[0], dict(fps=30, shape=(args2[0][0].shape[1], args2[0][0].shape[0])))
    af2, frame_in_csv, active_frame, save_file =write_pred_video_modified(frame_list, dict(fps=30, shape=(frame_list[0].shape[1], frame_list[0].shape[0])), pred_dict,prev_last_frame=None, save_file=out_file, add_frame=add_frame, traj_len=8)
    # last_frame_of_prev_vid = add_frame[-1]

    # print(frame_in_csv, active_frame, save_file, "printing details")

    # i2=0
    frame = frame_list[0]
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    frame_list = []
    frame_list, fps, (w, h) = generate_frames(save_file)
    # print("save file, " ,save_file)

    # 优化1: 使用更高效的编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 直接使用mp4v，避免H264初始化失败
    temp_output_path = f"{out_file[:-4]}_score_clip.mp4"
    out = None
    
    try:
        # print(f"Creating output video at: {temp_output_path}")
        # print(f"Video dimensions: {frame_width}x{frame_height}")
        
        # 优化2: 创建VideoWriter时使用更高效的参数
        out = cv2.VideoWriter(temp_output_path, fourcc, 30, (frame_width, frame_height), isColor=True)
        
        if not out.isOpened():
            raise Exception(f"Could not open VideoWriter for {temp_output_path}")
            
        # print("VideoWriter opened successfully")
        # print("active frame is ", active_frame)
        # print(f"Total frames to process: {len(frame_list)}")

        # 优化3: 预计算文本位置和样式，避免重复计算
        score_text_p1 = f"Player 1: {scores['p1']}"
        score_text_p2 = f"Player 2: {scores['p2']}"
        text_size, _ = cv2.getTextSize(score_text_p1, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_width = text_size[0]
        text_height = text_size[1]
        padding = 10
        top_right_p1 = (frame_width - text_width - padding, text_height + padding)
        top_right_p2 = (frame_width - text_width - padding, 2 * text_height + 2 * padding)
        
        # 优化4: 创建帧副本，避免修改原始帧
        frame_copy = None
        
        # 优化7: 初始化prev_scores变量
        prev_scores = {'p1': -1, 'p2': -1}
        
        # 优化8: 添加进度显示
        total_frames = len(frame_list)
        progress_interval = max(1, total_frames // 20)  # 每5%显示一次进度
        
        # 优化9: 预分配内存，减少动态分配
        frame_copy = np.empty((frame_height, frame_width, 3), dtype=np.uint8)
        
        # 优化10: 缓存字体参数，避免重复调用
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        font_color = (0, 0, 255)
        font_line_type = cv2.LINE_AA

        # Process each frame
        for i, frame in enumerate(frame_list):
            # 显示进度
            if i % progress_interval == 0:
                progress = (i / total_frames) * 100
                print(f"Processing video: {progress:.1f}% ({i}/{total_frames})")
            
            if(i in active_frame):
                scores = clip_start(frame_in_csv[active_frame.index(i)], i, save_file, scores, pointers_to_players, first_serve)
                if(set_scores["p1"]==1 and set_scores["p2"]==1):
                    if(scores["p1"]==11 or scores["p2"]==11):
                        pointers_to_players["closer"], pointers_to_players["farther"] = pointers_to_players["farther"], pointers_to_players["closer"]

                if(scores["p1"]==21 or scores["p2"]==21):
                    if(scores["p1"]==21):
                        set_scores["p1"]+=1
                    else:
                        set_scores["p2"]+=1
                    scores = {"p1" : 0, "p2" : 0}
                    first_serve = True
                    pointers_to_players["closer"], pointers_to_players["farther"] = pointers_to_players["farther"], pointers_to_players["closer"]
                first_serve = False 

            # Ensure frame is valid
            if frame is None or frame.size == 0:
                continue

            # 优化5: 只在分数变化时重新计算文本
            if i == 0 or scores['p1'] != prev_scores.get('p1', -1) or scores['p2'] != prev_scores.get('p2', -1):
                score_text_p1 = f"Player 1: {scores['p1']}"
                score_text_p2 = f"Player 2: {scores['p2']}"
                text_size, _ = cv2.getTextSize(score_text_p1, font, font_scale, font_thickness)
                text_width = text_size[0]
                text_height = text_size[1]
                top_right_p1 = (frame_width - text_width - padding, text_height + padding)
                top_right_p2 = (frame_width - text_width - padding, 2 * text_height + 2 * padding)
                prev_scores = {'p1': scores['p1'], 'p2': scores['p2']}

            # 优化6: 使用预分配的帧副本，避免重复分配内存
            np.copyto(frame_copy, frame)

            # Draw Player 1 score in red color
            cv2.putText(frame_copy, score_text_p1, top_right_p1, font, font_scale, font_color, font_thickness, font_line_type)

            # Draw Player 2 score in red color below Player 1 score
            cv2.putText(frame_copy, score_text_p2, top_right_p2, font, font_scale, font_color, font_thickness, font_line_type)

            # Write the modified frame to the output video file
            out.write(frame_copy)
            
        print(f"Finished processing all {total_frames} frames")
        
    except Exception as e:
        print(f"Error during video processing: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure VideoWriter is properly released
        if out is not None:
            # print("Releasing VideoWriter...")
            out.release()
            # print(f"VideoWriter released. Output file: {temp_output_path}")

    # Record overall end time and calculate duration
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    
    print(f"Video processing completed in {overall_duration:.2f} seconds")
    print("Done")

# def return_frame_list():
#     frame_list, pred_dict, out_file = create_frames()
#     return frame_list


testing()