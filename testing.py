import os
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

    frame_list, pred_dict, out_file = create_frames()
    print("legth of fl , pd , of is ",len(frame_list),len(pred_dict),len(out_file))
    print(pred_dict)
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
    print("save file, " ,save_file)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    temp_output_path = f"{out_file[:-4]}_score_clip.mp4"
    out = None
    
    try:
        print(f"Creating output video at: {temp_output_path}")
        print(f"Video dimensions: {frame_width}x{frame_height}")
        
        # Create VideoWriter with proper parameters
        out = cv2.VideoWriter(temp_output_path, fourcc, 30, (frame_width, frame_height))
        
        if not out.isOpened():
            raise Exception(f"Could not open VideoWriter for {temp_output_path}")
            
        print("VideoWriter opened successfully")
        print("active frame is ", active_frame)
        print(f"Total frames to process: {len(frame_list)}")

        # Process each frame
        for i, frame in enumerate(frame_list):
            if i % 100 == 0:
                print(f"Processing frame {i}/{len(frame_list)}")
                
            if(i in active_frame):
                scores = clip_start(frame_in_csv[active_frame.index(i)], i, save_file, scores, pointers_to_players, first_serve)
                print("scores", scores)
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
                print(f"Warning: Invalid frame at index {i}, skipping")
                continue

            score_text_p1 = f"Player 1: {scores['p1']}"
            score_text_p2 = f"Player 2: {scores['p2']}"
            text_size, _ = cv2.getTextSize(score_text_p1, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_width = text_size[0]
            text_height = text_size[1]

            # Position to place the text (top right corner with padding)
            padding = 10
            top_right_p1 = (frame.shape[1] - text_width - padding, text_height + padding)
            top_right_p2 = (frame.shape[1] - text_width - padding, 2 * text_height + 2 * padding)

            # Draw Player 1 score in red color
            cv2.putText(frame, score_text_p1, top_right_p1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Draw Player 2 score in red color below Player 1 score
            cv2.putText(frame, score_text_p2, top_right_p2, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

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


    print("Done")

# def return_frame_list():
#     frame_list, pred_dict, out_file = create_frames()
#     return frame_list


testing()