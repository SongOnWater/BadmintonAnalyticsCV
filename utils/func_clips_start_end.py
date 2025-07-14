from ultralytics import YOLO
import cv2
import numpy as np
import sys
from .left_right import visualize_boxes_xywh, get_bounding_boxes
from .ullasmodel import return_middle_line

import os
import pandas as pd

def get_video_dimensions(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was successfully loaded
    if not cap.isOpened():
        raise ValueError("Video not found or unable to load.")
    
    # Get the width and height of the frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Release the video capture object
    cap.release()
    
    return height, width

def score_updation_no_changes(path_to_mp4, scores):
    cap = cv2.VideoCapture(path_to_mp4)
    
    # Check if video capture is successful
    if not cap.isOpened():
        print(f"Error: Could not open video file at {path_to_mp4}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object for temporary file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    temp_output_path = "temp_output.mp4"
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error reading video frame")
            break
        
        # Update scores and draw on frame
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

        # Write the modified frame to the temporary output video file
        out.write(frame)

    # Release video capture and writer objects
    cap.release()
    out.release()

    # Replace the original video file with the temporary output file
    os.remove(path_to_mp4)  # Delete original file
    os.rename(temp_output_path, path_to_mp4)




def read_csv_for_mp4(mp4_file_path):
    # Check if the provided file path has an .mp4 extension
    if not mp4_file_path.endswith('.mp4'):
        raise ValueError("The provided file does not have an .mp4 extension")
    
    # Replace the .mp4 extension with .csv
    csv_file_path = mp4_file_path.replace('.mp4', '_ball.csv')
    
    # if not os.path.exists(csv_file_path):
    #     raise FileNotFoundError(f"The CSV file {csv_file_path} does not exist")

    return csv_file_path


def clip_end(frame_in_csv):
    print("clip ends here",frame_in_csv) 

def clip_start(frame_in_csv, frame_in_mp4, path_to_mp4, scores, pointers_to_players, first_serve):

    if(frame_in_csv==None):
        score_updation_no_changes(path_to_mp4, scores)
        return scores
    path_to_csv =read_csv_for_mp4(path_to_mp4)
    # print("converted into csv")
    print("clip starts here", frame_in_csv)
    # img_height, img_width = get_video_dimensions(path_to_mp4)

    mid_line_coord = return_middle_line(path_to_mp4)
    # print("hello")
    # print("t0")
    cap = cv2.VideoCapture(path_to_mp4)
    # print("t1 ", frame_in_mp4)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_in_mp4)
    while cap.isOpened():
        # print("hello2")
        ret, frame = cap.read()
        if not ret:
            break

        img_height, img_width = frame.shape[:2]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Convert frame to numpy array if it's PIL Image
            if hasattr(frame_rgb, 'size'):  # Check if it's PIL Image
                frame_np = np.array(frame_rgb)
            else:
                frame_np = frame_rgb
            
            # Ensure frame is in correct format for YOLO
            if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                boxes = get_bounding_boxes(frame_np, img_height=img_height, img_width=img_width)
            else:
                print(f"Invalid frame format at frame {i}, skipping detection")
                return scores
        except Exception as e:
            print(f"Error in object detection at frame {i}: {str(e)}")
            return scores
        try:
            frame, p1_side= visualize_boxes_xywh(mid_line_coord, frame_in_mp4, boxes, img_width, img_height)
        except:
            continue
        if (p1_side!=""):
            break
        # co+=1
    cap.release()

    
    with open(path_to_csv, "r") as csv_file:
        # print("frame in csv", frame_in_csv)
        # if frame_in_csv == 0 :
        #     frame_in_csv = 1
        # print("t2 ")
        for i in range(frame_in_csv+1):
            FrameNo,Visibility,X,Y = tuple(csv_file.readline().split(","))
            # print("here2", i)
            if i == 0:
                continue
            # FrameNo = int(FrameNo)
            # Visibility = int(Visibility)
            # X = int(X)
            # Y = int(Y)
        frame_counter = 0
        direction = ""
        direction_determiner = 0
        flag_direction=0
        y_diff_for_direction = 0

        while True:
            # try:
                # print("f1")
                list1 = tuple(csv_file.readline().strip().split(","))
                list2 = tuple(csv_file.readline().strip().split(","))

                # print("list1 = ",list1 , "\nlist2 = ",list2)

                if list1[0]=='' or list2[0]=='':
                    print("value:", direction_determiner)
                    if direction_determiner > 0:
                        direction = "Right"
                    elif direction_determiner < 0:
                        direction = "Left"
                    else:
                        direction = "stationary"
                    
                    print(f"Final Direction: {direction}")
                    break
                
                FrameNo_1, Visibility_1, X_1, Y_1 = list1[0],list1[1],list1[2],list1[3]
                FrameNo_2, Visibility_2, X_2, Y_2 = list2[0],list2[1],list2[2],list2[3]
                
                FrameNo_1 = int(FrameNo_1)
                Visibility_1 = int(Visibility_1)
                X_1 = int(X_1)
                Y_1 = int(Y_1)
                FrameNo_2 = int(FrameNo_2)
                Visibility_2 = int(Visibility_2)
                X_2 = int(X_2)
                Y_2 = int(Y_2)

                y_diff = Y_2 - Y_1
                x_diff = X_2 - X_1
                print("Frame =",FrameNo_1,"  frame counter =  ", frame_counter)
                if(y_diff!=0):
                    y_diff_for_direction = y_diff
                
                

                if (Visibility_1 == 0 and Visibility_2 == 0) and flag_direction!=0 and y_diff_for_direction<0:
                    direction_determiner += flag_direction
                    frame_counter += 1

                elif Visibility_1 == 1 and Visibility_2 == 1:
                    # y_diff = Y_2 - Y_1
                    # x_diff = X_2 - X_1
                    if y_diff <= 0:
                        if x_diff > 0:
                            direction_determiner += 1
                            flag_direction=1
                        elif x_diff < 0:
                            direction_determiner -= 1
                            flag_direction=-1
                        elif x_diff==0 and flag_direction!=0:
                            direction_determiner+=flag_direction                        
                    # print("f3")
                    frame_counter += 1
                
                    # else:
                    #     # print(frame_counter, "frame no")
                    #     frame_counter += 1
                
                if frame_counter == 24:
                    print("value:", direction_determiner)
                    if direction_determiner > 0:
                        direction = "Right"
                    elif direction_determiner < 0:
                        direction = "Left"
                    else:
                        direction = "stationary"
                    
                    print(f"Final Direction: {direction}")
                    break

                print("DD", direction_determiner, end= " ")

            # except ValueError:
            # # If a line read is incomplete or incorrect format, we break the loop
            #     print("Error in reading the frames or reached the end of file.")
                

    
    if(first_serve!=True):
        if direction == "Right" and p1_side == "Right":
            # assert p2_side == "Left"
            scores[pointers_to_players["farther"]]+=1
            print("closer Receive")
        elif direction == "Left" and p1_side == "Right":
            # assert p2_side == "Left"
            scores[pointers_to_players["closer"]]+=1
            print("closer Serve")
        elif direction == "Right" and p1_side == "Left":
            # assert p2_side == "Right"
            scores[pointers_to_players["closer"]]+=1
            print("closer Serve")
        elif direction == "Left" and p1_side == "Left":
            # assert p2_side == "Right"
            scores[pointers_to_players["farther"]]+=1
            print("closer Receive")
        else:
            print("Error in determining the serve")
    else:
        if direction == "Right" and p1_side == "Right":
            # assert p2_side == "Left"
            # scores[pointers_to_players["farther"]]+=1
            print("closer Receive")
        elif direction == "Left" and p1_side == "Right":
            # assert p2_side == "Left"
            # scores[pointers_to_players["closer"]]+=1
            print("closer Serve")
        elif direction == "Right" and p1_side == "Left":
            # assert p2_side == "Right"
            # scores[pointers_to_players["closer"]]+=1
            print("closer Serve")
        elif direction == "Left" and p1_side == "Left":
            # assert p2_side == "Right"
            # scores[pointers_to_players["farther"]]+=1
            print("closer Receive")
        else:
            print("Error in determining the serve")

    return scores
    

    

                