import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import math
from PIL import Image
from ultralytics import YOLO



model = YOLO("/workspace/BadmintonAnalyticsCV/ckpts/yolov8n.pt")

# Function to perform inference and get normalized bounding box coordinates in xywh format for each frame
def get_bounding_boxes(frame, img_height, img_width):
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(frame, Image.Image) or (hasattr(frame, 'mode') and hasattr(frame, 'getdata')):
            # This is a PIL Image
            frame_np = np.array(frame)
            if hasattr(frame, 'mode') and frame.mode == 'RGB':
                frame_np = cv.cvtColor(frame_np, cv.COLOR_RGB2BGR)
            frame = frame_np
        elif hasattr(frame, 'size'):  # Another way to check for PIL Image
            frame_np = np.array(frame)
            frame = frame_np
            
        # Ensure frame is a numpy array
        if not isinstance(frame, np.ndarray):
            print(f"Warning: frame is not a numpy array but {type(frame)}")
            # Try to convert it anyway
            try:
                frame = np.array(frame)
            except:
                print("Failed to convert frame to numpy array")
                return np.empty((0, 4))  # Return empty array
        
        # Perform inference
        try:
            results = model.predict(source=frame, save=False, verbose=False)
        except Exception as e:
            print(f"Error in YOLO prediction: {str(e)}")
            return np.empty((0, 4))  # Return empty array on prediction error
        
        # Check if results is empty or has no boxes
        if not results or len(results) == 0:
            print("No results from YOLO model")
            return np.empty((0, 4))
            
        # Check if any detections were made
        if len(results[0].boxes) == 0:
            print("No detections found in the frame")
            return np.empty((0, 4))
            
        # Extract bounding box coordinates in x_center, y_center, width, height format
        try:
            bounding_boxes = results[0].boxes.xywh  # [x_center, y_center, width, height]
            # Convert to numpy array
            bounding_boxes = bounding_boxes.cpu().numpy()
        except Exception as e:
            print(f"Error extracting bounding boxes: {str(e)}")
            return np.empty((0, 4))
        
        # Check if bounding_boxes is empty
        if bounding_boxes.size == 0:
            print("Empty bounding boxes array")
            return np.empty((0, 4))

        # Normalize the bounding box coordinates
        bounding_boxes[:, 0] /= img_width   # Normalize x_center
        bounding_boxes[:, 1] /= img_height  # Normalize y_center
        bounding_boxes[:, 2] /= img_width   # Normalize width
        bounding_boxes[:, 3] /= img_height  # Normalize height

        return bounding_boxes
    except Exception as e:
        print(f"Error in get_bounding_boxes: {str(e)}")
        return np.empty((0, 4))  # Return empty array instead of None

#Function to visualize bounding boxes on a frame and print the coordinates
def visualize_boxes_xywh(mid_line_coord, frame, boxes, img_width, img_height):
    #print("From left_right ", boxes)
    
    if boxes is None or len(boxes) == 0:
        print("Warning: No bounding boxes detected")
        return frame, "Left"  # Default to Left if no boxes detected
    
    try:
        x_center_player, _, _, _ = boxes[0]
        if x_center_player < mid_line_coord:
            p1_side = "Left"
        else:
            p1_side = "Right"
        
        #print("Player 1 is on the", p1_side)
        return frame, p1_side
        
    except Exception as e:
        print(f"Error in visualize_boxes_xywh: {str(e)}")
        return frame, "Left"  # Default to Left on error