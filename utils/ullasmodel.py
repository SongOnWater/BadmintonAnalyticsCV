import cv2
from ultralytics import YOLO
import logging

from PIL import Image, ImageDraw

def return_middle_line(video_path):
    #logging.basicConfig(level=logging.WARNING)
    # Initialize YOLO model
    # Using standard YOLOv8 nano model as replacement
    model = YOLO("/workspace/BadmintonAnalyticsCV/ckpts/yolov8n.pt")
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print("Error opening video file")
        return None
    # print("mid1")
    # Read the first frame
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("Error reading video frame")
            return None
        # print("mid1.5 ", ret)
                
        # Convert frame from BGR to RGB and to PIL Image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        # Predict bounding boxes using YOLO
        try:
            results = model.predict(source=pil_image, verbose=False)
        except Exception as e:
            print(f"Error in YOLO prediction: {str(e)}")
            return None
        # print("mid2")
        # Calculate middle line x-coordinate
        middle_line_x = 0.0
        num_boxes = 0
        
        for pred in results:
            for box in pred.boxes.xywh:
                x_center, _, _, _ = box
                middle_line_x += x_center
                num_boxes += 1
            # print("mid3")
        
        if num_boxes > 0:
            middle_line_x /= num_boxes
            middle_line_ratio = middle_line_x / image.shape[1]  # Normalize to image width
        else:
            middle_line_ratio = None
        # print("mid4")
        # Release video capture object and close any open windows    
        if middle_line_ratio != None:
            return float(middle_line_ratio)
        
    video.release()
    cv2.destroyAllWindows()
    # print("mid5")
    return None 

# import cv2
# from ultralytics import YOLO
# import logging
# import torch

# from PIL import Image, ImageDraw

# def return_middle_line(video_path):
#     #logging.basicConfig(level=logging.WARNING)

#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # print(f"Using device: {device}")  # Inform user about device usage

#     # Initialize YOLO model
#     model = YOLO("/mnt/e/Classes/Combined-files/ullasmodelb/best.pt")
#     # model.to(device)

#     # Open the video file
#     video = cv2.VideoCapture(video_path)
    
#     # Check if video opened successfully
#     if not video.isOpened():
#         print("Error opening video file")
#         return None
    
#     while video.isOpened():
#         # Read the first frame
#         ret, frame = video.read()
#         if not ret:
#             print("Error reading video frame")
#             return None
        
#         # Convert frame from BGR to RGB (OpenCV format to PIL format)
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image = torch.from_numpy(image).to(device)
        
#         # Predict bounding boxes using YOLO
#         results = model.predict(image)
        
#         # Calculate middle line x-coordinate
#         middle_line_x = 0.0
#         num_boxes = 0
        
#         for pred in results:
#             for box in pred.boxes.xywh:
#                 x_center, _, _, _ = box
#                 middle_line_x += x_center
#                 num_boxes += 1
        
#         if num_boxes > 0:
#             middle_line_x /= num_boxes
#             middle_line_ratio = middle_line_x / image.shape[1]  # Normalize to image width
#         else:
#             middle_line_ratio = None
        
#         # Release video capture object and close any open windows
        
#         if(middle_line_ratio!=None):
#             return float(middle_line_ratio)
#     video.release()
#     cv2.destroyAllWindows()
#     return None