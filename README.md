# BadmintonAnalyticsCV
Badminton Analytics using CV - so far, highlight creation and automatic score updation using TrackNet and YOLO.

[TrackNetV3 Model Repository](https://github.com/qaz812345/TrackNetV3)

[YOLOv8 Model Repository](https://github.com/ultralytics/ultralytics)

## 🚀 RTX 4060 Ti Optimized Version

**NEW**: For RTX 4060 Ti users, use the optimized version for **1.66x performance improvement**:

```bash
python run_rtx4060ti_final_optimized.py --video input/your_video.mp4
```

**Performance**: Processes 2-minute videos in 5.0 minutes (vs 8.3 minutes original), saving 3.3 minutes per video.

## Environment setup:

```
pip install -r requirements.txt
```

## Steps to run:

  ### Option 1: RTX 4060 Ti Optimized (Recommended)
  ```bash
  python run_rtx4060ti_final_optimized.py --video input/your_video.mp4
  ```

  ### Option 2: Original Method
  #### Shuttle Tracking Inference using TrackNetV3:
  1. Execute the following command line statement
     
     ```python3 pre_predict.py --video_file original_short.mp4 (or any other raw footage video)```
  (You can choose to add a --save_dir <dir> argument if you want the predictions to be stored elsewhere and not the default prediction directory)

  #### Using TrackNetV3 predictions for highlights generation and score updation:
  1. Execute the following command line statement
     
     ```python3 testing.py```

After running the above steps, filename_score_clip.mp4 will be created at cwd.


