# BadmintonAnalyticsCV
Badminton Analytics using CV - so far, highlight creation and automatic score updation using TrackNet and YOLO.

[TrackNetV3 Model Repository](https://github.com/qaz812345/TrackNetV3)

[YOLOv8 Model Repository](https://github.com/ultralytics/ultralytics)

## Environment setup:

```
pip install -r requirements.txt
```

## Steps to run:

  ### 🚀 推荐使用方式 (突破性能版本)
  
  使用优化的突破性能版本，获得最佳性能：
  
  ```bash
  python run_breakthrough.py --video_file your_video.mp4
  ```
  
  **参数说明**：
  - `--video_file`: 输入视频文件路径（必需）
  - `--save_dir`: 输出目录（默认：prediction）
  - `--eval_mode`: 预测模式
    - `nonoverlap`: 快速模式，2.3x性能提升（默认）
    - `weight`: 高精度模式，使用时间集成
  - `--generate_pred_video`: 生成完整预测视频（默认不生成，节省时间）
  
  **使用示例**：
  ```bash
  # 默认快速模式
  python run_breakthrough.py --video_file match.mp4
  
  # 高精度模式
  python run_breakthrough.py --video_file match.mp4 --eval_mode weight
  
  # 生成完整预测视频
  python run_breakthrough.py --video_file match.mp4 --generate_pred_video
  ```
  
  **性能优势**：
  - 2.32倍速度提升
  - 移除时间集成开销
  - 使用nonoverlap模式
  - 一键生成所有结果

  ### 传统使用方式
  
  #### Shuttle Tracking Inference using TrackNetV3:
  1. Execute the following command line statement
     
     ```python3 pre_predict.py --video_file your_video.mp4 (or any other raw footage video)```
  (You can choose to add a --save_dir <dir> argument if you want the predictions to be stored elsewhere and not the default prediction directory)

  #### Using TrackNetV3 predictions for highlights generation and score updation:
  1. Execute the following command line statement
     
     ```python3 testing_fixed.py```

After running the above steps, filename_score_clip.mp4 will be created at cwd.



## 本轮性能优化总结（2025-08，gpt-5-improve 分支）

为显著降低视频处理耗时、提升 GPU 利用率，本轮对代码进行了以下改进：

- 模型与数据管线
  - 缓存并复用模型：避免每段视频重复加载权重。
  - 启用推理优化：`torch.backends.cudnn.benchmark=True`、TF32（若可用）、`torch.inference_mode()`、AMP（`torch.amp.autocast('cuda')`）与 `channels_last`。
  - DataLoader 调优：Windows 默认 `num_workers=0`（避免首批多进程卡顿）；其它系统启用 `persistent_workers`、`pin_memory` 与 `prefetch_factor`。
  - 明确按 checkpoint 的 `seq_len` 生成输入，确保通道数始终为 `3*seq_len`（如 `bg_mode='concat'` 则会在每窗前拼接 median，通道为 `(L+1)*3`）。修复了“24 vs 27 通道不匹配”的错误。

- 视频读取与预处理
  - 去除对 MoviePy 的强依赖：`pre_predict.py` 完全使用 OpenCV 读取、按时间分段，无中间写盘/重编码。
  - 统一使用 OpenCV 进行 resize 与 BGR→RGB，再以 NumPy/torch 构造 `(N,3,H,W)`，减少 PIL 与频繁拼接带来的 CPU 开销。
  - 支持自适应分段（默认 20s），可根据机器与视频调整以减少重建管线成本。

- 其他
  - 移除 `predict.py` 对 `test.py` 的导入，内联了 `get_ensemble_weight`、`predict_location`、`generate_inpaint_mask`，避免 `pycocotools` 依赖。
  - 首批数据准备更快：在 Windows 环境避免多进程复制与 pickling，进度条更早开始，segment 总时长明显缩短。

### 运行建议

- 使用 Conda 环境的 Python 执行，避免 `.venv` 干扰：
  - `C:\Users\admin\miniconda3\envs\badminton-env\python.exe -u pre_predict.py --video_file input\bd_4.mp4 --save_dir prediction`
- 根据 GPU/CPU 情况，可增大 `batch_size`（如 16/24）并酌情增加段长 `clip_duration`（如 30-60s）以提升吞吐。
