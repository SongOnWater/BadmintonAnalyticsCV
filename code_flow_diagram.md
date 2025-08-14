# 羽毛球分析系统代码调用流程图

```mermaid
graph TD
    %% 主入口点
    A[用户运行 run_pipeline.py] --> B[解析命令行参数]
    B --> C[创建输出目录]
    
    %% 第一阶段：预预测
    C --> D[Step 1: 运行 pre_predict.py]
    D --> E[读取输入视频文件]
    E --> F[检查视频FPS，可选转换为30fps]
    F --> G[计算视频总时长，分段处理]
    G --> H[循环处理每个视频段]
    
    %% 视频段处理
    H --> I[读取视频段帧]
    I --> J[调用 predict.py 进行推理]
    J --> K[加载预训练模型]
    
    %% 模型加载
    K --> L[加载 TrackNet 模型]
    L --> M[可选加载 InpaintNet 模型]
    M --> N[模型推理处理]
    
    %% 推理过程
    N --> O[预处理帧数据]
    O --> P[TrackNet 预测羽毛球位置]
    P --> Q{是否使用 InpaintNet?}
    Q -->|是| R[InpaintNet 修复预测坐标]
    Q -->|否| S[直接使用 TrackNet 结果]
    R --> S
    
    %% 结果处理
    S --> T[后处理预测结果]
    T --> U[合并所有段的结果]
    U --> V[保存预测结果到CSV]
    V --> W[保存中间结果到二进制文件]
    
    %% 第二阶段：测试和评分
    W --> X[Step 2: 运行 testing.py]
    X --> Y[读取预测结果和视频]
    Y --> Z[调用 pred_dict_modify 优化预测]
    Z --> AA[生成视频片段]
    AA --> BB[调用 write_pred_video_modified]
    
    %% 评分系统
    BB --> CC[分析比赛片段]
    CC --> DD[调用 clip_start 计算得分]
    DD --> EE[更新玩家分数]
    EE --> FF[处理比赛规则逻辑]
    
    %% 视频生成
    FF --> GG[生成带分数叠加的视频]
    GG --> HH[创建最终输出文件]
    
    %% 输出文件
    HH --> II[CSV文件：羽毛球轨迹数据]
    HH --> JJ[MP4文件：带轨迹的视频]
    HH --> KK[MP4文件：带分数的视频片段]
    
    %% 样式定义
    classDef entryPoint fill:#e1f5fe
    classDef process fill:#f3e5f5
    classDef model fill:#e8f5e8
    classDef output fill:#fff3e0
    classDef decision fill:#ffebee
    
    class A entryPoint
    class D,X process
    class L,M model
    class II,JJ,KK output
    class Q decision
```

## 主要模块说明

### 1. 主控制流程 (run_pipeline.py)
- 协调整个分析流程
- 按顺序执行预预测和测试两个阶段
- 处理错误和计时

### 2. 预预测阶段 (pre_predict.py)
- 视频分段处理，避免内存溢出
- 调用核心推理模块
- 合并多个段的预测结果

### 3. 核心推理 (predict.py)
- 加载和缓存预训练模型
- 实现滑动窗口推理
- 支持 TrackNet + InpaintNet 组合

### 4. 模型架构 (model.py)
- **TrackNet**: U-Net结构的羽毛球检测网络
- **InpaintNet**: 1D卷积网络，修复预测坐标

### 5. 测试和评分 (testing.py)
- 优化预测结果，过滤噪声
- 实现羽毛球比赛规则逻辑
- 生成带分数叠加的输出视频

### 6. 工具函数 (utils/)
- **general.py**: 通用图像处理和数据转换
- **func_clips_start_end.py**: 比赛片段分析和得分计算
- **left_right.py**: 球员位置识别
- **ullasmodel.py**: 球场中线检测

## 数据流向

1. **输入**: MP4视频文件
2. **中间处理**: 帧序列 → 预测坐标 → 优化结果
3. **输出**: 
   - CSV轨迹数据
   - 带轨迹的视频
   - 带分数的视频片段

## 关键技术特点

- **分段处理**: 支持长视频，避免内存问题
- **模型缓存**: 避免重复加载模型
- **滑动窗口**: 时序预测，提高准确性
- **规则引擎**: 自动计算羽毛球比赛得分
- **多模型融合**: TrackNet + InpaintNet 提升性能
