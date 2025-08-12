# Performance Configuration for Badminton Analytics CV
import torch
import os

class Config:
    """Centralized configuration for performance optimization"""
    
    # Device Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_HALF_PRECISION = torch.cuda.is_available()
    
    # Model Configuration
    TRACKNET_MODEL_PATH = "ckpts/TrackNet_best.pt"
    INPAINTNET_MODEL_PATH = "ckpts/InpaintNet_best.pt"
    YOLO_MODEL_PATH = "ckpts/yolov8n.pt"
    
    # Processing Configuration
    if torch.cuda.is_available():
        BATCH_SIZE = 8
        NUM_WORKERS = 16
        CLIP_DURATION = 20  # seconds
    else:
        BATCH_SIZE = 2
        NUM_WORKERS = 8
        CLIP_DURATION = 15  # seconds
    
    # Video Processing
    TARGET_FPS = 30
    VIDEO_CODEC = 'libx264'
    
    # Memory Management
    ENABLE_MEMORY_OPTIMIZATION = True
    CLEAR_CACHE_FREQUENCY = 100  # frames
    
    # Paths
    TEMP_DIR = "temp"
    OUTPUT_DIR = "prediction"
    
    # Performance Monitoring
    ENABLE_PROFILING = False
    LOG_PERFORMANCE = True
    
    @classmethod
    def setup_torch_optimizations(cls):
        """Setup PyTorch optimizations"""
        if cls.DEVICE.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable memory efficient attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except:
                pass
    
    @classmethod
    def get_optimal_batch_size(cls, video_length_seconds):
        """Get optimal batch size based on video length"""
        if video_length_seconds > 300:  # 5 minutes
            return max(1, cls.BATCH_SIZE // 2)
        return cls.BATCH_SIZE
    
    @classmethod
    def create_temp_dir(cls):
        """Create temporary directory if it doesn't exist"""
        if not os.path.exists(cls.TEMP_DIR):
            os.makedirs(cls.TEMP_DIR)
    
    @classmethod
    def cleanup_temp_files(cls):
        """Clean up temporary files"""
        import glob
        temp_files = glob.glob(os.path.join(cls.TEMP_DIR, "*"))
        for file in temp_files:
            try:
                os.remove(file)
            except:
                pass