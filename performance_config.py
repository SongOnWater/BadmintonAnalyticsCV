"""
Performance configuration and optimization utilities
"""
import torch
import os
import psutil
from typing import Dict, Any

class PerformanceConfig:
    """Centralized performance configuration"""
    
    def __init__(self):
        self.device = self._get_optimal_device()
        self.batch_size = self._get_optimal_batch_size()
        self.num_workers = self._get_optimal_workers()
        self.memory_fraction = 0.95  # Use 95% of available GPU memory - AGGRESSIVE
        
    def _get_optimal_device(self) -> torch.device:
        """Get the best available device"""
        if torch.cuda.is_available():
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            print(f"GPU Memory: {gpu_memory / 1024**3:.1f} GB")
            return torch.device('cuda')
        return torch.device('cpu')
    
    def _get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size - MAXIMUM GPU UTILIZATION"""
        if self.device.type == 'cuda':
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            # 基于实际测试：1.6GB峰值 -> 可以用更大批处理
            if gpu_memory_gb >= 16:
                return 64  # 极大批处理，目标使用8-12GB GPU内存
            elif gpu_memory_gb >= 12:
                return 48
            elif gpu_memory_gb >= 8:
                return 32
            else:
                return 16
        else:
            ram_gb = psutil.virtual_memory().total / 1024**3
            return min(16, max(4, int(ram_gb // 2)))
    
    def _get_optimal_workers(self) -> int:
        """Calculate optimal number of workers - single threaded for stability"""
        return 0  # 单线程处理，避免多进程复杂性
    
    def setup_torch_optimizations(self):
        """Setup PyTorch optimizations - AGGRESSIVE"""
        if self.device.type == 'cuda':
            # Enable cuDNN benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # 牺牲确定性换取速度
            
            # Enable TensorFloat-32 for faster training on Ampere GPUs
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                # 启用更多CUDA优化
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            except AttributeError:
                pass
            
            # 设置CUDA流和内存池
            try:
                torch.cuda.set_sync_debug_mode(0)  # 禁用同步调试
                # 预分配GPU内存池
                torch.cuda.empty_cache()
                torch.cuda.memory._set_allocator_settings("expandable_segments:True")
            except:
                pass
            
            # Set memory fraction
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
        
        # 设置合理的CPU线程数
        cpu_threads = min(8, os.cpu_count() or 4)  # 减少线程数避免冲突
        torch.set_num_threads(cpu_threads)
        
        # 设置环境变量
        os.environ['OMP_NUM_THREADS'] = str(cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(cpu_threads)
        os.environ['NUMEXPR_MAX_THREADS'] = str(cpu_threads)
    
    def get_dataloader_config(self) -> Dict[str, Any]:
        """Get optimized DataLoader configuration"""
        config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.device.type == 'cuda',
            'persistent_workers': self.num_workers > 0,
            'drop_last': False,
            'shuffle': False
        }
        
        # 单线程模式，无需预取设置
            
        return config
    
    def print_config(self):
        """Print current configuration"""
        print(f"Performance Configuration:")
        print(f"  Device: {self.device}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Num Workers: {self.num_workers}")
        print(f"  Memory Fraction: {self.memory_fraction}")

# Global instance
perf_config = PerformanceConfig()