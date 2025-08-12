import time
import psutil
import torch
import gc
from functools import wraps
from typing import Dict, Any
import logging

class PerformanceMonitor:
    """Performance monitoring and optimization utilities"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def start_timer(self, name: str):
        """Start timing an operation"""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing an operation and return duration"""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            self.metrics[name] = duration
            del self.start_times[name]
            return duration
        return 0.0
    
    def log_memory_usage(self, stage: str = ""):
        """Log current memory usage"""
        # System memory
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        log_msg = f"[{stage}] CPU: {cpu_percent}%, RAM: {memory.percent}% ({memory.used / 1024**3:.1f}GB/{memory.total / 1024**3:.1f}GB)"
        
        # GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_cached = torch.cuda.memory_reserved() / 1024**3
            log_msg += f", GPU: {gpu_memory:.1f}GB allocated, {gpu_cached:.1f}GB cached"
        
        self.logger.info(log_msg)
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def optimize_memory(self, force_gc: bool = False):
        """Optimize memory usage"""
        if force_gc:
            gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024**3,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'cuda_version': torch.version.cuda,
            })
        
        return info
    
    def print_summary(self):
        """Print performance summary"""
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        
        # System info
        info = self.get_system_info()
        print(f"CPU Cores: {info['cpu_count']}")
        print(f"RAM: {info['memory_total_gb']:.1f}GB")
        if info['cuda_available']:
            print(f"GPU: {info['gpu_name']} ({info['gpu_memory_gb']:.1f}GB)")
        
        # Timing metrics
        if self.metrics:
            print("\nTiming Metrics:")
            for name, duration in self.metrics.items():
                print(f"  {name}: {duration:.2f}s")
        
        print("="*50)

def performance_timer(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def memory_efficient_processing(clear_cache_every: int = 100):
    """Decorator for memory efficient processing"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Clear cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            result = func(*args, **kwargs)
            
            # Clear cache after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return result
        return wrapper
    return decorator

# Global performance monitor instance
monitor = PerformanceMonitor()