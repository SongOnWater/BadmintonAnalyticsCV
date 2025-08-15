"""
Memory management utilities
"""
import gc
import torch
import psutil
import os

def clear_memory():
    """Clear Python and CUDA memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_memory_usage():
    """Get current memory usage statistics"""
    stats = {}
    
    # System RAM
    ram = psutil.virtual_memory()
    stats['ram_used_gb'] = ram.used / 1024**3
    stats['ram_total_gb'] = ram.total / 1024**3
    stats['ram_percent'] = ram.percent
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_stats()
        stats['gpu_allocated_gb'] = gpu_memory.get('allocated_bytes.all.current', 0) / 1024**3
        stats['gpu_reserved_gb'] = gpu_memory.get('reserved_bytes.all.current', 0) / 1024**3
        stats['gpu_max_allocated_gb'] = gpu_memory.get('allocated_bytes.all.peak', 0) / 1024**3
    
    return stats

def print_memory_usage(prefix=""):
    """Print current memory usage"""
    stats = get_memory_usage()
    print(f"{prefix}Memory Usage:")
    print(f"  RAM: {stats['ram_used_gb']:.1f}/{stats['ram_total_gb']:.1f} GB ({stats['ram_percent']:.1f}%)")
    
    if 'gpu_allocated_gb' in stats:
        print(f"  GPU Allocated: {stats['gpu_allocated_gb']:.1f} GB")
        print(f"  GPU Reserved: {stats['gpu_reserved_gb']:.1f} GB")
        print(f"  GPU Peak: {stats['gpu_max_allocated_gb']:.1f} GB")

class MemoryMonitor:
    """Context manager for monitoring memory usage"""
    
    def __init__(self, name="Operation"):
        self.name = name
        self.start_stats = None
    
    def __enter__(self):
        clear_memory()
        self.start_stats = get_memory_usage()
        print(f"Starting {self.name}")
        print_memory_usage("  Before: ")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        clear_memory()
        end_stats = get_memory_usage()
        print(f"Finished {self.name}")
        print_memory_usage("  After: ")
        
        # Calculate differences
        if 'gpu_allocated_gb' in end_stats and 'gpu_allocated_gb' in self.start_stats:
            gpu_diff = end_stats['gpu_allocated_gb'] - self.start_stats['gpu_allocated_gb']
            print(f"  GPU Memory Change: {gpu_diff:+.1f} GB")