#!/usr/bin/env python3
"""
自适应GPU优化器
基于GPU的SM数量、内存带宽、架构特性动态选择最优参数
"""

import torch
import time
import gc
import numpy as np
from typing import Dict, Tuple, Optional
import json
import os

class AdaptiveGPUOptimizer:
    """自适应GPU优化器 - 动态选择最优参数"""
    
    def __init__(self, model_memory_per_sample: float = 0.3):
        """
        初始化优化器
        
        Args:
            model_memory_per_sample: 每个样本的内存需求(GB)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_memory_per_sample = model_memory_per_sample
        
        if self.device.type == 'cuda':
            self.gpu_props = torch.cuda.get_device_properties(0)
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory_gb = self.gpu_props.total_memory / 1024**3
            self.sm_count = self.gpu_props.multi_processor_count
            self.compute_capability = torch.cuda.get_device_capability(0)
        else:
            self.gpu_props = None
            self.gpu_name = "CPU"
            self.gpu_memory_gb = 8.0
            self.sm_count = 4  # 假设4核CPU
            self.compute_capability = (0, 0)
        
        # 缓存优化结果
        self.optimization_cache = {}
        self.cache_file = "gpu_optimization_cache.json"
        self._load_cache()
    
    def _load_cache(self):
        """加载优化缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.optimization_cache = json.load(f)
            except:
                self.optimization_cache = {}
    
    def _save_cache(self):
        """保存优化缓存"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.optimization_cache, f, indent=2)
        except:
            pass
    
    def get_gpu_characteristics(self) -> Dict:
        """获取详细的GPU特性"""
        if self.device.type != 'cuda':
            return {
                "name": self.gpu_name,
                "type": "CPU",
                "memory_gb": self.gpu_memory_gb,
                "sm_count": self.sm_count,
                "memory_bandwidth_gbps": 50,  # 假设DDR4
                "optimal_load_factor": 0.8,
                "architecture": "CPU"
            }
        
        # GPU架构检测
        major, minor = self.compute_capability
        if major >= 9:
            architecture = "Hopper"
            optimal_load_factor = 0.8
        elif major >= 8:
            architecture = "Ada Lovelace" if "RTX 40" in self.gpu_name else "Ampere"
            optimal_load_factor = 0.7
        elif major >= 7:
            architecture = "Turing"
            optimal_load_factor = 0.6
        elif major >= 6:
            architecture = "Pascal"
            optimal_load_factor = 0.5
        else:
            architecture = "Legacy"
            optimal_load_factor = 0.4
        
        # 内存带宽估算 (基于GPU型号)
        memory_bandwidth = self._estimate_memory_bandwidth()
        
        return {
            "name": self.gpu_name,
            "type": "CUDA",
            "memory_gb": self.gpu_memory_gb,
            "sm_count": self.sm_count,
            "memory_bandwidth_gbps": memory_bandwidth,
            "optimal_load_factor": optimal_load_factor,
            "architecture": architecture,
            "compute_capability": self.compute_capability,
            "max_threads_per_sm": getattr(self.gpu_props, 'max_threads_per_multiprocessor', 2048)
        }
    
    def _estimate_memory_bandwidth(self) -> float:
        """估算GPU内存带宽 (GB/s)"""
        # 基于GPU型号的精确带宽数据
        bandwidth_database = {
            # RTX 40系列 (Ada Lovelace)
            "RTX 4090": 1008,
            "RTX 4080 SUPER": 736,
            "RTX 4080": 717,
            "RTX 4070 Ti SUPER": 672,
            "RTX 4070 Ti": 504,
            "RTX 4070 SUPER": 504,
            "RTX 4070": 504,
            "RTX 4060 Ti": 288,
            "RTX 4060": 272,
            
            # RTX 30系列 (Ampere)
            "RTX 3090 Ti": 1008,
            "RTX 3090": 936,
            "RTX 3080 Ti": 912,
            "RTX 3080": 760,
            "RTX 3070 Ti": 608,
            "RTX 3070": 448,
            "RTX 3060 Ti": 448,
            "RTX 3060": 360,
            
            # RTX 20系列 (Turing)
            "RTX 2080 Ti": 616,
            "RTX 2080 SUPER": 496,
            "RTX 2080": 448,
            "RTX 2070 SUPER": 448,
            "RTX 2070": 448,
            "RTX 2060 SUPER": 448,
            "RTX 2060": 336,
            
            # GTX系列
            "GTX 1080 Ti": 484,
            "GTX 1080": 320,
            "GTX 1070": 256,
            "GTX 1060": 192,
        }
        
        # 精确匹配
        for gpu_model, bandwidth in bandwidth_database.items():
            if gpu_model in self.gpu_name:
                return bandwidth
        
        # 模糊匹配
        if "4090" in self.gpu_name:
            return 1008
        elif "4080" in self.gpu_name:
            return 717
        elif "4070" in self.gpu_name:
            return 504
        elif "4060 Ti" in self.gpu_name:
            return 288
        elif "4060" in self.gpu_name:
            return 272
        elif "3090" in self.gpu_name:
            return 936
        elif "3080" in self.gpu_name:
            return 760
        elif "3070" in self.gpu_name:
            return 448
        elif "3060" in self.gpu_name:
            return 360
        elif "2080" in self.gpu_name:
            return 448
        elif "2070" in self.gpu_name:
            return 448
        elif "2060" in self.gpu_name:
            return 336
        elif "1080" in self.gpu_name:
            return 320
        elif "1070" in self.gpu_name:
            return 256
        elif "1060" in self.gpu_name:
            return 192
        
        # 基于计算能力的估算
        major, minor = self.compute_capability
        if major >= 8:
            return 400 + (self.gpu_memory_gb - 8) * 40
        elif major >= 7:
            return 300 + (self.gpu_memory_gb - 6) * 30
        elif major >= 6:
            return 200 + (self.gpu_memory_gb - 4) * 25
        else:
            return 150
    
    def calculate_optimal_batch_size(self, gpu_chars: Dict) -> Tuple[int, Dict]:
        """基于GPU特性计算最优批处理大小"""
        
        # 1. 基于内存限制的最大批处理大小
        usable_memory = gpu_chars["memory_gb"] * 0.8  # 保留20%给系统
        memory_limited_batch = int(usable_memory / self.model_memory_per_sample)
        
        # 2. 基于SM数量和最优负载系数的推荐批处理大小
        sm_optimal_batch = int(gpu_chars["sm_count"] * gpu_chars["optimal_load_factor"])
        
        # 3. 基于内存带宽的最优批处理大小
        # 内存带宽利用率在70-80%时最优
        # 简化计算：带宽(GB/s) / 每样本内存(GB) * 利用率
        bandwidth_optimal_batch = int(gpu_chars["memory_bandwidth_gbps"] * 0.8 / 
                                    self.model_memory_per_sample)
        
        # 4. 基于架构特性的调整
        arch_multipliers = {
            "Hopper": 1.2,
            "Ada Lovelace": 1.0,
            "Ampere": 0.9,
            "Turing": 0.8,
            "Pascal": 0.7,
            "Legacy": 0.6,
            "CPU": 0.3
        }
        
        arch_multiplier = arch_multipliers.get(gpu_chars["architecture"], 0.8)
        
        # 5. 综合计算最优批处理大小
        candidates = [
            memory_limited_batch,
            sm_optimal_batch,
            bandwidth_optimal_batch  # 重新启用带宽优化
        ]
        
        # 对于RTX 4060 Ti，使用最优批处理大小
        print(f"🔍 检查GPU名称: '{gpu_chars['name']}'")
        if "RTX 4060 Ti" in gpu_chars["name"]:
            # 使用40以获得最佳GPU利用率
            optimal_batch = 8
            base_batch = optimal_batch  # 用于记录
            print(f"🔧 RTX 4060 Ti优化: 设置批处理大小为 {optimal_batch} (最优配置)")
        elif "4060 Ti" in gpu_chars["name"]:
            optimal_batch = 8
            base_batch = optimal_batch
            print(f"🔧 4060 Ti优化: 设置批处理大小为 {optimal_batch} (最优配置)")
        else:
            # 其他GPU使用算法计算
            # 选择候选值中的较大者，但不超过内存限制
            base_batch = max(sm_optimal_batch, min(bandwidth_optimal_batch, memory_limited_batch))
            
            # 应用架构调整
            optimal_batch = int(base_batch * arch_multiplier)
            
            # 确保在合理范围内，但允许更大的批处理大小
            optimal_batch = max(8, min(optimal_batch, memory_limited_batch))
            
            # 调整到4的倍数 (GPU友好)
            optimal_batch = (optimal_batch // 4) * 4
        
        calculation_details = {
            "memory_limited": memory_limited_batch,
            "sm_optimal": sm_optimal_batch,
            "bandwidth_optimal": bandwidth_optimal_batch,
            "base_batch": base_batch,
            "arch_multiplier": arch_multiplier,
            "final_optimal": optimal_batch,
            "reasoning": f"基于{gpu_chars['sm_count']}个SM，{gpu_chars['optimal_load_factor']:.1f}负载系数，{gpu_chars['memory_bandwidth_gbps']}GB/s带宽"
        }
        
        return optimal_batch, calculation_details
    
    def validate_batch_size(self, batch_size: int, input_shape: Tuple) -> bool:
        """验证批处理大小是否可行"""
        if self.device.type != 'cuda':
            return True
        
        try:
            # 创建测试张量 - 修正输入形状
            # input_shape已经包含了batch维度，所以要去掉第一个维度
            actual_shape = input_shape[1:] if len(input_shape) > 3 else input_shape
            test_tensor = torch.randn(batch_size, *actual_shape, 
                                    device=self.device, dtype=torch.float16)
            
            # 模拟一些计算
            result = torch.sum(test_tensor, dim=1)
            
            # 清理
            del test_tensor, result
            torch.cuda.empty_cache()
            
            return True
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                return False
            raise e
    
    def optimize_for_model(self, input_shape: Tuple, force_recompute: bool = False) -> Dict:
        """为特定模型优化参数"""
        
        # 生成缓存键
        cache_key = f"{self.gpu_name}_{input_shape}_{self.model_memory_per_sample}"
        
        # 检查缓存
        if not force_recompute and cache_key in self.optimization_cache:
            cached_result = self.optimization_cache[cache_key]
            print(f"🔄 使用缓存的优化结果: batch_size={cached_result['optimal_batch_size']}")
            return cached_result
        
        print(f"🚀 为 {self.gpu_name} 动态优化参数...")
        
        # 获取GPU特性
        gpu_chars = self.get_gpu_characteristics()
        
        # 计算最优批处理大小
        optimal_batch, calc_details = self.calculate_optimal_batch_size(gpu_chars)
        
        print(f"📊 GPU特性分析:")
        print(f"   - SM数量: {gpu_chars['sm_count']}")
        print(f"   - 内存带宽: {gpu_chars['memory_bandwidth_gbps']} GB/s")
        print(f"   - 最优负载系数: {gpu_chars['optimal_load_factor']}")
        print(f"   - 架构: {gpu_chars['architecture']}")
        
        print(f"🎯 批处理大小计算:")
        print(f"   - 内存限制: {calc_details['memory_limited']}")
        print(f"   - SM最优: {calc_details['sm_optimal']}")
        print(f"   - 带宽最优: {calc_details['bandwidth_optimal']}")
        print(f"   - 架构调整: ×{calc_details['arch_multiplier']}")
        print(f"   - 最终推荐: {optimal_batch}")
        
        # 验证批处理大小
        print(f"🔍 验证批处理大小 {optimal_batch}...")
        if self.validate_batch_size(optimal_batch, input_shape):
            validated_batch = optimal_batch
            print(f"✅ 验证通过")
        else:
            # 如果验证失败，逐步减小
            print(f"⚠️ 验证失败，自动调整...")
            for test_batch in range(optimal_batch - 4, 4, -4):
                if self.validate_batch_size(test_batch, input_shape):
                    validated_batch = test_batch
                    print(f"✅ 调整后的批处理大小: {validated_batch}")
                    break
            else:
                validated_batch = 8
                print(f"⚠️ 使用保守批处理大小: {validated_batch}")
        
        # 其他优化参数
        optimization_result = {
            "optimal_batch_size": validated_batch,
            "gpu_characteristics": gpu_chars,
            "calculation_details": calc_details,
            "memory_conservative": gpu_chars["memory_gb"] < 8,  # 只有小于8GB才保守
            "aggressive_cleanup": gpu_chars["architecture"] in ["Pascal", "Legacy"],  # 只有老架构需要激进清理
            "use_mixed_precision": gpu_chars["compute_capability"][0] >= 7,  # Turing及以上支持
            "optimization_timestamp": time.time()
        }
        
        # 缓存结果
        self.optimization_cache[cache_key] = optimization_result
        self._save_cache()
        
        print(f"🎉 优化完成! 推荐配置:")
        print(f"   - 批处理大小: {validated_batch}")
        print(f"   - 内存保守模式: {optimization_result['memory_conservative']}")
        print(f"   - 激进清理: {optimization_result['aggressive_cleanup']}")
        print(f"   - 混合精度: {optimization_result['use_mixed_precision']}")
        
        return optimization_result
    
    def get_dynamic_parameters(self, input_shape: Tuple) -> Dict:
        """获取动态优化参数 - 主要接口"""
        return self.optimize_for_model(input_shape)

# 全局优化器实例
_global_optimizer = None

def get_adaptive_optimizer(model_memory_per_sample: float = 0.3) -> AdaptiveGPUOptimizer:
    """获取全局自适应优化器实例"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = AdaptiveGPUOptimizer(model_memory_per_sample)
    return _global_optimizer

def optimize_for_current_gpu(input_shape: Tuple) -> Dict:
    """为当前GPU优化参数的便捷函数"""
    optimizer = get_adaptive_optimizer()
    return optimizer.get_dynamic_parameters(input_shape)

if __name__ == "__main__":
    # 测试优化器
    optimizer = AdaptiveGPUOptimizer()
    
    # 模拟TrackNet输入形状
    input_shape = (12, 288, 512)  # channels, height, width
    
    result = optimizer.optimize_for_model(input_shape)
    
    print(f"\n🎯 最终优化结果:")
    print(f"批处理大小: {result['optimal_batch_size']}")
    print(f"GPU: {result['gpu_characteristics']['name']}")
    print(f"推荐理由: {result['calculation_details']['reasoning']}")