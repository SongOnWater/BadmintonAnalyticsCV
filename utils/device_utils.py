"""
设备管理工具
确保所有张量操作都在正确的设备上进行，避免设备不匹配错误
"""
import torch
from typing import Union, Any

def get_device() -> torch.device:
    """获取当前可用的设备"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensure_tensor_on_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """确保张量在指定设备上"""
    if tensor.device != device:
        return tensor.to(device)
    return tensor

def safe_cpu_conversion(tensor: torch.Tensor) -> torch.Tensor:
    """安全地将张量转换为CPU，避免设备不匹配错误"""
    try:
        # 先确保张量在正确的设备上，然后转移到CPU
        if hasattr(tensor, 'device'):
            return tensor.detach().cpu()
        else:
            return tensor.cpu()
    except Exception as e:
        print(f"Warning: Error converting tensor to CPU: {e}")
        # 如果转换失败，尝试直接转换
        try:
            return tensor.cpu()
        except:
            # 最后尝试强制转换
            return tensor.detach().cpu()

def safe_numpy_conversion(tensor: torch.Tensor) -> Any:
    """安全地将张量转换为numpy数组"""
    try:
        # 先安全地转移到CPU，然后转换为numpy
        cpu_tensor = safe_cpu_conversion(tensor)
        return cpu_tensor.numpy()
    except Exception as e:
        print(f"Warning: Error converting tensor to numpy: {e}")
        # 如果转换失败，返回None
        return None

def ensure_model_on_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """确保模型在指定设备上"""
    if next(model.parameters()).device != device:
        return model.to(device)
    return model
