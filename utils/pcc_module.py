import time
import numpy as np
from typing import List, Dict, Any, Union
import cv2
from pathlib import Path
import statistics

class PccResult:
    """PCC 结果数据类"""
    def __init__(self):
        self.compression_rate = []
        self.restored_compression_rate = []
        self.stream_compression_rates = []
        self.proj_times = []
        self.fit_times = []
        self.merge_times = []
        self.loss_rate = []
        self.restored_loss_rate = []
        self.match_pcts = []
        self.matchs = []
        self.unmatch_pcts = []
        self.unmatchs = []
        self.fitting_ratios = []


def map_projection(f_mat: np.ndarray, pcloud_data: List[Any], 
                   pitch_precision: float, yaw_precision: float, 
                   option: str) -> float:
    """
    将点云数据投影到范围矩阵
    
    Args:
        f_mat: 输出范围矩阵
        pcloud_data: 点云数据列表
        pitch_precision: 俯仰角精度
        yaw_precision: 偏航角精度
        option: 选项 ('e' 或其他)
    
    Returns:
        投影时间（秒）
    """
    print(f"Range Map: {f_mat.shape[0]}, {f_mat.shape[1]}")

    proj_start = time.time()

    # 将点云转换为范围矩阵，带有 x-y 精度图
    if option == 'e':
        pcloud_to_mat_vec4f(pcloud_data, f_mat, pitch_precision, yaw_precision)
    else:
        pcloud_to_mat_float(pcloud_data, f_mat, pitch_precision, yaw_precision)

    proj_end = time.time()
    proj_time = proj_end - proj_start

    print(f"Time taken by projection: {proj_time:.6f} sec.")
    return proj_time


def pcloud_to_mat_vec4f(pcloud_data: List[Any], f_mat: np.ndarray, 
                        pitch_precision: float, yaw_precision: float) -> None:
    """
    将点云数据转换为 Vec4f 类型的矩阵
    
    Note: 这是示例实现，需要根据实际的点云数据结构进行调整
    """
    # 清空矩阵
    f_mat.fill(0)
    
    for point in pcloud_data:
        # 根据实际点云数据结构计算坐标
        # 这里假设点云数据有 x, y, z 坐标
        if hasattr(point, 'x') and hasattr(point, 'y') and hasattr(point, 'z'):
            # 计算投影坐标
            u = int((point.y / yaw_precision) + f_mat.shape[1] / 2)
            v = int((point.x / pitch_precision) + f_mat.shape[0] / 2)
            
            if 0 <= u < f_mat.shape[1] and 0 <= v < f_mat.shape[0]:
                # 存储为 Vec4f 格式 [x, y, z, 1.0] 或其他需要的格式
                f_mat[v, u] = [point.x, point.y, point.z, 1.0]


def pcloud_to_mat_float(pcloud_data: List[Any], f_mat: np.ndarray,
                       pitch_precision: float, yaw_precision: float) -> None:
    """
    将点云数据转换为 float 类型的矩阵（范围图）
    
    Note: 这是示例实现，需要根据实际的点云数据结构进行调整
    """
    # 清空矩阵
    f_mat.fill(0.0)
    
    for point in pcloud_data:
        # 根据实际点云数据结构计算坐标和范围
        if hasattr(point, 'x') and hasattr(point, 'y') and hasattr(point, 'z'):
            # 计算投影坐标
            u = int((point.y / yaw_precision) + f_mat.shape[1] / 2)
            v = int((point.x / pitch_precision) + f_mat.shape[0] / 2)
            
            if 0 <= u < f_mat.shape[1] and 0 <= v < f_mat.shape[0]:
                # 计算范围（距离）
                range_val = np.sqrt(point.x**2 + point.y**2 + point.z**2)
                f_mat[v, u] = range_val


def average(data_list: List[Union[float, int]]) -> float:
    """计算列表中数值的平均值"""
    if not data_list:
        return 0.0
    return sum(data_list) / len(data_list)


def print_pcc_res(pcc_res: PccResult) -> None:
    """打印 PCC 结果统计信息"""
    avg_compression_rate = average(pcc_res.compression_rate)
    print(f"Compression rate: {avg_compression_rate:.6f} bpp")
    
    avg_restored_compression_rate = average(pcc_res.restored_compression_rate)
    print(f"Restored compression rate: {avg_restored_compression_rate:.6f} bpp")
    
    avg_stream_compression_rate = average(pcc_res.stream_compression_rates)
    print(f"Stream compression rate: {avg_stream_compression_rate:.6f} bpp")
    
    avg_proj_time = average(pcc_res.proj_times)
    print(f"2D projection time: {avg_proj_time:.6f} sec")
    
    avg_fit_time = average(pcc_res.fit_times)
    print(f"Average fitting time: {avg_fit_time:.6f} sec")
    
    avg_merge_time = average(pcc_res.merge_times)
    print(f"Average merging time: {avg_merge_time:.6f} sec")
    
    avg_loss_rate = average(pcc_res.loss_rate)
    print(f"Loss rate [PSNR]: {avg_loss_rate:.6f}")
    
    avg_loss_rate = average(pcc_res.restored_loss_rate)
    print(f"Restored Loss rate [PSNR]: {avg_loss_rate:.6f}")
    
    avg_match_pct = average(pcc_res.match_pcts)
    avg_match = average(pcc_res.matchs)
    print(f"Merge-Matched Percentage: {avg_match_pct:.6f} in {avg_match:.6f}")
    
    avg_unmatch_pct = average(pcc_res.unmatch_pcts)
    avg_unmatch = average(pcc_res.unmatchs)
    print(f"Merge-Unmatched Percentage: {avg_unmatch_pct:.6f} in {avg_unmatch:.6f}")
    
    avg_fitting_ratio = average(pcc_res.fitting_ratios)
    print(f"Fitting Ratio: {avg_fitting_ratio:.6f}")


# 可选：更详细的统计信息函数
def detailed_pcc_statistics(pcc_res: PccResult) -> Dict[str, Any]:
    """提供更详细的 PCC 统计信息"""
    stats = {}
    
    for attr_name in dir(pcc_res):
        if not attr_name.startswith('_'):
            attr_value = getattr(pcc_res, attr_name)
            if isinstance(attr_value, list) and attr_value:
                stats[attr_name] = {
                    'mean': average(attr_value),
                    'min': min(attr_value),
                    'max': max(attr_value),
                    'std': statistics.stdev(attr_value) if len(attr_value) > 1 else 0.0,
                    'count': len(attr_value)
                }
    
    return stats


def print_detailed_stats(pcc_res: PccResult) -> None:
    """打印详细的统计信息"""
    stats = detailed_pcc_statistics(pcc_res)
    
    print("\n=== Detailed PCC Statistics ===")
    for metric_name, metric_stats in stats.items():
        print(f"\n{metric_name}:")
        print(f"  Mean: {metric_stats['mean']:.6f}")
        print(f"  Min: {metric_stats['min']:.6f}")
        print(f"  Max: {metric_stats['max']:.6f}")
        print(f"  Std: {metric_stats['std']:.6f}")
        print(f"  Count: {metric_stats['count']}")