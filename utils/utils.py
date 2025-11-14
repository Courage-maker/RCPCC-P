import numpy as np
import struct
import open3d as o3d
from typing import List, Tuple, Any, Optional
import cv2
import math
from pathlib import Path

# 常量定义（需要根据实际情况调整）
PI = math.pi
ROW_OFFSET = 0.0  # 需要根据实际配置调整
COL_OFFSET = 0.0  # 需要根据实际配置调整
LIVOX_ROW_OFFSET = 0.0  # 需要根据实际配置调整
LIVOX_COL_OFFSET = 0.0  # 需要根据实际配置调整
USE_LIVOX = False
VERBOSE = True

class PointCloud:
    """点云数据类"""
    def __init__(self, x: float, y: float, z: float, r: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
    
    def to_string_xyz(self) -> str:
        """转换为 XYZ 字符串"""
        return f"[{self.x},{self.y},{self.z}]"
    
    def __repr__(self):
        return f"PointCloud(x={self.x}, y={self.y}, z={self.z}, r={self.r})"


def load_pcloud_ply(file_name: str) -> List[PointCloud]:
    """
    从 PLY 文件加载点云数据
    
    Args:
        file_name: PLY 文件名
        
    Returns:
        点云数据列表
    """
    pcloud_data = []
    
    try:
        # 使用 Open3D 读取 PLY 文件
        pcd = o3d.io.read_point_cloud(file_name)
        points = np.asarray(pcd.points)
        
        for point in points:
            x, y, z = point
            # 过滤距离原点太近的点
            if x*x + y*y + z*z < 2.0:
                continue
            pcloud_data.append(PointCloud(x, y, z, 0.0))
            
    except Exception as e:
        print(f"Error reading PLY file {file_name}: {e}")
    
    return pcloud_data


def load_pcloud(file_name: str) -> List[PointCloud]:
    """
    从二进制 XYZ 文件加载点云数据
    
    Args:
        file_name: 二进制文件名
        
    Returns:
        点云数据列表
    """
    pcloud_data = []
    
    try:
        with open(file_name, 'rb') as file:
            # 读取整个文件
            data = file.read()
            
            # 每点4个float (x, y, z, r)
            point_size = 4 * 4  # 4个float，每个4字节
            num_points = len(data) // point_size
            
            for i in range(num_points):
                offset = i * point_size
                x = struct.unpack('f', data[offset:offset+4])[0]
                y = struct.unpack('f', data[offset+4:offset+8])[0]
                z = struct.unpack('f', data[offset+8:offset+12])[0]
                r = struct.unpack('f', data[offset+12:offset+16])[0]
                
                pcloud_data.append(PointCloud(x, y, z, r))
                
    except Exception as e:
        print(f"Error reading binary point cloud file {file_name}: {e}")
    
    return pcloud_data


def export_pcloud(pcloud_data: List[PointCloud], file_name: str) -> None:
    """
    导出点云数据到二进制文件
    
    Args:
        pcloud_data: 点云数据列表
        file_name: 输出文件名
    """
    try:
        with open(file_name, 'wb') as file:
            for point in pcloud_data:
                file.write(struct.pack('f', point.x))
                file.write(struct.pack('f', point.y))
                file.write(struct.pack('f', point.z))
                file.write(struct.pack('f', point.r))
                
    except Exception as e:
        print(f"Error exporting point cloud to {file_name}: {e}")


def load_pcloud_xyz(file_name: str) -> List[PointCloud]:
    """
    从文本 XYZ 文件加载点云数据
    
    Args:
        file_name: 文本文件名
        
    Returns:
        点云数据列表
    """
    pcloud_data = []
    
    try:
        with open(file_name, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) < 3:
                    print(f"Invalid line: {line}")
                    continue
                    
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    pcloud_data.append(PointCloud(x, y, z, 0.0))
                except ValueError:
                    print(f"Invalid numeric data in line: {line}")
                    
    except Exception as e:
        print(f"Error reading XYZ file {file_name}: {e}")
    
    return pcloud_data


def pcloud2bin(pcloud_data: List[PointCloud], filename: str) -> None:
    """
    将点云数据导出为二进制格式
    
    Args:
        pcloud_data: 点云数据列表
        filename: 输出文件名
    """
    try:
        with open(filename, 'wb') as out_stream:
            for point in pcloud_data:
                out_stream.write(struct.pack('f', point.x))
                out_stream.write(struct.pack('f', point.y))
                out_stream.write(struct.pack('f', point.z))
                out_stream.write(struct.pack('f', point.r))
                
        print(f"Write {filename}")
        
    except Exception as e:
        print(f"Error writing binary point cloud to {filename}: {e}")


def compute_cartesian_coord(radius: float, yaw: float, pitch: float,
                           pitch_precision: float, yaw_precision: float) -> Tuple[float, float, float]:
    """
    从球坐标计算笛卡尔坐标
    
    Args:
        radius: 半径
        yaw: 偏航角
        pitch: 俯仰角
        pitch_precision: 俯仰角精度
        yaw_precision: 偏航角精度
        
    Returns:
        (x, y, z) 笛卡尔坐标
    """
    # 将球坐标转换为笛卡尔坐标
    xy_radius = radius * math.cos(yaw / 180.0 * PI * yaw_precision)
    
    # 计算坐标
    z = radius * math.sin(yaw / 180.0 * PI * yaw_precision)
    x = xy_radius * math.cos(pitch / 180.0 * PI * pitch_precision)
    y = xy_radius * math.sin(pitch / 180.0 * PI * pitch_precision)
    
    return x, y, z


def pcloud2string(pcloud_data: List[PointCloud]) -> str:
    """
    将点云数据转换为字符串
    
    Args:
        pcloud_data: 点云数据列表
        
    Returns:
        点云字符串表示
    """
    points_str = ",".join(point.to_string_xyz() for point in pcloud_data)
    return f"[{points_str}]"


def output_cloud(pcloud_data: List[PointCloud], file_name: str) -> None:
    """
    输出点云数据到 PLY 文件
    
    Args:
        pcloud_data: 点云数据列表
        file_name: 输出文件名
    """
    try:
        with open(file_name, 'w') as stream:
            stream.write("ply\nformat ascii 1.0\n")
            stream.write(f"element vertex {len(pcloud_data)}\n")
            stream.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            
            for point in pcloud_data:
                stream.write(f"{point.x} {point.y} {point.z}\n")
                
    except Exception as e:
        print(f"Error writing PLY file {file_name}: {e}")


def output_normalize_cloud(pcloud_data: List[PointCloud], file_name: str) -> List[float]:
    """
    输出归一化的点云数据到 PLY 文件
    
    Args:
        pcloud_data: 点云数据列表
        file_name: 输出文件名
        
    Returns:
        归一化范围 [min_x, max_x, min_y, max_y, min_z, max_z]
    """
    if not pcloud_data:
        return [0.0] * 6
    
    # 计算坐标范围
    x_coords = [point.x for point in pcloud_data]
    y_coords = [point.y for point in pcloud_data]
    z_coords = [point.z for point in pcloud_data]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    min_z, max_z = min(z_coords), max(z_coords)
    
    try:
        with open(file_name, 'w') as stream:
            stream.write("ply\nformat ascii 1.0\n")
            stream.write(f"element vertex {len(pcloud_data)}\n")
            stream.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            
            for point in pcloud_data:
                # 归一化到 0-255 范围
                x_norm = int((point.x - min_x) / (max_x - min_x) * 255) if max_x > min_x else 0
                y_norm = int((point.y - min_y) / (max_y - min_y) * 255) if max_y > min_y else 0
                z_norm = int((point.z - min_z) / (max_z - min_z) * 255) if max_z > min_z else 0
                
                stream.write(f"{x_norm} {y_norm} {z_norm}\n")
                
    except Exception as e:
        print(f"Error writing normalized PLY file {file_name}: {e}")
    
    return [min_x, max_x, min_y, max_y, min_z, max_z]


def neighbor_search(radius: float, img: np.ndarray, row: int, col: int) -> float:
    """
    在邻域内搜索最接近的半径值
    
    Args:
        radius: 目标半径
        img: 范围图像
        row: 行坐标
        col: 列坐标
        
    Returns:
        最接近的半径值
    """
    restored_radius = img[row, col]
    
    # 搜索 5x5 邻域
    for i in range(max(0, row - 2), min(img.shape[0], row + 3)):
        for j in range(max(0, col - 2), min(img.shape[1], col + 3)):
            if abs(restored_radius - radius) > abs(img[i, j] - radius):
                restored_radius = img[i, j]
    
    return restored_radius


def compute_loss_rate(img: np.ndarray, pcloud_data: List[PointCloud],
                     pitch_precision: float, yaw_precision: float) -> float:
    """
    计算点云重建的损失率（PSNR）
    
    Args:
        img: 范围图像
        pcloud_data: 原始点云数据
        pitch_precision: 俯仰角精度
        yaw_precision: 偏航角精度
        
    Returns:
        PSNR 值
    """
    if not pcloud_data:
        return 0.0
    
    # 初始化误差度量
    x_error, y_error, z_error = 0.0, 0.0, 0.0
    error, dist_error = 0.0, 0.0
    norm_error, norm_dist_error = 0.0, 0.0
    max_radius, min_radius = 0.0, float('inf')
    
    for point in pcloud_data:
        x, y, z = point.x, point.y, point.z
        
        # 计算球坐标参数
        dist = math.sqrt(x*x + y*y)
        radius = math.sqrt(x*x + y*y + z*z)
        pitch = math.atan2(y, x) * 180.0 / pitch_precision / PI
        yaw = math.atan2(z, dist) * 180.0 / yaw_precision / PI
        
        if USE_LIVOX:
            row_offset = LIVOX_ROW_OFFSET / yaw_precision
            col_offset = LIVOX_COL_OFFSET / pitch_precision
        else:
            row_offset = ROW_OFFSET / yaw_precision
            col_offset = COL_OFFSET / pitch_precision
        
        col_idx = min(img.shape[1] - 1, max(0, int(pitch + col_offset)))
        row_idx = min(img.shape[0] - 1, max(0, int(yaw + row_offset)))
        
        # 更新最大最小半径
        max_radius = max(max_radius, radius)
        min_radius = min(min_radius, radius)
        
        restored_radius = neighbor_search(radius, img, row_idx, col_idx)
        
        if (math.isnan(restored_radius) or restored_radius > 300.0 or 
            restored_radius < 1.0):
            print(f"[IGNORE]: {radius} vs. {restored_radius}. [x,y,z]: {x}, {y}, {z}")
            restored_radius = 0.1
        
        # 计算重建的笛卡尔坐标
        restored_x, restored_y, restored_z = compute_cartesian_coord(
            restored_radius, (row_idx + 0.5) - row_offset, (col_idx + 0.5) - col_offset,
            pitch_precision, yaw_precision
        )
        
        x_diff = abs(x - restored_x)
        y_diff = abs(y - restored_y)
        z_diff = abs(z - restored_z)
        diff = math.sqrt(x_diff*x_diff + y_diff*y_diff + z_diff*z_diff)
        
        x_error += x_diff
        y_error += y_diff
        z_error += z_diff
        error += diff
        
        norm_error += diff / radius
        dist_error += abs(restored_radius - radius)
        norm_dist_error += abs(restored_radius - radius) / radius
    
    if VERBOSE:
        num_points = len(pcloud_data)
        print(f"the x-error rate: {x_error / num_points}")
        print(f"the y-error rate: {y_error / num_points}")
        print(f"the z-error rate: {z_error / num_points}")
        print(f"the error rate: {error / num_points}")
        print(f"the normal error rate: {norm_error / num_points}")
        print(f"the distance error rate: {dist_error / num_points}")
        print(f"the normal distance error rate: {norm_dist_error / num_points}")
    
    # 计算边界框宽度
    bb_width = 2 * max_radius
    # 计算平均误差
    error = error / len(pcloud_data)
    
    # 计算峰值信噪比 (PSNR)
    if error == 0:
        return float('inf')
    
    return 10.0 * math.log10((bb_width * bb_width) / (error * error))


def restore_pcloud(img: np.ndarray, pitch_precision: float, yaw_precision: float) -> List[PointCloud]:
    """
    从范围图像重建点云数据
    
    Args:
        img: 范围图像
        pitch_precision: 俯仰角精度
        yaw_precision: 偏航角精度
        
    Returns:
        重建的点云数据列表
    """
    restored_pcloud = []
    
    print(f"{pitch_precision} : {yaw_precision}")
    
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            radius = img[row, col]
            if radius <= 0 or math.isinf(radius):
                continue
            
            if USE_LIVOX:
                pitch = (col + 0.5) * pitch_precision - LIVOX_COL_OFFSET
                yaw = (row + 0.5) * yaw_precision - LIVOX_ROW_OFFSET
            else:
                pitch = (col + 0.5) * pitch_precision - COL_OFFSET
                yaw = (row + 0.5) * yaw_precision - ROW_OFFSET
            
            z = radius * math.sin(yaw * PI / 180.0)
            dist = radius * math.cos(yaw * PI / 180.0)
            y = dist * math.sin(pitch * PI / 180.0)
            x = dist * math.cos(pitch * PI / 180.0)
            
            restored_pcloud.append(PointCloud(x, y, z, radius))
    
    return restored_pcloud


def restore_extra_pcloud(pitch_precision: float, yaw_precision: float, 
                        extra_pc: List[List[float]]) -> List[PointCloud]:
    """
    从额外点云数据重建点云
    
    Args:
        pitch_precision: 俯仰角精度
        yaw_precision: 偏航角精度
        extra_pc: 额外点云数据
        
    Returns:
        重建的点云数据列表
    """
    restored_pcloud = []
    
    print(f"{pitch_precision} : {yaw_precision}")
    
    for pc in extra_pc:
        if len(pc) < 3:
            continue
            
        radius = pc[2]
        if radius <= 0 or math.isinf(radius):
            continue
            
        row = pc[0]
        col = pc[1]
        
        if USE_LIVOX:
            pitch = (col + 0.5) * pitch_precision - LIVOX_COL_OFFSET
            yaw = (row + 0.5) * yaw_precision - LIVOX_ROW_OFFSET
        else:
            pitch = (col + 0.5) * pitch_precision - COL_OFFSET
            yaw = (row + 0.5) * yaw_precision - ROW_OFFSET
        
        z = radius * math.sin(yaw * PI / 180.0)
        dist = radius * math.cos(yaw * PI / 180.0)
        y = dist * math.sin(pitch * PI / 180.0)
        x = dist * math.cos(pitch * PI / 180.0)
        
        restored_pcloud.append(PointCloud(x, y, z, radius))
    
    return restored_pcloud


# 模板函数的 Python 版本
def pcloud_to_mat(pcloud_data: List[PointCloud], img: np.ndarray,
                 pitch_precision: float, yaw_precision: float, 
                 use_vec4f: bool = False) -> None:
    """
    将点云数据投影到矩阵中
    
    Args:
        pcloud_data: 点云数据列表
        img: 输出图像矩阵
        pitch_precision: 俯仰角精度
        yaw_precision: 偏航角精度
        use_vec4f: 是否使用 Vec4f 格式
    """
    for point in pcloud_data:
        x, y, z = point.x, point.y, point.z
        
        # 计算球坐标参数
        dist = math.sqrt(x*x + y*y)
        radius = math.sqrt(x*x + y*y + z*z)
        pitch = math.atan2(y, x) * 180.0 / pitch_precision / PI
        yaw = math.atan2(z, dist) * 180.0 / yaw_precision / PI
        
        if USE_LIVOX:
            row_offset = LIVOX_ROW_OFFSET / yaw_precision
            col_offset = LIVOX_COL_OFFSET / pitch_precision
        else:
            row_offset = ROW_OFFSET / yaw_precision
            col_offset = COL_OFFSET / pitch_precision
        
        col_idx = min(img.shape[1] - 1, max(0, int(pitch + col_offset)))
        row_idx = min(img.shape[0] - 1, max(0, int(yaw + row_offset)))
        
        if use_vec4f and len(img.shape) == 3 and img.shape[2] == 4:
            # 使用 Vec4f 格式 [radius, x, y, z]
            img[row_idx, col_idx] = [radius, x, y, z]
        else:
            # 使用标量格式
            img[row_idx, col_idx] = radius


def export_mat(mat: np.ndarray, file_name: str, precision: int = 6) -> None:
    """
    导出矩阵到文本文件
    
    Args:
        mat: 要导出的矩阵
        file_name: 输出文件名
        precision: 数值精度
    """
    try:
        with open(file_name, 'w') as outfile:
            for i in range(mat.shape[1]):  # 列
                for j in range(mat.shape[0]):  # 行
                    if len(mat.shape) == 3 and mat.shape[2] == 4:
                        # Vec4f 格式
                        vec = mat[j, i]
                        outfile.write(f"{vec[0]:.{precision}f}:{vec[1]:.{precision}f},")
                    else:
                        # 标量格式
                        outfile.write(f"{mat[j, i]:.{precision}f},")
                outfile.write("\n")
                
    except Exception as e:
        print(f"Error exporting matrix to {file_name}: {e}")