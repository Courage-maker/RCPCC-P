import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Union
import math

# 假设这些常量在 config 中定义
VERTICAL_DEGREE = 180.0
HORIZONTAL_DEGREE = 360.0

# 导入其他模块的函数
from binary_compressor import decompress_data
from dct import saidct, delta_saidct
from serializer import deserialize_data, q_deserialize_data
from decoder import single_channel_decode
from utils.struct import PointCloud

class DecoderModule:
    """
    解码器模块，负责将压缩数据解码为点云
    """
    
    def __init__(self, pitch_precision: float = None, yaw_precision: float = None, 
                 tile_size: int = None, data: bytes = None, use_compress: bool = False, 
                 ksample: int = -1):
        """
        初始化解码器模块
        
        Args:
            pitch_precision: 俯仰角精度
            yaw_precision: 偏航角精度
            tile_size: 瓦片大小
            data: 压缩数据 (如果提供，则从数据初始化)
            use_compress: 是否使用压缩数据
            ksample: 采样参数
        """
        self.tile_size = tile_size
        self.restored_pcloud = []
        
        if data is not None:
            # 从数据初始化
            if use_compress:
                success = q_deserialize_data(
                    decompress_data(data), self.yaw_precision, self.pitch_precision,
                    self.b_mat, self.idx_sizes, self.coefficients, self.occ_mat,
                    self.tile_fit_lengths, self.dct_mat
                )
            else:
                success = deserialize_data(
                    data, self.b_mat, self.idx_sizes, self.coefficients,
                    self.occ_mat, self.tile_fit_lengths, self.dct_mat
                )
            
            if not success:
                print("deserialize data failed")
                return
            
            self.row = self.idx_sizes[0] * tile_size
            self.col = self.idx_sizes[1] * tile_size
            self.r_mat = np.zeros((self.row, self.col), dtype=np.float32)
            self.unfit_mask_mat = np.zeros((self.row, self.col), dtype=bool)
            self.unfit_nums = np.zeros(self.row * self.col, dtype=np.float32)
            
            if ksample > 1:
                self.decode_with_ksample(self.b_mat, self.idx_sizes, self.coefficients,
                                       self.occ_mat, self.tile_fit_lengths, self.dct_mat, ksample)
                print(f"ksample: {ksample}")
            else:
                self.decode(self.b_mat, self.idx_sizes, self.coefficients,
                          self.occ_mat, self.tile_fit_lengths, self.dct_mat)
        else:
            # 使用参数初始化
            self.pitch_precision = pitch_precision
            self.yaw_precision = yaw_precision
            
            self.row = int(VERTICAL_DEGREE / yaw_precision)
            self.row = ((self.row + tile_size - 1) // tile_size) * tile_size
            self.col = int(HORIZONTAL_DEGREE / pitch_precision) + tile_size
            self.col = ((self.col + tile_size - 1) // tile_size) * tile_size
            
            self.r_mat = np.zeros((self.row, self.col), dtype=np.float32)
            self.b_mat = np.zeros((self.row // tile_size, self.col // tile_size), dtype=np.int32)
            self.occ_mat = np.zeros((self.row // tile_size, self.col // tile_size), dtype=np.int32)
            self.dct_mat = np.zeros((self.row, self.col), dtype=np.float64)
            self.unfit_mask_mat = np.zeros((self.row, self.col), dtype=bool)
            self.unfit_nums = np.zeros(self.row * self.col, dtype=np.float32)
            self.coefficients = []
            self.tile_fit_lengths = []
    
    def decode(self, b_mat: np.ndarray, idx_sizes: Tuple[int, int],
               coefficients: List, occ_mat: np.ndarray,
               tile_fit_lengths: List, dct_mat: np.ndarray):
        """
        解码数据
        
        Args:
            b_mat: 拟合标记矩阵
            idx_sizes: 索引尺寸
            coefficients: 平面系数列表
            occ_mat: 占据矩阵
            tile_fit_lengths: 瓦片拟合长度列表
            dct_mat: DCT 系数矩阵
        """
        # 单通道解码
        single_channel_decode(
            self.r_mat, b_mat, idx_sizes, coefficients, occ_mat,
            tile_fit_lengths, self.tile_size, dct_mat
        )
        
        print(f"tile_fit_lengths size: {len(tile_fit_lengths)}")
        print(f"coefficients size: {len(coefficients)}")
        print(f"b_mat shape: {b_mat.shape}")
        print(f"r_mat shape: {self.r_mat.shape}")
        print(f"occ_mat shape: {occ_mat.shape}")
        
        # 可视化 (可选)
        # mask = self.r_mat.copy()
        # cv2.imshow("r_mat", mask)
        # cv2.waitKey(0)
        
        # 恢复点云
        self.restore_pcloud(self.r_mat, self.pitch_precision, self.yaw_precision)
        print(f"pointcloud size: {len(self.restored_pcloud)}")
    
    def decode_with_ksample(self, b_mat: np.ndarray, idx_sizes: Tuple[int, int],
                           coefficients: List, occ_mat: np.ndarray,
                           tile_fit_lengths: List, dct_mat: np.ndarray, ksample: int):
        """
        带采样的解码
        
        Args:
            b_mat: 拟合标记矩阵
            idx_sizes: 索引尺寸
            coefficients: 平面系数列表
            occ_mat: 占据矩阵
            tile_fit_lengths: 瓦片拟合长度列表
            dct_mat: DCT 系数矩阵
            ksample: 采样参数
        """
        extra_pc = []
        single_channel_decode(
            self.r_mat, b_mat, idx_sizes, coefficients, occ_mat,
            tile_fit_lengths, self.tile_size, dct_mat, ksample, extra_pc
        )
        
        print(f"tile_fit_lengths size: {len(tile_fit_lengths)}")
        print(f"coefficients size: {len(coefficients)}")
        print(f"b_mat shape: {b_mat.shape}")
        print(f"r_mat shape: {self.r_mat.shape}")
        print(f"occ_mat shape: {occ_mat.shape}")
        
        # 恢复点云
        self.restore_pcloud(self.r_mat, self.pitch_precision, self.yaw_precision)
        self.restore_extra_pcloud(self.pitch_precision, self.yaw_precision, extra_pc)
        
        print(f"Extra pointcloud size: {len(extra_pc)}")
        print(f"pointcloud size: {len(self.restored_pcloud)}")
    
    def unpack_data(self, data: bytes):
        """
        解包数据
        
        Args:
            data: 序列化数据
        """
        success = deserialize_data(
            data, self.b_mat, self.idx_sizes, self.coefficients,
            self.occ_mat, self.tile_fit_lengths, self.dct_mat
        )
        return success
    
    def decode_from_data(self, data: bytes, use_compress: bool = False, ksample: int = -1):
        """
        从数据解码
        
        Args:
            data: 压缩数据
            use_compress: 是否使用压缩
            ksample: 采样参数
        """
        if use_compress:
            success = self.unpack_data(decompress_data(data))
        else:
            success = self.unpack_data(data)
        
        if not success:
            print("Failed to unpack data")
            return
        
        if ksample > 1:
            self.decode_with_ksample(
                self.b_mat, self.idx_sizes, self.coefficients,
                self.occ_mat, self.tile_fit_lengths, self.dct_mat, ksample
            )
        else:
            self.decode(
                self.b_mat, self.idx_sizes, self.coefficients,
                self.occ_mat, self.tile_fit_lengths, self.dct_mat
            )
    
    def restore_pcloud(self, r_mat: np.ndarray, pitch_precision: float, yaw_precision: float):
        """
        从范围矩阵恢复点云
        
        Args:
            r_mat: 范围矩阵
            pitch_precision: 俯仰角精度
            yaw_precision: 偏航角精度
        """
        self.restored_pcloud = []
        
        # 实现从球面坐标到笛卡尔坐标的转换
        for i in range(r_mat.shape[0]):
            for j in range(r_mat.shape[1]):
                range_val = r_mat[i, j]
                if range_val > 0:  # 有效的范围值
                    # 计算球面坐标
                    theta = i * yaw_precision  # 垂直角度
                    phi = j * pitch_precision  # 水平角度
                    
                    # 转换为弧度
                    theta_rad = np.radians(theta)
                    phi_rad = np.radians(phi)
                    
                    # 转换为笛卡尔坐标
                    x = range_val * np.sin(theta_rad) * np.cos(phi_rad)
                    y = range_val * np.sin(theta_rad) * np.sin(phi_rad)
                    z = range_val * np.cos(theta_rad)
                    
                    self.restored_pcloud.append(PointCloud(x, y, z))
    
    def restore_extra_pcloud(self, pitch_precision: float, yaw_precision: float, extra_pc: List):
        """
        恢复额外点云
        
        Args:
            pitch_precision: 俯仰角精度
            yaw_precision: 偏航角精度
            extra_pc: 额外点云数据
        """
        # 实现额外点云的恢复
        # 这里假设 extra_pc 已经是 PointCloud 对象列表
        self.restored_pcloud.extend(extra_pc)

# 辅助函数实现

def single_channel_decode(r_mat: np.ndarray, b_mat: np.ndarray, idx_sizes: Tuple[int, int],
                         coefficients: List, occ_mat: np.ndarray, tile_fit_lengths: List,
                         tile_size: int, dct_mat: np.ndarray, ksample: int = -1, 
                         extra_pc: List = None) -> bool:
    """
    单通道解码
    
    Args:
        r_mat: 范围矩阵
        b_mat: 拟合标记矩阵
        idx_sizes: 索引尺寸
        coefficients: 平面系数列表
        occ_mat: 占据矩阵
        tile_fit_lengths: 瓦片拟合长度列表
        tile_size: 瓦片大小
        dct_mat: DCT 系数矩阵
        ksample: 采样参数
        extra_pc: 额外点云列表
        
    Returns:
        bool: 解码是否成功
    """
    # 实现单通道解码逻辑
    # 这里应该包括平面拟合的解码和 DCT 系数的逆变换
    
    # 1. 处理平面拟合部分
    for r_idx in range(idx_sizes[0]):
        for c_idx in range(idx_sizes[1]):
            if b_mat[r_idx, c_idx] == 1:
                # 平面拟合区域
                coeff_idx = sum(tile_fit_lengths[:r_idx * idx_sizes[1] + c_idx])
                if coeff_idx < len(coefficients):
                    # 使用平面系数恢复数据
                    a, b, c, d = coefficients[coeff_idx]
                    # 在瓦片区域内应用平面方程
                    for i in range(tile_size):
                        for j in range(tile_size):
                            global_i = r_idx * tile_size + i
                            global_j = c_idx * tile_size + j
                            if occ_mat[r_idx, c_idx] == 1:  # 有数据的区域
                                # 使用平面方程计算范围值
                                r_mat[global_i, global_j] = -(a * i + b * j + d) / c if c != 0 else 0
    
    # 2. 处理 DCT 部分
    # 创建掩码矩阵
    mask = np.zeros_like(r_mat, dtype=bool)
    for i in range(r_mat.shape[0]):
        for j in range(r_mat.shape[1]):
            # 如果该位置没有被平面拟合覆盖且应该有数据
            # 这里需要根据实际情况确定掩码
            pass
    
    # 逆 DCT 变换
    # 根据条件选择使用哪种逆 DCT
    USE_SADCT = True  # 这应该是一个配置选项
    if USE_SADCT:
        idct_result = saidct(dct_mat, mask)
    else:
        idct_result = delta_saidct(dct_mat, mask)
    
    # 将逆 DCT 结果合并到范围矩阵中
    for i in range(r_mat.shape[0]):
        for j in range(r_mat.shape[1]):
            if mask[i, j]:
                r_mat[i, j] = idct_result[i, j]
    
    # 3. 处理采样 (如果启用)
    if ksample > 1 and extra_pc is not None:
        # 实现额外的采样逻辑
        pass
    
    return True

def deserialize_data(data: bytes, b_mat: np.ndarray, idx_sizes: Tuple[int, int],
                    coefficients: List, occ_mat: np.ndarray, 
                    tile_fit_lengths: List, dct_mat: np.ndarray) -> bool:
    """
    反序列化数据
    
    Args:
        data: 序列化数据
        b_mat: 拟合标记矩阵
        idx_sizes: 索引尺寸
        coefficients: 平面系数列表
        occ_mat: 占据矩阵
        tile_fit_lengths: 瓦片拟合长度列表
        dct_mat: DCT 系数矩阵
        
    Returns:
        bool: 反序列化是否成功
    """
    # 实现反序列化逻辑
    # 这里应该从二进制数据中解析出各个组件
    return True

def q_deserialize_data(data: bytes, yaw_precision: float, pitch_precision: float,
                      b_mat: np.ndarray, idx_sizes: Tuple[int, int],
                      coefficients: List, occ_mat: np.ndarray,
                      tile_fit_lengths: List, dct_mat: np.ndarray) -> bool:
    """
    量化反序列化数据
    
    Args:
        data: 序列化数据
        yaw_precision: 偏航角精度
        pitch_precision: 俯仰角精度
        b_mat: 拟合标记矩阵
        idx_sizes: 索引尺寸
        coefficients: 平面系数列表
        occ_mat: 占据矩阵
        tile_fit_lengths: 瓦片拟合长度列表
        dct_mat: DCT 系数矩阵
        
    Returns:
        bool: 反序列化是否成功
    """
    # 实现量化反序列化逻辑
    return True

# 使用示例
if __name__ == "__main__":
    # 示例1: 从数据解码
    with open("compressed_data.bin", "rb") as f:
        compressed_data = f.read()
    
    decoder = DecoderModule(data=compressed_data, tile_size=8, use_compress=True)
    print(f"Decoded point cloud size: {len(decoder.restored_pcloud)}")
    
    # 示例2: 使用参数初始化后解码
    decoder2 = DecoderModule(pitch_precision=1.0, yaw_precision=1.0, tile_size=8)
    with open("raw_data.bin", "rb") as f:
        raw_data = f.read()
    
    decoder2.decode_from_data(raw_data, use_compress=False)
    print(f"Decoded point cloud size: {len(decoder2.restored_pcloud)}")