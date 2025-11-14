import numpy as np
import cv2
import time
from typing import List, Tuple, Dict, Any
import math

# 假设这些常量在 config 中定义
VERTICAL_DEGREE = 180.0
HORIZONTAL_DEGREE = 360.0

# 量化字典 (需要根据实际情况调整)
quantization_dict = {
    0: [1.0, 1.0, 0.1],
    1: [0.5, 0.5, 0.05],
    2: [0.25, 0.25, 0.025],
    # 添加更多量化级别...
}

# 假设这些类型在其他模块中定义
class PointCloud:
    """点云数据结构"""
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class PccResult:
    """PCC 结果数据结构"""
    def __init__(self):
        self.proj_times = []
        self.compression_rate = []
        self.loss_rate = []
        self.fit_times = []

# 导入其他模块的函数 (需要在其他文件中实现)
from binary_compressor import compress_data
from dct import sadct, delta_sadct
from serializer import serialize_data, q_serialize_data
from pcc_module import map_projection, compute_loss_rate
from encoder import encode_occupation_mat, single_channel_encode

class EncoderModule:
    """
    编码器模块，负责将点云数据编码为压缩格式
    """
    
    def __init__(self, pitch_precision: float = None, yaw_precision: float = None, 
                 threshold: float = None, tile_size: int = None, q_level: int = None):
        """
        初始化编码器模块
        
        Args:
            pitch_precision: 俯仰角精度
            yaw_precision: 偏航角精度  
            threshold: 平面拟合阈值
            tile_size: 瓦片大小
            q_level: 量化级别
        """
        if q_level is not None and pitch_precision is None:
            # 使用量化级别初始化
            self.q_level = q_level
            self.yaw_precision = quantization_dict[q_level][0]
            self.pitch_precision = quantization_dict[q_level][1]
            self.threshold = quantization_dict[q_level][2]
        else:
            # 使用具体参数初始化
            self.pitch_precision = pitch_precision
            self.yaw_precision = yaw_precision
            self.threshold = threshold
            self.q_level = q_level if q_level is not None else -1
        
        self.tile_size = tile_size
        
        # 计算矩阵尺寸
        self.row = int(VERTICAL_DEGREE / self.yaw_precision)
        self.row = ((self.row + tile_size - 1) // tile_size) * tile_size
        self.col = int(HORIZONTAL_DEGREE / self.pitch_precision) + tile_size
        self.col = ((self.col + tile_size - 1) // tile_size) * tile_size
        
        # 初始化矩阵
        self.f_mat = np.zeros((self.row, self.col, 4), dtype=np.float32)  # range image
        self.b_mat = np.zeros((self.row // tile_size, self.col // tile_size), dtype=np.int32)
        self.occ_mat = np.zeros((self.row // tile_size, self.col // tile_size), dtype=np.int32)
        self.dct_mat = np.zeros((self.row, self.col), dtype=np.float64)
        
        # 初始化其他数据结构
        self.tile_fit_lengths = []
        self.coefficients = []  # 平面系数
        self.unfit_nums = []
        self.idx_sizes = (self.row // tile_size, self.col // tile_size)
        
        print(f"precision: {self.pitch_precision} {self.yaw_precision}")
    
    def encode(self, pcloud_data: List[PointCloud]) -> PccResult:
        """
        编码点云数据
        
        Args:
            pcloud_data: 点云数据列表
            
        Returns:
            PccResult: 编码结果
        """
        pcc_res = PccResult()
        
        # 初始化计时和度量
        proj_time, fit_time = 0.0, 0.0
        psnr, total_pcloud_size = 0.0, 0.0
        
        print(f"CURRENT pcloud size: {len(pcloud_data)}")
        
        # 地图投影
        proj_time = map_projection(self.f_mat, pcloud_data, self.pitch_precision, self.yaw_precision, 'e')
        pcc_res.proj_times.append(proj_time)
        
        # 计算压缩率
        compression_rate = 8.0 * self.f_mat.shape[1] * self.f_mat.shape[0] / len(pcloud_data)
        pcc_res.compression_rate.append(compression_rate)
        
        # 计算损失率 (PSNR)
        # psnr = compute_loss_rate(self.f_mat, pcloud_data, self.pitch_precision, self.yaw_precision)
        pcc_res.loss_rate.append(psnr)
        
        print(f"Loss rate [PSNR]: {psnr} Compression rate: {compression_rate} bpp.")
        
        # 拟合范围地图
        mat_div_tile_sizes = [self.row // self.tile_size, self.col // self.tile_size]
        
        # 编码占据矩阵
        encode_occupation_mat(self.f_mat, self.occ_mat, self.tile_size, mat_div_tile_sizes)
        
        # 单通道编码
        fit_time = single_channel_encode(
            self.f_mat, self.b_mat, mat_div_tile_sizes, self.coefficients,
            self.unfit_nums, self.tile_fit_lengths, self.threshold, self.tile_size
        )
        print(f"fit_time: {fit_time}")
        
        # 创建点云掩码
        mask = np.zeros((self.row, self.col), dtype=np.int32)
        for i in range(self.row):
            for j in range(self.col):
                if self.f_mat[i, j, 0] != 0:
                    mask[i, j] = 1
        
        # 清除已拟合点的瓦片
        for r_idx in range(mat_div_tile_sizes[0]):
            for c_idx in range(mat_div_tile_sizes[1]):
                if self.b_mat[r_idx, c_idx] == 1:
                    for i in range(self.tile_size):
                        for j in range(self.tile_size):
                            mask[r_idx * self.tile_size + i, c_idx * self.tile_size + j] = 0
        
        # 准备 SADCT 数据
        value_map = np.zeros((self.row, self.col), dtype=np.float64)
        for i in range(self.row):
            for j in range(self.col):
                value_map[i, j] = self.f_mat[i, j, 0]
        
        mask_map = np.zeros((self.row, self.col), dtype=bool)
        for i in range(self.row):
            for j in range(self.col):
                mask_map[i, j] = mask[i, j] == 1
        
        # SADCT 变换
        start_time = time.time()
        
        # 根据条件选择使用哪种 DCT
        USE_SADCT = True  # 这应该是一个配置选项
        if USE_SADCT:
            coeff_res = sadct(value_map, mask_map)
        else:
            coeff_res = delta_sadct(value_map, mask_map)
        
        end_time = time.time()
        print(f"SADCT time: {end_time - start_time}")
        
        self.dct_mat = coeff_res[0]  # 假设返回的是 (系数, 掩码) 元组
        pcc_res.fit_times.append(fit_time)
        
        return pcc_res
    
    def pack_data(self) -> bytes:
        """
        打包编码数据
        
        Returns:
            bytes: 序列化后的数据
        """
        if self.q_level != -1:
            data = q_serialize_data(
                self.q_level, self.b_mat, self.idx_sizes, self.coefficients,
                self.occ_mat, self.tile_fit_lengths, self.dct_mat
            )
        else:
            data = serialize_data(
                self.b_mat, self.idx_sizes, self.coefficients, 
                self.occ_mat, self.tile_fit_lengths, self.dct_mat
            )
        return data
    
    def encode_to_data(self, pcloud_data: List[PointCloud], use_compress: bool = False) -> bytes:
        """
        将点云数据编码为二进制数据
        
        Args:
            pcloud_data: 点云数据列表
            use_compress: 是否使用压缩
            
        Returns:
            bytes: 编码后的数据
        """
        self.encode(pcloud_data)
        
        start_time = time.time()
        temp_data = self.pack_data()
        
        if use_compress:
            temp_data = compress_data(temp_data)
        
        end_time = time.time()
        print(f"compress time: {end_time - start_time}")
        
        return temp_data

# 辅助函数实现 (需要在其他模块中完整实现)

def map_projection(f_mat: np.ndarray, pcloud_data: List[PointCloud], 
                  pitch_precision: float, yaw_precision: float, mode: str) -> float:
    """
    地图投影函数
    
    Args:
        f_mat: 目标矩阵
        pcloud_data: 点云数据
        pitch_precision: 俯仰角精度
        yaw_precision: 偏航角精度
        mode: 投影模式
        
    Returns:
        float: 投影时间
    """
    # 实现地图投影逻辑
    start_time = time.time()
    
    # 这里实现具体的投影算法
    # 将点云投影到球面坐标并填充到 f_mat
    
    end_time = time.time()
    return end_time - start_time

def encode_occupation_mat(f_mat: np.ndarray, occ_mat: np.ndarray, 
                         tile_size: int, mat_div_tile_sizes: List[int]):
    """
    编码占据矩阵
    
    Args:
        f_mat: 输入矩阵
        occ_mat: 占据矩阵
        tile_size: 瓦片大小
        mat_div_tile_sizes: 瓦片分割尺寸
    """
    # 实现占据矩阵编码逻辑
    pass

def single_channel_encode(f_mat: np.ndarray, b_mat: np.ndarray, 
                         mat_div_tile_sizes: List[int], coefficients: List,
                         unfit_nums: List, tile_fit_lengths: List,
                         threshold: float, tile_size: int) -> float:
    """
    单通道编码
    
    Args:
        f_mat: 输入矩阵
        b_mat: 拟合标记矩阵
        mat_div_tile_sizes: 瓦片分割尺寸
        coefficients: 平面系数列表
        unfit_nums: 不适合拟合的数量列表
        tile_fit_lengths: 瓦片拟合长度列表
        threshold: 拟合阈值
        tile_size: 瓦片大小
        
    Returns:
        float: 拟合时间
    """
    # 实现单通道编码逻辑
    start_time = time.time()
    
    # 这里实现平面拟合和编码逻辑
    
    end_time = time.time()
    return end_time - start_time

def serialize_data(b_mat: np.ndarray, idx_sizes: Tuple[int, int], 
                  coefficients: List, occ_mat: np.ndarray, 
                  tile_fit_lengths: List, dct_mat: np.ndarray) -> bytes:
    """
    序列化数据
    
    Args:
        b_mat: 拟合标记矩阵
        idx_sizes: 索引尺寸
        coefficients: 平面系数列表
        occ_mat: 占据矩阵
        tile_fit_lengths: 瓦片拟合长度列表
        dct_mat: DCT 系数矩阵
        
    Returns:
        bytes: 序列化后的数据
    """
    # 实现序列化逻辑
    # 这里应该将各个组件打包为二进制格式
    return b""

def q_serialize_data(q_level: int, b_mat: np.ndarray, idx_sizes: Tuple[int, int],
                    coefficients: List, occ_mat: np.ndarray,
                    tile_fit_lengths: List, dct_mat: np.ndarray) -> bytes:
    """
    量化序列化数据
    
    Args:
        q_level: 量化级别
        b_mat: 拟合标记矩阵
        idx_sizes: 索引尺寸
        coefficients: 平面系数列表
        occ_mat: 占据矩阵
        tile_fit_lengths: 瓦片拟合长度列表
        dct_mat: DCT 系数矩阵
        
    Returns:
        bytes: 序列化后的数据
    """
    # 实现量化序列化逻辑
    return b""

# 使用示例
if __name__ == "__main__":
    # 创建测试点云数据
    test_pcloud = [PointCloud(i * 0.1, j * 0.1, k * 0.1) 
                  for i in range(100) for j in range(100) for k in range(10)]
    
    # 使用量化级别初始化编码器
    encoder = EncoderModule(tile_size=8, q_level=1)
    
    # 编码点云数据
    encoded_data = encoder.encode_to_data(test_pcloud, use_compress=True)
    
    print(f"Encoded data size: {len(encoded_data)} bytes")
    print(f"Original points: {len(test_pcloud)}")