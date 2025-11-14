import numpy as np
import cv2
import time
from typing import List, Tuple, Dict, Any, Union
import math

# 导入 DCT 模块
from dct import saidct, delta_saidct

# 定义类型别名
Matrix = List[List[float]]
FloatMatrix = List[List[float]]

class Decoder:
    """
    解码器类，负责将压缩数据解码为范围图像
    """
    
    @staticmethod
    def copy_unfit_points(img: np.ndarray, unfit_nums: List[float], 
                         unfit_nums_itr: int, occ_code: int,
                         r_idx: int, c_idx: int, tile_size: int) -> int:
        """
        将不适合平面拟合的点复制到重建的范围图像中
        
        Args:
            img: 重建的图像
            unfit_nums: 不适合拟合的数值列表
            unfit_nums_itr: 不适合拟合数值的迭代器
            occ_code: 占据编码
            r_idx: 行索引
            c_idx: 列索引
            tile_size: 瓦片大小
            
        Returns:
            int: 更新后的迭代器位置
        """
        itr = 0
        for row in range(r_idx * tile_size, (r_idx + 1) * tile_size):
            for col in range(c_idx * tile_size, (c_idx + 1) * tile_size):
                if ((occ_code >> itr) & 1) == 1:
                    img[row, col] = unfit_nums[unfit_nums_itr]
                    unfit_nums_itr += 1
                itr += 1
        return unfit_nums_itr
    
    @staticmethod
    def calc_fit_nums(img: np.ndarray, c: Tuple[float, float, float, float], 
                     occ_code: int, c_idx: int, r_idx: int, 
                     len_itr: int, tile_size: int):
        """
        根据平面系数计算拟合的数值
        
        Args:
            img: 重建的图像
            c: 平面系数 (a, b, c, d)
            occ_code: 占据编码
            c_idx: 列索引
            r_idx: 行索引
            len_itr: 长度迭代器
            tile_size: 瓦片大小
        """
        a, b, c_val, d = c
        itr = 0
        for j in range(tile_size):
            for i in range(tile_size):
                if ((occ_code >> itr) & 1) == 1:
                    val = abs((d + b * j + c_val * (len_itr * tile_size + i)) / a)
                    if not math.isnan(val):
                        img[r_idx * tile_size + j, c_idx * tile_size + i] = val
                itr += 1
    
    @staticmethod
    def generate_uniform_list(k: int) -> List[float]:
        """
        生成均匀分布列表
        
        Args:
            k: 采样数量
            
        Returns:
            List[float]: 均匀分布的偏移值列表
        """
        result = []
        if k == 1:
            result.append(0.0)
        else:
            for i in range(k):
                result.append(float(i + 1) / (k + 1) - 0.5)
        return result
    
    @staticmethod
    def calc_fit_nums_with_ksample(img: np.ndarray, c: Tuple[float, float, float, float], 
                                  occ_code: int, c_idx: int, r_idx: int, 
                                  len_itr: int, tile_size: int, ksample: int, 
                                  extra_pc: List[List[float]]):
        """
        带采样的平面拟合数值计算
        
        Args:
            img: 重建的图像
            c: 平面系数 (a, b, c, d)
            occ_code: 占据编码
            c_idx: 列索引
            r_idx: 行索引
            len_itr: 长度迭代器
            tile_size: 瓦片大小
            ksample: 采样参数
            extra_pc: 额外点云列表
        """
        a, b, c_val, d = c
        itr = 0
        offset = Decoder.generate_uniform_list(ksample)
        
        for j in range(tile_size):
            for i in range(tile_size):
                if ((occ_code >> itr) & 1) == 1:
                    val = abs((d + b * j + c_val * (len_itr * tile_size + i)) / a)
                    if not math.isnan(val):
                        img[r_idx * tile_size + j, c_idx * tile_size + i] = val
                    
                    # ksample 采样
                    if ksample > 1:
                        for a_offset in range(ksample):
                            for b_offset in range(ksample):
                                pc_list = []
                                kval = abs((d + b * (j + offset[a_offset]) + 
                                          c_val * (len_itr * tile_size + i + offset[b_offset])) / a)
                                if not math.isnan(kval) and abs(kval - val) < 0.1:
                                    pc_list.append(r_idx * tile_size + j + offset[a_offset])  # row
                                    pc_list.append(c_idx * tile_size + i + offset[b_offset])  # col
                                    pc_list.append(kval)  # range value
                                    extra_pc.append(pc_list)
                itr += 1
    
    @staticmethod
    def single_channel_decode(img: np.ndarray, b_mat: np.ndarray, idx_sizes: Tuple[int, int],
                             coefficients: List[Tuple[float, float, float, float]], 
                             occ_mat: np.ndarray, tile_fit_lengths: List[int],
                             tile_size: int, dct_mat: Matrix = None, 
                             ksample: int = -1, extra_pc: List[List[float]] = None,
                             unfit_nums: List[float] = None, multi_mat: np.ndarray = None) -> float:
        """
        单通道解码
        
        Args:
            img: 重建的图像
            b_mat: 拟合标记矩阵
            idx_sizes: 索引尺寸
            coefficients: 平面系数列表
            occ_mat: 占据矩阵
            tile_fit_lengths: 瓦片拟合长度列表
            tile_size: 瓦片大小
            dct_mat: DCT 系数矩阵
            ksample: 采样参数
            extra_pc: 额外点云列表
            unfit_nums: 不适合拟合的数值列表
            multi_mat: 多通道矩阵
            
        Returns:
            float: 解码时间
        """
        decode_start = time.time()
        
        tt2 = tile_size * tile_size
        fit_itr = 0
        unfit_nums_itr = 0
        
        unfit_cnt = 0
        fit_cnt = 0
        
        # 处理 DCT 系数（如果提供）
        if dct_mat is not None:
            unfit_nums_list = []
            unfit_mask = [[False] * (idx_sizes[1] * tile_size) for _ in range(idx_sizes[0] * tile_size)]
            
            # 创建不适合拟合的掩码
            for r_idx in range(idx_sizes[0]):
                for c_idx in range(idx_sizes[1]):
                    if b_mat[r_idx, c_idx] == 0:
                        occ_code = occ_mat[r_idx, c_idx]
                        for i in range(tt2):
                            if ((occ_code >> i) & 1) == 1:
                                unfit_mask[r_idx * tile_size + i // tile_size][c_idx * tile_size + i % tile_size] = True
            
            # 逆 DCT 变换
            USE_SADCT = True  # 这应该是一个配置选项
            if USE_SADCT:
                idct_res = saidct(dct_mat, unfit_mask)
            else:
                idct_res = delta_saidct(dct_mat, unfit_mask)
            
            # 提取不适合拟合的数值
            for r_idx in range(idx_sizes[0]):
                for c_idx in range(idx_sizes[1]):
                    if b_mat[r_idx, c_idx] == 0:
                        occ_code = occ_mat[r_idx, c_idx]
                        for i in range(tt2):
                            if ((occ_code >> i) & 1) == 1:
                                unfit_nums_list.append(
                                    idct_res[r_idx * tile_size + i // tile_size][c_idx * tile_size + i % tile_size]
                                )
            
            unfit_nums = unfit_nums_list
            print(f"unfit_nums size: {len(unfit_nums)}")
        
        print(f"occ mat size: {occ_mat.shape[0]} {occ_mat.shape[1]}")
        
        # 解码主循环
        for r_idx in range(idx_sizes[0]):
            len_itr = 0
            current_len = 0
            c = (0.0, 0.0, 0.0, 0.0)
            
            for c_idx in range(idx_sizes[1]):
                tile_status = b_mat[r_idx, c_idx]
                occ_code = occ_mat[r_idx, c_idx]
                
                # 如果不适合拟合，复制不适合拟合的点
                if tile_status == 0:
                    if len_itr < current_len:
                        print(f"[ERROR]: should encode unfit nums right now!")
                        print(f"[INFO]: r_idx {r_idx} c_idx {c_idx} len_itr {len_itr} len {current_len}")
                        return 0.0
                    
                    if unfit_nums is not None:
                        unfit_nums_itr = Decoder.copy_unfit_points(
                            img, unfit_nums, unfit_nums_itr, occ_code, r_idx, c_idx, tile_size
                        )
                    unfit_cnt += 1
                else:
                    # 适合拟合的情况
                    if len_itr < current_len:
                        if multi_mat is None or multi_mat[r_idx, c_idx] == 0:
                            if ksample > 1 and extra_pc is not None:
                                Decoder.calc_fit_nums_with_ksample(
                                    img, c, occ_code, c_idx, r_idx, len_itr, tile_size, ksample, extra_pc
                                )
                            else:
                                Decoder.calc_fit_nums(img, c, occ_code, c_idx, r_idx, len_itr, tile_size)
                        len_itr += 1
                    else:
                        c = coefficients[fit_itr]
                        len_itr = 0
                        current_len = tile_fit_lengths[fit_itr]
                        
                        if multi_mat is None or multi_mat[r_idx, c_idx] == 0:
                            if ksample > 1 and extra_pc is not None:
                                Decoder.calc_fit_nums_with_ksample(
                                    img, c, occ_code, c_idx, r_idx, len_itr, tile_size, ksample, extra_pc
                                )
                            else:
                                Decoder.calc_fit_nums(img, c, occ_code, c_idx, r_idx, len_itr, tile_size)
                        
                        fit_itr += 1
                        len_itr += 1
                    fit_cnt += 1
        
        decode_end = time.time()
        decode_time = decode_end - decode_start
        
        print(f"Single with fitting_cnts: {fit_cnt} with unfitting_cnts: {unfit_cnt}")
        
        return decode_time
    
    @staticmethod
    def calc_fit_nums_with_offset(img: np.ndarray, c: Tuple[float, float, float, float], 
                                 occ_code: int, c_idx: int, r_idx: int, 
                                 len_itr: int, tile_size: int, offset: float):
        """
        带偏移的平面拟合数值计算
        
        Args:
            img: 重建的图像
            c: 平面系数 (a, b, c, d)
            occ_code: 占据编码
            c_idx: 列索引
            r_idx: 行索引
            len_itr: 长度迭代器
            tile_size: 瓦片大小
            offset: 偏移值
        """
        a, b, c_val, d = c
        itr = 0
        for j in range(tile_size):
            for i in range(tile_size):
                if ((occ_code >> itr) & 1) == 1:
                    val = abs((offset + b * j + c_val * (len_itr * tile_size + i)) / a)
                    img[r_idx * tile_size + j, c_idx * tile_size + i] = val
                itr += 1
    
    @staticmethod
    def multi_channel_decode(imgs: List[np.ndarray], b_mat: np.ndarray,
                            idx_sizes: Tuple[int, int], occ_mats: List[np.ndarray],
                            coefficients: List[Tuple[float, float, float, float]],
                            plane_offsets: List[List[float]], tile_fit_lengths: List[int],
                            threshold: float, tile_size: int):
        """
        多通道解码
        
        Args:
            imgs: 图像列表
            b_mat: 拟合标记矩阵
            idx_sizes: 索引尺寸
            occ_mats: 占据矩阵列表
            coefficients: 平面系数列表
            plane_offsets: 平面偏移列表
            tile_fit_lengths: 瓦片拟合长度列表
            threshold: 阈值
            tile_size: 瓦片大小
        """
        tt2 = tile_size * tile_size
        unfit_cnt = 0
        fit_cnt = 0
        
        for ch in range(len(imgs)):
            print(f"[CHANNEL] {ch}")
            fit_itr = 0
            for r_idx in range(idx_sizes[0]):
                len_itr = 0
                current_len = 0
                c = (0.0, 0.0, 0.0, 0.0)
                offsets = plane_offsets[fit_itr] if fit_itr < len(plane_offsets) else []
                
                for c_idx in range(idx_sizes[1]):
                    tile_status = b_mat[r_idx, c_idx]
                    occ_code = occ_mats[ch][r_idx, c_idx]
                    
                    if tile_status == 0:
                        if len_itr < current_len:
                            print(f"[ERROR]: should encode unfit nums right now!")
                            print(f"[INFO]: r_idx {r_idx} c_idx {c_idx} len_itr {len_itr} len {current_len}")
                            return
                        unfit_cnt += 1
                    else:
                        if len_itr < current_len:
                            Decoder.calc_fit_nums_with_offset(
                                imgs[ch], c, occ_code, c_idx, r_idx, len_itr, tile_size, offsets[ch]
                            )
                            len_itr += 1
                        else:
                            c = coefficients[fit_itr]
                            offsets = plane_offsets[fit_itr] if fit_itr < len(plane_offsets) else []
                            len_itr = 0
                            current_len = tile_fit_lengths[fit_itr]
                            Decoder.calc_fit_nums_with_offset(
                                imgs[ch], c, occ_code, c_idx, r_idx, len_itr, tile_size, offsets[ch]
                            )
                            fit_itr += 1
                            len_itr += 1
                        fit_cnt += 1
        
        print(f"Multi with fitting_cnts: {fit_cnt} with unfitting_cnts: {unfit_cnt}")

# 兼容性函数，保持与 C++ 相同的接口
def single_channel_decode(img: np.ndarray, b_mat: np.ndarray, idx_sizes: Tuple[int, int],
                         coefficients: List[Tuple[float, float, float, float]], 
                         occ_mat: np.ndarray, tile_fit_lengths: List[int],
                         unfit_nums: List[float], tile_size: int, 
                         multi_mat: np.ndarray = None) -> float:
    """
    单通道解码 (兼容 C++ 接口)
    
    Args:
        img: 重建的图像
        b_mat: 拟合标记矩阵
        idx_sizes: 索引尺寸
        coefficients: 平面系数列表
        occ_mat: 占据矩阵
        tile_fit_lengths: 瓦片拟合长度列表
        unfit_nums: 不适合拟合的数值列表
        tile_size: 瓦片大小
        multi_mat: 多通道矩阵
        
    Returns:
        float: 解码时间
    """
    return Decoder.single_channel_decode(
        img, b_mat, idx_sizes, coefficients, occ_mat, tile_fit_lengths,
        tile_size, None, -1, None, unfit_nums, multi_mat
    )

def single_channel_decode_with_dct(img: np.ndarray, b_mat: np.ndarray, idx_sizes: Tuple[int, int],
                                  coefficients: List[Tuple[float, float, float, float]], 
                                  occ_mat: np.ndarray, tile_fit_lengths: List[int],
                                  tile_size: int, dct_mat: Matrix) -> float:
    """
    带 DCT 的单通道解码 (兼容 C++ 接口)
    
    Args:
        img: 重建的图像
        b_mat: 拟合标记矩阵
        idx_sizes: 索引尺寸
        coefficients: 平面系数列表
        occ_mat: 占据矩阵
        tile_fit_lengths: 瓦片拟合长度列表
        tile_size: 瓦片大小
        dct_mat: DCT 系数矩阵
        
    Returns:
        float: 解码时间
    """
    return Decoder.single_channel_decode(
        img, b_mat, idx_sizes, coefficients, occ_mat, tile_fit_lengths,
        tile_size, dct_mat
    )

def single_channel_decode_with_ksample(img: np.ndarray, b_mat: np.ndarray, idx_sizes: Tuple[int, int],
                                      coefficients: List[Tuple[float, float, float, float]], 
                                      occ_mat: np.ndarray, tile_fit_lengths: List[int],
                                      tile_size: int, dct_mat: Matrix, 
                                      ksample: int, extra_pc: List[List[float]]) -> float:
    """
    带采样的单通道解码 (兼容 C++ 接口)
    
    Args:
        img: 重建的图像
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
        float: 解码时间
    """
    return Decoder.single_channel_decode(
        img, b_mat, idx_sizes, coefficients, occ_mat, tile_fit_lengths,
        tile_size, dct_mat, ksample, extra_pc
    )

def multi_channel_decode(imgs: List[np.ndarray], b_mat: np.ndarray,
                        idx_sizes: Tuple[int, int], occ_mats: List[np.ndarray],
                        coefficients: List[Tuple[float, float, float, float]],
                        plane_offsets: List[List[float]], tile_fit_lengths: List[int],
                        threshold: float, tile_size: int):
    """
    多通道解码 (兼容 C++ 接口)
    
    Args:
        imgs: 图像列表
        b_mat: 拟合标记矩阵
        idx_sizes: 索引尺寸
        occ_mats: 占据矩阵列表
        coefficients: 平面系数列表
        plane_offsets: 平面偏移列表
        tile_fit_lengths: 瓦片拟合长度列表
        threshold: 阈值
        tile_size: 瓦片大小
    """
    Decoder.multi_channel_decode(
        imgs, b_mat, idx_sizes, occ_mats, coefficients,
        plane_offsets, tile_fit_lengths, threshold, tile_size
    )

# 使用示例
if __name__ == "__main__":
    # 创建测试数据
    img_rows, img_cols = 64, 64
    test_img = np.zeros((img_rows, img_cols), dtype=np.float32)
    test_b_mat = np.random.randint(0, 2, (8, 8), dtype=np.int32)
    test_idx_sizes = (8, 8)
    test_coefficients = [(1.0, 0.1, 0.1, 10.0) for _ in range(5)]
    test_occ_mat = np.random.randint(0, 65536, (8, 8), dtype=np.int32)
    test_tile_fit_lengths = [2, 1, 3, 1, 1]
    test_tile_size = 8
    
    # 测试基本解码
    decode_time = single_channel_decode(
        test_img, test_b_mat, test_idx_sizes, test_coefficients,
        test_occ_mat, test_tile_fit_lengths, [], test_tile_size
    )
    print(f"Basic decoding time: {decode_time:.4f} seconds")
    
    # 测试带 DCT 的解码
    test_dct_mat = [[0.0] * img_cols for _ in range(img_rows)]
    decode_time_dct = single_channel_decode_with_dct(
        test_img, test_b_mat, test_idx_sizes, test_coefficients,
        test_occ_mat, test_tile_fit_lengths, test_tile_size, test_dct_mat
    )
    print(f"DCT decoding time: {decode_time_dct:.4f} seconds")
    
    # 测试带采样的解码
    test_extra_pc = []
    decode_time_ksample = single_channel_decode_with_ksample(
        test_img, test_b_mat, test_idx_sizes, test_coefficients,
        test_occ_mat, test_tile_fit_lengths, test_tile_size, test_dct_mat, 2, test_extra_pc
    )
    print(f"K-sample decoding time: {decode_time_ksample:.4f} seconds")
    print(f"Extra point cloud size: {len(test_extra_pc)}")