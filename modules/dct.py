import numpy as np
import cv2
from scipy.fftpack import dct, idct
from typing import Tuple, List, Union
import math

# 定义类型别名
Matrix = List[List[float]]
BoolMatrix = List[List[bool]]
SQRT_2 = math.sqrt(2)

def sadct(data: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对数据应用二维分离 DCT 变换
    
    Args:
        data: 输入数据 (CV_32FC4 类型)
        mask: 掩码矩阵 (CV_32SC1 类型)
    
    Returns:
        Tuple: (DCT 系数矩阵, 最终掩码)
    """
    assert data.dtype == np.float32 and len(data.shape) == 3 and data.shape[2] == 4
    assert mask.dtype == np.int32 and len(mask.shape) == 2
    
    rows, cols = data.shape[:2]
    
    temp_img = np.zeros((rows, cols), dtype=np.float64)
    mask_ = np.zeros((rows, cols), dtype=np.int32)
    
    # 对列进行 DCT
    for j in range(cols):
        col_indices = np.where(mask[:, j])[0]
        N = len(col_indices)
        
        if N > 0:
            col_data = np.array([data[i, j, 0] for i in col_indices], dtype=np.float64)
            
            # 使用 scipy 的 DCT
            dct_col = dct(col_data, type=2, norm='ortho')
            
            for idx, i in enumerate(col_indices):
                if idx < N:
                    temp_img[idx, j] = dct_col[idx]
                    mask_[idx, j] = 1
    
    temp_img_new = np.zeros((rows, cols), dtype=np.float64)
    mask_final = np.zeros((rows, cols), dtype=np.int32)
    
    # 对行进行 DCT
    for i in range(rows):
        row_indices = np.where(mask_[i, :])[0]
        N = len(row_indices)
        
        if N > 0:
            row_data = np.array([temp_img[i, j] for j in row_indices], dtype=np.float64)
            
            # 使用 scipy 的 DCT
            dct_row = dct(row_data, type=2, norm='ortho')
            
            for idx, j in enumerate(row_indices):
                if idx < N:
                    temp_img_new[i, idx] = dct_row[idx]
                    mask_final[i, idx] = 1
    
    return temp_img_new, mask_final

def saidct(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    对 DCT 系数应用二维分离逆 DCT 变换
    
    Args:
        data: DCT 系数 (CV_32FC4 类型)
        mask: 掩码矩阵 (CV_32SC1 类型)
    
    Returns:
        np.ndarray: 重构的数据
    """
    assert data.dtype == np.float32 and len(data.shape) == 3 and data.shape[2] == 4
    assert mask.dtype == np.int32 and len(mask.shape) == 2
    
    rows, cols = mask.shape
    
    # 步骤 1: 生成掩码和索引映射
    _mask1 = np.zeros((rows, cols), dtype=np.int32)
    _mask = np.zeros((rows, cols), dtype=np.int32)
    index_mask1 = np.full((rows, cols, 2), -1, dtype=np.int32)
    index_mask = np.full((rows, cols, 2), -1, dtype=np.int32)
    index_mask2 = np.full((rows, cols, 2), -1, dtype=np.int32)
    
    # 列掩码
    for j in range(cols):
        l = 0
        for i in range(rows):
            if mask[i, j]:
                _mask1[l, j] = 1
                index_mask1[l, j] = [i, j]
                l += 1
    
    # 行掩码
    for i in range(rows):
        l = 0
        for j in range(cols):
            if _mask1[i, j]:
                _mask[i, l] = 1
                index_mask[i, l] = index_mask1[i, j]
                index_mask2[i, l] = [i, j]
                l += 1
    
    # 步骤 2: 填充系数
    coeff = np.zeros((rows, cols), dtype=np.float64)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            coeff[i, j] = data[i, j, 0]
    
    # 步骤 3: 对每行进行 IDCT
    temp_coeff = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        valid_num = np.count_nonzero(_mask[i])
        if valid_num > 0:
            row_data = np.array([coeff[i, j] for j in range(valid_num)], dtype=np.float64)
            
            # 使用 scipy 的逆 DCT
            idct_row = idct(row_data, type=2, norm='ortho')
            
            for j in range(valid_num):
                temp_coeff[i, j] = idct_row[j]
    
    coeff = temp_coeff.copy()
    
    # 步骤 4: 映射回列位置
    temp_coeff.fill(0)
    for i in range(rows):
        for j in range(cols):
            if _mask[i, j]:
                idx_i, idx_j = index_mask2[i, j]
                temp_coeff[idx_i, idx_j] = coeff[i, j]
    
    coeff = temp_coeff.copy()
    
    # 步骤 5: 对每列进行 IDCT
    temp_coeff.fill(0)
    for j in range(cols):
        valid_num = np.count_nonzero(_mask1[:, j])
        if valid_num > 0:
            col_data = np.array([coeff[i, j] for i in range(valid_num)], dtype=np.float64)
            
            # 使用 scipy 的逆 DCT
            idct_col = idct(col_data, type=2, norm='ortho')
            
            for i in range(valid_num):
                temp_coeff[i, j] = idct_col[i]
    
    coeff = temp_coeff.copy()
    
    # 步骤 6: 映射回原始位置
    real_result = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            if _mask1[i, j]:
                orig_i, orig_j = index_mask1[i, j]
                real_result[orig_i, orig_j] = coeff[i, j]
    
    return real_result

def matrix_sadct(data: Matrix, mask: BoolMatrix) -> Tuple[Matrix, BoolMatrix]:
    """
    基于列表的二维分离 DCT 变换
    
    Args:
        data: 输入数据矩阵
        mask: 布尔掩码矩阵
    
    Returns:
        Tuple: (DCT 系数矩阵, 最终掩码)
    """
    rows = len(data)
    cols = len(data[0])
    
    temp_img = [[0.0] * cols for _ in range(rows)]
    mask_ = [[False] * cols for _ in range(rows)]
    
    # 对列进行 DCT
    for j in range(cols):
        col_data = []
        indices = []
        for i in range(rows):
            if mask[i][j]:
                col_data.append(data[i][j])
                indices.append(i)
        
        N = len(col_data)
        if N > 0:
            dct_col = dct(np.array(col_data), type=2, norm='ortho')
            
            for idx, i in enumerate(indices):
                if idx < N:
                    temp_img[idx][j] = dct_col[idx] / (SQRT_2 * math.sqrt(N))
                    mask_[idx][j] = True
    
    temp_img_new = [[0.0] * cols for _ in range(rows)]
    mask_final = [[False] * cols for _ in range(rows)]
    
    # 对行进行 DCT
    for i in range(rows):
        row_data = []
        indices = []
        for j in range(cols):
            if mask_[i][j]:
                row_data.append(temp_img[i][j])
                indices.append(j)
        
        N = len(row_data)
        if N > 0:
            dct_row = dct(np.array(row_data), type=2, norm='ortho')
            
            for idx, j in enumerate(indices):
                if idx < N:
                    temp_img_new[i][idx] = dct_row[idx] / (SQRT_2 * math.sqrt(N))
                    mask_final[i][idx] = True
    
    return temp_img_new, mask_final

def matrix_saidct(data: Matrix, mask: BoolMatrix) -> Matrix:
    """
    基于列表的二维分离逆 DCT 变换
    
    Args:
        data: DCT 系数矩阵
        mask: 布尔掩码矩阵
    
    Returns:
        Matrix: 重构的数据矩阵
    """
    rows = len(mask)
    cols = len(mask[0])
    
    coeff = [[0.0] * cols for _ in range(rows)]
    _mask = [[False] * cols for _ in range(rows)]
    _mask1 = [[False] * cols for _ in range(rows)]
    
    # 使用字典存储索引映射
    index_mask1 = [[(-1, -1) for _ in range(cols)] for _ in range(rows)]
    index_mask = [[(-1, -1) for _ in range(cols)] for _ in range(rows)]
    index_mask2 = [[(-1, -1) for _ in range(cols)] for _ in range(rows)]
    
    # 步骤 1: 生成掩码和索引映射
    for j in range(cols):
        l = 0
        for i in range(rows):
            if mask[i][j]:
                _mask1[l][j] = True
                index_mask1[l][j] = (i, j)
                l += 1
    
    for i in range(rows):
        l = 0
        for j in range(cols):
            if _mask1[i][j]:
                _mask[i][l] = True
                index_mask[i][l] = index_mask1[i][j]
                index_mask2[i][l] = (i, j)
                l += 1
    
    # 步骤 2: 填充系数
    for i in range(len(data)):
        for j in range(len(data[0])):
            coeff[i][j] = data[i][j]
    
    # 步骤 3: 对每行进行 IDCT
    temp_coeff = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        valid_num = sum(1 for x in _mask[i] if x)
        if valid_num > 0:
            row_data = [coeff[i][j] for j in range(valid_num)]
            idct_row = idct(np.array(row_data), type=2, norm='ortho')
            
            for j in range(valid_num):
                temp_coeff[i][j] = idct_row[j] / (SQRT_2 * math.sqrt(valid_num))
    
    coeff = [row[:] for row in temp_coeff]
    
    # 步骤 4: 映射回列位置
    temp_coeff = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if _mask[i][j]:
                orig_i, orig_j = index_mask2[i][j]
                temp_coeff[orig_i][orig_j] = coeff[i][j]
    
    coeff = [row[:] for row in temp_coeff]
    
    # 步骤 5: 对每列进行 IDCT
    temp_coeff = [[0.0] * cols for _ in range(rows)]
    for j in range(cols):
        valid_num = sum(1 for i in range(rows) if _mask1[i][j])
        if valid_num > 0:
            col_data = [coeff[i][j] for i in range(valid_num)]
            idct_col = idct(np.array(col_data), type=2, norm='ortho')
            
            for i in range(valid_num):
                temp_coeff[i][j] = idct_col[i] / (SQRT_2 * math.sqrt(valid_num))
    
    coeff = [row[:] for row in temp_coeff]
    
    # 步骤 6: 映射回原始位置
    real_result = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if _mask1[i][j]:
                orig_i, orig_j = index_mask1[i][j]
                real_result[orig_i][orig_j] = coeff[i][j]
    
    return real_result

def delta_sadct(data: Matrix, mask: BoolMatrix) -> Tuple[Matrix, BoolMatrix]:
    """
    带均值减去的 DCT 变换
    
    Args:
        data: 输入数据矩阵
        mask: 布尔掩码矩阵
    
    Returns:
        Tuple: (DCT 系数矩阵, 最终掩码)
    """
    temp_img = [row[:] for row in data]
    
    # 计算均值
    total_sum = 0.0
    count = 0
    for i in range(len(data)):
        for j in range(len(data[0])):
            if mask[i][j]:
                total_sum += data[i][j]
                count += 1
    
    mean_val = total_sum / count if count > 0 else 0.0
    
    # 减去均值
    for i in range(len(data)):
        for j in range(len(data[0])):
            if mask[i][j]:
                temp_img[i][j] -= mean_val
    
    scale = 10000.0
    dct_result, final_mask = matrix_sadct(temp_img, mask)
    
    # 将均值存储在 DC 系数中
    dct_result[0][0] = mean_val * scale
    
    return dct_result, final_mask

def delta_saidct(data: Matrix, mask: BoolMatrix) -> Matrix:
    """
    带均值恢复的逆 DCT 变换
    
    Args:
        data: DCT 系数矩阵
        mask: 布尔掩码矩阵
    
    Returns:
        Matrix: 重构的数据矩阵
    """
    rows = len(mask)
    cols = len(mask[0])
    scale = 10000.0
    
    # 从 DC 系数中恢复均值
    mean_val = data[0][0] / scale
    
    # 将 DC 系数置零
    modified_data = [row[:] for row in data]
    modified_data[0][0] = 0.0
    
    # 进行逆变换
    result = matrix_saidct(modified_data, mask)
    
    # 加上均值
    for i in range(rows):
        for j in range(cols):
            if mask[i][j]:
                result[i][j] += mean_val
    
    return result

# 使用示例
if __name__ == "__main__":
    # 创建测试数据
    rows, cols = 8, 8
    test_data = np.random.rand(rows, cols, 4).astype(np.float32)
    test_mask = np.random.randint(0, 2, (rows, cols), dtype=np.int32)
    
    # 测试 OpenCV 版本的 DCT
    dct_coeff, final_mask = sadct(test_data, test_mask)
    reconstructed = saidct(test_data, test_mask)
    
    print(f"Original shape: {test_data.shape}")
    print(f"DCT coefficients shape: {dct_coeff.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # 测试基于列表的版本
    list_data = [[float(i + j) for j in range(cols)] for i in range(rows)]
    list_mask = [[(i + j) % 2 == 0 for j in range(cols)] for i in range(rows)]
    
    dct_result, mask_result = matrix_sadct(list_data, list_mask)
    reconstructed_list = matrix_saidct(dct_result, list_mask)
    
    print(f"List DCT completed successfully")