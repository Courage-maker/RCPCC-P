import struct
import numpy as np
from typing import List, Tuple, Any
import os

def export_b_mat(b_mat: np.ndarray, filename: str) -> None:
    """导出二进制矩阵到文件"""
    with open(filename, 'wb') as out_stream:
        cnt = 0
        code = 0
        for row in range(b_mat.shape[0]):
            for col in range(b_mat.shape[1]):
                if cnt == 8:
                    out_stream.write(struct.pack('B', code))
                    cnt = 0
                    code = 0
                
                status = b_mat[row, col]
                code += (status << cnt)
                cnt += 1
        
        if cnt > 0:
            out_stream.write(struct.pack('B', code))

def import_b_mat(rows: int, cols: int, filename: str) -> np.ndarray:
    """从文件导入二进制矩阵"""
    b_mat = np.zeros((rows, cols), dtype=np.int32)
    
    with open(filename, 'rb') as in_stream:
        cnt = 0
        code_data = in_stream.read(1)
        if not code_data:
            return b_mat
            
        code = struct.unpack('B', code_data)[0]
        
        for row in range(rows):
            for col in range(cols):
                if cnt == 8:
                    code_data = in_stream.read(1)
                    if not code_data:
                        return b_mat
                    code = struct.unpack('B', code_data)[0]
                    cnt = 0
                
                if (code >> cnt) & 1 == 1:
                    b_mat[row, col] = 1
                else:
                    b_mat[row, col] = 0
                cnt += 1
    
    return b_mat

def export_coefficients(coefficients: List[Tuple[float, float, float, float]], filename: str) -> None:
    """导出系数到文件"""
    with open(filename, 'wb') as out_stream:
        for c in coefficients:
            for i in range(4):
                out_stream.write(struct.pack('f', c[i]))

def import_coefficients(filename: str) -> List[Tuple[float, float, float, float]]:
    """从文件导入系数"""
    coefficients = []
    
    with open(filename, 'rb') as in_stream:
        while True:
            data = in_stream.read(4 * 4)  # 4个float，每个4字节
            if len(data) < 16:
                break
                
            c = struct.unpack('4f', data)
            coefficients.append(c)
    
    return coefficients

def export_occ_mat(occ_mat: np.ndarray, filename: str) -> None:
    """导出占用矩阵到文件"""
    with open(filename, 'wb') as out_stream:
        for row in range(occ_mat.shape[0]):
            for col in range(occ_mat.shape[1]):
                code = np.uint16(occ_mat[row, col])
                out_stream.write(struct.pack('H', code))

def import_occ_mat(rows: int, cols: int, filename: str) -> np.ndarray:
    """从文件导入占用矩阵"""
    occ_mat = np.zeros((rows, cols), dtype=np.int32)
    
    with open(filename, 'rb') as in_stream:
        for row in range(rows):
            for col in range(cols):
                data = in_stream.read(2)
                if not data:
                    return occ_mat
                    
                code = struct.unpack('H', data)[0]
                occ_mat[row, col] = code
    
    return occ_mat

def export_unfit_nums(data: List[float], filename: str) -> None:
    """导出不适合数字到文件"""
    with open(filename, 'wb') as out_stream:
        for d in data:
            quantized_d = np.uint16(d * 256)
            out_stream.write(struct.pack('H', quantized_d))

def import_unfit_nums(filename: str) -> List[float]:
    """从文件导入不适合数字"""
    data = []
    
    with open(filename, 'rb') as in_stream:
        while True:
            byte_data = in_stream.read(2)
            if len(byte_data) < 2:
                break
                
            quantized_d = struct.unpack('H', byte_data)[0]
            d = float(quantized_d) / 256.0
            data.append(d)
    
    return data

def export_tile_fit_lengths(data: List[int], filename: str) -> None:
    """导出瓦片拟合长度到文件"""
    with open(filename, 'wb') as out_stream:
        for d in data:
            quantized_d = np.uint16(d)
            out_stream.write(struct.pack('H', quantized_d))

def import_tile_fit_lengths(filename: str) -> List[int]:
    """从文件导入瓦片拟合长度"""
    data = []
    
    with open(filename, 'rb') as in_stream:
        while True:
            byte_data = in_stream.read(2)
            if len(byte_data) < 2:
                break
                
            quantized_d = struct.unpack('H', byte_data)[0]
            data.append(int(quantized_d))
    
    return data

def export_plane_offsets(data: List[List[float]], filename: str) -> None:
    """导出平面偏移到文件"""
    with open(filename, 'wb') as out_stream:
        for vec in data:
            for d in vec:
                out_stream.write(struct.pack('f', d))

def import_plane_offsets(filename: str, size: int) -> List[List[float]]:
    """从文件导入平面偏移"""
    data = []
    vec = []
    
    with open(filename, 'rb') as in_stream:
        while True:
            byte_data = in_stream.read(4)
            if len(byte_data) < 4:
                break
                
            d = struct.unpack('f', byte_data)[0]
            vec.append(d)
            
            if len(vec) == size:
                data.append(vec.copy())
                vec.clear()
    
    return data

def export_filenames(data: List[str], filename: str) -> None:
    """导出文件名列表到文件"""
    with open(filename, 'w', encoding='utf-8') as out_stream:
        for s in data:
            out_stream.write(s + '\n')

def import_filenames(filename: str) -> List[str]:
    """从文件导入文件名列表"""
    data = []
    
    with open(filename, 'r', encoding='utf-8') as in_stream:
        for line in in_stream:
            data.append(line.strip())
    
    return data

def count_file_bytes(filenames: List[str]) -> int:
    """计算文件总字节数"""
    total_size = 0
    for filename in filenames:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"{filename} size: {file_size} bytes")
            total_size += file_size
    return total_size

# 模板函数的Python版本
def export_vectors(data: List[Any], filename: str, dtype: str) -> None:
    """导出向量数据到文件"""
    with open(filename, 'wb') as out_stream:
        for d in data:
            if dtype == 'f':
                out_stream.write(struct.pack('f', d))
            elif dtype == 'i':
                out_stream.write(struct.pack('i', d))
            elif dtype == 'H':
                out_stream.write(struct.pack('H', d))
            # 可以根据需要添加更多数据类型

def import_vectors(filename: str, dtype: str) -> List[Any]:
    """从文件导入向量数据"""
    data = []
    
    format_char = dtype
    bytes_per_element = struct.calcsize(format_char)
    
    with open(filename, 'rb') as in_stream:
        while True:
            byte_data = in_stream.read(bytes_per_element)
            if len(byte_data) < bytes_per_element:
                break
                
            d = struct.unpack(format_char, byte_data)[0]
            data.append(d)
    
    return data