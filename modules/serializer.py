import numpy as np
import cv2
import struct
from typing import List, Tuple, Dict, Any, Union
import io
import math

# 量化字典
quantization_dict = [
    [0.25, 0.5, 0.1, 0.1],    # 0x
    [0.25, 0.5, 0.2, 0.20],   # 0x
    [0.25, 0.5, 0.4, 0.20],   # good
    [0.5, 1.0, 0.1, 0.2],     # good
    [0.5, 1.0, 0.2, 0.2],     # good
    [1.0, 2.0, 0.4, 0.20],    # 1x
]

# 定义类型别名
Matrix = List[List[float]]

# Zigzag 编码
def zigzag_encode(n: int) -> int:
    """Zigzag 编码，将有符号整数转换为无符号整数"""
    return (n << 1) ^ (n >> 31)

# Zigzag 解码
def zigzag_decode(n: int) -> int:
    """Zigzag 解码，将无符号整数转换为有符号整数"""
    return (n >> 1) ^ (-(n & 1))

def get_dct_mask(occ_mat: np.ndarray, tile_size: int, b_mat: np.ndarray) -> np.ndarray:
    """
    获取 DCT 掩码矩阵
    
    Args:
        occ_mat: 占据矩阵
        tile_size: 瓦片大小
        b_mat: 拟合标记矩阵
        
    Returns:
        np.ndarray: DCT 掩码矩阵
    """
    rows, cols = occ_mat.shape
    tt2 = tile_size * tile_size
    raw_occ_mat = np.zeros((rows * tile_size, cols * tile_size), dtype=np.int32)
    
    # 重建原始占据矩阵
    for r_idx in range(rows):
        for c_idx in range(cols):
            if b_mat[r_idx, c_idx] == 0:  # 非平面拟合区域
                occ_code = occ_mat[r_idx, c_idx]
                for i in range(tt2):
                    if ((occ_code >> i) & 1) == 1:
                        raw_i = r_idx * tile_size + i // tile_size
                        raw_j = c_idx * tile_size + i % tile_size
                        raw_occ_mat[raw_i, raw_j] = 1
    
    # 列压缩
    dct_mask = np.zeros_like(raw_occ_mat, dtype=np.int32)
    for j in range(raw_occ_mat.shape[1]):
        l = 0
        for i in range(raw_occ_mat.shape[0]):
            if raw_occ_mat[i, j] == 1:
                dct_mask[l, j] = 1
                l += 1
    
    # 行压缩
    dct_mask_final = np.zeros_like(dct_mask, dtype=np.int32)
    for i in range(dct_mask.shape[0]):
        l = 0
        for j in range(dct_mask.shape[1]):
            if dct_mask[i, j] == 1:
                dct_mask_final[i, l] = 1
                l += 1
    
    return dct_mask_final

def serialize_bmat_to_stream(mat: np.ndarray, stream: io.BytesIO):
    """
    序列化二值矩阵到流
    
    Args:
        mat: 二值矩阵
        stream: 字节流
    """
    cnt = 0
    code = 0
    for row in range(mat.shape[0]):
        for col in range(mat.shape[1]):
            if cnt == 8:
                stream.write(struct.pack('B', code))
                cnt = 0
                code = 0
            
            status = 1 if mat[row, col] > 0 else 0
            code += (status << cnt)
            cnt += 1
    
    if cnt > 0:
        stream.write(struct.pack('B', code))

def deserialize_bmat_from_stream(stream: io.BytesIO, rows: int, cols: int) -> np.ndarray:
    """
    从流中反序列化二值矩阵
    
    Args:
        stream: 字节流
        rows: 行数
        cols: 列数
        
    Returns:
        np.ndarray: 二值矩阵
    """
    mat = np.zeros((rows, cols), dtype=np.int32)
    cnt = 0
    code_bytes = stream.read(1)
    if not code_bytes:
        return mat
    
    code = struct.unpack('B', code_bytes)[0]
    
    for row in range(rows):
        for col in range(cols):
            if cnt == 8:
                code_bytes = stream.read(1)
                if not code_bytes:
                    return mat
                code = struct.unpack('B', code_bytes)[0]
                cnt = 0
            
            if (code >> cnt) & 1 == 1:
                mat[row, col] = 1
            else:
                mat[row, col] = 0
            
            cnt += 1
    
    return mat

def serialize_occmat_to_stream(occ_mat: np.ndarray, stream: io.BytesIO):
    """
    序列化占据矩阵到流
    
    Args:
        occ_mat: 占据矩阵
        stream: 字节流
    """
    for row in range(occ_mat.shape[0]):
        for col in range(occ_mat.shape[1]):
            code = np.uint16(occ_mat[row, col])
            stream.write(struct.pack('H', code))

def deserialize_occmat_from_stream(stream: io.BytesIO, rows: int, cols: int) -> np.ndarray:
    """
    从流中反序列化占据矩阵
    
    Args:
        stream: 字节流
        rows: 行数
        cols: 列数
        
    Returns:
        np.ndarray: 占据矩阵
    """
    occ_mat = np.zeros((rows, cols), dtype=np.int32)
    for row in range(rows):
        for col in range(cols):
            code_bytes = stream.read(2)
            if not code_bytes:
                return occ_mat
            code = struct.unpack('H', code_bytes)[0]
            occ_mat[row, col] = code
    
    return occ_mat

def serialize_mat_to_stream(mat: np.ndarray, stream: io.BytesIO):
    """
    序列化矩阵到流
    
    Args:
        mat: NumPy 矩阵
        stream: 字节流
    """
    # 写入矩阵信息
    dtype_code = mat.dtype.str
    shape = mat.shape
    
    # 将 dtype 转换为字符串表示
    stream.write(struct.pack('i', len(dtype_code)))
    stream.write(dtype_code.encode('utf-8'))
    
    # 写入形状
    stream.write(struct.pack('i', len(shape)))
    for dim in shape:
        stream.write(struct.pack('i', dim))
    
    # 写入数据
    stream.write(mat.tobytes())

def deserialize_mat_from_stream(stream: io.BytesIO) -> np.ndarray:
    """
    从流中反序列化矩阵
    
    Args:
        stream: 字节流
        
    Returns:
        np.ndarray: NumPy 矩阵
    """
    # 读取 dtype
    dtype_len = struct.unpack('i', stream.read(4))[0]
    dtype_str = stream.read(dtype_len).decode('utf-8')
    
    # 读取形状
    shape_len = struct.unpack('i', stream.read(4))[0]
    shape = []
    for _ in range(shape_len):
        shape.append(struct.unpack('i', stream.read(4))[0])
    
    # 读取数据
    total_elements = np.prod(shape)
    dtype = np.dtype(dtype_str)
    data_size = total_elements * dtype.itemsize
    data_bytes = stream.read(data_size)
    
    # 重建矩阵
    mat = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
    return mat

def serialize_vec4f_vector_to_stream(vec: List, stream: io.BytesIO):
    """
    序列化 Vec4f 向量到流
    
    Args:
        vec: Vec4f 向量
        stream: 字节流
    """
    size = len(vec)
    stream.write(struct.pack('Q', size))
    for v in vec:
        # 假设 v 是包含 4 个浮点数的列表或元组
        stream.write(struct.pack('ffff', v[0], v[1], v[2], v[3]))

def deserialize_vec4f_vector_from_stream(stream: io.BytesIO) -> List:
    """
    从流中反序列化 Vec4f 向量
    
    Args:
        stream: 字节流
        
    Returns:
        List: Vec4f 向量
    """
    size_bytes = stream.read(8)
    if not size_bytes:
        return []
    
    size = struct.unpack('Q', size_bytes)[0]
    vec = []
    for _ in range(size):
        data_bytes = stream.read(16)  # 4 * float32
        if len(data_bytes) < 16:
            break
        v = struct.unpack('ffff', data_bytes)
        vec.append(v)
    
    return vec

def serialize_matrix_to_stream(mat: Matrix, stream: io.BytesIO):
    """
    序列化矩阵到流
    
    Args:
        mat: 二维矩阵
        stream: 字节流
    """
    rows = len(mat)
    stream.write(struct.pack('Q', rows))
    for row in mat:
        cols = len(row)
        stream.write(struct.pack('Q', cols))
        # 将每行数据打包为二进制
        row_data = struct.pack(f'{cols}d', *row)
        stream.write(row_data)

def deserialize_matrix_from_stream(stream: io.BytesIO) -> Matrix:
    """
    从流中反序列化矩阵
    
    Args:
        stream: 字节流
        
    Returns:
        Matrix: 二维矩阵
    """
    rows_bytes = stream.read(8)
    if not rows_bytes:
        return []
    
    rows = struct.unpack('Q', rows_bytes)[0]
    mat = []
    for _ in range(rows):
        cols_bytes = stream.read(8)
        if not cols_bytes:
            break
        cols = struct.unpack('Q', cols_bytes)[0]
        
        row_data_bytes = stream.read(cols * 8)  # double 是 8 字节
        if len(row_data_bytes) < cols * 8:
            break
        
        row = list(struct.unpack(f'{cols}d', row_data_bytes))
        mat.append(row)
    
    return mat

def serialize_int_vector_to_stream(vec: List[int], stream: io.BytesIO):
    """
    序列化整数向量到流
    
    Args:
        vec: 整数向量
        stream: 字节流
    """
    size = len(vec)
    stream.write(struct.pack('Q', size))
    for data in vec:
        quantized_d = np.uint16(data)
        stream.write(struct.pack('H', quantized_d))

def deserialize_int_vector_from_stream(stream: io.BytesIO) -> List[int]:
    """
    从流中反序列化整数向量
    
    Args:
        stream: 字节流
        
    Returns:
        List[int]: 整数向量
    """
    size_bytes = stream.read(8)
    if not size_bytes:
        return []
    
    size = struct.unpack('Q', size_bytes)[0]
    vec = []
    for _ in range(size):
        data_bytes = stream.read(2)
        if not data_bytes:
            break
        quantized_d = struct.unpack('H', data_bytes)[0]
        vec.append(int(quantized_d))
    
    return vec

def serialize_data(b_mat: np.ndarray, idx_sizes: Tuple[int, int],
                  coefficients: List, occ_mat: np.ndarray,
                  tile_fit_lengths: List[int], dct_mat: Matrix) -> bytes:
    """
    序列化所有数据
    
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
    stream = io.BytesIO()
    
    # 包头校验
    stream.write(b'N')
    
    # 序列化各个组件
    serialize_mat_to_stream(b_mat, stream)
    
    for size in idx_sizes:
        stream.write(struct.pack('i', size))
    
    serialize_vec4f_vector_to_stream(coefficients, stream)
    serialize_mat_to_stream(occ_mat, stream)
    serialize_int_vector_to_stream(tile_fit_lengths, stream)
    serialize_matrix_to_stream(dct_mat, stream)
    
    return stream.getvalue()

def deserialize_data(data: bytes) -> Tuple[bool, np.ndarray, Tuple[int, int], List, np.ndarray, List[int], Matrix]:
    """
    反序列化所有数据
    
    Args:
        data: 序列化数据
        
    Returns:
        Tuple: (成功标志, b_mat, idx_sizes, coefficients, occ_mat, tile_fit_lengths, dct_mat)
    """
    stream = io.BytesIO(data)
    
    # 检查包头
    header = stream.read(1)
    if header != b'N':
        return False, None, (0, 0), [], None, [], []
    
    # 反序列化各个组件
    b_mat = deserialize_mat_from_stream(stream)
    
    idx_sizes = (
        struct.unpack('i', stream.read(4))[0],
        struct.unpack('i', stream.read(4))[0]
    )
    
    coefficients = deserialize_vec4f_vector_from_stream(stream)
    occ_mat = deserialize_mat_from_stream(stream)
    tile_fit_lengths = deserialize_int_vector_from_stream(stream)
    dct_mat = deserialize_matrix_from_stream(stream)
    
    return True, b_mat, idx_sizes, coefficients, occ_mat, tile_fit_lengths, dct_mat

def serialize_dct_to_stream(mat: Matrix, stream: io.BytesIO, quantization_step: float, 
                           occ_mat: np.ndarray, b_mat: np.ndarray, use_zigzag: bool = False):
    """
    序列化 DCT 系数到流
    
    Args:
        mat: DCT 系数矩阵
        stream: 字节流
        quantization_step: 量化步长
        occ_mat: 占据矩阵
        b_mat: 拟合标记矩阵
        use_zigzag: 是否使用 Zigzag 编码
    """
    rows = len(mat)
    cols = len(mat[0]) if rows > 0 else 0
    
    stream.write(struct.pack('Q', rows))
    stream.write(struct.pack('Q', cols))
    
    # 获取有效 DCT 系数的索引
    indices = []
    dct_mask = get_dct_mask(occ_mat, 4, b_mat)
    
    for i in range(dct_mask.shape[0]):
        for j in range(dct_mask.shape[1]):
            if dct_mask[i, j] == 1 and i < rows and j < cols:
                indices.append(i * cols + j)
    
    # 序列化有效系数
    for idx in indices:
        row = idx // cols
        col = idx % cols
        
        if not use_zigzag:
            quantized_val = int(mat[row][col] / quantization_step)
            stream.write(struct.pack('i', quantized_val))
        else:
            quantized_val = zigzag_encode(int(mat[row][col] / quantization_step))
            stream.write(struct.pack('I', quantized_val))  # 无符号32位整数

def deserialize_dct_from_stream(stream: io.BytesIO, quantization_step: float,
                               occ_mat: np.ndarray, b_mat: np.ndarray, 
                               use_zigzag: bool = False) -> Matrix:
    """
    从流中反序列化 DCT 系数
    
    Args:
        stream: 字节流
        quantization_step: 量化步长
        occ_mat: 占据矩阵
        b_mat: 拟合标记矩阵
        use_zigzag: 是否使用 Zigzag 编码
        
    Returns:
        Matrix: DCT 系数矩阵
    """
    rows_bytes = stream.read(8)
    cols_bytes = stream.read(8)
    if not rows_bytes or not cols_bytes:
        return []
    
    rows = struct.unpack('Q', rows_bytes)[0]
    cols = struct.unpack('Q', cols_bytes)[0]
    
    mat = [[0.0] * cols for _ in range(rows)]
    
    # 获取有效 DCT 系数的索引
    indices = []
    dct_mask = get_dct_mask(occ_mat, 4, b_mat)
    
    for i in range(dct_mask.shape[0]):
        for j in range(dct_mask.shape[1]):
            if dct_mask[i, j] == 1 and i < rows and j < cols:
                indices.append(i * cols + j)
    
    # 反序列化有效系数
    for idx in indices:
        row = idx // cols
        col = idx % cols
        
        if not use_zigzag:
            quantized_val_bytes = stream.read(4)
            if not quantized_val_bytes:
                break
            quantized_val = struct.unpack('i', quantized_val_bytes)[0]
            mat[row][col] = quantized_val * quantization_step
        else:
            quantized_val_bytes = stream.read(4)
            if not quantized_val_bytes:
                break
            quantized_val = struct.unpack('I', quantized_val_bytes)[0]
            mat[row][col] = zigzag_decode(quantized_val) * quantization_step
    
    return mat

def q_serialize_data(q_level: int, b_mat: np.ndarray, idx_sizes: Tuple[int, int],
                    coefficients: List, occ_mat: np.ndarray,
                    tile_fit_lengths: List[int], dct_mat: Matrix) -> bytes:
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
    stream = io.BytesIO()
    
    # 添加包头
    stream.write(b'Q')
    
    # 写入量化参数
    parameters = quantization_dict[q_level][:4]
    for param in parameters:
        stream.write(struct.pack('f', param))
    
    # 序列化 b_mat
    bmat_row, bmat_col = b_mat.shape
    stream.write(struct.pack('H', bmat_row))
    stream.write(struct.pack('H', bmat_col))
    serialize_bmat_to_stream(b_mat, stream)
    
    # 序列化其他组件
    for size in idx_sizes:
        stream.write(struct.pack('i', size))
    
    serialize_vec4f_vector_to_stream(coefficients, stream)
    serialize_occmat_to_stream(occ_mat, stream)
    serialize_int_vector_to_stream(tile_fit_lengths, stream)
    
    # 序列化 DCT 系数（使用量化）
    quantization_step = quantization_dict[q_level][3]
    serialize_dct_to_stream(dct_mat, stream, quantization_step, occ_mat, b_mat)
    
    return stream.getvalue()

def q_deserialize_data(data: bytes) -> Tuple[bool, float, float, np.ndarray, Tuple[int, int], List, np.ndarray, List[int], Matrix]:
    """
    量化反序列化数据
    
    Args:
        data: 序列化数据
        
    Returns:
        Tuple: (成功标志, pitch_precision, yaw_precision, b_mat, idx_sizes, coefficients, occ_mat, tile_fit_lengths, dct_mat)
    """
    stream = io.BytesIO(data)
    
    # 检查包头
    header = stream.read(1)
    if header != b'Q':
        return False, 0.0, 0.0, None, (0, 0), [], None, [], []
    
    # 读取量化参数
    parameters = []
    for _ in range(4):
        param_bytes = stream.read(4)
        if not param_bytes:
            return False, 0.0, 0.0, None, (0, 0), [], None, [], []
        parameters.append(struct.unpack('f', param_bytes)[0])
    
    pitch_precision, yaw_precision, _, quantization_step = parameters
    
    # 读取 b_mat 尺寸
    bmat_row_bytes = stream.read(2)
    bmat_col_bytes = stream.read(2)
    if not bmat_row_bytes or not bmat_col_bytes:
        return False, pitch_precision, yaw_precision, None, (0, 0), [], None, [], []
    
    bmat_row = struct.unpack('H', bmat_row_bytes)[0]
    bmat_col = struct.unpack('H', bmat_col_bytes)[0]
    
    # 反序列化各个组件
    b_mat = deserialize_bmat_from_stream(stream, bmat_row, bmat_col)
    
    idx_sizes = (
        struct.unpack('i', stream.read(4))[0],
        struct.unpack('i', stream.read(4))[0]
    )
    
    coefficients = deserialize_vec4f_vector_from_stream(stream)
    occ_mat = deserialize_occmat_from_stream(stream, bmat_row, bmat_col)
    tile_fit_lengths = deserialize_int_vector_from_stream(stream)
    dct_mat = deserialize_dct_from_stream(stream, quantization_step, occ_mat, b_mat)
    
    return True, pitch_precision, yaw_precision, b_mat, idx_sizes, coefficients, occ_mat, tile_fit_lengths, dct_mat

# 使用示例
if __name__ == "__main__":
    # 创建测试数据
    test_b_mat = np.random.randint(0, 2, (8, 8), dtype=np.int32)
    test_idx_sizes = (16, 16)
    test_coefficients = [(1.0, 2.0, 3.0, 4.0) for _ in range(10)]
    test_occ_mat = np.random.randint(0, 65536, (8, 8), dtype=np.int32)
    test_tile_fit_lengths = [5, 3, 7, 2]
    test_dct_mat = [[float(i + j) for j in range(16)] for i in range(16)]
    
    # 测试序列化
    serialized_data = serialize_data(
        test_b_mat, test_idx_sizes, test_coefficients,
        test_occ_mat, test_tile_fit_lengths, test_dct_mat
    )
    print(f"Serialized data size: {len(serialized_data)} bytes")
    
    # 测试反序列化
    success, b_mat, idx_sizes, coefficients, occ_mat, tile_fit_lengths, dct_mat = deserialize_data(serialized_data)
    if success:
        print("Deserialization successful")
        print(f"b_mat shape: {b_mat.shape}")
        print(f"coefficients count: {len(coefficients)}")
    else:
        print("Deserialization failed")
    
    # 测试量化序列化
    q_serialized_data = q_serialize_data(
        0, test_b_mat, test_idx_sizes, test_coefficients,
        test_occ_mat, test_tile_fit_lengths, test_dct_mat
    )
    print(f"Quantized serialized data size: {len(q_serialized_data)} bytes")
    
    # 测试量化反序列化
    q_success, pitch_prec, yaw_prec, q_b_mat, q_idx_sizes, q_coefficients, q_occ_mat, q_tile_lengths, q_dct_mat = q_deserialize_data(q_serialized_data)
    if q_success:
        print("Quantized deserialization successful")
        print(f"Precision: {pitch_prec}, {yaw_prec}")
    else:
        print("Quantized deserialization failed")