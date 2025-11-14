"""
点云压缩配置参数
"""

# 数学常量
SQRT_2 = 1.4142135623730950488
PI = 3.14159266

# 算法开关
USE_SADCT = True      # 使用 SADCT 变换
USE_ZIGZAG = True     # 使用 Zigzag 扫描
VERBOSE = True        # 详细输出模式
# PERFORMANCE = True  # 性能模式（注释掉表示禁用）
FITTING = True        # 启用拟合

# 标准 LiDAR 参数
ROW_OFFSET = 32.0
COL_OFFSET = 180.0
VERTICAL_DEGREE = 32.0 + 5.0    # 37.0 度
HORIZONTAL_DEGREE = 180.0 + 180.0  # 360.0 度

# Livox LiDAR 参数（默认禁用）
USE_LIVOX = False
LIVOX_ROW_OFFSET = 38.6
LIVOX_COL_OFFSET = 36.0
LIVOX_VERTICAL_DEGREE = 77.0
LIVOX_HORIZONTAL_DEGREE = 72.0

# 计算得到的参数（运行时计算）
def get_vertical_resolution(use_livox: bool = False) -> float:
    """获取垂直方向分辨率"""
    if use_livox:
        return LIVOX_VERTICAL_DEGREE
    return VERTICAL_DEGREE

def get_horizontal_resolution(use_livox: bool = False) -> float:
    """获取水平方向分辨率"""
    if use_livox:
        return LIVOX_HORIZONTAL_DEGREE
    return HORIZONTAL_DEGREE

def get_row_offset(use_livox: bool = False) -> float:
    """获取行偏移量"""
    if use_livox:
        return LIVOX_ROW_OFFSET
    return ROW_OFFSET

def get_col_offset(use_livox: bool = False) -> float:
    """获取列偏移量"""
    if use_livox:
        return LIVOX_COL_OFFSET
    return COL_OFFSET

# 图像尺寸配置
def calculate_image_dimensions(pitch_precision: float, yaw_precision: float, 
                             use_livox: bool = False) -> tuple:
    """
    根据精度参数计算图像尺寸
    
    Args:
        pitch_precision: 俯仰角精度
        yaw_precision: 偏航角精度
        use_livox: 是否使用 Livox 参数
        
    Returns:
        (rows, cols) 图像尺寸
    """
    if use_livox:
        vertical_degree = LIVOX_VERTICAL_DEGREE
        horizontal_degree = LIVOX_HORIZONTAL_DEGREE
    else:
        vertical_degree = VERTICAL_DEGREE
        horizontal_degree = HORIZONTAL_DEGREE
    
    rows = int(vertical_degree / yaw_precision)
    cols = int(horizontal_degree / pitch_precision)
    
    return rows, cols

# 压缩参数预设
class CompressionPreset:
    """压缩参数预设类"""
    
    HIGH_QUALITY = {
        'pitch_precision': 0.1,
        'yaw_precision': 0.1,
        'quantization_bits': 16,
        'use_dct': True,
        'quality_factor': 95
    }
    
    BALANCED = {
        'pitch_precision': 0.2,
        'yaw_precision': 0.2,
        'quantization_bits': 12,
        'use_dct': True,
        'quality_factor': 85
    }
    
    HIGH_COMPRESSION = {
        'pitch_precision': 0.3,
        'yaw_precision': 0.3,
        'quantization_bits': 8,
        'use_dct': True,
        'quality_factor': 75
    }

# 性能监控配置
class PerformanceConfig:
    """性能监控配置"""
    ENABLE_TIMING = True
    LOG_MEMORY_USAGE = True
    PROFILE_COMPRESSION = True
    SAVE_STATISTICS = True

# 输出配置
class OutputConfig:
    """输出配置"""
    SAVE_INTERMEDIATE_RESULTS = False
    GENERATE_VISUALIZATION = True
    COMPRESSION_STATS_FORMAT = "json"  # "json", "csv", "txt"

# 错误处理配置
class ErrorConfig:
    """错误处理配置"""
    MAX_POINT_DISTANCE = 300.0  # 最大点距离
    MIN_POINT_DISTANCE = 1.0    # 最小点距离
    IGNORE_INVALID_POINTS = True
    LOG_ERROR_DETAILS = True

# 默认参数
DEFAULT_PITCH_PRECISION = 0.1
DEFAULT_YAW_PRECISION = 0.1
DEFAULT_USE_LIVOX = USE_LIVOX

# 配置验证函数
def validate_config() -> bool:
    """
    验证配置参数的合理性
    
    Returns:
        配置是否有效
    """
    issues = []
    
    if PI <= 0:
        issues.append("PI 必须为正数")
    
    if ROW_OFFSET < 0 or COL_OFFSET < 0:
        issues.append("偏移量不能为负数")
    
    if VERTICAL_DEGREE <= 0 or HORIZONTAL_DEGREE <= 0:
        issues.append("视角度数必须为正数")
    
    if USE_LIVOX:
        if LIVOX_VERTICAL_DEGREE <= 0 or LIVOX_HORIZONTAL_DEGREE <= 0:
            issues.append("Livox 视角度数必须为正数")
    
    if issues:
        print("配置验证发现问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    return True

# 配置信息函数
def print_config_summary(use_livox: bool = None):
    """
    打印配置摘要
    
    Args:
        use_livox: 是否使用 Livox 配置，None 表示使用全局设置
    """
    if use_livox is None:
        use_livox = USE_LIVOX
    
    print("=== 点云压缩配置 ===")
    print(f"LiDAR 类型: {'Livox' if use_livox else 'Standard'}")
    print(f"垂直视角: {get_vertical_resolution(use_livox):.1f}°")
    print(f"水平视角: {get_horizontal_resolution(use_livox):.1f}°")
    print(f"行偏移: {get_row_offset(use_livox):.1f}")
    print(f"列偏移: {get_col_offset(use_livox):.1f}")
    print(f"详细输出: {'启用' if VERBOSE else '禁用'}")
    print(f"拟合功能: {'启用' if FITTING else '禁用'}")
    print(f"SADCT: {'启用' if USE_SADCT else '禁用'}")
    print(f"Zigzag: {'启用' if USE_ZIGZAG else '禁用'}")

# 初始化时验证配置
if __name__ == "__main__":
    if validate_config():
        print("配置验证通过")
        print_config_summary()
    else:
        print("配置验证失败，请检查参数")