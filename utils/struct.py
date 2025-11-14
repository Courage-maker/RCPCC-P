from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any
import numpy as np

@dataclass
class PointCloud:
    """点云数据结构"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    r: float = 0.0
    color: Tuple[int, int, int] = (255, 255, 255)
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, r: float = 0.0):
        """初始化点云数据"""
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.r = float(r)
        self.color = (255, 255, 255)
    
    def to_vector_xyz(self) -> List[float]:
        """转换为 XYZ 向量"""
        return [self.x, self.y, self.z]
    
    def to_numpy_xyz(self) -> np.ndarray:
        """转换为 NumPy 数组"""
        return np.array([self.x, self.y, self.z])
    
    def to_string_xyz(self) -> str:
        """转换为 XYZ 字符串"""
        return f"[{self.x},{self.y},{self.z}]"
    
    def distance_to(self, other: 'PointCloud') -> float:
        """计算到另一个点的距离"""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    
    def magnitude(self) -> float:
        """计算向量的模长"""
        return np.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    
    def __repr__(self):
        return f"PointCloud(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, r={self.r:.3f})"
    
    def __eq__(self, other):
        if not isinstance(other, PointCloud):
            return False
        return (abs(self.x - other.x) < 1e-6 and 
                abs(self.y - other.y) < 1e-6 and 
                abs(self.z - other.z) < 1e-6)
    
    def __hash__(self):
        return hash((round(self.x, 6), round(self.y, 6), round(self.z, 6)))


@dataclass
class Range3D:
    """3D 范围结构"""
    x_min: float = 0.0
    y_min: float = 0.0
    z_min: float = 0.0
    x_max: float = 0.0
    y_max: float = 0.0
    z_max: float = 0.0
    
    def to_string(self) -> str:
        """转换为字符串表示"""
        return (f"[min]: {self.x_min},{self.y_min},{self.z_min}; "
                f"[max]: {self.x_max},{self.y_max},{self.z_max}")
    
    def width(self) -> float:
        """获取 X 方向宽度"""
        return self.x_max - self.x_min
    
    def height(self) -> float:
        """获取 Y 方向高度"""
        return self.y_max - self.y_min
    
    def depth(self) -> float:
        """获取 Z 方向深度"""
        return self.z_max - self.z_min
    
    def volume(self) -> float:
        """计算体积"""
        return self.width() * self.height() * self.depth()
    
    def center(self) -> Tuple[float, float, float]:
        """获取中心点坐标"""
        return ((self.x_min + self.x_max) / 2,
                (self.y_min + self.y_max) / 2,
                (self.z_min + self.z_max) / 2)
    
    def contains(self, point: PointCloud) -> bool:
        """检查点是否在范围内"""
        return (self.x_min <= point.x <= self.x_max and
                self.y_min <= point.y <= self.y_max and
                self.z_min <= point.z <= self.z_max)
    
    def expand_to_include(self, point: PointCloud) -> None:
        """扩展范围以包含指定点"""
        self.x_min = min(self.x_min, point.x)
        self.y_min = min(self.y_min, point.y)
        self.z_min = min(self.z_min, point.z)
        self.x_max = max(self.x_max, point.x)
        self.y_max = max(self.y_max, point.y)
        self.z_max = max(self.z_max, point.z)
    
    @classmethod
    def from_points(cls, points: List[PointCloud]) -> 'Range3D':
        """从点云列表创建范围"""
        if not points:
            return cls()
        
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        z_coords = [p.z for p in points]
        
        return cls(
            x_min=min(x_coords), y_min=min(y_coords), z_min=min(z_coords),
            x_max=max(x_coords), y_max=max(y_coords), z_max=max(z_coords)
        )
    
    def __repr__(self):
        return f"Range3D(x=[{self.x_min:.3f}, {self.x_max:.3f}], " \
               f"y=[{self.y_min:.3f}, {self.y_max:.3f}], " \
               f"z=[{self.z_min:.3f}, {self.z_max:.3f}])"


@dataclass
class PFrame:
    """点云帧结构"""
    range: Range3D = field(default_factory=Range3D)
    points: List[PointCloud] = field(default_factory=list)
    
    def __init__(self, range_obj: Optional[Range3D] = None, points: Optional[List[PointCloud]] = None):
        """初始化点云帧"""
        self.range = range_obj if range_obj is not None else Range3D()
        self.points = points if points is not None else []
        
        # 如果提供了点但没有范围，自动计算范围
        if points and (range_obj is None):
            self.range = Range3D.from_points(points)
    
    def add_point(self, point: PointCloud) -> None:
        """添加点到帧中"""
        self.points.append(point)
        self.range.expand_to_include(point)
    
    def point_count(self) -> int:
        """获取点的数量"""
        return len(self.points)
    
    def bounds(self) -> Range3D:
        """获取边界范围"""
        return self.range
    
    def filter_points(self, condition) -> 'PFrame':
        """根据条件过滤点"""
        filtered_points = [p for p in self.points if condition(p)]
        return PFrame(Range3D.from_points(filtered_points), filtered_points)
    
    def __repr__(self):
        return f"PFrame(points={len(self.points)}, range={self.range})"


@dataclass
class PccResult:
    """PCC 结果数据结构"""
    compression_rate: List[float] = field(default_factory=list)
    fitting_ratios: List[float] = field(default_factory=list)
    restored_compression_rate: List[float] = field(default_factory=list)
    stream_compression_rates: List[float] = field(default_factory=list)
    loss_rate: List[float] = field(default_factory=list)
    restored_loss_rate: List[float] = field(default_factory=list)
    match_pcts: List[float] = field(default_factory=list)
    unmatch_pcts: List[float] = field(default_factory=list)
    matchs: List[float] = field(default_factory=list)
    unmatchs: List[float] = field(default_factory=list)
    proj_times: List[float] = field(default_factory=list)
    fit_times: List[float] = field(default_factory=list)
    merge_times: List[float] = field(default_factory=list)
    
    def add_compression_result(self, 
                             compression_rate: float,
                             fitting_ratio: float,
                             restored_compression_rate: float,
                             stream_compression_rate: float,
                             loss_rate: float,
                             restored_loss_rate: float,
                             match_pct: float,
                             unmatch_pct: float,
                             match_val: float,
                             unmatch_val: float,
                             proj_time: float,
                             fit_time: float,
                             merge_time: float) -> None:
        """添加压缩结果"""
        self.compression_rate.append(compression_rate)
        self.fitting_ratios.append(fitting_ratio)
        self.restored_compression_rate.append(restored_compression_rate)
        self.stream_compression_rates.append(stream_compression_rate)
        self.loss_rate.append(loss_rate)
        self.restored_loss_rate.append(restored_loss_rate)
        self.match_pcts.append(match_pct)
        self.unmatch_pcts.append(unmatch_pct)
        self.matchs.append(match_val)
        self.unmatchs.append(unmatch_val)
        self.proj_times.append(proj_time)
        self.fit_times.append(fit_time)
        self.merge_times.append(merge_time)
    
    def get_averages(self) -> dict:
        """获取所有指标的平均值"""
        def safe_average(values: List[float]) -> float:
            return sum(values) / len(values) if values else 0.0
        
        return {
            'compression_rate': safe_average(self.compression_rate),
            'fitting_ratios': safe_average(self.fitting_ratios),
            'restored_compression_rate': safe_average(self.restored_compression_rate),
            'stream_compression_rates': safe_average(self.stream_compression_rates),
            'loss_rate': safe_average(self.loss_rate),
            'restored_loss_rate': safe_average(self.restored_loss_rate),
            'match_pcts': safe_average(self.match_pcts),
            'unmatch_pcts': safe_average(self.unmatch_pcts),
            'matchs': safe_average(self.matchs),
            'unmatchs': safe_average(self.unmatchs),
            'proj_times': safe_average(self.proj_times),
            'fit_times': safe_average(self.fit_times),
            'merge_times': safe_average(self.merge_times),
        }
    
    def clear(self) -> None:
        """清空所有结果"""
        self.compression_rate.clear()
        self.fitting_ratios.clear()
        self.restored_compression_rate.clear()
        self.stream_compression_rates.clear()
        self.loss_rate.clear()
        self.restored_loss_rate.clear()
        self.match_pcts.clear()
        self.unmatch_pcts.clear()
        self.matchs.clear()
        self.unmatchs.clear()
        self.proj_times.clear()
        self.fit_times.clear()
        self.merge_times.clear()
    
    def __repr__(self):
        avg = self.get_averages()
        return f"PccResult(frames={len(self.compression_rate)}, " \
               f"avg_compression={avg['compression_rate']:.3f}, " \
               f"avg_loss={avg['loss_rate']:.3f})"


# 实用函数
def points_to_numpy(points: List[PointCloud]) -> np.ndarray:
    """将点云列表转换为 NumPy 数组"""
    return np.array([[p.x, p.y, p.z] for p in points])


def numpy_to_points(array: np.ndarray) -> List[PointCloud]:
    """将 NumPy 数组转换为点云列表"""
    return [PointCloud(row[0], row[1], row[2]) for row in array]


def calculate_bounding_box(points: List[PointCloud]) -> Range3D:
    """计算点云的边界框"""
    return Range3D.from_points(points)


# 类型别名
PointCloudArray = np.ndarray  # shape: (n, 3) 或 (n, 4)
PointList = List[PointCloud]
FrameList = List[PFrame]