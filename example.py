#!/usr/bin/env python3
"""
点云压缩示例 - Python 版本
演示点云压缩、解压缩和可视化功能
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

# 导入项目模块
from utils.struct import PointCloud
from utils.utils import (
    load_pcloud, 
    load_pcloud_ply, 
    load_pcloud_xyz,
    export_pcloud,
    output_cloud
)
from utils.config import (
    DEFAULT_PITCH_PRECISION, 
    DEFAULT_YAW_PRECISION,
    USE_LIVOX,
    print_config_summary
)

# 尝试导入可视化库
try:
    import open3d as o3d
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("警告: Open3D 不可用，可视化功能将被禁用")
    VISUALIZATION_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("警告: Matplotlib 不可用，备用可视化功能将被禁用")
    MATPLOTLIB_AVAILABLE = False


class EncoderModule:
    """编码器模块模拟类"""
    
    # 5个压缩等级配置：数值越大，压缩比越大
    COMPRESSION_LEVELS = {
        1: {"compression_factor": 0.9, "noise_level": 0.002, "description": "高质量"},     # 保留90%数据
        2: {"compression_factor": 0.7, "noise_level": 0.005, "description": "平衡质量"},   # 保留70%数据
        3: {"compression_factor": 0.5, "noise_level": 0.01, "description": "标准压缩"},    # 保留50%数据
        4: {"compression_factor": 0.3, "noise_level": 0.02, "description": "高压缩"},     # 保留30%数据
        5: {"compression_factor": 0.1, "noise_level": 0.05, "description": "极限压缩"}     # 保留10%数据
    }
    
    def __init__(self, tile_size: int = 4, compression_level: int = 3):
        self.tile_size = tile_size
        self.compression_level = compression_level
        self.compression_ratio = 0.0
        
        if compression_level not in self.COMPRESSION_LEVELS:
            raise ValueError(f"压缩等级必须在 {list(self.COMPRESSION_LEVELS.keys())} 之间")
        
    def encode_to_data(self, pcloud_data: List[PointCloud], verbose: bool = True) -> bytes:
        """将点云数据编码为字节数据"""
        if verbose:
            print(f"编码点云数据: {len(pcloud_data)} 个点")
            print(f"瓦片大小: {self.tile_size}, 压缩等级: {self.compression_level}")
            
        level_config = self.COMPRESSION_LEVELS[self.compression_level]
        if verbose:
            print(f"压缩模式: {level_config['description']}")
            print(f"压缩因子: {level_config['compression_factor']}")
        
        # 模拟编码过程
        start_time = time.time()
        
        # 使用简单的序列化作为示例
        encoded_data = self._simple_serialize(pcloud_data)
        
        # 根据压缩等级决定保留的数据量
        compression_factor = level_config["compression_factor"]
        target_size = int(len(encoded_data) * compression_factor)
        
        # 截断数据来模拟压缩
        encoded_data = encoded_data[:target_size]
        
        encode_time = time.time() - start_time
        
        if verbose:
            print(f"编码完成: {len(encoded_data)} 字节, 耗时: {encode_time:.3f} 秒")
        
        return encoded_data
    
    def _simple_serialize(self, pcloud_data: List[PointCloud]) -> bytes:
        """简单的序列化实现"""
        import struct
        
        # 将点云数据序列化为二进制格式
        buffer = bytearray()
        
        # 写入点数量
        buffer.extend(struct.pack('I', len(pcloud_data)))
        
        # 写入每个点的坐标
        for point in pcloud_data:
            buffer.extend(struct.pack('fff', point.x, point.y, point.z))
        
        return bytes(buffer)


class DecoderModule:
    """解码器模块模拟类"""
    
    def __init__(self, encoded_data: bytes, tile_size: int = 4, 
                 verbose: bool = True, compression_level: int = 3):
        self.encoded_data = encoded_data
        self.tile_size = tile_size
        self.compression_level = compression_level
        self.restored_pcloud: List[PointCloud] = []
        
        if verbose:
            print(f"解码数据: {len(encoded_data)} 字节")
            print(f"瓦片大小: {tile_size}, 压缩等级: {compression_level}")
            
            level_config = EncoderModule.COMPRESSION_LEVELS[compression_level]
            print(f"压缩模式: {level_config['description']}")
        
        self._decode(verbose)
    
    def _decode(self, verbose: bool = True):
        """解码数据"""
        start_time = time.time()
        
        # 模拟解码过程
        self.restored_pcloud = self._simple_deserialize(self.encoded_data)
        
        # 根据压缩等级添加噪声模拟质量损失
        level_config = EncoderModule.COMPRESSION_LEVELS[self.compression_level]
        noise_level = level_config["noise_level"]
        
        if noise_level > 0:
            # 添加随机噪声来模拟质量损失
            import random
            for point in self.restored_pcloud:
                point.x += random.gauss(0, noise_level)
                point.y += random.gauss(0, noise_level)
                point.z += random.gauss(0, noise_level)
        
        decode_time = time.time() - start_time
        
        if verbose:
            print(f"解码完成: {len(self.restored_pcloud)} 个点, 耗时: {decode_time:.3f} 秒")
            print(f"噪声水平: {noise_level}")
    
    def _simple_deserialize(self, data: bytes) -> List[PointCloud]:
        """简单的反序列化实现"""
        import struct
        
        points = []
        
        try:
            # 读取点数量
            num_points = struct.unpack('I', data[:4])[0]
            
            # 计算预期的数据长度
            expected_length = 4 + num_points * 12  # 4字节头 + 每个点3个float(12字节)
            
            if len(data) < expected_length:
                # 数据不完整，尝试恢复尽可能多的点
                num_points = min(num_points, (len(data) - 4) // 12)
            
            # 读取每个点的坐标
            for i in range(num_points):
                offset = 4 + i * 12
                x, y, z = struct.unpack('fff', data[offset:offset+12])
                points.append(PointCloud(x, y, z))
        
        except Exception as e:
            print(f"反序列化错误: {e}")
            # 返回空列表或部分数据
        
        return points


def visualize_point_clouds_open3d_simultaneous(original_cloud: List[PointCloud], 
                                              restored_cloud: List[PointCloud],
                                              point_size: float = 2.0) -> None:
    """
    使用 Open3D 同时显示两个点云窗口
    
    Args:
        original_cloud: 原始点云数据
        restored_cloud: 重建点云数据
        point_size: 点的大小
    """
    if not VISUALIZATION_AVAILABLE:
        print("Open3D 不可用，无法进行可视化")
        return
    
    # 转换为 Open3D 点云对象
    original_pcd = o3d.geometry.PointCloud()
    restored_pcd = o3d.geometry.PointCloud()
    
    # 设置点云数据
    original_pcd.points = o3d.utility.Vector3dVector(
        [[p.x, p.y, p.z] for p in original_cloud]
    )
    restored_pcd.points = o3d.utility.Vector3dVector(
        [[p.x, p.y, p.z] for p in restored_cloud]
    )
    
    # 设置颜色
    original_pcd.paint_uniform_color([1, 0, 0])  # 红色 - 原始点云
    restored_pcd.paint_uniform_color([0, 1, 0])  # 绿色 - 重建点云
    
    print("同时打开两个 Open3D 可视化窗口...")
    print("说明:")
    print("  - 红色点: 原始点云")
    print("  - 绿色点: 重建点云")
    print("  - 点大小:", point_size)
    print("  - 按 'Q' 或关闭窗口退出")
    
    # 创建可视化器
    vis1 = o3d.visualization.Visualizer()
    vis2 = o3d.visualization.Visualizer()
    
    # 设置窗口
    vis1.create_window(window_name="原始点云", width=800, height=600, left=50, top=50)
    vis2.create_window(window_name="重建点云", width=800, height=600, left=900, top=50)
    
    # 添加几何体
    vis1.add_geometry(original_pcd)
    vis2.add_geometry(restored_pcd)
    
    # 设置渲染选项 - 点大小
    render_option1 = vis1.get_render_option()
    render_option1.point_size = point_size
    render_option2 = vis2.get_render_option()
    render_option2.point_size = point_size
    
    # 设置背景颜色为黑色以便更好显示
    render_option1.background_color = np.array([0, 0, 0])
    render_option2.background_color = np.array([0, 0, 0])
    
    # 同时运行两个可视化窗口
    try:
        while True:
            # 更新两个窗口
            vis1.poll_events()
            vis1.update_renderer()
            
            vis2.poll_events()
            vis2.update_renderer()
            
            # 检查窗口是否关闭
            if not vis1.poll_events() or not vis2.poll_events():
                break
                
            time.sleep(0.01)  # 短暂休眠以减少CPU使用
                
    except KeyboardInterrupt:
        print("可视化被用户中断")
    finally:
        # 关闭可视化器
        vis1.destroy_window()
        vis2.destroy_window()


def visualize_point_clouds_matplotlib_simultaneous(original_cloud: List[PointCloud],
                                                  restored_cloud: List[PointCloud],
                                                  point_size: float = 2.0) -> None:
    """
    使用 Matplotlib 同时显示两个点云窗口
    
    Args:
        original_cloud: 原始点云数据
        restored_cloud: 重建点云数据
        point_size: 点的大小
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib 不可用，无法进行可视化")
        return
    
    # 创建两个独立的图形窗口
    fig1 = plt.figure("原始点云", figsize=(10, 8))
    fig2 = plt.figure("重建点云", figsize=(10, 8))
    
    # 原始点云窗口
    ax1 = fig1.add_subplot(111, projection='3d')
    original_points = np.array([[p.x, p.y, p.z] for p in original_cloud])
    ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], 
               c='red', s=point_size, alpha=0.8, label='原始点云')
    ax1.set_title('原始点云')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # 重建点云窗口
    ax2 = fig2.add_subplot(111, projection='3d')
    restored_points = np.array([[p.x, p.y, p.z] for p in restored_cloud])
    ax2.scatter(restored_points[:, 0], restored_points[:, 1], restored_points[:, 2], 
               c='green', s=point_size, alpha=0.8, label='重建点云')
    ax2.set_title('重建点云')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    # 调整窗口位置
    fig1.canvas.manager.window.wm_geometry("+50+50")  # 第一个窗口位置
    fig2.canvas.manager.window.wm_geometry("+900+50")  # 第二个窗口位置
    
    plt.show(block=True)


def visualize_point_clouds_combined(original_cloud: List[PointCloud],
                                   restored_cloud: List[PointCloud],
                                   point_size: float = 2.0) -> None:
    """
    在同一个窗口中显示对比视图
    
    Args:
        original_cloud: 原始点云数据
        restored_cloud: 重建点云数据
        point_size: 点的大小
    """
    if not VISUALIZATION_AVAILABLE:
        print("Open3D 不可用，无法进行可视化")
        return
    
    # 转换为 Open3D 点云对象
    original_pcd = o3d.geometry.PointCloud()
    restored_pcd = o3d.geometry.PointCloud()
    
    # 设置点云数据
    original_pcd.points = o3d.utility.Vector3dVector(
        [[p.x, p.y, p.z] for p in original_cloud]
    )
    restored_pcd.points = o3d.utility.Vector3dVector(
        [[p.x, p.y, p.z] for p in restored_cloud]
    )
    
    # 设置颜色
    original_pcd.paint_uniform_color([1, 0, 0])  # 红色 - 原始点云
    restored_pcd.paint_uniform_color([0, 1, 0])  # 绿色 - 重建点云
    
    # 将重建点云向右偏移以便对比
    translation = np.identity(4)
    translation[0, 3] = 2.0  # 在x方向偏移2个单位
    restored_pcd.transform(translation)
    
    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="点云对比 (左:原始红色, 右:重建绿色)", 
                     width=1200, height=600)
    
    # 添加几何体
    vis.add_geometry(original_pcd)
    vis.add_geometry(restored_pcd)
    
    # 设置渲染选项 - 点大小
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.array([0, 0, 0])
    
    # 设置相机位置以便更好地查看两个点云
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_lookat([1, 0, 0])
    ctr.set_zoom(0.8)
    
    print("打开对比视图窗口...")
    print("说明:")
    print("  - 左侧红色点: 原始点云")
    print("  - 右侧绿色点: 重建点云")
    print("  - 点大小:", point_size)
    print("  - 按 'Q' 或关闭窗口退出")
    
    # 运行可视化
    vis.run()
    vis.destroy_window()


def calculate_compression_ratio(original_size: int, compressed_size: int) -> float:
    """计算压缩率"""
    if compressed_size == 0:
        return 0.0
    return original_size / compressed_size


def get_file_size(filepath: str) -> int:
    """获取文件大小"""
    return os.path.getsize(filepath)


def print_compression_levels():
    """打印压缩等级说明"""
    print("压缩等级说明 (数值越大，压缩比越大):")
    print("-" * 50)
    for level, config in EncoderModule.COMPRESSION_LEVELS.items():
        retention = config["compression_factor"] * 100
        print(f"等级 {level}: {config['description']} - 保留约 {retention:.0f}% 数据")
    print("-" * 50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='点云压缩示例')
    parser.add_argument('input_file', help='输入点云文件路径 (.bin, .ply, .xyz)')
    parser.add_argument('compression_level', type=int, help='压缩等级 (1-5)，数值越大压缩比越大', 
                       choices=[1, 2, 3, 4, 5], metavar='[1-5]')
    parser.add_argument('--no-visualization', action='store_true', 
                       help='禁用可视化')
    parser.add_argument('--output', '-o', default='decompressed.ply',
                       help='输出文件名 (默认: decompressed.ply)')
    parser.add_argument('--use-livox', action='store_true',
                       help='使用 Livox LiDAR 参数')
    parser.add_argument('--visualization-method', choices=['open3d', 'matplotlib', 'combined'], 
                       default='open3d', help='选择可视化方法')
    parser.add_argument('--point-size', type=float, default=2.0,
                       help='点的大小 (默认: 2.0)')
    parser.add_argument('--list-levels', action='store_true',
                       help='显示压缩等级说明并退出')
    
    args = parser.parse_args()
    
    # 如果请求显示压缩等级说明
    if args.list_levels:
        print_compression_levels()
        return 0
    
    # 验证输入文件
    if not os.path.isfile(args.input_file):
        print(f"错误: 文件不存在: {args.input_file}")
        return 1
    
    # 打印配置信息
    print_config_summary(args.use_livox)
    print_compression_levels()
    print(f"选择的压缩等级: {args.compression_level}")
    print(f"输入文件: {args.input_file}")
    print(f"可视化方法: {args.visualization_method}")
    print(f"点大小: {args.point_size}")
    print("-" * 50)
    
    # 加载点云数据
    pcloud_data: List[PointCloud] = []
    filename = args.input_file
    file_extension = Path(filename).suffix.lower()
    
    try:
        if file_extension == '.bin':
            pcloud_data = load_pcloud(filename)
        elif file_extension == '.ply':
            pcloud_data = load_pcloud_ply(filename)
        elif file_extension == '.xyz':
            pcloud_data = load_pcloud_xyz(filename)
        else:
            print(f"错误: 不支持的文件格式: {file_extension}")
            print("支持格式: .bin, .ply, .xyz")
            return 1
        
        if not pcloud_data:
            print("错误: 未能加载任何点云数据")
            return 1
            
        print(f"成功加载点云: {len(pcloud_data)} 个点")
        
    except Exception as e:
        print(f"加载点云时出错: {e}")
        return 1
    
    # 获取原始文件大小
    try:
        original_size = get_file_size(filename)
        print(f"原始文件大小: {original_size} 字节")
    except Exception as e:
        print(f"获取文件大小时出错: {e}")
        original_size = 0
    
    # 编码点云
    try:
        print("开始编码...")
        encoder = EncoderModule(tile_size=4, compression_level=args.compression_level)
        encoded_data = encoder.encode_to_data(pcloud_data, verbose=True)
        
        # 计算压缩率
        compressed_size = len(encoded_data)
        compression_ratio = calculate_compression_ratio(original_size, compressed_size)
        
        print("\n压缩结果:")
        print(f"  原始大小: {original_size} 字节")
        print(f"  压缩大小: {compressed_size} 字节")
        print(f"  压缩比率: {compression_ratio:.2f}:1")
        
    except Exception as e:
        print(f"编码过程中出错: {e}")
        return 1
    
    # 解码点云
    try:
        print("\n开始解码...")
        decoder = DecoderModule(
            encoded_data, 
            tile_size=4, 
            verbose=True, 
            compression_level=args.compression_level
        )
        restored_pcloud = decoder.restored_pcloud
        
        print(f"重建点云: {len(restored_pcloud)} 个点")
        
        # 计算点保留率
        retention_rate = len(restored_pcloud) / len(pcloud_data) * 100
        print(f"点保留率: {retention_rate:.1f}%")
        
    except Exception as e:
        print(f"解码过程中出错: {e}")
        return 1
    
    # 保存重建的点云
    try:
        output_cloud(restored_pcloud, args.output)
        output_size = get_file_size(args.output)
        print(f"保存重建点云到: {args.output} ({output_size} 字节)")
        
    except Exception as e:
        print(f"保存点云时出错: {e}")
        return 1
    
    # 可视化
    if not args.no_visualization:
        print("\n开始可视化...")
        
        if args.visualization_method == 'open3d' and VISUALIZATION_AVAILABLE:
            visualize_point_clouds_open3d_simultaneous(
                pcloud_data, restored_pcloud, point_size=args.point_size
            )
        elif args.visualization_method == 'matplotlib' and MATPLOTLIB_AVAILABLE:
            visualize_point_clouds_matplotlib_simultaneous(
                pcloud_data, restored_pcloud, point_size=args.point_size
            )
        elif args.visualization_method == 'combined' and VISUALIZATION_AVAILABLE:
            visualize_point_clouds_combined(
                pcloud_data, restored_pcloud, point_size=args.point_size
            )
        else:
            print(f"请求的可视化方法 '{args.visualization_method}' 不可用")
            # 尝试使用可用的方法
            if VISUALIZATION_AVAILABLE:
                print("使用 Open3D 同时显示作为替代")
                visualize_point_clouds_open3d_simultaneous(
                    pcloud_data, restored_pcloud, point_size=args.point_size
                )
            elif MATPLOTLIB_AVAILABLE:
                print("使用 Matplotlib 同时显示作为替代")
                visualize_point_clouds_matplotlib_simultaneous(
                    pcloud_data, restored_pcloud, point_size=args.point_size
                )
            else:
                print("没有可用的可视化库，跳过可视化")
    
    print("\n处理完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())