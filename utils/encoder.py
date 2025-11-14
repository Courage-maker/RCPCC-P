import numpy as np
import cv2
import time
from typing import List, Tuple, Dict, Any, Union
import math
from collections import defaultdict

class Encoder:
    """
    编码器类，负责将范围图像编码为压缩格式
    """
    
    @staticmethod
    def filter_vals(img: np.ndarray, c_idx: int, r_idx: int, 
                   length: int, height: int) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        过滤出非零值点
        
        Args:
            img: 输入图像
            c_idx: 列起始索引
            r_idx: 行起始索引
            length: 长度
            height: 高度
            
        Returns:
            Tuple: (索引值列表, 范围值列表)
        """
        idx_vals = []
        range_vals = []
        
        for r in range(r_idx, min(height + r_idx, img.shape[0])):
            for c in range(c_idx, min(length + c_idx, img.shape[1])):
                vec = img[r, c]
                if vec[0] > 0.0:  # 假设第一个通道是范围值
                    idx_vals.append((float(r - r_idx), float(c - c_idx)))
                    range_vals.append(vec[0])
        
        return idx_vals, range_vals
    
    @staticmethod
    def plane_fitting(idx_vals: List[Tuple[float, float]], 
                     range_vals: List[float]) -> Tuple[float, float, float, float]:
        """
        3D 点云平面拟合
        
        Args:
            idx_vals: 索引值列表
            range_vals: 范围值列表
            
        Returns:
            Tuple: 平面系数 (a, b, c, d)
            
        Raises:
            ValueError: 当点数不足或拟合失败时
        """
        if len(idx_vals) < 3:
            raise ValueError("Not enough points for plane fitting")
        
        # 计算质心
        sum_x, sum_y, sum_z = 0.0, 0.0, 0.0
        for i in range(len(idx_vals)):
            x = range_vals[i]  # 范围值作为 x 坐标
            y = idx_vals[i][0]  # 行偏移作为 y 坐标
            z = idx_vals[i][1]  # 列偏移作为 z 坐标
            sum_x += x
            sum_y += y
            sum_z += z
        
        centroid_x = sum_x / len(idx_vals)
        centroid_y = sum_y / len(idx_vals)
        centroid_z = sum_z / len(idx_vals)
        
        # 计算协方差矩阵的元素
        xx, xy, xz = 0.0, 0.0, 0.0
        yy, yz, zz = 0.0, 0.0, 0.0
        
        for i in range(len(idx_vals)):
            r_x = range_vals[i] - centroid_x
            r_y = idx_vals[i][0] - centroid_y
            r_z = idx_vals[i][1] - centroid_z
            
            xx += r_x * r_x
            xy += r_x * r_y
            xz += r_x * r_z
            yy += r_y * r_y
            yz += r_y * r_z
            zz += r_z * r_z
        
        # 归一化
        n = len(idx_vals)
        xx /= n
        xy /= n
        xz /= n
        yy /= n
        yz /= n
        zz /= n
        
        # 计算法向量
        dir_x, dir_y, dir_z = 0.0, 0.0, 0.0
        
        # X 轴方向
        det_x = yy * zz - yz * yz
        x_axis_dir_x = det_x
        x_axis_dir_y = xz * yz - xy * zz
        x_axis_dir_z = xy * yz - xz * yy
        x_weight = det_x * det_x
        
        if (dir_x * x_axis_dir_x + dir_y * x_axis_dir_y + dir_z * x_axis_dir_z) < 0:
            x_weight = -x_weight
        
        dir_x += x_axis_dir_x * x_weight
        dir_y += x_axis_dir_y * x_weight
        dir_z += x_axis_dir_z * x_weight
        
        # Y 轴方向
        det_y = xx * zz - xz * xz
        y_axis_dir_x = xz * yz - xy * zz
        y_axis_dir_y = det_y
        y_axis_dir_z = xy * xz - yz * xx
        y_weight = det_y * det_y
        
        if (dir_x * y_axis_dir_x + dir_y * y_axis_dir_y + dir_z * y_axis_dir_z) < 0:
            y_weight = -y_weight
        
        dir_x += y_axis_dir_x * y_weight
        dir_y += y_axis_dir_y * y_weight
        dir_z += y_axis_dir_z * y_weight
        
        # Z 轴方向
        det_z = xx * yy - xy * xy
        z_axis_dir_x = xy * yz - xz * yy
        z_axis_dir_y = xy * xz - yz * xx
        z_axis_dir_z = det_z
        z_weight = det_z * det_z
        
        if (dir_x * z_axis_dir_x + dir_y * z_axis_dir_y + dir_z * z_axis_dir_z) < 0:
            z_weight = -z_weight
        
        dir_x += z_axis_dir_x * z_weight
        dir_y += z_axis_dir_y * z_weight
        dir_z += z_axis_dir_z * z_weight
        
        # 归一化法向量
        norm = math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
        if norm == 0.0:
            raise ValueError("Zero norm in plane fitting")
        
        dir_x /= norm
        dir_y /= norm
        dir_z /= norm
        
        # 计算平面方程系数: a*x + b*y + c*z + d = 0
        a = dir_x
        b = dir_y
        c = dir_z
        d = -(dir_x * centroid_x + dir_y * centroid_y + dir_z * centroid_z)
        
        if c == 0.0 or d == 0.0:
            raise ValueError("Invalid plane coefficients")
        
        return (a, b, c, d)
    
    @staticmethod
    def check_mat(img: np.ndarray, coefficients: Tuple[float, float, float, float],
                 threshold: float, c_idx: int, r_idx: int, 
                 length: int, height: int) -> bool:
        """
        检查平面拟合是否满足阈值要求
        
        Args:
            img: 输入图像
            coefficients: 平面系数
            threshold: 阈值
            c_idx: 列起始索引
            r_idx: 行起始索引
            length: 长度
            height: 高度
            
        Returns:
            bool: 是否满足阈值要求
        """
        a, b, c, d = coefficients
        if a == 0.0 and b == 0.0 and c == 0.0:
            return False
        
        for j in range(height):
            for i in range(length):
                vec = img[r_idx + j, c_idx + i]
                actual_value = vec[0]  # 假设第一个通道是范围值
                if actual_value > 0:
                    # 计算拟合值
                    val = abs(-d / (a * vec[0] + b * j + c * i)) * vec[0]
                    diff = abs(val - actual_value)
                    if diff > threshold:
                        return False
        
        return True
    
    @staticmethod
    def delta_coding(mat: np.ndarray, idx_sizes: Tuple[int, int, int], 
                    tile_size: int) -> int:
        """
        Delta 编码
        
        Args:
            mat: 输入矩阵
            idx_sizes: 索引尺寸
            tile_size: 瓦片大小
            
        Returns:
            int: 总计数
        """
        cnt = 0
        total_cnt = 0
        
        for i in range(0, idx_sizes[2], tile_size):
            for j in range(0, idx_sizes[1], tile_size):
                for k in range(idx_sizes[0]):
                    check = False
                    for ii in range(tile_size):
                        for jj in range(tile_size):
                            val = mat[k, j + jj, i + ii]
                            if val[0] > 0.0:  # 假设第一个通道是范围值
                                check = True
                                total_cnt += 1
                    
                    if check:
                        cnt += 1
        
        print(f"[DELTA] The unfitted tiles: {cnt}")
        print(f"Delta coding: {total_cnt / (tile_size * tile_size)}")
        return total_cnt
    
    @staticmethod
    def characterize_occupation(occ_m: np.ndarray, tile_size: int, 
                              mat_sizes: Tuple[int, int], pcloud_size: float) -> float:
        """
        表征占据矩阵（已弃用）
        
        Args:
            occ_m: 占据矩阵
            tile_size: 瓦片大小
            mat_sizes: 矩阵尺寸
            pcloud_size: 点云大小
            
        Returns:
            float: 比特率
        """
        m = defaultdict(int)
        
        for i in range(0, occ_m.shape[0] // tile_size):
            for j in range(0, occ_m.shape[1] // tile_size):
                val = 0
                for ii in range(tile_size):
                    for jj in range(tile_size):
                        val = val * 2
                        if occ_m[i * tile_size + ii, j * tile_size + jj][0] > 0.0:
                            val += 1
                
                m[val] += 1
        
        vec = list(m.values())
        vec.sort(reverse=True)
        
        bits = 0
        for i, count in enumerate(vec):
            if i < 8:
                bits += (2 + 3) * count
            elif i < 40:
                bits += (2 + 5) * count
            elif i < 168:
                bits += (2 + 7) * count
            else:
                bits += count * 18
        
        print(f"additional bits for occupation map({len(vec)}): {bits}")
        print(f": {pcloud_size} = {bits / pcloud_size}")
        
        return bits / pcloud_size
    
    @staticmethod
    def normalized_img(img: np.ndarray, tile_size: int) -> np.ndarray:
        """
        归一化图像
        
        Args:
            img: 输入图像
            tile_size: 瓦片大小
            
        Returns:
            np.ndarray: 归一化后的图像
        """
        # 找到最大范围值
        max_r = 0.0
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                max_r = max(max_r, img[r, c][0])
        
        # 创建归一化图像
        norm_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        for r in range(0, img.shape[0], tile_size):
            for c in range(0, img.shape[1], tile_size):
                sum_val = 0.0
                for rr in range(tile_size):
                    for cc in range(tile_size):
                        if r + rr < img.shape[0] and c + cc < img.shape[1]:
                            val = int(img[r, c][0] / max_r * 255)
                            norm_img[r + rr, c + cc] = val % 256
                            sum_val += img[r + rr, c + cc][0]
                
                if sum_val < 1.0:
                    for rr in range(tile_size):
                        for cc in range(tile_size):
                            if r + rr < img.shape[0] and c + cc < img.shape[1]:
                                norm_img[r + rr, c + cc] = 255
        
        return norm_img
    
    @staticmethod
    def export_mats(imgs: List[np.ndarray], tile_size: int):
        """
        导出矩阵为图像文件
        
        Args:
            imgs: 图像列表
            tile_size: 瓦片大小
        """
        for k, img in enumerate(imgs):
            img_file = f"indi_{k:06d}.jpg"
            norm_img = Encoder.normalized_img(img, tile_size)
            cv2.imwrite(img_file, norm_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    @staticmethod
    def merge(img: np.ndarray, c_idx: int, r_idx: int, 
             coefficients: Tuple[float, float, float, float],
             length: int, height: int, scale: int, threshold: float) -> bool:
        """
        测试并合并瓦片
        
        Args:
            img: 输入图像
            c_idx: 列索引
            r_idx: 行索引
            coefficients: 平面系数
            length: 长度
            height: 高度
            scale: 缩放因子
            threshold: 阈值
            
        Returns:
            bool: 是否成功合并
        """
        # 转换为矩阵索引
        c_idx_scaled = c_idx * scale
        r_idx_scaled = r_idx * scale
        length_scaled = length * scale
        height_scaled = height * scale
        
        # 过滤数值
        idx_vals, range_vals = Encoder.filter_vals(
            img, c_idx_scaled, r_idx_scaled, length_scaled, height_scaled
        )
        
        # 如果没有点，返回成功但系数为零
        if len(idx_vals) == 0:
            coefficients = (0.0, 0.0, 0.0, 0.0)
            return True
        
        # 平面拟合
        try:
            new_coefficients = Encoder.plane_fitting(idx_vals, range_vals)
            coefficients = new_coefficients
        except ValueError:
            return False
        
        # 检查是否满足阈值
        if not Encoder.check_mat(img, coefficients, threshold, 
                                c_idx_scaled, r_idx_scaled, 
                                length_scaled, height_scaled):
            return False
        
        return True
    
    @staticmethod
    def copy_unfit_points(img: np.ndarray, unfit_nums: List[float],
                         r_idx: int, c_idx: int, tile_size: int):
        """
        复制不适合拟合的点到 unfit_nums 列表
        
        Args:
            img: 输入图像
            unfit_nums: 不适合拟合的数值列表
            r_idx: 行索引
            c_idx: 列索引
            tile_size: 瓦片大小
        """
        for row in range(r_idx * tile_size, (r_idx + 1) * tile_size):
            for col in range(c_idx * tile_size, (c_idx + 1) * tile_size):
                num = img[row, col][0]  # 假设第一个通道是范围值
                if num != 0.0:
                    unfit_nums.append(num)
    
    @staticmethod
    def encode_occupation_mat(img: np.ndarray, occ_mat: np.ndarray, 
                             tile_size: int, idx_sizes: Tuple[int, int]):
        """
        编码占据矩阵
        
        Args:
            img: 输入图像
            occ_mat: 占据矩阵
            tile_size: 瓦片大小
            idx_sizes: 索引尺寸
        """
        for r_idx in range(idx_sizes[0]):
            for c_idx in range(idx_sizes[1]):
                code = 0
                for row in range(r_idx * tile_size, (r_idx + 1) * tile_size):
                    for col in range(c_idx * tile_size, (c_idx + 1) * tile_size):
                        num = img[row, col][0]  # 假设第一个通道是范围值
                        if num > 0.1:
                            pos = (row - r_idx * tile_size) * tile_size + (col - c_idx * tile_size)
                            code += (1 << pos)
                occ_mat[r_idx, c_idx] = code
    
    @staticmethod
    def single_channel_encode(img: np.ndarray, b_mat: np.ndarray, 
                             idx_sizes: Tuple[int, int],
                             coefficients: List[Tuple[float, float, float, float]],
                             unfit_nums: List[float], tile_fit_lengths: List[int],
                             threshold: float, tile_size: int) -> float:
        """
        单通道编码
        
        Args:
            img: 输入图像
            b_mat: 拟合标记矩阵
            idx_sizes: 索引尺寸
            coefficients: 平面系数列表
            unfit_nums: 不适合拟合的数值列表
            tile_fit_lengths: 瓦片拟合长度列表
            threshold: 阈值
            tile_size: 瓦片大小
            
        Returns:
            float: 拟合时间
        """
        fit_start = time.time()
        
        fit_cnt, unfit_cnt = 0, 0
        tt2 = tile_size * tile_size
        
        # 主编码循环
        for r_idx in range(idx_sizes[0]):
            c_idx = 0
            current_len = 1
            prev_c = (0.0, 0.0, 0.0, 0.0)
            c = (0.0, 0.0, 0.0, 0.0)
            
            while c_idx + current_len <= idx_sizes[1]:
                # 当长度为1时，尝试拟合单个瓦片
                if current_len == 1:
                    if not Encoder.merge(img, c_idx, r_idx, prev_c, 1, 1, tile_size, threshold):
                        # 设置 b_mat 为 0，表示该瓦片不能拟合
                        b_mat[r_idx, c_idx] = 0
                        c_idx += 1
                        current_len = 1
                        continue
                    else:
                        fit_cnt += 1
                        # 瓦片可以拟合
                        b_mat[r_idx, c_idx] = 1
                
                # 尝试合并下一个瓦片
                if Encoder.merge(img, c_idx, r_idx, c, current_len + 1, 1, tile_size, threshold):
                    # 如果已经到达列末尾
                    if current_len + c_idx >= idx_sizes[1]:
                        coefficients.append(c)
                        tile_fit_lengths.append(min(current_len + 1, idx_sizes[1] - c_idx))
                        b_mat[r_idx, idx_sizes[1] - 1] = 1
                        break
                    else:
                        prev_c = c
                        b_mat[r_idx, c_idx + current_len] = 1
                    
                    current_len += 1
                    fit_cnt += 1
                else:
                    coefficients.append(prev_c)
                    tile_fit_lengths.append(current_len)
                    c_idx = c_idx + current_len
                    current_len = 1
                    prev_c = (0.0, 0.0, 0.0, 0.0)
                    
                    if current_len + c_idx >= idx_sizes[1]:
                        break
            
            # 复制所有不适合拟合的点
            for c_idx in range(idx_sizes[1]):
                if b_mat[r_idx, c_idx] == 0:
                    Encoder.copy_unfit_points(img, unfit_nums, r_idx, c_idx, tile_size)
                    unfit_cnt += 1
        
        fit_end = time.time()
        fit_time = fit_end - fit_start
        
        print(f"Single with fitting_cnts: {fit_cnt} with unfitting_cnts: {unfit_cnt}")
        return fit_time
    
    @staticmethod
    def test_tile(img: np.ndarray, coefficients: Tuple[float, float, float, float],
                 threshold: float, c_idx: int, r_idx: int, 
                 length: int, height: int, offsets: List[float]) -> bool:
        """
        测试瓦片是否满足阈值要求
        
        Args:
            img: 输入图像
            coefficients: 平面系数
            threshold: 阈值
            c_idx: 列索引
            r_idx: 行索引
            length: 长度
            height: 高度
            offsets: 偏移列表
            
        Returns:
            bool: 是否满足阈值要求
        """
        a, b, c, d = coefficients
        if a == 0.0 and b == 0.0 and c == 0.0:
            return False
        
        cnt = 0
        sum_val = 0.0
        for j in range(height):
            for i in range(length):
                vec = img[r_idx + j, c_idx + i]
                actual_value = vec[0]  # 假设第一个通道是范围值
                if actual_value > 0:
                    cnt += 1
                    sum_val += a * vec[0] + b * j + c * i
        
        offset = sum_val / cnt if cnt > 0 else 0.0
        
        for j in range(height):
            for i in range(length):
                vec = img[r_idx + j, c_idx + i]
                actual_value = vec[0]
                if actual_value > 0:
                    val = abs(-offset / (a * vec[0] + b * j + c * i)) * actual_value
                    diff = abs(val - actual_value)
                    
                    if diff > threshold:
                        return False
        
        offsets.append(-offset)
        return True
    
    @staticmethod
    def multi_merge(imgs: List[np.ndarray], c_idx: int, r_idx: int,
                   coefficients: Tuple[float, float, float, float],
                   offsets: List[float], length: int, height: int, 
                   scale: int, threshold: float) -> bool:
        """
        多通道合并
        
        Args:
            imgs: 图像列表
            c_idx: 列索引
            r_idx: 行索引
            coefficients: 平面系数
            offsets: 偏移列表
            length: 长度
            height: 高度
            scale: 缩放因子
            threshold: 阈值
            
        Returns:
            bool: 是否成功合并
        """
        # 转换为矩阵索引
        c_idx_scaled = c_idx * scale
        r_idx_scaled = r_idx * scale
        length_scaled = length * scale
        height_scaled = height * scale
        offsets.clear()
        
        idx_vals, range_vals = [], []
        
        # 找到有足够点的通道
        for img in imgs:
            idx_vals, range_vals = Encoder.filter_vals(
                img, c_idx_scaled, r_idx_scaled, length_scaled, height_scaled
            )
            if len(idx_vals) >= 4:
                break
        
        # 如果点数不足
        if len(idx_vals) < 4:
            coefficients = (0.0, 0.0, 0.0, 0.0)
            offsets.extend([0.0] * len(imgs))
            return False
        
        # 平面拟合
        try:
            new_coefficients = Encoder.plane_fitting(idx_vals, range_vals)
            coefficients = new_coefficients
        except ValueError:
            coefficients = (0.0, 0.0, 0.0, 0.0)
            offsets.extend([0.0] * len(imgs))
            return False
        
        # 测试所有通道
        for img in imgs:
            if not Encoder.test_tile(img, coefficients, threshold, 
                                    c_idx_scaled, r_idx_scaled, 
                                    length_scaled, height_scaled, offsets):
                coefficients = (0.0, 0.0, 0.0, 0.0)
                offsets.clear()
                offsets.extend([0.0] * len(imgs))
                return False
        
        return True
    
    @staticmethod
    def remove_fit_points(img: np.ndarray, r_idx: int, c_idx: int, tile_size: int):
        """
        移除已拟合的点
        
        Args:
            img: 输入图像
            r_idx: 行索引
            c_idx: 列索引
            tile_size: 瓦片大小
        """
        for row in range(r_idx * tile_size, (r_idx + 1) * tile_size):
            for col in range(c_idx * tile_size, (c_idx + 1) * tile_size):
                img[row, col] = (0.0, 0.0, 0.0, 0.0)  # 设置为零
    
    @staticmethod
    def multi_channel_encode(imgs: List[np.ndarray], b_mat: np.ndarray,
                            idx_sizes: Tuple[int, int],
                            coefficients: List[Tuple[float, float, float, float]],
                            plane_offsets: List[List[float]], 
                            tile_fit_lengths: List[int],
                            threshold: float, tile_size: int) -> float:
        """
        多通道编码
        
        Args:
            imgs: 图像列表
            b_mat: 拟合标记矩阵
            idx_sizes: 索引尺寸
            coefficients: 平面系数列表
            plane_offsets: 平面偏移列表
            tile_fit_lengths: 瓦片拟合长度列表
            threshold: 阈值
            tile_size: 瓦片大小
            
        Returns:
            float: 拟合时间
        """
        fit_start = time.time()
        
        fit_cnt, unfit_cnt = 0, 0
        tt2 = tile_size * tile_size
        
        # 主编码循环
        for r_idx in range(idx_sizes[0]):
            c_idx = 0
            current_len = 1
            prev_c = (0.0, 0.0, 0.0, 0.0)
            c = (0.0, 0.0, 0.0, 0.0)
            offsets, prev_offsets = [], []
            
            while c_idx + current_len <= idx_sizes[1]:
                # 当长度为1时，尝试拟合单个瓦片
                if current_len == 1:
                    if not Encoder.multi_merge(imgs, c_idx, r_idx, prev_c, prev_offsets,
                                              1, 1, tile_size, threshold):
                        # 设置 b_mat 为 0，表示该瓦片不能拟合
                        b_mat[r_idx, c_idx] = 0
                        c_idx += 1
                        current_len = 1
                        continue
                    else:
                        fit_cnt += 1
                        # 瓦片可以拟合
                        b_mat[r_idx, c_idx] = 1
                
                # 尝试合并下一个瓦片
                if Encoder.multi_merge(imgs, c_idx, r_idx, c, offsets,
                                      current_len + 1, 1, tile_size, threshold):
                    # 如果已经到达列末尾
                    if current_len + c_idx >= idx_sizes[1]:
                        coefficients.append(c)
                        plane_offsets.append(offsets.copy())
                        tile_fit_lengths.append(min(current_len + 1, idx_sizes[1] - c_idx))
                        b_mat[r_idx, idx_sizes[1] - 1] = 1
                        break
                    else:
                        prev_c = c
                        prev_offsets = offsets.copy()
                        b_mat[r_idx, c_idx + current_len] = 1
                    
                    current_len += 1
                    fit_cnt += 1
                else:
                    coefficients.append(prev_c)
                    plane_offsets.append(prev_offsets.copy())
                    tile_fit_lengths.append(current_len)
                    c_idx = c_idx + current_len
                    current_len = 1
                    prev_c = (0.0, 0.0, 0.0, 0.0)
                    
                    if current_len + c_idx >= idx_sizes[1]:
                        break
            
            # 移除所有已拟合的点
            for ch, img in enumerate(imgs):
                for c_idx in range(idx_sizes[1]):
                    if b_mat[r_idx, c_idx] == 1:
                        Encoder.remove_fit_points(img, r_idx, c_idx, tile_size)
        
        fit_end = time.time()
        fit_time = fit_end - fit_start
        
        print(f"Multi with fitting_cnts: {fit_cnt} with unfitting_cnts: {unfit_cnt}")
        return fit_time

# 兼容性函数，保持与 C++ 相同的接口
def encode_occupation_mat(img: np.ndarray, occ_mat: np.ndarray, 
                         tile_size: int, idx_sizes: Tuple[int, int]):
    """
    编码占据矩阵 (兼容 C++ 接口)
    
    Args:
        img: 输入图像
        occ_mat: 占据矩阵
        tile_size: 瓦片大小
        idx_sizes: 索引尺寸
    """
    Encoder.encode_occupation_mat(img, occ_mat, tile_size, idx_sizes)

def single_channel_encode(img: np.ndarray, b_mat: np.ndarray, 
                         idx_sizes: Tuple[int, int],
                         coefficients: List[Tuple[float, float, float, float]],
                         unfit_nums: List[float], tile_fit_lengths: List[int],
                         threshold: float, tile_size: int) -> float:
    """
    单通道编码 (兼容 C++ 接口)
    
    Args:
        img: 输入图像
        b_mat: 拟合标记矩阵
        idx_sizes: 索引尺寸
        coefficients: 平面系数列表
        unfit_nums: 不适合拟合的数值列表
        tile_fit_lengths: 瓦片拟合长度列表
        threshold: 阈值
        tile_size: 瓦片大小
        
    Returns:
        float: 拟合时间
    """
    return Encoder.single_channel_encode(
        img, b_mat, idx_sizes, coefficients, unfit_nums, 
        tile_fit_lengths, threshold, tile_size
    )

def multi_channel_encode(imgs: List[np.ndarray], b_mat: np.ndarray,
                        idx_sizes: Tuple[int, int],
                        coefficients: List[Tuple[float, float, float, float]],
                        plane_offsets: List[List[float]], 
                        tile_fit_lengths: List[int],
                        threshold: float, tile_size: int) -> float:
    """
    多通道编码 (兼容 C++ 接口)
    
    Args:
        imgs: 图像列表
        b_mat: 拟合标记矩阵
        idx_sizes: 索引尺寸
        coefficients: 平面系数列表
        plane_offsets: 平面偏移列表
        tile_fit_lengths: 瓦片拟合长度列表
        threshold: 阈值
        tile_size: 瓦片大小
        
    Returns:
        float: 拟合时间
    """
    return Encoder.multi_channel_encode(
        imgs, b_mat, idx_sizes, coefficients, plane_offsets,
        tile_fit_lengths, threshold, tile_size
    )

# 使用示例
if __name__ == "__main__":
    # 创建测试数据
    img_rows, img_cols = 64, 64
    test_img = np.random.rand(img_rows, img_cols, 4).astype(np.float32)
    test_b_mat = np.zeros((8, 8), dtype=np.int32)
    test_idx_sizes = (8, 8)
    test_coefficients = []
    test_unfit_nums = []
    test_tile_fit_lengths = []
    test_threshold = 0.1
    test_tile_size = 8
    
    # 测试单通道编码
    fit_time = single_channel_encode(
        test_img, test_b_mat, test_idx_sizes, test_coefficients,
        test_unfit_nums, test_tile_fit_lengths, test_threshold, test_tile_size
    )
    print(f"Single channel encoding time: {fit_time:.4f} seconds")
    print(f"Fitted coefficients: {len(test_coefficients)}")
    print(f"Unfit numbers: {len(test_unfit_nums)}")
    
    # 测试多通道编码
    test_imgs = [test_img, test_img.copy()]
    test_plane_offsets = []
    fit_time_multi = multi_channel_encode(
        test_imgs, test_b_mat, test_idx_sizes, test_coefficients,
        test_plane_offsets, test_tile_fit_lengths, test_threshold, test_tile_size
    )
    print(f"Multi channel encoding time: {fit_time_multi:.4f} seconds")
    print(f"Plane offsets: {len(test_plane_offsets)}")