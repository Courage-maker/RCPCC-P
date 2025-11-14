# project_config.py
"""
点云压缩项目配置
Python 版本的项目配置和依赖管理
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import platform

class ProjectConfig:
    """项目配置类"""
    
    # 项目基本信息
    PROJECT_NAME = "pcc_project"
    VERSION = "1.0.0"
    PYTHON_VERSION = "3.8+"
    
    # 依赖库版本要求
    DEPENDENCIES = {
        # 核心数据处理
        "numpy": ">=1.21.0",
        "opencv-python": ">=4.5.0",  # OpenCV
        "open3d": ">=0.15.0",        # 3D 数据处理，替代 PCL
        "scipy": ">=1.7.0",          # 科学计算，包含 FFT
        "pyzstd": ">=0.15.0",        # Zstandard 压缩
        "grpcio": ">=1.40.0",        # gRPC
        
        # 数据处理和序列化
        "protobuf": ">=3.19.0",
        "pillow": ">=8.3.0",
        "matplotlib": ">=3.4.0",
        
        # 工具库
        "click": ">=8.0.0",          # 命令行界面
        "tqdm": ">=4.62.0",          # 进度条
        "psutil": ">=5.8.0",         # 系统监控
    }
    
    # 可选依赖
    OPTIONAL_DEPENDENCIES = {
        "cupy": ">=9.0.0",           # CUDA 加速（可选）
        "pycuda": ">=2021.0",        # CUDA 支持（可选）
        "mpi4py": ">=3.1.0",         # MPI 并行（可选）
    }
    
    # 开发依赖
    DEV_DEPENDENCIES = {
        "pytest": ">=6.2.0",
        "pytest-cov": ">=2.12.0",
        "black": ">=21.0.0",
        "mypy": ">=0.910",
        "flake8": ">=3.9.0",
        "pylint": ">=2.11.0",
    }

class PathConfig:
    """路径配置"""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        
        # 主要目录
        self.source_dir = self.base_dir / "src"
        self.utils_dir = self.source_dir / "utils"
        self.modules_dir = self.source_dir / "modules"
        self.proto_dir = self.source_dir / "proto"
        self.examples_dir = self.base_dir / "examples"
        self.tests_dir = self.base_dir / "tests"
        self.docs_dir = self.base_dir / "docs"
        self.data_dir = self.base_dir / "data"
        
        # 输出目录
        self.build_dir = self.base_dir / "build"
        self.dist_dir = self.base_dir / "dist"
        self.logs_dir = self.base_dir / "logs"
        
    def setup_directories(self) -> None:
        """创建项目目录结构"""
        directories = [
            self.source_dir,
            self.utils_dir,
            self.modules_dir,
            self.proto_dir,
            self.examples_dir,
            self.tests_dir,
            self.docs_dir,
            self.data_dir,
            self.build_dir,
            self.dist_dir,
            self.logs_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")

class BuildConfig:
    """构建配置"""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.optimization_level = "O3"  # 模拟 C++ 的 -O3 优化
        
        # 平台特定配置
        if self.platform == "linux":
            self.opencl_available = self._check_opencl()
            self.cuda_available = self._check_cuda()
        else:
            self.opencl_available = False
            self.cuda_available = False
    
    def _check_opencl(self) -> bool:
        """检查 OpenCL 支持"""
        try:
            import pyopencl
            return True
        except ImportError:
            return False
    
    def _check_cuda(self) -> bool:
        """检查 CUDA 支持"""
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def get_build_flags(self) -> Dict[str, any]:
        """获取构建标志"""
        flags = {
            "optimization": self.optimization_level,
            "debug_info": True,  # 模拟 -g 标志
            "platform": self.platform,
            "opencl": self.opencl_available,
            "cuda": self.cuda_available,
            "vectorization": True,  # 启用向量化优化
            "parallel": True,       # 启用并行处理
        }
        return flags

class DependencyManager:
    """依赖管理"""
    
    def __init__(self):
        self.config = ProjectConfig()
    
    def check_installed(self, package: str) -> bool:
        """检查包是否已安装"""
        try:
            __import__(package.replace("-", "_"))
            return True
        except ImportError:
            return False
    
    def install_dependencies(self, include_optional: bool = False, 
                           include_dev: bool = False) -> bool:
        """安装项目依赖"""
        try:
            import pip
            
            # 安装必需依赖
            for package, version in self.config.DEPENDENCIES.items():
                if not self.check_installed(package):
                    print(f"Installing {package}{version}...")
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", 
                        f"{package}{version}"
                    ], check=True)
            
            # 安装可选依赖
            if include_optional:
                for package, version in self.config.OPTIONAL_DEPENDENCIES.items():
                    if not self.check_installed(package):
                        print(f"Installing optional dependency {package}{version}...")
                        subprocess.run([
                            sys.executable, "-m", "pip", "install", 
                            f"{package}{version}"
                        ], check=True)
            
            # 安装开发依赖
            if include_dev:
                for package, version in self.config.DEV_DEPENDENCIES.items():
                    if not self.check_installed(package):
                        print(f"Installing dev dependency {package}{version}...")
                        subprocess.run([
                            sys.executable, "-m", "pip", "install", 
                            f"{package}{version}"
                        ], check=True)
            
            return True
            
        except Exception as e:
            print(f"Failed to install dependencies: {e}")
            return False
    
    def generate_requirements_file(self, filename: str = "requirements.txt") -> None:
        """生成 requirements.txt 文件"""
        with open(filename, 'w') as f:
            f.write(f"# {ProjectConfig.PROJECT_NAME} requirements\n")
            f.write(f"# Python {ProjectConfig.PYTHON_VERSION}\n\n")
            
            # 必需依赖
            f.write("# Core dependencies\n")
            for package, version in ProjectConfig.DEPENDENCIES.items():
                f.write(f"{package}{version}\n")
            
            # 可选依赖
            f.write("\n# Optional dependencies\n")
            for package, version in ProjectConfig.OPTIONAL_DEPENDENCIES.items():
                f.write(f"# {package}{version}\n")
            
            # 开发依赖
            f.write("\n# Development dependencies\n")
            for package, version in ProjectConfig.DEV_DEPENDENCIES.items():
                f.write(f"# {package}{version}\n")

class ProjectBuilder:
    """项目构建器"""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.paths = PathConfig(base_dir)
        self.build_config = BuildConfig()
        self.dependency_manager = DependencyManager()
    
    def setup_project(self) -> bool:
        """设置项目"""
        try:
            print("Setting up project directories...")
            self.paths.setup_directories()
            
            print("Installing dependencies...")
            if not self.dependency_manager.install_dependencies():
                return False
            
            print("Generating requirements file...")
            self.dependency_manager.generate_requirements_file()
            
            print("Project setup completed successfully!")
            return True
            
        except Exception as e:
            print(f"Project setup failed: {e}")
            return False
    
    def build(self, clean: bool = False) -> bool:
        """构建项目"""
        try:
            if clean:
                self.clean()
            
            print("Building project...")
            flags = self.build_config.get_build_flags()
            print(f"Build flags: {flags}")
            
            # 在 Python 中，"构建"主要是准备环境和验证依赖
            if not self.validate_environment():
                return False
            
            print("Build completed successfully!")
            return True
            
        except Exception as e:
            print(f"Build failed: {e}")
            return False
    
    def clean(self) -> None:
        """清理构建文件"""
        import shutil
        
        if self.paths.build_dir.exists():
            shutil.rmtree(self.paths.build_dir)
            print(f"Removed {self.paths.build_dir}")
        
        if self.paths.dist_dir.exists():
            shutil.rmtree(self.paths.dist_dir)
            print(f"Removed {self.paths.dist_dir}")
        
        if self.paths.logs_dir.exists():
            shutil.rmtree(self.paths.logs_dir)
            print(f"Removed {self.paths.logs_dir}")
    
    def validate_environment(self) -> bool:
        """验证环境配置"""
        print("Validating environment...")
        
        # 检查 Python 版本
        python_version = sys.version_info
        required_version = (3, 8)
        if python_version < required_version:
            print(f"Python version {python_version} is below required {required_version}")
            return False
        
        # 检查必需依赖
        missing_deps = []
        for package in ProjectConfig.DEPENDENCIES:
            if not self.dependency_manager.check_installed(package):
                missing_deps.append(package)
        
        if missing_deps:
            print(f"Missing dependencies: {missing_deps}")
            return False
        
        print("Environment validation passed!")
        return True

# 命令行接口
def main():
    """主函数 - 命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description=f"{ProjectConfig.PROJECT_NAME} 项目管理工具")
    parser.add_argument("command", choices=["setup", "build", "clean", "deps", "validate"],
                       help="要执行的命令")
    parser.add_argument("--base-dir", help="项目基础目录")
    parser.add_argument("--include-optional", action="store_true", 
                       help="包含可选依赖")
    parser.add_argument("--include-dev", action="store_true", 
                       help="包含开发依赖")
    parser.add_argument("--clean", action="store_true", 
                       help="构建前清理")
    
    args = parser.parse_args()
    
    builder = ProjectBuilder(args.base_dir)
    
    if args.command == "setup":
        success = builder.setup_project()
    elif args.command == "build":
        success = builder.build(clean=args.clean)
    elif args.command == "clean":
        builder.clean()
        success = True
    elif args.command == "deps":
        success = builder.dependency_manager.install_dependencies(
            include_optional=args.include_optional,
            include_dev=args.include_dev
        )
    elif args.command == "validate":
        success = builder.validate_environment()
    else:
        print(f"Unknown command: {args.command}")
        success = False
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()