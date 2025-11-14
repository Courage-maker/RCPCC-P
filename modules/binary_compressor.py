import zstandard as zstd
import os
import sys

def read_file(file_path):
    """Read a file into a bytes object"""
    try:
        with open(file_path, 'rb') as file:
            return file.read()
    except Exception as e:
        raise RuntimeError(f"Could not open or read file: {file_path}") from e

def write_file(file_path, data):
    """Write data to a file"""
    try:
        with open(file_path, 'wb') as file:
            file.write(data)
    except Exception as e:
        raise RuntimeError(f"Could not open or write file: {file_path}") from e

def compress_data(data):
    """Compress data using Zstandard"""
    if not data:
        return b''
    
    # Create compressor with maximum compression level
    compressor = zstd.ZstdCompressor(level=zstd.MAX_COMPRESSION_LEVEL)
    
    try:
        compressed_data = compressor.compress(data)
        
        # Calculate compression ratio
        compression_ratio = len(data) / len(compressed_data) if len(compressed_data) > 0 else 0
        print(f"Binary Compression ratio: {compression_ratio:.2f}")
        
        return compressed_data
    except Exception as e:
        raise RuntimeError(f"Compression failed: {str(e)}") from e

def decompress_data(compressed_data):
    """Decompress data using Zstandard"""
    if not compressed_data:
        return b''
    
    # Create decompressor
    decompressor = zstd.ZstdDecompressor()
    
    try:
        decompressed_data = decompressor.decompress(compressed_data)
        return decompressed_data
    except Exception as e:
        raise RuntimeError(f"Decompression failed: {str(e)}") from e

# 使用示例
if __name__ == "__main__":
    # 示例用法
    try:
        # 读取文件
        original_data = read_file("input.bin")
        print(f"Original size: {len(original_data)} bytes")
        
        # 压缩数据
        compressed = compress_data(original_data)
        print(f"Compressed size: {len(compressed)} bytes")
        
        # 写入压缩文件
        write_file("compressed.bin.zst", compressed)
        
        # 解压数据
        decompressed = decompress_data(compressed)
        print(f"Decompressed size: {len(decompressed)} bytes")
        
        # 验证数据完整性
        if original_data == decompressed:
            print("Compression/Decompression successful - data integrity verified")
        else:
            print("ERROR: Data mismatch after compression/decompression")
            
    except Exception as e:
        print(f"Error: {e}")