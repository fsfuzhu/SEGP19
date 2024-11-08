import sys
import os
import pyvips
from concurrent.futures import ThreadPoolExecutor

def process_tile(image, scale, tx, ty, tile_size, level, output_dir):
    """
    处理单个瓦片并保存为 JPG 文件。
    """
    try:
        x = tx * tile_size * scale
        y = ty * tile_size * scale
        w = tile_size * scale
        h = tile_size * scale

        # 提取区域
        region = image.crop(x, y, w, h).resize(1 / scale)

        # 转换为 RGB 模式
        if region.bands == 4:
            region = region[:3]  # 去除 alpha 通道
        elif region.bands == 1:
            region = region.colourspace("srgb")

        # 定义输出文件名
        output_filename = f"tile_{ty}_{tx}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # 保存为 JPG 文件
        region.write_to_file(output_path, Q=100)  # Q=90 表示 JPEG 质量
        print(f"保存瓦片: {output_path}")
    except Exception as e:
        print(f"瓦片 ({tx}, {ty}) 处理失败: {e}")

def convert_svs_to_jpg_tiles_parallel(input_path, output_dir, tile_size=1024, level=0, max_workers=4):
    """
    并行将 SVS 文件分割成多个 JPG 瓦片文件。

    参数：
    - input_path: 输入的 SVS 文件路径
    - output_dir: 输出 JPG 文件的目录
    - tile_size: 每个瓦片的大小（默认为 1024x1024 像素）
    - level: 要读取的图像级别（默认 0 为最高分辨率）
    - max_workers: 并行工作的线程数（默认为 4）
    """
    try:
        # 加载 SVS 文件
        image = pyvips.Image.new_from_file(input_path, access='sequential')

        # 计算缩放因子
        scale_str = image.get('openslide.level[{}].downsample'.format(level))
        if scale_str is None:
            scale = 1.0  # 如果没有指定级别的缩放因子，默认为1.0
        else:
            scale = float(scale_str)  # 将字符串转换为浮点数

        # 获取指定级别的尺寸
        width = int(image.width / scale)
        height = int(image.height / scale)
        print(f"读取级别 {level} 的图像尺寸: {width}x{height}")

        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)

        # 计算瓦片数量
        tiles_x = (width + tile_size - 1) // tile_size
        tiles_y = (height + tile_size - 1) // tile_size
        print(f"分割为 {tiles_x} x {tiles_y} = {tiles_x * tiles_y} 个瓦片")

        # 使用线程池并行处理瓦片
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    executor.submit(process_tile, image, scale, tx, ty, tile_size, level, output_dir)

        print("转换完成！")

    except Exception as e:
        print(f"转换失败: {e}")

def main():
    if len(sys.argv) < 3:
        print("用法: python split_svs_to_jpg_pyvips.py <输入.svs> <输出目录> [tile_size] [level] [max_workers]")
        print("示例: python split_svs_to_jpg_pyvips.py input.svs output_tiles 1024 0 8")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    # 默认参数
    tile_size = 1024
    level = 0
    max_workers = 4

    # 解析可选参数
    if len(sys.argv) >= 4:
        try:
            tile_size = int(sys.argv[3])
        except ValueError:
            print("tile_size 必须是整数，例如 1024")
            sys.exit(1)
    if len(sys.argv) >= 5:
        try:
            level = int(sys.argv[4])
        except ValueError:
            print("level 必须是整数，例如 0")
            sys.exit(1)
    if len(sys.argv) == 6:
        try:
            max_workers = int(sys.argv[5])
        except ValueError:
            print("max_workers 必须是整数，例如 8")
            sys.exit(1)

    # 检查输入文件是否存在
    if not os.path.isfile(input_file):
        print(f"输入文件不存在: {input_file}")
        sys.exit(1)

    convert_svs_to_jpg_tiles_parallel(input_file, output_dir, tile_size, level, max_workers)

if __name__ == "__main__":
    main()
