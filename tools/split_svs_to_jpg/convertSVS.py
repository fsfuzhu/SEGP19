import sys
import os
import pyvips
from concurrent.futures import ThreadPoolExecutor

def process_tile(image, scale, tx, ty, tile_size, level, output_dir):
    """
    Process a single tile and save it as a JPG file.
    """
    try:
        x = tx * tile_size * scale
        y = ty * tile_size * scale
        w = tile_size * scale
        h = tile_size * scale

        # Extract the region
        region = image.crop(x, y, w, h).resize(1 / scale)

        # Convert to RGB mode
        if region.bands == 4:
            region = region[:3]  # Remove the alpha channel
        elif region.bands == 1:
            region = region.colourspace("srgb")

        # Define output filename
        output_filename = f"tile_{ty}_{tx}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Save as JPG file
        region.write_to_file(output_path, Q=100)  # Q=90 for JPEG quality
        print(f"Saved tile: {output_path}")
    except Exception as e:
        print(f"Failed to process tile ({tx}, {ty}): {e}")

def convert_svs_to_jpg_tiles_parallel(input_path, output_dir, tile_size=1024, level=0, max_workers=4):
    """
    Split an SVS file into multiple JPG tiles in parallel.

    Parameters:
    - input_path: Input SVS file path
    - output_dir: Directory for output JPG files
    - tile_size: Size of each tile (default is 1024x1024 pixels)
    - level: Image level to read (default 0 is the highest resolution)
    - max_workers: Number of threads for parallel processing (default is 4)
    """
    try:
        # Load the SVS file
        image = pyvips.Image.new_from_file(input_path, access='sequential')

        # Calculate scale factor
        scale_str = image.get('openslide.level[{}].downsample'.format(level))
        if scale_str is None:
            scale = 1.0  # Default to 1.0 if no downscale factor is specified
        else:
            scale = float(scale_str)  # Convert string to float

        # Get dimensions for the specified level
        width = int(image.width / scale)
        height = int(image.height / scale)
        print(f"Image dimensions at level {level}: {width}x{height}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Calculate number of tiles
        tiles_x = (width + tile_size - 1) // tile_size
        tiles_y = (height + tile_size - 1) // tile_size
        print(f"Dividing into {tiles_x} x {tiles_y} = {tiles_x * tiles_y} tiles")

        # Process tiles in parallel using a thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    executor.submit(process_tile, image, scale, tx, ty, tile_size, level, output_dir)

        print("Conversion complete!")

    except Exception as e:
        print(f"Conversion failed: {e}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python split_svs_to_jpg_pyvips.py <input.svs> <output_directory> [tile_size] [level] [max_workers]")
        print("Example: python split_svs_to_jpg_pyvips.py input.svs output_tiles 1024 0 8")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    # Default parameters
    tile_size = 1024
    level = 0
    max_workers = 4

    # Parse optional parameters
    if len(sys.argv) >= 4:
        try:
            tile_size = int(sys.argv[3])
        except ValueError:
            print("tile_size must be an integer, e.g., 1024")
            sys.exit(1)
    if len(sys.argv) >= 5:
        try:
            level = int(sys.argv[4])
        except ValueError:
            print("level must be an integer, e.g., 0")
            sys.exit(1)
    if len(sys.argv) == 6:
        try:
            max_workers = int(sys.argv[5])
        except ValueError:
            print("max_workers must be an integer, e.g., 8")
            sys.exit(1)

    # Check if the input file exists
    if not os.path.isfile(input_file):
        print(f"Input file does not exist: {input_file}")
        sys.exit(1)

    convert_svs_to_jpg_tiles_parallel(input_file, output_dir, tile_size, level, max_workers)

if __name__ == "__main__":
    main()
