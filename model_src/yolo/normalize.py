
import os
from PIL import Image
import numpy as np

def preprocess_images(directory):
    output_directory = os.path.join(directory, 'processed')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".bmp"):
            file_path = os.path.join(directory, filename)
            image = Image.open(file_path)

            # 转换为黑白（灰度）
            grayscale_image = image.convert("L")

            # 归一化：将图像像素值从 [0, 255] 缩放到 [0, 1]
            normalized_image_array = np.array(grayscale_image) / 255.0
            normalized_image = Image.fromarray((normalized_image_array * 255).astype(np.uint8))

            output_path = os.path.join(output_directory, filename)
            normalized_image.save(output_path)
            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    current_directory = os.getcwd()
    preprocess_images(current_directory)
