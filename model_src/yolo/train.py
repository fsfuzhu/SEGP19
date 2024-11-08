import os
import shutil
import random
from ultralytics import YOLO

def split_data(image_path, train_dir, val_dir, split_ratio=0.8):
    # 获取所有图片文件
    all_files = [f for f in os.listdir(image_path) if f.endswith(('.jpg', '.png'))]
    random.shuffle(all_files)  # 随机打乱文件顺序

    # 分割为训练集和验证集
    train_files = all_files[:int(len(all_files) * split_ratio)]
    val_files = all_files[int(len(all_files) * split_ratio):]

    # 创建训练集和验证集文件夹
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 将文件复制到各自的文件夹中
    for file in train_files:
        # 复制图片文件
        shutil.copy(os.path.join(image_path, file), os.path.join(train_dir, file))
        
        # 检查并复制对应的标签文件
        txt_file = os.path.splitext(file)[0] + '.txt'
        txt_src = os.path.join(image_path, txt_file)
        if os.path.exists(txt_src):
            shutil.copy(txt_src, os.path.join(train_dir, txt_file))
    
    for file in val_files:
        # 复制图片文件
        shutil.copy(os.path.join(image_path, file), os.path.join(val_dir, file))
        
        # 检查并复制对应的标签文件
        txt_file = os.path.splitext(file)[0] + '.txt'
        txt_src = os.path.join(image_path, txt_file)
        if os.path.exists(txt_src):
            shutil.copy(txt_src, os.path.join(val_dir, txt_file))

    print(f"训练集大小: {len(train_files)}, 验证集大小: {len(val_files)}")

def main():
    # 定义图片和标注数据的路径
    image_path = r'C:\Users\Administrator\Desktop\Pap Semar\AP_18 labeled'
    train_dir = os.path.join(image_path, 'train')
    val_dir = os.path.join(image_path, 'val')
    
    # 分割数据集
    split_data(image_path, train_dir, val_dir)

    #for identifying all cells
    # 定义训练配置
    data_config = f"""
    # YOLO数据配置
    train: {train_dir}
    val: {val_dir}

    # 类别名称列表
    names:
      0: cells
    """

    # 创建一个新的数据配置文件
    data_file = 'smear_data.yaml'
    with open(data_file, 'w') as file:
        file.write(data_config)

    # 初始化YOLO模型并加载更大的预训练权重（精度优先）
    model = YOLO('yolov8n.pt')  # 使用更大的模型 'yolov8x.pt' 提升精度

    # 设置自定义的超参数
    model.train(
        data=data_file,
        epochs=1000,  # 增加 epoch 数，进行更长时间的训练
        batch=16,  # 如果显存允许，使用更大的 batch size，建议从 16 或 32 开始
        imgsz=640,  # 保持图像大小为 640x640
        save_period=10,  # 每隔 10 个 epoch 保存一次模型
        name='smear_cell_detection',
        lr0=1e-4,  # 初始学习率
        optimizer='AdamW',  # 使用 AdamW 优化器，提升训练稳定性
        amp=False,  # 启用混合精度训练
        augment=True,  # 启用数据增强
        patience=10  # 使用 early stopping，等待 50 个 epoch 后验证不再提升才停止
    )

if __name__ == '__main__':
    main()
