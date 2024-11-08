import os
import shutil
import random
from ultralytics import YOLO

def split_data(image_path, train_dir, val_dir, split_ratio=0.8):
    # Get all image files
    all_files = [f for f in os.listdir(image_path) if f.endswith(('.jpg', '.png'))]
    random.shuffle(all_files)  # Shuffle file order randomly

    # Split into training and validation sets
    train_files = all_files[:int(len(all_files) * split_ratio)]
    val_files = all_files[int(len(all_files) * split_ratio):]

    # Create training and validation directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Copy files into respective directories
    for file in train_files:
        # Copy image file
        shutil.copy(os.path.join(image_path, file), os.path.join(train_dir, file))
        
        # Check and copy corresponding label file
        txt_file = os.path.splitext(file)[0] + '.txt'
        txt_src = os.path.join(image_path, txt_file)
        if os.path.exists(txt_src):
            shutil.copy(txt_src, os.path.join(train_dir, txt_file))
    
    for file in val_files:
        # Copy image file
        shutil.copy(os.path.join(image_path, file), os.path.join(val_dir, file))
        
        # Check and copy corresponding label file
        txt_file = os.path.splitext(file)[0] + '.txt'
        txt_src = os.path.join(image_path, txt_file)
        if os.path.exists(txt_src):
            shutil.copy(txt_src, os.path.join(val_dir, txt_file))

    print(f"Training set size: {len(train_files)}, Validation set size: {len(val_files)}")

def main():
    # Define the paths for image and label data
    image_path = r'C:\Users\Administrator\Desktop\Pap Semar\AP_18 labeled'
    train_dir = os.path.join(image_path, 'train')
    val_dir = os.path.join(image_path, 'val')
    
    # Split the dataset
    split_data(image_path, train_dir, val_dir)

    # Configuration for identifying all cells
    # Define training configuration
    data_config = f"""
    # YOLO data configuration
    train: {train_dir}
    val: {val_dir}

    # List of class names
    names:
      0: cells
    """

    # Create a new data configuration file
    data_file = 'smear_data.yaml'
    with open(data_file, 'w') as file:
        file.write(data_config)

    # Initialize the YOLO model and load larger pretrained weights (accuracy-focused)
    model = YOLO('yolov8n.pt')  # Use a larger model 'yolov8x.pt' for higher accuracy

    # Set custom hyperparameters
    model.train(
        data=data_file,
        epochs=1000,  # Train for 1000 epochs
        batch=16,  # Increase batch size if memory allows
        imgsz=640,  # Keep image size as 640x640
        save_period=10,  # Save model every 10 epochs
        name='smear_cell_detection',
        lr0=1e-4,  # Initial learning rate
        optimizer='AdamW',  # Use AdamW optimizer for stable training
        amp=False,  # Enable mixed precision (set False for GTX to avoid bugs)
        augment=True,  # Enable data augmentation
        patience=10  # Apply early stopping, stop if no improvement after 10 epochs
    )

if __name__ == '__main__':
    main()
