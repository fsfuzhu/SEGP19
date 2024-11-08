import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO
import torch.nn as nn
import torch.nn.functional as F



'''
STEP 1: Install the required libraries
install needed library
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install ultralytics

STEP 2: 
cd RESNET18
python inference.py
'''
 
# --------------------------------------------------------------
# below are are the arguments needed for the inference.py script
# --------------------------------------------------------------
# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to models (update these paths as needed)
RESNET_MODEL_PATH = './models/resnet18_model.pth'
YOLO_MODEL_PATH = './models/yolov8x_model.pt'

# Test images directory
INFER_IMAGES_DIR = f'./images'
OUTPUT_DIR = f'./outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Detection and classification thresholds
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Adjust based on model performance
RESNET_CONFIDENCE_THRESHOLD = 0.0  # Adjust based on desired specificity

# Class names
CLASS_NAMES = ['abnormal', 'benign', 'normal']

CLASS_COLOURS = {
    "normal": (102, 204, 0),    # Green for Normal
    "abnormal": (0, 0, 255),    # Red for Abnormal
    "benign": (255, 0, 0),      # Blue for Benigh
}

# Load YOLO model (Assuming a PyTorch model)
def load_yolo_model(model_path):
    # Replace with your YOLO model loading code
    yolo_model = YOLO(model_path)
    return yolo_model

# Load ResNet model
def load_resnet_model(model_path, num_classes=3):
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

# Preprocessing for ResNet model
def preprocess_cell_image(cell_image):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])
    return preprocess(cell_image)

# Function to perform cell detection using YOLO
def detect_cells(yolo_model, image):
    results = yolo_model.predict(source=image, save=False)
    boxes = []
    for result in results:
        if result.boxes is None:
            continue  # No boxes detected in this result

        # Iterate over each bounding box
        for i, box in enumerate(result.boxes.xyxy):
            # Move the box tensor to CPU and convert to NumPy
            box_np = box.cpu().numpy()
            
            # Check for NaN values
            if np.isnan(box_np).any():
                continue  # Skip invalid boxes

            # Convert box coordinates to integers
            x1, y1, x2, y2 = map(int, box_np)
            conf = float(result.boxes.conf[i].cpu().numpy()) if len(result.boxes.conf) > 0 else 0.0
            if conf > YOLO_CONFIDENCE_THRESHOLD:
                boxes.append({
                    'bbox': [x1, y1, x2, y2]
                })
    return boxes

# Function to classify cropped cells using ResNet
def classify_cells(resnet_model, cell_images):
    cell_tensors = []
    for cell_img in cell_images:
        processed = preprocess_cell_image(cell_img)
        cell_tensors.append(processed)
    cell_batch = torch.stack(cell_tensors).to(device)
    with torch.no_grad():
        outputs = resnet_model(cell_batch)
        probabilities = F.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, dim=1)
    return predictions.cpu().numpy(), confidences.cpu().numpy()

# Main testing function
def test_cells(save_images = False):
    # Load models
    yolo_model = load_yolo_model(YOLO_MODEL_PATH)
    resnet_model = load_resnet_model(RESNET_MODEL_PATH)
    normal_total, abnormal_total, benign_total = 0, 0, 0
    # Iterate over test images
    for img_filename in os.listdir(INFER_IMAGES_DIR):
        img_path = os.path.join(INFER_IMAGES_DIR, img_filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image {img_filename}")
            continue

        # Detect cells
        detections = detect_cells(yolo_model, image)

        # List to hold cropped cell images
        cropped_cells = []
        boxes_to_draw = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cell_image = image[y1:y2, x1:x2]
            cropped_cells.append(cell_image)
            boxes_to_draw.append({
                'bbox': det['bbox'],
                'class': None,  # Placeholder, will be filled after classification
                'class_confidence': None,  # Placeholder
            })

        if len(cropped_cells) == 0:
            continue

        # Classify cells
        predictions, confidences = classify_cells(resnet_model, cropped_cells)

        # Filter and annotate detections
        for idx, box in enumerate(boxes_to_draw):
            class_idx = predictions[idx]
            class_confidence = confidences[idx]
            if class_confidence < RESNET_CONFIDENCE_THRESHOLD:
                continue  # Skip low-confidence predictions
            
            if class_idx == 0:
                abnormal_total += 1
            elif class_idx == 1:
                benign_total += 1
            else:
                normal_total += 1
            class_name = CLASS_NAMES[class_idx]
            box['class'] = class_name
            box['class_confidence'] = class_confidence

            # Draw bounding box and label on the image
            x1, y1, x2, y2 = map(int, box['bbox'])
            label = f"{class_name}: {class_confidence:.2f}"
            color = CLASS_COLOURS.get(class_name, (255, 255, 255))  # Default to white if class ID not in class_colors
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1 + 1, y1 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # uncomment to Save annotated image
        if save_images:
            output_path = os.path.join(OUTPUT_DIR, img_filename)
            cv2.imwrite(output_path, image)
            print(f"Processed and saved annotated image: {output_path}")

    total_cells = normal_total + abnormal_total + benign_total
    print(f"Normal: {normal_total} ({(normal_total/total_cells)*100}%), Abnormal: {abnormal_total} ({(abnormal_total/total_cells)*100}%), Benign: {benign_total} ({(benign_total/total_cells)*100}%)")

if __name__ == '__main__':
    test_cells(save_images=True)
