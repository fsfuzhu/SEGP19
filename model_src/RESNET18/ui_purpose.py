# scripts/test_cells.py

import os
import cv2
import torch
import json
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO
import torch.nn as nn
import torch.nn.functional as F

# Load YOLO model
def load_yolo_model(model_path):
    yolo_model = YOLO(model_path)
    return yolo_model

# Load ResNet model using the new "weights" parameter
def load_resnet_model(model_path, num_classes=3):
    model = resnet18(weights=None)  # Use 'weights=None' instead of pretrained=False
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
def test_cells():
    # Dictionary to store JSON detection results
    detection_results = {}
    
    # Load models
    yolo_model = load_yolo_model(YOLO_MODEL_PATH)
    resnet_model = load_resnet_model(RESNET_MODEL_PATH)
    normal_total, abnormal_total, benign_total = 0, 0, 0
    
    # Mapping from model prediction index to our JSON keys (lowercase)
    class_mapping = {0: "abnormal", 1: "benign", 2: "normal"}
    
    # Iterate over test images
    for img_filename in required_image_names:
        img_path = os.path.join(TEST_IMAGES_DIR, img_filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image {img_filename}")
            continue
        
        # Initialize dictionary for current image
        detection_results[img_filename] = {
            "normal": [],
            "abnormal": [],
            "benign": []
        }
        
        # Detect cells
        detections = detect_cells(yolo_model, image)

        # List to hold cropped cell images and corresponding bounding boxes for later use
        cropped_cells = []
        boxes_to_draw = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cell_image = image[y1:y2, x1:x2]
            cropped_cells.append(cell_image)
            boxes_to_draw.append({
                'bbox': det['bbox'],
                'class': None,           # Placeholder, will be filled after classification
                'class_confidence': None,  # Placeholder
            })

        if len(cropped_cells) == 0:
            print(f"No cells detected in image {img_filename}")
            continue

        # Classify cells
        predictions, confidences = classify_cells(resnet_model, cropped_cells)

        ID = 1
        # Iterate over detections to annotate and save JSON data
        for idx, box in enumerate(boxes_to_draw):
            class_idx = predictions[idx]
            class_confidence = confidences[idx]
            if class_confidence < RESNET_CONFIDENCE_THRESHOLD:
                continue  # Skip low-confidence predictions
            
            # Update totals
            if class_idx == 0:
                abnormal_total += 1
            elif class_idx == 1:
                benign_total += 1
            else:
                normal_total += 1
                
            class_name = CLASS_NAMES[class_idx]  # e.g. 'Abnormal'
            cell_id = f"{class_name[0]}{ID}"  # e.g. 'A1'
            box['class'] = class_name
            box['class_confidence'] = class_confidence

            # Draw bounding box and label on the image
            x1, y1, x2, y2 = map(int, box['bbox'])
            label = cell_id
            color = CLASS_COLOURS.get(class_name, (255, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1 + 1, y1 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Ensure the key for this image exists (using setdefault)
            detection_results.setdefault(img_filename, {"normal": [], "abnormal": [], "benign": []})
            
            # Save detection information in the JSON structure
            detection_info = {
                "cell_ID": cell_id,
                "x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2,
                "confidence": float(class_confidence)
            }
            # Use the mapping to get the proper JSON key
            json_key = class_mapping[class_idx]
            detection_results[img_filename][json_key].append(detection_info)
            
            ID += 1

        # Save annotated image
        output_path = os.path.join(OUTPUT_DIR, img_filename)
        # cv2.imwrite(output_path, image)
    
    # Write the detection results to a JSON file
    with open(JSON_PATH, 'w') as json_file:
        json.dump(detection_results, json_file, indent=4)
    
    return detection_results

if __name__ == '__main__':
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths to models (update these paths as needed)
    RESNET_MODEL_PATH = '/media/nine/HD_1/HD_2_from_seven/Yann/pap_smear/RESNET18/outputs/2024-11-05/21-45-21/weights/best_model_epoch_31.pth'
    YOLO_MODEL_PATH = '/media/nine/HD_1/HD_2_from_seven/Yann/pap_smear/YOLO/runs/detect/independent_cell_train/weights/best.pt'

    # Test images directory
    patient_name = 'AP19'
    TEST_IMAGES_DIR = f'/media/nine/HD_1/HD_2_from_seven/Yann/pap_smear/data/jpg_unlabeled_data/{patient_name}'
    OUTPUT_DIR = f'./outputs'
    JSON_FILENAME = f'{patient_name}.json'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    JSON_PATH = os.path.join(OUTPUT_DIR, JSON_FILENAME)

    AP19_required_image_names = ['tile_18_18.jpg', 'tile_17_6.jpg', 'tile_11_7.jpg', 'tile_36_8.jpg', 
                                'tile_15_37.jpg', 'tile_27_30.jpg', 'tile_49_21.jpg', 'tile_37_10.jpg', 
                                'tile_24_3.jpg', 'tile_36_10.jpg', 'tile_41_14.jpg', 'tile_49_22.jpg', 'tile_41_5.jpg', 
                                'tile_48_23.jpg', 'tile_25_27.jpg', 'tile_35_41.jpg', 'tile_42_21.jpg', 'tile_32_50.jpg', 
                                'tile_11_8.jpg', 'tile_29_33.jpg', 'tile_36_2.jpg', 'tile_37_17.jpg', 'tile_20_47.jpg', 
                                'tile_34_50.jpg', 'tile_17_44.jpg',"tile_12_21.jpg", "tile_20_36.jpg","tile_23_8.jpg","tile_25_16.jpg",
                                "tile_29_42.jpg","tile_29_53.jpg","tile_30_25.jpg","tile_31_53.jpg",
                                "tile_35_52.jpg","tile_40_50.jpg"]

    AP18_required_image_names = [
        "tile_15_32.jpg", "tile_16_8.jpg", "tile_18_42.jpg", "tile_21_3.jpg",
        "tile_24_30.jpg", "tile_24_46.jpg", "tile_24_47.jpg", "tile_30_33.jpg",
        "tile_30_42.jpg", "tile_32_7.jpg", "tile_35_21.jpg", "tile_36_14.jpg",
        "tile_39_32.jpg", "tile_43_49.jpg", "tile_44_8.jpg", "tile_47_26.jpg",
        "tile_48_12.jpg", "tile_50_30.jpg", "tile_55_22.jpg", "tile_55_34.jpg",
        "tile_55_36.jpg", "tile_6_18.jpg", "tile_8_36.jpg", "tile_9_23.jpg",
        "tile_9_27.jpg", 'tile_42_1.jpg', 'tile_37_33.jpg', 'tile_16_3.jpg', 'tile_53_16.jpg'
    ]

    if patient_name == 'AP19':
        required_image_names = AP19_required_image_names
    else:
        required_image_names = AP18_required_image_names

    # Detection and classification thresholds
    YOLO_CONFIDENCE_THRESHOLD = 0.5  # Adjust based on model performance
    RESNET_CONFIDENCE_THRESHOLD = 0.0  # Adjust based on desired specificity

    # Class names and associated colours
    CLASS_NAMES = ['Abnormal', 'Benign', 'Normal']
    CLASS_COLOURS = {
        "Normal": (102, 204, 0),    # Green for Normal
        "Abnormal": (0, 0, 255),    # Red for Abnormal
        "Benign": (255, 0, 0),      # Blue for Benign
    }
    results = test_cells()