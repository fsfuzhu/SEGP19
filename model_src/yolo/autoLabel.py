import os
from ultralytics import YOLO
import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the trained YOLO model
model = YOLO('model_all_cells.pt')  # Replace with the path to your model weights file

# Set input and output directories
input_folder = './'  # Current directory
output_image_folder = './yolo_output/images/'  # Directory for output annotated images
output_label_folder = './yolo_output/labels/'  # Directory for output label files

# Create output directories if they don't exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

# Define colors for each class (modify as needed)
class_colors = {
    0: (0, 255, 0),    # Green for healthy
    1: (255, 0, 0),    # Blue for rubbish
    2: (0, 0, 255),    # Red for unhealthy
    3: (255, 255, 0)   # Yellow for bothcells
}

# Supported image formats
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
image_files = [file for file in os.listdir(input_folder) if os.path.splitext(file)[1].lower() in image_extensions]

def compute_iou(box1, box2):
    """
    Compute IoU between two bounding boxes.
    Box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute intersection area
    inter_width = max(0, x2 - x1 + 1)
    inter_height = max(0, y2 - y1 + 1)
    inter_area = inter_width * inter_height

    # Compute individual areas
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Compute IoU
    iou = inter_area / float(area1 + area2 - inter_area)
    return iou

def non_max_suppression(boxes, iou_threshold=0.5):
    """
    IoU-based Non-Maximum Suppression, keeping larger bounding boxes.
    boxes: List of dictionaries with keys 'box', 'class', 'conf'
    """
    if not boxes:
        return []

    # Sort by area in descending order
    boxes = sorted(boxes, key=lambda x: (x['box'][2] - x['box'][0]) * (x['box'][3] - x['box'][1]), reverse=True)
    kept_boxes = []

    while boxes:
        current = boxes.pop(0)
        kept_boxes.append(current)
        boxes = [box for box in boxes if compute_iou(current['box'], box['box']) <= iou_threshold]

    return kept_boxes

# Process each image
for image_file in image_files:
    try:
        # Full path of the input image
        input_image_path = os.path.join(input_folder, image_file)
        
        # Read image
        image = cv2.imread(input_image_path)
        if image is None:
            logging.warning(f"Unable to read image {image_file}. Skipping.")
            continue

        height, width = image.shape[:2]

        # Run inference with YOLO model
        results = model.predict(source=input_image_path, save=False)  # Disable auto-save

        # Create an annotated copy of the image
        annotated_image = image.copy()

        # Prepare label file
        label_lines = []

        # Collect all detected bounding boxes
        detected_boxes = []

        # Iterate over each result
        for result in results:
            if result.boxes is None:
                continue  # No boxes detected in this result

            # Iterate over each bounding box
            for i, box in enumerate(result.boxes.xyxy):
                # Move tensor to CPU and convert to NumPy array
                box_np = box.cpu().numpy()
                
                # Check for NaN values
                if np.isnan(box_np).any():
                    logging.warning(f"NaN detected in bounding box for image {image_file}. Skipping this box.")
                    continue  # Skip invalid box

                # Get bounding box coordinates
                x1, y1, x2, y2 = box_np
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Get class ID and confidence
                cls = int(result.boxes.cls[i].cpu().numpy()) if len(result.boxes.cls) > 0 else -1
                conf = float(result.boxes.conf[i].cpu().numpy()) if len(result.boxes.conf) > 0 else 0.0

                if cls not in model.names:
                    logging.warning(f"Unknown class ID {cls} detected in image {image_file}. Skipping this box.")
                    continue  # Skip unknown class

                # Add to detected box list
                detected_boxes.append({
                    'box': [x1, y1, x2, y2],
                    'class': cls,
                    'conf': conf
                })

        # Apply Non-Maximum Suppression (NMS)
        filtered_boxes = non_max_suppression(detected_boxes, iou_threshold=0.5)

        # Annotate filtered bounding boxes
        for box_info in filtered_boxes:
            x1, y1, x2, y2 = box_info['box']
            cls = box_info['class']
            conf = box_info['conf']

            # Calculate YOLO format labels (relative to image dimensions)
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            bbox_width = (x2 - x1) / width
            bbox_height = (y2 - y1) / height

            # Ensure all values are between 0 and 1
            x_center = min(max(x_center, 0), 1)
            y_center = min(max(y_center, 0), 1)
            bbox_width = min(max(bbox_width, 0), 1)
            bbox_height = min(max(bbox_height, 0), 1)

            # Create label line
            label_line = f"{cls} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
            label_lines.append(label_line)

            # Get class name and color
            label = f"{model.names[cls]} {conf:.2f}"
            color = class_colors.get(cls, (255, 255, 255))  # Default to white if class ID not in class_colors

            # Draw bounding box on image
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            # Draw label on image
            cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 2)

        # Save annotated image
        output_image_path = os.path.join(output_image_folder, image_file)
        cv2.imwrite(output_image_path, annotated_image)

        # Save label file
        label_file_name = os.path.splitext(image_file)[0] + '.txt'
        label_file_path = os.path.join(output_label_folder, label_file_name)
        with open(label_file_path, 'w') as f:
            f.write('\n'.join(label_lines))

        logging.info(f'Processed {image_file}, saved annotated image to {output_image_path}, label file to {label_file_path}')

    except Exception as e:
        logging.error(f"Error processing {image_file}: {e}")
        continue  # Continue to the next image

logging.info('Processing complete. All results saved to: {}'.format(output_image_folder))
