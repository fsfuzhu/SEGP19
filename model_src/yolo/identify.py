import os
from ultralytics import YOLO
import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the trained YOLO model
model = YOLO('best.pt')  # Replace with your model's weights file path

# Set input and output directories
input_folder = './'  # Current directory
output_folder = './yolo_output/'  # Output directory for results

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define colors for each class (healthy, rubbish, unhealthy, bothcells)
class_colors = {
    0: (0, 255, 0),    # Green for Normal
    1: (0, 0, 255),    # Red for Abnormal
    2: (255, 0, 0),    # Blue for Benigh
}

# Supported image formats
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
image_files = [file for file in os.listdir(input_folder) if os.path.splitext(file)[1].lower() in image_extensions]

# Process each image
for image_file in image_files:
    try:
        # Full path to the input image
        input_image_path = os.path.join(input_folder, image_file)
        
        # Read the image
        image = cv2.imread(input_image_path)
        if image is None:
            logging.warning(f"Unable to read image {image_file}. Skipping.")
            continue

        # Perform inference using the YOLO model
        results = model.predict(source=image, save=False)  # Disable automatic saving
        
        # Create a copy of the image to draw annotations
        annotated_image = image.copy()

        # Iterate over each result
        for result in results:
            if result.boxes is None:
                continue  # No boxes detected in this result

            # Iterate over each bounding box
            for i, box in enumerate(result.boxes.xyxy):
                # Move the box tensor to CPU and convert to NumPy
                box_np = box.cpu().numpy()
                
                # Check for NaN values
                if np.isnan(box_np).any():
                    logging.warning(f"Detected NaN in bounding box for image {image_file}. Skipping this box.")
                    continue  # Skip invalid boxes

                # Convert box coordinates to integers
                x1, y1, x2, y2 = map(int, box_np)

                # Get the class ID and confidence
                cls = int(result.boxes.cls[i].cpu().numpy()) if len(result.boxes.cls) > 0 else -1
                conf = float(result.boxes.conf[i].cpu().numpy()) if len(result.boxes.conf) > 0 else 0.0
                
                # Get the class name and color
                label = f"{model.names[cls]} {conf:.2f}" if cls in model.names else f"Class {cls} {conf:.2f}"
                color = class_colors.get(cls, (255, 255, 255))  # Default to white if class ID not in class_colors

                # Draw the bounding box on the image
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

                # Put the label on the image
                cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 2)

        # Save the annotated image to the output directory
        output_image_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_image_path, annotated_image)

        logging.info(f'Processed {image_file} and saved the results to {output_image_path}')

    except Exception as e:
        logging.error(f"Error processing {image_file}: {e}")
        continue  # Proceed to the next image

logging.info('Processing complete. All results saved to: {}'.format(output_folder))
